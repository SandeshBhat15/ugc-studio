"""Production Manager: project state, stage execution, QC, exports."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import comfy as comfy_mod
import llm
import media
import planner
import voices
from comfy import Comfy, ComfyError, frames_for, resolve_size
from core import (PROJECTS_DIR, comfy_dir, ensure_dirs, load_settings, now,
                  read_json, slugify, unique_dir, write_json)

# ComfyUI lives outside this repo, so every path into it is derived from one
# configurable root instead of a hardcoded drive letter.
COMFY_ROOT = comfy_dir()
OUTPUT_DIR = COMFY_ROOT / "output"
COMFY_PREFIX = "UGCStudio"
CHARACTERS_DIR = COMFY_ROOT / "models" / "characters"

STAGE_KEYS = ("stage01", "stage02", "stage03", "stage04")


# ---------------------------------------------------------------- store

def _dirs(root: Path) -> None:
    for sub in ("brief", "references/products", "references/creators",
                "references/videos", "storyboard", "prompts", "previews",
                "audio", "captions", "exports", "logs", "scene_images", "scene_videos"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def project_path(pid: str) -> Path:
    return PROJECTS_DIR / pid / "project.json"


def get(pid: str) -> dict:
    data = read_json(project_path(pid))
    if not data:
        raise RuntimeError(f"Project '{pid}' not found.")
    return data


def save(project: dict) -> dict:
    project["updated"] = now()
    write_json(project_path(project["id"]), project)
    return project


def list_projects() -> list[dict]:
    ensure_dirs()
    out = []
    for p in sorted(PROJECTS_DIR.glob("*/project.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        d = read_json(p, {}) or {}
        out.append({
            "id": d.get("id"), "name": d.get("name"), "client": d.get("client"),
            "updated": d.get("updated"), "duration": d.get("duration"),
            "status": _overall(d),
        })
    return out


def _overall(p: dict) -> str:
    if p.get("stage04", {}).get("exports"):
        return "completed"
    if p.get("stage03", {}).get("status") == "completed":
        return "editing"
    if p.get("stage02", {}).get("status") in ("completed", "awaiting_approval"):
        return "review"
    if p.get("stage01", {}).get("status") == "awaiting_approval":
        return "planning"
    return "draft"


def new_project(data: dict) -> dict:
    ensure_dirs()
    name = (data.get("name") or "Untitled Project").strip()
    pid = slugify(name) + "_" + uuid.uuid4().hex[:6]
    root = unique_dir(PROJECTS_DIR, pid)
    _dirs(root)
    project = {
        "id": root.name,
        "name": name,
        "client": data.get("client", ""),
        "brief": data.get("brief", ""),
        "objective": data.get("objective", ""),
        "platform": data.get("platform", "Instagram Reels"),
        "style": data.get("style", "talking-head UGC"),
        "aspect": data.get("aspect", "9:16"),
        "duration": int(data.get("duration", 15)),
        "concepts": int(data.get("concepts", 1)),
        "single_take": bool(data.get("single_take")),
        "created": now(),
        "updated": now(),
        "refs": {"products": [], "creators": [], "videos": [], "extra": []},
        "llm": {},
        "stage01": {"status": "not_started"},
        "stage02": {"status": "not_started", "scenes": {}},
        "stage03": {"status": "not_started", "scenes": {}},
        "stage04": {"status": "not_started", "exports": []},
        "log": [],
    }
    (root / "brief" / "brief.txt").write_text(project["brief"], encoding="utf-8")
    return save(project)


def log(project: dict, msg: str, level: str = "info") -> None:
    project.setdefault("log", []).append({"t": now(), "level": level, "msg": msg})
    project["log"] = project["log"][-300:]
    p = PROJECTS_DIR / project["id"] / "logs" / "studio.log"
    with p.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {level.upper()} {msg}\n")


def add_reference(pid: str, kind: str, src: Path, original: str = "") -> dict:
    project = get(pid)
    root = PROJECTS_DIR / pid
    kind = kind if kind in ("products", "creators", "videos", "scenes", "extra") else "extra"
    dest_dir = root / "references" / kind
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    safe = slugify(Path(original or src.name).stem) + src.suffix.lower()
    dest = dest_dir / safe
    i = 2
    while dest.exists():
        dest = dest_dir / f"{slugify(Path(original or src.name).stem)}_{i}{src.suffix.lower()}"
        i += 1
    shutil.copy2(src, dest)
    entry = {"file": str(dest.relative_to(root)).replace("\\", "/"),
             "original": original or src.name, "added": now()}
    project["refs"][kind].append(entry)
    save(project)
    return entry


# ---------------------------------------------------------------- stage 01

def run_brief(pid: str) -> dict:
    project = get(pid)
    settings = load_settings()
    project["stage01"]["status"] = "planning"
    save(project)
    try:
        fields = None
        if settings.get("llm_mode") != "offline":
            try:
                fields = planner.extract_with_llm(settings, project["brief"])
                project["llm"] = {"provider": fields.get("source"), "model": fields.get("model")}
            except Exception as e:
                log(project, f"LLM brief extraction failed, using offline parser: {e}", "warn")
        if not fields:
            fields = planner.extract_offline(project["brief"])
        project["stage01"].update({
            "status": "awaiting_approval", "fields": fields,
            "finished": now(),
        })
        save(project)
        root = PROJECTS_DIR / pid
        write_json(root / "brief" / "brief_fields.json", fields)
        log(project, "Brief extracted.")
        return project["stage01"]
    except Exception as e:
        project["stage01"].update({"status": "failed", "error": str(e)})
        log(project, f"Brief extraction failed: {e}", "error")
        save(project)
        raise


def run_plan(pid: str) -> dict:
    project = get(pid)
    settings = load_settings()
    project["stage01"]["status"] = "planning"
    save(project)
    try:
        fields = project["stage01"].get("fields")
        if not fields:
            # Prefer the LLM for brief extraction; fall back to the regex
            # reader so planning still works when no provider is configured.
            try:
                fields = planner.extract_with_llm(settings, project["brief"])
            except Exception:
                fields = planner.extract_offline(project["brief"])
        opts = {"duration": project["duration"], "style": project["style"],
                "platform": project["platform"], "aspect": project["aspect"],
                "single_take": bool(project.get("single_take"))}
        # The script has to sound like this specific creator, not like a
        # generic ad reader, so the character's own tone goes into the brief.
        ch = project.get("character") or {}
        opts["tone"] = str(ch.get("tone") or "").strip()
        opts["creator"] = str(ch.get("description") or "").strip()
        sb = planner.plan(settings, fields, opts)
        sb = planner.normalize_storyboard(sb, int(project["duration"]),
                                          planner._scene_count(int(project["duration"]),
                                                               opts["single_take"]),
                                          single_take=opts["single_take"],
                                          brief_fields=fields)
        root = PROJECTS_DIR / pid
        (root / "brief").mkdir(parents=True, exist_ok=True)
        write_json(root / "brief" / "brief_fields.json", fields)
        project["stage01"].update({
            "status": "awaiting_approval", "storyboard": sb, "fields": fields,
            "llm": {"provider": sb.get("source"), "model": sb.get("model", "")},
        })
        write_json(root / "storyboard" / "storyboard.json", sb)
        (root / "storyboard" / "storyboard.md").write_text(_storyboard_md(sb), encoding="utf-8")
        save(project)
        log(project, f"Storyboard planned ({len(sb['scenes'])} scenes).")
        return project["stage01"]
    except Exception as e:
        project["stage01"].update({"status": "failed", "error": str(e)})
        log(project, f"Storyboard planning failed: {e}", "error")
        save(project)
        raise


def _storyboard_md(sb: dict) -> str:
    lines = [f"# {sb.get('concept_name', 'Storyboard')}", "",
             f"**Hook:** {sb.get('hook', '')}", "",
             f"**Total:** {sb.get('total_seconds', 0)}s", ""]
    for s in sb.get("scenes", []):
        lines += [f"## Scene {s['n']} — {s.get('purpose', '')} ({s.get('seconds', 0)}s)", "",
                  f"- **Script:** {s.get('script', '')}",
                  f"- **Visual:** {s.get('visual', '')}",
                  f"- **Camera:** {s.get('camera_framing', '')} / {s.get('camera_movement', '')}",
                  f"- **Image prompt:** {s.get('image_prompt', '')}",
                  f"- **Video prompt:** {s.get('video_prompt', '')}", ""]
    return "\n".join(lines)


def update_scene(pid: str, n: int, patch: dict) -> dict:
    project = get(pid)
    sb = project["stage01"].get("storyboard") or {}
    for s in sb.get("scenes", []):
        if int(s["n"]) == int(n):
            s.update({k: v for k, v in patch.items() if k in s})
            break
    project["stage01"]["storyboard"] = sb
    save(project)
    return sb


def reorder_scenes(pid: str, order: list[int]) -> dict:
    project = get(pid)
    sb = project["stage01"]["storyboard"]
    by_n = {int(s["n"]): s for s in sb["scenes"]}
    sb["scenes"] = [by_n[n] for n in order if n in by_n]
    for i, s in enumerate(sb["scenes"], start=1):
        s["n"] = i
    project["stage01"]["storyboard"] = sb
    save(project)
    return sb


def approve_storyboard(pid: str) -> dict:
    project = get(pid)
    sb = project["stage01"].get("storyboard")
    if not sb or not sb.get("scenes"):
        raise RuntimeError("Nothing to approve — plan a storyboard first.")
    issues = qc_storyboard(sb, project)
    project["stage01"].update({"status": "completed", "approved": now(), "qc": issues})
    save(project)
    log(project, "Storyboard approved.")
    return project["stage01"]


def qc_storyboard(sb: dict, project: dict) -> list[dict]:
    issues = []
    scenes = sb.get("scenes", [])
    if not scenes:
        issues.append({"level": "error", "msg": "No scenes in storyboard."})
    total = sum(int(s.get("seconds", 0)) for s in scenes)
    if total != int(project["duration"]):
        issues.append({"level": "warn",
                       "msg": f"Scene durations total {total}s but the brief asks for {project['duration']}s."})
    for s in scenes:
        if not s.get("image_prompt") or len(s["image_prompt"]) < 15:
            issues.append({"level": "warn", "msg": f"Scene {s['n']} has a very short image prompt."})
        if int(s.get("seconds", 0)) < 1:
            issues.append({"level": "error", "msg": f"Scene {s['n']} has an invalid duration."})
    # Copy problems the planner already detected, so the user sees them at the
    # approval gate instead of discovering them after a render.
    for flag in sb.get("slop_flags") or []:
        issues.append({"level": "warn", "msg": flag})
    return issues


# ---------------------------------------------------------------- stage 02

def comfy_client() -> Comfy:
    return Comfy(load_settings().get("comfy_url", comfy_mod.Comfy("http://127.0.0.1:8188").base))


def _png_size(path: Path) -> tuple[int, int]:
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return (0, 0)
        w, h = struct.unpack(">II", head[16:24])
        return (w, h)
    except Exception:
        return (0, 0)


def _refs_for_scene(project: dict, scene: dict, c: Comfy) -> list[str]:
    refs = scene.get("refs") or {}
    out: list[str] = []
    root = PROJECTS_DIR / project["id"]

    def upload(entry_key: str, idx: int):
        lst = project["refs"].get(entry_key) or []
        if not lst:
            return
        idx = max(1, min(idx, len(lst)))
        entry = lst[idx - 1]
        if not entry.get("comfy_name"):
            entry["comfy_name"] = c.upload_image(root / entry["file"])
            save(project)
        out.append(entry["comfy_name"])

    # 1) CHARACTER IDENTITY ANCHOR — the same approved sheet in every scene.
    #    It goes first so it becomes <Picture 1> and anchors *who* the creator is.
    char = project.get("character") or {}
    if char.get("status") == "ready" and char.get("file"):
        if not char.get("comfy_name"):
            char["comfy_name"] = c.upload_image(root / char["file"])
            save(project)
        out.append(char["comfy_name"])

    prods = refs.get("products") or ([1] if project["refs"].get("products") else [])
    creators = refs.get("creators") or ([1] if project["refs"].get("creators") else [])
    for i in prods:
        upload("products", int(i))
    for i in creators:
        upload("creators", int(i))
    # A scene reference locks framing, background and blocking for ONE scene.
    # Scene N uses the Nth uploaded scene reference, so nothing needs mapping.
    scene_ref = None
    scene_pool = project["refs"].get("scenes") or []
    if scene_pool:
        idx = max(1, min(int(scene.get("n") or 1), len(scene_pool)))
        upload("scenes", idx)
        scene_ref = len(out)          # 1-based <Picture N> position
    # any extra visual references the artist uploaded
    for i in range(1, len(project["refs"].get("extra") or []) + 1):
        upload("extra", i)
    return out[:9], {"scene_ref": scene_ref}


def run_character(pid: str, opts: dict | None = None) -> dict:
    """Generate ONE locked character sheet that every scene then reuses.

    This is the character-generation step: instead of hoping the model keeps a
    face consistent across independent scene renders, we render the person once,
    approve that sheet, and feed the exact same image into every scene as
    <Picture 1>.
    """
    project = get(pid)
    opts = opts or {}
    settings = load_settings()
    c = comfy_client()
    if not c.online():
        raise RuntimeError("ComfyUI is not running. Start ComfyUI Desktop and try again.")

    profile = settings.get("profile", "standard")
    width, height = resolve_size(project.get("aspect") or "9:16", profile)
    root = PROJECTS_DIR / pid
    (root / "references" / "creators").mkdir(parents=True, exist_ok=True)

    # optional: artist-uploaded creator photos steer the look
    refs: list[str] = []
    for entry in project["refs"].get("creators") or []:
        if not entry.get("comfy_name"):
            entry["comfy_name"] = c.upload_image(root / entry["file"])
            save(project)
        refs.append(entry["comfy_name"])

    desc = (opts.get("description") or "").strip()
    gender = str(opts.get("gender") or "").strip().lower()
    if gender not in ("female", "male"):
        gender = infer_gender(desc, str(project.get("brief") or "")[:600])
    if desc:
        who = desc
    else:
        # The old gender-neutral wording is exactly how a male still appeared
        # in a brief that asked for a woman, so the default now carries gender.
        word = {"female": "woman", "male": "man"}.get(gender, "person")
        who = (f"a friendly, natural-looking {word} in their late twenties, "
               "genuine warm smile, casual everyday clothing")
    prompt = (f"Create a clean character reference portrait of {who}. "
              "Head and shoulders, looking at the camera, neutral seamless background, "
              "single soft key light from 45 degrees, subtle shadow side, sharp focus "
              "on the face, photorealistic, natural skin texture.")
    if refs:
        prompt += (" Keep this exact person identical to the reference photos: "
                   "same face, hair, skin tone and age.")
    prompt = (prompt.rstrip(". ") +
              ". Do not render any captions, titles, slogans, watermarks or invented text.")

    seed = int(opts.get("seed") or 7)
    graph = c.image_graph(
        prompt=prompt, width=width, height=height, ref_images=refs,
        prefix=f"{COMFY_PREFIX}/{pid}/character",
        steps=20 if profile != "draft" else 14, seed=seed)
    qid = c.queue(graph)
    result = c.wait(qid, timeout=2400)
    produced = _collect_outputs(result, "images")
    if not produced:
        raise ComfyError("Character generation finished but produced no image.")

    src = OUTPUT_DIR / produced[0].get("subfolder", "") / produced[0]["filename"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = root / "references" / "creators" / f"character_{stamp}.png"
    shutil.copy2(src, dest)
    wh = _png_size(dest)

    previous = project.get("character") or {}
    prev_voice = previous.get("voice") if isinstance(previous.get("voice"), dict) else {}
    project["character"] = {
        "file": str(dest.relative_to(root)).replace("\\", "/"),
        "status": "ready",
        "width": wh[0], "height": wh[1],
        "prompt": prompt,
        "description": who,
        "gender": gender,
        "voice": {"name": (str(opts.get("voice") or "").strip()
                           or str(prev_voice.get("name") or "")),
                  "style": (str(opts.get("voice_style") or "").strip()
                            or str(prev_voice.get("style") or ""))},
        "tone": str(opts.get("tone") or "").strip() or str(previous.get("tone") or ""),
        "seed": seed,
        "profile": profile,
        "used_references": len(refs),
        "created": now(),
    }
    save(project)
    log(project, f"Character sheet generated: {dest.name}")
    return project["character"]


# ---------------------------------------------------------------- characters

def save_character(opts: dict) -> dict:
    """Encode reference photos into a reusable Omnichar `.char` identity.

    Separate from run_character: that one renders a portrait sheet inside a
    single project, while this writes a standalone identity object that the
    video pipeline can load by name for any project (settings["character"]).
    """
    name = (opts.get("name") or "").strip()
    description = (opts.get("description") or "").strip()
    if not name:
        raise RuntimeError("A character name is required.")
    slug = slugify(name, "")
    if not slug:
        raise RuntimeError("Character name must contain letters or digits.")
    if not description:
        raise RuntimeError("A character description is required.")

    images: dict[str, list[str]] = {}
    total = 0
    for role in ("face", "body", "cloths"):
        paths = [str(raw).strip() for raw in (opts.get(role) or [])
                 if str(raw or "").strip()]
        if len(paths) > 3:
            raise RuntimeError(f"At most 3 {role} images are allowed.")
        for text in paths:
            p = Path(text)
            if not p.is_absolute():
                raise RuntimeError(f"'{text}' is not an absolute path.")
            if not p.is_file():
                raise RuntimeError(f"{role} reference not found: {p}")
        images[role] = paths
        total += len(paths)
    if not total:
        raise RuntimeError("At least one reference image (face, body or outfit) is required.")

    voice = str(opts.get("voice") or "").strip() or None
    if voice:
        vp = Path(voice)
        if not vp.is_absolute():
            raise RuntimeError(f"'{voice}' is not an absolute path.")
        if not vp.is_file():
            raise RuntimeError(f"Voice clip not found: {vp}")
        voice = str(vp)

    c = comfy_client()
    if not c.online():
        raise RuntimeError("ComfyUI is not running. Start ComfyUI Desktop and try again.")

    graph = c.character_graph(
        name=name, description=description, images=images, voice=voice,
        resolution=int(opts.get("resolution") or 1024),
        prefix=f"{COMFY_PREFIX}/character_{slug}")
    qid = c.queue(graph)
    result = c.wait(qid, timeout=1800)

    path = _collect_char_path(result)
    fallback = CHARACTERS_DIR / f"{slug}.char"
    if not path or not Path(path).exists():
        # The save node's STRING output is the authoritative path, but if the
        # history entry dropped it, the file it just wrote still proves success.
        if fallback.exists():
            path = str(fallback)
        else:
            raise ComfyError("Character encode finished but wrote no .char file.")
    return {"name": name, "char": Path(path).name, "path": str(path),
            "images": total, "voice": bool(voice)}


def _collect_char_path(entry: dict) -> str:
    """Find the path the save node returned, whatever bucket ComfyUI used.

    OmnicharSaveCharacter reports a plain STRING, and raw strings land under
    different output keys depending on the node, so scan every value.
    """
    for node_out in (entry.get("outputs") or {}).values():
        for value in (node_out or {}).values():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, str) and item.lower().endswith(".char"):
                    return item
    return ""


def list_characters() -> list[dict]:
    """Every `.char` identity installed on this machine, newest first."""
    out = []
    if not CHARACTERS_DIR.exists():
        return out
    for p in CHARACTERS_DIR.glob("*.char"):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({
            "name": p.name,
            "size_mb": round(st.st_size / (1024 ** 2), 1),
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                .isoformat(timespec="seconds"),
        })
    out.sort(key=lambda x: x["modified"], reverse=True)
    return out


def delete_character(name: str) -> dict:
    """Remove one installed `.char`. The name must be a bare filename."""
    name = Path(str(name or "")).name
    if not name.endswith(".char"):
        raise RuntimeError("Only .char files can be removed.")
    target = (CHARACTERS_DIR / name).resolve()
    if target.parent != CHARACTERS_DIR.resolve():
        raise RuntimeError("That character is outside the characters folder.")
    if not target.exists():
        raise RuntimeError(f"{name} is not installed.")
    target.unlink()
    # Never leave the pipeline pointing at an identity that no longer exists.
    settings = load_settings()
    if settings.get("character") == name:
        from core import save_settings
        settings["character"] = ""
        save_settings(settings)
    return {"deleted": name}


# ---------------------------------------------------------------- project files

EDITABLE_PROJECT_FIELDS = ("name", "client", "brief", "objective", "platform",
                           "style", "aspect", "duration", "single_take")


def update_project(pid: str, patch: dict) -> dict:
    """Edit the brief-level fields of a project after it was created."""
    project = get(pid)
    for key in EDITABLE_PROJECT_FIELDS:
        if key not in patch:
            continue
        value = patch[key]
        if key == "duration":
            value = max(3, int(value or project.get("duration", 15)))
        elif key == "single_take":
            value = bool(value)
        elif key == "aspect":
            value = value if value in ("9:16", "1:1", "16:9") else project.get("aspect")
        else:
            value = str(value or "").strip()
        project[key] = value
    project["updated"] = now()
    save(project)
    log(project, "Project details updated.")
    return project


def _kind_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        return "image"
    if suffix in (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"):
        return "video"
    if suffix in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
        return "audio"
    if suffix in (".json", ".txt", ".md", ".log", ".csv"):
        return "text"
    return "file"


def project_files(pid: str) -> dict:
    """Every file the project owns, so the UI can manage them like a real folder."""
    root = (PROJECTS_DIR / pid).resolve()
    if not root.exists():
        raise RuntimeError(f"Project '{pid}' not found.")
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        files.append({
            "path": rel,
            "name": p.name,
            "dir": str(Path(rel).parent).replace("\\", "/").replace(".", "") or "root",
            "kind": _kind_of(p),
            "size": st.st_size,
            "size_mb": round(st.st_size / (1024 ** 2), 2),
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                .isoformat(timespec="seconds"),
        })
    files.sort(key=lambda x: x["path"])
    total = sum(f["size"] for f in files)
    return {"files": files, "count": len(files),
            "total_mb": round(total / (1024 ** 2), 1), "root": str(root)}


def delete_project_file(pid: str, rel: str) -> dict:
    """Delete one project file and scrub every reference to it from the state."""
    project = get(pid)
    root = (PROJECTS_DIR / pid).resolve()
    target = (root / str(rel or "")).resolve()
    if root not in target.parents:
        raise RuntimeError("That path is outside the project folder.")
    if not target.is_file():
        raise RuntimeError("That file does not exist.")
    rel = str(target.relative_to(root)).replace("\\", "/")
    target.unlink()

    # References (uploads) are listed by path, so drop the matching entry.
    for kind, entries in (project.get("refs") or {}).items():
        project["refs"][kind] = [e for e in (entries or []) if e.get("file") != rel]

    # A deleted still or clip must not keep claiming to be approved/completed.
    for n, entry in (project.get("stage02", {}).get("scenes") or {}).items():
        if entry.get("file") == rel:
            entry.pop("file", None)
            entry["status"] = "not_started"
        if entry.get("approved_file") == rel:
            entry.pop("approved_file", None)
            entry.pop("approved_at", None)
            entry["approved"] = False
            entry["status"] = "generated" if entry.get("file") else "not_started"
    for n, entry in (project.get("stage03", {}).get("scenes") or {}).items():
        if entry.get("file") == rel:
            entry.pop("file", None)
            entry["status"] = "not_started"
    project["stage04"]["exports"] = [
        e for e in (project.get("stage04", {}).get("exports") or [])
        if e.get("file") != rel
    ]
    project["updated"] = now()
    save(project)
    log(project, f"Deleted {rel}.")
    return {"deleted": rel}


def set_character_sheet(pid: str, src, gender: str | None = None,
                        voice: str | None = None, voice_style: str | None = None,
                        tone: str | None = None, description: str | None = None) -> dict:
    """Import a character sheet drawn anywhere instead of generating one.

    A supplied sheet is the strongest identity lock the pipeline can get: it
    becomes <Picture 1> for every scene and for the video pass.
    """
    project = get(pid)
    root = PROJECTS_DIR / pid
    src = Path(str(src or ""))
    if not src.is_file():
        raise RuntimeError("That image file does not exist.")
    if src.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise RuntimeError("Use a PNG, JPG, WebP or BMP image for the sheet.")
    dest_dir = root / "character"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"sheet_imported_{time.strftime('%Y%m%d_%H%M%S')}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    w, h = _img_size(dest)
    previous = project.get("character") or {}
    project["character"] = {
        "file": str(dest.relative_to(root)).replace("\\", "/"),
        "status": "ready", "source": "imported", "width": w, "height": h,
        "created": now(), "prompt": previous.get("prompt", ""),
        "description": (str(description or "").strip()
                        or previous.get("description") or ""),
        "gender": (str(gender or "").strip().lower()
                   if str(gender or "").strip().lower() in ("female", "male")
                   else previous.get("gender") or ""),
        "voice": {"name": (str(voice or "").strip()
                           or (previous.get("voice") or {}).get("name") or ""),
                  "style": (str(voice_style or "").strip()
                            or (previous.get("voice") or {}).get("style") or "")},
        "tone": str(tone or "").strip() or previous.get("tone") or "",
    }
    save(project)
    log(project, f"Character sheet imported from {src.name}.")
    return project["character"]


def delete_project(pid: str) -> dict:
    """Remove a project folder and every file inside it."""
    pid = Path(str(pid or "")).name
    root = (PROJECTS_DIR / pid).resolve()
    if root.parent != PROJECTS_DIR.resolve():
        raise RuntimeError("That project is outside the projects folder.")
    if not root.is_dir():
        raise RuntimeError(f"Project '{pid}' not found.")
    shutil.rmtree(root)
    return {"deleted": pid}


def duplicate_project(pid: str) -> dict:
    """Copy a project so a variant can be tried without touching the original."""
    src = get(pid)
    root = PROJECTS_DIR / pid
    new_id = f"{slugify(src.get('name') or 'project')}_{uuid.uuid4().hex[:6]}"
    dest = PROJECTS_DIR / new_id
    shutil.copytree(root, dest)
    project = read_json(dest / "project.json", {}) or {}
    project["id"] = new_id
    project["name"] = f"{src.get('name') or 'Untitled'} (copy)"
    project["created"] = now()
    project["updated"] = now()
    project["log"] = []
    # Cached ComfyUI names point at the original project's uploaded images.
    for entries in (project.get("refs") or {}).values():
        for e in entries or []:
            e.pop("comfy_name", None)
    (project.get("character") or {}).pop("comfy_name", None)
    write_json(dest / "project.json", project)
    return project


# ---------------------------------------------------------------- speaking

# H3 animates what the prompt describes. A prompt that lists only a silent
# physical performance gets exactly that: speech in the audio track, a closed
# mouth in the picture. So the pipeline appends these itself — the planner is
# not allowed to drop them.
SPEAKING_STILL = (
    "The creator is mid-sentence: mouth slightly open, lips parted as if "
    "speaking the next word, eyes on the lens, eyebrows raised in conversation."
)

SPEAKING_VIDEO = (
    "The creator is talking to the camera for the entire shot: mouth open, lips "
    "and jaw moving continuously with the words, natural blinking, small head "
    "movement on the stressed syllables. Never silent, never mouth closed, "
    "never a voiceover laid over a still face."
)

# What separates a real face from a rendered one at this scale. The turbo LoRA
# and low step counts flatten skin into wax and turn teeth into a white smear,
# so this is appended to every video prompt and the sampler runs the quality
# path. It is deliberately phrased as photographic description, not as a
# negative prompt: the graph has no negative input.
REALISM_VIDEO = (
    "Photographic realism: real human skin with visible pores, fine vellus hair, "
    "freckles and slight unevenness in tone, subsurface scattering in the cheeks "
    "and nose, a little shine on the forehead. Individual teeth are visible and "
    "correctly shaped, with natural gums, a visible tongue and a wet inner lip; "
    "no fused teeth, no blurred or glowing white teeth. Eyes have catchlights, "
    "wet lower lids and visible iris detail. Not waxy, not plastic, not "
    "airbrushed, no beauty filter, no smooth digital skin."
)

FEMALE_WORDS = ("woman", "women", "girl", "female", " she ", " her ", "lady",
                "mother", " mum", " mom", "actress", "wife", "girlfriend")
MALE_WORDS = (" man", " men", " boy", " male", " he ", " his ", " guy",
              "father", " dad", "actor", "husband", "boyfriend")


def infer_gender(*texts) -> str:
    """Guess the on-screen creator's gender from whatever text we have."""
    blob = " " + " ".join(str(t or "").lower() for t in texts) + " "
    female = sum(blob.count(w) for w in FEMALE_WORDS)
    male = sum(blob.count(w) for w in MALE_WORDS)
    if female > male:
        return "female"
    if male > female:
        return "male"
    return ""


def creator_gender(project: dict) -> str:
    """Gender of the person actually on screen, or '' when nobody knows.

    A character sheet is the authority. Without one we fall back to the text we
    do have, because a wrong voice is worse than an inferred one.
    """
    ch = project.get("character") or {}
    g = str(ch.get("gender") or "").strip().lower()
    if g in ("female", "male"):
        return g
    fields = (project.get("stage01") or {}).get("fields") or {}
    audience = fields.get("target_audience") if isinstance(fields, dict) else ""
    return infer_gender(ch.get("description"), ch.get("prompt"), audience,
                        str(project.get("brief") or "")[:600])


ACCENTS = {"indian": "Indian English accent", "british": "British accent",
           "american": "American accent", "australian": "Australian accent",
           "nigerian": "Nigerian accent", "irish": "Irish accent",
           "scottish": "Scottish accent", "south african": "South African accent"}


def creator_voice(project: dict) -> dict:
    """The creator's own voice: the prebuilt voice, its delivery, and the tone."""
    ch = project.get("character") or {}
    voice = ch.get("voice") if isinstance(ch.get("voice"), dict) else {}
    return {
        "name": str(voice.get("name") or ch.get("voice_name") or "").strip(),
        "style": str(voice.get("style") or ch.get("voice_style") or "").strip(),
        "tone": str(ch.get("tone") or "").strip(),
    }


def _default_style(project: dict) -> str:
    """Build a delivery note from the character so the voice is never generic."""
    ch = project.get("character") or {}
    blob = " ".join(str(ch.get(k) or "") for k in
                    ("description", "prompt", "tone")).lower()
    parts = []
    for key, label in ACCENTS.items():
        if key in blob:
            parts.append(label)
            break
    if re.search(r"mid-?20s|2[0-9] years|2[0-9]-year", blob):
        parts.append("in their twenties")
    parts.append("warm, natural conversational delivery")
    parts.append("not a presenter, not an announcer")
    return ", ".join(parts)


def resolve_voice(project: dict, settings: dict) -> tuple[str, str, str]:
    """Pick the voice that belongs to this creator. Returns (name, style, why).

    A voice is part of a character, not a global preference: the character
    stores its own prebuilt voice and its own delivery note, and the gender is
    only a fallback for characters created before that existed.
    """
    cv = creator_voice(project)
    style = cv["style"] or _default_style(project)
    if cv["name"]:
        return cv["name"], style, "set on the character"
    gender = creator_gender(project)
    # No voice was chosen by hand, so read the character and pick one. This is
    # the difference between "a voice" and *this* person's voice: a stock
    # prebuilt name is only right if it came from what the character is.
    ch = project.get("character") or {}
    blob = " ".join([
        str(ch.get("description") or ""), str(ch.get("prompt") or ""),
        str(ch.get("tone") or ""), style,
        str(project.get("brief") or "")[:400],
    ])
    name, why = voices.derive(blob, gender)
    if name:
        return name, style, why
    if gender == "male":
        return (settings.get("tts_voice_male") or "Puck"), style, "male creator default"
    if gender == "female":
        return (settings.get("tts_voice_female") or "Kore"), style, "female creator default"
    return ((settings.get("tts_voice") or "Kore"), style,
            "gender unknown - using the default voice")


def derive_voice_for(blob: str, gender: str = "") -> tuple[str, str]:
    """Public entry point for the picker and the preview endpoint."""
    return voices.derive(blob, gender)


def voice_options() -> list[dict]:
    return voices.options()


def run_stills(pid: str, scenes: list[int] | None = None) -> dict:
    project = get(pid)
    sb = project["stage01"].get("storyboard") or {}
    if not sb.get("scenes"):
        raise RuntimeError("Approve a storyboard before generating images.")
    if project["stage01"].get("status") != "completed":
        raise RuntimeError("Storyboard must be approved before image generation.")

    settings = load_settings()
    c = comfy_client()
    if not c.online():
        raise RuntimeError("ComfyUI is not running. Start ComfyUI Desktop and try again.")

    profile = settings.get("profile", "standard")
    width, height = resolve_size(project["aspect"], profile)
    root = PROJECTS_DIR / pid
    targets = [s for s in sb["scenes"] if not scenes or int(s["n"]) in scenes]

    project["stage02"]["status"] = "generating"
    project["stage02"]["resolution"] = f"{width}x{height}"
    save(project)

    for s in targets:
        n = int(s["n"])
        entry = project["stage02"]["scenes"].setdefault(str(n), {})
        entry.update({"status": "generating", "started": now()})
        save(project)
        try:
            refs, ref_meta = _refs_for_scene(project, s, c)
            prompt = s.get("image_prompt") or s.get("visual") or "product photo"
            has_char = (project.get("character") or {}).get("status") == "ready"
            if has_char:
                prompt = (prompt.rstrip(". ") +
                          ". Keep the creator exactly as in <Picture 1>: the same face, "
                          "hair, skin tone, age and body — it is the same person in every scene.")
            # anchor product identity when a product reference is supplied
            if refs and (s.get("refs") or {}).get("products"):
                pic = 2 if has_char else 1
                prompt = (prompt.rstrip(". ") +
                          f". The product from <Picture {pic}> must stay visibly identical: "
                          "same shape, same label, same colours, clearly in frame.")
            if ref_meta.get("scene_ref"):
                prompt = (prompt.rstrip(". ") +
                          f". Match the framing, background and blocking of <Picture "
                          f"{ref_meta['scene_ref']}> - same location, same camera angle, "
                          "same composition - while keeping the creator and the product "
                          "exactly as they are in the other references.")
            # Never let the model draw overlay text. Brand, CTA and captions are
            # burned in later with FFmpeg, where glyphs are always pixel-sharp;
            # model-rendered lettering is the main cause of distorted text.
            prompt = (prompt.rstrip(". ") +
                      ". Do not render any captions, titles, slogans, subtitles, "
                      "watermarks or invented text of any kind. If the product has a "
                      "label, reproduce every word on it exactly as it appears in the "
                      "reference photo — same spelling, same typeface, same layout. "
                      "Never invent, restyle or reletter a label. If no label is "
                      "visible in the reference, leave the product surface blank and "
                      "unprinted.")
            # A closed-mouth still is what produced a voiceover over a silent
            # face, so any scene with a spoken line asks for a speaking pose.
            if str(s.get("script") or "").strip():
                prompt = (prompt.rstrip(". ") + ". " + SPEAKING_STILL)
            seed = int(entry.get("seed") or 42)
            graph = c.image_graph(
                prompt=prompt, width=width, height=height, ref_images=refs,
                prefix=f"{COMFY_PREFIX}/{pid}/scene_{n}",
                steps=20 if profile != "draft" else 14, seed=seed)
            qid = c.queue(graph)
            entry["prompt_id"] = qid
            save(project)
            result = c.wait(qid, timeout=2400)

            produced = _collect_outputs(result, "images")
            if not produced:
                raise ComfyError("Generation finished but produced no image.")
            src = OUTPUT_DIR / produced[0]["subfolder"] / produced[0]["filename"]
            scene_dir = root / "scene_images" / f"scene_{n:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            dest = scene_dir / f"scene_{n:02d}_{stamp}.png"
            shutil.copy2(src, dest)
            w, h = _png_size(dest)
            qc = []
            if w == 0:
                qc.append({"level": "error", "msg": "File is not a readable PNG."})
            if w and abs((w / h) - (width / height)) > 0.02:
                qc.append({"level": "warn",
                           "msg": f"Image is {w}x{h}, expected ratio {width}x{height}."})
            entry.update({
                "status": "awaiting_approval", "file": str(dest.relative_to(root)).replace("\\", "/"),
                "width": w, "height": h, "prompt": prompt, "seed": seed,
                "refs_used": refs, "model": comfy_mod.REF2VA,
                "profile": profile, "finished": now(), "qc": qc, "error": None,
            })
            write_json(root / "prompts" / f"scene_{n:02d}_image.json",
                       {"prompt": prompt, "seed": seed, "refs": refs,
                        "width": width, "height": height, "model": comfy_mod.REF2VA})
            save(project)
            log(project, f"Scene {n} image generated.")
        except Exception as e:
            entry.update({"status": "failed", "error": str(e), "finished": now()})
            save(project)
            log(project, f"Scene {n} image failed: {e}", "error")

    states = [v.get("status") for v in project["stage02"]["scenes"].values()]
    project["stage02"]["status"] = (
        "awaiting_approval" if "awaiting_approval" in states else
        "failed" if "failed" in states else "completed")
    save(project)
    return project["stage02"]


def approve_still(pid: str, n: int, filename: str | None = None) -> dict:
    project = get(pid)
    root = PROJECTS_DIR / pid
    entry = project["stage02"]["scenes"].get(str(n))
    if not entry:
        raise RuntimeError(f"Scene {n} has no generated image.")
    rel = entry.get("file")
    if filename:
        rel = f"scene_images/scene_{int(n):02d}/{filename}"
    src = root / rel
    if not src.exists():
        raise RuntimeError("That image file does not exist.")
    dest = root / "scene_images" / f"scene_{int(n):02d}" / "approved.png"
    shutil.copy2(src, dest)
    entry.update({"approved": True, "approved_file": str(dest.relative_to(root)).replace("\\", "/"),
                  "approved_at": now(), "status": "approved"})
    _refresh_stage02(project)
    save(project)
    log(project, f"Scene {n} image approved.")
    return entry


def _refresh_stage02(project: dict) -> None:
    """Recompute the stage-02 aggregate from its per-scene entries.

    Approving a scene only updated that scene, so the stage kept reporting
    'awaiting_approval' even when every image had been approved.
    """
    scenes = (project.get("stage02") or {}).get("scenes") or {}
    if not scenes:
        return
    states = [v.get("status") for v in scenes.values()]
    if all(s == "approved" for s in states):
        project["stage02"]["status"] = "completed"
    elif any(s == "approved" for s in states):
        project["stage02"]["status"] = "in_progress"
    elif any(s == "failed" for s in states):
        project["stage02"]["status"] = "failed"
    else:
        project["stage02"]["status"] = "awaiting_approval"


def regenerate_still(pid: str, n: int, prompt: str | None = None,
                     seed: int | None = None) -> dict:
    project = get(pid)
    sb = project["stage01"]["storyboard"]
    for s in sb["scenes"]:
        if int(s["n"]) == int(n) and prompt:
            s["image_prompt"] = prompt
    if seed is not None:
        project["stage02"]["scenes"].setdefault(str(n), {})["seed"] = seed
    project["stage02"]["scenes"].setdefault(str(n), {})["seed"] = (
        seed if seed is not None else int(time.time()) % 100000)
    save(project)
    return run_stills(pid, [n])


def _img_size(path: Path) -> tuple[int, int]:
    """PNG from its header; anything else via ffprobe, which reads JPG/WebP too."""
    if path.suffix.lower() == ".png":
        w, h = _png_size(path)
        if w:
            return w, h
    info = media.probe(path)
    return int(info.get("width") or 0), int(info.get("height") or 0)


def replace_still(pid: str, n: int, src: str) -> dict:
    """Import a still made outside the app (Krea, Blender, a camera) for one scene.

    The file is copied into the project so the project stays self-contained, and
    approval is deliberately reset: an imported image still has to pass the gate.
    """
    project = get(pid)
    sb = project["stage01"].get("storyboard") or {}
    if not any(int(s["n"]) == int(n) for s in (sb.get("scenes") or [])):
        raise RuntimeError(f"Scene {n} does not exist in the storyboard.")
    src_path = Path(str(src or ""))
    if not src_path.is_file():
        raise RuntimeError("That image no longer exists — upload it again.")
    suffix = src_path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise RuntimeError("Use a PNG, JPG or WebP image.")

    root = PROJECTS_DIR / pid
    scene_dir = root / "scene_images" / f"scene_{n:02d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = scene_dir / f"scene_{n:02d}_imported_{stamp}{suffix}"
    i = 2
    while dest.exists():
        dest = scene_dir / f"scene_{n:02d}_imported_{stamp}_{i}{suffix}"
        i += 1
    shutil.copy2(src_path, dest)

    w, h = _img_size(dest)
    qc = []
    if w == 0:
        qc.append({"level": "error", "msg": "Could not read the image dimensions."})
    else:
        want_w, want_h = resolve_size(project.get("aspect") or "9:16", "draft")
        if abs((w / h) - (want_w / want_h)) > 0.02:
            qc.append({"level": "warn",
                       "msg": f"Image is {w}x{h}, the project is {want_w}x{want_h}. "
                              "It will be cropped to fit."})

    entry = project["stage02"]["scenes"].setdefault(str(n), {})
    entry.update({
        "status": "awaiting_approval", "file": str(dest.relative_to(root)).replace("\\", "/"),
        "width": w, "height": h, "profile": "imported", "source": "imported",
        "approved": False, "finished": now(), "qc": qc, "error": None,
    })
    entry.pop("approved_file", None)
    entry.pop("approved_at", None)
    states = [v.get("status") for v in project["stage02"]["scenes"].values()]
    project["stage02"]["status"] = (
        "awaiting_approval" if "awaiting_approval" in states else
        "generating" if "generating" in states else "completed")
    save(project)
    log(project, f"Scene {n} still imported from {src_path.name}.")
    return project


# ---------------------------------------------------------------- stage 03

def align32(value: float) -> int:
    """Round a dimension to the multiple of 32 the H3 latent upscaler requires."""
    return max(32, int(round(float(value) / 32.0)) * 32)


def run_videos(pid: str, scenes: list[int] | None = None,
               seeds: dict | None = None) -> dict:
    project = get(pid)
    sb = project["stage01"].get("storyboard") or {}
    settings = load_settings()
    c = comfy_client()
    if not c.online():
        raise RuntimeError("ComfyUI is not running. Start ComfyUI Desktop and try again.")

    profile = settings.get("video_profile", "draft")
    width, height = resolve_size(project["aspect"], profile)
    # Render the draft small, then resize the *latent* before decoding. The
    # sampler never sees the larger size, so time and VRAM stay at draft level
    # and the face cannot drift on the way up - it is the same latent.
    upscale = None
    if settings.get("video_upscale", True):
        factor = float(settings.get("video_upscale_factor", 2.5) or 2.5)
        if factor > 1.0:
            upscale = (align32(width * factor), align32(height * factor))
    root = PROJECTS_DIR / pid
    # Resolution and sampler quality are independent knobs. What makes the
    # video look plastic is the turbo LoRA plus a handful of steps, NOT the
    # frame size, so the quality path is the default and turbo is an explicit
    # opt-in for throwaway previews.
    turbo = bool(settings.get("video_turbo", False))
    if turbo:
        log(project, "Turbo LoRA is on: expect waxy skin and soft teeth. "
                     "Turn it off in Render settings for the real pass.", "warn")
    steps = int(settings.get("video_steps", 0) or
                {"draft": 24, "standard": 30, "high": 36}.get(profile, 30))
    voice, voice_style, voice_reason = resolve_voice(project, settings)
    if voice_reason.startswith("gender unknown"):
        log(project, f"Voice: {voice_reason} ({voice}). Set the creator's gender "
                     "in the Characters tab to match it to the person on screen.", "warn")
    # Identity for the video pass. The approved still already carries the face
    # the user approved, so a second identity from a *different* person is
    # strictly harmful - that is how a male still became a drifting video.
    proj_char = (project.get("character") or {})
    video_character = proj_char.get("char") or None
    if not video_character and proj_char.get("file") and settings.get("character"):
        log(project, f"Ignoring the global character '{settings['character']}': "
                     "this project has its own creator sheet and mixing two "
                     "identities makes the face drift.", "warn")
    want_tts = bool(settings.get("tts_enabled", True)) and bool(settings.get("gemini_key"))
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    targets = [s for s in sb.get("scenes", []) if not scenes or int(s["n"]) in scenes]

    project["stage03"]["status"] = "generating"
    project["stage03"]["resolution"] = f"{width}x{height}"
    save(project)

    for s in targets:
        n = int(s["n"])
        entry = project["stage03"]["scenes"].setdefault(str(n), {})
        if seeds and str(n) in seeds and seeds[str(n)] is not None:
            entry["seed"] = int(seeds[str(n)])
            save(project)
        still = project["stage02"]["scenes"].get(str(n), {})
        approved = still.get("approved_file")
        if not approved:
            entry.update({"status": "needs_attention",
                          "error": "No approved image for this scene."})
            continue
        still_path = root / approved
        if not still_path.exists():
            entry.update({"status": "needs_attention", "error": "Approved image file is missing."})
            continue
        entry.update({"status": "generating", "started": now()})
        save(project)
        try:
            # 1) Speak the line first. The clip is generated FROM this audio,
            #    so the mouth follows the real words, and the frame count comes
            #    from the actual speech length so nothing drifts afterwards.
            line = str(s.get("script") or "").strip()
            ref_wav = None
            if want_tts and line:
                # Key the file by voice + text. Reusing a plain scene_01.wav
                # meant an old female take survived a voice fix and got played
                # over a male creator.
                digest = hashlib.sha1(
                    f"{voice}\n{voice_style}\n{line}".encode("utf-8")).hexdigest()[:8]
                candidate = audio_dir / f"scene_{n:02d}_{digest}.wav"
                try:
                    if candidate.exists() and not llm.audio_ok(candidate.read_bytes()):
                        log(project, f"Scene {n}: cached voiceover was silent, "
                                     "regenerating it.", "warn")
                        candidate.unlink(missing_ok=True)
                    if not candidate.exists():
                        candidate.write_bytes(llm.gemini_tts(
                            settings["gemini_key"], line, voice=voice,
                            style=voice_style))
                        if not llm.audio_ok(candidate.read_bytes()):
                            candidate.unlink(missing_ok=True)
                            raise RuntimeError("the voice model returned silence")
                    ref_wav = candidate
                except Exception as e:
                    # Rendering without a reference track means H3 invents its
                    # own speech and the mouth no longer matches the script -
                    # the exact bug this stage exists to prevent. Fail the
                    # scene instead of shipping it.
                    log(project, f"Scene {n} voiceover failed: {e}", "error")
                    entry.update({"status": "needs_attention",
                                  "error": f"Voiceover failed: {e}"})
                    save(project)
                    continue

            seconds = float(s.get("seconds", 5))
            if ref_wav is not None:
                ainfo = media.probe(ref_wav)
                if ainfo.get("duration"):
                    seconds = float(ainfo["duration"])
            length = frames_for(seconds)
            if project.get("single_take") and length > 362:
                raise RuntimeError(
                    "A continuous take is capped at about 15 seconds by H3. "
                    "Shorten the project, or turn off single take and cut it into scenes.")
            seed = int(entry.get("seed") or 42)
            video_prompt = str(s.get("video_prompt") or s.get("visual") or "subtle motion")
            if line:
                # The prompt is the only thing telling H3 to move the mouth.
                video_prompt = (video_prompt.rstrip(". ") + ". " + SPEAKING_VIDEO)
            video_prompt = video_prompt.rstrip(". ") + ". " + REALISM_VIDEO
            def _graph(up):
                return c.video_graph(
                    image_name=str(still_path).replace("\\", "/"),
                    prompt=video_prompt,
                    width=width, height=height, length=length,
                    prefix=f"{COMFY_PREFIX}/{pid}/scene_{n}_video",
                    seed=seed, turbo=turbo, audio=True, steps=steps,
                    character=video_character,
                    upscale=up,
                    ref_audio=str(ref_wav).replace("\\", "/") if ref_wav else None)

            upscale_note = None
            qid = c.queue(_graph(upscale))
            entry["prompt_id"] = qid
            save(project)
            try:
                result = c.wait(qid, timeout=5400)
            except Exception as e:
                if not upscale:
                    raise
                # The upscale is an upgrade, not a requirement. If this card
                # cannot decode the larger frame, ship the draft and say so
                # instead of losing the whole scene.
                upscale_note = (f"The {upscale[0]}x{upscale[1]} upscale pass failed "
                                f"({str(e)[:120]}); this scene is at draft size.")
                log(project, f"Scene {n}: {upscale_note}", "warn")
                upscale = None
                qid = c.queue(_graph(None))
                entry["prompt_id"] = qid
                save(project)
                result = c.wait(qid, timeout=5400)
            produced = _collect_videos(result)
            if not produced:
                raise ComfyError("Generation finished but produced no video.")
            item = produced[0]
            src = OUTPUT_DIR / item.get("subfolder", "") / item["filename"]
            scene_dir = root / "scene_videos" / f"scene_{n:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            dest = scene_dir / f"scene_{n:02d}_{stamp}.mp4"
            shutil.copy2(src, dest)
            info = media.probe(dest)
            qc = []
            if upscale_note:
                qc.append({"level": "warn", "msg": upscale_note})
            if info.get("error"):
                qc.append({"level": "error", "msg": f"Cannot read the video: {info['error']}"})
            if not info.get("duration"):
                qc.append({"level": "error", "msg": "Video has zero duration."})
            if want_tts and not creator_gender(project):
                qc.append({"level": "warn",
                           "msg": "Voice may not match the creator: no gender is "
                                  "recorded for this character. Set it in the "
                                  "Characters tab."})
            if want_tts and voice_style and not llm.LAST_TTS.get("styled"):
                qc.append({"level": "warn",
                           "msg": "The delivery style was not applied: the voice "
                                  "model that answered does not accept a style. "
                                  "The words still follow the character's tone."})
            entry.update({
                "status": "completed", "file": str(dest.relative_to(root)).replace("\\", "/"),
                "width": info.get("width"), "height": info.get("height"),
                "fps": info.get("fps"), "duration": info.get("duration"),
                "has_audio": info.get("has_audio"), "frames": length,
                "voiceover": (str(ref_wav.relative_to(root)).replace("\\", "/")
                              if ref_wav is not None else None),
                "turbo": turbo, "steps": steps,
                "voice": voice, "voice_style": voice_style,
                "voice_reason": voice_reason,
                "tts_model": llm.LAST_TTS.get("model") or None,
                "voice_styled": bool(llm.LAST_TTS.get("styled")),
                "prompt": video_prompt, "seed": seed,
                "source_image": approved, "finished": now(), "qc": qc, "error": None,
            })
            write_json(root / "prompts" / f"scene_{n:02d}_video.json",
                       {"prompt": video_prompt, "seed": seed, "frames": length,
                        "width": width, "height": height, "voice": voice,
                        "source_image": approved, "model": comfy_mod.REF2VA})
            save(project)
            log(project, f"Scene {n} video generated.")
        except Exception as e:
            entry.update({"status": "failed", "error": str(e), "finished": now()})
            save(project)
            log(project, f"Scene {n} video failed: {e}", "error")

    states = [v.get("status") for v in project["stage03"]["scenes"].values()]
    project["stage03"]["status"] = (
        "failed" if "failed" in states else
        "needs_attention" if "needs_attention" in states else "completed")
    save(project)
    return project["stage03"]


def _collect_outputs(entry: dict, key: str) -> list[dict]:
    out = []
    for node_out in (entry.get("outputs") or {}).values():
        for item in node_out.get(key, []) or []:
            if isinstance(item, dict) and item.get("filename"):
                out.append(item)
    return out


VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")


def _collect_videos(entry: dict) -> list[dict]:
    """Collect video outputs regardless of which bucket ComfyUI used.

    ComfyUI 0.44.x core SaveVideo reports its result under "images"
    (with metadata kind="video"), while VHS-style nodes use "gifs".
    Check every bucket and filter by extension so a still PNG can
    never be mistaken for a video.
    """
    out = []
    for key in ("videos", "gifs", "images"):
        for item in _collect_outputs(entry, key):
            if item["filename"].lower().endswith(VIDEO_EXTS):
                out.append(item)
    return out


# ---------------------------------------------------------------- stage 04

def build_export(pid: str, opts: dict | None = None) -> dict:
    project = get(pid)
    opts = opts or {}
    root = PROJECTS_DIR / pid
    profile = opts.get("profile") or load_settings().get("profile", "standard")
    width, height = resolve_size(project["aspect"], profile)
    fps = int(opts.get("fps", 24))

    clips = []
    for s in project["stage01"]["storyboard"]["scenes"]:
        n = int(s["n"])
        v = project["stage03"]["scenes"].get(str(n), {})
        if v.get("status") == "completed" and v.get("file"):
            clips.append((n, root / v["file"]))
    if not clips:
        raise RuntimeError("No completed scene videos to export.")

    # Never shrink a clip that was already rendered bigger. The latent-upscale
    # stage produces a larger frame than the still profile asks for, and
    # exporting at the profile size would throw that resolution away.
    if not opts.get("profile"):
        for _n, clip in clips:
            info = media.probe(clip)
            if info.get("width") and info.get("height"):
                if info["width"] * info["height"] > width * height:
                    width, height = int(info["width"]), int(info["height"])
                break

    work = root / "exports" / "_work"
    work.mkdir(parents=True, exist_ok=True)
    want_caps = bool(opts.get("captions", load_settings().get("overlay_captions", True)))
    # The persistent CTA band and a per-scene caption can end up saying the
    # same sentence twice — the last scene usually IS the call to action.
    # Normalise the overlay wording so the caption can stand down when it
    # would just repeat it.
    overlay_texts = {re.sub(r"[^a-z0-9]+", " ", str(o.get("text") or "").lower()).strip()
                     for o in (opts.get("overlays") or [])}
    settings = load_settings()
    want_tts = bool(opts.get("voiceover", settings.get("tts_enabled", True)))
    # "native" keeps the speech MiniMax H3 generated in the same diffusion pass
    # as the frames, so words and mouth are frame-accurate by construction.
    # "tts" overwrites it with Gemini speech (legacy behaviour).
    audio_mode = str(opts.get("audio_mode") or settings.get("audio_mode") or "native").lower()
    voice, voice_style, _voice_reason = resolve_voice(project, settings)
    script_of = {}
    for s in project["stage01"]["storyboard"]["scenes"]:
        line = str(s.get("script") or "").strip()
        # Strip form-field labels such as "Call to action:" so captions read
        # as spoken copy. Older storyboards saved before the planner fix
        # still contain them.
        line = re.sub(r"^\s*(call\s*-?\s*to\s*-?\s*action|cta)\s*[:\-\u2013]\s*",
                      "", line, flags=re.I).strip()
        script_of[int(s["n"])] = line

    norm = []
    for n, clip in clips:
        if not clip.exists():
            raise RuntimeError(f"Scene {n} video file is missing.")
        info = media.probe(clip)
        dst = work / f"norm_{n:02d}.mp4"
        media.normalise(clip, dst, width, height, fps,
                        with_audio=bool(info.get("has_audio")))
        # Real spoken voiceover replaces the model's babble track. Done before
        # the caption burn so its -c:a copy keeps the new audio.
        line = script_of.get(n, "")
        # H3 speaks and renders in one pass, so its own track is already in
        # sync. Only replace it when asked for TTS, or when the clip came back
        # with no audio at all and the scene would otherwise be silent.
        needs_tts = bool(line) and want_tts and (
            audio_mode == "tts" or not info.get("has_audio"))
        if needs_tts:
            try:
                # Reuse the exact take the video was lip-synced against. A new
                # TTS render would be a different performance and would drift.
                cached = None
                recorded = (project["stage03"]["scenes"].get(str(n)) or {}).get("voiceover")
                if recorded:
                    p = root / recorded
                    if p.is_file():
                        cached = p
                if cached is None:
                    # Older projects stored an unhashed take; the hashed name
                    # is what the render actually used.
                    for p in sorted((root / "audio").glob(f"scene_{n:02d}_*.wav")):
                        if llm.audio_ok(p.read_bytes()):
                            cached = p
                            break
                wav = work / f"tts_{n:02d}.wav"
                if cached is not None and llm.audio_ok(cached.read_bytes()):
                    shutil.copy2(cached, wav)
                else:
                    wav.write_bytes(llm.gemini_tts(settings["gemini_key"], line,
                                                   voice=voice, style=voice_style))
                    if not llm.audio_ok(wav.read_bytes()):
                        raise RuntimeError("the voice model returned silence")
                muxed = work / f"voice_{n:02d}.mp4"
                media.replace_audio(dst, wav, muxed, work)
                dst = muxed
            except Exception as e:
                log(project, f"Scene {n} voiceover failed: {e}", "error")
        # Per-scene captions are burned onto each clip BEFORE concatenation so
        # every scene shows its own line of dialogue. Real glyphs, never
        # model-drawn text.
        cap = script_of.get(n, "")
        if want_caps and cap:
            # If the caption repeats wording that a persistent overlay band
            # already shows, strip that phrase so the screen says it once.
            # Longest phrase first: removing "GTA 6" before "Pre-order GTA 6
            # today" would leave a broken "Pre-order today" behind, which is
            # just as redundant as the full line.
            for otext in sorted(overlay_texts, key=lambda t: -len(t.split())):
                words = otext.split()
                if not words:
                    continue
                pat = re.compile(r"[^a-z0-9]+".join(re.escape(w) for w in words),
                                 re.I)
                cap = pat.sub(" ", cap)
            cap = re.sub(r"\s+", " ", cap).strip(" .,!-;:")
            if cap:
                cap = cap[0].upper() + cap[1:]
        if want_caps and cap:
            capped = work / f"cap_{n:02d}.mp4"
            media.burn_text(dst, capped,
                            [{"text": cap, "pos": "bottom",
                              "size": 0.042, "bold": True}],
                            width, height, work)
            dst = capped
        norm.append(dst)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    final = root / "exports" / f"{slugify(project['name'])}_{width}x{height}_{stamp}.mp4"
    joined = work / f"joined_{stamp}.mp4"
    media.concat(norm, joined, work)
    if audio_mode != "tts":
        # The native track is what the mouth was animated to; levelling it
        # keeps speech audible over room tone without touching the frames.
        levelled = work / f"level_{stamp}.mp4"
        try:
            media.loudnorm(joined, levelled)
            joined = levelled
        except Exception as e:
            log(project, f"Loudness normalise skipped: {e}", "error")

    overlays = [o for o in (opts.get("overlays") or [])
                if str(o.get("text") or "").strip()]
    if overlays:
        media.burn_text(joined, final, overlays, width, height, work)
    else:
        shutil.move(str(joined), str(final))

    info = media.probe(final)
    rec = {"file": str(final.relative_to(root)).replace("\\", "/"),
           "width": info.get("width"), "height": info.get("height"),
           "fps": info.get("fps"), "duration": info.get("duration"),
           "size_mb": info.get("size_mb"), "created": now(),
           "scenes": [n for n, _ in clips],
           "overlays": [str(o.get("text")) for o in overlays]}
    project["stage04"]["exports"].append(rec)
    project["stage04"]["status"] = "completed"
    save(project)
    log(project, f"Exported {final.name}")
    return rec
