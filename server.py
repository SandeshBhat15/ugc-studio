"""UGC Studio — local web app (standard library only)."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import inventory
import llm
import pipeline
from core import (APP_DIR, PROJECTS_DIR, STATIC_DIR, ensure_dirs, load_settings,
                  masked, now, save_settings)

HOST = "127.0.0.1"
PORT = int(os.environ.get("UGC_PORT", "8756"))

# Files dropped on the dashboard land here first. ComfyUI reads absolute
# paths, so a staged file can be handed to a node without re-uploading it.
UPLOAD_DIR = APP_DIR / "uploads"

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------- helpers

def _reveal(path: Path) -> None:
    """Open a folder in the OS file manager. Local app, local user."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError("That folder does not exist yet.")
    try:
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - intended desktop behaviour
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not open the folder: {e}") from e


def _kind_of(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        return "image"
    if suffix in (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"):
        return "video"
    if suffix in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
        return "audio"
    return "file"

def _job_start(pid: str, kind: str, fn, *args, **kwargs) -> dict:
    with JOBS_LOCK:
        if JOBS.get(pid, {}).get("running"):
            raise RuntimeError("A job is already running for this project.")
        JOBS[pid] = {"running": True, "kind": kind, "started": now(), "error": None}
    def run():
        try:
            fn(*args, **kwargs)
            with JOBS_LOCK:
                JOBS[pid].update(running=False, finished=now())
        except Exception as e:  # noqa: BLE001
            with JOBS_LOCK:
                JOBS[pid].update(running=False, error=str(e), finished=now())
            traceback.print_exc()
    threading.Thread(target=run, daemon=True).start()
    return JOBS[pid]


def _job_state(pid: str) -> dict:
    with JOBS_LOCK:
        return dict(JOBS.get(pid, {"running": False}))


def _parse_multipart(body: bytes, content_type: str) -> dict:
    m = re.search(r"boundary=([^;]+)", content_type)
    if not m:
        return {}
    boundary = ("--" + m.group(1).strip('"')).encode()
    out: dict = {"files": {}, "fields": {}}
    for chunk in body.split(boundary):
        if not chunk.strip(b"-\r\n"):
            continue
        if b"\r\n\r\n" not in chunk:
            continue
        head, data = chunk.split(b"\r\n\r\n", 1)
        data = data.rstrip(b"\r\n")
        head_s = head.decode("utf-8", "replace")
        fn = re.search(r'filename="([^"]*)"', head_s)
        name = re.search(r'name="([^"]*)"', head_s)
        if not name:
            continue
        if fn:
            out["files"][name.group(1)] = (fn.group(1), data)
        else:
            out["fields"][name.group(1)] = data.decode("utf-8", "replace")
    return out


# ---------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    server_version = "UGCStudio/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        if "/static/" not in fmt % args:
            print(f"  {self.address_string()} {fmt % args}")

    # -------------------------------------------------------- send
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, data, code: int = 200):
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, msg: str, code: int = 400):
        self._json({"ok": False, "error": str(msg)}, code)

    def _body(self) -> bytes:
        # Cache the payload: a route that ignores the body must still drain it,
        # otherwise the leftover bytes are parsed as the next request line and
        # the browser sees "501 Unsupported method ('{}POST')".
        if getattr(self, "_raw_body", None) is not None:
            return self._raw_body
        n = int(self.headers.get("Content-Length") or 0)
        self._raw_body = self.rfile.read(n) if n else b""
        return self._raw_body

    def _jsonbody(self) -> dict:
        raw = self._body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _handle_upload(self):
        """Stage dropped files on disk and hand back absolute paths."""
        ctype = self.headers.get("Content-Type", "")
        parsed = _parse_multipart(self._body(), ctype)
        if not parsed.get("files"):
            return self._err("No files were received.")
        dest_dir = UPLOAD_DIR / time.strftime("%Y%m%d_%H%M%S")
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for _field, (fname, data) in parsed["files"].items():
            if not data:
                continue
            safe = re.sub(r"[^\w.\- ]+", "_", Path(fname or "file").name).strip() or "file"
            dest = dest_dir / safe
            i = 2
            while dest.exists():
                dest = dest_dir / f"{Path(safe).stem}_{i}{Path(safe).suffix}"
                i += 1
            dest.write_bytes(data)
            out.append({"name": dest.name, "path": str(dest),
                        "size": len(data), "kind": _kind_of(dest.name)})
        return self._json({"ok": True, "files": out, "folder": str(dest_dir)})

    # -------------------------------------------------------- file serve
    def _serve_download(self, path: Path):
        if not path.exists() or not path.is_file():
            self._err("File not found", 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{path.name}"')
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 262144)

    def _serve_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self._err("File not found", 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s or 0)
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(262144, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            except Exception:
                pass
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 262144)

    # -------------------------------------------------------- router
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self._serve_file(STATIC_DIR / "index.html")
            if p.startswith("/static/"):
                return self._serve_file(STATIC_DIR / p[len("/static/"):])
            if p == "/api/health":
                s = load_settings()
                from comfy import Comfy
                c = Comfy(s.get("comfy_url"))
                return self._json({"ok": True, "comfy_online": c.online(),
                                   "llm": llm.status(), "now": now()})
            if p == "/api/inventory":
                return self._json(inventory.inventory())
            if p == "/api/settings":
                s = load_settings()
                s["openrouter_key_masked"] = masked(s.get("openrouter_key", ""))
                s.pop("openrouter_key", None)
                s["gemini_key_masked"] = masked(s.get("gemini_key", ""))
                s.pop("gemini_key", None)
                # The UI opens this folder, so the server reports where it is
                # instead of the browser hardcoding one machine's path.
                s["characters_dir"] = str(pipeline.CHARACTERS_DIR)
                return self._json(s)
            if p == "/api/characters":
                return self._json({"ok": True,
                                   "characters": pipeline.list_characters()})
            if p == "/api/llm/status":
                return self._json(llm.status())
            if p == "/api/projects":
                return self._json(pipeline.list_projects())
            m = re.match(r"^/api/projects/([^/]+)/files$", p)
            if m:
                return self._json({"ok": True, **pipeline.project_files(m.group(1))})
            m = re.match(r"^/api/projects/([^/]+)$", p)
            if m:
                proj = pipeline.get(m.group(1))
                proj["job"] = _job_state(m.group(1))
                return self._json(proj)
            if p == "/api/file":
                pid = q.get("project", "")
                rel = q.get("path", "")
                root = (PROJECTS_DIR / pid).resolve()
                target = (root / rel).resolve()
                if not str(target).startswith(str(root)):
                    return self._err("Invalid path", 403)
                if q.get("download"):
                    self._serve_download(target)
                    return
                return self._serve_file(target)
            if p == "/api/staged":
                # Preview a file dropped for character creation, before it is
                # encoded into a .char. Only the staging folder is readable.
                target = Path(q.get("path", "")).resolve()
                base = UPLOAD_DIR.resolve()
                if not (target == base or base in target.parents):
                    return self._err("Invalid path", 403)
                return self._serve_file(target)
            if p == "/api/cfg":
                return self._json({"ok": True})
            self._err("Not found", 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._err(e, 500)

    def do_PATCH(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        # Drain the payload even when the route ignores it (see _body).
        self._body()
        try:
            m = re.match(r"^/api/projects/([^/]+)/scene/(\d+)$", p)
            if m:
                return self._json(pipeline.update_scene(m.group(1), int(m.group(2)), self._jsonbody()))
            m = re.match(r"^/api/projects/([^/]+)$", p)
            if m:
                return self._json(pipeline.update_project(m.group(1), self._jsonbody()))
            self._err("Not found", 404)
        except Exception as e:  # noqa: BLE001
            self._err(e, 500)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        # Drain the payload even when the route ignores it (see _body).
        self._body()
        try:
            if p == "/api/settings":
                body = self._jsonbody()
                s = load_settings()
                if "openrouter_key" in body:
                    key = (body.get("openrouter_key") or "").strip()
                    if key:
                        s["openrouter_key"] = key
                if body.get("clear_key"):
                    s["openrouter_key"] = ""
                for k in ("llm_mode", "openrouter_model", "allow_cloud_refs",
                          "profile", "video_profile", "comfy_url",
                          "overlay_enabled", "overlay_brand", "overlay_cta",
                          "overlay_captions", "gemini_key", "gemini_model",
                          "tts_enabled", "tts_voice",
                          "video_turbo", "video_steps", "audio_mode",
                          "character"):
                    if k in body:
                        s[k] = body[k]
                save_settings(s)
                return self._json({"ok": True})
            if p == "/api/llm/test":
                body = self._jsonbody()
                s = load_settings()
                gkey = s.get("gemini_key", "")
                key = s.get("openrouter_key", "")
                # Prefer Gemini when a key is configured: it is the provider
                # the user actually set up, and it costs no local VRAM.
                if gkey:
                    model = body.get("model") or s.get("gemini_model") or "gemini-3.8-flash"
                    res = llm.probe_structured(model, "gemini", gkey)
                else:
                    model = body.get("model") or s.get("openrouter_model")
                    if not model:
                        free = llm.free_models(key)
                        model = free[0]["id"] if free else "openrouter/free"
                    res = llm.probe_structured(model, "openrouter", key)
                return self._json({"ok": res.get("ok", False), "model": model, "detail": res})
            if p == "/api/projects":
                return self._json(pipeline.new_project(self._jsonbody()))
            if p == "/api/characters":
                # Reference photos are absolute local paths: VHS_LoadImagePath
                # and VHS_LoadAudio read them from disk, so nothing is uploaded.
                return self._json({"ok": True,
                                   **pipeline.save_character(self._jsonbody())})
            if p == "/api/characters/delete":
                return self._json({"ok": True,
                                   **pipeline.delete_character(self._jsonbody().get("name"))})
            if p == "/api/upload":
                return self._handle_upload()
            if p == "/api/reveal":
                target = Path(self._jsonbody().get("path") or APP_DIR).resolve()
                allowed = (APP_DIR.resolve(), PROJECTS_DIR.resolve(),
                           pipeline.CHARACTERS_DIR.resolve())
                if not any(target == a or a in target.parents for a in allowed):
                    return self._err("That path is outside the app's folders.", 403)
                _reveal(target)
                return self._json({"ok": True})
            m = re.match(r"^/api/projects/([^/]+)/files/delete$", p)
            if m:
                body = self._jsonbody()
                return self._json({"ok": True,
                                   **pipeline.delete_project_file(m.group(1),
                                                                  body.get("path"))})
            m = re.match(r"^/api/projects/([^/]+)/delete$", p)
            if m:
                return self._json({"ok": True,
                                   **pipeline.delete_project(m.group(1))})
            m = re.match(r"^/api/projects/([^/]+)/duplicate$", p)
            if m:
                return self._json({"ok": True,
                                   **pipeline.duplicate_project(m.group(1))})
            m = re.match(r"^/api/projects/([^/]+)/character/sheet$", p)
            if m:
                body = self._jsonbody()
                return self._json({"ok": True,
                                   "character": pipeline.set_character_sheet(
                                       m.group(1), body.get("path"))})
            m = re.match(r"^/api/projects/([^/]+)/(brief|plan|approve-storyboard|stills|videos|export|reorder|generate-campaign|character|stills/approve|stills/regenerate|stills/replace)$", p)
            if m:
                pid, action = m.group(1), m.group(2)
                body = self._jsonbody()
                if action == "brief":
                    return self._json(pipeline.run_brief(pid))
                if action == "plan":
                    return self._json(pipeline.run_plan(pid))
                if action == "approve-storyboard":
                    return self._json(pipeline.approve_storyboard(pid))
                if action == "reorder":
                    return self._json(pipeline.reorder_scenes(pid, body.get("order", [])))
                if action == "character":
                    _job_start(pid, "character", pipeline.run_character, pid, body)
                    return self._json({"ok": True, "job": _job_state(pid)})
                if action == "stills":
                    scenes = body.get("scenes")
                    _job_start(pid, "stills", pipeline.run_stills, pid, scenes)
                    return self._json({"ok": True, "job": _job_state(pid)})
                if action == "stills/approve":
                    return self._json(pipeline.approve_still(pid, int(body.get("scene")),
                                                             body.get("filename")))
                if action == "stills/regenerate":
                    scenes = body.get("scenes") or [body.get("scene")]
                    scenes = [int(x) for x in scenes if x]
                    for n in scenes:
                        pipeline.regenerate_still(pid, n, body.get("prompt"), body.get("seed"))
                    return self._json({"ok": True})
                if action == "stills/replace":
                    return self._json(pipeline.replace_still(
                        pid, int(body.get("scene")), body.get("path")))
                if action == "videos":
                    scenes = body.get("scenes")
                    seeds = body.get("seeds")
                    _job_start(pid, "videos", pipeline.run_videos, pid, scenes, seeds)
                    return self._json({"ok": True, "job": _job_state(pid)})
                if action == "export":
                    return self._json(pipeline.build_export(pid, body))
                if action == "generate-campaign":
                    def campaign():
                        project = pipeline.get(pid)
                        project["stage02"]["status"] = "generating"
                        pipeline.save(project)
                        pipeline.run_stills(pid)
                        pipeline.run_videos(pid)
                        pipeline.build_export(pid, {})
                    _job_start(pid, "campaign", campaign)
                    return self._json({"ok": True, "job": _job_state(pid)})
            m = re.match(r"^/api/projects/([^/]+)/upload$", p)
            if m:
                pid = m.group(1)
                ctype = self.headers.get("Content-Type", "")
                parsed = _parse_multipart(self._body(), ctype)
                kind = parsed["fields"].get("kind", "products")
                created = []
                tmpdir = Path(tempfile.mkdtemp(prefix="ugc_up_"))
                try:
                    for _name, (fname, data) in parsed["files"].items():
                        if not data:
                            continue
                        tmp = tmpdir / (fname or "upload.png")
                        tmp.write_bytes(data)
                        created.append(pipeline.add_reference(pid, kind, tmp, fname))
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return self._json({"ok": True, "added": created})
            self._err("Not found", 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._err(e, 500)


def main():
    ensure_dirs()
    (STATIC_DIR / "index.html").exists()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n  UGC Studio running at  http://{HOST}:{PORT}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
