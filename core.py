"""Core helpers: paths, JSON, HTTP. Standard library only."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = APP_DIR / "projects"
STATIC_DIR = APP_DIR / "static"
CONFIG_PATH = APP_DIR / "settings.json"

DEFAULT_COMFY = os.environ.get("UGC_COMFY_URL", "http://127.0.0.1:8188")

# ComfyUI is a separate install, so its location differs per machine. The
# env var wins, then the `comfy_dir` setting, then the default below.
DEFAULT_COMFY_DIR = os.environ.get("UGC_COMFY_DIR", r"E:\Comfy\ComfyUI\ComfyUI")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default=None):
    try:
        # utf-8-sig tolerates a UTF-8 BOM. Without it a BOM written by e.g.
        # PowerShell's `Set-Content -Encoding utf8` makes json.loads fail and
        # silently downgrades settings to defaults.
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def slugify(text: str, fallback: str = "project") -> str:
    s = re.sub(r"[^\w\s-]", "", (text or "").strip().lower())
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:60] or fallback


def unique_dir(base: Path, name: str) -> Path:
    cand = base / name
    if not cand.exists():
        return cand
    for i in range(2, 500):
        c = base / f"{name}_{i}"
        if not c.exists():
            return c
    return base / f"{name}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------- settings

def load_settings() -> dict:
    s = read_json(CONFIG_PATH, {}) or {}
    s.setdefault("comfy_url", DEFAULT_COMFY)
    s.setdefault("comfy_dir", DEFAULT_COMFY_DIR)  # ComfyUI install folder
    s.setdefault("llm_mode", "auto")          # auto | local | cloud
    s.setdefault("openrouter_key", "")
    s.setdefault("openrouter_model", "")      # "" = auto-pick free
    s.setdefault("allow_cloud_refs", False)   # privacy gate for uploading images
    s.setdefault("profile", "standard")       # draft | standard | high (stills)
    s.setdefault("video_profile", "draft")    # render size only: draft 384x672
    s.setdefault("video_turbo", True)         # turbo LoRA keeps 8 GB cards fast
    s.setdefault("video_steps", 0)            # 0 = auto by profile
    s.setdefault("audio_mode", "native")      # native = H3's own synced speech
    s.setdefault("character", "")             # .char name, e.g. "ananya.char"
    s.setdefault("overlay_enabled", True)     # burn real text into the final cut
    s.setdefault("overlay_brand", True)       # brand / product name
    s.setdefault("overlay_cta", True)         # call to action
    s.setdefault("overlay_captions", True)    # scene dialogue as captions
    s.setdefault("gemini_key", "")            # Google Gemini API key
    s.setdefault("gemini_model", "gemini-3.8-flash")  # planning model
    s.setdefault("tts_enabled", True)         # real voiceover via Gemini TTS
    s.setdefault("tts_voice", "Kore")         # fallback when gender is unknown
    # A voice must match the person on screen, so the pipeline picks by gender
    # instead of using one global voice for every creator.
    s.setdefault("tts_voice_female", "Kore")
    s.setdefault("tts_voice_male", "Puck")
    s.setdefault("last_model_check", 0)
    return s


def save_settings(settings: dict) -> None:
    write_json(CONFIG_PATH, settings)


def comfy_dir() -> Path:
    """Where ComfyUI is installed on this machine.

    Env var beats the saved setting so a portable install can be pointed at
    another drive without editing settings.json.
    """
    env = os.environ.get("UGC_COMFY_DIR")
    if env:
        return Path(env).expanduser()
    saved = (read_json(CONFIG_PATH, {}) or {}).get("comfy_dir") or DEFAULT_COMFY_DIR
    return Path(saved).expanduser()


def masked(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


# ---------------------------------------------------------------- http

def http_json(url: str, payload=None, headers=None, timeout: float = 60.0, method=None):
    data = None
    hdrs = {"Accept": "application/json", "User-Agent": "UGC-Studio/1.0"}
    if headers:
        hdrs.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


def http_get_text(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "UGC-Studio/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def http_upload(url: str, file_path: Path, field: str = "image", extra: dict | None = None) -> dict:
    """multipart/form-data upload without third-party libs."""
    boundary = "----UGCStudio" + uuid.uuid4().hex
    file_path = Path(file_path)
    parts = []
    for k, v in (extra or {}).items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{file_path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = b"".join(parts) + head + file_path.read_bytes() + tail
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(len(body))},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8", "replace") or "{}")


def wait_for_url(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 500
    except Exception:
        return False


def disk_free_gb(path: str | Path) -> float:
    try:
        return round(shutil.disk_usage(str(path)).free / (1024 ** 3), 1)
    except OSError:
        return -1.0


def retry(fn, attempts: int = 3, delay: float = 2.0, on_error=None):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if on_error:
                on_error(e, i)
            if i < attempts - 1:
                time.sleep(delay * (i + 1))
    raise last  # type: ignore[misc]
