"""FFmpeg helpers: probing, normalising, concatenation, export."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# FFMPEG_BIN / FFPROBE_BIN win, then a bare `ffmpeg` / `ffprobe` on PATH.
# Nothing else is required: install ffmpeg anywhere and put it on PATH.
FFMPEG_CANDIDATES = [
    c for c in (os.environ.get("FFMPEG_BIN"), "ffmpeg") if c
]
FFPROBE_CANDIDATES = [
    c for c in (os.environ.get("FFPROBE_BIN"), "ffprobe") if c
]


def _first_available(cands) -> str | None:
    for c in cands:
        if c == "ffmpeg" or c == "ffprobe":
            if shutil.which(c):
                return c
            continue
        if Path(c).exists():
            return c
    return None


def ffmpeg() -> str | None:
    return _first_available(FFMPEG_CANDIDATES)


def ffprobe_bin() -> str | None:
    return _first_available(FFPROBE_CANDIDATES)


def probe(path: str | Path) -> dict:
    exe = ffprobe_bin()
    if not exe:
        return {"error": "ffprobe not found"}
    cmd = [exe, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(out.stdout or "{}")
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fps = 0.0
    try:
        num, den = (v.get("avg_frame_rate") or "0/1").split("/")
        fps = round(float(num) / float(den), 3) if float(den) else 0.0
    except Exception:
        pass
    return {
        "width": v.get("width"), "height": v.get("height"),
        "fps": fps, "codec": v.get("codec_name"),
        "has_audio": bool(a), "audio_codec": a.get("codec_name"),
        "duration": round(float(data.get("format", {}).get("duration") or 0), 3),
        "size_mb": round(float(data.get("format", {}).get("size") or 0) / 1048576, 2),
    }


def normalise(src: Path, dst: Path, width: int, height: int, fps: int = 24,
              with_audio: bool = True) -> Path:
    """Re-encode one clip to a canonical canvas so concatenation is safe."""
    exe = ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found")
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}")
    cmd = [exe, "-y", "-i", str(src)]
    if not with_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += [str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg normalise failed: {res.stderr[-600:]}")
    return dst


def concat(clips: list[Path], dst: Path, workdir: Path) -> Path:
    exe = ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found")
    listfile = workdir / "_concat.txt"
    listfile.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
    cmd = [exe, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
           "-c", "copy", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        # fall back to re-encode
        cmd = [exe, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
               "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(dst)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {res.stderr[-600:]}")
    return dst


def grab_frame(video: Path, dst: Path, at: float = 1.0) -> Path:
    exe = ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found")
    subprocess.run([exe, "-y", "-ss", str(at), "-i", str(video), "-frames:v", "1", str(dst)],
                   capture_output=True, text=True, timeout=120)
    return dst


def extract_audio(video: Path, dst: Path) -> Path:
    exe = ffmpeg()
    subprocess.run([exe, "-y", "-i", str(video), "-vn", "-c:a", "aac", str(dst)],
                   capture_output=True, text=True, timeout=600)
    return dst


def _filter_path(p: Path) -> str:
    """Path in the form ffmpeg filter arguments expect (C:/a/b -> C\\:/a/b)."""
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _pick_font(bold: bool = True) -> str | None:
    names = (["arialbd.ttf", "seguisb.ttf", "arial.ttf"] if bold
             else ["arial.ttf", "segoeui.ttf"])
    for n in names:
        p = Path("C:/Windows/Fonts") / n
        if p.exists():
            return _filter_path(p)
    return None


def burn_text(video: Path, dst: Path, lines, width: int, height: int,
              workdir: Path | None = None) -> Path:
    """Composite REAL text over the video (never model-drawn glyphs).

    lines: iterable of dicts {"text": str, "pos": "top"|"center"|"lower"|"bottom",
                              "size": float (multiplier of width), "bold": bool}
    Uses ffmpeg's drawtext with textfile= so no escaping or % expansion issues.
    """
    exe = ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found")
    lines = [l for l in (lines or []) if (l or {}).get("text")]
    if not lines:
        shutil.copy2(video, dst)
        return dst

    workdir = Path(workdir or dst.parent)
    workdir.mkdir(parents=True, exist_ok=True)

    filters = []
    for i, spec in enumerate(lines):
        tf = workdir / f"_txt_{os.getpid()}_{i}.txt"
        tf.write_text(str(spec["text"]), encoding="utf-8")
        size = int(width * float(spec.get("size", 0.055)))
        raw = str(spec["text"]).strip()
        # Auto-shrink so a long line can never overflow the frame. Rough
        # bold-sans advance width is ~0.55em per character; keep it inside 92%.
        max_w = int(width * 0.92)
        est = int(len(raw) * size * 0.55)
        if est > max_w and est > 0:
            size = max(10, int(size * max_w / est))
        pos = spec.get("pos", "bottom")
        if pos == "top":
            y = "h*0.06"
        elif pos == "center":
            y = "(h-text_h)/2"
        elif pos == "lower":
            y = "h*0.74-text_h"
        else:
            y = "h*0.88-text_h"
        parts = [
            "drawtext=expansion=none",
            f"textfile='{_filter_path(tf)}'",
            "fontcolor=white",
            f"fontsize={size}",
            f"borderw={max(2, size // 12)}",
            "bordercolor=black@0.85",
            "box=1", "boxcolor=black@0.28", f"boxborderw={max(4, size // 4)}",
            "x=(w-text_w)/2", f"y={y}",
        ]
        font = _pick_font(bool(spec.get("bold", True)))
        if font:
            parts.insert(1, f"fontfile='{font}'")
        filters.append(":".join(parts))

    cmd = [exe, "-y", "-i", str(video), "-vf", ",".join(filters),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
           str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    for i in range(len(filters)):
        try:
            (workdir / f"_txt_{os.getpid()}_{i}.txt").unlink()
        except OSError:
            pass
    if r.returncode != 0 or not dst.exists():
        err = (r.stderr or r.stdout or "unknown ffmpeg error").strip().splitlines()
        raise RuntimeError("text overlay failed: " + " | ".join(err[-3:]))
    return dst


def loudnorm(src, dst, target: float = -16.0) -> Path:
    """Level the audio track to a consistent loudness; frames are copied."""
    exe = ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found")
    cmd = [exe, "-y", "-i", str(src), "-c:v", "copy",
           "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
           "-movflags", "+faststart", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0 or not Path(dst).exists():
        raise RuntimeError("loudnorm failed: " + (p.stderr or "")[-400:])
    return dst


def replace_audio(video, audio, dst, workdir=None):
    """Mux `audio` over `video`, padding with silence and trimming to the clip.

    The video stream is copied untouched; only the audio track is replaced, so
    this costs no re-encode of the frames.
    """
    cmd = [
        ffmpeg(), "-y",
        "-i", str(video),
        "-i", str(audio),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(dst),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("audio mux failed: " + (p.stderr or "")[-400:])
    return dst