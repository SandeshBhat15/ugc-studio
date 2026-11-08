"""Hardware, model inventory and recommended generation profiles."""
from __future__ import annotations

import json
import os
import subprocess

from core import disk_free_gb, comfy_dir, load_settings
from comfy import Comfy


def _disk_map() -> dict:
    """Free space on the drives that exist here, keyed by drive letter."""
    roots = []
    for p in (comfy_dir(), os.path.abspath(os.sep)):
        anchor = os.path.splitdrive(str(p))[0] + os.sep
        if anchor not in roots:
            roots.append(anchor)
    try:
        for d in os.listdrives():
            if d not in roots:
                roots.append(d)
    except (AttributeError, OSError):
        pass
    return {os.path.splitdrive(r)[0].rstrip(":") or r: disk_free_gb(r) for r in roots}

OPTIONAL_MODELS = [
    {"name": "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors",
     "size_gb": 4.22, "folder": "model_patches",
     "adds": "ControlNet-guided video: follow a reference video's motion/layout.",
     "why": "Only needed if you want reference-video control in Stage 03.",
     "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"},
    {"name": "minimax_h3_flf2v_lora_v1.safetensors",
     "size_gb": 0.15, "folder": "loras",
     "adds": "360-degree orbit / first-last-frame camera moves.",
     "why": "Optional cinematic orbit shots; not needed for standard UGC scenes.",
     "url": "https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/resolve/main/minimax_h3_flf2v_lora_v1.safetensors"},
    {"name": "p-h3-image_vae_int8_convrot.safetensors",
     "size_gb": 2.93, "folder": "vae",
     "adds": "Dedicated still-image VAE for the p-h3-image model family.",
     "why": "Not required — the installed H3 video VAE is verified to work for stills.",
     "url": "https://huggingface.co/PrunaAI/p-h3-image-comfy/resolve/main/vae/p-h3-image_vae_int8_convrot.safetensors"},
]

INCOMPATIBLE = [
    {"name": "p-h3-image_lora_bf16.safetensors",
     "reason": "Installed, but trained for the PrunaAI p-h3-image base model. Stacking it "
               "on MiniMax H3 Ref2VA produces chromatic corruption (verified by A/B test). "
               "Stage 02 deliberately does not use it."},
]

PROFILES = {
    "draft": {"label": "Draft", "short_edge": 384, "steps": 14,
              "note": "Fastest. Rough detail, good for blocking scenes."},
    "standard": {"label": "Standard", "short_edge": 576, "steps": 20,
                 "note": "Balanced. Recommended default."},
    "high": {"label": "High Quality", "short_edge": 768, "steps": 20,
             "note": "Native trained detail. Slow on 8 GB cards; video may run out of memory."},
}


def gpu_info() -> dict:
    try:
        ps = ("Get-CimInstance Win32_VideoController | "
              "Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
        items = [{"name": d.get("Name"), "driver": d.get("DriverVersion"),
                  "reported_ram_gb": round((d.get("AdapterRAM") or 0) / 1073741824, 1)}
                 for d in data]
    except Exception:
        items = []
    return {"adapters": items}


def inventory() -> dict:
    settings = load_settings()
    c = Comfy(settings.get("comfy_url", "http://127.0.0.1:8188"))
    stats, info = None, None
    online = c.online()
    if online:
        try:
            stats = c.stats()
        except Exception:
            stats = None
        try:
            info = c.object_info()
        except Exception:
            info = None

    devices = []
    if stats:
        for d in stats.get("devices", []):
            devices.append({
                "name": d.get("name"), "type": d.get("type"),
                "vram_total_gb": round((d.get("vram_total") or 0) / 1073741824, 2),
                "vram_free_gb": round((d.get("vram_free") or 0) / 1073741824, 2),
            })
    sysinfo = (stats or {}).get("system", {})

    def enum(cls: str, key: str) -> list[str]:
        try:
            return info[cls]["input"]["required"][key][0]
        except Exception:
            return []

    installed = {
        "diffusion_models": enum("UNETLoader", "unet_name"),
        "text_encoders": enum("CLIPLoader", "clip_name"),
        "vae": enum("VAELoader", "vae_name"),
        "loras": enum("LoraLoaderModelOnly", "lora_name"),
        "upscale_models": enum("UpscaleModelLoader", "model_name"),
    }
    have = {n for v in installed.values() for n in v}

    caps = c.capabilities() if online else {"online": False, "missing": c.REQUIRED}

    biggest = max(devices, key=lambda d: d["vram_total_gb"], default=None)
    vram = biggest["vram_total_gb"] if biggest else 0

    if vram >= 24:
        allowed = ["draft", "standard", "high"]
        rec = "high"
    elif vram >= 10:
        allowed = ["draft", "standard", "high"]
        rec = "standard"
    elif vram >= 7:
        allowed = ["draft", "standard"]
        rec = "standard"
    else:
        allowed = ["draft"]
        rec = "draft"

    optional = [dict(m, present=m["name"] in have,
                     disk_free_gb=disk_free_gb(comfy_dir())) for m in OPTIONAL_MODELS]

    return {
        "comfy": {
            "online": online, "url": settings.get("comfy_url"),
            "version": sysinfo.get("comfyui_version"),
            "python": sysinfo.get("python_version"),
            "torch": sysinfo.get("pytorch_version"),
            "classes": len(info) if info else 0,
            "missing_required": caps.get("missing", []),
        },
        "hardware": {
            "os": sysinfo.get("os"), "ram_total_gb": round((sysinfo.get("ram_total") or 0) / 1073741824, 1),
            "ram_free_gb": round((sysinfo.get("ram_free") or 0) / 1073741824, 1),
            "devices": devices,
            "adapters": gpu_info().get("adapters", []),
            "disk_free_gb": _disk_map(),
        },
        "models": installed,
        "optional_models": optional,
        "incompatible_models": INCOMPATIBLE,
        "profiles": PROFILES,
        "recommended_profile": rec,
        "allowed_profiles": allowed,
        "notes": [
            "Only one diffusion model is installed (Ref2VA). It drives both stills and video.",
            "No image checkpoint is installed or needed — H3 generates stills natively.",
            "No local LLM detected; the planner uses OpenRouter free models or an offline rule-based planner.",
        ],
    }
