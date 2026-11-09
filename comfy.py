"""ComfyUI client + API-format graph builders for native MiniMax H3."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from core import http_json, http_upload, slugify

REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
QWEN = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
# MiniMax H3 3D latent upscaler (658 MB, Efficient-Large-Model/Open-H3-Regenerate-2K-IC).
# Resizes an already-denoised H3 latent instead of re-rendering at the larger
# size, so the face and the lip timing cannot drift on the way up.
LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
UPSCALE = "RealESRGAN_x2.pth"

# Resolution profiles. Mimics Mickmumpitz MinimaxH3Resolution:
# short edge 768, area cap 768*1344 = 1032192, multiple of 32.
SCALES = {"draft": 0.5, "standard": 0.75, "high": 1.0}
ASPECTS = {"9:16": (9, 16), "1:1": (1, 1), "16:9": (16, 9)}


MAX_PIXELS = 768 * 1344  # 1032192, the trained pixel budget


def resolve_size(aspect: str, profile: str) -> tuple[int, int]:
    ratio = ASPECTS.get(aspect, (9, 16))
    scale = SCALES.get(profile, 0.75)
    short = int(round(768 * scale / 32)) * 32
    w, h = ratio
    long_edge = int(round(short * max(w, h) / min(w, h) / 32)) * 32
    # keep within the trained pixel budget (constant, not scaled)
    while short * long_edge > MAX_PIXELS:
        long_edge -= 32
    long_edge = max(long_edge, short)
    return (long_edge, short) if w > h else (short, long_edge)


def frames_for(seconds: float, fps: int = 24) -> int:
    """Snap to the model's valid 5 + 17k grid."""
    raw = max(5, round(seconds * fps))
    return raw + ((5 - (raw % 17)) % 17)


class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex

    # ---------------------------------------------------------- probes
    def online(self) -> bool:
        try:
            http_json(f"{self.base}/system_stats", timeout=4)
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        return http_json(f"{self.base}/system_stats", timeout=10)

    def object_info(self) -> dict:
        return http_json(f"{self.base}/object_info", timeout=90)

    def queue_state(self) -> dict:
        return http_json(f"{self.base}/queue", timeout=10)

    # ---------------------------------------------------------- inputs
    def upload_image(self, path: Path) -> str:
        name = f"ugc_{uuid.uuid4().hex[:10]}_{Path(path).name}"
        res = http_upload(f"{self.base}/upload/image", Path(path), field="image",
                          extra={"overwrite": "true"})
        return res.get("name", name)

    # ---------------------------------------------------------- run
    def queue(self, graph: dict) -> str:
        try:
            res = http_json(f"{self.base}/prompt",
                            {"prompt": graph, "client_id": self.client_id}, timeout=120)
        except RuntimeError as e:
            raise ComfyError(self._explain(str(e))) from e
        if "prompt_id" not in res:
            raise ComfyError(f"ComfyUI rejected the job: {res}")
        return res["prompt_id"]

    @staticmethod
    def _explain(msg: str) -> str:
        low = msg.lower()
        if "missing node" in low or "class_type" in low:
            return f"A required node is not installed in ComfyUI. {msg}"
        if "value not in list" in low or "not in list" in low:
            return f"A model file is missing from this machine. {msg}"
        return msg

    def history(self, prompt_id: str) -> dict:
        return http_json(f"{self.base}/history/{prompt_id}", timeout=30)

    def wait(self, prompt_id: str, timeout: float = 3600) -> dict:
        t0 = time.time()
        delay = 1.5
        while time.time() - t0 < timeout:
            try:
                h = self.history(prompt_id)
            except Exception:
                h = {}
            entry = h.get(prompt_id) if isinstance(h, dict) else None
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise ComfyError(f"Generation failed: {msgs[-1] if msgs else 'unknown error'}")
                if status.get("completed"):
                    return entry
            time.sleep(delay)
            delay = min(delay + 0.5, 5.0)
        raise ComfyError("Timed out waiting for ComfyUI.")

    def cancel(self, prompt_id: str) -> None:
        try:
            http_json(f"{self.base}/interrupt", {}, timeout=10)
        except Exception:
            pass

    # ---------------------------------------------------------- graphs
    def image_graph(self, prompt: str, width: int, height: int,
                    ref_images: list[str], prefix: str, steps: int = 20,
                    seed: int | None = None, upscale: bool = True) -> dict:
        """Native H3 still: PH3ImageConditioning (text-to-image or edit)."""
        g: dict = {
            "1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": REF2VA, "weight_dtype": "default"}},
            "2": {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["1", 0], "shift_video": 1.0, "shift_audio": 1.0}},
            "3": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": QWEN, "type": "minimax", "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
            "5": {"class_type": "PH3ImageConditioning",
                  "inputs": {"clip": ["3", 0], "prompt": prompt,
                             "width": int(width), "height": int(height),
                             "vae": ["4", 0]}},
            "6": {"class_type": "BasicGuider",
                  "inputs": {"model": ["2", 0], "conditioning": ["5", 0]}},
            "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "8": {"class_type": "BasicScheduler",
                  "inputs": {"model": ["2", 0], "scheduler": "beta",
                             "steps": int(steps), "denoise": 1.0}},
            "9": {"class_type": "RandomNoise",
                  "inputs": {"noise_seed": int(seed if seed is not None else 42)}},
            "10": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["9", 0], "guider": ["6", 0], "sampler": ["7", 0],
                              "sigmas": ["8", 0], "latent_image": ["5", 1]}},
            "11": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        }
        # reference images -> autogrow inputs (image_1..image_9)
        for i, name in enumerate(ref_images[:9], start=1):
            key = f"li{i}"
            g[key] = {"class_type": "LoadImage",
                      "inputs": {"image": name, "upload": "image"}}
            g["5"]["inputs"][f"images.image_{i}"] = [key, 0]
        out = ["11", 0]
        if upscale:
            g["12"] = {"class_type": "UpscaleModelLoader",
                       "inputs": {"model_name": UPSCALE}}
            g["13"] = {"class_type": "ImageUpscaleWithModel",
                       "inputs": {"upscale_model": ["12", 0], "image": ["11", 0]}}
            out = ["13", 0]
        g["99"] = {"class_type": "SaveImage",
                   "inputs": {"images": out, "filename_prefix": prefix}}
        return g

    def video_graph(self, image_name: str, prompt: str, width: int, height: int,
                    length: int, prefix: str, seed: int | None = None,
                    turbo: bool = True, audio: bool = True,
                    ref_audio: str | None = None,
                    steps: int | None = None,
                    character: str | None = None,
                    upscale: tuple[int, int] | None = None) -> dict:
        """Native H3 video from an approved still, official pack defaults.

        Passing `ref_audio` (an absolute WAV path) conditions the model on the
        real voiceover, so mouth movement is generated to match the speech
        instead of the model inventing its own babble track.

        Passing `character` (a .char name, e.g. "ananya.char") swaps the single
        still for an Omnichar character: its face/body/outfit references become
        <Picture 1..3> and the prompt is prefixed so the model knows which
        reference is which. That is what keeps the same person across scenes.
        """
        g: dict = {
            "1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": REF2VA, "weight_dtype": "default"}},
            "2": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": TURBO_LORA,
                             # The turbo LoRA is distilled for ~8 steps. On the
                             # quality path (more steps, res_multistep) it
                             # over-cooks the image, so neutralise it there.
                             "strength_model": 1.0 if turbo else 0.0}},
            "3": {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["2", 0], "shift_video": 12.0, "shift_audio": 3.0}},
            "4": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": QWEN, "type": "minimax", "device": "default"}},
            "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
            "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
            "7": {"class_type": "VHS_LoadImagePath",
                  "inputs": {"image": image_name, "custom_width": 0, "custom_height": 0}},
            "8": {"class_type": "MiniMaxH3ReferenceToVideo",
                  "inputs": {"clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0],
                             "ref_images.ref_image_0": ["7", 0],
                             "prompt": prompt, "width": int(width), "height": int(height),
                             "length": int(length), "ref_image_size": "match"}},
            "9": {"class_type": "BasicGuider",
                  "inputs": {"model": ["3", 0], "conditioning": ["8", 0]}},
            "10": {"class_type": "KSamplerSelect",
                   "inputs": {"sampler_name": "euler" if turbo else "res_multistep"}},
            "11": {"class_type": "BasicScheduler",
                   "inputs": {"model": ["3", 0], "scheduler": "beta",
                              "steps": int(steps if steps is not None
                                           else (8 if turbo else 50)),
                              "denoise": 1.0}},
            "12": {"class_type": "RandomNoise",
                   "inputs": {"noise_seed": int(seed if seed is not None else 42)}},
            "13": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["12", 0], "guider": ["9", 0], "sampler": ["10", 0],
                              "sigmas": ["11", 0], "latent_image": ["8", 1]}},
            "14": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        }
        if character:
            # The approved still STAYS at ref_image_0: it is the frame being
            # animated, so it owns the composition, blocking and wardrobe the
            # user actually approved. The character supplies identity only,
            # riding in the slots after it. Dropping the still here (as this
            # used to) silently replaced every scene with the .char identity -
            # which is how male stills turned into a female video.
            g["20"] = {"class_type": "OmnicharLoadCharacter",
                       "inputs": {"char": character, "char_path": ""}}
            g["21"] = {"class_type": "OmnicharDecodeCharacter",
                       "inputs": {"char": ["20", 0], "style": "token",
                                  "prompt": prompt, "arch": "originals",
                                  "max_references": 0, "size_from": "first",
                                  "fit": "pad",
                                  # We supply our own per-scene voiceover below,
                                  # so the stored voice clip is not sent.
                                  "voice": "never", "audio_position": 1}}
            # Do not swap the prompt for the character's stored description:
            # a .char written for one person would otherwise overrule a still
            # of someone else. Identity comes from the reference images, the
            # same mechanism the stills pass uses.
            for i in range(3):
                node = str(22 + i)
                g[node] = {"class_type": "OmnicharCharacterReference",
                           "inputs": {"refs": ["21", 2], "index": i}}
                # +1 keeps ref_image_0 for the approved still.
                g["8"]["inputs"][f"ref_images.ref_image_{i + 1}"] = [node, 0]

        if audio:
            g["15"] = {"class_type": "VAEDecodeAudio",
                       "inputs": {"samples": ["13", 0], "vae": ["6", 0]}}
            g["16"] = {"class_type": "CreateVideo",
                       "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24}}
        else:
            g["16"] = {"class_type": "CreateVideo",
                       "inputs": {"images": ["14", 0], "fps": 24}}
        if ref_audio:
            # Driving the generation with the actual voiceover is what makes
            # the lips match the words: the audio VAE conditions the model on
            # this exact signal, so the clip length must equal the speech.
            g["17"] = {"class_type": "VHS_LoadAudio",
                       "inputs": {"audio_file": str(ref_audio)}}
            g["8"]["inputs"]["ref_audios.ref_audio_0"] = ["17", 0]

        if upscale:
            # Resize the finished latent, then decode at the bigger size. The
            # audio branch reads the same latent, so the voiceover and its lip
            # sync pass through the upscale untouched.
            up_w, up_h = int(upscale[0]), int(upscale[1])
            g["30"] = {"class_type": "LatentUpscaleModelLoader",
                       "inputs": {"model_name": LATENT_UPSCALER}}
            g["31"] = {"class_type": "MiniMaxH3LatentUpscaleWithModel",
                       "inputs": {"model": ["30", 0], "samples": ["13", 0],
                                  "width": up_w, "height": up_h}}
            g["14"]["inputs"]["samples"] = ["31", 0]
            g["15"]["inputs"]["samples"] = ["31", 0]

        g["99"] = {"class_type": "SaveVideo",
                   "inputs": {"video": ["16", 0], "filename_prefix": prefix,
                              "format": "mp4", "codec": "h264"}}
        return g

    def character_graph(self, name: str, description: str,
                        images: dict[str, list[str]], voice: str | None = None,
                        resolution: int = 1024, prefix: str | None = None) -> dict:
        """Omnichar `.char`: one reusable identity built from reference photos.

        The encoder takes each reference ROLE as its own numbered slot rather
        than one flat list, because the SDK bakes a different part of the
        identity from each: `face` drives the head, `body` the build, and
        `cloths` the wardrobe. Supplying all three is what keeps the same
        person — and the same outfit — consistent in every later scene.

        `prefix` is accepted for symmetry with the other builders; the save
        node decides its own path inside ComfyUI's models/characters folder.
        """
        g: dict = {}
        n = 0

        def add(class_type: str, inputs: dict) -> str:
            nonlocal n
            n += 1
            g[str(n)] = {"class_type": class_type, "inputs": inputs}
            return str(n)

        enc: dict = {"name": name, "description": description,
                     "resolution": int(resolution)}
        for role in ("face", "body", "cloths"):
            # VHS_LoadImagePath reads the absolute path directly, so reference
            # photos never need uploading into ComfyUI's input folder.
            for i, path in enumerate((images.get(role) or [])[:3]):
                node = add("VHS_LoadImagePath",
                           {"image": str(Path(path).as_posix()),
                            "custom_width": 0, "custom_height": 0})
                # first slot of a role is unsuffixed, then _2 / _3
                slot = role if i == 0 else f"{role}_{i + 1}"
                enc[slot] = [node, 0]

        if voice:
            enc["voice"] = [add("VHS_LoadAudio",
                                {"audio_file": str(Path(voice).as_posix())}), 0]

        encode = add("OmnicharEncodeCharacter", enc)
        add("OmnicharSaveCharacter",
            {"char": [encode, 0], "filename": f"{slugify(name, 'character')}.char",
             "overwrite": True})
        return g

    # ---------------------------------------------------------- capability check
    REQUIRED = ["PH3ImageConditioning", "MiniMaxH3ReferenceToVideo",
                "MiniMaxH3SigmaShift", "UNETLoader", "CLIPLoader", "VAELoader",
                "VHS_LoadImagePath", "CreateVideo", "SaveVideo", "SaveImage",
                "UpscaleModelLoader", "ImageUpscaleWithModel"]

    def capabilities(self) -> dict:
        try:
            info = self.object_info()
        except Exception as e:
            return {"online": False, "error": str(e), "missing": self.REQUIRED}
        missing = [c for c in self.REQUIRED if c not in info]
        return {"online": True, "classes": len(info), "missing": missing}
