"""LLM provider manager: local (Ollama / LM Studio) + OpenRouter free tier.

No third-party deps. Never spends money: only models priced at 0 are eligible.
"""
from __future__ import annotations

import base64
import json
import struct
import time
from dataclasses import dataclass

from core import http_json, http_get_text, load_settings

OPENROUTER = "https://openrouter.ai/api/v1"
OLLAMA = "http://127.0.0.1:11434"
LMSTUDIO = "http://127.0.0.1:1234"
GEMINI = "https://generativelanguage.googleapis.com/v1beta"

# Tried in order when the primary model's free-tier quota is spent (HTTP 429).
CHAT_FALLBACKS = ["gemini-3.5-flash", "gemini-2.5-flash"]
TTS_FALLBACKS = ["gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview"]

# Filled in by gemini_tts so callers can record which model actually spoke and
# whether the delivery style survived.
LAST_TTS: dict = {"model": "", "styled": False}


@dataclass
class Provider:
    kind: str          # "ollama" | "lmstudio" | "openrouter" | "offline"
    model: str
    label: str
    free: bool = True


# ------------------------------------------------------------------ local

def detect_ollama() -> dict:
    try:
        tags = http_json(f"{OLLAMA}/api/tags", timeout=3)
        models = [m.get("name", "") for m in tags.get("models", [])]
        return {"available": True, "models": models, "url": OLLAMA}
    except Exception as e:
        return {"available": False, "models": [], "error": str(e)[:200]}


def detect_lmstudio() -> dict:
    try:
        data = http_json(f"{LMSTUDIO}/v1/models", timeout=3)
        models = [m.get("id", "") for m in data.get("data", [])]
        return {"available": True, "models": models, "url": LMSTUDIO}
    except Exception as e:
        return {"available": False, "models": [], "error": str(e)[:200]}


def local_providers() -> list[Provider]:
    out: list[Provider] = []
    ol = detect_ollama()
    if ol["available"]:
        for m in ol["models"]:
            out.append(Provider("ollama", m, f"Ollama · {m}", True))
    lm = detect_lmstudio()
    if lm["available"]:
        for m in lm["models"]:
            out.append(Provider("lmstudio", m, f"LM Studio · {m}", True))
    return out


# ------------------------------------------------------------------ openrouter

def openrouter_models(key: str = "") -> list[dict]:
    """Full catalogue. Works without a key (catalogue is public)."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = http_json(f"{OPENROUTER}/models", headers=headers, timeout=30)
    return data.get("data", [])


def _is_free(m: dict) -> bool:
    pricing = m.get("pricing", {}) or {}
    try:
        return float(pricing.get("prompt", 1)) == 0.0 and float(pricing.get("completion", 1)) == 0.0
    except (TypeError, ValueError):
        return False


def _ctx(m: dict) -> int:
    n = m.get("context_length") or (m.get("top_provider") or {}).get("context_length") or 0
    try:
        return int(n)
    except (TypeError, ValueError):
        return 0


def free_models(key: str = "", min_context: int = 8000) -> list[dict]:
    """Free models that can plausibly do structured output, best first."""
    try:
        models = openrouter_models(key)
    except Exception:
        return []
    cand = []
    for m in models:
        if not _is_free(m):
            continue
        if _ctx(m) < min_context:
            continue
        mods = (m.get("architecture") or {}).get("input_modalities") or ["text"]
        cand.append({
            "id": m.get("id"),
            "name": m.get("name") or m.get("id"),
            "context": _ctx(m),
            "vision": "image" in mods,
        })
    # prefer larger context, then vision, then shorter ids (usually flagship)
    cand.sort(key=lambda m: (m["vision"], m["context"]), reverse=True)
    return cand


def openrouter_chat(key: str, model: str, messages: list[dict],
                    temperature: float = 0.6, max_tokens: int = 3000,
                    timeout: float = 120) -> str:
    if not key:
        raise RuntimeError("OpenRouter API key is not configured.")
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    res = http_json(f"{OPENROUTER}/chat/completions", payload,
                    headers={"Authorization": f"Bearer {key}",
                             "HTTP-Referer": "http://localhost",
                             "X-Title": "UGC Studio"},
                    timeout=timeout)
    if "error" in res:
        raise RuntimeError(res["error"].get("message", str(res["error"])))
    choices = res.get("choices") or []
    if not choices:
        raise RuntimeError("Empty response from model.")
    return choices[0]["message"].get("content") or ""


# ------------------------------------------------------------------ unified

def probe_structured(model: str, provider: str, key: str = "") -> dict:
    """Small deterministic JSON test before trusting a model."""
    msgs = [{"role": "user",
             "content": 'Reply with ONLY minified JSON, no prose: {"ok": true, "n": 3}'}]
    t0 = time.time()
    try:
        if provider == "openrouter":
            raw = openrouter_chat(key, model, msgs, temperature=0, max_tokens=100, timeout=60)
        elif provider == "gemini":
            raw = gemini_chat(key, model, msgs, temperature=0, max_tokens=800, timeout=60)
        elif provider == "ollama":
            res = http_json(f"{OLLAMA}/api/chat",
                            {"model": model, "messages": msgs, "stream": False,
                             "options": {"temperature": 0}}, timeout=120)
            raw = res.get("message", {}).get("content", "")
        elif provider == "lmstudio":
            res = http_json(f"{LMSTUDIO}/v1/chat/completions",
                            {"model": model, "messages": msgs, "temperature": 0,
                             "max_tokens": 100}, timeout=120)
            raw = res["choices"][0]["message"]["content"]
        else:
            return {"ok": False, "error": "unknown provider"}
        txt = raw.strip()
        start, end = txt.find("{"), txt.rfind("}")
        parsed = json.loads(txt[start:end + 1]) if start >= 0 else None
        return {"ok": bool(parsed and parsed.get("ok") is True),
                "seconds": round(time.time() - t0, 2), "sample": txt[:120]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "seconds": round(time.time() - t0, 2)}


def chat(settings: dict, messages: list[dict], task: str = "plan",
         temperature: float = 0.6, max_tokens: int = 4000) -> dict:
    """Try providers per priority. Returns {text, model, provider}."""
    mode = settings.get("llm_mode", "auto")
    key = settings.get("openrouter_key", "")
    errors: list[str] = []

    if mode in ("auto", "local"):
        for p in local_providers():
            try:
                if p.kind == "ollama":
                    res = http_json(f"{OLLAMA}/api/chat",
                                    {"model": p.model, "messages": messages, "stream": False,
                                     "options": {"temperature": temperature}}, timeout=240)
                    text = res.get("message", {}).get("content", "")
                else:
                    res = http_json(f"{LMSTUDIO}/v1/chat/completions",
                                    {"model": p.model, "messages": messages,
                                     "temperature": temperature, "max_tokens": max_tokens},
                                    timeout=240)
                    text = res["choices"][0]["message"]["content"]
                if text.strip():
                    return {"text": text, "model": p.model, "provider": p.kind}
            except Exception as e:
                errors.append(f"{p.label}: {str(e)[:150]}")

    gkey = settings.get("gemini_key", "")
    if mode in ("auto", "cloud") and gkey:
        gmodel = settings.get("gemini_model", "") or "gemini-3.8-flash"
        try:
            text = gemini_chat(gkey, gmodel, messages,
                               temperature=temperature, max_tokens=max_tokens)
            if text.strip():
                return {"text": text, "model": gmodel, "provider": "gemini"}
        except Exception as e:
            errors.append(f"gemini/{gmodel}: {str(e)[:150]}")

    if mode in ("auto", "cloud") and key:
        preferred = settings.get("openrouter_model", "")
        order: list[str] = []
        if preferred:
            order.append(preferred)
        for m in free_models(key)[:6]:
            if m["id"] not in order:
                order.append(m["id"])
        for model in order:
            try:
                text = openrouter_chat(key, model, messages,
                                       temperature=temperature, max_tokens=max_tokens)
                if text.strip():
                    return {"text": text, "model": model, "provider": "openrouter"}
            except Exception as e:
                errors.append(f"{model}: {str(e)[:150]}")

    raise RuntimeError(
        "No LLM provider available. " + ("; ".join(errors[-3:]) if errors
                                         else "Configure OpenRouter or start Ollama/LM Studio.")
    )


def status() -> dict:
    s = load_settings()
    ol = detect_ollama()
    lm = detect_lmstudio()
    key = s.get("openrouter_key", "")
    gkey = s.get("gemini_key", "")
    free: list[dict] = []
    orerr = ""
    # The catalogue is public, so populate the picker even before a key is
    # pasted — otherwise the Settings dropdown is empty and looks broken.
    try:
        free = free_models(key)[:8]
    except Exception as e:
        orerr = str(e)[:200]
    return {
        "mode": s.get("llm_mode", "auto"),
        "has_openrouter_key": bool(key),
        "openrouter_key_masked": (key[:6] + "..." + key[-4:]) if len(key) > 10 else ("" if not key else "***"),
        "has_gemini_key": bool(gkey),
        "gemini_key_masked": (gkey[:6] + "..." + gkey[-4:]) if len(gkey) > 10 else ("" if not gkey else "***"),
        "gemini_model": s.get("gemini_model", "") or "gemini-3.8-flash",
        "ollama": ol,
        "lmstudio": lm,
        "free_models": free,
        "openrouter_error": orerr,
        "selected_model": s.get("openrouter_model", ""),
        "allow_cloud_refs": s.get("allow_cloud_refs", False),
    }


def gemini_models(key: str) -> list[dict]:
    """List Gemini models that support generateContent."""
    res = http_json(f"{GEMINI}/models", None,
                    headers={"x-goog-api-key": key}, timeout=20, method="GET")
    out: list[dict] = []
    for m in (res.get("models") or []):
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        name = str(m.get("name", "")).replace("models/", "")
        out.append({
            "id": name,
            "name": m.get("displayName") or name,
            "context": m.get("inputTokenLimit") or 0,
            "vision": True,
        })
    return out


def gemini_chat(key: str, model: str, messages: list[dict],
                temperature: float = 0.6, max_tokens: int = 3000,
                thinking_budget: int = 0,
                timeout: float = 180.0) -> str:
    """Chat via Gemini generateContent.

    System prompts are folded into the first user turn, which Gemini handles
    reliably for structured-output tasks.
    """
    contents: list[dict] = []
    system = ""
    for m in messages:
        role = m.get("role")
        text = str(m.get("content") or "")
        if role == "system":
            system = (system + "\n" + text).strip()
            continue
        contents.append({"role": "model" if role == "assistant" else "user",
                         "parts": [{"text": text}]})
    if system:
        if contents and contents[0]["role"] == "user":
            contents[0]["parts"][0]["text"] = system + "\n\n" + contents[0]["parts"][0]["text"]
        else:
            contents.insert(0, {"role": "user", "parts": [{"text": system}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Continue."}]}]

    payload = {"contents": contents,
               "generationConfig": {
                   "temperature": temperature,
                   "maxOutputTokens": max_tokens,
                   # Gemini 3.x burns output tokens on internal reasoning
                   # (thoughtsTokenCount) and can return EMPTY content with
                   # finishReason MAX_TOKENS. 0 disables thinking so structured
                   # output is reliable and fast.
                   "thinkingConfig": {"thinkingBudget": thinking_budget},
               }}
    # Free-tier keys allow only ~20 requests/day per model, and a spent quota
    # surfaces as HTTP 429 or a stalled read. Falling through to another
    # model keeps planning working instead of failing the whole stage.
    tried: list[str] = []
    last: Exception | None = None
    for name in [model] + [m for m in CHAT_FALLBACKS if m != model]:
        tried.append(name)
        try:
            res = http_json(f"{GEMINI}/models/{name}:generateContent",
                            payload, headers={"x-goog-api-key": key},
                            timeout=timeout, method="POST")
            break
        except Exception as e:
            msg = str(e)
            retryable = ("HTTP 429" in msg or "quota" in msg.lower()
                         or "timed out" in msg.lower() or "Timeout" in type(e).__name__)
            last = e
            if not retryable:
                raise
    else:
        raise RuntimeError("Gemini quota exhausted on every model tried "
                           f"({', '.join(tried)}): {last}")
    cands = res.get("candidates") or [{}]
    parts = ((cands[0].get("content") or {}).get("parts")) or []
    return "".join(str(p.get("text") or "") for p in parts)


def _ensure_wav(data: bytes, rate: int = 24000) -> bytes:
    """Wrap headerless PCM in a RIFF header.

    Some models ignore the requested `audio/wav` and answer with raw
    `audio/l16`. Written to disk as-is that file has no duration, so ffprobe
    reports 0 s and the render gets a broken reference track.
    """
    if data[:4] == b"RIFF":
        return data
    n = len(data) - len(data) % 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + n, b"WAVE",
        b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16,
        b"data", n)
    return header + data[:n]


def pcm_peak(wav: bytes) -> int:
    """Largest absolute 16-bit sample, so silence can be told from speech."""
    if wav[:4] == b"RIFF":
        i = wav.find(b"data", 12)
        payload = wav[i + 8:] if i != -1 else wav[44:]
    else:
        payload = wav          # older models return headerless PCM
    payload = payload[:len(payload) // 2 * 2]
    peak = 0
    for (s,) in struct.iter_unpack("<h", payload):
        a = s if s >= 0 else -s
        if a > peak:
            peak = a
    return peak


def audio_ok(wav: bytes) -> bool:
    """Reject the failure mode that shipped a silent take: valid but silent.

    `gemini-3.8-flash-lite-tts` answers a styled request with 228 kB of zero
    samples. Without this check that file becomes the lip-sync reference and
    the finished clip is silent.
    """
    if not wav or len(wav) < 4000:
        return False
    if wav[:4] != b"RIFF":
        return False
    return pcm_peak(wav) > 64


def _tts_legacy(key: str, text: str, model: str, voice: str,
                timeout: float) -> bytes:
    """Plain speech, no delivery control. Used when the style path fails."""
    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }
    res = http_json(f"{GEMINI}/models/{model}:generateContent", body,
                    headers={"x-goog-api-key": key},
                    timeout=timeout, method="POST")
    cands = res.get("candidates") or [{}]
    parts = ((cands[0].get("content") or {}).get("parts")) or []
    for p in parts:
        inline = p.get("inlineData") or {}
        data = inline.get("data")
        if data:
            return _ensure_wav(base64.b64decode(data))
    raise RuntimeError("TTS returned no audio data.")


def _tts_styled(key: str, text: str, model: str, voice: str, style: str,
                timeout: float) -> bytes:
    """Speech with a turn-level `style`, which is a real API field.

    Do NOT put the style into the prompt text. That looks like it works, but
    the model reads the instruction aloud: a 33-word line measured 10.8 s plain
    and 23.8 s with a spoken prefix, the extra 13 s being the instruction
    itself. `speech_metadata.style` is the supported channel.
    """
    body = {
        "model": model,
        "input": [{
            "type": "user_input",
            "content": [{
                "type": "text", "text": text,
                "annotations": [{"type": "speech_metadata", "style": style}],
            }],
        }],
        "response_format": {"type": "audio", "mime_type": "audio/wav",
                            "sample_rate": 24000},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    res = http_json(f"{GEMINI}/interactions", body,
                    headers={"x-goog-api-key": key},
                    timeout=timeout, method="POST")
    data = (res.get("output_audio") or {}).get("data")
    if not data:
        raise RuntimeError("TTS returned no audio data.")
    return _ensure_wav(base64.b64decode(data))


def gemini_tts(key: str, text: str, model: str = "gemini-3.8-flash-tts",
               voice: str = "Kore", style: str = "",
               timeout: float = 180.0) -> bytes:
    """Generate speech for `text` and return WAV bytes.

    This is only the *reference* voice. MiniMax H3 renders picture and speech
    in a single pass, so its own track is the one that actually matches the
    mouth. We feed this WAV in as `ref_audios` to steer the performance, then
    keep H3's native audio at export (`audio_mode="native"`). Use
    `audio_mode="tts"` only when the generated track is unusable.

    `style` is the character's own delivery - accent, pace, warmth, attitude.
    The prebuilt voices are all American-presenter neutral, so without it a
    creator sounds like a stock announcer rather than themselves.
    """
    # Same free-tier quota trap as chat: fall through to another TTS model
    # rather than losing the voiceover and the lip-sync reference with it.
    models = [model] + [m for m in TTS_FALLBACKS if m != model]
    tried: list[str] = []
    last: Exception | None = None
    if style:
        for name in models:
            tried.append(f"{name} (styled)")
            try:
                wav = _tts_styled(key, text, name, voice, style, timeout)
                if audio_ok(wav):
                    LAST_TTS.update(model=name, styled=True)
                    return wav
                last = RuntimeError(f"{name} returned silent audio")
            except Exception as e:
                last = e
    for name in models:
        tried.append(f"{name} (plain)")
        try:
            wav = _tts_legacy(key, text, name, voice, timeout)
            if audio_ok(wav):
                LAST_TTS.update(model=name, styled=False)
                return wav
            last = RuntimeError(f"{name} returned silent audio")
        except Exception as e:
            msg = str(e)
            retryable = ("HTTP 429" in msg or "quota" in msg.lower()
                         or "timed out" in msg.lower() or "Timeout" in type(e).__name__)
            last = e
            if not retryable:
                raise
    raise RuntimeError("TTS produced no usable speech on any model "
                       f"({', '.join(tried)}): {last}")