"""LLM provider manager: local (Ollama / LM Studio) + OpenRouter free tier.

No third-party deps. Never spends money: only models priced at 0 are eligible.
"""
from __future__ import annotations

import base64
import json
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


def gemini_tts(key: str, text: str, model: str = "gemini-3.8-flash-tts",
               voice: str = "Kore", timeout: float = 180.0) -> bytes:
    """Generate speech for `text` and return WAV bytes.

    This is only the *reference* voice. MiniMax H3 renders picture and speech
    in a single pass, so its own track is the one that actually matches the
    mouth. We feed this WAV in as `ref_audios` to steer the performance, then
    keep H3's native audio at export (`audio_mode="native"`). Use
    `audio_mode="tts"` only when the generated track is unusable.
    """
    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }
    # Same free-tier quota trap as chat: fall through to another TTS model
    # rather than losing the voiceover and the lip-sync reference with it.
    tried: list[str] = []
    last: Exception | None = None
    for name in [model] + [m for m in TTS_FALLBACKS if m != model]:
        tried.append(name)
        try:
            res = http_json(f"{GEMINI}/models/{name}:generateContent", body,
                            headers={"x-goog-api-key": key},
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
        raise RuntimeError("Gemini TTS quota exhausted on every model tried "
                           f"({', '.join(tried)}): {last}")
    cands = res.get("candidates") or [{}]
    parts = ((cands[0].get("content") or {}).get("parts")) or []
    for p in parts:
        inline = p.get("inlineData") or {}
        data = inline.get("data")
        if data:
            return base64.b64decode(data)
    raise RuntimeError("TTS returned no audio data.")