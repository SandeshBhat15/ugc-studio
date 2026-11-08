"""Stage 01: brief extraction + storyboard planning (offline rules or LLM)."""
from __future__ import annotations

import json
import re

import llm

STYLE_HINTS = {
    "talking-head UGC": "creator speaks directly to camera, handheld selfie framing",
    "testimonial": "creator gives an honest first-person review to camera",
    "product demonstration": "close-up hands-on demo of the product in use",
    "lifestyle": "candid lifestyle moment with the product in natural context",
    "unboxing": "hands opening the packaging and revealing the product",
    "product-focused commercial": "stylised product hero shots, minimal or no people",
}

PLATFORM_HINTS = {
    "instagram reels": "vertical 9:16, captions near the lower third",
    "youtube shorts": "vertical 9:16, fast hook in the first second",
    "tiktok": "vertical 9:16, native casual pacing",
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def extract_offline(brief: str) -> dict:
    """Deterministic extraction. Never invents claims."""
    sents = _sentences(brief)
    low = brief.lower()

    brand = ""
    m = re.search(r"([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})\s+is\s+"
                  r"(?:launching|introducing|releasing|a|an)\b", brief)
    if m:
        brand = m.group(1).strip()
    if not brand:
        m = re.search(r"\b(?:from|by|at)\s+([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})", brief)
        if m:
            brand = m.group(1).strip()

    product = ""
    m = re.search(r"(?:launching|introducing|releasing)\s+"
                  r"([A-Z][\w\-.']*(?:\s+[A-Za-z][\w\-.']*){0,4})", brief)
    if m:
        product = m.group(1).strip().rstrip(".,")
    if not product:
        m = re.search(r"\b(?:a|an|the)\s+(\d+\s*ml\s+)?([a-z][\w-]*(?:\s+[\w-]+){0,2}?"
                      r"\s+(?:serum|cream|cleanser|moisturiser|moisturizer|lotion|oil|balm|"
                      r"shampoo|device|supplement|drink|snack|bag|shoe|app|tool))\b", brief, re.I)
        if m:
            product = (m.group(1) or "") + m.group(2)
    if product:
        product = product.strip()

    benefits = []
    for s in sents:
        if re.search(r"\b(help|helps|reduce|reduces|improve|improves|hydrat|clean|soft|"
                     r"bright|smooth|protect|repair|boost|fast|long[- ]lasting)\b", s, re.I):
            benefits.append(s)
    ctas = []
    for s in sents:
        if re.search(r"\b(buy|shop|order|try|get|download|book|visit|link)\b", s, re.I):
            ctas.append(s)
    unknown = [k for k, v in (("brand", brand), ("product_name", product),
                              ("category", ""), ("target_audience", ""),
                              ("objective", ""), ("tone", "")) if not v]
    return {
        "brand": brand,
        "product_name": product,
        "category": "",
        "features": [],
        "benefits": benefits[:4],
        "target_audience": "",
        "objective": "",
        "tone": "",
        "selling_points": benefits[:3],
        "call_to_action": ctas[0] if ctas else "",
        "mandatory_messaging": [],
        "restrictions": [],
        "unknown": unknown,
        "source": "offline",
        "raw_brief": brief,
    }


def extract_with_llm(settings: dict, brief: str) -> dict:
    schema = {
        "brand": "string", "product_name": "string", "category": "string",
        "features": ["string"], "benefits": ["string"], "target_audience": "string",
        "objective": "string", "tone": "string", "selling_points": ["string"],
        "call_to_action": "string", "mandatory_messaging": ["string"],
        "restrictions": ["string"], "unknown": ["string"],
    }
    msgs = [
        {"role": "system", "content":
         "You extract advertising briefs. Output ONLY valid minified JSON matching the "
         "requested schema. Never invent product specifications, ingredients, certifications, "
         "testimonials, medical or performance claims. Put anything not stated into 'unknown'."},
        {"role": "user", "content":
         f"Schema: {json.dumps(schema)}\n\nBRIEF:\n{brief}\n\nReturn JSON only."},
    ]
    res = llm.chat(settings, msgs, task="extract", temperature=0.2, max_tokens=1500)
    obj = _parse_json(res["text"])
    obj["source"] = res["provider"]
    obj["model"] = res["model"]
    obj["raw_brief"] = brief
    return obj


def _parse_json(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("Model did not return JSON.")
    return json.loads(t[start:end + 1])


# ---------------------------------------------------------------- storyboard

def _scene_count(seconds: int, single_take: bool = False) -> int:
    if single_take:
        # One continuous take: the camera never cuts, so there is exactly one scene.
        return 1
    if seconds <= 5:
        return 1
    if seconds <= 10:
        return 2
    if seconds <= 15:
        return 3
    if seconds <= 30:
        return 5
    return min(8, max(6, seconds // 8))


def _allocate(total: int, n: int) -> list[int]:
    base = total / n
    out = [max(1, int(round(base)))] * n
    # fix rounding drift
    drift = total - sum(out)
    i = 0
    while drift != 0 and i < 1000:
        idx = i % n
        if drift > 0:
            out[idx] += 1
            drift -= 1
        elif out[idx] > 1:
            out[idx] -= 1
            drift += 1
        i += 1
    return out


BEATS = [
    ("Hook", "grab attention in the first second", "creator looks at camera and opens with the hook"),
    ("Problem", "name the audience's need", "creator gestures at the everyday frustration"),
    ("Product intro", "reveal the product and brand", "creator brings the product into frame"),
    ("Demonstration", "show the product being used", "creator uses the product on camera"),
    ("Benefits", "state the key benefits from the brief", "creator reacts to the result"),
    ("Proof", "reinforce credibility without inventing claims", "creator shows the product detail"),
    ("Call to action", "tell the viewer what to do next", "creator points to the CTA"),
    ("Close", "leave a lasting brand impression", "clean product hero shot"),
]


BEAT_ORDER = {
    1: ["Hook"],
    2: ["Hook", "Call to action"],
    3: ["Hook", "Demonstration", "Call to action"],
    4: ["Hook", "Problem", "Demonstration", "Call to action"],
    5: ["Hook", "Problem", "Product intro", "Demonstration", "Call to action"],
    6: ["Hook", "Problem", "Product intro", "Demonstration", "Benefits", "Call to action"],
    7: ["Hook", "Problem", "Product intro", "Demonstration", "Benefits", "Proof", "Call to action"],
    8: ["Hook", "Problem", "Product intro", "Demonstration", "Benefits", "Proof",
        "Call to action", "Close"],
}

BEAT_META = {
    "Hook": ("grab attention in the first second",
             "creator looks straight into the camera and opens with the hook"),
    "Problem": ("name the audience's everyday need",
                "creator gestures at the frustration the product solves"),
    "Product intro": ("reveal the product and the brand",
                      "creator brings the product up into frame, label facing camera"),
    "Demonstration": ("show the product being used",
                      "creator uses the product on camera in a real setting"),
    "Benefits": ("state the benefits taken from the brief",
                 "creator reacts to the result and mentions the benefit"),
    "Proof": ("reinforce credibility without inventing claims",
              "creator shows a close detail of the product"),
    "Call to action": ("tell the viewer exactly what to do next",
                       "creator delivers the call to action to camera"),
    "Close": ("leave a lasting brand impression",
              "clean product hero shot with the brand visible"),
}


def plan_offline(brief_fields: dict, opts: dict) -> dict:
    seconds = int(opts.get("duration", 15))
    n = _scene_count(seconds, bool(opts.get("single_take")))
    durs = _allocate(seconds, n)
    style = opts.get("style", "talking-head UGC")
    style_hint = STYLE_HINTS.get(style, style)
    plat_hint = PLATFORM_HINTS.get(opts.get("platform", "").lower(), "")
    brand = brief_fields.get("brand") or "the brand"
    product = brief_fields.get("product_name") or "the product"
    benefits = brief_fields.get("benefits") or []
    if isinstance(benefits, str):
        benefits = [b.strip() for b in re.split(r"(?<=[.!?])\s+", benefits) if b.strip()]
    benefits = [
        re.sub(r"^\s*(key features|benefits mentioned by the client|benefits|features)\s*[:\-\u2013]\s*",
               "", str(b), flags=re.I).strip(" .")
        for b in benefits
    ]
    benefits = [b for b in benefits if b]
    cta = brief_fields.get("call_to_action") or "Follow for more."
    # Strip a leading label such as "Call to action:" so captions read as
    # spoken copy, never as a form field.
    cta = re.sub(r"^\s*(call\s*-?\s*to\s*-?\s*action|cta)\s*[:\-\u2013]\s*", "",
                 str(cta), flags=re.I).strip().rstrip(".")
    objective = brief_fields.get("objective") or ""
    audience = brief_fields.get("target_audience") or ""

    beats = BEAT_ORDER.get(n, BEAT_ORDER[8])
    if n > 8:
        beats = BEAT_ORDER[8] + ["Demonstration"] * (n - 8)

    scenes = []
    for i in range(n):
        purpose = beats[i] if i < len(beats) else "Demonstration"
        goal, action = BEAT_META.get(purpose, ("support the story", "creator holds the product"))
        benefit = benefits[i % len(benefits)] if benefits else ""
        if purpose == "Hook":
            script = f"Stop scrolling — this is {product}." if product != "the product" \
                else f"Stop scrolling — {objective or 'you need to see this'}."
        elif purpose == "Call to action":
            script = cta
        elif purpose == "Benefits":
            script = benefit or f"{product} does what it promises."
        elif purpose == "Problem":
            script = f"If you're {audience or 'always busy'}, this is for you."
        else:
            script = f"Here's {product} in action."
        env = ("bright bathroom or bedroom" if style in ("talking-head UGC", "testimonial")
               else "clean minimal interior")
        scenes.append({
            "n": i + 1,
            "purpose": purpose,
            "seconds": durs[i],
            "script": script,
            "visual": f"{style_hint}. {purpose}: {goal}. {plat_hint}. {env}, natural light.",
            "creator_action": action,
            "product_action": f"{product} visible, label readable, in a natural context.",
            "camera_framing": "selfie close-up" if i == 0 else "medium close-up with product in frame",
            "camera_movement": "gentle push-in" if i % 2 == 0 else "subtle handheld drift",
            "lighting": "single hard key light from camera-left, deep shadow falloff",
            "image_prompt": (
                f"{style_hint}. {purpose} moment for {brand}: {action}. "
                f"Product: {product}. {env}, one dominant window light from the side, "
                f"strong shadow contrast, authentic smartphone photography, shallow "
                f"depth of field, vertical composition, product label facing the camera."
            ),
            "video_prompt": (
                f"{action}. Handheld smartphone camera with gentle {('push-in' if i % 2 == 0 else 'drift')}, "
                f"directional daylight with visible shadows, authentic UGC energy, "
                f"{product} clearly visible and readable, subtle realistic motion."
            ),
            "audio": "creator voiceover" if style != "product-focused commercial" else "music only",
            "refs": {"products": [1] if brief_fields.get("product_name") or brief_fields.get("brand") else [],
                     "creators": [1] if style != "product-focused commercial" else []},
            "continuity": "same creator, same wardrobe, same location, consistent product label",
        })

    return {
        "concept_name": f"{brand} · {style}",
        "hook": scenes[0]["script"] if scenes else "",
        "total_seconds": seconds,
        "scenes": scenes,
        "source": "offline",
    }


def plan_with_llm(settings: dict, brief_fields: dict, opts: dict) -> dict:
    seconds = int(opts.get("duration", 15))
    single = bool(opts.get("single_take"))
    n = _scene_count(seconds, single)
    schema = {
        "concept_name": "string",
        "hook": "string",
        "scenes": [{
            "n": "int", "purpose": "string", "seconds": "int", "script": "string",
            "visual": "string", "creator_action": "string", "product_action": "string",
            "camera_framing": "string", "camera_movement": "string", "lighting": "string",
            "image_prompt": "string", "video_prompt": "string", "audio": "string",
            "refs": {"products": ["int"], "creators": ["int"]},
            "continuity": "string",
        }],
    }
    msgs = [
        {"role": "system", "content":
         "You are a senior UGC advertising director. Produce a shot-by-shot storyboard as "
         "minified JSON only. Never invent product specifications, ingredients, benefits, "
         "certifications, testimonials, medical or performance claims — use only the supplied "
         "brief. image_prompt must be a still-photograph prompt with no camera-motion words. "
         "video_prompt must describe motion for an image-to-video model. Scene durations must "
         "sum exactly to the requested total. "
         # The speech model renders what it reads, so titles it cannot pronounce
         # must be written out: it says 'These final 6' for 'GTA 6'.
         "script is spoken dialogue: keep every line SHORT (about 12 words maximum), "
         "and always spell out titles instead of using acronyms or digits "
         "(write 'Grand Theft Auto 6', never 'GTA 6'). "
         # A distilled sampler cannot invent skin texture, so ask for it directly.
         # Even light is what makes a render look plastic, so force a direction.
         "image_prompt and video_prompt must both specify natural skin texture, visible "
         "pores, subtle skin imperfections and no beauty filter. Every image_prompt must "
         "also name ONE dominant light source with its direction and quality, for example "
         "'hard afternoon sun from camera-left, deep shadow falloff' or 'single window "
         "light from the right, unlit background'. Never write 'soft even lighting', "
         "'soft practical lighting', 'flat lighting' or 'well lit'."},
        {"role": "user", "content":
         f"Schema: {json.dumps(schema)}\n\n"
         f"Duration: {seconds}s across {n} scenes. Style: {opts.get('style')}. "
         f"Platform: {opts.get('platform')}. Aspect: {opts.get('aspect')}.\n"
         f"Brief data: {json.dumps(brief_fields)}\n\nReturn JSON only."},
    ]
    if single:
        # "No camera cuts" is a hard constraint, not a style hint: one scene,
        # one unbroken line, and a line long enough to fill the take.
        msgs[0]["content"] += (
            " This advert is ONE CONTINUOUS TAKE: return exactly one scene. The camera "
            "never cuts — no second shot, no scene change, no montage. camera_movement "
            "must describe a single slow drift or push-in that lasts the whole take. "
            "The script is one unbroken spoken line that fills the entire duration "
            f"(about 2.2 words per second, so roughly {max(12, int(seconds * 2.2))} words).")
    res = llm.chat(settings, msgs, task="plan", temperature=0.8, max_tokens=8000)
    obj = _parse_json(res["text"])
    obj["source"] = res["provider"]
    obj["model"] = res["model"]
    return normalize_storyboard(obj, seconds, n)


def normalize_storyboard(obj: dict, seconds: int, n: int) -> dict:
    scenes = obj.get("scenes") or []
    if not scenes:
        raise ValueError("Storyboard has no scenes.")
    # keep requested scene count, trim or pad from the offline template
    if len(scenes) > n:
        scenes = scenes[:n]
    scenes.sort(key=lambda s: s.get("n", 0))
    for i, s in enumerate(scenes):
        s["n"] = i + 1
        s.setdefault("refs", {"products": [1], "creators": [1]})
        s.setdefault("seconds", 5)
        for k in ("purpose", "script", "visual", "creator_action", "product_action",
                  "camera_framing", "camera_movement", "lighting", "image_prompt",
                  "video_prompt", "audio", "continuity"):
            s.setdefault(k, "")
    # enforce exact total duration
    durs = _allocate(seconds, len(scenes))
    for s, d in zip(scenes, durs):
        s["seconds"] = d
    obj["scenes"] = scenes
    obj["total_seconds"] = sum(durs)
    obj.setdefault("concept_name", "UGC concept")
    obj.setdefault("hook", scenes[0].get("script", "") if scenes else "")
    return obj


def plan(settings: dict, brief_fields: dict, opts: dict) -> dict:
    n = _scene_count(int(opts.get("duration", 15)), bool(opts.get("single_take")))
    try:
        return plan_with_llm(settings, brief_fields, opts)
    except Exception as e:
        # Never fall back silently: a storyboard written by the offline
        # template instead of the LLM must be visible in the UI and the log,
        # otherwise a transient provider failure looks like a success.
        sb = plan_offline(brief_fields, opts)
        sb["plan_error"] = f"{type(e).__name__}: {e}"
        sb["source"] = "offline"
        return sb
