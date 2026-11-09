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

# Real UGC is not a short TV ad. It is one person talking to one person, and the
# beat order below is what actually holds retention on a 15-30 second cut:
# hook (0-3s) -> problem -> solution/demo -> proof -> CTA (last 3-5s).
# The weights are relative scene lengths: the hook is short and sharp, the
# demonstration carries the ad, the CTA is one breath.
BEAT_LIBRARY = {
    "Hook": (0.7, "open on the most specific true thing you have - a number, a "
                  "contrarian claim, or the moment just before the result"),
    "Problem": (1.0, "the exact situation the viewer is in, in their own words - "
                     "small and concrete, never a category"),
    "Agitate": (0.9, "what the problem costs them: the money, the time, the "
                     "thing they settled for"),
    "Solution": (1.4, "the product, in frame, doing the job - name it plainly "
                      "and let the demonstration carry the claim"),
    "Proof": (1.0, "one specific the viewer can check: a number, a duration, a "
                   "before/after visible in the same frame"),
    "Result": (1.0, "the after-state, shown not described - quiet, not triumphant"),
    "Objection": (0.9, "the real reason they would not buy, said out loud, then "
                       "answered in a few words"),
    "CTA": (0.8, "one action in the creator's own words, low pressure"),
}

BEAT_ORDER = {
    1: ["Hook"],
    2: ["Hook", "CTA"],
    3: ["Hook", "Solution", "CTA"],
    4: ["Hook", "Problem", "Solution", "CTA"],
    5: ["Hook", "Problem", "Solution", "Proof", "CTA"],
    6: ["Hook", "Problem", "Solution", "Proof", "Objection", "CTA"],
    7: ["Hook", "Problem", "Agitate", "Solution", "Proof", "Objection", "CTA"],
    8: ["Hook", "Problem", "Agitate", "Solution", "Proof", "Result",
        "Objection", "CTA"],
}

# Shot sizes rotate so two consecutive scenes never look the same. Repeated
# framing is the biggest "AI slideshow" tell in a storyboard.
SHOT_ROTATION = ["selfie close-up", "medium close-up", "tight product close-up",
                 "wide room shot", "over-shoulder"]

# Appended to every prompt so the phone-camera look cannot be lost, even when
# the planner ignores the instructions. This is what separates native UGC from
# a commercial: flat colour, off-centre framing, one hard light, real skin.
PHONE_LOOK_PARTS = [
    ("Photographed on a phone, vertical, held at chest height.",
     "photographed on a phone"),
    ("One dominant light source from one side with real falloff into shadow; "
     "available light only.", "dominant light"),
    ("Slightly off-centre framing, flat colour profile, mild sensor grain.",
     "off-centre"),
    ("Natural skin texture with visible pores and small imperfections.",
     "visible pores"),
    ("No beauty filter.", "beauty filter"),
    ("No retouching.", "retouching"),
    ("No studio lighting.", "studio lighting"),
    ("No seamless backdrop.", "seamless backdrop"),
    ("No ring light.", "ring light"),
    ("No professional photography.", "professional photography"),
]
PHONE_LOOK = " ".join(p for p, _ in PHONE_LOOK_PARTS)

MOTION_LOOK_PARTS = [
    ("Handheld phone camera with micro-jitter, natural exposure breathing and "
     "one small refocus.", "handheld phone camera"),
    ("Real skin texture, no beauty filter, no skin smoothing.", "no skin smoothing"),
    ("The room stays still; only the creator and the product move.",
     "room stays still"),
]
MOTION_LOOK = " ".join(p for p, _ in MOTION_LOOK_PARTS)


def _with_look(prompt: str, parts: list) -> str:
    """Append only the look clauses the prompt does not already contain.

    Without this the planner's own skin/light wording and the guarantee block
    both land in the prompt and every quality cue gets weighted twice.
    """
    low = prompt.lower()
    missing = [s for s, key in parts if key not in low]
    if not missing:
        return prompt
    return (prompt.rstrip(". ") + ". " + " ".join(missing)).strip()

# Every phrase here is a tell that a machine wrote the ad.
AD_SPEAK = [
    "elevate", "unleash", "revolutionary", "game-chang", "next-level",
    "the ultimate", "seamless", "empower", "transform your", "unlock",
    "dive into", "cutting-edge", "state-of-the-art", "effortless", "curated",
    "bespoke", "say goodbye to", "look no further", "in today's",
    "introducing", "experience the", "discover the power",
]

BAD_HOOKS = ["stop scrolling", "wait for it", "you won't believe",
             "this changed everything", "you need to see this"]

# Poster copy. A real creator does not say these out loud, so a script that
# contains one reads as an advertisement, not a recommendation.
TAGLINE_PHRASES = ["shop now", "buy now", "order today", "order now",
                   "available now", "don't miss", "dont miss", "get yours",
                   "your favourite", "your favorite", "ready whenever",
                   "treat yourself", "limited time", "while stocks last"]

FLAT_LIGHT = ["soft even lighting", "soft practical lighting", "flat lighting",
              "well lit", "evenly lit"]

LIGHT_WORDS = ("light", "sun", "window", "lamp", "shadow", "daylight", "glow",
               "backlit", "rim")

MOTION_WORDS = ("drift", "push", "pan", "tilt", "handheld", "hand-held", "move",
                "turn", "reach", "lean", "nod", "step", "lift", "open", "set",
                "place", "pick", "hold up", "walk", "refocus", "shake", "sway")


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


def _allocate_weighted(total: int, weights: list[float]) -> list[int]:
    """Split `total` seconds across scenes by weight, never below 1s each.

    Equal splits are what make an ad feel machine-made: a 15s cut needs a short
    hook, a long demonstration and a short CTA, not three 5s boxes.
    """
    n = len(weights)
    if n == 0:
        return []
    total = max(n, int(total))
    s = sum(weights) or float(n)
    raw = [total * (w / s) for w in weights]
    out = [max(1, int(r)) for r in raw]
    diff = total - sum(out)
    order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    k = 0
    while diff > 0:
        out[order[k % n]] += 1
        diff -= 1
        k += 1
    while diff < 0:
        i = max(range(n), key=lambda j: out[j])
        if out[i] <= 1:
            break
        out[i] -= 1
        diff += 1
    return out


def _allocate(total: int, n: int) -> list[int]:
    return _allocate_weighted(total, [1.0] * n)


def _beat_plan(seconds: int, n: int, single_take: bool = False) -> list[dict]:
    """Purpose + relative length for each scene, in real UGC beat order."""
    if single_take:
        # One unbroken take still has a hook and a CTA inside it; it just has no
        # cut to separate them, so the whole line is one scene.
        return [{"purpose": "Hook", "weight": 1.0,
                 "goal": "one unbroken take: hook, then the product, then the ask"}]
    names = BEAT_ORDER.get(n, BEAT_ORDER[8])
    if n > 8:
        names = BEAT_ORDER[8] + ["Solution"] * (n - 8)
    return [{"purpose": p,
             "weight": BEAT_LIBRARY.get(p, (1.0, ""))[0],
             "goal": BEAT_LIBRARY.get(p, (1.0, "support the story"))[1]}
            for p in names[:n]]


def _trim_words(text: str, limit: int = 12) -> str:
    """Spoken lines longer than this collapse at low step counts."""
    words = str(text or "").split()
    if len(words) <= limit:
        return str(text or "").strip()
    return " ".join(words[:limit]).rstrip(" ,;:—-") + "."


# Opinion frames only: a hook may not claim a fact the brief did not supply.
HOOK_LINES = [
    "Okay, this is not what I expected.",
    "I did not want to like this.",
    "Nobody told me about this one.",
    "I have been putting this off for weeks.",
]

BEAT_ACTION = {
    "Hook": "already mid-sentence, looking straight at the lens",
    "Problem": "half-turned to camera, phone lowered",
    "Agitate": "sighs and looks off-frame",
    "Solution": "holds the product up into frame",
    "Proof": "turns the product to show the detail",
    "Result": "looks at the product, then back at the lens",
    "Objection": "shrugs with one hand out",
    "CTA": "leans in slightly, straight to lens",
}

LIGHTS = [
    "hard afternoon sun from camera-left, deep shadow falloff",
    "single window light from the right, the background unlit",
    "one warm lamp just off frame left, dark room behind",
    "grey daylight through one doorway behind her, bright rim on the hair",
]


def plan_offline(brief_fields: dict, opts: dict) -> dict:
    seconds = int(opts.get("duration", 15))
    single = bool(opts.get("single_take"))
    n = _scene_count(seconds, single)
    beats = _beat_plan(seconds, n, single)
    durs = _allocate_weighted(seconds, [b["weight"] for b in beats])
    style = opts.get("style", "talking-head UGC")
    style_hint = STYLE_HINTS.get(style, style)
    plat_hint = PLATFORM_HINTS.get(opts.get("platform", "").lower(), "")
    brand = brief_fields.get("brand") or "the brand"
    product = brief_fields.get("product_name") or ""
    benefits = brief_fields.get("benefits") or []
    if isinstance(benefits, str):
        benefits = [b.strip() for b in re.split(r"(?<=[.!?])\s+", benefits) if b.strip()]
    benefits = [
        re.sub(r"^\s*(key features|benefits mentioned by the client|benefits|features)\s*[:\-\u2013]\s*",
               "", str(b), flags=re.I).strip(" .")
        for b in benefits
    ]
    benefits = [b for b in benefits if b]
    cta = brief_fields.get("call_to_action") or "Have a look."
    # Strip a leading label such as "Call to action:" so captions read as
    # spoken copy, never as a form field.
    cta = re.sub(r"^\s*(call\s*-?\s*to\s*-?\s*action|cta)\s*[:\-\u2013]\s*", "",
                 str(cta), flags=re.I).strip().rstrip(".")
    # A brief's CTA is usually poster copy ("Shop now"). Spoken by a creator it
    # reads as an advert, so swap it for the way a person actually says it.
    if any(p in cta.lower() for p in TAGLINE_PHRASES):
        cta = "Link's below if you want one"
    audience = brief_fields.get("target_audience") or ""
    env = ("an ordinary lived-in room: unmade bed, charger cable, a mug on the desk"
           if style in ("talking-head UGC", "testimonial")
           else "a real room, not a set")
    moves = ["slow push-in", "gentle handheld drift", "small reframe as she leans in"]

    scenes = []
    for i, beat in enumerate(beats):
        purpose = beat["purpose"]
        shot = SHOT_ROTATION[i % len(SHOT_ROTATION)]
        light = LIGHTS[i % len(LIGHTS)]
        action = BEAT_ACTION.get(purpose, "holds the product, talking to camera")
        move = moves[i % len(moves)]
        benefit = benefits[i % len(benefits)] if benefits else ""

        if purpose == "Hook":
            script = HOOK_LINES[i % len(HOOK_LINES)]
        elif purpose == "Problem":
            script = (f"If you're {audience}, you know this one." if audience
                      else "You know that thing that never quite works?")
        elif purpose == "Agitate":
            script = "And you keep paying for it anyway."
        elif purpose == "Solution":
            script = f"So I tried {product}." if product else "So I tried this."
        elif purpose == "Proof":
            script = benefit or "This is the part that got me."
        elif purpose == "Result":
            script = "That's it. That's the whole thing."
        elif purpose == "Objection":
            script = "Yes, I know what you're thinking."
        elif purpose == "CTA":
            script = cta
        else:
            script = f"{product}." if product else ""

        scenes.append({
            "n": i + 1,
            "purpose": purpose,
            "seconds": durs[i],
            "script": _trim_words(script),
            "visual": f"{shot}, creator {action}. {env}. {light}.",
            "creator_action": action,
            "product_action": (f"{product} in frame, unlabelled." if product
                               else "product in frame, unlabelled."),
            "camera_framing": shot,
            "camera_movement": move,
            "lighting": light,
            "image_prompt": (
                f"{shot} of the creator {action}, {env}. {light}. "
                f"Product: {product or 'the product'} clearly in frame, plain "
                f"unlabelled bottle with no lettering, held naturally. {style_hint}. "
                f"{plat_hint} {PHONE_LOOK}"
            ).strip(),
            "video_prompt": (
                f"Creator {action} and keeps talking to camera. {move}. "
                f"{MOTION_LOOK} {product or 'The product'} stays clearly in frame."
            ).strip(),
            "audio": "creator voiceover" if style != "product-focused commercial" else "music only",
            "refs": {"products": [1] if brief_fields.get("product_name") or brief_fields.get("brand") else [],
                     "creators": [1] if style != "product-focused commercial" else []},
            "continuity": "same creator, same wardrobe, same room, same plain unlabelled bottle",
        })

    return {
        "concept_name": f"{brand} · {style}",
        "hook": scenes[0]["script"] if scenes else "",
        "total_seconds": seconds,
        "scenes": scenes,
        "source": "offline",
    }


DIRECTOR_SYSTEM = """You are a UGC creative director writing for a creator with a real \
following, not a brand copywriter. The result must read like one person talking to one \
person, shot on a phone — not like a commercial that happens to be vertical.

STRUCTURE — fill the skeleton you are given; do not change its order or purposes.
- HOOK owns the first 3 seconds whatever the total length. One idea only. Never name the \
brand or the product in the hook: a brand name in the first line is the fastest way to \
signal "advertisement" and lose the scroll. It must work with the sound off, so the \
picture alone has to carry it.
- PROBLEM names the exact situation in the viewer's own words. Concrete and small, not a \
category.
- SOLUTION shows the product doing the job. Name it plainly; let the demonstration carry \
the claim.
- CTA is the last 3-5 seconds: one action, low pressure, in the creator's words. The \
creator is talking to a friend, not reading the brand's poster, so it sounds like \
"the link is right there if you want it" and never like "Your favourite coffee, ready \
whenever you need a break. Shop now." Never use shop now, buy now, order today, \
available now, don't miss out, get yours, or any line a brand would print on a poster.

SCRIPT is spoken dialogue, not copy:
- Write in THIS creator's voice. If a creator is described, match their vocabulary, \
rhythm and attitude — a deadpan creator and a bubbly creator do not say the same \
sentence. Never write lines that any generic presenter could have said.
- Write what a person would actually say out loud. Fragments are fine. Fifth-grade \
reading level. Maximum 12 words per scene.
- Say each line in your head before writing it down. If a real person would not say that \
sentence in a conversation, cut it.
- Specific beats clever. A number is an argument; "amazing results" is noise.
- Banned ad-speak: elevate, unleash, revolutionary, game-changing, next-level, the \
ultimate, seamless, empower, transform your, unlock, dive into, cutting-edge, \
state-of-the-art, effortless, curated, bespoke, say goodbye to, look no further, in \
today's world, introducing, experience the, discover the power.
- Never open with "Stop scrolling", "Wait for it", "You won't believe" or "This changed \
everything". Those are the four phrases every machine-written ad uses.
- Leave one small human imperfection in the script — "honestly", a "so", a half-finished \
thought. Do not polish it into a slogan.
- Never invent specifications, ingredients, results, timelines, testimonials, \
certifications, medical or performance claims. Use only the brief.
- Always spell out titles instead of using acronyms or digits: write "Grand Theft Auto 6", \
never "GTA 6" — the speech model reads digits and acronyms as gibberish.

PICTURE — this is where AI UGC usually fails, so be literal:
- Every scene is shot on a phone by the creator or a friend. Never a studio, never a \
commercial set, never a model shoot.
- image_prompt describes ONE still: who, doing what, shot size, light direction and \
quality, location, wardrobe, what is in the background. No motion words.
- video_prompt describes only what MOVES: camera behaviour (handheld micro-jitter, a slow \
push-in, a small refocus) plus the one physical action that changes during the shot. It is \
an image-to-video model — it animates the still, it cannot cut.
- Every image_prompt names ONE dominant light source with its direction, e.g. "hard \
afternoon sun from camera-left, deep shadow falloff" or "single window light from the \
right, the background unlit". Never write "soft even lighting", "soft practical \
lighting", "flat lighting", "well lit", or "natural light" on its own — even light is \
what makes a render look plastic.
- Both prompts must ask for natural skin texture, visible pores, subtle skin imperfections \
and no beauty filter.
- Vary the shot size every scene: selfie close-up, medium close-up, tight product \
close-up, wide room shot, over-shoulder. Two identical framings in a row reads as a \
slideshow.
- Keep one creator, one wardrobe, one location across scenes unless the brief says \
otherwise. continuity states exactly what must not change.
- Never ask for a label, logo, brand name, price or any lettering on the product or \
anywhere in the scene. The image model renders lettering as gibberish, and garbled \
brand text is worse than no text. Describe a plain unlabelled container; the brand \
lives in the spoken line and the burned-in caption.
- Do not put text-bearing props in frame either: no branded mug, poster, book cover, \
packet, phone screen or clothing print. At draft size those are an unreadable blur, \
and an upscale turns them into legible nonsense. If a prop is needed, name a plain \
version of it (a plain white mug, a plain notebook).

Output minified JSON only, matching the schema exactly. Scene durations must sum exactly \
to the requested total."""


def _storyboard_flags(scenes: list[dict], single_take: bool = False) -> list[str]:
    """Concrete reasons a storyboard would read as generic AI slop."""
    flags = []
    if not scenes:
        return ["No scenes were returned."]
    for i, s in enumerate(scenes):
        n = s.get("n", i + 1)
        script = str(s.get("script") or "")
        low = script.lower()
        img = str(s.get("image_prompt") or "")
        vid = str(s.get("video_prompt") or "")
        if i == 0:
            for bad in BAD_HOOKS:
                if low.startswith(bad) or f" {bad}" in low:
                    flags.append(f"Scene {n} hook opens with the cliché '{bad}'.")
        for bad in BAD_HOOKS:
            if bad in low:
                flags.append(f"Scene {n} script contains the cliché '{bad}'.")
        for phrase in AD_SPEAK:
            if phrase in low:
                flags.append(f"Scene {n} script uses ad-speak: '{phrase}'.")
        for phrase in TAGLINE_PHRASES:
            if phrase in low:
                flags.append(f"Scene {n} script sounds like poster copy: '{phrase}'.")
        words = len(script.split())
        limit = 40 if single_take else 14
        if words > limit:
            flags.append(f"Scene {n} script is {words} words; keep it under {limit}.")
        if img:
            if not any(w in img.lower() for w in LIGHT_WORDS):
                flags.append(f"Scene {n} image_prompt names no light source or direction.")
            for bad in FLAT_LIGHT:
                if bad in img.lower():
                    flags.append(f"Scene {n} image_prompt uses flat light '{bad}'.")
        if vid and not any(w in vid.lower() for w in MOTION_WORDS):
            flags.append(f"Scene {n} video_prompt describes no camera or body motion.")
        if i and s.get("camera_framing") and \
                s.get("camera_framing") == scenes[i - 1].get("camera_framing"):
            flags.append(f"Scene {n} repeats the same framing as scene {n - 1}.")
    return flags


def plan_with_llm(settings: dict, brief_fields: dict, opts: dict) -> dict:
    seconds = int(opts.get("duration", 15))
    single = bool(opts.get("single_take"))
    n = _scene_count(seconds, single)
    beats = _beat_plan(seconds, n, single)
    durs = _allocate_weighted(seconds, [b["weight"] for b in beats])
    skeleton = "\n".join(
        f"  scene {i + 1}: {b['purpose']} — {d} s — {b['goal']}"
        for i, (b, d) in enumerate(zip(beats, durs)))
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
    system = DIRECTOR_SYSTEM
    if single:
        # "No camera cuts" is a hard constraint, not a style hint: one scene,
        # one unbroken line, and a line long enough to fill the take.
        system += (
            " This advert is ONE CONTINUOUS TAKE: return exactly one scene. The camera "
            "never cuts — no second shot, no scene change, no montage. camera_movement "
            "must describe a single slow drift or push-in that lasts the whole take. "
            "The script is one unbroken spoken line that fills the entire duration "
            f"(about 2.2 words per second, so roughly {max(12, int(seconds * 2.2))} words).")
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"Schema: {json.dumps(schema)}\n\n"
         f"Duration: {seconds}s across {n} scenes. Style: {opts.get('style')}. "
         f"Platform: {opts.get('platform')}. Aspect: {opts.get('aspect')}.\n"
         f"Creator: {opts.get('creator') or 'not described'}."
         + (f" Their tone: {opts['tone']}." if opts.get("tone") else "")
         + "\n"
         f"Required skeleton (purpose, seconds, what that beat must achieve):\n"
         f"{skeleton}\n\n"
         f"Brief data: {json.dumps(brief_fields)}\n\nReturn JSON only."},
    ]
    obj = None
    for attempt in (1, 2):
        res = llm.chat(settings, msgs, task="plan", temperature=0.8, max_tokens=8000)
        obj = _parse_json(res["text"])
        obj["source"] = res["provider"]
        obj["model"] = res["model"]
        obj = normalize_storyboard(obj, seconds, n, single_take=single,
                                   brief_fields=brief_fields)
        flags = obj.get("slop_flags") or []
        if not flags or attempt == 2:
            break
        # One automatic rewrite, with the offending lines quoted back. Cheaper
        # than the user hand-fixing a storyboard that reads like an advert.
        msgs = msgs + [
            {"role": "user", "content":
             "That storyboard broke these rules:\n" +
             "\n".join(f"- {f}" for f in flags) +
             "\nRewrite it and return the full JSON again. Same scene count, same "
             "purposes, same durations, same brief — only fix what is flagged."},
        ]
    return obj


def _clean_script(text: str) -> str:
    """Strip the cliché openers that make an ad read as machine-written."""
    t = str(text or "").strip()
    for bad in BAD_HOOKS:
        if t.lower().startswith(bad):
            t = t[len(bad):].lstrip(" —-,.:;!").strip()
            if t:
                t = t[0].upper() + t[1:]
    return t


def normalize_storyboard(obj: dict, seconds: int, n: int,
                         single_take: bool = False,
                         brief_fields: dict | None = None) -> dict:
    scenes = obj.get("scenes") or []
    if not scenes:
        raise ValueError("Storyboard has no scenes.")
    # keep requested scene count
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
        s["script"] = _clean_script(s.get("script"))
        if not s.get("camera_framing"):
            s["camera_framing"] = SHOT_ROTATION[i % len(SHOT_ROTATION)]
        # Guarantee the phone-camera look even when the planner ignored it.
        # This floor is the difference between native UGC and a studio render.
        img = _with_look((s.get("image_prompt") or s.get("visual") or "").strip(),
                         PHONE_LOOK_PARTS)
        s["image_prompt"] = img
        vid = _with_look(str(s.get("video_prompt") or "").strip(), MOTION_LOOK_PARTS)
        s["video_prompt"] = vid
    # Weighted beats: short hook, long demo, short CTA — never three equal boxes.
    beats = _beat_plan(seconds, len(scenes), single_take)
    durs = _allocate_weighted(seconds, [b["weight"] for b in beats])
    for s, d in zip(scenes, durs):
        s["seconds"] = d
    obj["scenes"] = scenes
    obj["total_seconds"] = sum(durs)
    obj.setdefault("concept_name", "UGC concept")
    obj["hook"] = scenes[0].get("script", "") if scenes else ""
    obj["slop_flags"] = _storyboard_flags(scenes, single_take)
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
