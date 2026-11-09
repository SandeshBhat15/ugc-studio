"""Character -> prebuilt voice.

The Gemini prebuilt voices are not interchangeable. Each one is documented with
a one-word character ("Firm", "Youthful", "Casual", "Gravelly"), and when the
delivery-style annotation is unavailable that word is the only thing that can
actually be steered. So the voice is chosen from the character's own
description and tone rather than from a global default.

Gender for each voice is inferred from the vendor's own samples, not from a
documented field, so it is used as a preference and never as a hard filter:
a trait match on the wrong side of the guess is still better than a generic
voice.
"""

import re

# name: (inferred gender, traits the vendor documents, plain-English blurb)
PREBUILT = {
    "Kore": ("female", ["firm", "steady", "neutral"], "firm and even"),
    "Leda": ("female", ["youthful", "bright", "light"], "youthful"),
    "Aoede": ("female", ["breezy", "light", "airy"], "breezy"),
    "Callirrhoe": ("female", ["easy-going", "relaxed", "casual"], "easy-going"),
    "Autonoe": ("female", ["bright", "lively"], "bright"),
    "Despina": ("female", ["smooth", "even"], "smooth"),
    "Erinome": ("female", ["clear", "crisp"], "clear"),
    "Laomedeia": ("female", ["upbeat", "energetic"], "upbeat"),
    "Achernar": ("female", ["soft", "gentle", "quiet"], "soft"),
    "Pulcherrima": ("female", ["forward", "confident", "direct"], "forward"),
    "Vindemiatrix": ("female", ["gentle", "warm", "slow"], "gentle"),
    "Sadachbia": ("female", ["lively", "playful"], "lively"),
    "Sulafat": ("female", ["warm", "rich"], "warm"),
    "Zephyr": ("female", ["bright", "clear"], "bright"),
    "Gacrux": ("female", ["mature", "grounded"], "mature"),
    "Puck": ("male", ["upbeat", "energetic"], "upbeat"),
    "Charon": ("male", ["informative", "deep", "measured"], "informative"),
    "Fenrir": ("male", ["excitable", "loud"], "excitable"),
    "Orus": ("male", ["firm", "confident"], "firm"),
    "Enceladus": ("male", ["breathy", "soft"], "breathy"),
    "Iapetus": ("male", ["clear", "crisp"], "clear"),
    "Umbriel": ("male", ["easy-going", "relaxed"], "easy-going"),
    "Algieba": ("male", ["smooth", "warm"], "smooth"),
    "Algenib": ("male", ["gravelly", "rough"], "gravelly"),
    "Rasalgethi": ("male", ["informative", "dry"], "informative"),
    "Alnilam": ("male", ["firm", "steady"], "firm"),
    "Schedar": ("male", ["even", "flat", "deadpan"], "even"),
    "Achird": ("male", ["friendly", "warm"], "friendly"),
    "Zubenelgenubi": ("male", ["casual", "laid-back"], "casual"),
    "Sadaltager": ("male", ["knowledgeable", "measured"], "knowledgeable"),
}

# How a word in the character's own tone text maps onto documented traits.
# Order matters: the first trait is the closest fit, the rest are near misses.
SYNONYMS = {
    "deadpan": ["deadpan", "flat", "even"],
    "monotone": ["flat", "even", "deadpan"],
    "dry": ["dry", "deadpan", "flat", "casual"],
    "sarcastic": ["dry", "deadpan", "casual"],
    "wry": ["dry", "casual", "deadpan"],
    "tired": ["soft", "low", "slow", "even"],
    "low-key": ["even", "casual", "soft"],
    "quiet": ["quiet", "soft", "gentle"],
    "calm": ["even", "steady", "gentle"],
    "warm": ["warm", "gentle", "friendly"],
    "friendly": ["friendly", "warm", "casual"],
    "kind": ["gentle", "warm", "soft"],
    "energetic": ["energetic", "upbeat", "lively"],
    "upbeat": ["upbeat", "lively", "bright"],
    "excited": ["excitable", "lively", "upbeat"],
    "loud": ["loud", "excitable", "forward"],
    "confident": ["confident", "firm", "forward"],
    "bold": ["forward", "confident", "firm"],
    "shy": ["soft", "quiet", "gentle"],
    "soft": ["soft", "gentle", "quiet"],
    "gentle": ["gentle", "soft", "quiet"],
    "bright": ["bright", "lively"],
    "playful": ["playful", "lively", "casual"],
    "funny": ["playful", "casual", "lively"],
    "silly": ["playful", "lively", "excitable"],
    "serious": ["firm", "measured", "steady"],
    "professional": ["measured", "informative", "steady"],
    "authoritative": ["firm", "deep", "measured"],
    "smooth": ["smooth", "warm"],
    "gravelly": ["gravelly", "rough"],
    "husky": ["breathy", "gravelly", "soft"],
    "breathy": ["breathy", "soft"],
    "deep": ["deep", "rich", "measured"],
    "rich": ["rich", "warm", "deep"],
    "nervous": ["soft", "quick", "breathy"],
    "fast": ["quick", "energetic", "upbeat"],
    "slow": ["slow", "gentle", "measured"],
    "casual": ["casual", "laid-back", "easy-going"],
    "relaxed": ["relaxed", "easy-going", "laid-back"],
    "laid-back": ["laid-back", "easy-going", "relaxed"],
}

# Strong style signals: these describe *how* someone talks, so they outweigh
# a stray "funny" in the same sentence.
WEIGHT = {"deadpan": 2.0, "dry": 2.0, "sarcastic": 2.0, "monotone": 2.0,
          "wry": 2.0, "tired": 1.5, "low-key": 1.5, "quiet": 1.5}

YOUNG = re.compile(r"\b(1[89]|2[0-9]|30)\b|mid-?20s|twenties|teen|young", re.I)
OLD = re.compile(r"\b(4[0-9]|5[0-9]|6[0-9])\b|forties|fifties|sixties|older|"
                 r"middle-aged|mature", re.I)


def _wanted(text: str) -> tuple[dict, list[str]]:
    """Turn the character's own words into weighted trait requests."""
    wanted: dict[str, float] = {}
    hits: list[str] = []
    for word, traits in SYNONYMS.items():
        if not re.search(rf"\b{re.escape(word)}\b", text):
            continue
        hits.append(word)
        weight = WEIGHT.get(word, 1.0)
        for i, trait in enumerate(traits):
            # first trait 3 points, second 2, third 1, times the word weight
            wanted[trait] = max(wanted.get(trait, 0.0),
                                weight * (3.0 - i))
    if YOUNG.search(text):
        wanted["youthful"] = max(wanted.get("youthful", 0.0), 2.0)
        hits.append("young")
    if OLD.search(text):
        wanted["mature"] = max(wanted.get("mature", 0.0), 2.0)
        hits.append("older")
    return wanted, hits


def derive(blob: str, gender: str = "") -> tuple[str, str]:
    """Pick the prebuilt voice that best fits `blob`. Returns (voice, why).

    `blob` is the character's description plus tone plus delivery note, i.e.
    everything we know about how this person actually sounds.
    """
    text = " " + re.sub(r"[^a-z0-9\- ]+", " ", (blob or "").lower()) + " "
    wanted, hits = _wanted(text)

    def _best(pool):
        name_, score_ = "", 0.0
        traits_: list[str] = []
        for name, (voice_gender, traits, _blurb) in pool:
            score = sum(wanted.get(t, 0.0) for t in traits)
            if not score:
                continue
            if gender and voice_gender == gender:
                score += 1.0
            if score > score_:
                name_, score_ = name, score
                traits_ = [t for t in traits if wanted.get(t)]
        return name_, traits_

    # Gender is a hard filter when it is known: a trait match on the wrong
    # voice is not worth a woman sounding like a man. Only fall back across
    # genders when nothing on the right side scores at all.
    best_name, best_traits = "", []
    if gender:
        best_name, best_traits = _best([(n, v) for n, v in PREBUILT.items()
                                        if v[0] == gender])
    if not best_name:
        best_name, best_traits = _best(list(PREBUILT.items()))

    if not best_name:
        return "", "no tone described"

    why = "derived from " + ", ".join(hits[:4]) if hits else "derived"
    if best_traits:
        why += f" -> {best_name} ({', '.join(best_traits[:2])})"
    return best_name, why


def options() -> list[dict]:
    """Every prebuilt voice, for the picker."""
    return [{"name": n, "gender": g, "blurb": b, "traits": t}
            for n, (g, t, b) in PREBUILT.items()]
