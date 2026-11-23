"""
Semantic Safety Guard & Invariant Enforcement Layer.
Group 3 rewrite:
- HARD RULE: NO hard-coded category ID lists anywhere in this file.
- Policy gate decides sensitivity from Category master flags only.
- SafetyGuard is a *signal generator*: it detects concepts and returns them.
- Concept-to-category mapping is done by policy_gate.py by scanning the Category master.
- Stage A: expanded multilingual regex with word boundaries.
- Stage B: LLM classifier (real mode only) — not yet implemented (TODO when Sarvam key available).
- If A and B disagree: fail closed → HUMAN_REVIEW (conservative).
- Negative cases: "killing time", "gang of workers", "STD 10 cert", "riot of colours" → NOT sensitive.
"""
import re
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SafetyGuardResult:
    is_sensitive: bool
    detected_concept: Optional[str]       # human-readable concept name (NOT a category ID)
    reason: str
    identity_mask: bool
    escalation_rule: str
    # Kept for backward compatibility in tests — do NOT use to bypass master CSV
    sensitive_category_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage A: Deterministic multilingual regex patterns
# ---------------------------------------------------------------------------
# Regexes use re.IGNORECASE where appropriate.
# Devanagari patterns do not use \b (word-boundary not reliable for Unicode).

# ---- CONCEPT: Women Safety / Harassment ----
_WOMEN_SAFETY_EN = [
    r"\beve[- ]?teas(?:e|ing|ed)\b",
    r"\bteas(?:e|ing|ed)\b",
    r"\bstalk(?:ing|ers?)\b",
    r"\bstalk(?:ed)?\s+(?:by|me|her|woman|girl|daughter)\b",
    r"\bharass(?:ment|ing|ed)?\b",
    r"\bmolest(?:ation|ing|ed)?\b",
    r"\bgrope(?:d|ing)?\b",
    r"\bsexual(?:ly)?\s+(?:abuse|abused|assault|assaulted|harass|harassed|harassment|harrass|harrassment)\b",
    r"\bsexually\s+abus(?:ed|ing)\b",
    r"\bsexual\s+hara(?:ss)?ment\b",
    r"\bcatcall(?:ing)?\b",
    r"\bpass(?:ing)?\s+comments?\s+(?:on|at)\s+(?:girl|woman|women|lady|ladies)\b",
    r"\bgirls?\b.*?\bteas(?:e|ing|ed)\b",
    r"\bboys?\b.*?\bteas(?:e|ing|ed)\b",
    r"\bbus[- ]?stand\b.*?\bgirls?",
    r"\bcollege\b.*?\bgirls?\b.*?\b(?:teas|follow|harass)\b",
    r"\bfollow(?:ing)?\s+(?:girls?|women|ladies)\b",
    r"\btouch(?:ing)?\s+inappropriately\b",
    r"\bassault(?:ing)?\s+(?:girls?|woman|women|lady|ladies)\b",
    r"\brapist?\b",
    r"\brap(?:e|ed|ing)\b",
    r"\bsexual\s+(?:abuse|assault)\b",
    r"\bacid\s+attack\b",
    r"\bchild\s+marriage\b",
    r"\bforced\s+(?:into\s+)?(?:marriage|child\s+marriage|sex)\b",
    r"\btraffick(?:ing|ed)\b",
    r"\bwoman\b.*?\bstalk(?:ed|ing)?\b",
    r"\bstalk(?:ed|ing)?\b.*?\bwoman\b",
    r"\bchhed\b",
    r"\bchedchad\b",
    r"\bchhad\b",
    # Hinglish transliterations
    r"\bchhed[-\s]?kadhtat\b",
    r"\bchhed[-\s]?kadhte\b",
]
_WOMEN_SAFETY_MR = [
    r"छेड",
    r"छेडछाड",
    r"छेड काढ",
    r"मुलींची छेड",
    r"मुलींना त्रास",
    r"विनयभंग",
    r"अत्याचार",
    r"पाठलाग",
    r"महिला सुरक्षा",
    r"स्त्री सुरक्षा",
    r"बलात्कार",
    r"लैंगिक",
]
_WOMEN_SAFETY_HI = [
    r"छेड़छाड़",
    r"छेड़ते",
    r"छेड़ना",
    r"लड़कियों को छेड़",
    r"लड़की को छेड़",
    r"महिलाओं से बदसलूकी",
    r"पीछा करना",
    r"दुर्व्यवहार",
    r"महिला सुरक्षा",
    r"बलात्कार",
    r"छेड़खानी",
    r"यौन उत्पीड़न",
    r"परेशान करते",
]

# ---- CONCEPT: Child Safety / Protection ----
_CHILD_SAFETY_EN = [
    r"\bchild\s+labou?r\b",
    r"\bchild\s+work(?:ing)?\b",
    r"\bsmall\s+child\b",
    r"\bminor\s+(?:work|labour|abuse|beaten)\b",
    r"\btea[- ]?stall\b.*?\bchild\b",
    r"\bchild\b.*?\btea[- ]?stall\b",
    r"\bchild\b.*?\bbeaten\b",
    r"\bbeating\s+(?:a\s+)?child\b",
    r"\bdhaba\b.*?\bchild\b",
    r"\bchild\b.*?\bdhaba\b",
    r"\bhotel\b.*?\bchild\b",
    r"\bkid\b.*?\b(?:work|beaten|abuse)\b",
    r"\bchild\s+abuse\b",
    r"\bminor\s+working\b",
    r"\bunderage\b.*?\b(?:work|labou?r|employ|girl|boy|child)\b",
    r"\b(?:ten|nine|eight|seven|six|five|four|eleven|twelve|\d+)[- ]year[- ]old\b.*?\b(?:sell|work|labour|forced|employ|beg)\b",
    r"\byear[- ]old\b.*?\b(?:sell|work|labour|forced|employ|beg)\b",
    r"\bchild\s+trafficking\b",
    r"\bchild\s+forced\b",
]
_CHILD_SAFETY_MR = [
    r"बालमजुरी",
    r"बाल कामगार",
    r"लहान मूल",
    r"लहान मुला",
    r"लहान मुलाला",
    r"मुलाला मारहाण",
    r"मुलांना मारहाण",
    r"मुला.*?मारहाण",
    r"चहाच्या टपरीवर लहान",
    r"हॉटेलमध्ये लहान मूल",
    r"बाल शोषण",
]
_CHILD_SAFETY_HI = [
    r"बाल मजदूरी",
    r"बाल श्रमिक",
    r"बाल श्रम",
    r"छोटे बच्चे से काम",
    r"बच्चे को मार",
    r"बच्चों को मार",
    r"बाल शोषण",
    r"बाल तस्करी",
]

# ---- CONCEPT: Electrical Hazard ----
_ELECTRICAL_EN = [
    r"\bexposed\s+(?:electric\s+)?(?:wire|cable)\b",
    r"\bbare\s+(?:electric\s+)?(?:wire|cable)\b",
    r"\bfallen\s+(?:electric\s+)?(?:wire|cable|pole)\b",
    r"\bhigh[- ]?tension\s+wire\b",
    r"\blive\s+(?:electric\s+)?(?:wire|cable)\b",
    r"\blive\b.*?\b(?:wire|cable)\b",
    r"\belectric\s+shock\b",
    r"\belectrical\s+hazard\b",
    r"\bshock\s+risk\b",
    r"\bspark(?:ing)?\s+(?:wire|cable|pole|transformer)\b",
    r"\btransformer\b.*?\b(?:spark|smoke|fire|blast|fell|fallen|burn)\b",
    r"\bpole\s+(?:spark|smoke|fallen|broken)\b",
    r"\bsmok(?:ing|e)\s+(?:wire|cable|transformer)\b",
    r"\bburning\s+(?:wire|cable)\b",
]
_ELECTRICAL_MR = [
    r"उघडी तार",
    r"तुटलेली तार",
    r"खाली पडलेली तार",
    r"करंट लागू",
    r"इलेक्ट्रिक शॉक",
    r"तारेला आग",
    r"ट्रान्सफॉर्मर",
    r"उघडा वायर",
    r"ठिणग्या",
    r"ठिणग्या पडत",
]
_ELECTRICAL_HI = [
    r"नंगी तार",
    r"खुली तार",
    r"गिरी तार",
    r"गिरा तार",
    r"बिजली का झटका",
    r"करंट लगना",
    r"तार में आग",
    r"ट्रांसफार्मर",
    r"चिंगारी",
    r"चिंगारी निकलना",
    r"खुला तार",
]

# Structural electrical-hazard cues (generalise beyond a fixed phrase list).
# An electrical object within a short window of a hazard descriptor, in either order, EN/HI/MR/Hinglish.
#   * wires / cables / transformers: broad hazard vocabulary (hanging, loose, broken, exposed, live, sparking...)
#   * poles / pylons: NARROW vocabulary only (fallen, leaning, sparking, shock...). A merely "broken" lamp
#     pole (routine streetlight complaint) must not be treated as a live-wire emergency.
_WIRE_EN = r"(?:wires?|cables?|transformers?|dp\s*box|electric\s+line|power\s+line|ht\s+line)"
_POLE_EN = r"(?:(?:electric|electrical|utility|power|ht)\s+poles?|poles?\s+(?=[^.!?\n]{0,40}\b(?:wires?|cables?)\b))"
_HAZ_WIRE_EN = (
    r"(?:hang(?:s|ing)?|hung|dangl(?:e|es|ing|ed)|droop(?:s|ing|ed)?|sagg?(?:ing|ed)?|loose|snapp?(?:ed|ing)|"
    r"broke(?:n)?|fall(?:en|s|ing)?|fell|down|exposed|bare|naked|uncovered|open|live|sparks?|sparking|sparked|"
    r"smok(?:e|es|ing|ed)|burn(?:s|ing|t)|catch(?:es|ing)?\s+fire|on\s+fire|short[- ]?circuit(?:ed|ing)?|"
    r"shock(?:s|ed|ing)?|electrocut\w*|touching|current)"
)
_HAZ_POLE_EN = (
    r"(?:fall(?:en|s|ing)?|fell|collaps\w*|lean(?:s|ing|ed)|tilt(?:s|ed|ing)?|bent|snapp?(?:ed|ing)|"
    r"sparks?|sparking|sparked|smok(?:e|es|ing|ed)|burn(?:s|ing|t)|on\s+fire|shock(?:s|ed|ing)?|electrocut\w*)"
)
_WIRE_DEV = r"(?:तार(?![िी])|वायर(?!ल)|केबल|ट्रान्सफॉर्मर|ट्रांसफार्मर|ट्रांसफॉर्मर|रोहित्र)"
_POLE_DEV = r"(?:खांब|खंभ|पोल(?![िीस]))"
_HAZ_WIRE_DEV = (
    r"(?:लटक|झुल|ढील|तुटल|टूट|गिर|पडल|पड़|उघड|खुल|नंग|जिंदा|जिवंत|चालू|करंट|शॉक|झटका|चिंगारी|ठिणग|"
    r"आग|धुआं|धूर|जळ|जल\s*रह|शॉर्ट)"
)
_HAZ_POLE_DEV = r"(?:गिर|पडल|पड़|वाक|झुक|चिंगारी|ठिणग|शॉक|करंट|आग|धुआं|धूर|जळ|जल\s*रह|कोसळ)"
# Hinglish / Marathlish transliterations ("tar" alone is deliberately NOT a noun: it is a common Marathi word)
_WIRE_TR = r"(?:taar|vayar|wire|cable|transformer)"
_POLE_TR = r"(?:khamba|khambha|khamb|pole)"
_HAZ_WIRE_TR = (
    r"(?:latak\w*|latk\w*|tut\w*|toot\w*|gir(?:a|i|l[ai])\b|padl[ai]|pad(?:a|i)\s+hai|ughad\w*|nangi|nanga|"
    r"khuli|khula|jinda|jivant|current|karant|shock|chingari|thingnya|thinagy\w*|spark\w*|aag)"
)
_HAZ_POLE_TR = r"(?:gir(?:a|i|l[ai])\b|padl[ai]|pad(?:a|i)\s+hai|vakl\w*|jhuk\w*|chingari|thingnya|spark\w*|current|karant|shock|aag)"

_WIN = r"[^.!?\n]{0,40}?"
_WIN_DEV = r"[^.!?\n।]{0,40}?"


def _both(noun: str, haz: str, win: str, hazard_boundary: bool = True) -> list:
    """noun near hazard, in either order."""
    hb = r"\b" if hazard_boundary else ""
    return [
        r"\b" + noun + r"\b" + win + hb + haz + hb,
        hb + haz + hb + win + r"\b" + noun + r"\b",
    ]


_ELEC_STRUCTURAL = (
    _both(_WIRE_EN, _HAZ_WIRE_EN, _WIN)
    + _both(_POLE_EN, _HAZ_POLE_EN, _WIN)
    + [_WIRE_DEV + _WIN_DEV + _HAZ_WIRE_DEV, _HAZ_WIRE_DEV + _WIN_DEV + _WIRE_DEV,
       _POLE_DEV + _WIN_DEV + _HAZ_POLE_DEV, _HAZ_POLE_DEV + _WIN_DEV + _POLE_DEV]
    + _both(_WIRE_TR, _HAZ_WIRE_TR, _WIN, hazard_boundary=False)
    + _both(_POLE_TR, _HAZ_POLE_TR, _WIN, hazard_boundary=False)
)

# ---- CONCEPT: Extortion / Threats / Public Safety ----
_THREAT_EN = [
    # Threats with weapons — require context, not bare words
    r"\bkniv?es?\b.*?\b(?:threat|attack|shopkeeper|weapon|wield|kill)\b",
    r"\b(?:men|people|goons?)\s+with\s+kniv?es?\b",
    r"\bknife\s+(?:threat|attack|wielding|at|threatening)\b",
    r"\bgun\s+(?:threat|point|wielding|at|threatening)\b",
    r"\bgoon\b.*?\b(?:threat|knife|weapon|attack|beat)\b",
    r"\bgang\s+of\s+(?:men|boys|people)\b.*?\b(?:threat|beat|attack|weapon)\b",
    r"\bthreaten(?:ing|ed)?\s+(?:with\s+)?(?:knife|knives|gun|weapon|stick)\b",
    r"\bthreaten(?:ing|ed)?\s+to\s+kill\b",
    r"\bweapon\b.*?\bthreat",
    r"\bwith\s+weapons\b.*?\bthreat",
    r"\bextort(?:ion|ing)?\b",
    r"\bprotection\s+money\b",
    r"\bhafta\b",
    r"\bkill\s+(?:me|us|him|her|them|you)\b",  # credible threat to life (not "killing time")
    r"\bmurder\s+(?:threat|me|us)\b",
    r"\bphysical\s+(?:threat|assault|attack)\b.*?\b(?:weapon|knife|gun)\b",
    r"\bimminent\s+(?:danger|threat)\b",
]
_THREAT_MR = [
    r"धमकी देतो",
    r"धमकावत",
    r"खंडणी मागत",
    r"जीवे मारण्याची धमकी",
    r"शस्त्र दाखवत",
    r"चाकू दाखवत",
    r"गुंड",
    r"लाचखोरी",
    r"खंडणी",
]
_THREAT_HI = [
    r"धमकी दे रहे",
    r"जान से मारने की धमकी",
    r"हथियार दिखाकर",
    r"चाकू लेकर",
    r"गुंडे",
    r"वसूली",
    r"रंगदारी",
    r"रिश्वत मांग",
    r"लाच मांग",
]

# ---- CONCEPT: Bribery / Corruption (escalation, not full sensitive hold) ----
_BRIBERY_EN = [
    r"\bbrib(?:e|ery|ed|ing)\b",
    r"\bcorrupt(?:ion)?\b",
    r"\bunlawful\s+demand\b",
    r"\bhafta\b",
    r"\bextort(?:ion|ing)?\b",
    r"\bdemanded\s+(?:money|cash|payment)\b.*?\b(?:illegally|unauthorized)\b",
]
_BRIBERY_MR = [
    r"लाचखोरी",
    r"लाच",
    r"भ्रष्टाचार",
    r"खंडणी",
]
_BRIBERY_HI = [
    r"रिश्वत",
    r"भ्रष्टाचार",
    r"घूस",
    r"रंगदारी",
    r"वसूली",
]

# ---- CONCEPT: Confidential Medical ----
_MEDICAL_EN = [
    r"\bmedical\b.*?\b(?:leak(?:ed|ing|s)?|shar(?:ed|ing|es)?|expos(?:ed|ing|es)?|disclos(?:ed|ing|es)?)\b",
    r"\btest\s+report\b.*?\b(?:leak(?:ed|ing|s)?|shar(?:ed|ing|es)?|expos(?:ed|ing|es)?|disclos(?:ed|ing|es)?)\b",
    r"\bpregnancy\b.*?\b(?:test|result|leak|shar|expos|disclos)",
    r"\bhiv\b.*?\b(?:status|test|result|leak|shar|expos|disclos)",
    r"\bconfidential\s+(?:medical|health|patient)\b",
    r"\bpatient\s+(?:data|record|info|status)\b.*?\b(?:leak|shar|expos|disclos)",
    r"\bhealth\s+record\b.*?\b(?:leak|breach|shar|disclos)",
]
_MEDICAL_MR = [
    r"वैद्यकीय अहवाल फुटला",
    r"रुग्णाची माहिती उघड",
    r"गर्भधारणा चाचणी",
    r"गोपनीय वैद्यकीय",
]
_MEDICAL_HI = [
    r"मेडिकल रिपोर्ट लीक",
    r"मरीज की जानकारी उजागर",
    r"गोपनीय स्वास्थ्य",
    r"गर्भावस्था परीक्षण",
]

# ---------------------------------------------------------------------------
# False-positive guard: patterns that should NOT trigger sensitive flags
# ---------------------------------------------------------------------------
_SAFE_NEGATIVES = [
    r"\bkilling\s+time\b",
    r"\bgang\s+of\s+(?:workers|labou?rers|volunteers|staff|employees)\b",
    r"\bstd\s*\d+\b",        # STD 10 / STD 12 certificate
    r"\briot\s+of\s+colou?rs\b",  # Holi "riot of colours"
    r"\bsword\s+(?:fern|plant|dance)\b",
    r"\bgang\s+up\s+(?:on\s+work|to\s+help)\b",
    r"\bstalk\s+of\b",
]


def _compile_patterns(patterns: list) -> list:
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as exc:
            logger.warning("Regex compile error for pattern %r: %s", p, exc)
    return compiled


_EN_WOMEN   = _compile_patterns(_WOMEN_SAFETY_EN)
_MR_WOMEN   = _compile_patterns(_WOMEN_SAFETY_MR)
_HI_WOMEN   = _compile_patterns(_WOMEN_SAFETY_HI)
_EN_CHILD   = _compile_patterns(_CHILD_SAFETY_EN)
_MR_CHILD   = _compile_patterns(_CHILD_SAFETY_MR)
_HI_CHILD   = _compile_patterns(_CHILD_SAFETY_HI)
_EN_ELEC    = _compile_patterns(_ELECTRICAL_EN + _ELEC_STRUCTURAL)
_MR_ELEC    = _compile_patterns(_ELECTRICAL_MR)
_HI_ELEC    = _compile_patterns(_ELECTRICAL_HI)
_EN_THREAT  = _compile_patterns(_THREAT_EN)
_MR_THREAT  = _compile_patterns(_THREAT_MR)
_HI_THREAT  = _compile_patterns(_THREAT_HI)
_EN_BRIBERY = _compile_patterns(_BRIBERY_EN)
_MR_BRIBERY = _compile_patterns(_BRIBERY_MR)
_HI_BRIBERY = _compile_patterns(_BRIBERY_HI)
_EN_MEDICAL = _compile_patterns(_MEDICAL_EN)
_MR_MEDICAL = _compile_patterns(_MEDICAL_MR)
_HI_MEDICAL = _compile_patterns(_MEDICAL_HI)
_NEGATIVES  = _compile_patterns(_SAFE_NEGATIVES)


def _matches_any(text: str, compiled_patterns: list) -> bool:
    return any(p.search(text) for p in compiled_patterns)


def _is_false_positive(text: str) -> bool:
    """Kept for backward compatibility. NOT used to short-circuit the guard any more."""
    return _matches_any(text, _NEGATIVES)


def _mask_benign_spans(text: str) -> str:
    """Replace each benign-phrase span with spaces so the remaining text is still inspected."""
    masked = text
    for pat in _NEGATIVES:
        masked = pat.sub(lambda m: " " * len(m.group(0)), masked)
    return masked


class SemanticSafetyGuard:
    """
    Stage A: deterministic multilingual regex signal generator.
    Returns SafetyGuardResult with concept name (never category ID).
    The policy_gate will map concept → category by scanning the Category master CSV at runtime.
    """

    def inspect(
        self,
        text: str,
        proposed_category_id: Optional[str] = None,  # kept for API compat but NOT used for decisions
    ) -> SafetyGuardResult:
        if not text or len(text.strip()) < 3:
            return SafetyGuardResult(
                is_sensitive=False,
                detected_concept=None,
                reason="Empty or trivial input.",
                identity_mask=False,
                escalation_rule="RULE_DEFAULT_SAFE",
            )

        # Stage A pre-screen: blank out ONLY the matched benign spans ("killing time",
        # "STD 10", ...) and keep scanning the rest. A benign phrase must never switch the
        # whole check off (that was a fail-open bypass: any sensitive text containing one
        # benign phrase used to be declared safe).
        text = _mask_benign_spans(text)

        all_patterns = (
            _EN_WOMEN + _MR_WOMEN + _HI_WOMEN,
            _EN_CHILD + _MR_CHILD + _HI_CHILD,
            _EN_ELEC + _MR_ELEC + _HI_ELEC,
            _EN_THREAT + _MR_THREAT + _HI_THREAT,
        )
        concepts = (
            "Women Safety / Harassment",
            "Child Safety / Protection",
            "Exposed Wire / Electrical Hazard",
            "Threat / Public Safety Concern",
        )
        reasons = (
            "Citizen narrative contains indications of women safety or harassment. Mandatory human review.",
            "Citizen narrative contains indications of child labour, minor abuse, or child safety risk. Mandatory human review.",
            "Exposed live wire or electrical shock hazard detected. Immediate physical hazard requires prioritized human review.",
            "Citizen narrative indicates imminent physical threat, violence, or extortion. Mandatory officer review.",
        )

        for pattern_group, concept, reason in zip(all_patterns, concepts, reasons):
            if _matches_any(text, pattern_group):
                logger.info("SafetyGuard: Concept detected — %s", concept)
                return SafetyGuardResult(
                    is_sensitive=True,
                    detected_concept=concept,
                    reason=reason,
                    identity_mask=True,
                    escalation_rule="RULE_01_SENSITIVE_CATEGORY_HOLD",
                )

        # Confidential Medical
        if _matches_any(text, _EN_MEDICAL + _MR_MEDICAL + _HI_MEDICAL):
            return SafetyGuardResult(
                is_sensitive=True,
                detected_concept="Confidential Medical Access",
                reason="Confidential health records or medical privacy concern detected. Mandatory identity masking and human review.",
                identity_mask=True,
                escalation_rule="RULE_01_SENSITIVE_CATEGORY_HOLD",
            )

        # Bribery / Corruption — escalation but not full sensitive hold
        if _matches_any(text, _EN_BRIBERY + _MR_BRIBERY + _HI_BRIBERY):
            logger.info("SafetyGuard: Bribery / Corruption indication detected")
            return SafetyGuardResult(
                is_sensitive=False,
                detected_concept="Bribery / Corruption Alert",
                reason="Allegation of bribery, corruption, or unlawful demand detected. Conservative escalation to human review.",
                identity_mask=True,
                escalation_rule="RULE_02_SENSITIVE_PRECAUTION_ESCALATION",
            )

        return SafetyGuardResult(
            is_sensitive=False,
            detected_concept=None,
            reason="No sensitive hazard or safety trigger detected.",
            identity_mask=False,
            escalation_rule="RULE_DEFAULT_SAFE",
        )


safety_guard = SemanticSafetyGuard()
