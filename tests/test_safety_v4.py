"""
v4 safety hardening. Offline: the LLM stage is injected, no network.
Covers: the Hindi/Marathi hanging-wire hole, the benign-phrase bypass, two-stage OR / fail-closed
behaviour, and master-driven concept -> category resolution (no hard-coded IDs).
"""
from types import SimpleNamespace
import pytest

from app.services.safety_guard import safety_guard
from app.services.safety_assessment import assess, resolve_category_for_concept
from app.services.policy_gate import evaluate_policy_gate
from app.db.models import Category


def _no_llm(text):
    return None, "not_configured"


HANGING_WIRE = [
    "स्कूल के गेट के पास बिजली का तार लटक रहा है",                       # Hindi (the reported leak)
    "शाळेच्या गेटजवळ विजेची तार लटकत आहे",                               # Marathi
    "स्कूल के गेट के पास जिंदा तार नीचे लटकी हुई है बच्चों को खतरा",
    "school gate jawal vayar latkat aahe current lagu shakto",           # Marathlish
    "school ke paas taar latak raha hai",                                # Hinglish
    "A live electric wire is hanging near the school gate",
    "Electric cable dangling low over the road near the market",
    "The pole is leaning dangerously and wires are loose near the primary school",
    "शाळेजवळील खांब वाकला आहे आणि तार तुटली आहे",
    "बिजली के खंभे से चिंगारी निकल रही है",
    "An electric pole has fallen on the road near the school",
    "विजेचा खांब पडला आहे",
    "wire tutli aahe ani current yet aahe",
]


@pytest.mark.parametrize("text", HANGING_WIRE)
def test_hanging_wire_is_sensitive(text):
    assert safety_guard.inspect(text).is_sensitive, text


BENIGN = [
    "सुनवाई की तारीख आगे बढ़ा दी गई है",            # 'तारीख' (date) contains 'तार'
    "कामाची तारीख पडली आहे पण काम सुरू नाही",
    "पोलीस स्टेशनच्या रस्त्यावर खड्डे आहेत",          # 'पोलीस' (police) contains 'पोल'
    "वायरल बुखार के मरीज बढ़ रहे हैं, अस्पताल में दवा नहीं",
    "Wireless internet tower fee is wrongly charged on my bill",
    "Garbage truck is killing time and never comes on schedule",
    "Street light is not working near the school for two weeks",
    "The water pipe is leaking near the temple",
    "Streetlight broken pole lamp bulb dark",                        # routine: a broken lamp pole is not a live wire
    "The street light pole is broken and the lamp is not working",
    "विजेचा खांब तुटला आहे आणि दिवा बंद आहे",
]


@pytest.mark.parametrize("text", BENIGN)
def test_benign_look_alikes_not_sensitive(text):
    assert not safety_guard.inspect(text).is_sensitive, text


@pytest.mark.parametrize("text", [
    "We are killing time at the office but boys are harassing girls at the bus stand",
    "STD 10 results awaited. Also, a man threatened to kill me yesterday",
    "gang of workers came, and then a live wire was hanging near our house",
])
def test_benign_phrase_cannot_switch_the_guard_off(text):
    """Regression: one 'false-positive' phrase used to declare the WHOLE text safe (fail-open)."""
    assert safety_guard.inspect(text).is_sensitive


def test_stage_b_alone_can_hold_a_case_stage_a_misses():
    text = "ek vichitra goshta ghadli, mala bhiti vatte"  # nothing the regex knows
    a = assess(text, llm_safety=lambda t: ({"sensitive": True, "concept": "Threat / Public Safety Concern",
                                            "confidence": 0.8}, "ok"))
    assert a.is_sensitive and not a.stage_a_sensitive and a.disagreement and a.identity_mask


def test_stage_a_alone_holds_when_stage_b_says_safe():
    a = assess(HANGING_WIRE[5], llm_safety=lambda t: ({"sensitive": False, "concept": None, "confidence": 0.9}, "ok"))
    assert a.is_sensitive and a.stage_a_sensitive and a.disagreement


def test_routine_complaint_passes_both_stages():
    a = assess("Garbage has not been collected in our lane for a week",
               llm_safety=lambda t: ({"sensitive": False, "concept": None, "confidence": 0.95}, "ok"))
    assert not a.is_sensitive and not a.disagreement


def _cat(**kw):
    base = dict(category_id="C009", category="Street Lighting and Electricity", subcategory="Streetlight not working",
                responsible_department_id="D005", responsible_department="Electricity Dept",
                default_assignee="Officer", sla_days=5, sensitive=False, identity_hiding=False,
                default_priority="MEDIUM", auto_resolution_allowed=True)
    base.update(kw)
    return Category(**base)


def test_stage_b_error_fails_closed_to_human_review():
    a = assess("Garbage has not been collected in our lane", llm_safety=lambda t: (None, "error"))
    d = evaluate_policy_gate(_cat(), 0.99, "STRONG", 0, "Garbage has not been collected in our lane",
                             llm_confidence=0.99, safety_result=a)
    assert d.verdict == "HUMAN_REVIEW" and d.rule_id == "RULE_07_SAFETY_STAGE_B_UNAVAILABLE"
    assert not d.can_auto_resolve


def test_stage_b_not_configured_is_not_an_error():
    a = assess("Garbage has not been collected in our lane", llm_safety=_no_llm)
    d = evaluate_policy_gate(_cat(), 0.99, "STRONG", 0, "Garbage has not been collected in our lane",
                             llm_confidence=0.99, safety_result=a)
    assert d.verdict == "AUTO_HANDLE"


def test_agent_can_only_make_the_gate_stricter():
    d = evaluate_policy_gate(_cat(), 0.99, "STRONG", 0, "Garbage not collected", llm_confidence=0.99,
                             agent_escalation="could involve a hazard")
    assert d.verdict == "HUMAN_REVIEW" and d.rule_id == "RULE_08_AGENT_ESCALATION"
    # ...and never overrides a hold
    d2 = evaluate_policy_gate(_cat(sensitive=True, identity_hiding=True), 1.0, "STRONG", 0, "x", agent_escalation=None)
    assert d2.verdict == "SENSITIVE_HOLD"


def test_sensitive_proximity_forces_human_review():
    d = evaluate_policy_gate(_cat(), 0.99, "STRONG", 0, "Garbage not collected", llm_confidence=0.99,
                             sensitive_proximity={"hit": True, "category_id": "C010", "ratio": 0.95})
    assert d.verdict == "HUMAN_REVIEW" and d.rule_id == "RULE_02_SENSITIVE_PRECAUTION_ESCALATION"


MASTER = [
    _cat(category_id="C009"),
    _cat(category_id="C010", subcategory="Exposed wire / electrical hazard", sensitive=True, identity_hiding=True),
    _cat(category_id="C017", category="Women and Child Welfare", subcategory="Women safety / harassment concern",
         sensitive=True, identity_hiding=True),
    _cat(category_id="C018", category="Women and Child Welfare", subcategory="Child safety / protection concern",
         sensitive=True, identity_hiding=True),
    _cat(category_id="C026", category="Public Safety", subcategory="Threat / immediate public safety concern",
         sensitive=True, identity_hiding=True),
]


@pytest.mark.parametrize("concept,expected", [
    ("Exposed Wire / Electrical Hazard", "C010"),
    ("Women Safety / Harassment", "C017"),
    ("Child Safety / Protection", "C018"),
    ("Threat / Public Safety Concern", "C026"),
])
def test_concept_resolves_against_master_rows(concept, expected):
    assert resolve_category_for_concept(concept, MASTER).category_id == expected


def test_concept_resolution_follows_the_master_not_a_fixed_table():
    """Re-label the master: the resolver must follow it (no hard-coded IDs)."""
    swapped = [
        _cat(category_id="C900", subcategory="Exposed wire / electrical hazard", sensitive=True, identity_hiding=True),
        _cat(category_id="C901", subcategory="Streetlight outage", sensitive=False),
    ]
    assert resolve_category_for_concept("Exposed Wire / Electrical Hazard", swapped).category_id == "C900"
    assert resolve_category_for_concept("Exposed Wire / Electrical Hazard",
                                        [_cat(category_id="C900", sensitive=False)]) is None
