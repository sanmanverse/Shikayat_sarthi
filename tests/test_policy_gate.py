"""
Policy Gate Tests — Groups 3 & 13.
Key invariants:
- No sensitive/identity_hiding case ever reaches AUTO_HANDLE, regardless of confidence.
- Policy flags come from Category object (simulating master CSV row) — no hard-coded ID lists.
- Flipping a category's flags changes gate behavior.
- identity_hiding=True alone forces masking + HUMAN_REVIEW.
- Negative texts (killing time, gang of workers, STD 10 cert) → NOT sensitive.
"""
import pytest
from app.db.models import Category
from app.services.policy_gate import evaluate_policy_gate, detect_sensitive_keywords
from app.services.safety_guard import safety_guard


# ---------------------------------------------------------------------------
# Fixtures — simulate Category master rows
# ---------------------------------------------------------------------------

@pytest.fixture
def routine_category():
    return Category(
        category_id="C009",
        category="Street Lighting and Electricity",
        subcategory="Streetlight not working",
        responsible_department_id="D005",
        responsible_department="Electricity Distribution Department",
        default_assignee="Electricity Distribution Department - Duty Officer (Demo)",
        sla_days=5,
        sensitive=False,
        identity_hiding=False,
        default_priority="MEDIUM",
        auto_resolution_allowed=True,
    )


@pytest.fixture
def sensitive_women_safety_category():
    return Category(
        category_id="C017",
        category="Women and Child Welfare",
        subcategory="Women safety / harassment concern",
        responsible_department_id="D017",
        responsible_department="District Women Support Cell",
        default_assignee="District Women Support Cell - Duty Officer (Demo)",
        sla_days=1,
        sensitive=True,
        identity_hiding=True,
        default_priority="CRITICAL",
        auto_resolution_allowed=False,
    )


@pytest.fixture
def sensitive_child_safety_category():
    return Category(
        category_id="C018",
        category="Women and Child Welfare",
        subcategory="Child safety / protection concern",
        responsible_department_id="D018",
        responsible_department="District Child Protection Unit",
        default_assignee="District Child Protection Unit - Duty Officer (Demo)",
        sla_days=1,
        sensitive=True,
        identity_hiding=True,
        default_priority="CRITICAL",
        auto_resolution_allowed=False,
    )


@pytest.fixture
def sensitive_wire_category():
    return Category(
        category_id="C010",
        category="Street Lighting and Electricity",
        subcategory="Exposed wire / electrical hazard",
        responsible_department_id="D005",
        responsible_department="Electricity Distribution Department",
        default_assignee="Electricity Distribution Department - Duty Officer (Demo)",
        sla_days=1,
        sensitive=True,
        identity_hiding=True,
        default_priority="CRITICAL",
        auto_resolution_allowed=False,
    )


@pytest.fixture
def identity_hiding_only_category():
    """Category with identity_hiding=True but sensitive=False."""
    return Category(
        category_id="C028",
        category="Legal Aid",
        subcategory="Anonymous legal complaint",
        responsible_department_id="D015",
        responsible_department="Legal Aid Cell",
        default_assignee="Legal Aid Officer",
        sla_days=7,
        sensitive=False,
        identity_hiding=True,
        default_priority="MEDIUM",
        auto_resolution_allowed=True,
    )


# ---------------------------------------------------------------------------
# Core sensitive-hold invariants
# ---------------------------------------------------------------------------

def test_sensitive_category_hard_invariant(sensitive_women_safety_category):
    """INVARIANT: Category.sensitive=True → SENSITIVE_HOLD at any confidence."""
    decision = evaluate_policy_gate(
        category=sensitive_women_safety_category,
        fused_confidence=1.0,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Regular complaint wording",
    )
    assert decision.verdict == "SENSITIVE_HOLD"
    assert decision.rule_id == "RULE_01_SENSITIVE_CATEGORY_HOLD"
    assert decision.can_auto_resolve is False
    assert decision.identity_mask is True


def test_sensitive_leak_prevention_girls_teased(routine_category):
    """'Boys tease the girls near the bus stand' MUST NOT auto-handle."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.98,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Boys tease the girls near the bus stand every evening",
    )
    assert decision.verdict == "SENSITIVE_HOLD"
    assert decision.can_auto_resolve is False
    assert decision.identity_mask is True


def test_sensitive_leak_prevention_child_beaten(routine_category):
    """'A small child is working at a tea stall and is being beaten' → SENSITIVE_HOLD."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.95,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="A small child is working at a tea stall and is being beaten",
    )
    assert decision.verdict == "SENSITIVE_HOLD"
    assert decision.can_auto_resolve is False
    assert decision.identity_mask is True


def test_sensitive_leak_prevention_exposed_wire(routine_category):
    """Live sparking wire → SENSITIVE_HOLD."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.92,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="A live electric wire has fallen on the street and is sparking, high shock risk",
    )
    assert decision.verdict == "SENSITIVE_HOLD"
    assert decision.can_auto_resolve is False
    assert decision.identity_mask is True


def test_sensitive_leak_prevention_knife_threat(routine_category):
    """Knife threat → SENSITIVE_HOLD."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.90,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Goon with a knife is threatening local market shopkeepers",
    )
    assert decision.verdict == "SENSITIVE_HOLD"
    assert decision.can_auto_resolve is False
    assert decision.identity_mask is True


# ---------------------------------------------------------------------------
# identity_hiding=True tests (Group 3)
# ---------------------------------------------------------------------------

def test_identity_hiding_alone_forces_masking_and_human_review(identity_hiding_only_category):
    """identity_hiding=True, sensitive=False → identity_mask + HUMAN_REVIEW (not AUTO_HANDLE)."""
    decision = evaluate_policy_gate(
        category=identity_hiding_only_category,
        fused_confidence=0.95,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Anonymous legal matter requiring discretion",
    )
    assert decision.verdict == "HUMAN_REVIEW"
    assert decision.identity_mask is True
    assert decision.can_auto_resolve is False
    assert decision.rule_id == "RULE_01b_IDENTITY_HIDING"


def test_no_ai_path_closes_sensitive_case(sensitive_women_safety_category):
    """Even at max confidence and STRONG retrieval, sensitive cases cannot be AUTO_HANDLE."""
    for conf in [0.0, 0.5, 0.75, 0.9, 1.0]:
        decision = evaluate_policy_gate(
            category=sensitive_women_safety_category,
            fused_confidence=conf,
            retrieval_quality="STRONG",
            sensitive_neighbor_count=0,
            complaint_text="A girl was harassed",
        )
        assert decision.verdict == "SENSITIVE_HOLD", f"Leaked at confidence={conf}"
        assert decision.can_auto_resolve is False


# ---------------------------------------------------------------------------
# Flag-flip test (Group 3)
# ---------------------------------------------------------------------------

def test_flag_flip_routine_to_sensitive(routine_category):
    """
    Flipping a normally-routine category's sensitive=True flag
    makes the gate return SENSITIVE_HOLD — proving no hard-coded ID dependency.
    """
    # First confirm it auto-handles with routine flags
    decision_before = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.89,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Streetlight on pole 14 is dark since Monday",
    )
    assert decision_before.verdict == "AUTO_HANDLE"

    # Now flip the flag in the same category object
    routine_category.sensitive = True
    decision_after = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.89,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Streetlight on pole 14 is dark since Monday",
    )
    assert decision_after.verdict == "SENSITIVE_HOLD"
    assert decision_after.can_auto_resolve is False

    # Restore
    routine_category.sensitive = False


# ---------------------------------------------------------------------------
# Conservative escalation
# ---------------------------------------------------------------------------

def test_sensitive_keyword_conservative_escalation(routine_category):
    """Bribe/corruption → HUMAN_REVIEW (conservative escalation)."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.95,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="The duty officer demanded 1000 rupees bribe to replace the lamp",
    )
    assert decision.verdict == "HUMAN_REVIEW"
    assert decision.rule_id == "RULE_02_SENSITIVE_PRECAUTION_ESCALATION"
    assert decision.can_auto_resolve is False
    assert decision.identity_mask is True


def test_sensitive_neighbor_conservative_escalation(routine_category):
    """Sensitive neighbors → escalate to HUMAN_REVIEW."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.90,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=2,
        complaint_text="Streetlight is broken near the crossroad",
    )
    assert decision.verdict == "HUMAN_REVIEW"
    assert decision.rule_id == "RULE_02_SENSITIVE_PRECAUTION_ESCALATION"
    assert decision.can_auto_resolve is False


# ---------------------------------------------------------------------------
# Confidence and retrieval checks
# ---------------------------------------------------------------------------

def test_low_confidence_escalation(routine_category):
    """Confidence below threshold → HUMAN_REVIEW."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.62,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Light flickering",
    )
    assert decision.verdict == "HUMAN_REVIEW"
    assert decision.rule_id == "RULE_04_LOW_CONFIDENCE"
    assert decision.can_auto_resolve is False


def test_null_confidence_forces_human_review(routine_category):
    """When fused_confidence is None (LLM didn't provide), must route to HUMAN_REVIEW."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=None,  # type: ignore
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Streetlight broken",
    )
    assert decision.verdict == "HUMAN_REVIEW"
    assert decision.can_auto_resolve is False


def test_auto_resolution_disallowed_by_category(sensitive_wire_category):
    """Category master says no auto-resolution → SENSITIVE_HOLD (sensitive=True overrides)."""
    decision = evaluate_policy_gate(
        category=sensitive_wire_category,
        fused_confidence=0.92,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Exposed wire on pole",
    )
    assert decision.verdict == "SENSITIVE_HOLD"
    assert decision.can_auto_resolve is False


def test_weak_retrieval_escalation(routine_category):
    """Weak retrieval → HUMAN_REVIEW."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.88,
        retrieval_quality="WEAK",
        sensitive_neighbor_count=0,
        complaint_text="Streetlight flickering irregularly",
    )
    assert decision.verdict == "HUMAN_REVIEW"
    assert decision.rule_id == "RULE_05_WEAK_RETRIEVAL"
    assert decision.can_auto_resolve is False


def test_routine_auto_handle_success(routine_category):
    """Routine case with strong confidence and no flags → AUTO_HANDLE."""
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.89,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text="Streetlight on pole 14 is dark since Monday",
    )
    assert decision.verdict == "AUTO_HANDLE"
    assert decision.rule_id == "RULE_06_AUTO_HANDLE_PERMITTED"
    assert decision.can_auto_resolve is True
    assert decision.identity_mask is False


# ---------------------------------------------------------------------------
# Negative tests — should NOT be sensitive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I am killing time at the park while waiting for the office to open",
    "A gang of workers is repairing the road outside my house making noise",
    "I need a STD 10 certificate from the municipal office",
    "It was a riot of colours at the Holi celebration in our ward",
    "The sword fern plants near the entrance need trimming",
    "STD 12 mark sheet copy required for college admission",
    "Gun salute ceremony at the parade ground disturbed area residents",
    "Stalk of bananas from government scheme not received this month",
    "River flooding — not a safety threat, need embankment repair",
])
def test_negative_texts_not_sensitive(text, routine_category):
    """Known false-positive-risk texts must NOT trigger sensitive hold."""
    guard_res = safety_guard.inspect(text)
    assert not guard_res.is_sensitive, (
        f"False positive: {text!r} → is_sensitive={guard_res.is_sensitive}, "
        f"concept={guard_res.detected_concept}"
    )
    decision = evaluate_policy_gate(
        category=routine_category,
        fused_confidence=0.9,
        retrieval_quality="STRONG",
        sensitive_neighbor_count=0,
        complaint_text=text,
    )
    assert decision.verdict != "SENSITIVE_HOLD", (
        f"False SENSITIVE_HOLD for: {text!r} → rule={decision.rule_id}"
    )


# ---------------------------------------------------------------------------
# Multilingual sensitive keyword detection
# ---------------------------------------------------------------------------

def test_multilingual_sensitive_keywords():
    assert detect_sensitive_keywords("लाचखोरी चालू आहे") is True
    assert detect_sensitive_keywords("रिश्वत मांग रहे हैं") is True
    assert detect_sensitive_keywords("मुलींची छेड काढली जात आहे") is True
    assert detect_sensitive_keywords("बस स्टॉप पर लड़कियों को छेड़ते हैं") is True
    assert detect_sensitive_keywords("चहाच्या टपरीवर लहान मुलाला मारहाण") is True
    assert detect_sensitive_keywords("उघडी तार लटकत आहे करंट लागू शकतो") is True
    assert detect_sensitive_keywords("regular water supply problem") is False


# ---------------------------------------------------------------------------
# Property test: no sensitive/identity_hiding case reaches AUTO_HANDLE
# ---------------------------------------------------------------------------

def test_property_no_sensitive_reaches_auto_handle():
    """
    For all combinations of sensitive/identity_hiding flags,
    no case with either flag True should reach AUTO_HANDLE.
    """
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
    ]
    for sensitive_flag, id_hiding_flag in test_cases:
        cat = Category(
            category_id="C_TEST",
            category="Test",
            subcategory="Test subcategory",
            responsible_department_id="D001",
            responsible_department="Test Dept",
            default_assignee="Test Officer",
            sla_days=7,
            sensitive=sensitive_flag,
            identity_hiding=id_hiding_flag,
            default_priority="MEDIUM",
            auto_resolution_allowed=True,  # even if allowed, sensitive/id flags must override
        )
        decision = evaluate_policy_gate(
            category=cat,
            fused_confidence=1.0,
            retrieval_quality="STRONG",
            sensitive_neighbor_count=0,
            complaint_text="A regular complaint text",
        )
        assert decision.verdict != "AUTO_HANDLE", (
            f"AUTO_HANDLE reached for sensitive={sensitive_flag}, identity_hiding={id_hiding_flag}"
        )
        assert not decision.can_auto_resolve
