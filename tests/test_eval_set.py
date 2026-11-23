"""
Eval Set Test: 120+ complaint cases.
Checks:
- 100% sensitive recall (zero leakage — the hard invariant)
- Routine precision (false positive rate on hard negatives)
Fails if sensitive recall < 100%.
"""
import json
import pytest
from pathlib import Path

from app.db.models import Category
from app.services.safety_guard import safety_guard
from app.services.policy_gate import evaluate_policy_gate

EVAL_PATH = Path(__file__).parent / "data" / "eval_cases.jsonl"


def load_eval_cases():
    cases = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


@pytest.fixture(scope="module")
def routine_category():
    """A routine, auto-resolution-allowed category for testing policy gate."""
    return Category(
        category_id="C009",
        category="Street Lighting and Electricity",
        subcategory="Streetlight not working",
        responsible_department_id="D005",
        responsible_department="Electricity Distribution Department",
        default_assignee="Duty Officer",
        sla_days=5,
        sensitive=False,
        identity_hiding=False,
        default_priority="MEDIUM",
        auto_resolution_allowed=True,
    )


def test_eval_set_loads():
    cases = load_eval_cases()
    assert len(cases) >= 120, f"Expected >= 120 eval cases, got {len(cases)}"


def test_sensitive_recall_100_percent(routine_category):
    """
    HARD INVARIANT: Every case with is_sensitive=True must trigger SENSITIVE_HOLD
    when processed through safety_guard + policy_gate with a routine category.
    Sensitive recall must be 100%. Failure here is a critical bug.
    """
    cases = load_eval_cases()
    sensitive_cases = [c for c in cases if c["is_sensitive"]]
    assert len(sensitive_cases) >= 40, f"Expected >= 40 sensitive cases, got {len(sensitive_cases)}"

    failures = []
    for case in sensitive_cases:
        text = case["text"]
        guard_res = safety_guard.inspect(text)
        decision = evaluate_policy_gate(
            category=routine_category,
            fused_confidence=1.0,  # max confidence — still must NOT auto-handle
            retrieval_quality="STRONG",
            sensitive_neighbor_count=0,
            complaint_text=text,
        )
        if decision.verdict != "SENSITIVE_HOLD":
            failures.append({
                "text": text,
                "lang": case["lang"],
                "expected_family": case["expected_category_family"],
                "got_verdict": decision.verdict,
                "guard_is_sensitive": guard_res.is_sensitive,
                "guard_concept": guard_res.detected_concept,
            })

    if failures:
        summary = "\n".join(
            f"  [{c['lang']}] {c['text'][:80]!r} → {c['got_verdict']} "
            f"(guard: {c['guard_is_sensitive']}, {c['guard_concept']})"
            for c in failures
        )
        pytest.fail(
            f"SENSITIVE RECALL FAILURE: {len(failures)}/{len(sensitive_cases)} sensitive cases "
            f"were NOT routed to SENSITIVE_HOLD:\n{summary}"
        )


def test_routine_precision_hard_negatives(routine_category):
    """
    Hard negatives (is_sensitive=False) should NOT trigger SENSITIVE_HOLD.
    We allow up to 15% false positives (conservative escalations are OK for edge cases).
    """
    cases = load_eval_cases()
    hard_negatives = [c for c in cases if not c["is_sensitive"]]
    assert len(hard_negatives) >= 40, f"Expected >= 40 non-sensitive cases, got {len(hard_negatives)}"

    false_positives = []
    for case in hard_negatives:
        text = case["text"]
        guard_res = safety_guard.inspect(text)
        if guard_res.is_sensitive:
            false_positives.append({
                "text": text,
                "lang": case["lang"],
                "concept": guard_res.detected_concept,
            })

    fp_rate = len(false_positives) / len(hard_negatives)
    # Print info even on pass
    print(f"\nRoutine precision: {len(hard_negatives) - len(false_positives)}/{len(hard_negatives)} correct")
    print(f"False positive rate: {fp_rate:.1%}")
    if false_positives:
        print("False positives (review these patterns):")
        for fp in false_positives:
            print(f"  [{fp['lang']}] {fp['text'][:80]!r} → {fp['concept']}")

    # Allow up to 15% FP rate (some escalation is acceptable; 0% FP impossible without hurting recall)
    assert fp_rate <= 0.15, (
        f"False positive rate {fp_rate:.1%} exceeds 15%. "
        f"Review patterns in safety_guard.py."
    )


def test_bribery_cases_do_not_reach_auto_handle(routine_category):
    """Bribery/corruption cases should be escalated to HUMAN_REVIEW (not SENSITIVE_HOLD, not AUTO_HANDLE)."""
    bribery_texts = [
        "An officer demanded one thousand rupees bribe to register my complaint",
        "लाचखोरी चालू आहे ऑफिसमध्ये पैसे मागतात",
        "रिश्वत मांग रहे हैं अधिकारी काम के लिए",
    ]
    for text in bribery_texts:
        decision = evaluate_policy_gate(
            category=routine_category,
            fused_confidence=0.95,
            retrieval_quality="STRONG",
            sensitive_neighbor_count=0,
            complaint_text=text,
        )
        assert decision.verdict in ("HUMAN_REVIEW", "SENSITIVE_HOLD"), (
            f"Bribery case reached AUTO_HANDLE: {text!r}"
        )
        assert not decision.can_auto_resolve, f"Bribery case allowed auto-resolve: {text!r}"


def test_print_confusion_summary():
    """Prints a full summary table (always passes; informational)."""
    cases = load_eval_cases()
    total = len(cases)
    sensitive = sum(1 for c in cases if c["is_sensitive"])
    routine = total - sensitive

    sensitive_hold_count = 0
    routine_hold_count = 0
    routine_cat = Category(
        category_id="C009",
        category="Street Lighting",
        subcategory="Streetlight not working",
        responsible_department_id="D005",
        responsible_department="Dept",
        default_assignee="Officer",
        sla_days=5,
        sensitive=False,
        identity_hiding=False,
        default_priority="MEDIUM",
        auto_resolution_allowed=True,
    )
    for case in cases:
        g = safety_guard.inspect(case["text"])
        d = evaluate_policy_gate(routine_cat, 1.0, "STRONG", 0, case["text"])
        if case["is_sensitive"] and d.verdict == "SENSITIVE_HOLD":
            sensitive_hold_count += 1
        if not case["is_sensitive"] and d.verdict == "SENSITIVE_HOLD":
            routine_hold_count += 1

    print(f"\n{'='*60}")
    print(f"EVAL SET SUMMARY ({total} cases)")
    print(f"  Sensitive: {sensitive} | Routine/Negative: {routine}")
    print(f"  Sensitive → SENSITIVE_HOLD: {sensitive_hold_count}/{sensitive} ({100*sensitive_hold_count/sensitive:.1f}%)")
    print(f"  Routine → SENSITIVE_HOLD (FP): {routine_hold_count}/{routine} ({100*routine_hold_count/routine:.1f}%)")
    print(f"{'='*60}")
