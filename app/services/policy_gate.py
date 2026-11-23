"""
Deterministic Policy Gate (Pure Python / Zero Hallucination).
Group 3 rewrite — HARD RULES:
- ALL sensitivity decisions come from Category.sensitive and Category.identity_hiding
  flags in the Category master CSV at runtime.
- NO hard-coded category ID lists anywhere in this file.
- Safety guard provides concepts; policy gate checks the master for sensitivity flags.
- identity_hiding=True alone (even sensitive=False) → mask + HUMAN_REVIEW.
- Gate order:
  0. Independent Safety Guard signal
  1. Unknown/None category
  2. Category.sensitive=True → SENSITIVE_HOLD
  3. Category.identity_hiding=True → HUMAN_REVIEW (identity mask)
  4. Conservative escalation (bribery / sensitive neighbors / sensitive-precedent proximity)
  5. auto_resolution_allowed=False → HUMAN_REVIEW
  5b. Safety stage B configured but failed → HUMAN_REVIEW (fail closed)
  5c. Agent asked for human review → HUMAN_REVIEW (agents can only make the gate stricter)
  6. Low confidence → HUMAN_REVIEW
  7. Weak retrieval → HUMAN_REVIEW
  8. All criteria met → AUTO_HANDLE
"""
from dataclasses import dataclass
from typing import Optional
from app.db.models import Category
from app.core.config import settings
from app.services.safety_guard import safety_guard


@dataclass
class PolicyGateDecision:
    verdict: str           # "SENSITIVE_HOLD", "HUMAN_REVIEW", "AUTO_HANDLE"
    rule_id: str
    reason: str
    identity_mask: bool
    can_auto_resolve: bool
    confidence: float
    # Additional for audit trail
    confidence_components: Optional[dict] = None


def detect_sensitive_keywords(text: str) -> bool:
    """Public helper: returns True if text contains any sensitive signal (for tests)."""
    res = safety_guard.inspect(text)
    return res.is_sensitive or res.escalation_rule == "RULE_02_SENSITIVE_PRECAUTION_ESCALATION"


def evaluate_policy_gate(
    category: Optional[Category],
    fused_confidence: float,
    retrieval_quality: str = "STRONG",
    sensitive_neighbor_count: int = 0,
    complaint_text: str = "",
    llm_confidence: Optional[float] = None,
    top_similarity: Optional[float] = None,
    neighbor_agreement: Optional[float] = None,
    safety_result=None,                       # SafetyAssessment / SafetyGuardResult (computed once upstream)
    sensitive_proximity: Optional[dict] = None,   # {"hit": bool, "category_id": str, "ratio": float}
    agent_escalation: Optional[str] = None,       # reason string when the agent loop asked for a human
) -> PolicyGateDecision:
    """
    Deterministic policy gate. Never auto-handles sensitive or identity-hidden cases.
    Policy flags come ONLY from the Category master row passed in — no hardcoded IDs.
    """
    confidence_components = {
        "llm_confidence": llm_confidence,
        "top_similarity": top_similarity,
        "neighbor_agreement": neighbor_agreement,
        "fused_confidence": fused_confidence,
    }

    proposed_cat_id = category.category_id if category else None

    # 0. Independent Safety Guard signal (text-level, language-aware)
    guard_res = safety_result if safety_result is not None else safety_guard.inspect(
        complaint_text, proposed_category_id=proposed_cat_id
    )

    if guard_res.is_sensitive:
        return PolicyGateDecision(
            verdict="SENSITIVE_HOLD",
            rule_id="RULE_01_SENSITIVE_CATEGORY_HOLD",
            reason=f"Safety guard detected sensitive concept: {guard_res.reason} Automated resolution is permanently disabled.",
            identity_mask=True,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 1. Unknown or missing category → route to human
    if category is None:
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_00_UNKNOWN_CATEGORY",
            reason="Unrecognized category. Routed to human grievance officer for triage.",
            identity_mask=guard_res.identity_mask,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 2. Category master says SENSITIVE → SENSITIVE_HOLD (no hardcoded IDs)
    if category.sensitive:
        return PolicyGateDecision(
            verdict="SENSITIVE_HOLD",
            rule_id="RULE_01_SENSITIVE_CATEGORY_HOLD",
            reason=(
                f"Category '{category.category_id}: {category.subcategory}' is flagged SENSITIVE "
                "in Category Master. Mandatory human review. Automated resolution permanently disabled."
            ),
            identity_mask=True,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 3. identity_hiding=True alone → mask + HUMAN_REVIEW (even if not sensitive)
    if category.identity_hiding:
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_01b_IDENTITY_HIDING",
            reason=(
                f"Category '{category.category_id}: {category.subcategory}' has identity_hiding=True. "
                "Citizen identity masked. Routed to officer for review."
            ),
            identity_mask=True,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 4. Bribery / corruption signal or sensitive neighbors → conservative escalation
    proximity_hit = bool(sensitive_proximity and sensitive_proximity.get("hit"))
    if (
        guard_res.escalation_rule == "RULE_02_SENSITIVE_PRECAUTION_ESCALATION"
        or sensitive_neighbor_count > 0
        or proximity_hit
    ):
        trigger_desc = []
        if sensitive_neighbor_count > 0:
            trigger_desc.append(f"{sensitive_neighbor_count} similar retrieved past case(s) are sensitive")
        if proximity_hit:
            trigger_desc.append(
                "a sensitive precedent "
                f"({sensitive_proximity.get('category_id')}) is almost as similar as the best match "
                f"(ratio {sensitive_proximity.get('ratio'):.2f}) although the proposed category is non-sensitive"
            )
        if guard_res.escalation_rule == "RULE_02_SENSITIVE_PRECAUTION_ESCALATION":
            trigger_desc.append(guard_res.reason)

        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_02_SENSITIVE_PRECAUTION_ESCALATION",
            reason=(
                f"Conservative escalation: proposed category is non-sensitive, but {'; '.join(trigger_desc)}. "
                "Held for officer verification."
            ),
            identity_mask=True,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # Derive base identity mask from master
    identity_mask = bool(category.identity_hiding)

    # 5. Auto-resolution not permitted in Category Master
    if not category.auto_resolution_allowed:
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_03_AUTOMATION_PROHIBITED",
            reason=f"Category '{category.subcategory}' does not permit automated resolution per Category Master policy.",
            identity_mask=identity_mask,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 5b. Safety stage B was configured but failed → cannot prove the text is safe → human
    if getattr(guard_res, "stage_b_status", None) == "error":
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_07_SAFETY_STAGE_B_UNAVAILABLE",
            reason=(
                "The LLM safety classifier was configured but did not return a usable answer, "
                "so the text could not be cleared as safe. Failing closed: officer review."
            ),
            identity_mask=True,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 5c. Agent loop asked for a human (agents may only make the gate stricter, never looser)
    if agent_escalation:
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_08_AGENT_ESCALATION",
            reason=f"Triage agent recommended human review: {agent_escalation[:240]}",
            identity_mask=identity_mask,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 6. Null/unknown confidence → fail closed
    if fused_confidence is None:
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_04_NULL_CONFIDENCE",
            reason="Confidence could not be computed (LLM confidence unavailable). Requires officer review.",
            identity_mask=identity_mask,
            can_auto_resolve=False,
            confidence=0.0,
            confidence_components=confidence_components,
        )

    # 7. Low confidence
    if fused_confidence < settings.AUTO_HANDLE_CONFIDENCE_THRESHOLD:
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_04_LOW_CONFIDENCE",
            reason=(
                f"Confidence {fused_confidence:.2f} is below automated handling threshold "
                f"({settings.AUTO_HANDLE_CONFIDENCE_THRESHOLD:.2f}). Requires officer review."
            ),
            identity_mask=identity_mask,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 8. Weak retrieval signal
    if retrieval_quality.upper() == "WEAK":
        return PolicyGateDecision(
            verdict="HUMAN_REVIEW",
            rule_id="RULE_05_WEAK_RETRIEVAL",
            reason="Historical precedent retrieval is weak or inconclusive. Requires officer review.",
            identity_mask=identity_mask,
            can_auto_resolve=False,
            confidence=fused_confidence,
            confidence_components=confidence_components,
        )

    # 9. All criteria met: routine auto-handle
    return PolicyGateDecision(
        verdict="AUTO_HANDLE",
        rule_id="RULE_06_AUTO_HANDLE_PERMITTED",
        reason=(
            f"Routine grievance approved for standard automated workflow (confidence: {fused_confidence:.2f}). "
            "AI routes and issues work order; final closure requires citizen confirmation."
        ),
        identity_mask=identity_mask,
        can_auto_resolve=True,
        confidence=fused_confidence,
        confidence_components=confidence_components,
    )
