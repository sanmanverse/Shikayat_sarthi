"""
Two-stage safety assessment (signal generator for the deterministic policy gate).

Stage A  deterministic multilingual regex guard (always runs, offline, free)
Stage B  LLM safety classifier via Sarvam (runs only when a key is configured)

Combination is OR / fail-closed:
  * either stage says sensitive            -> sensitive (disagreement is recorded)
  * stage B configured but failed/unparsed -> status "error": the gate sends the case to a human
  * stage B not configured                 -> status "not_configured": Stage A (+ retrieval proximity)
                                              is the only text-level net; this is stated in the trace

This module never decides a category ID. concept -> category is resolved against the Category
master at runtime by `resolve_category_for_concept` (no hard-coded ID lists).
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, Iterable

from app.services.safety_guard import safety_guard, SafetyGuardResult

logger = logging.getLogger(__name__)


@dataclass
class SafetyAssessment:
    # Same attribute names as SafetyGuardResult so the gate can consume either.
    is_sensitive: bool
    detected_concept: Optional[str]
    reason: str
    identity_mask: bool
    escalation_rule: str
    sensitive_category_id: Optional[str] = None
    # Two-stage details (audit / trace)
    stage_a_sensitive: bool = False
    stage_b_status: str = "not_configured"      # ok | not_configured | error
    stage_b_sensitive: Optional[bool] = None
    stage_b_confidence: Optional[float] = None
    disagreement: bool = False


def assess(text: str, llm_safety=None) -> SafetyAssessment:
    """
    llm_safety: callable(text) -> (result_dict|None, status). Defaults to the Sarvam client.
    Injected in tests so no network is needed.
    """
    a: SafetyGuardResult = safety_guard.inspect(text)

    if llm_safety is None:
        from app.services.sarvam import sarvam_client  # local import: keeps this module light
        llm_safety = sarvam_client.safety_classify

    b_result, b_status = llm_safety(text) if text and text.strip() else (None, "not_configured")
    b_sensitive = bool(b_result["sensitive"]) if b_result else None
    b_conf = b_result.get("confidence") if b_result else None
    b_concept = b_result.get("concept") if b_result else None

    is_sensitive = a.is_sensitive or bool(b_sensitive)
    disagreement = b_status == "ok" and (a.is_sensitive != bool(b_sensitive))

    concept = a.detected_concept
    reason = a.reason
    if is_sensitive and not a.is_sensitive:
        concept = b_concept or "Threat / Public Safety Concern"
        reason = (
            "LLM safety classifier flagged this complaint as potentially sensitive "
            f"({concept}); the deterministic guard did not. Failing closed: mandatory human review."
        )

    return SafetyAssessment(
        is_sensitive=is_sensitive,
        detected_concept=concept,
        reason=reason,
        identity_mask=a.identity_mask or is_sensitive,
        escalation_rule="RULE_01_SENSITIVE_CATEGORY_HOLD" if is_sensitive else a.escalation_rule,
        stage_a_sensitive=a.is_sensitive,
        stage_b_status=b_status,
        stage_b_sensitive=b_sensitive,
        stage_b_confidence=b_conf,
        disagreement=disagreement,
    )


# ---------------------------------------------------------------------------
# concept -> category (resolved from the Category master, never from a fixed ID table)
# ---------------------------------------------------------------------------
_STOP = {"and", "or", "the", "of", "for", "to", "a", "an", "in", "on", "services", "service", "concern"}


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]+", (text or "").lower()) if len(t) > 2 and t not in _STOP}


def resolve_category_for_concept(concept: Optional[str], categories: Iterable) -> Optional[object]:
    """
    Pick the Category-master row (sensitive or identity_hiding only) whose
    category + subcategory text overlaps most with the detected concept.
    Returns None when nothing overlaps (caller keeps its own proposal and the gate still holds).
    """
    if not concept:
        return None
    want = _tokens(concept)
    best, best_score = None, 0
    for cat in categories:
        if not (getattr(cat, "sensitive", False) or getattr(cat, "identity_hiding", False)):
            continue
        have = _tokens(f"{cat.category} {cat.subcategory}")
        score = len(want & have)
        if score > best_score:
            best, best_score = cat, score
    return best
