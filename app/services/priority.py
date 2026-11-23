"""
Holistic Multi-Factor Priority Engine.
Group 8:
- Computes priority score from:
  1. Base default priority from Category Master (CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1)
  2. Urgency boost from intake text analysis (never lowers CRITICAL or HIGH)
  3. Vulnerable group detection (women, children, elderly, disabled)
  4. Affected population volume signal (capped)
- priority_up_only(): priority never decreases on citizen dispute or re-triage.
  Decreasing priority strictly requires OFFICER role with recorded rationale.
"""
import re
from typing import Dict, Any, Optional

PRIORITY_RANKS = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
RANK_TO_PRIORITY = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}

VULNERABLE_KEYWORDS = [
    r"\bchildren\b", r"\bchild\b", r"\bminor\b", r"\bgirls?\b", r"\bwomen\b",
    r"\belderly\b", r"\bsenior\s+citizen\b", r"\bhandicap(?:ped)?\b", r"\bdisabled\b",
    r"\bdivyang\b", r"\bलहान\s+मूल\b", r"\bवृद्ध\b", r"\bअपंग\b", r"\bदिव्यांग\b",
    r"\bमहिला\b", r"\bबच्चे\b", r"\bबुजुर्ग\b"
]

HIGH_POPULATION_KEYWORDS = [
    r"\bentire\s+(?:village|colony|ward|street|locality)\b",
    r"\bhundreds\s+of\s+people\b",
    r"\bthousands\b",
    r"\ball\s+residents\b",
    r"\bschool\s+(?:children|students)\b",
    r"\bhospital\s+area\b",
    r"\bसर्व\s+नागरिक\b",
    r"\bसंपूर्ण\s+गाव\b",
]

def calculate_priority_score(
    base_priority: str,
    complaint_text: str,
    urgency_hint: str = "MEDIUM",
) -> Dict[str, Any]:
    """
    Calculates composite priority score.
    Returns:
        level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        score: float
        components: Dict[str, Any]
    """
    base_score = float(PRIORITY_RANKS.get(base_priority.upper(), 2))
    urgency_score = float(PRIORITY_RANKS.get(urgency_hint.upper(), 2))

    # Vulnerable group bonus
    vulnerable_detected = any(
        re.search(pat, complaint_text, re.IGNORECASE) for pat in VULNERABLE_KEYWORDS
    )
    vulnerable_bonus = 0.5 if vulnerable_detected else 0.0

    # Large population affected bonus (capped at 0.5)
    population_detected = any(
        re.search(pat, complaint_text, re.IGNORECASE) for pat in HIGH_POPULATION_KEYWORDS
    )
    population_bonus = 0.5 if population_detected else 0.0

    # Composite score
    composite = max(base_score, urgency_score * 0.8) + vulnerable_bonus + population_bonus

    # Map to priority level (never lower CRITICAL or HIGH base)
    if base_priority == "CRITICAL" or composite >= 3.8:
        level = "CRITICAL"
    elif base_priority == "HIGH" or composite >= 2.8:
        level = "HIGH"
    elif composite >= 1.8:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "priority_level": level,
        "composite_score": round(composite, 2),
        "components": {
            "base_priority": base_priority,
            "base_score": base_score,
            "urgency_hint": urgency_hint,
            "vulnerable_bonus": vulnerable_bonus,
            "population_bonus": population_bonus,
        }
    }


def priority_up_only(
    current_priority: str,
    suggested_priority: str,
    role: str = "SYSTEM",
) -> str:
    """
    Ensures priority can only escalate upward automatically.
    De-escalation requires OFFICER/SUPERVISOR role.
    """
    curr_rank = PRIORITY_RANKS.get(current_priority.upper(), 2)
    sugg_rank = PRIORITY_RANKS.get(suggested_priority.upper(), 2)

    if sugg_rank >= curr_rank:
        return suggested_priority.upper()

    # Decreasing priority: only officers can do this
    if role in ("OFFICER", "SUPERVISOR"):
        return suggested_priority.upper()

    # Reject automatic downgrade
    return current_priority.upper()
