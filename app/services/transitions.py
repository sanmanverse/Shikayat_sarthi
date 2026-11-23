"""
Central State Transitions, Role Invariants, and Closure Safety.
Group 4:
- Central transition_status() function with strict transition graph.
- Hard safety invariant: For sensitive or identity_hiding grievances,
  closing statuses (RESOLVED, CITIZEN_SATISFIED) can ONLY be executed
  by an officer with role 'OFFICER' or 'SUPERVISOR', NEVER by 'AI_SYSTEM' or unauthenticated citizen.
- Records StatusHistory and AuditLog for every transition.
"""
from typing import Optional, Set
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.db.models import Grievance, Category, StatusHistory, AuditLog

# Allowed status transitions
ALLOWED_TRANSITIONS: dict[str, Set[str]] = {
    "RECEIVED": {"TRIAGED", "IN_PROGRESS", "SENSITIVE_HOLD"},
    "TRIAGED": {"IN_PROGRESS", "SENSITIVE_HOLD", "RESOLVED"},
    "IN_PROGRESS": {"RESOLVED", "SENSITIVE_HOLD", "TRIAGED"},
    "SENSITIVE_HOLD": {"IN_PROGRESS", "RESOLVED", "TRIAGED"},  # only by OFFICER
    "RESOLVED": {"CITIZEN_SATISFIED", "DISPUTED_REOPENED"},
    "DISPUTED_REOPENED": {"IN_PROGRESS", "TRIAGED", "SENSITIVE_HOLD"},
    "CITIZEN_SATISFIED": {"DISPUTED_REOPENED"},
}

CLOSING_STATUSES = {"RESOLVED", "CITIZEN_SATISFIED"}


def transition_status(
    grievance: Grievance,
    new_status: str,
    actor: str,
    role: str,
    note: Optional[str],
    db: Session,
    trace_id: Optional[str] = None,
) -> Grievance:
    """
    Validates and executes a status transition.
    Enforces role permissions and sensitive closure safety.
    """
    current_status = grievance.status or "RECEIVED"
    cat = db.query(Category).filter(Category.category_id == grievance.category_id).first()
    is_sensitive = bool(cat.sensitive if cat else False) or bool(grievance.identity_mask)

    # 1. State machine check
    allowed = ALLOWED_TRANSITIONS.get(current_status, set())
    if new_status not in allowed and new_status != current_status:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Invalid status transition from '{current_status}' to '{new_status}'. "
                f"Allowed target statuses: {sorted(allowed)}"
            ),
        )

    # 2. Sensitive closure safety rule:
    # Sensitive or identity-masked cases cannot be transitioned to closing status by AI or citizen.
    if new_status in CLOSING_STATUSES and is_sensitive:
        if role not in ("OFFICER", "SUPERVISOR"):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"SAFETY VIOLATION: Sensitive grievance {grievance.reference_no} "
                    f"cannot be closed by '{actor}' (role='{role}'). "
                    "Only a verified senior officer may resolve or close sensitive cases."
                ),
            )

    # 3. Moving out of SENSITIVE_HOLD requires OFFICER role
    if current_status == "SENSITIVE_HOLD" and new_status != "SENSITIVE_HOLD":
        if role not in ("OFFICER", "SUPERVISOR"):
            raise HTTPException(
                status_code=403,
                detail="Only an officer can de-escalate or progress a case from SENSITIVE_HOLD.",
            )

    # Execute transition
    prev_status = grievance.status
    grievance.status = new_status
    grievance.updated_at = datetime.now(timezone.utc)

    # Record history
    history = StatusHistory(
        grievance_id=grievance.reference_no,
        from_status=prev_status,
        to_status=new_status,
        actor=actor,
        note=note or f"Status changed from {prev_status} to {new_status} by {actor}.",
    )
    db.add(history)

    # Record audit log
    audit = AuditLog(
        actor=actor,
        agent_name="TransitionManager",
        model_name="deterministic_state_machine",
        tool_action="transition_status",
        affected_record=grievance.reference_no,
        summary=f"Transitioned {prev_status} -> {new_status} by {actor} ({role}). Note: {note or 'N/A'}",
        trace_id=trace_id,
    )
    db.add(audit)
    db.commit()
    db.refresh(grievance)
    return grievance
