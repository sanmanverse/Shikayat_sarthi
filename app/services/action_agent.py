"""
Action Agent & Workflow Executor.
Groups 0 fixes:
- email domain → example.invalid with per-grievance hash
- WhatsApp → masked placeholder per grievance
- Notifications always SIMULATED, never DELIVERED
- from_status derived from DB, not hardcoded
Groups 10 fixes:
- SENSITIVE_HOLD: restricted payload (no raw text, no citizen ID)
- HUMAN_REVIEW: officer desk task
- Identity-masked work orders exclude raw text
- All AI actions log real model name, trace_id
"""
import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.models import (
    Grievance, Category, Department, WorkOrder, Notification, StatusHistory, AuditLog
)
from app.services.policy_gate import PolicyGateDecision

logger = logging.getLogger(__name__)


def _masked_email(citizen_id: str) -> str:
    """Per-grievance non-resolvable email placeholder."""
    h = hashlib.sha256(citizen_id.encode()).hexdigest()[:8]
    return f"citizen-{h}@example.invalid"


def _masked_phone(citizen_id: str) -> str:
    """Per-grievance masked WhatsApp placeholder."""
    h = hashlib.sha256(citizen_id.encode()).hexdigest()[:6].upper()
    return f"+91-XXXXXX-{h}"


class ActionAgent:
    def execute_workflow(
        self,
        grievance: Grievance,
        category: Optional[Category],
        decision: PolicyGateDecision,
        db: Session,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes actions dictated by the Policy Gate decision.
        Sensitive cases never get raw text in work orders or notifications.
        """
        # Derive real previous status from StatusHistory (not hardcoded "RECEIVED")
        last_history = (
            db.query(StatusHistory)
            .filter(StatusHistory.grievance_id == grievance.reference_no)
            .order_by(StatusHistory.timestamp.desc())
            .first()
        )
        from_status = last_history.to_status if last_history else "RECEIVED"

        # 1. Update Grievance with Policy Gate outcome
        grievance.policy_verdict = decision.verdict
        grievance.policy_rule = decision.rule_id
        grievance.confidence = decision.confidence
        grievance.identity_mask = decision.identity_mask

        # 2. Assign Department & Default Assignee from Master
        if category:
            grievance.category_id = category.category_id
            grievance.department_id = category.responsible_department_id
            grievance.assigned_to = category.default_assignee
            grievance.priority = category.default_priority

            sla_days = category.sla_days or 7
            grievance.due_date = grievance.created_at + timedelta(days=sla_days)

        actions_taken = []

        # 3. Route according to Verdict
        if decision.verdict == "AUTO_HANDLE":
            grievance.status = "IN_PROGRESS"
            new_note = f"Auto-handling approved under {decision.rule_id}. Work order dispatched."

            work_order = self._create_work_order(grievance, category, db)
            actions_taken.append(
                f"WorkOrder created (ID: {work_order.id}) dispatched to webhook {work_order.mock_webhook_id}"
            )

            cat_name = category.subcategory if category else "your issue"
            sla_days_str = str(category.sla_days) if category else "7"
            self._send_notifications(
                grievance,
                f"Your grievance {grievance.reference_no} regarding {cat_name} "
                f"has been auto-triaged and assigned. Resolution expected within {sla_days_str} days.",
                db,
            )
            actions_taken.append("Simulated notification dispatched to citizen (SIMULATED)")

        elif decision.verdict == "SENSITIVE_HOLD":
            grievance.status = "SENSITIVE_HOLD"
            new_note = (
                f"Sensitive grievance locked under {decision.rule_id}. "
                "Automated resolution strictly prohibited. Placed in Confidential Officer Queue."
            )

            # Restricted escalation webhook (minimal payload — NO raw text, NO citizen ID)
            work_order = self._create_sensitive_work_order(grievance, category, db)
            actions_taken.append(
                f"Restricted escalation webhook dispatched (WO ID: {work_order.id})"
            )

            # Discreet citizen notification (no details about category or content)
            self._send_notifications(
                grievance,
                f"Your confidential submission {grievance.reference_no} has been received "
                "and securely assigned to a designated senior officer. Your identity is protected.",
                db,
            )
            actions_taken.append(
                "Discreet citizen notification sent (identity protected, SIMULATED)"
            )

        else:  # HUMAN_REVIEW
            grievance.status = "TRIAGED"
            new_note = f"Held for officer verification under {decision.rule_id}: {decision.reason}"

            # Officer desk task (non-identifying if identity_mask)
            self._create_officer_task(grievance, category, decision, db)
            actions_taken.append("Officer desk task created")

            self._send_notifications(
                grievance,
                f"Your grievance {grievance.reference_no} is under review by our nodal officer desk.",
                db,
            )
            actions_taken.append("Citizen status notification sent (SIMULATED)")

        # 4. Record Status History
        status_entry = StatusHistory(
            grievance_id=grievance.reference_no,
            from_status=from_status,
            to_status=grievance.status,
            actor="AI_ACTION_AGENT",
            note=new_note,
        )
        db.add(status_entry)

        # 5. Record Audit Log with trace_id
        from app.core.config import settings as _settings
        model_label = "mock_keyword_v1" if not _sarvam_is_real() else _settings.SARVAM_CHAT_MODEL
        audit = AuditLog(
            actor="AI_ACTION_AGENT",
            agent_name="ActionAgent",
            model_name=model_label,
            tool_action="execute_workflow",
            affected_record=grievance.reference_no,
            confidence=decision.confidence,
            policy_rule=decision.rule_id,
            summary=f"Verdict: {decision.verdict}. Actions: {'; '.join(actions_taken)}",
            trace_id=trace_id,
        )
        db.add(audit)
        db.commit()

        return {
            "reference_no": grievance.reference_no,
            "verdict": decision.verdict,
            "rule_id": decision.rule_id,
            "status": grievance.status,
            "assigned_to": grievance.assigned_to,
            "identity_mask": grievance.identity_mask,
            "actions_taken": actions_taken,
        }

    def _create_work_order(
        self, grievance: Grievance, category: Optional[Category], db: Session
    ) -> WorkOrder:
        dept = db.query(Department).filter(Department.department_id == grievance.department_id).first()
        webhook_id = dept.mock_webhook_id if dept else f"MOCK-WEBHOOK-{grievance.department_id or 'D001'}"

        payload = {
            "work_order_id": f"WO-{grievance.reference_no}",
            "grievance_id": grievance.reference_no,
            "department": dept.department_name if dept else "Public Works Department",
            "assignee": grievance.assigned_to or "Duty Officer",
            "location": f"{grievance.locality}, {grievance.taluka}",
            "description": grievance.raw_text,  # allowed for non-sensitive
            "due_date": grievance.due_date.strftime("%Y-%m-%d") if grievance.due_date else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        wo = WorkOrder(
            grievance_id=grievance.reference_no,
            department_id=grievance.department_id or "D001",
            assignee=grievance.assigned_to or "Duty Officer",
            mock_webhook_id=webhook_id,
            payload=json.dumps(payload),
            status="DISPATCHED",
        )
        db.add(wo)
        db.flush()
        return wo

    def _create_sensitive_work_order(
        self, grievance: Grievance, category: Optional[Category], db: Session
    ) -> WorkOrder:
        """Restricted payload: no raw text, no citizen identifier."""
        dept = db.query(Department).filter(Department.department_id == grievance.department_id).first()
        webhook_id = dept.mock_webhook_id if dept else f"MOCK-WEBHOOK-{grievance.department_id or 'D001'}"

        payload = {
            "work_order_id": f"WO-{grievance.reference_no}",
            "grievance_id": grievance.reference_no,
            "department": dept.department_name if dept else "Specialised Dept",
            "category_id": grievance.category_id,
            # NO raw_text, NO citizen_id in sensitive payload
            "masked_location": f"[LOCATION MASKED] {grievance.taluka}",
            "priority": grievance.priority,
            "due_date": grievance.due_date.strftime("%Y-%m-%d") if grievance.due_date else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensitivity_note": "CONFIDENTIAL — restricted to authorised officers only",
        }

        wo = WorkOrder(
            grievance_id=grievance.reference_no,
            department_id=grievance.department_id or "D001",
            assignee=grievance.assigned_to or "Duty Officer",
            mock_webhook_id=webhook_id,
            payload=json.dumps(payload),
            status="DISPATCHED",
        )
        db.add(wo)
        db.flush()
        return wo

    def _create_officer_task(
        self,
        grievance: Grievance,
        category: Optional[Category],
        decision: PolicyGateDecision,
        db: Session,
    ):
        """Create a simple officer-desk work-order for HUMAN_REVIEW cases."""
        dept = db.query(Department).filter(Department.department_id == grievance.department_id).first()
        webhook_id = dept.mock_webhook_id if dept else f"MOCK-WEBHOOK-{grievance.department_id or 'D001'}"

        # If identity masked, do not include raw text even in HUMAN_REVIEW
        desc = (
            "[IDENTITY MASKED — text suppressed for privacy]"
            if grievance.identity_mask
            else grievance.raw_text
        )
        payload = {
            "task_type": "HUMAN_REVIEW",
            "grievance_id": grievance.reference_no,
            "category_id": grievance.category_id,
            "rule": decision.rule_id,
            "reason": decision.reason,
            "description": desc,
            "location": f"{grievance.locality}, {grievance.taluka}",
            "priority": grievance.priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        wo = WorkOrder(
            grievance_id=grievance.reference_no,
            department_id=grievance.department_id or "D001",
            assignee=grievance.assigned_to or "Duty Officer",
            mock_webhook_id=webhook_id,
            payload=json.dumps(payload),
            status="PENDING_OFFICER",
        )
        db.add(wo)
        db.flush()

    def _send_notifications(self, grievance: Grievance, message: str, db: Session):
        """Always SIMULATED; per-grievance masked recipients."""
        citizen_id = grievance.citizen_id or "unknown"
        n_email = Notification(
            grievance_id=grievance.reference_no,
            channel="EMAIL_SIMULATED",
            recipient=_masked_email(citizen_id),
            message=message,
            status="SIMULATED",
        )
        n_wa = Notification(
            grievance_id=grievance.reference_no,
            channel="WHATSAPP_SIMULATED",
            recipient=_masked_phone(citizen_id),
            message=message,
            status="SIMULATED",
        )
        db.add(n_email)
        db.add(n_wa)


def _sarvam_is_real() -> bool:
    """Check if Sarvam API key is configured."""
    from app.core.config import settings
    return bool(settings.SARVAM_API_KEY and len(settings.SARVAM_API_KEY) > 5)


action_agent = ActionAgent()
