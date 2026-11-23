"""
Predictive SLA Intelligence & Bottleneck Monitor (Module 05).

Real, history-driven prediction (v4):
- Per-category resolution durations are computed from the mock history (resolution_date - submission_date
  of Resolved/Closed rows; see sla_stats.py). The previous version averaged the configured `sla_days`
  column and filtered on status == "RESOLVED" (the data says "Resolved"), so it never used real durations.
- Breach probability is a conditional survival estimate: P(duration > SLA | still open at its current age).
- Department backlog and same-area recurrence are small, reported adjustments (never a fixed sentence).
- The alert text is built from the numbers actually used.
- A raised alert is dispatched as an escalation (mock work order to the department webhook + simulated
  supervisor notification) so it is an action, not just a row.
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.db.models import (
    Grievance, Category, Department, HistoryGrievance, SLAAlert, AuditLog, WorkOrder, Notification,
)
from app.services.sla_stats import build_duration_table, estimate_breach, duration_days, DurationTable

logger = logging.getLogger(__name__)

OPEN_STATUSES = ["RECEIVED", "TRIAGED", "IN_PROGRESS", "DISPUTED_REOPENED", "SENSITIVE_HOLD"]


class SLAPredictor:
    def __init__(self):
        self._table: Optional[DurationTable] = None
        self._table_rows = -1

    def duration_table(self, db: Session) -> DurationTable:
        """Cached per-category duration table; rebuilt if the history row count changes."""
        count = db.query(func.count(HistoryGrievance.grievance_id)).scalar() or 0
        if self._table is None or count != self._table_rows:
            rows = db.query(
                HistoryGrievance.category_id,
                HistoryGrievance.department_id,
                HistoryGrievance.status,
                HistoryGrievance.submission_date,
                HistoryGrievance.resolution_date,
            ).all()
            self._table = build_duration_table(rows)
            self._table_rows = count
            logger.info("SLA duration table built from %d resolved history rows.", self._table.size)
        return self._table

    def evaluate_case_sla(self, grievance: Grievance, db: Session) -> Optional[SLAAlert]:
        """
        Estimates breach probability for an open grievance and raises a pre-emptive alert
        (plus escalation dispatch) when it reaches SLA_ALERT_THRESHOLD.
        """
        if grievance.status in ("RESOLVED", "CITIZEN_SATISFIED"):
            return None

        cat = db.query(Category).filter(Category.category_id == grievance.category_id).first()
        sla_days = cat.sla_days if cat else 7
        created = grievance.created_at
        created_time = created.replace(tzinfo=timezone.utc) if created.tzinfo is None else created
        elapsed_days = (datetime.now(timezone.utc) - created_time).total_seconds() / 86400.0

        dept_open_count = (
            db.query(func.count(Grievance.id))
            .filter(Grievance.department_id == grievance.department_id, Grievance.status.in_(OPEN_STATUSES))
            .scalar() or 0
        )
        recent_same_area = (
            db.query(func.count(Grievance.id))
            .filter(
                Grievance.category_id == grievance.category_id,
                Grievance.taluka == grievance.taluka,
                Grievance.locality == grievance.locality,
                Grievance.id != grievance.id,
                Grievance.created_at >= (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None),
            )
            .scalar() or 0
        )

        est = estimate_breach(
            self.duration_table(db),
            category_id=grievance.category_id,
            department_id=grievance.department_id,
            sla_days=sla_days,
            elapsed_days=elapsed_days,
            backlog_open=dept_open_count,
            recurring=recent_same_area >= 2,
            min_samples=settings.SLA_MIN_SAMPLES,
        )
        breach_prob = round(est.probability, 2)

        if breach_prob < settings.SLA_ALERT_THRESHOLD:
            return None

        existing_alert = (
            db.query(SLAAlert)
            .filter(SLAAlert.grievance_id == grievance.reference_no, SLAAlert.status == "ACTIVE")
            .first()
        )
        if existing_alert:
            return None

        label = (cat.subcategory if cat else "comparable")
        due_dt = grievance.due_date or (created_time + timedelta(days=sla_days))
        alert = SLAAlert(
            grievance_id=grievance.reference_no,
            breach_probability=breach_prob,
            reason=est.explain(sla_days, elapsed_days, label),
            due_date=due_dt,
            status="ACTIVE",
        )
        db.add(alert)
        db.add(AuditLog(
            actor="AI_SLA_MONITOR",
            agent_name="SLAPredictor",
            model_name=f"empirical_survival:{est.method}",
            tool_action="raise_sla_alert",
            affected_record=grievance.reference_no,
            confidence=breach_prob,
            policy_rule="RULE_PREEMPTIVE_SLA_BREACH",
            summary=(
                f"SLA alert for {grievance.reference_no}: P(breach)={breach_prob:.2f} via {est.method} "
                f"(n={est.n_samples}, conditional n={est.n_conditional}, elapsed {elapsed_days:.1f}d / SLA {sla_days}d)."
            ),
        ))
        self._dispatch_escalation(grievance, alert, breach_prob, est.already_overdue, db)
        db.commit()
        return alert

    def _dispatch_escalation(self, grievance: Grievance, alert: SLAAlert, prob: float, overdue: bool, db: Session):
        """Mock work order to the department webhook + simulated supervisor notification. Minimal payload."""
        dept = db.query(Department).filter(Department.department_id == grievance.department_id).first()
        webhook_id = dept.mock_webhook_id if dept else f"MOCK-WEBHOOK-{grievance.department_id or 'D001'}"
        payload = {
            "task_type": "SLA_ESCALATION",
            "grievance_id": grievance.reference_no,
            "category_id": grievance.category_id,
            "breach_probability": prob,
            "already_overdue": overdue,
            "priority": grievance.priority,
            # No raw text and no citizen identifier in escalation payloads.
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        db.add(WorkOrder(
            grievance_id=grievance.reference_no,
            department_id=grievance.department_id or "D001",
            assignee="Department Supervisor",
            mock_webhook_id=webhook_id,
            payload=json.dumps(payload),
            status="ESCALATED",
        ))
        db.add(Notification(
            grievance_id=grievance.reference_no,
            channel="EMAIL_SIMULATED",
            recipient=(dept.contact_email_demo if dept and getattr(dept, "contact_email_demo", None) else "supervisor@example.invalid"),
            message=f"SLA escalation for {grievance.reference_no}: breach probability {prob:.0%}.",
            status="SIMULATED",
        ))

    def get_department_bottlenecks(self, db: Session) -> List[Dict[str, Any]]:
        """Department bottleneck analytics: live backlog + real historical resolution performance."""
        table = self.duration_table(db)
        departments = db.query(Department).all()
        insights = []

        for dept in departments:
            total_cases = db.query(func.count(Grievance.id)).filter(Grievance.department_id == dept.department_id).scalar() or 0
            open_cases = db.query(func.count(Grievance.id)).filter(
                Grievance.department_id == dept.department_id,
                Grievance.status.in_(OPEN_STATUSES),
            ).scalar() or 0

            alerts_count = (
                db.query(func.count(SLAAlert.id))
                .join(Grievance, SLAAlert.grievance_id == Grievance.reference_no)
                .filter(Grievance.department_id == dept.department_id, SLAAlert.status == "ACTIVE")
                .scalar() or 0
            )

            # Chronic recurrence: the SAME category repeating in the SAME area (last 30 days)
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None)
            hotspots = (
                db.query(Grievance.category_id, Grievance.taluka, Grievance.locality, func.count(Grievance.id).label("count"))
                .filter(Grievance.department_id == dept.department_id, Grievance.created_at >= cutoff)
                .group_by(Grievance.category_id, Grievance.taluka, Grievance.locality)
                .having(func.count(Grievance.id) >= 2)
                .all()
            )

            # Historical performance of this department (real durations)
            hist = table.by_department.get(dept.department_id, [])
            hist_rows = db.query(HistoryGrievance.sla_days, HistoryGrievance.status, HistoryGrievance.submission_date,
                                 HistoryGrievance.resolution_date).filter(
                HistoryGrievance.department_id == dept.department_id).all()
            breach_n = resolved_n = 0
            for sla, status, sub, res in hist_rows:
                if (status or "").lower() in ("resolved", "closed"):
                    d = duration_days(sub, res)
                    if d is not None:
                        resolved_n += 1
                        breach_n += 1 if d > sla else 0
            median = sorted(hist)[len(hist) // 2] if hist else None

            insights.append({
                "department_id": dept.department_id,
                "department_name": dept.department_name,
                "total_tickets": total_cases,
                "active_backlog": open_cases,
                "active_sla_alerts": alerts_count,
                "historical_resolved_cases": resolved_n,
                "historical_breach_rate": round(breach_n / resolved_n, 2) if resolved_n else None,
                "historical_median_days": round(median, 1) if median is not None else None,
                "risk_level": "CRITICAL" if alerts_count >= 2 or open_cases > 5 else ("ELEVATED" if open_cases > 2 else "NORMAL"),
                "recurring_hotspots": [
                    {"category_id": h[0], "taluka": h[1], "locality": h[2], "count": h[3]}
                    for h in hotspots
                ],
            })

        return sorted(insights, key=lambda x: (x["active_sla_alerts"], x["active_backlog"]), reverse=True)


sla_predictor = SLAPredictor()
