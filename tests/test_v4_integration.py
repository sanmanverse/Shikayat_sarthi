"""DB-backed checks for the v4 fixes (mock mode: no key, no network)."""
import uuid
from datetime import datetime, timezone, timedelta

from app.db.models import Grievance, WorkOrder, AuditLog
from app.services.orchestrator import orchestrator
from app.services.sla_predictor import sla_predictor


def _grievance(db, text, category_id, **kw):
    g = Grievance(
        reference_no=f"V4-{uuid.uuid4().hex[:6].upper()}", citizen_id=f"citizen_{uuid.uuid4().hex[:6]}",
        raw_text=text, language="Hindi", taluka="Nanded", locality="Central Area",
        urgency_level="MEDIUM", priority="MEDIUM", status="RECEIVED", category_id=category_id, **kw,
    )
    db.add(g)
    db.commit()
    return g


def test_reported_leak_hindi_hanging_wire_is_held_even_if_intake_says_routine(db):
    """Intake proposed a routine category with high confidence; retrieval would not save it."""
    g = _grievance(db, "स्कूल के गेट के पास बिजली का तार लटक रहा है", "C009")
    ev = orchestrator.process_grievance(grievance=g, db=db, llm_confidence=0.99)
    assert ev["decision"]["verdict"] == "SENSITIVE_HOLD"
    assert ev["category"]["sensitive_flag"] is True and ev["category"]["category_id"] == "C010"
    assert ev["safety"]["category_override"] and "C009" in ev["safety"]["category_override"]
    assert g.status not in ("RESOLVED", "CITIZEN_SATISFIED", "CLOSED")
    assert db.query(AuditLog).filter(AuditLog.affected_record == g.reference_no,
                                     AuditLog.tool_action == "override_category_by_safety_signal").count() == 1


def test_sensitive_traces_are_restricted_and_routine_are_public_when_enabled(db):
    g = _grievance(db, "A live electric wire is hanging near the school gate", "C009")
    ev = orchestrator.process_grievance(grievance=g, db=db, llm_confidence=0.9)
    assert ev["public_trace_link"] is None                    # never public for a protected case
    assert ev["trace_visibility"] in ("restricted", "disabled")


def test_pipeline_reports_agent_and_retrieval_modes_honestly(db):
    g = _grievance(db, "Garbage has not been collected in our lane for a week", "C006")
    ev = orchestrator.process_grievance(grievance=g, db=db, llm_confidence=0.9)
    assert ev["agent"]["mode"] == "not_run" and "SARVAM_API_KEY" in ev["agent"]["reason"]
    assert ev["retrieval_evidence"]["retrieval_mode"] in ("bm25", "hybrid")
    assert ev["safety"]["stage_b_status"] == "not_configured"


def test_sla_alert_uses_history_and_escalates_without_raw_text(db):
    created = datetime.now(timezone.utc) - timedelta(days=2)
    g = Grievance(
        reference_no=f"V4-SLA-{uuid.uuid4().hex[:6].upper()}", citizen_id="citizen_sla",
        raw_text="Large pothole outside the market, private details inside", category_id="C001",
        department_id="D001", taluka="Nanded", locality="Market Road", status="IN_PROGRESS",
        created_at=created, due_date=created + timedelta(days=5),
    )
    db.add(g)
    db.commit()
    alert = sla_predictor.evaluate_case_sla(g, db)
    assert alert is not None and alert.breach_probability >= 0.75
    assert "historical" in alert.reason.lower()
    wo = db.query(WorkOrder).filter(WorkOrder.grievance_id == g.reference_no, WorkOrder.status == "ESCALATED").first()
    assert wo is not None and "private details" not in wo.payload and "SLA_ESCALATION" in wo.payload
