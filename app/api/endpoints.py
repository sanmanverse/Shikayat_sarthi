from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import (
    Category, Department, HistoryGrievance, Grievance, StatusHistory, WorkOrder,
    AuditLog, SLAAlert, Incident, IncidentMember
)
from app.services.intake_agent import intake_agent
from app.services.orchestrator import orchestrator
from app.services.sarvam import sarvam_client
from app.services.sla_predictor import sla_predictor
from app.services.clustering import clustering_engine
from app.services.transitions import transition_status
from app.services.priority import priority_up_only
from app.core.config import settings
from app.core.observability import tracer

router = APIRouter()
_RECEIVED_WEBHOOK_PAYLOADS: Dict[str, List[Dict[str, Any]]] = {}


# --- Health Endpoint (Group 1) ---
@router.get("/health")
def health_check():
    """Reports the live mode of each subsystem (no secrets)."""
    llm_mode = "real" if (settings.SARVAM_API_KEY and len(settings.SARVAM_API_KEY) > 5) else "mock"
    langfuse_enabled = tracer.is_enabled
    try:
        from app.nlp.embedding_retrieval import get_embedding_engine
        engine = get_embedding_engine()
        emb_ready = bool(engine is not None and engine.is_ready)
    except Exception:
        emb_ready = False

    langfuse_msg = None
    if not langfuse_enabled:
        langfuse_msg = "Tracing disabled: set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env"

    return {
        "status": "healthy",
        "app_version": settings.APP_VERSION,
        "llm_mode": llm_mode,
        "llm_model": settings.SARVAM_CHAT_MODEL if llm_mode == "real" else "mock_keyword_v1",
        "stt_mode": llm_mode,
        "stt_model": settings.SARVAM_STT_MODEL if llm_mode == "real" else None,
        "agent_loop": "guarded_agentic" if (llm_mode == "real" and settings.AGENT_LOOP_ENABLED) else "deterministic_pipeline",
        "safety_stage_b": "llm_classifier" if llm_mode == "real" else "not_configured (regex stage + retrieval proximity only)",
        "langfuse_enabled": langfuse_enabled,
        "langfuse_message": langfuse_msg,
        "langfuse_errors": tracer.error_count,
        "langfuse_last_error": tracer.last_error,
        "embedding_backend": settings.EMBEDDING_MODEL if emb_ready else "none",
        "retrieval_mode": "hybrid" if emb_ready else "bm25",
    }

# --- Request / Response Models ---
class ChatTurnRequest(BaseModel):
    conversation_history: List[Dict[str, str]] = []
    message: str
    citizen_id: str = "citizen_demo_01"

class ConfirmDraftRequest(BaseModel):
    draft: Dict[str, Any]
    citizen_id: str = "citizen_demo_01"

class DisputeRequest(BaseModel):
    dispute_reason: str

class StatusUpdateRequest(BaseModel):
    status: str
    actor: str = "NODAL_OFFICER"
    note: Optional[str] = None

class AssignUpdateRequest(BaseModel):
    department_id: str
    assignee: str
    actor: str = "NODAL_OFFICER"

class MemberActionRequest(BaseModel):
    officer_name: str = "Nodal Officer"


# --- 1. Master Data Endpoints (Guide Section 6) ---
@router.get("/categories")
def get_categories(db: Session = Depends(get_db)):
    """Source of truth for grievance categories, SLAs, and policy flags."""
    categories = db.query(Category).all()
    return [
        {
            "category_id": c.category_id,
            "category": c.category,
            "subcategory": c.subcategory,
            "responsible_department_id": c.responsible_department_id,
            "default_assignee": c.default_assignee,
            "sla_days": c.sla_days,
            "sensitive": c.sensitive,
            "identity_hiding": c.identity_hiding,
            "default_priority": c.default_priority,
            "auto_resolution_allowed": c.auto_resolution_allowed,
        }
        for c in categories
    ]

@router.get("/departments")
def get_departments(db: Session = Depends(get_db)):
    """Department directory and mock routing webhooks."""
    departments = db.query(Department).all()
    return [
        {
            "department_id": d.department_id,
            "department_name": d.department_name,
            "service_scope": d.service_scope,
            "jurisdiction_type": d.jurisdiction_type,
            "jurisdiction_mapping": d.jurisdiction_mapping,
            "contact_email_demo": d.contact_email_demo,
            "contact_channel": d.contact_channel,
            "mock_webhook_id": d.mock_webhook_id,
        }
        for d in departments
    ]

@router.get("/grievances/history")
def get_grievance_history(
    limit: int = 50,
    category_id: Optional[str] = None,
    taluka: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Retrieve sample historical cases from the 800-case corpus."""
    q = db.query(HistoryGrievance)
    if category_id:
        q = q.filter(HistoryGrievance.category_id == category_id)
    if taluka:
        q = q.filter(HistoryGrievance.broad_location_taluka == taluka)
    records = q.limit(limit).all()
    return records


# --- 2. Conversational Intake Endpoints ---
@router.post("/intake/chat")
def intake_chat_turn(req: ChatTurnRequest, db: Session = Depends(get_db)):
    """Citizen chat turn in Marathi, Hindi, or English."""
    result = intake_agent.process_turn(
        conversation_history=req.conversation_history,
        current_message=req.message,
        db=db
    )
    return result

@router.post("/intake/confirm")
def intake_confirm_draft(req: ConfirmDraftRequest, db: Session = Depends(get_db)):
    """Citizen confirms draft. Grievance is registered, clustered for duplicates, and processed by Orchestrator."""
    grievance = intake_agent.confirm_and_submit(
        draft_data=req.draft,
        citizen_id=req.citizen_id,
        db=db
    )
    
    # Check for suggested duplicate grouping (Module 04)
    suggested_incident = clustering_engine.check_and_cluster(grievance=grievance, db=db)

    # Run Orchestrator (RAG -> Fusion -> Policy Gate -> Action Agent -> Tracing)
    # Pass intake draft's real confidence into process_grievance (Group 11)
    llm_conf = req.draft.get("confidence")
    evidence_packet = orchestrator.process_grievance(
        grievance=grievance, db=db, llm_confidence=llm_conf
    )
    
    if suggested_incident:
        evidence_packet["duplicate_grouping"] = {
            "incident_id": suggested_incident.id,
            "status": "SUGGESTED",
            "note": "Suggested group found with nearby similar complaints. Held for officer confirmation."
        }
    
    return evidence_packet

@router.post("/intake/voice")
async def intake_voice(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Transcribes audio using Sarvam STT and feeds text into the intake agent."""
    contents = await file.read()
    stt_res = sarvam_client.transcribe_audio(contents, filename=file.filename or "audio.wav")
    
    turn_res = intake_agent.process_turn(
        conversation_history=[],
        current_message=stt_res["transcript"],
        db=db
    )
    turn_res["transcription"] = stt_res
    return turn_res


# --- 3. Grievance Queue & Evidence (Officer/Admin Portal) ---
@router.get("/grievances")
def list_grievances(
    status: Optional[str] = None,
    department_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lists grievances with policy verdicts and masked PII for sensitive cases."""
    q = db.query(Grievance)
    if status:
        q = q.filter(Grievance.status == status)
    if department_id:
        q = q.filter(Grievance.department_id == department_id)
    
    grievances = q.order_by(Grievance.created_at.desc()).all()
    results = []
    for g in grievances:
        display_citizen = "ANONYMOUS_PROTECTED" if g.identity_mask else g.citizen_id
        display_text = "[REDACTED — identity-masked complaint]" if g.identity_mask else g.raw_text
        results.append({
            "reference_no": g.reference_no,
            "citizen_id": display_citizen,
            "raw_text": display_text,
            "category_id": g.category_id,
            "department_id": g.department_id,
            "taluka": g.taluka,
            "locality": g.locality,
            "priority": g.priority,
            "status": g.status,
            "confidence": g.confidence,
            "policy_verdict": g.policy_verdict,
            "policy_rule": g.policy_rule,
            "identity_mask": g.identity_mask,
            "assigned_to": g.assigned_to,
            "created_at": g.created_at.isoformat() if g.created_at else None,
            "due_date": g.due_date.isoformat() if g.due_date else None,
        })
    return results

@router.get("/grievances/{ref_no}")
def get_grievance_details(ref_no: str, db: Session = Depends(get_db)):
    """Full detail view with precedent evidence, status history, work orders, and notifications."""
    g = db.query(Grievance).filter(Grievance.reference_no == ref_no).first()
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    
    cat = db.query(Category).filter(Category.category_id == g.category_id).first()
    history = db.query(StatusHistory).filter(StatusHistory.grievance_id == ref_no).all()
    work_orders = db.query(WorkOrder).filter(WorkOrder.grievance_id == ref_no).all()
    
    return {
        "grievance": {
            "reference_no": g.reference_no,
            "citizen_id": "ANONYMOUS_PROTECTED" if g.identity_mask else g.citizen_id,
            "raw_text": g.raw_text,
            "taluka": g.taluka,
            "locality": g.locality,
            "status": g.status,
            "priority": g.priority,
            "confidence": g.confidence,
            "policy_verdict": g.policy_verdict,
            "policy_rule": g.policy_rule,
            "assigned_to": g.assigned_to,
            "due_date": g.due_date.isoformat() if g.due_date else None,
        },
        "category": {
            "category_id": cat.category_id if cat else None,
            "category": cat.category if cat else None,
            "subcategory": cat.subcategory if cat else None,
            "sla_days": cat.sla_days if cat else None,
            "sensitive": cat.sensitive if cat else False,
            "identity_hiding": cat.identity_hiding if cat else False,
            "auto_resolution_allowed": cat.auto_resolution_allowed if cat else False,
        },
        "status_history": [
            {
                "from_status": sh.from_status,
                "to_status": sh.to_status,
                "actor": sh.actor,
                "note": sh.note,
                "timestamp": sh.timestamp.isoformat()
            }
            for sh in history
        ],
        "work_orders": [
            {
                "id": wo.id,
                "department_id": wo.department_id,
                "assignee": wo.assignee,
                "webhook_id": wo.mock_webhook_id,
                "status": wo.status,
                "created_at": wo.created_at.isoformat()
            }
            for wo in work_orders
        ]
    }


# --- 4. Officer Actions & State Transitions ---
@router.patch("/grievances/{ref_no}/status")
def update_grievance_status(ref_no: str, req: StatusUpdateRequest, db: Session = Depends(get_db)):
    """Officer status update with mandatory audit log and state transition validation."""
    g = db.query(Grievance).filter(Grievance.reference_no == ref_no).first()
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    
    role = "OFFICER" if ("OFFICER" in req.actor or "SUPERVISOR" in req.actor) else (
        "SYSTEM" if ("AI" in req.actor or "SYSTEM" in req.actor) else "CITIZEN"
    )
    transition_status(
        grievance=g,
        new_status=req.status,
        actor=req.actor,
        role=role,
        note=req.note,
        db=db,
    )
    return {"reference_no": ref_no, "new_status": g.status}


@router.patch("/grievances/{ref_no}/assign")
def assign_grievance(ref_no: str, req: AssignUpdateRequest, db: Session = Depends(get_db)):
    """Assigns department and officer to grievance with validation and audit."""
    g = db.query(Grievance).filter(Grievance.reference_no == ref_no).first()
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    
    dept = db.query(Department).filter(Department.department_id == req.department_id).first()
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{req.department_id}' not found in directory")
    
    old_dept = g.department_id
    old_assignee = g.assigned_to
    g.department_id = req.department_id
    g.assigned_to = req.assignee
    
    db.add(AuditLog(
        actor=req.actor,
        agent_name="AssignHandler",
        model_name="deterministic_routing",
        tool_action="reassign_grievance",
        affected_record=ref_no,
        summary=f"Reassigned from {old_dept}/{old_assignee} to {req.department_id}/{req.assignee}"
    ))
    db.commit()
    return {"reference_no": ref_no, "department_id": g.department_id, "assigned_to": g.assigned_to}


# --- 5. Citizen Dispute Loop (PRD R25 / Module 07) ---
@router.post("/grievances/{ref_no}/dispute")
def dispute_resolution(ref_no: str, req: DisputeRequest, db: Session = Depends(get_db)):
    """Citizen disputes the resolution. Automatically reopens grievance with elevated priority."""
    g = db.query(Grievance).filter(Grievance.reference_no == ref_no).first()
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    
    transition_status(
        grievance=g,
        new_status="DISPUTED_REOPENED",
        actor="CITIZEN_DISPUTE",
        role="CITIZEN",
        note=f"Citizen disputed resolution: '{req.dispute_reason}'. Reopened automatically.",
        db=db,
    )
    g.priority = priority_up_only(g.priority or "MEDIUM", "HIGH", role="CITIZEN")
    db.commit()
    return {
        "reference_no": ref_no,
        "status": "DISPUTED_REOPENED",
        "message": "Your dispute has been registered. The case has been reopened and escalated to senior oversight."
    }

@router.post("/grievances/{ref_no}/confirm-resolution")
def confirm_resolution(ref_no: str, db: Session = Depends(get_db)):
    """Citizen confirms resolution. Only valid from RESOLVED status."""
    g = db.query(Grievance).filter(Grievance.reference_no == ref_no).first()
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    
    transition_status(
        grievance=g,
        new_status="CITIZEN_SATISFIED",
        actor="CITIZEN_FEEDBACK",
        role="CITIZEN",
        note="Citizen confirmed successful resolution.",
        db=db,
    )
    return {"reference_no": ref_no, "status": "CITIZEN_SATISFIED"}


# --- Work Orders Endpoints ---
class CreateWorkOrderRequest(BaseModel):
    grievance_id: str
    department_id: str
    assignee: str
    mock_webhook_id: str
    payload: Dict[str, Any]

@router.post("/work-orders")
def create_work_order(req: CreateWorkOrderRequest, db: Session = Depends(get_db)):
    """Creates and dispatches a work order."""
    import json
    wo = WorkOrder(
        grievance_id=req.grievance_id,
        department_id=req.department_id,
        assignee=req.assignee,
        mock_webhook_id=req.mock_webhook_id,
        payload=json.dumps(req.payload),
        status="DISPATCHED"
    )
    db.add(wo)
    db.commit()
    db.refresh(wo)
    return {"id": wo.id, "grievance_id": wo.grievance_id, "status": wo.status}

@router.get("/work-orders")
def list_work_orders(grievance_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Lists dispatched work orders."""
    q = db.query(WorkOrder)
    if grievance_id:
        q = q.filter(WorkOrder.grievance_id == grievance_id)
    orders = q.order_by(WorkOrder.created_at.desc()).all()
    return [
        {
            "id": wo.id,
            "grievance_id": wo.grievance_id,
            "department_id": wo.department_id,
            "assignee": wo.assignee,
            "webhook_id": wo.mock_webhook_id,
            "status": wo.status,
            "created_at": wo.created_at.isoformat() if wo.created_at else None,
        }
        for wo in orders
    ]


# --- Dev Backdate & Mock Webhooks ---
class BackdateRequest(BaseModel):
    reference_no: str
    days: int

@router.post("/dev/backdate")
def dev_backdate(req: BackdateRequest, db: Session = Depends(get_db)):
    """Dev utility for SLA breach simulation. Disabled unless ENABLE_DEV_BACKDATE is True."""
    if not settings.ENABLE_DEV_BACKDATE:
        raise HTTPException(status_code=403, detail="Dev backdate is disabled in configuration.")
    from datetime import timedelta
    g = db.query(Grievance).filter(Grievance.reference_no == req.reference_no).first()
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    g.created_at = g.created_at - timedelta(days=req.days)
    if g.due_date:
        g.due_date = g.due_date - timedelta(days=req.days)
    db.commit()
    return {"reference_no": req.reference_no, "backdated_days": req.days, "new_created_at": g.created_at.isoformat()}

@router.post("/mock-webhooks/{mock_webhook_id}")
def receive_mock_webhook(mock_webhook_id: str, payload: Dict[str, Any]):
    """Receives simulated webhook calls for integration testing."""
    if mock_webhook_id not in _RECEIVED_WEBHOOK_PAYLOADS:
        _RECEIVED_WEBHOOK_PAYLOADS[mock_webhook_id] = []
    _RECEIVED_WEBHOOK_PAYLOADS[mock_webhook_id].append(payload)
    return {"status": "received", "mock_webhook_id": mock_webhook_id, "payload_count": len(_RECEIVED_WEBHOOK_PAYLOADS[mock_webhook_id])}

@router.get("/mock-webhooks/{mock_webhook_id}")
def get_mock_webhook_payloads(mock_webhook_id: str):
    """Retrieves recorded payloads for a mock webhook."""
    return {"mock_webhook_id": mock_webhook_id, "payloads": _RECEIVED_WEBHOOK_PAYLOADS.get(mock_webhook_id, [])}


# --- 6. Duplicate Detection / Incident Clusters (Module 04) ---
@router.get("/incidents")
def list_incidents(db: Session = Depends(get_db)):
    """Lists suggested and confirmed duplicate incident clusters."""
    incidents = db.query(Incident).all()
    results = []
    for inc in incidents:
        members = db.query(IncidentMember).filter(IncidentMember.incident_id == inc.id).all()
        results.append({
            "incident_id": inc.id,
            "root_grievance_id": inc.root_grievance_id,
            "category_id": inc.category_id,
            "taluka": inc.taluka,
            "locality": inc.locality,
            "status": inc.status,
            "created_at": inc.created_at.isoformat() if inc.created_at else None,
            "members": [
                {
                    "member_id": m.id,
                    "grievance_id": m.grievance_id,
                    "similarity": m.similarity,
                    "status": m.status,
                    "confirmed_by": m.confirmed_by
                }
                for m in members
            ]
        })
    return results

@router.post("/incidents/members/{member_id}/confirm")
def confirm_duplicate_member(member_id: int, req: MemberActionRequest, db: Session = Depends(get_db)):
    """Officer confirms that a complaint belongs to this incident group."""
    success = clustering_engine.confirm_member(member_id, req.officer_name, db)
    if not success:
        raise HTTPException(status_code=404, detail="Incident member not found")
    return {"member_id": member_id, "status": "CONFIRMED"}

@router.post("/incidents/members/{member_id}/reject")
def reject_duplicate_member(member_id: int, req: MemberActionRequest, db: Session = Depends(get_db)):
    """Officer rejects grouping for this complaint."""
    success = clustering_engine.reject_member(member_id, req.officer_name, db)
    if not success:
        raise HTTPException(status_code=404, detail="Incident member not found")
    return {"member_id": member_id, "status": "REJECTED"}


# --- 7. Predictive SLA Intelligence & Alerts (Module 05) ---
@router.get("/sla/alerts")
def get_sla_alerts(status: str = "ACTIVE", db: Session = Depends(get_db)):
    """Returns active pre-emptive SLA breach alerts."""
    alerts = db.query(SLAAlert).filter(SLAAlert.status == status).all()
    return [
        {
            "id": a.id,
            "grievance_id": a.grievance_id,
            "breach_probability": a.breach_probability,
            "reason": a.reason,
            "due_date": a.due_date.isoformat() if a.due_date else None,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "status": a.status
        }
        for a in alerts
    ]

@router.post("/sla/scan")
def trigger_sla_scan(db: Session = Depends(get_db)):
    """Evaluates all open grievances and raises pre-emptive alerts."""
    open_cases = db.query(Grievance).filter(
        Grievance.status.in_(["RECEIVED", "TRIAGED", "IN_PROGRESS", "DISPUTED_REOPENED"])
    ).all()
    raised = []
    for g in open_cases:
        alert = sla_predictor.evaluate_case_sla(g, db)
        if alert:
            raised.append({"grievance_id": g.reference_no, "prob": alert.breach_probability})
    return {"scanned_count": len(open_cases), "alerts_raised": raised}

@router.get("/analytics/bottlenecks")
def get_bottlenecks(db: Session = Depends(get_db)):
    """Supervisor bottleneck insights and chronic recurrence hotspots."""
    return sla_predictor.get_department_bottlenecks(db)


# --- 8. Audit Logs ---
@router.get("/audit-logs")
def get_audit_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Audit log trail for AI and human actions."""
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "timestamp": l.timestamp.isoformat(),
            "actor": l.actor,
            "agent_name": l.agent_name,
            "tool_action": l.tool_action,
            "affected_record": l.affected_record,
            "policy_rule": l.policy_rule,
            "confidence": l.confidence,
            "summary": l.summary,
        }
        for l in logs
    ]
