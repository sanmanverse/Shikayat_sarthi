import pytest
from app.db.session import init_db, SessionLocal
from app.api.endpoints import (
    get_categories,
    get_departments,
    intake_chat_turn,
    intake_confirm_draft,
    get_grievance_details,
    list_grievances,
    update_grievance_status,
    dispute_resolution,
    get_audit_logs,
    ChatTurnRequest,
    ConfirmDraftRequest,
    StatusUpdateRequest,
    DisputeRequest,
)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()

@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def test_master_data_endpoints(db):
    # Verify 30 Categories from Official Master
    cats = get_categories(db)
    assert len(cats) == 30
    assert any(c["category_id"] == "C017" and c["sensitive"] is True for c in cats)
    assert any(c["category_id"] == "C018" and c["sensitive"] is True for c in cats)
    assert any(c["category_id"] == "C010" and c["sensitive"] is True for c in cats)
    assert any(c["category_id"] == "C009" and c["auto_resolution_allowed"] is True for c in cats)
    assert any(c["category_id"] == "C001" and c["auto_resolution_allowed"] is True for c in cats)

    # Verify 20 Departments from Official Directory
    depts = get_departments(db)
    assert len(depts) == 20
    assert any(d["department_id"] == "D001" for d in depts)
    assert any(d["department_id"] == "D017" for d in depts)

def test_conversational_intake_and_auto_handle(db):
    # Turn 1: Citizen complains without location
    req1 = ChatTurnRequest(
        conversation_history=[],
        message="Streetlight broken pole lamp bulb dark",
        citizen_id="citizen_demo_01"
    )
    t1 = intake_chat_turn(req1, db)
    assert t1["action_required"] == "FOLLOWUP"
    assert "location" in t1["draft"]["missing_details"]

    # Turn 2: Citizen supplies location in Nanded
    req2 = ChatTurnRequest(
        conversation_history=[
            {"role": "user", "content": "Streetlight broken pole lamp bulb dark"},
            {"role": "assistant", "content": t1["assistant_reply"]}
        ],
        message="It is in Vazirabad locality in Nanded",
        citizen_id="citizen_demo_01"
    )
    t2 = intake_chat_turn(req2, db)
    assert t2["action_required"] == "CONFIRMATION"
    assert t2["draft"]["is_ready_for_confirmation"] is True

    # Citizen Confirms Draft -> Triggers End-to-End Orchestrator
    confirm_req = ConfirmDraftRequest(
        draft=t2["draft"],
        citizen_id="citizen_marathi_01"
    )
    res_data = intake_confirm_draft(confirm_req, db)
    
    # Assert Routine Auto-Handling & Evidence
    assert res_data["decision"]["verdict"] == "AUTO_HANDLE"
    assert res_data["decision"]["rule_id"] == "RULE_06_AUTO_HANDLE_PERMITTED"
    assert res_data["status"] == "IN_PROGRESS"
    assert len(res_data["retrieval_evidence"]["similar_precedents"]) > 0
    assert len(res_data["actions_executed"]) >= 2

def test_sensitive_case_safety_lock(db):
    # Sensitive complaint (women safety concern in Degloor)
    draft = {
        "category_id": "C017",
        "category_name": "Women and Child Welfare - Women safety / harassment concern",
        "description": "Boys tease the girls near the bus stand every evening in Degloor",
        "taluka": "Degloor",
        "locality": "Bus Stand Area",
        "urgency": "CRITICAL",
        "language": "English",
        "is_ready_for_confirmation": True
    }

    confirm_req = ConfirmDraftRequest(
        draft=draft,
        citizen_id="citizen_confidential_99"
    )
    res_data = intake_confirm_draft(confirm_req, db)

    # Verify SENSITIVE_HOLD and Safety Boundaries
    assert res_data["decision"]["verdict"] == "SENSITIVE_HOLD"
    assert res_data["decision"]["rule_id"] == "RULE_01_SENSITIVE_CATEGORY_HOLD"
    assert res_data["decision"]["can_auto_resolve"] is False
    assert res_data["decision"]["identity_masked"] is True
    assert res_data["status"] == "SENSITIVE_HOLD"

    ref_no = res_data["reference_no"]

    # Verify queue listing masks citizen identity
    queue_list = list_grievances(db=db)
    sensitive_item = next(item for item in queue_list if item["reference_no"] == ref_no)
    assert sensitive_item["citizen_id"] == "ANONYMOUS_PROTECTED"
    assert sensitive_item["identity_mask"] is True

def test_child_protection_safety_lock(db):
    # Sensitive child safety complaint
    draft = {
        "category_id": "C018",
        "category_name": "Women and Child Welfare - Child safety / protection concern",
        "description": "A small child is working at a tea stall and is being beaten by the owner",
        "taluka": "Naigaon",
        "locality": "Station Naka",
        "urgency": "CRITICAL",
        "language": "English",
        "is_ready_for_confirmation": True
    }

    confirm_req = ConfirmDraftRequest(
        draft=draft,
        citizen_id="citizen_child_witness"
    )
    res_data = intake_confirm_draft(confirm_req, db)

    # Must be SENSITIVE_HOLD
    assert res_data["decision"]["verdict"] == "SENSITIVE_HOLD"
    assert res_data["decision"]["can_auto_resolve"] is False
    assert res_data["status"] == "SENSITIVE_HOLD"

def test_citizen_dispute_loop(db):
    # Create routine case
    draft = {
        "category_id": "C001",
        "description": "Minor potholes on Station Road Nanded",
        "taluka": "Nanded",
        "locality": "Station Road",
        "urgency": "MEDIUM",
        "is_ready_for_confirmation": True
    }
    confirm_res = intake_confirm_draft(ConfirmDraftRequest(draft=draft, citizen_id="citizen_45"), db)
    ref_no = confirm_res["reference_no"]

    # Mark as resolved
    update_grievance_status(
        ref_no=ref_no,
        req=StatusUpdateRequest(status="RESOLVED", actor="NODAL_OFFICER", note="Road patch completed."),
        db=db
    )

    # Citizen disputes resolution
    disp_res = dispute_resolution(
        ref_no=ref_no,
        req=DisputeRequest(dispute_reason="Potholes were only filled with loose gravel and washed away in rain!"),
        db=db
    )
    assert disp_res["status"] == "DISPUTED_REOPENED"

    # Check status history
    details = get_grievance_details(ref_no=ref_no, db=db)
    assert details["grievance"]["status"] == "DISPUTED_REOPENED"
    assert details["grievance"]["priority"] == "HIGH"
    
    actors = [sh["actor"] for sh in details["status_history"]]
    assert "CITIZEN_DISPUTE" in actors

def test_audit_logs_recorded(db):
    logs = get_audit_logs(limit=50, db=db)
    assert len(logs) > 0
    actions = [l["tool_action"] for l in logs]
    assert "confirm_and_submit" in actions
    assert "execute_workflow" in actions
