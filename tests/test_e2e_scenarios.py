"""
End-to-End Tests for 3 Mandatory Demo Scenarios.
Group 13:
Scenario A: Routine Auto-Handled Grievance
  - Passes intake, RAG retrieval, policy gate (RULE_06_AUTO_HANDLE_PERMITTED)
  - Work order dispatched with mock webhook
  - Simulated notifications created (status SIMULATED)
  - Status becomes IN_PROGRESS
  - Final closure only after citizen confirmation

Scenario B: Sensitive Case Safety Lock
  - Intercepted by Safety Guard / Category Master
  - Policy Gate enforces SENSITIVE_HOLD (RULE_01_SENSITIVE_CATEGORY_HOLD)
  - Identity masked; no raw complaint text in work order or public trace
  - Closure strictly blocked for AI or unauthenticated citizen
  - Officer escalation webhook dispatched

Scenario C: Duplicate Detection & Incident Grouping
  - Multiple complaints for the same issue in the same locality
  - Duplicate engine links them into a SUGGESTED Incident cluster
  - Zero silent merges: both maintain separate reference numbers and history
  - Officer can confirm or reject the cluster membership
"""
import pytest
from app.api.endpoints import (
    intake_chat_turn, intake_confirm_draft, update_grievance_status,
    confirm_resolution, list_incidents, confirm_duplicate_member,
    ChatTurnRequest, ConfirmDraftRequest, StatusUpdateRequest
)
from app.db.models import Grievance, WorkOrder, Notification, Incident, IncidentMember

def test_scenario_a_routine_auto_handle(db):
    """Scenario A: Routine grievance processed, dispatched, resolved, citizen-confirmed."""
    # 1. Intake turn
    req = ChatTurnRequest(
        conversation_history=[],
        message="Streetlight not working on pole 12 in Vazirabad Nanded since two days",
        citizen_id="citizen_demo_scenario_a"
    )
    t = intake_chat_turn(req, db)
    assert t["draft"]["is_ready_for_confirmation"] is True
    assert t["draft"]["category_id"] == "C009"

    # 2. Citizen confirms draft
    confirm_req = ConfirmDraftRequest(
        draft=t["draft"],
        citizen_id="citizen_demo_scenario_a"
    )
    res = intake_confirm_draft(confirm_req, db)

    ref_no = res["reference_no"]
    assert res["status"] == "IN_PROGRESS"
    assert res["decision"]["verdict"] == "AUTO_HANDLE"
    assert res["decision"]["rule_id"] == "RULE_06_AUTO_HANDLE_PERMITTED"

    # 3. Check work order created
    wo = db.query(WorkOrder).filter(WorkOrder.grievance_id == ref_no).first()
    assert wo is not None
    assert wo.status == "DISPATCHED"

    # 4. Check simulated notifications
    notifs = db.query(Notification).filter(Notification.grievance_id == ref_no).all()
    assert len(notifs) >= 2
    for n in notifs:
        assert n.status == "SIMULATED"
        assert "@example.invalid" in n.recipient or "+91-XXXXXX" in n.recipient

    # 5. Officer marks resolved
    status_req = StatusUpdateRequest(
        status="RESOLVED",
        actor="NODAL_OFFICER",
        note="Replaced LED bulb on pole 12."
    )
    update_res = update_grievance_status(ref_no, status_req, db)
    assert update_res["new_status"] == "RESOLVED"

    # 6. Citizen confirms resolution
    conf_res = confirm_resolution(ref_no, db)
    assert conf_res["status"] == "CITIZEN_SATISFIED"


def test_scenario_b_sensitive_hold_and_closure_protection(db):
    """Scenario B: Sensitive case triggers safety lock, masking, and forbids unauthorized closure."""
    # 1. Sensitive grievance submission
    draft = {
        "category_id": "C017",
        "category_name": "Women and Child Welfare - Women safety / harassment concern",
        "description": "Some boys are eve teasing college girls near bus stand every evening",
        "taluka": "Nanded",
        "locality": "Bus Stand Area",
        "urgency": "CRITICAL",
        "language": "English",
        "is_ready_for_confirmation": True,
        "confidence": 0.95
    }
    confirm_req = ConfirmDraftRequest(
        draft=draft,
        citizen_id="citizen_victim_confidential_01"
    )
    res = intake_confirm_draft(confirm_req, db)
    ref_no = res["reference_no"]

    # Invariants
    assert res["status"] == "SENSITIVE_HOLD"
    assert res["decision"]["verdict"] == "SENSITIVE_HOLD"
    assert res["decision"]["identity_masked"] is True

    # Check work order does NOT contain raw complaint text
    wo = db.query(WorkOrder).filter(WorkOrder.grievance_id == ref_no).first()
    assert wo is not None
    assert "eve teasing" not in wo.payload.lower()

    # Attempting to resolve via AI or citizen must fail (HTTP 403)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        update_grievance_status(
            ref_no,
            StatusUpdateRequest(status="RESOLVED", actor="AI_SYSTEM_AGENT", note="Attempted auto resolve"),
            db
        )
    assert exc_info.value.status_code == 403

    # Attempting citizen confirmation directly must fail
    with pytest.raises(HTTPException) as exc_info2:
        confirm_resolution(ref_no, db)
    assert exc_info2.value.status_code in (403, 409)


def test_scenario_c_duplicate_clustering_and_officer_review(db):
    """Scenario C: Two complaints for same incident grouped without silent merge."""
    # 1. First complaint
    draft1 = {
        "category_id": "C003",
        "category_name": "Water Supply - Pipeline leakage",
        "description": "Drinking water pipeline burst near Shivaji statue, water flooding the road",
        "taluka": "Nanded",
        "locality": "Vazirabad",
        "urgency": "HIGH",
        "language": "English",
        "is_ready_for_confirmation": True,
        "confidence": 0.90
    }
    res1 = intake_confirm_draft(ConfirmDraftRequest(draft=draft1, citizen_id="citizen_one"), db)
    ref1 = res1["reference_no"]

    # 2. Second complaint for same incident
    draft2 = {
        "category_id": "C003",
        "category_name": "Water Supply - Pipeline leakage",
        "description": "Drinking water pipeline leakage and flooding near Shivaji statue",
        "taluka": "Nanded",
        "locality": "Vazirabad",
        "urgency": "HIGH",
        "language": "English",
        "is_ready_for_confirmation": True,
        "confidence": 0.90
    }
    res2 = intake_confirm_draft(ConfirmDraftRequest(draft=draft2, citizen_id="citizen_two"), db)
    ref2 = res2["reference_no"]

    # Invariants: Separate reference numbers
    assert ref1 != ref2

    # Duplicate grouping suggested
    incidents = list_incidents(db)
    assert len(incidents) > 0

    # Locate incident containing ref1 or ref2
    target_inc = next((i for i in incidents if any(m["grievance_id"] == ref1 for m in i["members"])), None)
    assert target_inc is not None
    members = target_inc["members"]
    assert any(m["grievance_id"] == ref2 for m in members)

    # Officer confirms member
    target_member = next(m for m in members if m["grievance_id"] == ref2)
    from app.api.endpoints import MemberActionRequest
    confirm_duplicate_member(
        target_member["member_id"],
        MemberActionRequest(officer_name="Officer Deshmukh"),
        db
    )

    # Verify confirmed in DB
    updated_m = db.query(IncidentMember).filter(IncidentMember.id == target_member["member_id"]).first()
    assert updated_m.status == "CONFIRMED"
    assert updated_m.confirmed_by == "Officer Deshmukh"
