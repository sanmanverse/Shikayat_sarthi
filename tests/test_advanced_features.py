from datetime import datetime, timezone, timedelta
import pytest
from app.db.session import init_db, SessionLocal
from app.db.models import Grievance, Incident, IncidentMember, SLAAlert
from app.api.endpoints import (
    intake_confirm_draft,
    list_incidents,
    confirm_duplicate_member,
    reject_duplicate_member,
    get_sla_alerts,
    trigger_sla_scan,
    get_bottlenecks,
    ConfirmDraftRequest,
    MemberActionRequest
)
from app.core.observability import redact_pii
from app.services.sla_predictor import sla_predictor

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

def test_duplicate_incident_grouping_without_silent_merge(db):
    """
    Scenario 3: Two similar complaints in same taluka/locality form a SUGGESTED group.
    Neither complaint is dropped or merged away.
    """
    # Complaint 1
    draft1 = {
        "category_id": "C003",
        "description": "Drinking water pipeline burst on main road causing severe flooding",
        "taluka": "Degloor",
        "locality": "Bazar Chowk",
        "urgency": "HIGH",
        "is_ready_for_confirmation": True
    }
    c1 = intake_confirm_draft(ConfirmDraftRequest(draft=draft1, citizen_id="citizen_deg_01"), db)
    ref1 = c1["reference_no"]

    # Complaint 2 (Similar issue in same locality)
    draft2 = {
        "category_id": "C003",
        "description": "Drinking water pipeline leak and flooding near shops",
        "taluka": "Degloor",
        "locality": "Bazar Chowk",
        "urgency": "HIGH",
        "is_ready_for_confirmation": True
    }
    c2 = intake_confirm_draft(ConfirmDraftRequest(draft=draft2, citizen_id="citizen_deg_02"), db)
    ref2 = c2["reference_no"]

    # Assert individual identities preserved
    assert ref1 != ref2
    g1 = db.query(Grievance).filter(Grievance.reference_no == ref1).first()
    g2 = db.query(Grievance).filter(Grievance.reference_no == ref2).first()
    assert g1 is not None and g2 is not None

    # Check that a suggested Incident was formed
    incidents = list_incidents(db=db)
    target_inc = next((i for i in incidents if i["taluka"] == "Degloor" and i["locality"] == "Bazar Chowk"), None)
    assert target_inc is not None
    assert target_inc["status"] == "SUGGESTED"
    assert len(target_inc["members"]) >= 2

    # Human officer confirmation gate
    target_member = next(m for m in target_inc["members"] if m["grievance_id"] == ref2)
    conf_res = confirm_duplicate_member(
        member_id=target_member["member_id"],
        req=MemberActionRequest(officer_name="Officer Patil"),
        db=db
    )
    assert conf_res["status"] == "CONFIRMED"

def test_predictive_sla_breach_alert(db):
    """
    Predictive Intelligence (Module 05): Cases likely to breach SLA trigger
    pre-emptive alert before deadline.
    """
    import uuid
    ref_test = f"GRV-SLA-{uuid.uuid4().hex[:6].upper()}"
    # Create grievance with created_at set to 6 days ago for a 7-day SLA
    past_date = datetime.now(timezone.utc) - timedelta(days=6)
    g = Grievance(
        reference_no=ref_test,
        citizen_id="citizen_alert_01",
        raw_text="Contaminated muddy drinking water issue unresolved",
        category_id="C005",
        department_id="D002",
        taluka="Nanded",
        locality="Taroda Naka",
        status="IN_PROGRESS",
        created_at=past_date,
        due_date=past_date + timedelta(days=2)
    )
    db.add(g)
    db.commit()

    # Run SLA predictor evaluation
    alert = sla_predictor.evaluate_case_sla(g, db)
    assert alert is not None
    assert alert.breach_probability >= 0.65
    assert "pre-emptive" in alert.reason.lower() or "sla" in alert.reason.lower()

    # Check alert endpoint
    active_alerts = get_sla_alerts(status="ACTIVE", db=db)
    assert any(a["grievance_id"] == ref_test for a in active_alerts)

def test_department_bottleneck_analytics(db):
    """Admin bottleneck overview and chronic recurrence hotspots."""
    bottlenecks = get_bottlenecks(db)
    assert len(bottlenecks) == 20
    assert "active_backlog" in bottlenecks[0]
    assert "recurring_hotspots" in bottlenecks[0]

def test_pii_redaction_layer():
    """Verify personal information is stripped from public observability traces."""
    raw_payload = {
        "citizen_name": "Ramesh Patil",
        "phone": "Citizen contact is +91-9822334455 please call",
        "email": "Write to citizen.ramesh@gmail.com",
        "user_id": "citizen_secret_1234",
        "aadhaar": "Aadhaar: 1234 5678 9012 provided"
    }
    sanitized = redact_pii(raw_payload, is_identity_masked=True)
    assert "[REDACTED_PHONE]" in sanitized["phone"]
    assert "[REDACTED_EMAIL]" in sanitized["email"]
    assert "[REDACTED_ID]" in sanitized["aadhaar"]
    assert sanitized["user_id"] == "ANONYMOUS_PROTECTED"
