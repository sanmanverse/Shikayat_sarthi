from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Category(Base):
    __tablename__ = "categories"

    category_id = Column(String(50), primary_key=True)  # C001 to C030
    category = Column(String(100), nullable=False)
    subcategory = Column(String(150), nullable=False)
    responsible_department_id = Column(String(50), ForeignKey("departments.department_id"), nullable=False)
    responsible_department = Column(String(150), nullable=True)
    default_assignee = Column(String(150), nullable=False)
    sla_days = Column(Integer, nullable=False, default=7)
    sensitive = Column(Boolean, nullable=False, default=False)
    identity_hiding = Column(Boolean, nullable=False, default=False)
    default_priority = Column(String(20), nullable=False, default="MEDIUM")
    auto_resolution_allowed = Column(Boolean, nullable=False, default=False)
    policy_note = Column(Text, nullable=True)

    department = relationship("Department", back_populates="categories")


class Department(Base):
    __tablename__ = "departments"

    department_id = Column(String(50), primary_key=True)  # D001 to D020
    department_name = Column(String(150), nullable=False)
    service_scope = Column(Text, nullable=False)
    jurisdiction_type = Column(String(50), nullable=False)
    jurisdiction_mapping = Column(Text, nullable=False)
    contact_email_demo = Column(String(100), nullable=False)
    contact_channel = Column(String(50), nullable=False)
    mock_webhook_id = Column(String(100), nullable=False)
    is_real_contact = Column(Boolean, nullable=False, default=False)
    notes = Column(Text, nullable=True)

    categories = relationship("Category", back_populates="department")


class HistoryGrievance(Base):
    __tablename__ = "mock_grievance_history"

    grievance_id = Column(String(50), primary_key=True)
    complaint_text = Column(Text, nullable=False)
    category_id = Column(String(50), ForeignKey("categories.category_id"), nullable=False)
    category = Column(String(100), nullable=False)
    subcategory = Column(String(150), nullable=False)
    department_id = Column(String(50), ForeignKey("departments.department_id"), nullable=False)
    broad_location_taluka = Column(String(100), nullable=False)
    broad_location_locality = Column(String(100), nullable=False)
    submission_date = Column(String(50), nullable=False)
    resolution_date = Column(String(50), nullable=True)
    status = Column(String(50), nullable=False)
    resolution_text = Column(Text, nullable=True)
    citizen_feedback = Column(Text, nullable=True)
    priority = Column(String(20), nullable=False)
    sla_days = Column(Integer, nullable=False)
    sensitive = Column(Boolean, nullable=False)
    identity_hiding = Column(Boolean, nullable=False)
    resolution_actor = Column(String(50), nullable=True)
    ai_auto_closed = Column(Boolean, nullable=False, default=False)
    data_origin = Column(String(50), nullable=True, default="SYNTHETIC_DEMO")


class Grievance(Base):
    __tablename__ = "grievances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reference_no = Column(String(50), unique=True, index=True, nullable=False)
    citizen_id = Column(String(50), nullable=False)
    raw_text = Column(Text, nullable=False)
    normalized_text = Column(Text, nullable=True)  # English-normalized text for multilingual RAG
    language = Column(String(20), nullable=False, default="English")
    
    # Classification fields
    category_id = Column(String(50), ForeignKey("categories.category_id"), nullable=True)
    department_id = Column(String(50), ForeignKey("departments.department_id"), nullable=True)
    taluka = Column(String(100), nullable=True)
    locality = Column(String(100), nullable=True)
    urgency_level = Column(String(20), default="MEDIUM")
    priority = Column(String(20), default="MEDIUM")
    status = Column(String(50), default="RECEIVED")  # RECEIVED, TRIAGED, IN_PROGRESS, RESOLVED, DISPUTED, SENSITIVE_HOLD
    
    # Policy and Agent outputs
    confidence = Column(Float, default=0.0)
    policy_verdict = Column(String(50), nullable=True)  # AUTO_HANDLE, HUMAN_REVIEW, SENSITIVE_HOLD
    policy_rule = Column(String(100), nullable=True)
    identity_mask = Column(Boolean, default=False)
    sentiment = Column(String(50), default="NEUTRAL")
    summary = Column(Text, nullable=True)
    ai_auto_closed = Column(Boolean, default=False)
    assigned_to = Column(String(150), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    due_date = Column(DateTime, nullable=True)

    # Relationships
    status_history = relationship("StatusHistory", back_populates="grievance")
    work_orders = relationship("WorkOrder", back_populates="grievance")
    notifications = relationship("Notification", back_populates="grievance")


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    root_grievance_id = Column(String(50), nullable=False)
    category_id = Column(String(50), nullable=False)
    taluka = Column(String(100), nullable=False)
    locality = Column(String(100), nullable=False)
    status = Column(String(50), default="SUGGESTED")  # SUGGESTED, CONFIRMED, REJECTED
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    members = relationship("IncidentMember", back_populates="incident")


class IncidentMember(Base):
    __tablename__ = "incident_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    grievance_id = Column(String(50), nullable=False)
    similarity = Column(Float, nullable=False)
    status = Column(String(50), default="SUGGESTED")  # SUGGESTED, CONFIRMED, REJECTED
    confirmed_by = Column(String(100), nullable=True)
    confirmed_at = Column(DateTime, nullable=True)

    incident = relationship("Incident", back_populates="members")


class StatusHistory(Base):
    __tablename__ = "status_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    grievance_id = Column(String(50), ForeignKey("grievances.reference_no"), nullable=False)
    from_status = Column(String(50), nullable=True)
    to_status = Column(String(50), nullable=False)
    actor = Column(String(100), nullable=False)  # AI_SYSTEM or OFFICER name
    note = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    grievance = relationship("Grievance", back_populates="status_history")


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    grievance_id = Column(String(50), ForeignKey("grievances.reference_no"), nullable=False)
    department_id = Column(String(50), nullable=False)
    assignee = Column(String(150), nullable=False)
    mock_webhook_id = Column(String(100), nullable=False)
    payload = Column(Text, nullable=False)
    status = Column(String(50), default="DISPATCHED")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    grievance = relationship("Grievance", back_populates="work_orders")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    grievance_id = Column(String(50), ForeignKey("grievances.reference_no"), nullable=False)
    channel = Column(String(50), nullable=False)  # EMAIL_SIMULATED, WHATSAPP_SIMULATED
    recipient = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String(50), default="SENT")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    grievance = relationship("Grievance", back_populates="notifications")


class SLAAlert(Base):
    __tablename__ = "sla_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    grievance_id = Column(String(50), nullable=False)
    breach_probability = Column(Float, nullable=False)
    reason = Column(Text, nullable=False)
    due_date = Column(DateTime, nullable=False)
    status = Column(String(50), default="ACTIVE")
    acknowledged_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    actor = Column(String(100), nullable=False)  # AI_SYSTEM or OFFICER username
    agent_name = Column(String(100), nullable=True)
    model_name = Column(String(100), nullable=True)
    tool_action = Column(String(100), nullable=False)
    affected_record = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=True)
    policy_rule = Column(String(100), nullable=True)
    summary = Column(Text, nullable=False)
    trace_id = Column(String(100), nullable=True)
