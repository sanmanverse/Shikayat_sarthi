import csv
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.db.models import Base, Category, Department, HistoryGrievance

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(db: Session = None):
    Base.metadata.create_all(bind=engine)
    
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        # 1. Load Departments (D001 - D020)
        if db.query(Department).count() == 0:
            dept_csv = settings.DATA_DIR / "department_directory.csv"
            if dept_csv.exists():
                with open(dept_csv, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dept = Department(
                            department_id=row["department_id"].strip(),
                            department_name=row["department_name"].strip(),
                            service_scope=row["service_scope"].strip(),
                            jurisdiction_type=row["jurisdiction_type"].strip(),
                            jurisdiction_mapping=row["jurisdiction_mapping"].strip(),
                            contact_email_demo=row["contact_email_demo"].strip(),
                            contact_channel=row["contact_channel"].strip(),
                            mock_webhook_id=row["mock_webhook_id"].strip(),
                            is_real_contact=row.get("is_real_contact", "False").lower() in ("true", "1", "yes"),
                            notes=row.get("notes", "").strip(),
                        )
                        db.add(dept)
                db.commit()

        # 2. Load Categories (C001 - C030)
        if db.query(Category).count() == 0:
            cat_csv = settings.DATA_DIR / "grievance_category_master.csv"
            if cat_csv.exists():
                with open(cat_csv, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cat = Category(
                            category_id=row["category_id"].strip(),
                            category=row["category"].strip(),
                            subcategory=row["subcategory"].strip(),
                            responsible_department_id=row["responsible_department_id"].strip(),
                            responsible_department=row.get("responsible_department", "").strip(),
                            default_assignee=row["default_assignee"].strip(),
                            sla_days=int(row["sla_days"]),
                            sensitive=row["sensitive"].lower() in ("true", "1", "yes"),
                            identity_hiding=row["identity_hiding"].lower() in ("true", "1", "yes"),
                            default_priority=row["default_priority"].strip(),
                            auto_resolution_allowed=row["auto_resolution_allowed"].lower() in ("true", "1", "yes"),
                            policy_note=row.get("policy_note", "").strip(),
                        )
                        db.add(cat)
                db.commit()

        # 3. Load 800 History Grievances
        if db.query(HistoryGrievance).count() == 0:
            hist_csv = settings.DATA_DIR / "mock_grievance_history.csv"
            if hist_csv.exists():
                with open(hist_csv, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        hist = HistoryGrievance(
                            grievance_id=row["grievance_id"].strip(),
                            complaint_text=row["complaint_text"].strip(),
                            category_id=row["category_id"].strip(),
                            category=row["category"].strip(),
                            subcategory=row["subcategory"].strip(),
                            department_id=row["department_id"].strip(),
                            broad_location_taluka=row["broad_location_taluka"].strip(),
                            broad_location_locality=row["broad_location_locality"].strip(),
                            submission_date=row["submission_date"].strip(),
                            resolution_date=row["resolution_date"].strip() or None,
                            status=row["status"].strip(),
                            resolution_text=row["resolution_text"].strip() or None,
                            citizen_feedback=row["citizen_feedback"].strip() or None,
                            priority=row["priority"].strip(),
                            sla_days=int(row["sla_days"]),
                            sensitive=row["sensitive"].lower() in ("true", "1", "yes"),
                            identity_hiding=row["identity_hiding"].lower() in ("true", "1", "yes"),
                            resolution_actor=row["resolution_actor"].strip() or None,
                            ai_auto_closed=row["ai_auto_closed"].lower() in ("true", "1", "yes"),
                            data_origin=row.get("data_origin", "SYNTHETIC_DEMO").strip(),
                        )
                        db.add(hist)
                db.commit()
    finally:
        if close_db:
            db.close()
