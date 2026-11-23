"""
Duplicate Detection & Suggested Incident Grouping Engine (Module 04).
Group 7:
- Uses embedding cosine similarity (via multilingual-e5-small) when available,
  with calibrated token-overlap fallback.
- Settings: CLUSTER_SIMILARITY_THRESHOLD (0.82) and CLUSTER_WINDOW_DAYS (15).
- Candidate pool includes SENSITIVE_HOLD and DISPUTED_REOPENED cases.
- Excludes records with location_unknown or missing taluka/locality.
- In incident listings: never reveal identity or raw text for identity-masked cases.
- STRICT SAFETY RULE: Incident groups are SUGGESTED and confirmable by a human officer.
- ZERO SILENT MERGES: Every grievance maintains its own unique reference number, status history,
  and citizen notifications.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from app.db.models import Grievance, Incident, IncidentMember, AuditLog
from app.nlp.retrieval import tokenize
from app.core.config import settings

def calculate_token_similarity(text1: str, text2: str) -> float:
    """Calculates Jaccard similarity between two complaint token sets."""
    s1 = set(tokenize(text1))
    s2 = set(tokenize(text2))
    if not s1 or not s2:
        return 0.0
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    return round(len(intersection) / len(union), 3)


def compute_pair_similarity(text1: str, text2: str) -> float:
    """
    Computes semantic similarity using multilingual-e5-small if ready,
    otherwise falls back to token overlap.
    """
    try:
        from app.nlp.embedding_retrieval import embedding_engine
        if embedding_engine and embedding_engine.is_ready and embedding_engine.model:
            import numpy as np
            embs = embedding_engine.model.encode(
                [f"query: {text1}", f"query: {text2}"],
                convert_to_numpy=True
            )
            n1 = np.linalg.norm(embs[0])
            n2 = np.linalg.norm(embs[1])
            if n1 > 0 and n2 > 0:
                cos_sim = float(np.dot(embs[0], embs[1]) / (n1 * n2))
                return round(max(0.0, min(1.0, cos_sim)), 3)
    except Exception:
        pass

    # Token overlap fallback
    tok_sim = calculate_token_similarity(text1, text2)
    # Map token Jaccard (where 0.25+ indicates high civic similarity) to 0-1 scale
    return round(min(tok_sim * 2.5, 1.0), 3)


class DuplicateClusteringEngine:
    def check_and_cluster(
        self,
        grievance: Grievance,
        db: Session,
        similarity_threshold: Optional[float] = None,
        window_days: Optional[int] = None,
    ) -> Optional[Incident]:
        """
        Scans recent open grievances in the same taluka and locality.
        If a similar issue exists, attaches as a SUGGESTED group member.
        """
        if similarity_threshold is None:
            similarity_threshold = settings.CLUSTER_SIMILARITY_THRESHOLD
        if window_days is None:
            window_days = settings.CLUSTER_WINDOW_DAYS

        # Skip clustering if location is missing or unknown
        if not grievance.taluka or grievance.taluka in ("Unknown", "None", ""):
            return None
        if not grievance.locality or grievance.locality in ("Unknown", "None", ""):
            return None

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)

        # 1. Candidate pool: Same category, taluka, locality within time window
        # Includes SENSITIVE_HOLD and DISPUTED_REOPENED cases
        candidates = (
            db.query(Grievance)
            .filter(
                Grievance.id != grievance.id,
                Grievance.category_id == grievance.category_id,
                Grievance.taluka == grievance.taluka,
                Grievance.locality == grievance.locality,
                Grievance.created_at >= cutoff,
                Grievance.status.in_([
                    "RECEIVED", "TRIAGED", "IN_PROGRESS", "SENSITIVE_HOLD", "DISPUTED_REOPENED"
                ])
            )
            .all()
        )

        for candidate in candidates:
            sim = compute_pair_similarity(grievance.raw_text, candidate.raw_text)
            # Accept if similarity meets threshold (or legacy token threshold 0.25)
            if sim >= similarity_threshold or sim >= 0.65:
                # Found a matching candidate!
                # Check if an Incident cluster already exists for the candidate
                incident = (
                    db.query(Incident)
                    .filter(
                        Incident.category_id == grievance.category_id,
                        Incident.taluka == grievance.taluka,
                        Incident.locality == grievance.locality,
                        Incident.status.in_(["SUGGESTED", "CONFIRMED"])
                    )
                    .first()
                )

                if not incident:
                    # Create new suggested incident
                    incident = Incident(
                        root_grievance_id=candidate.reference_no,
                        category_id=grievance.category_id or "C001",
                        taluka=grievance.taluka,
                        locality=grievance.locality,
                        status="SUGGESTED",
                        created_at=now
                    )
                    db.add(incident)
                    db.flush()

                    # Add root candidate as member
                    db.add(IncidentMember(
                        incident_id=incident.id,
                        grievance_id=candidate.reference_no,
                        similarity=1.0,
                        status="SUGGESTED"
                    ))

                # Add new grievance as member
                existing_member = (
                    db.query(IncidentMember)
                    .filter(
                        IncidentMember.incident_id == incident.id,
                        IncidentMember.grievance_id == grievance.reference_no
                    )
                    .first()
                )
                if not existing_member:
                    member = IncidentMember(
                        incident_id=incident.id,
                        grievance_id=grievance.reference_no,
                        similarity=sim,
                        status="SUGGESTED"
                    )
                    db.add(member)

                    # Audit log for suggestion
                    db.add(AuditLog(
                        actor="AI_DUPLICATE_ENGINE",
                        tool_action="suggest_incident_cluster",
                        affected_record=grievance.reference_no,
                        confidence=sim,
                        policy_rule="RULE_SUGGESTED_DUPLICATE_GROUP",
                        summary=(
                            f"Grievance {grievance.reference_no} suggested for Incident #{incident.id} "
                            f"(matches {candidate.reference_no} in {grievance.locality}, similarity: {sim:.2f}). "
                            f"Preserved independent ticket record."
                        )
                    ))
                    db.commit()
                return incident

        return None

    def confirm_member(self, member_id: int, officer_name: str, db: Session) -> bool:
        """Officer confirms duplicate grouping."""
        member = db.query(IncidentMember).filter(IncidentMember.id == member_id).first()
        if member:
            member.status = "CONFIRMED"
            member.confirmed_by = officer_name
            member.confirmed_at = datetime.now(timezone.utc)
            db.commit()
            return True
        return False

    def reject_member(self, member_id: int, officer_name: str, db: Session) -> bool:
        """Officer rejects duplicate grouping."""
        member = db.query(IncidentMember).filter(IncidentMember.id == member_id).first()
        if member:
            member.status = "REJECTED"
            member.confirmed_by = officer_name
            member.confirmed_at = datetime.now(timezone.utc)
            db.commit()
            return True
        return False


clustering_engine = DuplicateClusteringEngine()
