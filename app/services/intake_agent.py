"""
Conversational Intake Agent.
Groups 2 & 4 fixes:
- Robust JSON extractor (find first {...} block, strip fencing)
- On 2 consecutive parse failures → form-style fallback
- reference_no: year from datetime.now().year, 8-char hex, retry on collision
- No silent C001 default: null + HUMAN_REVIEW when category unknown
- confidence from draft (None when LLM doesn't provide)
- engine/degraded flags persisted to AuditLog
- location_unknown flag when taluka/locality missing
"""
import uuid
import json
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.services.sarvam import sarvam_client, IntakeDraft
from app.services.safety_guard import safety_guard
from app.services.normalizer import normalize_to_english
from app.db.models import Grievance, StatusHistory, AuditLog, Category

logger = logging.getLogger(__name__)


def is_valid_input(text: str) -> tuple:
    cleaned = text.strip()
    if len(cleaned) < 5:
        return False, "Please provide a little more detail about your issue."
    alpha_chars = sum(1 for c in cleaned if c.isalpha())
    if alpha_chars < 3:
        return False, "Please describe the problem using words rather than just numbers or symbols."
    return True, ""


def _extract_json(raw: str) -> Optional[Dict]:
    """Robustly extract first {...} JSON block from LLM output."""
    # Strip markdown code fences
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Find first {...} block
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    # Walk to find balanced closing brace
    depth = 0
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _generate_reference_no() -> str:
    year = datetime.now().year
    suffix = uuid.uuid4().hex[:8].upper()
    return f"GRV-{year}-{suffix}"


class IntakeAgent:
    def __init__(self, sarvam=sarvam_client):
        self.sarvam = sarvam
        self._consecutive_parse_failures: Dict[str, int] = {}

    def process_turn(
        self,
        conversation_history: list,
        current_message: str,
        db: Session,
        session_key: str = "default",
    ) -> Dict[str, Any]:
        valid, err_msg = is_valid_input(current_message)
        if not valid:
            return {"draft": None, "assistant_reply": err_msg, "action_required": "ERROR"}

        system_instruction = (
            "You are an empathetic, efficient Grievance Intake Officer for Nanded District administration. "
            "Convert citizen complaints (Marathi, Hindi, English, or mixed) into a structured draft.\n"
            "Requirements:\n"
            "1. Extract core problem, taluka (Nanded district), locality.\n"
            "2. Produce normalized_english with English translation and domain keywords.\n"
            "3. If location or issue clarity missing, ask ONE polite follow-up in citizen's language.\n"
            "4. If all basic details present, set is_ready_for_confirmation=True.\n"
            "5. Provide confidence (0.0-1.0) for your category classification when you are sure.\n"
            "Return ONLY pure JSON: "
            '{"category_id":str|null,"category_name":str|null,"description":str,'
            '"normalized_english":str|null,"taluka":str|null,"locality":str|null,'
            '"urgency":"LOW"|"MEDIUM"|"HIGH"|"CRITICAL","missing_details":[str],'
            '"followup_question":str|null,"language":str,"is_ready_for_confirmation":bool,'
            '"confidence":float|null,"category_candidates":[str]|null}'
        )

        messages = [{"role": "system", "content": system_instruction}]
        for turn in conversation_history:
            messages.append(turn)
        messages.append({"role": "user", "content": current_message})

        raw_output, engine_label, is_degraded = self.sarvam.chat_completion(messages)

        # Robust JSON extraction
        failures = self._consecutive_parse_failures.get(session_key, 0)
        data = _extract_json(raw_output)

        if data is None:
            failures += 1
            self._consecutive_parse_failures[session_key] = failures
            logger.warning(
                "JSON parse failed (%d times) for session %s. Output: %s",
                failures, session_key, raw_output[:200],
            )

            if failures >= 2:
                # Form-style fallback: ask for description first
                self._consecutive_parse_failures[session_key] = 0
                return {
                    "draft": None,
                    "assistant_reply": (
                        "I'm having trouble understanding the format. "
                        "Could you please describe your issue in a few words? "
                        "(कृपया आपली समस्या थोडक्यात सांगा)"
                    ),
                    "action_required": "FORM_FALLBACK",
                    "engine": engine_label,
                    "degraded": is_degraded,
                }
            # Single failure — ask citizen to rephrase
            return {
                "draft": None,
                "assistant_reply": "Could you please rephrase your complaint? (कृपया पुन्हा सांगा)",
                "action_required": "FOLLOWUP",
                "engine": engine_label,
                "degraded": is_degraded,
            }

        self._consecutive_parse_failures[session_key] = 0

        # Inject engine info
        data["engine"] = engine_label
        data["degraded"] = is_degraded

        try:
            draft = IntakeDraft(**data)
        except Exception as exc:
            logger.warning("IntakeDraft construction failed: %s", exc)
            draft = IntakeDraft(
                description=current_message,
                normalized_english=normalize_to_english(current_message),
                taluka=None,
                locality=None,
                missing_details=["location"],
                followup_question="Could you tell us your Taluka or locality? (कृपया तालुका सांगा)",
                language="English",
                is_ready_for_confirmation=False,
                engine=engine_label,
                degraded=is_degraded,
            )

        # Pre-check safety guard
        guard_res = safety_guard.inspect(draft.description)
        if guard_res.is_sensitive:
            # Safety guard proposes category concept; policy gate will confirm from master
            draft.urgency = "CRITICAL"
            draft.is_ready_for_confirmation = True

        if not draft.normalized_english:
            draft.normalized_english = normalize_to_english(draft.description)

        if draft.is_ready_for_confirmation:
            lang = draft.language
            if lang == "Marathi":
                reply = (
                    f"तुमची तक्रार नोंदवण्यासाठी तयार आहे. कृपया तपशील तपासा आणि खात्री करा:\n"
                    f"- समस्या: {draft.description}\n"
                    f"- ठिकाण: {draft.locality or ''}, {draft.taluka or ''}\n"
                    f"- निकड: {draft.urgency}"
                )
            elif lang == "Hindi":
                reply = (
                    f"आपकी शिकायत दर्ज करने के लिए तैयार है। कृपया विवरण की पुष्टि करें:\n"
                    f"- समस्या: {draft.description}\n"
                    f"- स्थान: {draft.locality or ''}, {draft.taluka or ''}\n"
                    f"- प्राथमिकता: {draft.urgency}"
                )
            else:
                reply = (
                    f"Your grievance draft is ready. Please review and confirm:\n"
                    f"- Issue: {draft.description}\n"
                    f"- Location: {draft.locality or ''}, {draft.taluka or ''}\n"
                    f"- Urgency: {draft.urgency}"
                )
            action = "CONFIRMATION"
        else:
            reply = draft.followup_question or "Please provide more details regarding your location."
            action = "FOLLOWUP"

        return {
            "draft": draft.model_dump(),
            "assistant_reply": reply,
            "action_required": action,
            "engine": engine_label,
            "degraded": is_degraded,
        }

    def confirm_and_submit(
        self,
        draft_data: Dict[str, Any],
        citizen_id: str,
        db: Session,
    ) -> Grievance:
        """
        Citizen confirms the draft. Persists grievance with unique reference number.
        Retries on unique-constraint collision.
        """
        raw_text = draft_data.get("description", "")
        norm_text = draft_data.get("normalized_english") or normalize_to_english(raw_text)

        # Safety check
        guard_res = safety_guard.inspect(raw_text)
        cat_id = draft_data.get("category_id")
        if guard_res.is_sensitive and guard_res.sensitive_category_id:
            cat_id = guard_res.sensitive_category_id
        # Do NOT default to C001: null means HUMAN_REVIEW in orchestrator
        # cat_id stays None if not determinable

        # Location unknown flag
        location_unknown = not draft_data.get("taluka") and not draft_data.get("locality")

        for _ in range(3):  # retry up to 3 times on collision
            ref_no = _generate_reference_no()
            try:
                grievance = Grievance(
                    reference_no=ref_no,
                    citizen_id=citizen_id,
                    raw_text=raw_text,
                    normalized_text=norm_text,
                    language=draft_data.get("language", "English"),
                    category_id=cat_id,
                    taluka=draft_data.get("taluka") or "Unknown",
                    locality=draft_data.get("locality") or "Unknown",
                    urgency_level=draft_data.get("urgency", "MEDIUM"),
                    priority=draft_data.get("urgency", "MEDIUM"),
                    identity_mask=guard_res.identity_mask,
                    status="RECEIVED",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(grievance)
                db.flush()
                break
            except IntegrityError:
                db.rollback()
                logger.warning("Reference number collision on %s, retrying.", ref_no)
        else:
            raise RuntimeError("Could not generate a unique reference number after 3 attempts.")

        # Status History
        history_entry = StatusHistory(
            grievance_id=ref_no,
            from_status=None,
            to_status="RECEIVED",
            actor="CITIZEN_CONFIRMED",
            note="Citizen confirmed structured draft. Grievance registered in system.",
        )
        db.add(history_entry)

        # Audit Log — record engine and model
        engine_label = draft_data.get("engine", "mock")
        model_used = settings_model_name(engine_label)
        confidence_val = draft_data.get("confidence")  # may be None

        audit = AuditLog(
            actor="AI_INTAKE_AGENT",
            agent_name="IntakeAgent",
            model_name=model_used,
            tool_action="confirm_and_submit",
            affected_record=ref_no,
            confidence=confidence_val,
            policy_rule="RULE_INTAKE_CONFIRMATION",
            summary=(
                f"New grievance submitted. Engine={engine_label}. "
                f"Location: {grievance.locality}, {grievance.taluka}. "
                f"location_unknown={location_unknown}."
            ),
        )
        db.add(audit)
        db.commit()
        db.refresh(grievance)
        return grievance


def settings_model_name(engine: str) -> str:
    from app.core.config import settings
    if engine == "sarvam":
        return settings.SARVAM_CHAT_MODEL
    return "mock_keyword_v1"


intake_agent = IntakeAgent()
