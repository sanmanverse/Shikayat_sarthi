"""
Langfuse Observability & Public Trace Manager with PII Redaction Layer.
Groups 1 requirements:
- Langfuse Python SDK v3+ (OpenTelemetry based): start_as_current_observation(trace_context=..., as_type="span")
- Salted hash for user_id (never raw citizen_id)
- Null trace link when Langfuse not configured
- Full PII redaction: Aadhaar first, then phone, email, names, addresses
- Nested span structure: Intake -> Safety -> RAG -> Policy -> Action
- flush() after each pipeline run; shutdown() on app shutdown
"""
import re
import hashlib
import logging
import warnings
from typing import Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII Redaction – ORDER MATTERS: Aadhaar (12-digit) BEFORE phone regex
# ---------------------------------------------------------------------------
AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")
PHONE_PATTERN = re.compile(r"(\+?91[\-\s]?)?[6-9]\d{9}")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
NAME_AFTER_PATTERN = re.compile(
    r"(?:my name is|मी|मैं)\s+([A-Za-z\u0900-\u097F]{2,30})",
    re.IGNORECASE,
)
HOUSE_NUMBER_PATTERN = re.compile(
    r"\b(?:plot|house|flat|door|h\.?no\.?|block)\s*[#\-]?\s*\w{1,10}\b",
    re.IGNORECASE,
)


def redact_pii(data: Any, is_identity_masked: bool = False) -> Any:
    """Recursively redact PII from trace payloads. Run Aadhaar first to avoid partial eating by phone regex."""
    if isinstance(data, str):
        text = AADHAAR_PATTERN.sub("[REDACTED_ID]", data)
        text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
        text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        text = NAME_AFTER_PATTERN.sub(r"my name is [REDACTED_NAME]", text)
        text = HOUSE_NUMBER_PATTERN.sub("[REDACTED_ADDRESS]", text)
        if is_identity_masked:
            text = re.sub(r"citizen_[a-zA-Z0-9_]+", "ANONYMOUS_PROTECTED", text)
        return text
    elif isinstance(data, dict):
        return {k: redact_pii(v, is_identity_masked) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_pii(item, is_identity_masked) for item in data]
    return data


def _salted_user_id(citizen_id: str) -> str:
    """One-way salted hash so raw citizen_id never enters any external trace."""
    raw = f"{settings.LANGFUSE_SALT}{citizen_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# GrievanceTracer
# ---------------------------------------------------------------------------
def make_trace_id(reference_no: str) -> str:
    """Deterministic 32-hex trace id (valid W3C/OpenTelemetry id); never contains the citizen id."""
    return hashlib.sha256(f"{settings.LANGFUSE_SALT}|trace|{reference_no}".encode()).hexdigest()[:32]


class GrievanceTracer:
    """
    Imperative wrapper over the Langfuse Python SDK (v3+, OpenTelemetry based).

    * Every span of one grievance shares one trace id (passed via trace_context).
    * Spans are sent with redacted payloads; sensitive complaint text is stubbed.
    * A trace is marked PUBLIC only in finalize(), after the policy decision is known, and only for
      non-sensitive, non-identity-masked cases (TRACE_PUBLIC_FOR_SENSITIVE=true overrides, for demos).
    * Nothing is fabricated: no keys / failed auth -> link is None and visibility is "disabled".
    """

    def __init__(self):
        self.client = None
        self.langfuse_enabled = False
        self.error_count = 0
        self.last_error: Optional[str] = None
        self._public: set = set()
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._init_langfuse()

    def _init_langfuse(self):
        pk, sk, host = settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY, settings.LANGFUSE_HOST
        if not (pk and sk):
            logger.warning(
                "LANGFUSE_* keys not set - tracing disabled. Set LANGFUSE_PUBLIC_KEY, "
                "LANGFUSE_SECRET_KEY, LANGFUSE_HOST (and LANGFUSE_PROJECT_ID) in .env to enable."
            )
            return
        try:
            from langfuse import Langfuse  # type: ignore
        except Exception as exc:
            logger.warning("langfuse package not importable (%s). Tracing disabled.", exc)
            return
        client = None
        for kw in ("base_url", "host"):          # constructor argument name differs between SDK majors
            try:
                client = Langfuse(public_key=pk, secret_key=sk, **{kw: host})
                break
            except TypeError:
                continue
            except Exception as exc:
                logger.warning("Failed to initialise Langfuse: %s. Tracing disabled.", exc)
                return
        if client is None:
            logger.warning("Could not construct a Langfuse client. Tracing disabled.")
            return
        try:
            if hasattr(client, "auth_check") and not client.auth_check():
                logger.warning("Langfuse credentials rejected (auth_check failed). Tracing disabled.")
                return
        except Exception as exc:
            logger.warning("Langfuse auth_check could not run (%s); continuing, spans may not arrive.", exc)
        self.client = client
        self.langfuse_enabled = True
        logger.info("Langfuse tracing enabled (host=%s).", host)

    @property
    def is_enabled(self) -> bool:
        return self.langfuse_enabled and self.client is not None

    def _record_error(self, what: str, exc: Exception):
        self.error_count += 1
        self.last_error = f"{what}: {exc}"[:200]
        logger.warning("Langfuse %s failed: %s", what, exc)

    # ------------------------------------------------------------------ lifecycle
    def create_trace(self, reference_no: str, citizen_id: str, is_identity_masked: bool = False) -> str:
        """Allocates the trace id for a grievance. citizen_id is only ever stored as a salted hash."""
        trace_id = make_trace_id(reference_no)
        self._meta[trace_id] = {
            "user_id": _salted_user_id(citizen_id),
            "reference_no": reference_no,
            "masked": is_identity_masked,
        }
        return trace_id

    def log_span(
        self,
        trace_id: str,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        is_identity_masked: bool = False,
        is_sensitive_text: bool = False,
    ):
        """Logs a named span (PII-redacted; complaint/query text stubbed for sensitive cases)."""
        clean_input = redact_pii(input_data, is_identity_masked)
        clean_output = redact_pii(output_data, is_identity_masked)
        clean_meta = redact_pii(metadata or {}, is_identity_masked)

        if is_sensitive_text or is_identity_masked:
            for key in ("complaint_text", "query", "raw_query"):
                if key in clean_input:
                    clean_input[key] = f"[PROTECTED - length={len(str(input_data.get(key, '')))}]"

        if not self.is_enabled:
            return
        try:
            with self.client.start_as_current_observation(  # type: ignore[union-attr]
                trace_context={"trace_id": trace_id},
                name=name,
                as_type="span",
                input=clean_input,
                output=clean_output,
                metadata=clean_meta,
            ):
                pass  # span is recorded when the context manager exits
        except Exception as exc:
            self._record_error(f"span '{name}'", exc)

    def finalize(self, trace_id: str, public: bool, output: Optional[Dict[str, Any]] = None):
        """
        Closes the trace: names it, tags it, and marks it public iff `public` is True.
        Called once, after the policy decision, so a case later found sensitive is never public.
        """
        meta = self._meta.get(trace_id, {})
        if not self.is_enabled:
            return
        try:
            with self.client.start_as_current_observation(  # type: ignore[union-attr]
                trace_context={"trace_id": trace_id},
                name="Trace_Finalized",
                as_type="span",
                input={"visibility": "public" if public else "restricted"},
                output=redact_pii(output or {}, bool(meta.get("masked"))),
            ) as span:
                try:
                    span.update_trace(
                        name=f"grievance:{meta.get('reference_no', '')}",
                        user_id=meta.get("user_id"),
                        tags=["hackfusion", "public" if public else "restricted"],
                    )
                except Exception as exc:
                    self._record_error("update_trace", exc)
                if public:
                    try:
                        span.set_trace_as_public()
                    except AttributeError:
                        self.client.set_current_trace_as_public()  # type: ignore[union-attr]
            if public:
                self._public.add(trace_id)
        except Exception as exc:
            self._record_error("finalize", exc)

    # ------------------------------------------------------------------ links
    def _trace_url(self, trace_id: str) -> Optional[str]:
        if not self.is_enabled:
            return None
        try:
            url = self.client.get_trace_url(trace_id=trace_id)  # type: ignore[union-attr]
            if url:
                return url
        except Exception as exc:
            logger.debug("get_trace_url failed (%s); building from settings.", exc)
        if settings.LANGFUSE_HOST and settings.LANGFUSE_PROJECT_ID:
            return f"{settings.LANGFUSE_HOST.rstrip('/')}/project/{settings.LANGFUSE_PROJECT_ID}/traces/{trace_id}"
        return None

    def get_public_trace_link(self, trace_id: str) -> Optional[str]:
        """Link that opens WITHOUT a Langfuse login. None unless the trace was marked public."""
        if trace_id not in self._public:
            return None
        return self._trace_url(trace_id)

    def get_internal_trace_link(self, trace_id: str) -> Optional[str]:
        """Link for project members (requires Langfuse login). Used for restricted traces."""
        return self._trace_url(trace_id)

    def visibility(self, trace_id: str) -> str:
        if not self.is_enabled:
            return "disabled"
        return "public" if trace_id in self._public else "restricted"

    def flush(self):
        if self.is_enabled:
            try:
                self.client.flush()  # type: ignore[union-attr]
            except Exception as exc:
                self._record_error("flush", exc)

    def shutdown(self):
        if self.is_enabled:
            try:
                self.client.shutdown()  # type: ignore[union-attr]
                logger.info("Langfuse client shut down cleanly.")
            except Exception as exc:
                self._record_error("shutdown", exc)


tracer = GrievanceTracer()
