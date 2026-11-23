"""Tracing: valid ids, correct SDK call shape, deferred public flag, no fabricated links."""
import re
from contextlib import contextmanager

from app.core.observability import GrievanceTracer, make_trace_id


class FakeSpan:
    def __init__(self, log):
        self.log = log

    def update_trace(self, **kw):
        self.log.append(("update_trace", kw))

    def set_trace_as_public(self):
        self.log.append(("public", None))


class FakeClient:
    def __init__(self):
        self.calls, self.inner = [], []

    @contextmanager
    def start_as_current_observation(self, **kw):
        self.calls.append(kw)
        yield FakeSpan(self.inner)

    def get_trace_url(self, trace_id=None):
        return f"https://lf.example/project/p1/traces/{trace_id}"

    def flush(self):
        self.calls.append({"flush": True})


def _enabled():
    t = GrievanceTracer()
    t.client, t.langfuse_enabled = FakeClient(), True
    return t


def test_trace_id_is_valid_otel_hex_and_deterministic():
    tid = make_trace_id("GRV-2026-ABC123")
    assert re.fullmatch(r"[0-9a-f]{32}", tid) and tid == make_trace_id("GRV-2026-ABC123")
    assert "GRV" not in tid


def test_disabled_tracer_never_fabricates_a_link():
    t = GrievanceTracer()
    t.client, t.langfuse_enabled = None, False
    tid = t.create_trace("GRV-1", "citizen_x")
    t.log_span(tid, "x", {}, {})
    t.finalize(tid, public=True)
    assert t.get_public_trace_link(tid) is None and t.visibility(tid) == "disabled"


def test_spans_use_trace_context_and_valid_sdk_arguments():
    t = _enabled()
    tid = t.create_trace("GRV-1", "citizen_x")
    t.log_span(tid, "Policy_Agent_Master_Gate", {"a": 1}, {"verdict": "AUTO_HANDLE"})
    kw = t.client.calls[0]
    assert kw["trace_context"] == {"trace_id": tid} and kw["as_type"] == "span" and kw["name"] == "Policy_Agent_Master_Gate"
    assert "trace_id" not in kw and "type" not in kw      # the old (invalid) kwargs


def test_trace_is_private_until_finalized_public():
    t = _enabled()
    tid = t.create_trace("GRV-1", "citizen_x")
    t.log_span(tid, "x", {}, {})
    assert t.get_public_trace_link(tid) is None and t.visibility(tid) == "restricted"
    t.finalize(tid, public=True, output={"verdict": "AUTO_HANDLE"})
    assert ("public", None) in t.client.inner
    assert t.get_public_trace_link(tid).endswith(tid) and t.visibility(tid) == "public"


def test_restricted_trace_is_never_marked_public():
    t = _enabled()
    tid = t.create_trace("GRV-2", "citizen_y", is_identity_masked=True)
    t.finalize(tid, public=False)
    assert ("public", None) not in t.client.inner
    assert t.get_public_trace_link(tid) is None and t.get_internal_trace_link(tid).endswith(tid)


def test_user_id_is_hashed_and_complaint_text_is_stubbed_for_protected_cases():
    t = _enabled()
    tid = t.create_trace("GRV-3", "citizen_secret_99", is_identity_masked=True)
    t.log_span(tid, "Vector_RAG_Retrieval", {"query": "girl harassed near the college gate"}, {},
               is_identity_masked=True, is_sensitive_text=True)
    t.finalize(tid, public=False)
    assert "girl harassed" not in str(t.client.calls)
    assert "citizen_secret_99" not in str(t.client.calls) + str(t.client.inner)


def test_span_failures_are_counted_not_hidden():
    t = _enabled()

    @contextmanager
    def boom(**kw):
        raise TypeError("bad kwarg")
        yield

    t.client.start_as_current_observation = boom
    t.log_span("a" * 32, "x", {}, {})
    assert t.error_count == 1 and "span 'x'" in t.last_error
