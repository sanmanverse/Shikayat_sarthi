"""Sarvam client request format (per docs.sarvam.ai), retries and honesty. No network: requests.post is faked."""
import json
import pytest
from fastapi import HTTPException

from app.services import sarvam as sv
from app.services.sarvam import SarvamClient, extract_json_object, _strip_reasoning

KEY = "sk_test_123456"


class FakeResp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text or json.dumps(body or {})

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _chat_body(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}]}


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(sv.time, "sleep", lambda s: None)


def test_chat_request_uses_published_endpoint_header_and_model(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        seen.update(url=url, headers=headers, payload=json)
        return FakeResp(200, _chat_body("hello"))

    monkeypatch.setattr(sv.requests, "post", fake_post)
    out, engine, degraded = SarvamClient(api_key=KEY).chat_completion([{"role": "user", "content": "hi"}])
    assert (out, engine, degraded) == ("hello", "sarvam", False)
    assert seen["url"] == "https://api.sarvam.ai/v1/chat/completions"
    assert seen["headers"]["api-subscription-key"] == KEY
    assert seen["payload"]["model"] == "sarvam-105b"
    assert "reasoning_effort" not in seen["payload"]      # only sent when configured


def test_inline_reasoning_is_stripped():
    assert _strip_reasoning("<think>secret chain</think>{\"a\": 1}") == '{"a": 1}'
    assert extract_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert _strip_reasoning("answer <think>never closed") == "answer"


def test_auth_failure_is_not_retried_and_is_marked_degraded(monkeypatch):
    calls = []
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: calls.append(1) or FakeResp(401, {}, "bad key"))
    out, engine, degraded = SarvamClient(api_key=KEY).chat_completion([{"role": "user", "content": "water leak"}])
    assert len(calls) == 1 and engine == "mock" and degraded is True


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    seq = iter([FakeResp(429, {}, "slow down"), FakeResp(200, _chat_body("ok"))])
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: next(seq))
    assert SarvamClient(api_key=KEY).chat_completion([{"role": "user", "content": "x"}])[:2] == ("ok", "sarvam")


def test_empty_content_counts_as_failure(monkeypatch):
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: FakeResp(200, _chat_body("")))
    assert SarvamClient(api_key=KEY).chat_completion([{"role": "user", "content": "x"}])[2] is True


def test_not_configured_is_mock_not_degraded():
    out, engine, degraded = SarvamClient(api_key="").chat_completion([{"role": "user", "content": "x"}])
    assert engine == "mock" and degraded is False


def test_tool_calling_sends_tools_and_returns_calls(monkeypatch):
    seen = {}
    tc = [{"id": "c1", "function": {"name": "finalize_triage", "arguments": "{}"}}]

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        seen["payload"] = json
        return FakeResp(200, _chat_body("", tc))

    monkeypatch.setattr(sv.requests, "post", fake_post)
    msg = SarvamClient(api_key=KEY).chat_with_tools([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert msg["tool_calls"] == tc and seen["payload"]["tools"] == [{"type": "function"}]


def test_stt_request_format_and_no_canned_transcript(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None, **kw):
        seen.update(url=url, headers=headers, files=files, data=data)
        return FakeResp(200, {"transcript": "नमस्कार", "language_code": "hi-IN"})

    monkeypatch.setattr(sv.requests, "post", fake_post)
    res = SarvamClient(api_key=KEY).transcribe_audio(b"RIFF", filename="clip.webm")
    assert seen["url"].endswith("/speech-to-text")
    assert seen["data"]["model"] == "saaras:v3" and seen["data"]["mode"] == "transcribe"
    assert seen["files"]["file"][2] == "audio/webm"
    assert res["transcript"] == "नमस्कार" and res["engine"] == "sarvam"

    with pytest.raises(HTTPException) as e:
        SarvamClient(api_key="").transcribe_audio(b"x")
    assert e.value.status_code == 503


def test_stt_failure_is_503_not_a_fake_transcript(monkeypatch):
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(HTTPException) as e:
        SarvamClient(api_key=KEY).transcribe_audio(b"x")
    assert e.value.status_code == 503


def test_safety_classify_statuses(monkeypatch):
    c = SarvamClient(api_key=KEY)
    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: FakeResp(
        200, _chat_body('{"sensitive": true, "concept": "electrical_hazard", "confidence": 0.9}')))
    res, status = c.safety_classify("तार लटक रहा है")
    assert status == "ok" and res["sensitive"] is True and res["concept"] == "Exposed Wire / Electrical Hazard"

    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: FakeResp(200, _chat_body("I think it is fine")))
    assert c.safety_classify("x") == (None, "error")

    monkeypatch.setattr(sv.requests, "post", lambda *a, **k: FakeResp(500, {}, "oops"))
    assert c.safety_classify("x") == (None, "error")

    assert SarvamClient(api_key="").safety_classify("x") == (None, "not_configured")
