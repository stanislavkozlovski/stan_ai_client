from __future__ import annotations

import json

import pytest

from stan_ai_client.grok_parser import (
    parse_grok_json_payload,
    summarize_grok_error_text,
    try_parse_grok_json_payload,
)
from stan_ai_client.types import GrokJsonPayload


def test_parse_success():
    text = '{"text": "hello", "stopReason": "EndTurn", "sessionId": "abc", "requestId": "req1"}'
    payload = parse_grok_json_payload(text)
    assert isinstance(payload, GrokJsonPayload)
    assert payload.text == "hello"
    assert payload.stop_reason == "EndTurn"
    assert payload.session_id == "abc"
    assert payload.has_structured_output is False


def test_parse_structured():
    text = '{"text": "{\\"x\\":1}", "stopReason": "EndTurn", "structuredOutput": {"x": 1}}'
    payload = parse_grok_json_payload(text)
    assert payload.has_structured_output is True
    assert payload.structured_output == {"x": 1}


def test_parse_error_envelope():
    text = '{"type": "error", "message": "boom"}'
    payload = parse_grok_json_payload(text)
    assert payload.text is None
    assert "error" in payload.extras.get("type", "")


def test_try_parse_bad_json_returns_none():
    assert try_parse_grok_json_payload("not json") is None
    assert try_parse_grok_json_payload("") is None


def test_summarize_error_prefers_text():
    payload = GrokJsonPayload(text="result here", stop_reason=None, session_id=None, request_id=None, thought=None, structured_output=None)
    assert summarize_grok_error_text(payload=payload, stdout="", stderr="") == "result here"


def test_summarize_falls_back_to_stderr():
    assert "stderr" in summarize_grok_error_text(payload=None, stdout="", stderr="some stderr error") 
