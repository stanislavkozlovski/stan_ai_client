from __future__ import annotations

from stan_ai_client.grok_parser import (
    parse_grok_json_payload,
    summarize_grok_error_text,
    try_parse_grok_json_payload,
)
from stan_ai_client.types import GrokJsonPayload


def test_parse_success() -> None:
    text = '{"text": "hello", "stopReason": "EndTurn", "sessionId": "abc", "requestId": "req1"}'
    payload = parse_grok_json_payload(text)
    assert isinstance(payload, GrokJsonPayload)
    assert payload.text == "hello"
    assert payload.stop_reason == "EndTurn"
    assert payload.session_id == "abc"
    assert payload.has_structured_output is False


def test_parse_structured() -> None:
    text = '{"text": "{\\"x\\":1}", "stopReason": "EndTurn", "structuredOutput": {"x": 1}}'
    payload = parse_grok_json_payload(text)
    assert payload.has_structured_output is True
    assert payload.structured_output == {"x": 1}


def test_parse_raw_structured_output() -> None:
    payload = parse_grok_json_payload('{"x": 1}', raw_structured_output=True)
    assert payload.has_structured_output is True
    assert payload.structured_output == {"x": 1}
    assert payload.extras == {}


def test_parse_raw_structured_output_preserves_structured_output_key() -> None:
    payload = parse_grok_json_payload(
        '{"structuredOutput": "ok"}',
        raw_structured_output=True,
    )
    assert payload.has_structured_output is True
    assert payload.structured_output == {"structuredOutput": "ok"}


def test_parse_raw_structured_output_accepts_envelope_with_metadata() -> None:
    payload = parse_grok_json_payload(
        '{"text": "{\\"x\\":1}", "structuredOutput": {"x": 1}}',
        raw_structured_output=True,
    )
    assert payload.has_structured_output is True
    assert payload.structured_output == {"x": 1}
    assert payload.text == '{"x":1}'


def test_parse_preserves_falsy_structured_output() -> None:
    payload = parse_grok_json_payload('{"text": "false", "structuredOutput": false}')
    assert payload.has_structured_output is True
    assert payload.structured_output is False


def test_parse_error_envelope() -> None:
    text = '{"type": "error", "message": "boom"}'
    payload = parse_grok_json_payload(text)
    assert payload.text is None
    assert "error" in payload.extras.get("type", "")


def test_try_parse_bad_json_returns_none() -> None:
    assert try_parse_grok_json_payload("not json") is None
    assert try_parse_grok_json_payload("") is None


def test_summarize_error_prefers_text() -> None:
    payload = GrokJsonPayload(
        text="result here",
        stop_reason=None,
        session_id=None,
        request_id=None,
        thought=None,
        structured_output=None,
    )
    assert summarize_grok_error_text(payload=payload, stdout="", stderr="") == "result here"


def test_summarize_error_prefers_error_envelope_message() -> None:
    payload = parse_grok_json_payload('{"type": "error", "message": "boom"}')
    assert summarize_grok_error_text(payload=payload, stdout="", stderr="") == "boom"


def test_summarize_falls_back_to_stderr() -> None:
    assert "stderr" in summarize_grok_error_text(
        payload=None, stdout="", stderr="some stderr error"
    )
