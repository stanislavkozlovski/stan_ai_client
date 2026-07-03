from __future__ import annotations

from stan_ai_client.codex_parser import (
    parse_codex_jsonl_payload,
    try_parse_codex_jsonl_payload,
)


def test_parse_codex_jsonl_payload_extracts_final_message_and_usage() -> None:
    payload = parse_codex_jsonl_payload(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
                '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}',
            ]
        )
    )

    assert payload.thread_id == "thread-1"
    assert payload.result == "done"
    assert payload.usage == {"input_tokens": 10, "output_tokens": 2}
    assert len(payload.events) == 3
    assert payload.error is None


def test_parse_codex_jsonl_payload_preserves_error_event() -> None:
    payload = parse_codex_jsonl_payload(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"error","message":"Rate limit exceeded"}',
            ]
        )
    )

    assert payload.error == {"type": "error", "message": "Rate limit exceeded"}


def test_parse_codex_jsonl_payload_skips_blank_lines() -> None:
    payload = parse_codex_jsonl_payload(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"thread-1"}',
                "",
                "  ",
                '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
            ]
        )
    )

    assert payload.thread_id == "thread-1"
    assert payload.result == "done"
    assert len(payload.events) == 2


def test_parse_codex_jsonl_payload_rejects_non_event_object() -> None:
    try:
        parse_codex_jsonl_payload('{"summary":"brief"}')
    except ValueError as exc:
        assert "string type" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-event JSON object to be rejected")


def test_try_parse_codex_jsonl_payload_returns_none_for_invalid_jsonl() -> None:
    assert try_parse_codex_jsonl_payload("not json") is None
