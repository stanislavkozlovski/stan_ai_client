from __future__ import annotations

from stan_ai_client.codex_parser import (
    parse_codex_jsonl_payload,
    recover_codex_jsonl_prefix_payload,
    summarize_codex_error_text,
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


def test_recover_codex_jsonl_prefix_payload_stops_before_malformed_tail() -> None:
    error_event = {"type": "error", "message": "Network is unreachable"}
    text = "\n".join(
        [
            '{"type":"error","message":"Network is unreachable"}',
            '{"type":"turn.failed"',
        ]
    )

    assert try_parse_codex_jsonl_payload(text) is None

    payload = recover_codex_jsonl_prefix_payload(text)

    assert payload is not None
    assert payload.events == (error_event,)
    assert payload.error == error_event


def test_recover_codex_jsonl_prefix_payload_matches_strict_parse_when_well_formed() -> None:
    """Both entry points scan lines through the same rule, so a well-formed
    stream must recover to exactly what strict parsing produces."""
    text = "\n".join(
        [
            "",
            '  {"type":"thread.started","thread_id":"thread-1"}  ',
            "",
            '{"type":"error","message":"Network is unreachable"}',
            '{"type":"turn.completed","usage":{"input_tokens":10}}',
            "",
        ]
    )

    assert recover_codex_jsonl_prefix_payload(text) == parse_codex_jsonl_payload(text)


def test_summarize_codex_error_text_prefers_structured_error_event() -> None:
    payload = parse_codex_jsonl_payload('{"type":"error","message":"Rate limit exceeded"}')

    summary = summarize_codex_error_text(
        payload=payload,
        stdout="",
        stderr="banner noise\nERROR: something else entirely",
    )

    assert summary == "Rate limit exceeded"


def test_summarize_codex_error_text_finds_provider_error_at_stderr_tail() -> None:
    provider_error = (
        "ERROR: unexpected status 400 Bad Request: "
        '{"error":{"message":"\'uniqueItems\' is not permitted.",'
        '"code":"invalid_json_schema"}}'
    )
    stderr = "\n".join(
        [
            ">_ You are using OpenAI Codex in ~/uzealot",
            "user",
            "Generate the weekly Delta Growth report with evidence identifiers.",
            *(f"progress line {index}" for index in range(50)),
            provider_error,
        ]
    )

    summary = summarize_codex_error_text(payload=None, stdout="", stderr=stderr)

    assert summary == provider_error
    assert "invalid_json_schema" in summary
    assert "uniqueItems" in summary
    assert "Delta Growth" not in summary


def test_summarize_codex_error_text_ignores_trailing_noise_after_error_line() -> None:
    stderr = "\n".join(
        [
            "banner",
            "ERROR: stream disconnected before completion",
            "shutting down",
        ]
    )

    summary = summarize_codex_error_text(payload=None, stdout="", stderr=stderr)

    assert summary == "ERROR: stream disconnected before completion"


def test_summarize_codex_error_text_falls_back_to_bounded_stderr_tail() -> None:
    stderr = "x" * 600 + "\n" + "final diagnostic without markers"

    summary = summarize_codex_error_text(payload=None, stdout="", stderr=stderr)

    assert len(summary) == 500
    assert summary.endswith("final diagnostic without markers")


def test_summarize_codex_error_text_keeps_short_stderr_unchanged() -> None:
    summary = summarize_codex_error_text(
        payload=None,
        stdout="",
        stderr="429 rate limit exceeded, retry after 5\n",
    )

    assert summary == "429 rate limit exceeded, retry after 5"


def test_summarize_codex_error_text_falls_back_to_stdout_head() -> None:
    summary = summarize_codex_error_text(
        payload=None, stdout="plain failure\n", stderr="  "
    )

    assert summary == "plain failure"
