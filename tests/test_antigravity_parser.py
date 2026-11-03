from __future__ import annotations

import json
from typing import Any

from stan_ai_client.antigravity_parser import (
    antigravity_error_lines,
    decode_antigravity_stream,
    has_auto_mode_unavailable_warning,
    has_current_structured_output,
    has_missing_conversation_warning,
    has_print_timeout_notice,
    headless_denial_notice,
    parse_antigravity_agent_error,
    recover_antigravity_partial_payload,
    stream_input_message,
    sum_step_usage,
    summarize_antigravity_error_text,
    try_parse_antigravity_payload,
)
from stan_ai_client.rate_limits import is_antigravity_rate_limit_text
from stan_ai_client.types import AntigravityJsonPayload

# Captured from agy 1.4.12 (`agy -p "Reply with the single word: ok"
# --output-format stream-json`), tool list shortened.
LIVE_STREAM = "\n".join(
    [
        '{"event":"init","conversation_id":"a2335fa4-b3a3-4b43-b180-075b61ae57bc","init":{"cwd":"/home/ops","tools":["run_command","view_file"],"permission_mode":"request-review"}}',
        '{"event":"step_update","step_update":{"conversation_id":"a2335fa4-b3a3-4b43-b180-075b61ae57bc","step_index":0,"state":"DONE","step_type":"user_input"}}',
        '{"event":"step_update","step_update":{"conversation_id":"a2335fa4-b3a3-4b43-b180-075b61ae57bc","step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"ok"}}',
        '{"event":"step_update","step_update":{"conversation_id":"a2335fa4-b3a3-4b43-b180-075b61ae57bc","step_index":1,"state":"DONE","step_type":"agent_response","text_delta":"\\n","duration_seconds":2.457760979,"usage":{"input_tokens":7500,"output_tokens":25,"thinking_tokens":24,"cache_read_tokens":0,"total_tokens":7525}}}',
        '{"event":"result","result":{"conversation_id":"a2335fa4-b3a3-4b43-b180-075b61ae57bc","status":"SUCCESS","response":"ok\\n","duration_seconds":2.521645668,"num_turns":1,"usage":{"input_tokens":7500,"output_tokens":25,"thinking_tokens":24,"cache_read_tokens":0,"total_tokens":7525}}}',
    ]
)
LIVE_AGENT_ERROR = (
    'AGY_ERROR: {"short_error":"failed to construct executor: plan model not specified; '
    'you must specify a valid model","status":"UNKNOWN","error_code":2,"code_kind":"grpc",'
    '"retryable":false,"error_id":"560d6d86-fbc6-429b-9496-68661f1291e1-1"}'
)


def step(index: Any, step_type: str, state: str = "DONE", **fields: Any) -> dict[str, Any]:
    return {
        "event": "step_update",
        "step_update": {"step_index": index, "state": state, "step_type": step_type, **fields},
    }


def test_stream_input_message_matches_agys_user_event() -> None:
    line = stream_input_message('line one\nline "two"')
    assert line.endswith("\n") and line.count("\n") == 1
    assert json.loads(line) == {
        "event": "user",
        "message": {"role": "user", "content": 'line one\nline "two"'},
    }


def test_live_stream_parses_into_payload() -> None:
    payload = try_parse_antigravity_payload(LIVE_STREAM)
    assert payload is not None
    assert payload.conversation_id == "a2335fa4-b3a3-4b43-b180-075b61ae57bc"
    assert payload.response == "ok\n"
    assert payload.permission_mode == "request-review"
    assert payload.step_usage == payload.usage
    assert payload.stream_diagnostics == ()
    assert [event["event"] for event in payload.events] == [
        "init",
        "step_update",
        "step_update",
        "step_update",
        "result",
    ]


def test_stream_decoding_counts_lines_that_are_not_events() -> None:
    stream = decode_antigravity_stream('\n["list"]\n{"no":"event"}\nplain\n{"event":"init"}\n')
    assert stream.invalid_lines == 3
    assert stream.result is None
    assert stream.events == ({"event": "init"},)


def test_payload_requires_a_result_event() -> None:
    assert try_parse_antigravity_payload('{"event":"init","conversation_id":"c"}\n') is None
    assert try_parse_antigravity_payload("") is None


def test_payload_from_dict_normalizes_types() -> None:
    payload = AntigravityJsonPayload.from_dict(
        {
            "conversation_id": "",
            "status": 3,
            "response": None,
            "duration_seconds": True,
            "num_turns": "2",
            "usage": [],
            "denied_actions": [{"action": "command"}, "junk"],
            "structured_output": None,
        }
    )
    assert payload.conversation_id is None
    assert payload.status is None
    assert payload.duration_seconds is None
    assert payload.num_turns is None
    assert payload.usage == {}
    assert payload.denied_actions == ({"action": "command"},)
    assert payload.has_structured_output and payload.structured_output is None


def test_step_usage_sums_last_done_report_per_step() -> None:
    events = (
        step(1, "agent_response", "ACTIVE", usage={"input_tokens": 999}),
        step(1, "agent_response", usage={"input_tokens": 10, "output_tokens": 1}),
        step(1, "agent_response", usage={"input_tokens": 12, "output_tokens": 2}),
        step(2, "agent_response", usage={"input_tokens": 20, "output_tokens": 3}),
        step(None, "agent_response", usage={"input_tokens": 1}),
        step(3, "agent_response", conversation_id="subagent", usage={"input_tokens": 500}),
    )
    assert sum_step_usage(events, conversation_id="parent") == {
        "input_tokens": 33,
        "output_tokens": 5,
    }


def test_step_usage_is_unknown_when_a_counter_is_unusable() -> None:
    events = (step(1, "agent_response", usage={"input_tokens": -1}),)
    assert sum_step_usage(events) is None
    events = (step(1, "agent_response", usage={"input_tokens": True}),)
    assert sum_step_usage(events) is None


def test_current_structured_output_requires_a_completed_finish_step() -> None:
    def payload(*events: dict[str, Any]) -> AntigravityJsonPayload:
        return AntigravityJsonPayload.from_dict({}, events=events)

    assert has_current_structured_output(payload(step(4, "finish")))
    assert not has_current_structured_output(payload(step(4, "finish", "ACTIVE")))
    assert not has_current_structured_output(
        payload(step(6, "tool", "ERROR", tool_name="finish"))
    )
    assert not has_current_structured_output(payload())


def test_partial_payload_keeps_conversation_and_streamed_text() -> None:
    lines = LIVE_STREAM.splitlines()[:3]
    payload = recover_antigravity_partial_payload("\n".join(lines))
    assert payload is not None
    assert payload.conversation_id == "a2335fa4-b3a3-4b43-b180-075b61ae57bc"
    assert payload.response == "ok"
    assert payload.status is None
    assert "stream ended before its result event" in payload.stream_diagnostics
    assert recover_antigravity_partial_payload("") is None
    complete = recover_antigravity_partial_payload(LIVE_STREAM)
    assert complete is not None and complete.status == "SUCCESS"


def test_agent_error_record_parses_the_last_valid_line() -> None:
    stderr = "\n".join(
        [
            "error: failed to construct executor: plan model not specified",
            "AGY_ERROR: {not json",
            LIVE_AGENT_ERROR,
            "AGY_ERROR: []",
        ]
    )
    record = parse_antigravity_agent_error(stderr)
    assert record is not None
    assert record.status == "UNKNOWN"
    assert record.error_code == 2
    assert record.code_kind == "grpc"
    assert record.retryable is False
    assert record.short_error is not None and "plan model" in record.short_error
    assert record.raw["error_id"] == "560d6d86-fbc6-429b-9496-68661f1291e1-1"
    assert parse_antigravity_agent_error("error: plain\n") is None


def test_error_lines_and_notices() -> None:
    stderr = "\n".join(
        [
            "warning: --mode auto is not available because the terminal sandbox is not "
            "available on this host; continuing in the default mode.",
            'warning: conversation "abc" not found',
            "Error: --json-schema can only be used when --output-format is 'json' or 'stream-json'",
            "[agy] print timeout after 4s with turn in progress; returning partial output",
            "jetski: no output produced — a tool required the \"command\" permission "
            "that headless mode cannot prompt for, so it was auto-denied.",
        ]
    )
    assert antigravity_error_lines(stderr) == (
        "Error: --json-schema can only be used when --output-format is 'json' or 'stream-json'",
    )
    assert has_auto_mode_unavailable_warning(stderr)
    assert has_missing_conversation_warning(stderr)
    assert has_print_timeout_notice(stderr)
    notice = headless_denial_notice(stderr)
    assert notice is not None and notice.startswith("jetski: no output produced")
    assert not has_missing_conversation_warning('note: conversation "abc" not found')
    assert headless_denial_notice("") is None


def test_error_summary_prefers_declared_errors() -> None:
    record = parse_antigravity_agent_error(LIVE_AGENT_ERROR)
    declared = AntigravityJsonPayload.from_dict({"error": "result error"})
    assert (
        summarize_antigravity_error_text(
            payload=declared, agent_error=record, stderr="error: x", returncode=3
        )
        == "result error"
    )
    assert summarize_antigravity_error_text(
        payload=None, agent_error=record, stderr="", returncode=3
    ).startswith("failed to construct executor")
    assert (
        summarize_antigravity_error_text(
            payload=None, agent_error=None, stderr="noise\nerror: first\n", returncode=1
        )
        == "error: first"
    )
    early = AntigravityJsonPayload.from_dict({"finished_reason": "interrupted"})
    assert "interrupted" in summarize_antigravity_error_text(
        payload=early, agent_error=None, stderr="", returncode=0
    )
    assert (
        summarize_antigravity_error_text(payload=None, agent_error=None, stderr="", returncode=2)
        == "Antigravity exited with code 2"
    )


def test_antigravity_rate_limit_markers() -> None:
    for text in (
        "RESOURCE_EXHAUSTED",
        "You have exhausted your quota on this model.",
        "individual quota reached",
        "STOP_REASON_QUOTA_EXHAUSTED",
        "Quota exceeded for quota metric 'Generate Content API requests per minute'",
        "RATE_LIMIT_EXCEEDED",
        "Your AI credits balance is too low to continue.",
        "HTTP 429 Too Many Requests",
        "status 429",
    ):
        assert is_antigravity_rate_limit_text(text), text
    for text in (
        "error_id 560d6d86-fbc6-429b-9496-68661f1291e1-1",
        "port 4290",
        "The model API is currently overloaded",
        "context limit reached",
        "write /tmp/x: disk quota exceeded",
    ):
        assert not is_antigravity_rate_limit_text(text), text
