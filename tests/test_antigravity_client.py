from __future__ import annotations

import json
import logging
import subprocess
from typing import Any
from unittest.mock import Mock, patch

import pytest

from stan_ai_client import (
    PROVIDER_RETRY_PREFIX,
    AntigravityApprovalRequiredError,
    AntigravityCapacityError,
    AntigravityClient,
    AntigravityCodeError,
    AntigravityExecutableNotFoundError,
    AntigravityNetworkUnavailableError,
    AntigravityProcessError,
    AntigravityProtocolError,
    AntigravityRateLimitError,
    AntigravityRunOptions,
    AntigravitySchemaValidationError,
    AntigravityServerError,
    AntigravitySessionNotFoundError,
    AntigravityStructuredOutputMissingError,
    AntigravityStructuredOutputValidationError,
    AntigravityTimeoutError,
    ApprovalRequiredError,
    CapacityError,
    NetworkUnavailableError,
    ProviderRetryPolicy,
    RateLimitError,
    RateLimitRetryPolicy,
    ServerError,
    StructuredSchema,
    classify_provider_failure,
    normalize_ai_usage,
    validate_antigravity_output_schema,
)
from stan_ai_client.antigravity import _redact_argv

CONVERSATION = "2a385ab1-271c-46f8-83f6-6b6ef29289a0"
SCHEMA: StructuredSchema[dict[str, Any]] = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {"count": {"type": "integer", "maximum": 3}},
        "required": ["count"],
        "additionalProperties": False,
    }
)
DENIAL_NOTICE = (
    "jetski: no output produced — a tool required the \"command\" permission that "
    "headless mode cannot prompt for, so it was auto-denied. Add an allow-rule under "
    "permissions.allow in settings.json (e.g. command(<target>)). Alternatively, re-run "
    "with --dangerously-skip-permissions to auto-approve all tools."
)


def usage(
    input_tokens: int = 7500, output: int = 25, thinking: int = 24, cache_read: int = 0
) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output,
        "thinking_tokens": thinking,
        "cache_read_tokens": cache_read,
        "total_tokens": input_tokens + output,
    }


def init_event(
    conversation: str = CONVERSATION, permission_mode: str = "always-proceed"
) -> dict[str, Any]:
    return {
        "event": "init",
        "conversation_id": conversation,
        "init": {"cwd": "/work", "tools": ["run_command"], "permission_mode": permission_mode},
    }


def step(
    index: int,
    step_type: str,
    *,
    state: str = "DONE",
    conversation: str = CONVERSATION,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "event": "step_update",
        "step_update": {
            "conversation_id": conversation,
            "step_index": index,
            "state": state,
            "step_type": step_type,
            **fields,
        },
    }


def result_event(
    conversation: str = CONVERSATION,
    *,
    status: str = "SUCCESS",
    response: str = "ok\n",
    usage_counts: dict[str, int] | None = None,
    num_turns: int = 1,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "event": "result",
        "result": {
            "conversation_id": conversation,
            "status": status,
            "response": response,
            "duration_seconds": 2.5,
            "num_turns": num_turns,
            "usage": usage() if usage_counts is None else usage_counts,
            **fields,
        },
    }


def stream(*events: dict[str, Any]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


def simple_stream(response: str = "ok\n", **result_fields: Any) -> str:
    return stream(
        init_event(),
        step(0, "user_input"),
        step(1, "agent_response", text_delta=response, usage=usage()),
        result_event(response=response, **result_fields),
    )


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def agent_error_line(**fields: Any) -> str:
    record = {
        "short_error": "boom",
        "status": "UNKNOWN",
        "error_code": 2,
        "code_kind": "grpc",
        "retryable": False,
        "error_id": "560d6d86-fbc6-429b-9496-68661f1291e1-1",
        **fields,
    }
    return "AGY_ERROR: " + json.dumps(record)


def argv_of(mock_exec: Mock, call: int = -1) -> tuple[str, ...]:
    return tuple(mock_exec.call_args_list[call][0][0].argv)


def flag_value(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --- construction and command building ---------------------------------


def test_defaults_match_the_package_automation_preferences() -> None:
    client = AntigravityClient()
    assert client.executable == "agy"
    assert client.default_model == "gemini-4-argon"
    assert client.default_effort == "high"
    assert client.default_permission_mode == "bypassPermissions"
    assert client.default_timeout_seconds == 120.0


@patch("stan_ai_client.antigravity.execute_command")
def test_default_call_sends_prompt_as_stream_json_on_stdin(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(simple_stream())

    result = AntigravityClient().run_text("say -ok")

    prepared = mock_exec.call_args[0][0]
    assert prepared.argv == (
        "agy",
        "--model",
        "gemini-4-argon",
        "--effort",
        "high",
        "--dangerously-skip-permissions",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
    )
    assert json.loads(prepared.input_text) == {
        "event": "user",
        "message": {"role": "user", "content": "say -ok"},
    }
    assert prepared.input_text.endswith("\n")
    assert not any("say -ok" in value for value in prepared.argv)
    assert result.text == "ok"
    assert result.returncode == 0


@patch("stan_ai_client.antigravity.execute_command")
def test_stdin_message_is_ascii_safe_for_any_locale(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(simple_stream())

    AntigravityClient().run_text("café ☕ \U0001f600")

    input_text = mock_exec.call_args[0][0].input_text
    assert input_text.isascii()
    assert json.loads(input_text)["message"]["content"] == "café ☕ \U0001f600"


@patch("stan_ai_client.antigravity.execute_command")
def test_argv_text_mode_attaches_prompt_and_reads_plain_stdout(mock_exec: Mock) -> None:
    mock_exec.return_value = completed("  plain answer\n")

    result = AntigravityClient().run_text(
        "--looks-like-a-flag", options=AntigravityRunOptions(input_mode="argv")
    )

    prepared = mock_exec.call_args[0][0]
    assert prepared.argv[-3:] == ("--output-format", "text", "--print=--looks-like-a-flag")
    assert "--input-format" not in prepared.argv
    assert prepared.input_text == ""
    assert result.text == "plain answer"
    assert result.stdout == "  plain answer\n"


@patch("stan_ai_client.antigravity.execute_command")
def test_argv_json_mode_still_reads_stream_json(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(simple_stream())

    result = AntigravityClient().run_json(
        "hello", options=AntigravityRunOptions(input_mode="argv")
    )

    argv = argv_of(mock_exec)
    assert flag_value(argv, "--output-format") == "stream-json"
    assert argv[-1] == "--print=hello"
    assert result.payload.response == "ok\n"


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("default", ()),
        ("acceptEdits", ("--mode", "accept-edits")),
        ("plan", ("--mode", "plan")),
        ("auto", ("--mode", "auto")),
        ("bypassPermissions", ("--dangerously-skip-permissions",)),
    ],
)
@patch("stan_ai_client.antigravity.execute_command")
def test_permission_modes_map_to_agy_flags(
    mock_exec: Mock, mode: Any, expected: tuple[str, ...]
) -> None:
    mock_exec.return_value = completed(simple_stream())

    AntigravityClient().run_json("hi", options=AntigravityRunOptions(permission_mode=mode))

    argv = argv_of(mock_exec)
    emitted = tuple(
        value
        for index, value in enumerate(argv)
        if value in {"--mode", "--dangerously-skip-permissions"}
        or (index > 0 and argv[index - 1] == "--mode")
    )
    assert emitted == expected


@patch("stan_ai_client.antigravity.execute_command")
def test_typed_options_map_to_flags_before_protocol_flags(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(simple_stream(), returncode=0)

    AntigravityClient().run_json(
        "hi",
        options=AntigravityRunOptions(
            cwd="/work",
            model="claude-opus-5-5",
            effort="medium",
            add_dirs=("/a", "/b"),
            agent="reviewer",
            project="proj",
            settings='{"theme":"dark"}',
            sandbox=False,
            disable_slash_commands=True,
            print_timeout_seconds=1.5,
            log_file="/tmp/agy.log",
            extra_args=("--output-format", "text", "--new-project"),
        ),
    )

    prepared = mock_exec.call_args[0][0]
    argv = prepared.argv
    assert prepared.cwd == "/work"
    assert flag_value(argv, "--model") == "claude-opus-5-5"
    assert flag_value(argv, "--effort") == "medium"
    assert [argv[i + 1] for i, value in enumerate(argv) if value == "--add-dir"] == ["/a", "/b"]
    assert flag_value(argv, "--agent") == "reviewer"
    assert flag_value(argv, "--project") == "proj"
    assert flag_value(argv, "--settings") == '{"theme":"dark"}'
    assert "--sandbox=false" in argv
    assert "--disable-slash-commands" in argv
    assert flag_value(argv, "--print-timeout") == "1500ms"
    assert flag_value(argv, "--log-file") == "/tmp/agy.log"
    # extra_args precede the protocol flags, so the client's wire format wins.
    assert argv.index("--new-project") < argv.index("--input-format")
    assert argv[-2:] == ("--output-format", "stream-json")


@pytest.mark.parametrize("seconds,expected", [(2, "2s"), (0, "0s"), (90.0, "90s"), (0.25, "250ms")])
@patch("stan_ai_client.antigravity.execute_command")
def test_print_timeout_uses_go_duration_units(
    mock_exec: Mock, seconds: float, expected: str
) -> None:
    mock_exec.return_value = completed(simple_stream())

    AntigravityClient().run_json(
        "hi", options=AntigravityRunOptions(print_timeout_seconds=seconds)
    )

    assert flag_value(argv_of(mock_exec), "--print-timeout") == expected


@pytest.mark.parametrize("seconds", [-1.0, float("inf"), float("nan")])
def test_invalid_print_timeout_is_rejected_locally(seconds: float) -> None:
    with pytest.raises(ValueError, match="print_timeout_seconds"):
        AntigravityClient().run_json(
            "hi", options=AntigravityRunOptions(print_timeout_seconds=seconds)
        )


@patch("stan_ai_client.antigravity.execute_command")
def test_default_options_merge_and_per_call_overrides_win(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(simple_stream())
    client = AntigravityClient(
        default_model=None,
        default_effort=None,
        default_options=AntigravityRunOptions(
            agent="default-agent", sandbox=True, permission_mode="plan"
        ),
    )

    client.run_json("hi")
    argv = argv_of(mock_exec)
    assert "--model" not in argv and "--effort" not in argv
    assert flag_value(argv, "--agent") == "default-agent"
    assert "--sandbox" in argv
    assert flag_value(argv, "--mode") == "plan"

    client.run_json(
        "hi", options=AntigravityRunOptions(agent="call-agent", permission_mode="default")
    )
    argv = argv_of(mock_exec)
    assert flag_value(argv, "--agent") == "call-agent"
    assert "--mode" not in argv and "--dangerously-skip-permissions" not in argv


@patch("stan_ai_client.antigravity.execute_command")
def test_resume_flags(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(simple_stream())
    client = AntigravityClient()

    client.run_json("hi", options=AntigravityRunOptions(session_id=CONVERSATION))
    assert flag_value(argv_of(mock_exec), "--conversation") == CONVERSATION

    client.run_json("hi", options=AntigravityRunOptions(continue_last_session=True))
    argv = argv_of(mock_exec)
    assert "--continue" in argv and "--conversation" not in argv

    with pytest.raises(ValueError, match="session_id and continue_last_session"):
        client.run_json(
            "hi",
            options=AntigravityRunOptions(session_id=CONVERSATION, continue_last_session=True),
        )


@patch("stan_ai_client.antigravity.execute_command")
def test_headless_runs_disable_agy_self_update_unless_caller_overrides(
    mock_exec: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGY_CLI_DISABLE_AUTO_UPDATE", raising=False)
    mock_exec.return_value = completed(simple_stream())
    client = AntigravityClient()

    client.run_json("hi", options=AntigravityRunOptions(env={"EXTRA": "1"}))
    env = mock_exec.call_args[0][0].env
    assert env["AGY_CLI_DISABLE_AUTO_UPDATE"] == "1"
    assert env["EXTRA"] == "1"

    client.run_json(
        "hi", options=AntigravityRunOptions(env={"AGY_CLI_DISABLE_AUTO_UPDATE": ""})
    )
    assert mock_exec.call_args[0][0].env["AGY_CLI_DISABLE_AUTO_UPDATE"] == ""


def test_argv_redaction_hides_prompt_session_settings_and_schema() -> None:
    argv = (
        "agy",
        "--conversation",
        CONVERSATION,
        "--settings",
        '{"secret": 1}',
        "--settings={inline}",
        "--json-schema",
        '{"type":"object"}',
        "--print=secret prompt",
    )
    assert _redact_argv(argv) == (
        "agy",
        "--conversation",
        "<redacted>",
        "--settings",
        "<redacted>",
        "--settings=<redacted>",
        "--json-schema",
        "<json-schema>",
        "--print=<prompt>",
    )


@patch("stan_ai_client.antigravity.execute_command")
def test_prompts_are_not_logged_by_default(
    mock_exec: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_exec.return_value = completed(simple_stream())
    with caplog.at_level(logging.DEBUG, logger="stan_ai_client"):
        AntigravityClient().run_text(
            "top secret prompt", options=AntigravityRunOptions(input_mode="argv")
        )
    assert "top secret prompt" not in caplog.text
    assert "--print=<prompt>" in caplog.text


# --- JSON payloads and usage ---------------------------------------------


@patch("stan_ai_client.antigravity.execute_command")
def test_run_json_exposes_result_envelope_events_and_invocation_usage(mock_exec: Mock) -> None:
    stdout = stream(
        init_event(permission_mode="request-review"),
        step(0, "user_input"),
        step(1, "agent_response", state="ACTIVE", text_delta="o"),
        step(1, "agent_response", text_delta="k\n", usage=usage(7000, 30, 20, 5000)),
        result_event(
            response="ok\n",
            usage_counts=usage(7000, 30, 20, 5000),
            json_schema={"type": "object"},
            future_field=True,
        ),
    )
    mock_exec.return_value = completed(stdout)

    result = AntigravityClient().run_json("hi", options=AntigravityRunOptions(model="m1"))

    payload = result.payload
    assert result.stdout == stdout
    assert payload.conversation_id == payload.session_id == CONVERSATION
    assert payload.status == "SUCCESS"
    assert payload.response == "ok\n"
    assert payload.num_turns == 1
    assert payload.duration_seconds == 2.5
    assert payload.duration_ms is not None
    assert payload.permission_mode == "request-review"
    assert payload.requested_model == "m1"
    assert payload.usage_scope == "invocation"
    assert payload.invocation_usage == usage(7000, 30, 20, 5000)
    assert payload.step_usage == usage(7000, 30, 20, 5000)
    assert payload.extras == {"json_schema": {"type": "object"}, "future_field": True}
    assert len(payload.events) == 5
    assert not payload.has_structured_output

    facts = normalize_ai_usage("antigravity", payload)
    assert facts.session_id == CONVERSATION
    # agy's input_tokens exclude cache reads; its total_tokens (7030) omits them.
    assert facts.tokens.fresh_input_tokens == 7000
    assert facts.tokens.cache_read_input_tokens == 5000
    assert facts.tokens.cache_write_input_tokens is None
    assert facts.tokens.output_tokens == 30
    assert facts.tokens.reasoning_output_tokens == 20
    assert facts.tokens.total_tokens == 12030
    assert facts.models[0].model == "m1"
    assert facts.models[0].model_source == "requested"


@patch("stan_ai_client.antigravity.execute_command")
def test_resumed_run_marks_cumulative_usage_and_sums_this_invocations_steps(
    mock_exec: Mock,
) -> None:
    stdout = stream(
        init_event(),
        step(5, "user_input"),
        step(6, "agent_response", text_delta="a", usage=usage(7000, 10, 5)),
        # A repeated DONE report for one step counts once; a subagent step is excluded.
        step(6, "agent_response", text_delta="", usage=usage(7000, 10, 5)),
        step(7, "agent_response", conversation="sub-conv", usage=usage(9999, 9, 9)),
        step(8, "agent_response", text_delta="b\n", usage=usage(7100, 20, 10)),
        result_event(response="a\nb\n", usage_counts=usage(30000, 90, 40), num_turns=3),
    )
    mock_exec.return_value = completed(stdout)

    payload = AntigravityClient().run_json(
        "continue", options=AntigravityRunOptions(session_id=CONVERSATION)
    ).payload

    assert payload.usage_scope == "cumulative"
    assert payload.usage == usage(30000, 90, 40)
    assert payload.invocation_usage == usage(14100, 30, 15)
    facts = normalize_ai_usage("antigravity", payload)
    assert facts.tokens.total_tokens == 14130
    assert "invocation usage summed from this invocation's step events" in facts.diagnostics
    assert facts.raw["usage"] == usage(30000, 90, 40)


@patch("stan_ai_client.antigravity.execute_command")
def test_undecodable_stream_lines_make_step_usage_unknown(mock_exec: Mock) -> None:
    stdout = (
        json.dumps(init_event())
        + "\nnot json\n"
        + json.dumps(result_event())
        + "\n"
    )
    mock_exec.return_value = completed(stdout)

    payload = AntigravityClient().run_json(
        "hi", options=AntigravityRunOptions(session_id=CONVERSATION)
    ).payload

    assert payload.step_usage is None
    assert payload.stream_diagnostics
    facts = normalize_ai_usage("antigravity", payload)
    assert facts.tokens.total_tokens is None
    assert any("undecodable" in item for item in facts.diagnostics)


# --- structured mode -------------------------------------------------------


def structured_stream(
    value: Any = None,
    *,
    finish: bool = True,
    present: bool = True,
    **result_fields: Any,
) -> str:
    events = [init_event(), step(0, "user_input"), step(1, "agent_response", usage=usage())]
    if finish:
        events.append(step(2, "finish"))
    fields: dict[str, Any] = {"json_schema": SCHEMA.schema, **result_fields}
    if present:
        fields["structured_output"] = {"count": 2} if value is None else value
    events.append(result_event(response='{"count":2}\n', **fields))
    return stream(*events)


@patch("stan_ai_client.antigravity.execute_command")
def test_run_structured_validates_and_passes_schema(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(structured_stream())

    result = AntigravityClient().run_structured("count", schema=SCHEMA)

    assert result.structured_output == {"count": 2}
    assert result.payload.has_structured_output
    assert flag_value(argv_of(mock_exec), "--json-schema") == SCHEMA.cli_json


def test_run_structured_rejects_non_object_root_before_running() -> None:
    with patch("stan_ai_client.antigravity.execute_command") as mock_exec:
        with pytest.raises(AntigravitySchemaValidationError):
            AntigravityClient().run_structured(
                "list", schema=StructuredSchema.from_dict({"type": "array"})
            )
        mock_exec.assert_not_called()
    validate_antigravity_output_schema(SCHEMA)


@patch("stan_ai_client.antigravity.execute_command")
def test_missing_structured_output_names_denied_actions(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        structured_stream(
            present=False,
            finish=False,
            denied_actions=[{"action": "command", "display_name": "RunCommand"}],
        )
    )

    with pytest.raises(AntigravityStructuredOutputMissingError) as excinfo:
        AntigravityClient().run_structured(
            "count", schema=SCHEMA, options=AntigravityRunOptions(permission_mode="default")
        )

    assert "RunCommand" in str(excinfo.value)
    assert excinfo.value.payload.response == '{"count":2}\n'


@patch("stan_ai_client.antigravity.execute_command")
def test_structured_output_is_validated_locally(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(structured_stream({"count": 10}))

    with pytest.raises(AntigravityStructuredOutputValidationError) as excinfo:
        AntigravityClient().run_structured("count", schema=SCHEMA)

    assert excinfo.value.structured_output == {"count": 10}


@patch("stan_ai_client.antigravity.execute_command")
def test_resumed_structured_run_rejects_stale_output_from_an_earlier_turn(
    mock_exec: Mock,
) -> None:
    # agy 1.4.12 reports the previous turn's structured_output after this turn's
    # finish call fails schema validation; only a completed finish step is fresh.
    stdout = stream(
        init_event(),
        step(3, "user_input"),
        step(4, "agent_response", text_delta='{"count":10}\n', usage=usage()),
        step(5, "tool", state="ACTIVE", tool_name="finish"),
        step(5, "tool", state="ERROR", tool_name="finish"),
        result_event(structured_output={"count": 2}, num_turns=2),
    )
    mock_exec.return_value = completed(stdout)

    with pytest.raises(AntigravityStructuredOutputMissingError) as excinfo:
        AntigravityClient().run_structured(
            "count 10", schema=SCHEMA, options=AntigravityRunOptions(session_id=CONVERSATION)
        )

    assert "earlier turn" in str(excinfo.value)
    assert excinfo.value.payload.structured_output == {"count": 2}


@patch("stan_ai_client.antigravity.execute_command")
def test_resumed_structured_run_accepts_output_with_a_completed_finish_step(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value = completed(structured_stream({"count": 1}))

    result = AntigravityClient().run_structured(
        "count", schema=SCHEMA, options=AntigravityRunOptions(continue_last_session=True)
    )

    assert result.structured_output == {"count": 1}


# --- agy contract edges ------------------------------------------------------


@patch("stan_ai_client.antigravity.execute_command")
def test_missing_conversation_warning_raises_with_the_new_conversation(mock_exec: Mock) -> None:
    other = "acb5e349-467e-4f18-84b7-a5d474099116"
    mock_exec.return_value = completed(
        stream(init_event(other), result_event(other)),
        stderr=f'warning: conversation "{CONVERSATION}" not found\n',
    )

    with pytest.raises(AntigravitySessionNotFoundError) as excinfo:
        AntigravityClient().run_json(
            "hi", options=AntigravityRunOptions(session_id=CONVERSATION)
        )

    error = excinfo.value
    assert isinstance(error, AntigravityProcessError)
    assert error.returncode == 0
    assert error.payload is not None and error.payload.conversation_id == other
    assert error.conversation_id == other


@patch("stan_ai_client.antigravity.execute_command")
def test_conversation_id_mismatch_alone_is_a_missing_conversation(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(stream(result_event("other-conversation")))

    with pytest.raises(AntigravitySessionNotFoundError):
        AntigravityClient().run_json(
            "hi", options=AntigravityRunOptions(session_id=CONVERSATION)
        )


@patch("stan_ai_client.antigravity.execute_command")
def test_argv_text_mode_detects_missing_conversation_from_stderr(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        "fresh answer\n", stderr='warning: conversation "not-a-uuid" not found\n'
    )

    with pytest.raises(AntigravitySessionNotFoundError):
        AntigravityClient().run_text(
            "hi", options=AntigravityRunOptions(session_id="not-a-uuid", input_mode="argv")
        )


@patch("stan_ai_client.antigravity.execute_command")
def test_auto_mode_denials_raise_approval_required_with_verbatim_response(
    mock_exec: Mock,
) -> None:
    response = "  I will create the file first.\n"
    mock_exec.return_value = completed(
        simple_stream(
            response, denied_actions=[{"action": "command", "display_name": "RunCommand"}]
        )
    )

    with pytest.raises(AntigravityApprovalRequiredError) as excinfo:
        AntigravityClient().run_json("hi", options=AntigravityRunOptions(permission_mode="auto"))

    error = excinfo.value
    assert isinstance(error, ApprovalRequiredError)
    assert isinstance(error, AntigravityProcessError)
    assert error.approval_prompt == response
    assert error.denied_actions == ({"action": "command", "display_name": "RunCommand"},)
    assert error.returncode == 0


@patch("stan_ai_client.antigravity.execute_command")
def test_auto_mode_denial_without_output_relays_agys_notice(
    mock_exec: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_exec.return_value = completed(
        simple_stream("", denied_actions=[{"action": "command", "display_name": "RunCommand"}]),
        stderr=(
            "warning: --mode auto is not available because the terminal sandbox is not "
            "available on this host; continuing in the default mode.\n"
            f"{DENIAL_NOTICE}\n"
        ),
    )

    with caplog.at_level(logging.WARNING, logger="stan_ai_client"):
        with pytest.raises(AntigravityApprovalRequiredError) as excinfo:
            AntigravityClient().run_text(
                "hi", options=AntigravityRunOptions(permission_mode="auto")
            )

    assert excinfo.value.approval_prompt == DENIAL_NOTICE
    assert "auto mode is unavailable on this host" in caplog.text


@patch("stan_ai_client.antigravity.execute_command")
def test_denials_outside_auto_mode_are_returned_on_the_payload(mock_exec: Mock) -> None:
    denied = [{"action": "command", "display_name": "RunCommand"}]
    mock_exec.return_value = completed(simple_stream("", denied_actions=denied))

    payload = AntigravityClient().run_json(
        "hi", options=AntigravityRunOptions(permission_mode="default")
    ).payload

    assert payload.denied_actions == tuple(denied)


@patch("stan_ai_client.antigravity.execute_command")
def test_print_timeout_is_a_timeout_error_with_the_partial_run(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        simple_stream("", finished_reason="print_timeout"),
        stderr="[agy] print timeout after 4s with turn in progress; returning partial output\n",
    )

    with pytest.raises(AntigravityTimeoutError) as excinfo:
        AntigravityClient().run_json(
            "essay", options=AntigravityRunOptions(print_timeout_seconds=4)
        )

    error = excinfo.value
    assert error.timeout_seconds == 4
    assert error.conversation_id == CONVERSATION
    assert error.payload is not None and error.payload.finished_reason == "print_timeout"


@patch("stan_ai_client.antigravity.execute_command")
def test_argv_text_print_timeout_notice_is_a_timeout_error(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        "partial", stderr="[agy] print timeout after 4s with turn in progress; returning partial output\n"
    )

    with pytest.raises(AntigravityTimeoutError) as excinfo:
        AntigravityClient().run_text(
            "essay", options=AntigravityRunOptions(input_mode="argv")
        )

    assert excinfo.value.stdout == "partial"


@pytest.mark.parametrize(
    "fields",
    [
        {"finished_reason": "interrupted"},
        {"finished_reason": "some_future_reason"},
        {"status": "ERROR", "error": "agent failed"},
    ],
)
@patch("stan_ai_client.antigravity.execute_command")
def test_zero_exit_runs_that_ended_early_are_process_errors(
    mock_exec: Mock, fields: dict[str, Any]
) -> None:
    mock_exec.return_value = completed(simple_stream(**fields))

    with pytest.raises(AntigravityProcessError) as excinfo:
        AntigravityClient().run_json("hi")

    assert excinfo.value.returncode == 0
    assert excinfo.value.payload is not None


@patch("stan_ai_client.antigravity.execute_command")
def test_startup_error_envelope_becomes_a_process_error(mock_exec: Mock) -> None:
    message = (
        'invalid model selection (--model "gemini-4-argon" --effort "max"): '
        'gemini-4-argon has no "max" effort (available: low, medium, high)'
    )
    mock_exec.return_value = completed(
        stream(
            result_event(
                "",
                status="ERROR",
                response="",
                error=message,
                usage_counts=usage(0, 0, 0),
                num_turns=0,
                finished_reason="error",
            )
        ),
        stderr=f"error: {message}\n",
        returncode=1,
    )

    with pytest.raises(AntigravityProcessError) as excinfo:
        AntigravityClient().run_json("hi", options=AntigravityRunOptions(effort="max"))

    error = excinfo.value
    assert type(error) is AntigravityProcessError
    assert str(error) == message
    assert error.payload is not None and error.payload.conversation_id is None


@patch("stan_ai_client.antigravity.execute_command")
def test_agent_failure_exposes_the_agy_error_record(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        stream(init_event(), result_event(status="ERROR", error="executor failed")),
        stderr="error: executor failed\n" + agent_error_line(short_error="executor failed") + "\n",
        returncode=3,
    )

    with pytest.raises(AntigravityProcessError) as excinfo:
        AntigravityClient().run_json("hi")

    error = excinfo.value
    # The error ID spells "429"; it must not be read as a rate limit.
    assert type(error) is AntigravityProcessError
    assert error.agent_error is not None
    assert error.agent_error.status == "UNKNOWN"
    assert error.agent_error.error_id == "560d6d86-fbc6-429b-9496-68661f1291e1-1"
    assert error.agent_error.retryable is False
    assert classify_provider_failure(error) is None


@pytest.mark.parametrize(
    "agent_fields,stderr_lines,error_type,signal",
    [
        ({"status": "UNAVAILABLE", "error_code": 503, "code_kind": "http", "retryable": True}, (), AntigravityServerError, "503"),
        ({"status": "INTERNAL", "error_code": 500, "code_kind": "http"}, (), AntigravityServerError, "500"),
        ({"status": "UNAVAILABLE", "error_code": 529, "code_kind": "http"}, (), AntigravityCapacityError, "529"),
        (
            {"status": "RESOURCE_EXHAUSTED", "error_code": 429, "code_kind": "http", "short_error": "MODEL_CAPACITY_EXHAUSTED: no capacity"},
            (),
            AntigravityCapacityError,
            "MODEL_CAPACITY_EXHAUSTED",
        ),
        (None, ("error: The model API is currently overloaded and may experience intermittent errors.",), AntigravityCapacityError, "overloaded"),
    ],
)
@patch("stan_ai_client.antigravity.execute_command")
def test_capacity_and_server_failures_are_classified(
    mock_exec: Mock,
    agent_fields: dict[str, Any] | None,
    stderr_lines: tuple[str, ...],
    error_type: type[AntigravityProcessError],
    signal: str,
) -> None:
    lines = list(stderr_lines)
    if agent_fields is not None:
        lines.append(agent_error_line(**agent_fields))
    mock_exec.return_value = completed(stdout="", stderr="\n".join(lines) + "\n", returncode=3)

    with pytest.raises(error_type) as excinfo:
        AntigravityClient().run_text("hi")

    error = excinfo.value
    assert isinstance(error, (CapacityError, ServerError))
    assert error.signal == signal
    assert classify_provider_failure(error) == error.failure


def test_classify_provider_failure_reads_raw_antigravity_process_errors() -> None:
    from stan_ai_client import CommandMetadata

    error = AntigravityProcessError(
        "failed",
        command=CommandMetadata(("agy",), None, 0),
        returncode=3,
        stdout="",
        stderr=agent_error_line(status="UNAVAILABLE", error_code=502, code_kind="http") + "\n",
        payload=None,
    )

    failure = classify_provider_failure(error)
    assert failure is not None and (failure.kind, failure.signal) == ("server_error", "502")


@pytest.mark.parametrize(
    "agent_fields,stderr_lines",
    [
        ({"status": "RESOURCE_EXHAUSTED", "error_code": 8, "short_error": "You have exhausted your quota on this model."}, ()),
        ({"status": "RESOURCE_EXHAUSTED", "error_code": 429, "code_kind": "http", "short_error": "quota"}, ()),
        (None, ("error: individual quota reached",)),
        (None, ("error: Your AI credits balance is too low to continue.",)),
        (None, ("error: request failed with HTTP 429",)),
    ],
)
@patch("stan_ai_client.antigravity.execute_command")
def test_quota_failures_are_rate_limits(
    mock_exec: Mock, agent_fields: dict[str, Any] | None, stderr_lines: tuple[str, ...]
) -> None:
    lines = list(stderr_lines)
    if agent_fields is not None:
        lines.append(agent_error_line(**agent_fields))
    mock_exec.return_value = completed(stderr="\n".join(lines) + "\n", returncode=3)

    with pytest.raises(AntigravityRateLimitError) as excinfo:
        AntigravityClient().run_text("hi")

    assert isinstance(excinfo.value, RateLimitError)
    assert not isinstance(excinfo.value, (CapacityError, ServerError))


@patch("stan_ai_client.antigravity.execute_command")
def test_rate_limit_reset_hints_are_parsed(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        stderr="error: Rate limit exceeded. Retry after 30 seconds.\n", returncode=3
    )

    with pytest.raises(AntigravityRateLimitError) as excinfo:
        AntigravityClient().run_text("hi")

    assert excinfo.value.retry_after_seconds == 90  # 30s hint plus the shared buffer


@patch("stan_ai_client.antigravity.execute_command")
def test_warnings_and_uuids_are_not_trusted_failure_evidence(mock_exec: Mock) -> None:
    # Only error: lines, AGY_ERROR short errors, and the result error are trusted.
    mock_exec.return_value = completed(
        stderr=(
            'warning: conversation "4f42-429" not found\n'
            "warning: no route to host; rate limit exceeded; MODEL_CAPACITY_EXHAUSTED\n"
            + agent_error_line(error_id="0429-4f42-a429-429") + "\n"
            + "error: something unexpected\n"
        ),
        returncode=1,
    )

    with pytest.raises(AntigravityProcessError) as excinfo:
        AntigravityClient().run_text("hi")

    assert type(excinfo.value) is AntigravityProcessError


@pytest.mark.parametrize(
    "line",
    [
        "error: dial tcp: lookup daily-cloudcode-pa.googleapis.com on 127.0.0.53:53: no such host",
        "error: dial tcp 142.250.1.1:443: connect: network is unreachable",
        "error: lookup oauth2.googleapis.com on 127.0.0.53:53: server misbehaving",
    ],
)
@patch("stan_ai_client.antigravity.execute_command")
def test_network_failures_are_classified(mock_exec: Mock, line: str) -> None:
    mock_exec.return_value = completed(stderr=line + "\n", returncode=3)

    with pytest.raises(AntigravityNetworkUnavailableError) as excinfo:
        AntigravityClient().run_text("hi")

    assert isinstance(excinfo.value, NetworkUnavailableError)
    assert isinstance(excinfo.value, AntigravityCodeError)


@patch("stan_ai_client.antigravity.execute_command")
def test_failed_resume_with_missing_conversation_warning(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        stderr=f'warning: conversation "{CONVERSATION}" not found\nerror: boom\n',
        returncode=1,
    )

    with pytest.raises(AntigravitySessionNotFoundError):
        AntigravityClient().run_text(
            "hi", options=AntigravityRunOptions(session_id=CONVERSATION)
        )


@pytest.mark.parametrize("stdout,message", [("", "empty output"), ("{bad", "without a result")])
@patch("stan_ai_client.antigravity.execute_command")
def test_stream_without_result_is_a_protocol_error(
    mock_exec: Mock, stdout: str, message: str
) -> None:
    mock_exec.return_value = completed(stdout)

    with pytest.raises(AntigravityProtocolError, match=message):
        AntigravityClient().run_json("hi")


@patch("stan_ai_client.antigravity.execute_command")
def test_subprocess_timeout_recovers_the_conversation_from_partial_stream(
    mock_exec: Mock,
) -> None:
    partial = stream(
        init_event(),
        step(0, "user_input"),
        step(1, "agent_response", state="ACTIVE", text_delta="Partial "),
        step(1, "agent_response", state="ACTIVE", text_delta="answer"),
    )
    mock_exec.side_effect = subprocess.TimeoutExpired(
        cmd=["agy"], timeout=5, output=partial.encode(), stderr=b"slow"
    )

    with pytest.raises(AntigravityTimeoutError) as excinfo:
        AntigravityClient().run_json("hi", options=AntigravityRunOptions(timeout_seconds=5))

    error = excinfo.value
    assert error.timeout_seconds == 5
    assert error.stderr == "slow"
    assert error.conversation_id == CONVERSATION
    assert error.payload is not None
    assert error.payload.response == "Partial answer"
    assert error.payload.status is None


@patch("stan_ai_client.antigravity.execute_command")
def test_missing_executable_and_cwd(mock_exec: Mock) -> None:
    mock_exec.side_effect = FileNotFoundError(2, "No such file", "agy")
    with pytest.raises(AntigravityExecutableNotFoundError):
        AntigravityClient().run_text("hi")

    mock_exec.side_effect = FileNotFoundError(2, "No such file", "/missing")
    with pytest.raises(AntigravityProcessError) as excinfo:
        AntigravityClient().run_text("hi", options=AntigravityRunOptions(cwd="/missing"))
    assert excinfo.value.returncode == 127


# --- retry policies ---------------------------------------------------------


@patch("stan_ai_client.provider_retry.retry_delay", return_value=0.0)
@patch("stan_ai_client.antigravity.execute_command")
def test_provider_retry_policy_resends_prefixed_prompt_after_capacity(
    mock_exec: Mock, _delay: Mock
) -> None:
    mock_exec.side_effect = [
        completed(
            stderr=agent_error_line(status="UNAVAILABLE", error_code=529, code_kind="http") + "\n",
            returncode=3,
        ),
        completed(simple_stream()),
    ]

    result = AntigravityClient().run_json(
        "original", provider_retry_policy=ProviderRetryPolicy()
    )

    assert result.payload.response == "ok\n"
    prompts = [
        json.loads(call[0][0].input_text)["message"]["content"]
        for call in mock_exec.call_args_list
    ]
    assert prompts == ["original", PROVIDER_RETRY_PREFIX + "original"]


@patch("stan_ai_client._retry.time.sleep")
@patch("stan_ai_client.antigravity.execute_command")
def test_rate_limit_policy_waits_for_parsed_reset(mock_exec: Mock, mock_sleep: Mock) -> None:
    mock_exec.side_effect = [
        completed(stderr="error: Rate limit exceeded. Retry after 5 seconds.\n", returncode=3),
        completed(simple_stream()),
    ]

    result = AntigravityClient().run_text(
        "hi", rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=600)
    )

    assert result.text == "ok"
    mock_sleep.assert_called_once_with(65.0)


@patch("stan_ai_client.antigravity.execute_command")
def test_print_timeout_placeholder_usage_is_not_reported_as_measured(mock_exec: Mock) -> None:
    # agy 1.4.12 reports all-zero usage for a print timeout with no completed step.
    mock_exec.return_value = completed(
        stream(
            init_event(),
            step(0, "user_input"),
            step(1, "agent_response", state="ACTIVE", text_delta="Bread"),
            result_event(
                response="",
                usage_counts=usage(0, 0, 0),
                finished_reason="print_timeout",
            ),
        ),
        stderr="[agy] print timeout after 4s with turn in progress; returning partial output\n",
    )

    with pytest.raises(AntigravityTimeoutError) as excinfo:
        AntigravityClient().run_json(
            "essay", options=AntigravityRunOptions(print_timeout_seconds=4)
        )

    payload = excinfo.value.payload
    assert payload is not None
    assert payload.usage_scope == "unknown"
    assert payload.invocation_usage == {}
    facts = normalize_ai_usage("antigravity", payload)
    assert facts.tokens.total_tokens is None
    assert "turn ended early; usage covers completed steps only" in facts.diagnostics


# Captured from agy 1.4.12 in the default stdin mode when the network failed
# after 572 characters had streamed: exit 0, an ERROR result, and AGY_ERROR.
LIVE_STREAM_FAILURE_STDERR = (
    'error: agent executor error: generating and executing: request failed: Post '
    '"https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse": '
    "Bad Gateway (response may be truncated)\n"
    'AGY_ERROR: {"short_error":"agent executor error: generating and executing: request '
    'failed: Post \\"https://daily-cloudcode-pa.googleapis.com/v1internal:'
    'streamGenerateContent?alt=sse\\": Bad Gateway","status":"UNKNOWN","error_code":2,'
    '"code_kind":"grpc","retryable":false,"error_id":"704d45a3-7f36-400f-b74d-d894f1b3a2db-3"}\n'
)


@patch("stan_ai_client.antigravity.execute_command")
def test_stdin_mode_failure_after_partial_output_exits_zero_but_raises(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value = completed(
        stream(
            init_event(),
            step(0, "user_input"),
            step(1, "agent_response", state="ACTIVE", text_delta="# Spanning the Chasm"),
            step(1, "agent_response", usage=usage(6895, 100, 0)),
            step(2, "error_message"),
            step(3, "error_message"),
            result_event(
                status="ERROR",
                response="# Spanning the Chasm",
                error=(
                    "agent executor error: generating and executing: request failed: "
                    "Bad Gateway"
                ),
                usage_counts=usage(6895, 100, 0),
                finished_reason="error",
            ),
        ),
        stderr=LIVE_STREAM_FAILURE_STDERR,
        returncode=0,
    )

    with pytest.raises(AntigravityProcessError) as excinfo:
        AntigravityClient().run_text("essay")

    error = excinfo.value
    assert type(error) is AntigravityProcessError
    assert error.returncode == 0
    assert error.agent_error is not None and error.agent_error.code_kind == "grpc"
    assert error.payload is not None
    assert error.payload.response == "# Spanning the Chasm"
    assert error.payload.usage_scope == "unknown"
    assert error.payload.invocation_usage == usage(6895, 100, 0)


@patch("stan_ai_client.antigravity.execute_command")
def test_disk_quota_errors_are_not_rate_limits(mock_exec: Mock) -> None:
    mock_exec.return_value = completed(
        stderr="error: write /home/ops/.gemini/x.db: disk quota exceeded\n", returncode=1
    )

    with pytest.raises(AntigravityProcessError) as excinfo:
        AntigravityClient().run_text("hi")

    assert type(excinfo.value) is AntigravityProcessError
