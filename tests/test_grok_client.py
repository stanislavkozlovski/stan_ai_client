from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

import pytest

from stan_ai_client import (
    GrokCancelledError,
    GrokClient,
    GrokMalformedStructuredOutputError,
    GrokRunOptions,
)
from stan_ai_client.exceptions import (
    GrokExecutableNotFoundError,
    GrokProcessError,
    GrokProtocolError,
    GrokRateLimitError,
    GrokStructuredOutputMissingError,
)
from stan_ai_client.schema import StructuredSchema
from stan_ai_client.types import GrokJsonPayload


def test_grok_client_init_defaults() -> None:
    client = GrokClient()
    assert client.executable == "grok"
    assert client.default_model == "grok-4.5"
    assert client.default_timeout_seconds == 120.0


def test_grok_options_no_input_mode() -> None:
    opts = GrokRunOptions(session_id="sid123")
    assert opts.session_id == "sid123"
    assert not hasattr(opts, "input_mode")


@patch("stan_ai_client.grok.execute_command")
def test_run_text_success(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = "hello world\n"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    result = client.run_text("say hi")
    assert result.text == "hello world"
    argv = mock_exec.call_args[0][0].argv
    assert "grok" in " ".join(argv)
    assert "--no-auto-update" in argv
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "grok-4.5"


@patch("stan_ai_client.grok.execute_command")
def test_session_id_uses_named_session_flag(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = "ok\n"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    client.run_text("say hi", options=GrokRunOptions(session_id="sid123"))

    argv = mock_exec.call_args[0][0].argv
    assert "--session-id" in argv
    assert argv[argv.index("--session-id") + 1] == "sid123"
    assert "--resume" not in argv


@patch("stan_ai_client.grok.execute_command")
def test_policy_rules_are_repeated_flags(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = "ok\n"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    with pytest.warns(DeprecationWarning, match="deprecated permission-rule alias"):
        options = GrokRunOptions(
            allowed_tools=("Bash(git *)", "Read"),
            disallowed_tools=("Write", "Edit"),
        )

    client = GrokClient()
    client.run_text("say hi", options=options)

    argv = mock_exec.call_args[0][0].argv
    assert argv.count("--allow") == 2
    assert argv[argv.index("--allow") + 1] == "Bash(git *)"
    assert argv[argv.index("--allow", argv.index("--allow") + 1) + 1] == "Read"
    assert argv.count("--deny") == 2
    assert argv[argv.index("--deny") + 1] == "Write"
    assert argv[argv.index("--deny", argv.index("--deny") + 1) + 1] == "Edit"


@patch("stan_ai_client.grok.execute_command")
def test_permission_rules_and_tool_inventory_use_distinct_flags(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = "ok\n"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    client.run_text(
        "say hi",
        options=GrokRunOptions(
            permission_allow_rules=("read_file",),
            permission_deny_rules=("run_terminal_cmd",),
            tools=("read_file", "grep", "list_dir"),
            excluded_tools=("web_fetch", "web_search"),
        ),
    )

    argv = mock_exec.call_args[0][0].argv
    assert argv.count("--tools") == 1
    assert argv.count("--disallowed-tools") == 1
    assert argv[argv.index("--allow") + 1] == "read_file"
    assert argv[argv.index("--deny") + 1] == "run_terminal_cmd"
    assert argv[argv.index("--tools") + 1] == "read_file,grep,list_dir"
    assert argv[argv.index("--disallowed-tools") + 1] == "web_fetch,web_search"


def test_deprecated_alias_warning_points_at_the_caller() -> None:
    with pytest.warns(DeprecationWarning) as records:
        GrokRunOptions(allowed_tools=("Read",))

    # Not the dataclass-generated __init__, which callers cannot act on.
    assert records[0].filename == __file__


def test_grok_options_reject_permission_alias_conflicts() -> None:
    with pytest.raises(ValueError, match="both allowed_tools and permission_allow_rules"):
        GrokRunOptions(
            allowed_tools=("legacy",),
            permission_allow_rules=("canonical",),
        )
    with pytest.raises(ValueError, match="both disallowed_tools and permission_deny_rules"):
        GrokRunOptions(
            disallowed_tools=("legacy",),
            permission_deny_rules=("canonical",),
        )


@patch("stan_ai_client.grok.execute_command")
def test_tool_and_permission_options_preserve_default_and_empty_override_semantics(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = "ok\n"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0
    client = GrokClient(
        default_options=GrokRunOptions(
            permission_allow_rules=("read_file",),
            permission_deny_rules=("run_terminal_cmd",),
            tools=("read_file",),
            excluded_tools=("web_fetch",),
        )
    )

    client.run_text("use defaults", options=GrokRunOptions())
    default_argv = mock_exec.call_args[0][0].argv
    assert default_argv[default_argv.index("--allow") + 1] == "read_file"
    assert default_argv[default_argv.index("--deny") + 1] == "run_terminal_cmd"
    assert default_argv[default_argv.index("--tools") + 1] == "read_file"
    assert default_argv[default_argv.index("--disallowed-tools") + 1] == "web_fetch"

    client.run_text(
        "clear defaults",
        options=GrokRunOptions(
            permission_allow_rules=(),
            permission_deny_rules=(),
            tools=(),
            excluded_tools=(),
        ),
    )
    cleared_argv = mock_exec.call_args[0][0].argv
    assert "--allow" not in cleared_argv
    assert "--deny" not in cleared_argv
    assert cleared_argv[cleared_argv.index("--tools") + 1] == ""
    assert cleared_argv[cleared_argv.index("--disallowed-tools") + 1] == ""


@patch("stan_ai_client.grok.execute_command")
def test_add_dirs_do_not_emit_cwd_flags(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = "ok\n"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    client.run_text(
        "say hi",
        options=GrokRunOptions(cwd="/tmp/repo", add_dirs=("/tmp/extra",)),
    )

    prepared = mock_exec.call_args[0][0]
    assert prepared.cwd == "/tmp/repo"
    assert "--cwd" not in prepared.argv


@patch("stan_ai_client.grok.execute_command")
def test_run_json_success(mock_exec: Mock) -> None:
    payload = '{"text": "ok", "stopReason": "EndTurn", "sessionId": "s1", "requestId": "r1"}'
    mock_exec.return_value.stdout = payload
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    res = client.run_json("test")
    assert isinstance(res.payload, GrokJsonPayload)
    assert res.payload.text == "ok"
    assert res.payload.session_id == "s1"
    assert isinstance(res.payload.duration_ms, int)


@patch("stan_ai_client.grok.execute_command")
def test_run_json_error_envelope_raises_process_error(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"type": "error", "message": "boom"}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    with pytest.raises(GrokProcessError) as exc:
        client.run_json("test")

    assert "boom" in str(exc.value)
    assert exc.value.returncode == 0
    assert exc.value.payload is not None
    assert exc.value.payload.extras["type"] == "error"


@patch("stan_ai_client.grok.execute_command")
def test_run_text_error_envelope_raises_process_error(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"type": "error", "message": "boom"}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    with pytest.raises(GrokProcessError) as exc:
        client.run_text("test")

    assert "boom" in str(exc.value)
    assert exc.value.returncode == 0
    assert exc.value.payload is not None
    assert exc.value.payload.extras["type"] == "error"


@patch("stan_ai_client.grok.execute_command")
def test_run_structured(mock_exec: Mock) -> None:
    payload = '{"text": "{\\"ans\\":42}", "stopReason": "EndTurn", "structuredOutput": {"ans": 42}}'
    mock_exec.return_value.stdout = payload
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"ans": {"type": "integer"}},
            "required": ["ans"],
            "additionalProperties": False,
        }
    )
    client = GrokClient()
    res = client.run_structured("return ans", schema=schema)
    assert res.structured_output == {"ans": 42}


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_recovers_one_json_value_from_text_when_output_is_null(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = json.dumps(
        {
            "text": '{"ans": 42}',
            "stopReason": "EndTurn",
            "sessionId": "s1",
            "structuredOutput": None,
        }
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"ans": {"type": "integer"}},
            "required": ["ans"],
            "additionalProperties": False,
        }
    )
    result = GrokClient().run_structured("return ans", schema=schema)

    assert result.structured_output == {"ans": 42}
    assert result.payload.session_id == "s1"


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_classifies_concatenated_text_before_schema_validation(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = json.dumps(
        {
            "text": '{"ans": 1}{"ans": 2}{"ans": 3}{"ans": 4}',
            "stopReason": "EndTurn",
            "sessionId": "s1",
            "requestId": "r1",
            "structuredOutput": None,
        }
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {"type": "object", "properties": {"ans": {"type": "integer"}}}
    )
    with pytest.raises(GrokMalformedStructuredOutputError) as exc:
        GrokClient().run_structured("return ans", schema=schema)

    assert exc.value.json_value_count == 4
    assert "4 concatenated top-level JSON values" in exc.value.detail
    assert exc.value.session_id == "s1"
    assert exc.value.request_id == "r1"
    assert exc.value.stop_reason == "EndTurn"
    assert exc.value.payload.duration_ms is not None


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_classifies_concatenated_direct_stdout(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"ans": 1}{"ans": 2}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {"type": "object", "properties": {"ans": {"type": "integer"}}}
    )
    with pytest.raises(GrokMalformedStructuredOutputError) as exc:
        GrokClient().run_structured("return ans", schema=schema)

    assert exc.value.json_value_count == 2
    assert "2 concatenated top-level JSON values" in exc.value.detail


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_classifies_truncated_direct_stdout(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"ans":'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {"type": "object", "properties": {"ans": {"type": "integer"}}}
    )
    with pytest.raises(GrokMalformedStructuredOutputError) as exc:
        GrokClient().run_structured("return ans", schema=schema)

    assert exc.value.json_value_count == 0
    assert "before completing a top-level value" in exc.value.detail


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_treats_non_ascii_digit_output_as_non_json(mock_exec: Mock) -> None:
    # "²".isdigit() is True, but a JSON number can only start with an ASCII digit
    # or "-", so this output never was JSON rather than being malformed JSON.
    mock_exec.return_value.stdout = "²3 tokens remaining"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {"type": "object", "properties": {"ans": {"type": "integer"}}}
    )
    with pytest.raises(GrokProtocolError) as exc:
        GrokClient().run_structured("return ans", schema=schema)

    assert not isinstance(exc.value, GrokMalformedStructuredOutputError)
    assert "non-JSON output" in str(exc.value)


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_classifies_truncated_envelope_text(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = json.dumps(
        {
            "text": '{"ans":',
            "stopReason": "EndTurn",
            "sessionId": "s1",
            "structuredOutput": None,
        }
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {"type": "object", "properties": {"ans": {"type": "integer"}}}
    )
    with pytest.raises(GrokMalformedStructuredOutputError) as exc:
        GrokClient().run_structured("return ans", schema=schema)

    assert exc.value.json_value_count == 0
    assert exc.value.session_id == "s1"
    assert "before completing a top-level value" in exc.value.detail


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_classifies_end_turn_with_null_output_as_missing(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = json.dumps(
        {
            "text": "",
            "stopReason": "EndTurn",
            "sessionId": "s1",
            "structuredOutput": None,
        }
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    with pytest.raises(GrokStructuredOutputMissingError) as exc:
        GrokClient().run_structured("return object", schema=schema)

    assert exc.value.session_id == "s1"
    assert exc.value.stop_reason == "EndTurn"
    assert exc.value.payload.structured_output is None


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_classifies_permission_cancellation_before_validation(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = json.dumps(
        {
            "text": "",
            "stopReason": "Cancelled",
            "sessionId": "s1",
            "requestId": "r1",
            "cancellationCategory": "permission_cancelled",
            "structuredOutput": None,
        }
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    with pytest.raises(GrokCancelledError) as exc:
        GrokClient().run_structured("return object", schema=schema)

    assert exc.value.session_id == "s1"
    assert exc.value.request_id == "r1"
    assert exc.value.stop_reason == "Cancelled"
    assert exc.value.cancellation_category == "permission_cancelled"
    assert exc.value.payload.duration_ms is not None
    assert "permission_cancelled" in str(exc.value)
    assert exc.value.returncode == 0
    assert isinstance(exc.value, GrokProcessError)
    assert not isinstance(exc.value, GrokProtocolError)


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_cancellation_wins_over_partial_structured_output(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = json.dumps(
        {
            "text": '{"ans": 42}',
            "stopReason": "Cancelled",
            "sessionId": "s1",
            "structuredOutput": {"ans": 42},
        }
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {"type": "object", "properties": {"ans": {"type": "integer"}}}
    )
    with pytest.raises(GrokCancelledError):
        GrokClient().run_structured("return ans", schema=schema)


@pytest.mark.parametrize(
    "raw_value",
    [
        {"stopReason": "Cancelled", "structuredOutput": None},
        {"text": "domain state", "stopReason": "Cancelled", "structuredOutput": None},
    ],
)
@patch("stan_ai_client.grok.execute_command")
def test_raw_schema_with_cancelled_stop_reason_is_not_control_metadata(
    mock_exec: Mock,
    raw_value: dict[str, object],
) -> None:
    mock_exec.return_value.stdout = json.dumps(raw_value)
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "stopReason": {"const": "Cancelled"},
                "structuredOutput": {"type": "null"},
            },
            "required": ["stopReason", "structuredOutput"],
            "additionalProperties": False,
        }
    )
    result = GrokClient().run_structured("return domain state", schema=schema)

    assert result.structured_output == raw_value


@pytest.mark.parametrize("identifier", ["sessionId", "requestId"])
@patch("stan_ai_client.grok.execute_command")
def test_raw_schema_with_cancelled_stop_reason_and_identifier_is_preserved(
    mock_exec: Mock,
    identifier: str,
) -> None:
    raw_value = {"stopReason": "Cancelled", identifier: "domain-id"}
    mock_exec.return_value.stdout = json.dumps(raw_value)
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "stopReason": {"const": "Cancelled"},
                identifier: {"type": "string"},
            },
            "required": ["stopReason", identifier],
            "additionalProperties": False,
        }
    )
    result = GrokClient().run_structured("return domain state", schema=schema)

    assert result.structured_output == raw_value


@patch("stan_ai_client.grok.execute_command")
def test_raw_schema_with_non_string_stop_reason_is_preserved(mock_exec: Mock) -> None:
    raw_value = {"stopReason": 123}
    mock_exec.return_value.stdout = json.dumps(raw_value)
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"stopReason": {"type": "integer"}},
            "required": ["stopReason"],
            "additionalProperties": False,
        }
    )
    result = GrokClient().run_structured("return numeric state", schema=schema)

    assert result.structured_output == raw_value


@patch("stan_ai_client.grok.execute_command")
def test_run_json_classifies_cancelled_envelope(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = json.dumps(
        {
            "text": "",
            "stopReason": "Cancelled",
            "sessionId": "s1",
            "cancellationCategory": "permission_cancelled",
        }
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    with pytest.raises(GrokCancelledError) as exc:
        GrokClient().run_json("test")

    assert exc.value.session_id == "s1"
    assert exc.value.cancellation_category == "permission_cancelled"


@patch("stan_ai_client.grok.execute_command")
def test_malformed_protocol_error_keeps_raw_stdout_out_of_message(mock_exec: Mock) -> None:
    raw_stdout = "sensitive non-json model output"
    mock_exec.return_value.stdout = raw_stdout
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    with pytest.raises(GrokProtocolError) as exc:
        GrokClient().run_structured("return object", schema=schema)

    assert exc.value.stdout == raw_stdout
    assert raw_stdout not in str(exc.value)
    assert "captured on the exception" in str(exc.value)


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_extracts_envelope_before_permissive_raw_schema(
    mock_exec: Mock,
) -> None:
    payload = (
        '{"text": "{\\"ans\\":42}", "stopReason": "EndTurn", '
        '"sessionId": "s1", "structuredOutput": {"ans": 42}}'
    )
    mock_exec.return_value.stdout = payload
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    client = GrokClient()
    res = client.run_structured("return ans", schema=schema)
    assert res.structured_output == {"ans": 42}
    assert res.payload.text == '{"ans":42}'
    assert res.payload.session_id == "s1"


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_falls_back_to_raw_on_envelope_schema_mismatch(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = (
        '{"stopReason": "domain-state", "sessionId": "domain-session", '
        '"structuredOutput": "ok"}'
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "stopReason": {"type": "string"},
                "sessionId": {"type": "string"},
                "structuredOutput": {"type": "string"},
            },
            "required": ["stopReason", "sessionId", "structuredOutput"],
            "additionalProperties": False,
        }
    )
    client = GrokClient()
    res = client.run_structured("return envelope-like fields", schema=schema)
    assert res.structured_output == {
        "stopReason": "domain-state",
        "sessionId": "domain-session",
        "structuredOutput": "ok",
    }
    assert res.payload.structured_output == res.structured_output


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_accepts_falsy_output(mock_exec: Mock) -> None:
    payload = '{"text": "{}", "stopReason": "EndTurn", "structuredOutput": {}}'
    mock_exec.return_value.stdout = payload
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object", "properties": {}, "additionalProperties": False}
    )
    client = GrokClient()
    res = client.run_structured("return empty object", schema=schema)
    assert res.structured_output == {}


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_accepts_raw_schema_object(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"ans": 42}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, int]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"ans": {"type": "integer"}},
            "required": ["ans"],
            "additionalProperties": False,
        }
    )
    client = GrokClient()
    res = client.run_structured("return ans", schema=schema)
    assert res.structured_output == {"ans": 42}
    assert res.payload.has_structured_output is True
    assert res.payload.structured_output == {"ans": 42}


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_rejects_error_envelope_before_permissive_validation(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = '{"type": "error", "message": "domain error"}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    client = GrokClient()
    with pytest.raises(GrokProcessError) as exc:
        client.run_structured("return domain error", schema=schema)

    assert exc.value.payload is not None
    assert exc.value.payload.extras["type"] == "error"


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_rejects_envelope_without_structured_output(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = (
        '{"text": "plain text", "stopReason": "EndTurn", "sessionId": "s1"}'
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    client = GrokClient()
    with pytest.raises(GrokStructuredOutputMissingError) as exc:
        client.run_structured("return object", schema=schema)

    assert exc.value.payload is not None
    assert exc.value.payload.session_id == "s1"


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_accepts_raw_text_with_metadata_like_keys(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"text": "desc", "sessionId": "s1"}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "sessionId": {"type": "string"},
            },
            "required": ["text", "sessionId"],
            "additionalProperties": False,
        }
    )
    client = GrokClient()
    res = client.run_structured("return text and session id", schema=schema)

    assert res.structured_output == {"text": "desc", "sessionId": "s1"}


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_rejects_structured_output_error_envelope(
    mock_exec: Mock,
) -> None:
    mock_exec.return_value.stdout = (
        '{"structuredOutputError": "schema failed", "text": "plain text"}'
    )
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    client = GrokClient()
    with pytest.raises(GrokStructuredOutputMissingError) as exc:
        client.run_structured("return object", schema=schema)

    assert exc.value.payload is not None
    assert "structuredOutputError" in exc.value.payload.extras


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_accepts_raw_metadata_like_keys(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"sessionId": "abc", "requestId": "req1"}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "sessionId": {"type": "string"},
                "requestId": {"type": "string"},
            },
            "required": ["sessionId", "requestId"],
            "additionalProperties": False,
        }
    )
    client = GrokClient()
    res = client.run_structured("return metadata-like fields", schema=schema)

    assert res.structured_output == {"sessionId": "abc", "requestId": "req1"}


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_preserves_raw_structured_output_key(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"structuredOutput": "ok"}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"structuredOutput": {"type": "string"}},
            "required": ["structuredOutput"],
            "additionalProperties": False,
        }
    )
    client = GrokClient()
    res = client.run_structured("return structuredOutput", schema=schema)
    assert res.structured_output == {"structuredOutput": "ok"}
    assert res.payload.structured_output == {"structuredOutput": "ok"}


@patch("stan_ai_client.grok.execute_command")
def test_run_structured_preserves_raw_envelope_like_keys(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = '{"text": "desc", "structuredOutput": "ok"}'
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "structuredOutput": {"type": "string"},
            },
            "required": ["text", "structuredOutput"],
            "additionalProperties": False,
        }
    )
    client = GrokClient()
    res = client.run_structured("return envelope-like fields", schema=schema)
    assert res.structured_output == {"text": "desc", "structuredOutput": "ok"}
    assert res.payload.structured_output == {"text": "desc", "structuredOutput": "ok"}


@pytest.mark.parametrize("text", ['{"ans": 1}', "plain text"])
@patch("stan_ai_client.grok.execute_command")
def test_run_structured_preserves_raw_text_only_value_with_null_output(
    mock_exec: Mock,
    text: str,
) -> None:
    raw_value = {"text": text, "structuredOutput": None}
    mock_exec.return_value.stdout = json.dumps(raw_value)
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    schema: StructuredSchema[dict[str, object]] = StructuredSchema.from_dict(
        {"type": "object"}
    )
    result = GrokClient().run_structured("return raw value", schema=schema)

    assert result.structured_output == raw_value
    assert result.payload.structured_output == raw_value


@patch("stan_ai_client.grok.execute_command")
def test_error_raises_process_error(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = ""
    mock_exec.return_value.stderr = "Error: something bad"
    mock_exec.return_value.returncode = 1

    client = GrokClient()
    with pytest.raises(GrokProcessError) as exc:
        client.run_text("fail")
    assert "something bad" in str(exc.value)


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_EXHAUSTED quota exceeded",
        "Resource exhausted quota exceeded",
        "resource-exhausted quota exceeded",
    ],
)
@patch("stan_ai_client.grok.execute_command")
def test_resource_exhausted_raises_rate_limit_error(
    mock_exec: Mock,
    message: str,
) -> None:
    mock_exec.return_value.stdout = ""
    mock_exec.return_value.stderr = message
    mock_exec.return_value.returncode = 1

    client = GrokClient()
    with pytest.raises(GrokRateLimitError):
        client.run_text("fail")


@patch("stan_ai_client.grok.execute_command")
def test_missing_cwd_is_process_error(mock_exec: Mock) -> None:
    missing_cwd = "/tmp/missing-grok-cwd"
    mock_exec.side_effect = FileNotFoundError(2, "No such file or directory", missing_cwd)

    client = GrokClient()
    with pytest.raises(GrokProcessError) as exc:
        client.run_text("hi", options=GrokRunOptions(cwd=missing_cwd))

    assert exc.value.returncode == 127
    assert exc.value.command.cwd == missing_cwd
    assert "working directory" in str(exc.value)


def test_executable_not_found() -> None:
    client = GrokClient(executable="nonexistent-grok-binary-xyz")
    with pytest.raises(GrokExecutableNotFoundError):
        client.run_text("hi")


@patch("stan_ai_client.grok.execute_command")
def test_long_prompt_uses_file(mock_exec: Mock) -> None:
    long_prompt = "x" * 5000
    mock_exec.return_value.stdout = "done"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    client.run_text(long_prompt)
    argv = mock_exec.call_args[0][0].argv
    assert "-p" not in argv
    assert "--prompt-file" in argv
    assert argv[argv.index("--prompt-file") + 1].endswith(".txt")


@patch("stan_ai_client.grok.execute_command")
def test_long_prompt_temp_files_are_per_prepared_command(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = "done"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    prepared_one, _ = client._prepare(
        "x" * 5000,
        output_format="plain",
        options=None,
    )
    prepared_two, _ = client._prepare(
        "y" * 5000,
        output_format="plain",
        options=None,
    )
    path_one = prepared_one.prompt_file_path
    path_two = prepared_two.prompt_file_path

    try:
        assert path_one is not None
        assert path_two is not None
        assert path_one != path_two
        assert os.path.exists(path_one)
        assert os.path.exists(path_two)

        client._execute(prepared_one)

        assert not os.path.exists(path_one)
        assert os.path.exists(path_two)
    finally:
        client._cleanup_tmp(path_one)
        client._cleanup_tmp(path_two)
