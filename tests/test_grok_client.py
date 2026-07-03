from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from stan_ai_client import GrokClient, GrokRunOptions
from stan_ai_client.exceptions import (
    GrokExecutableNotFoundError,
    GrokProcessError,
)
from stan_ai_client.schema import StructuredSchema
from stan_ai_client.types import GrokJsonPayload


def test_grok_client_init_defaults() -> None:
    client = GrokClient()
    assert client.executable == "grok"
    assert client.default_model == "grok-build"
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
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "grok-build"


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

    client = GrokClient()
    client.run_text(
        "say hi",
        options=GrokRunOptions(
            allowed_tools=("Bash(git *)", "Read"),
            disallowed_tools=("Write", "Edit"),
        ),
    )

    argv = mock_exec.call_args[0][0].argv
    assert argv.count("--allow") == 2
    assert argv[argv.index("--allow") + 1] == "Bash(git *)"
    assert argv[argv.index("--allow", argv.index("--allow") + 1) + 1] == "Read"
    assert argv.count("--deny") == 2
    assert argv[argv.index("--deny") + 1] == "Write"
    assert argv[argv.index("--deny", argv.index("--deny") + 1) + 1] == "Edit"


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
def test_error_raises_process_error(mock_exec: Mock) -> None:
    mock_exec.return_value.stdout = ""
    mock_exec.return_value.stderr = "Error: something bad"
    mock_exec.return_value.returncode = 1

    client = GrokClient()
    with pytest.raises(GrokProcessError) as exc:
        client.run_text("fail")
    assert "something bad" in str(exc.value)


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
