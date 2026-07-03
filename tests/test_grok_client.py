from __future__ import annotations

from unittest.mock import patch

import pytest

from stan_ai_client import GrokClient, GrokRunOptions
from stan_ai_client.exceptions import (
    GrokExecutableNotFoundError,
    GrokProcessError,
)
from stan_ai_client.types import GrokJsonPayload


def test_grok_client_init_defaults():
    client = GrokClient()
    assert client.executable == "grok"
    assert client.default_model == "grok-build"
    assert client.default_timeout_seconds == 120.0


def test_grok_options_no_input_mode():
    opts = GrokRunOptions(session_id="sid123")
    assert opts.session_id == "sid123"
    # input_mode should not be an attribute
    assert not hasattr(opts, "input_mode") or opts.input_mode is None  # type: ignore[attr-defined]


@patch("stan_ai_client.grok.execute_command")
def test_run_text_success(mock_exec):
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
def test_run_json_success(mock_exec):
    payload = '{"text": "ok", "stopReason": "EndTurn", "sessionId": "s1", "requestId": "r1"}'
    mock_exec.return_value.stdout = payload
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    res = client.run_json("test")
    assert isinstance(res.payload, GrokJsonPayload)
    assert res.payload.text == "ok"
    assert res.payload.session_id == "s1"


@patch("stan_ai_client.grok.execute_command")
def test_run_structured(mock_exec):
    payload = '{"text": "{\\"ans\\":42}", "stopReason": "EndTurn", "structuredOutput": {"ans": 42}}'
    mock_exec.return_value.stdout = payload
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    from stan_ai_client.schema import StructuredSchema

    schema = StructuredSchema.from_dict(
        {"type": "object", "properties": {"ans": {"type": "integer"}}, "required": ["ans"], "additionalProperties": False}
    )
    client = GrokClient()
    res = client.run_structured("return ans", schema=schema)
    assert res.structured_output == {"ans": 42}


@patch("stan_ai_client.grok.execute_command")
def test_error_raises_process_error(mock_exec):
    mock_exec.return_value.stdout = ""
    mock_exec.return_value.stderr = "Error: something bad"
    mock_exec.return_value.returncode = 1

    client = GrokClient()
    with pytest.raises(GrokProcessError) as exc:
        client.run_text("fail")
    assert "something bad" in str(exc.value)


def test_executable_not_found():
    client = GrokClient(executable="nonexistent-grok-binary-xyz")
    with pytest.raises(GrokExecutableNotFoundError):
        client.run_text("hi")


@patch("stan_ai_client.grok.execute_command")
def test_long_prompt_uses_file(mock_exec):
    long_prompt = "x" * 5000
    mock_exec.return_value.stdout = "done"
    mock_exec.return_value.stderr = ""
    mock_exec.return_value.returncode = 0

    client = GrokClient()
    client.run_text(long_prompt)
    argv = mock_exec.call_args[0][0].argv
    assert "--prompt-file" in argv
