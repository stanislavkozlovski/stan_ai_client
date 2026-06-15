from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from typing import Any

import pytest

from stan_ai_client import (
    ClaudeCodeClient,
    CommandMetadata,
    ClaudeExecutableNotFoundError,
    ClaudeLimitError,
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeRateLimitError,
    ClaudeStructuredOutputMissingError,
    ClaudeStructuredOutputValidationError,
    RateLimitInfo,
    RateLimitRetryPolicy,
    RunOptions,
    StructuredRunResult,
    StructuredSchema,
)


class RunRecorder:
    def __init__(
        self,
        completed: subprocess.CompletedProcess[str] | list[subprocess.CompletedProcess[str]],
    ) -> None:
        self.completed = completed if isinstance(completed, list) else [completed]
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: str | None,
        text: bool,
        capture_output: bool,
        timeout: float,
        input: str | None,
        env: Mapping[str, str] | None,
    ) -> subprocess.CompletedProcess[str]:
        call_index = len(self.calls)
        self.calls.append(
            {
                "argv": tuple(argv),
                "cwd": cwd,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
                "input": input,
                "env": env,
            }
        )
        return self.completed[min(call_index, len(self.completed) - 1)]


def test_run_json_uses_stdin_and_parses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"result":"ok","session_id":"sess-1","total_cost_usd":0.12}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient(default_model="claude-opus-4-6", default_effort="max")
    result = client.run_json("hello")

    assert result.payload.result == "ok"
    assert result.payload.session_id == "sess-1"
    assert result.payload.total_cost_usd == 0.12
    assert recorder.calls[0]["input"] == "hello"
    assert "hello" not in recorder.calls[0]["argv"]
    assert recorder.calls[0]["argv"][:2] == ("claude", "-p")
    assert "--output-format" in recorder.calls[0]["argv"]
    assert "json" in recorder.calls[0]["argv"]


def test_run_text_can_use_argv_and_extra_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="tagged\n",
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient(
        default_options=RunOptions(
            allowed_tools=("Read",),
            timeout_seconds=90,
        )
    )
    result = client.run_text(
        "tag this",
        options=RunOptions(
            cwd="/tmp/article",
            input_mode="argv",
            allowed_tools=(),
            extra_args=("--debug",),
        ),
    )

    assert result.text == "tagged"
    assert recorder.calls[0]["cwd"] == "/tmp/article"
    assert recorder.calls[0]["input"] is None
    assert recorder.calls[0]["timeout"] == 90
    assert recorder.calls[0]["argv"][-1] == "tag this"
    assert "--debug" in recorder.calls[0]["argv"]
    allowed_index = recorder.calls[0]["argv"].index("--allowed-tools")
    assert recorder.calls[0]["argv"][allowed_index + 1] == ""


def test_run_json_raises_protocol_error_on_non_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="plain text",
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    with pytest.raises(ClaudeProtocolError):
        client.run_json("hello")


def test_run_json_raises_process_error_with_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"is_error": true, "result": "permission denied"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    with pytest.raises(ClaudeProcessError) as excinfo:
        client.run_json("hello")

    assert "permission denied" in str(excinfo.value)
    assert excinfo.value.payload is not None
    assert excinfo.value.payload.result == "permission denied"


def test_run_json_raises_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"is_error": true, "result": "Rate limit exceeded, retry after 3600"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    with pytest.raises(ClaudeRateLimitError) as excinfo:
        client.run_json("hello")

    assert excinfo.value.rate_limit.retry_after_seconds == 3660


def test_run_json_raises_limit_error_for_hit_your_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"is_error": true, "result": "You\'ve hit your limit · resets 1am (Europe/Sofia)"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    with pytest.raises(ClaudeLimitError) as excinfo:
        client.run_json("hello")

    assert isinstance(excinfo.value, ClaudeRateLimitError)
    assert excinfo.value.reset_at is not None
    assert excinfo.value.rate_limit.reset_at == excinfo.value.reset_at


def test_rate_limit_policy_retries_json_after_parsed_wait(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout='{"is_error": true, "result": "Rate limit exceeded, retry after 3"}',
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"result": "ok"}',
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.rate_limit_retry")
    caplog.set_level(logging.WARNING, logger=logger.name)
    client = ClaudeCodeClient(logger=logger)
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.client.time.sleep", sleeps.append)

    result = client.run_json(
        "hello",
        rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=65, label="json test"),
    )

    assert result.payload.result == "ok"
    assert len(recorder.calls) == 2
    assert sleeps == [63.0]
    assert "Claude rate limited; retrying after reset" in caplog.text
    assert "wait_seconds=63.0" in caplog.text
    assert "label=json test" in caplog.text


def test_rate_limit_policy_refuses_json_wait_over_budget(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"is_error": true, "result": "Rate limit exceeded, retry after 10"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.rate_limit_over_budget")
    caplog.set_level(logging.WARNING, logger=logger.name)
    client = ClaudeCodeClient(logger=logger)
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.client.time.sleep", sleeps.append)

    with pytest.raises(ClaudeRateLimitError) as excinfo:
        client.run_json(
            "hello",
            rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=5, label="json test"),
        )

    assert excinfo.value.retry_after_seconds == 70
    assert len(recorder.calls) == 1
    assert sleeps == []
    assert "Claude rate limit exceeds wait budget" in caplog.text
    assert "remaining_wait_seconds=5.0" in caplog.text


def test_rate_limit_policy_refuses_json_without_retry_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"is_error": true, "result": "Rate limit exceeded"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.rate_limit_missing_metadata")
    caplog.set_level(logging.WARNING, logger=logger.name)
    client = ClaudeCodeClient(logger=logger)
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.client.time.sleep", sleeps.append)

    with pytest.raises(ClaudeRateLimitError) as excinfo:
        client.run_json(
            "hello",
            rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=None),
        )

    assert excinfo.value.retry_after_seconds is None
    assert len(recorder.calls) == 1
    assert sleeps == []
    assert "no retry metadata was parsed" in caplog.text


def test_rate_limit_policy_refuses_non_positive_retry_wait(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("stan_ai_client.tests.rate_limit_zero_wait")
    caplog.set_level(logging.WARNING, logger=logger.name)
    client = ClaudeCodeClient(logger=logger)
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.client.time.sleep", sleeps.append)
    error = ClaudeRateLimitError(
        "rate limited",
        command=CommandMetadata(argv=("claude",), cwd=None, elapsed_ms=0.0),
        returncode=1,
        stdout="",
        stderr="",
        payload=None,
        rate_limit=RateLimitInfo(
            message="rate limited",
            retry_after_seconds=0,
            reset_at=None,
        ),
    )

    def operation() -> None:
        raise error

    with pytest.raises(ClaudeRateLimitError) as excinfo:
        client._run_with_rate_limit_policy(
            operation,
            rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=0),
        )

    assert excinfo.value is error
    assert sleeps == []
    assert "non-positive retry wait" in caplog.text


def test_rate_limit_policy_uses_cumulative_wait_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout='{"is_error": true, "result": "Rate limit exceeded, retry after 4"}',
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout='{"is_error": true, "result": "Rate limit exceeded, retry after 4"}',
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.client.time.sleep", sleeps.append)

    with pytest.raises(ClaudeRateLimitError):
        client.run_json(
            "hello",
            rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=100),
        )

    assert len(recorder.calls) == 2
    assert sleeps == [64.0]


def test_rate_limit_policy_retries_text_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout='{"is_error": true, "result": "Rate limit exceeded, retry after 2"}',
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="done\n",
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.client.time.sleep", sleeps.append)

    result = client.run_text(
        "hello",
        rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=62),
    )

    assert result.text == "done"
    assert len(recorder.calls) == 2
    assert sleeps == [62.0]


def test_rate_limit_policy_retries_structured_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout='{"is_error": true, "result": "Rate limit exceeded, retry after 1"}',
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"result": "ok", "structured_output": {"summary": "brief"}}',
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.client.time.sleep", sleeps.append)
    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
    )

    result = client.run_structured(
        "hello",
        schema=schema,
        rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=61),
    )

    assert result.structured_output == {"summary": "brief"}
    assert len(recorder.calls) == 2
    assert sleeps == [61.0]


def test_rate_limit_policy_validates_max_wait_seconds() -> None:
    with pytest.raises(ValueError, match="max_wait_seconds"):
        RateLimitRetryPolicy(max_wait_seconds=-1)


def test_missing_executable_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("claude not found")

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", raise_not_found)

    client = ClaudeCodeClient(executable="claude")
    with pytest.raises(ClaudeExecutableNotFoundError):
        client.run_text("hello")


def test_logging_hides_prompt_text_by_default(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"result":"ok","session_id":"sess-1","total_cost_usd":0.12}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.default_logging")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    client = ClaudeCodeClient(logger=logger)
    client.run_json("super secret prompt")

    assert "Claude run starting" in caplog.text
    assert "Claude run finished" in caplog.text
    assert "prompt_chars=19" in caplog.text
    assert "super secret prompt" not in caplog.text
    assert "<prompt>" not in caplog.text


def test_logging_can_include_prompt_text_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok",
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.prompt_logging")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    client = ClaudeCodeClient(logger=logger, log_prompts=True)
    client.run_text("super secret prompt")

    assert "Claude prompt=super secret prompt" in caplog.text


def test_logging_redacts_json_schema_in_debug_argv(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"result":"ok","structured_output":{"summary":"brief"}}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.schema_logging")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        }
    )

    client = ClaudeCodeClient(logger=logger)
    client.run_structured("summarize this", schema=schema)

    assert "--json-schema" in caplog.text
    assert "<json-schema>" in caplog.text
    assert schema.cli_json not in caplog.text


def test_run_structured_passes_schema_and_returns_validated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"result":"ok","session_id":"sess-1","total_cost_usd":0.12,'
                '"usage":{"input_tokens":10},"structured_output":{"summary":"brief"}}'
            ),
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        }
    )

    client = ClaudeCodeClient(default_model="claude-opus-4-6", default_effort="max")
    result = client.run_structured("summarize this", schema=schema)

    assert result.structured_output == {"summary": "brief"}
    assert result.payload.structured_output == {"summary": "brief"}
    assert result.payload.session_id == "sess-1"
    assert result.payload.total_cost_usd == 0.12
    assert result.payload.usage == {"input_tokens": 10}
    assert recorder.calls[0]["input"] == "summarize this"
    schema_index = recorder.calls[0]["argv"].index("--json-schema")
    assert recorder.calls[0]["argv"][schema_index + 1] == schema.cli_json


def test_run_structured_accepts_null_structured_output_when_schema_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"result":"ok","structured_output":null}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    null_schema: StructuredSchema[None] = StructuredSchema.from_dict({"type": "null"})
    result: StructuredRunResult[None] = client.run_structured(
        "return null",
        schema=null_schema,
    )

    assert result.structured_output is None
    assert result.payload.has_structured_output is True


def test_run_structured_raises_when_structured_output_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"result":"ok","session_id":"sess-1"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    with pytest.raises(ClaudeStructuredOutputMissingError) as excinfo:
        client.run_structured(
            "summarize this",
            schema=StructuredSchema.from_dict({"type": "object"}),
        )

    assert excinfo.value.payload.session_id == "sess-1"


def test_run_structured_raises_when_structured_output_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"result":"ok","structured_output":{"summary":1}}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    with pytest.raises(ClaudeStructuredOutputValidationError) as excinfo:
        client.run_structured(
            "summarize this",
            schema=StructuredSchema.from_dict(
                {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                }
            ),
        )

    assert excinfo.value.structured_output == {"summary": 1}
    assert "does not match the schema" in str(excinfo.value)


def test_run_structured_raises_protocol_error_on_non_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="plain text",
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = ClaudeCodeClient()
    with pytest.raises(ClaudeProtocolError) as excinfo:
        client.run_structured(
            "hello",
            schema=StructuredSchema.from_dict({"type": "object"}),
        )

    assert "structured mode" in str(excinfo.value)
