from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from stan_ai_client import (
    CodexApprovalRequiredError,
    CodexClient,
    CodexNetworkUnavailableError,
    CodexProcessError,
    CodexProtocolError,
    CodexRateLimitError,
    CodexRunOptions,
    CodexStructuredOutputMissingError,
    CodexStructuredOutputValidationError,
    CodexTimeoutError,
    RateLimitRetryPolicy,
    StructuredSchema,
    normalize_ai_usage,
)
from stan_ai_client.codex_parser import (
    parse_codex_usage_payload,
    try_parse_codex_jsonl_payload,
)

SCHEMA: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
)
FINAL = '{"answer":"ok"}'
USAGE = {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 200}


def stream(*events: dict[str, Any]) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


SUCCESS = stream(
    {"type": "thread.started", "thread_id": "session-1"},
    {"type": "turn.completed", "usage": USAGE},
)


class CapturedRun:
    def __init__(
        self,
        stdout: str = SUCCESS,
        *,
        final: str | None = FINAL,
        returncode: int = 0,
        timeout: bool = False,
    ) -> None:
        self.stdout, self.final, self.returncode, self.timeout = (
            stdout,
            final,
            returncode,
            timeout,
        )
        self.paths: list[Path] = []
        self.argv: list[tuple[str, ...]] = []

    def __call__(
        self, argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.argv.append(tuple(argv))
        schema_path = Path(argv[argv.index("--output-schema") + 1])
        result_path = Path(argv[argv.index("--output-last-message") + 1])
        self.paths.extend((schema_path, result_path))
        assert json.loads(schema_path.read_text()) == SCHEMA.schema
        assert not result_path.exists()
        if self.final is not None:
            result_path.write_text(self.final)
        if self.timeout:
            raise subprocess.TimeoutExpired(
                argv, 1, output=self.stdout.encode(), stderr=b"partial stderr"
            )
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "")

    def assert_clean(self) -> None:
        assert self.paths
        assert all(not path.exists() for path in self.paths)
        assert all(not path.parent.exists() for path in self.paths[1::2])


@pytest.mark.parametrize(
    "options",
    [
        None,
        CodexRunOptions(session_id="session-1"),
        CodexRunOptions(continue_last_session=True),
    ],
)
def test_capture_returns_final_answer_and_raw_events(
    monkeypatch: pytest.MonkeyPatch, options: CodexRunOptions | None
) -> None:
    runner = CapturedRun()
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", runner)
    result = CodexClient().run_structured(
        "ok", schema=SCHEMA, options=options, capture_usage=True
    )
    assert result.structured_output == {"answer": "ok"}
    assert result.payload.structured_output == result.structured_output
    assert result.payload.has_structured_output
    assert result.payload.thread_id == "session-1"
    assert result.payload.usage == USAGE
    assert result.stdout == SUCCESS
    assert result.payload.usage_diagnostics == ()
    assert runner.argv[0].count("--json") == 1
    if options is not None:
        assert runner.argv[0].index("--json") < runner.argv[0].index("resume")
        assert result.payload.usage_scope == "cumulative"
        assert normalize_ai_usage("codex", result.payload).tokens.total_tokens is None
    else:
        assert result.payload.usage_scope == "invocation"
    runner.assert_clean()


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "not-json",
        SUCCESS + '{"type":',
        stream({"type": "turn.completed", "usage": "bad"}),
        stream({"type": "turn.completed", "usage": {"input_tokens": True}}),
    ],
)
def test_bad_accounting_does_not_retry_a_valid_answer(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    runner = CapturedRun(stdout)
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", runner)
    result = CodexClient().run_structured(
        "ok",
        schema=SCHEMA,
        capture_usage=True,
        rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=3600),
    )
    assert result.structured_output == {"answer": "ok"}
    assert normalize_ai_usage("codex", result.payload).diagnostics
    assert len(runner.argv) == 1
    runner.assert_clean()


@pytest.mark.parametrize(
    ("final", "error"),
    [
        (None, CodexStructuredOutputMissingError),
        ("", CodexStructuredOutputMissingError),
        ("invalid", CodexProtocolError),
        ('{"wrong":1}', CodexStructuredOutputValidationError),
    ],
)
def test_invalid_final_still_fails_with_usage_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, final: str | None, error: type[CodexProtocolError]
) -> None:
    runner = CapturedRun(final=final)
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", runner)
    with pytest.raises(error) as caught:
        CodexClient().run_structured("ok", schema=SCHEMA, capture_usage=True)
    assert caught.value.payload is not None
    assert caught.value.payload.usage == USAGE
    runner.assert_clean()


@pytest.mark.parametrize(
    ("stdout", "returncode", "error"),
    [
        (SUCCESS, 1, CodexProcessError),
        (
            stream({"type": "turn.failed", "error": {"message": "failed"}}),
            0,
            CodexProcessError,
        ),
        (
            "bad line\n"
            + stream({"type": "error", "message": "429 Too Many Requests"}),
            0,
            CodexRateLimitError,
        ),
        (
            stream(
                {
                    "type": "error",
                    "message": "error sending request: dns error: failed to lookup address information",
                }
            ),
            1,
            CodexNetworkUnavailableError,
        ),
    ],
)
def test_real_provider_failures_remain_typed(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
    error: type[CodexProcessError],
) -> None:
    runner = CapturedRun(stdout, returncode=returncode)
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", runner)
    with pytest.raises(error) as caught:
        CodexClient().run_structured("ok", schema=SCHEMA, capture_usage=True)
    assert caught.value.payload is not None
    assert caught.value.stdout == stdout
    assert caught.value.payload.usage_scope == "invocation"
    runner.assert_clean()


@pytest.mark.parametrize(
    ("stdout", "returncode", "options"),
    [
        (SUCCESS, 1, CodexRunOptions(continue_last_session=True)),
        (
            stream({"type": "turn.failed", "error": {"message": "failed"}}),
            0,
            CodexRunOptions(session_id="session-1"),
        ),
    ],
)
def test_resumed_provider_failures_preserve_cumulative_scope(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
    options: CodexRunOptions,
) -> None:
    runner = CapturedRun(stdout, returncode=returncode)
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", runner)
    with pytest.raises(CodexProcessError) as caught:
        CodexClient().run_structured(
            "ok", schema=SCHEMA, capture_usage=True, options=options
        )
    assert caught.value.payload is not None
    assert caught.value.payload.usage_scope == "cumulative"
    runner.assert_clean()


def test_quoted_errors_do_not_fail_and_recovered_error_is_not_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = stream(
        {"type": "error", "message": "connection interrupted, reconnecting"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "429 Too Many Requests"},
        },
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "aggregated_output": "DNS error"},
        },
        {"type": "turn.completed", "usage": USAGE},
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", CapturedRun(stdout))
    assert CodexClient().run_structured(
        "ok", schema=SCHEMA, capture_usage=True
    ).structured_output == {"answer": "ok"}


def test_capture_preserves_auto_approval_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denial = "This action was rejected due to unacceptable risk."
    runner = CapturedRun(
        stream(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "status": "declined",
                    "aggregated_output": denial,
                },
            }
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", runner)
    with pytest.raises(CodexApprovalRequiredError) as caught:
        CodexClient().run_structured(
            "ok",
            schema=SCHEMA,
            capture_usage=True,
            options=CodexRunOptions(permission_mode="auto"),
        )
    assert caught.value.approval_prompt == denial
    runner.assert_clean()


@pytest.mark.parametrize(
    ("options", "expected_scope"),
    [
        (None, "invocation"),
        (CodexRunOptions(session_id="session-1"), "cumulative"),
        (CodexRunOptions(continue_last_session=True), "cumulative"),
    ],
)
def test_capture_preserves_timeout_and_scoped_partial_payload(
    monkeypatch: pytest.MonkeyPatch,
    options: CodexRunOptions | None,
    expected_scope: str,
) -> None:
    runner = CapturedRun(timeout=True)
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", runner)
    with pytest.raises(CodexTimeoutError) as caught:
        CodexClient().run_structured(
            "ok", schema=SCHEMA, capture_usage=True, options=options
        )
    assert caught.value.stdout == SUCCESS
    assert caught.value.stderr == "partial stderr"
    assert caught.value.payload is not None and caught.value.payload.usage == USAGE
    assert caught.value.payload.usage_scope == expected_scope
    normalized = normalize_ai_usage("codex", caught.value.payload)
    assert normalized.tokens.total_tokens == (
        1200 if expected_scope == "invocation" else None
    )
    runner.assert_clean()


def test_rate_limit_retry_uses_new_result_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CapturedRun()
    calls = 0

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        runner.stdout = (
            stream({"type": "error", "message": "Rate limit exceeded, retry after 2"})
            if calls == 1
            else SUCCESS
        )
        runner.returncode = 1 if calls == 1 else 0
        return runner(argv, **kwargs)

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
    monkeypatch.setattr("stan_ai_client._retry.time.sleep", lambda seconds: None)
    result = CodexClient().run_structured(
        "ok",
        schema=SCHEMA,
        capture_usage=True,
        rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=3600),
    )
    assert result.structured_output == {"answer": "ok"}
    assert calls == 2 and len(set(runner.paths)) == 4
    runner.assert_clean()


def test_accounting_parser_keeps_later_failures_and_last_terminal_only() -> None:
    text = (
        SUCCESS
        + "broken\n"
        + stream({"type": "turn.completed"}, {"type": "turn.failed"})
    )
    payload = parse_codex_usage_payload(text, usage_scope="invocation")
    assert payload.usage == {}
    assert payload.usage_scope == "invocation"
    assert payload.events[-1]["type"] == "turn.failed"
    assert payload.usage_diagnostics
    assert try_parse_codex_jsonl_payload(text) is None
    duplicated = parse_codex_usage_payload(
        SUCCESS + SUCCESS, usage_scope="invocation"
    )
    assert normalize_ai_usage("codex", duplicated).tokens.total_tokens == 1200
