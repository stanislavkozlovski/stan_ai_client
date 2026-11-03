from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any

import pytest

from stan_ai_client import (
    CapacityError,
    ClaudeCapacityError,
    ClaudeCodeClient,
    ClaudeProcessError,
    ClaudeRateLimitError,
    ClaudeServerError,
    ClaudeUnclassifiedProviderError,
    CodexCapacityError,
    CodexClient,
    CodexProcessError,
    CodexRateLimitError,
    CodexServerError,
    CommandMetadata,
    GrokClient,
    GrokRateLimitError,
    NetworkUnavailableError,
    ProcessError,
    ServerError,
    classify_provider_failure,
)
from stan_ai_client.codex_parser import try_parse_codex_jsonl_payload
from stan_ai_client.parser import try_parse_json_payload


def raw_process_error(provider: str, stdout: str = "", stderr: str = "") -> ProcessError:
    error_type = ClaudeProcessError if provider == "claude" else CodexProcessError
    payload = (
        try_parse_json_payload(stdout)
        if provider == "claude"
        else try_parse_codex_jsonl_payload(stdout)
    )
    return error_type(
        "Unclassified process failure",
        command=CommandMetadata((provider,), None, 0),
        returncode=1,
        stdout=stdout,
        stderr=stderr,
        payload=payload,
    )


def fail_json(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    stdout: str,
    stderr: str = "",
    returncode: int = 1,
) -> ProcessError:
    calls = []

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
    client = ClaudeCodeClient() if provider == "claude" else CodexClient()
    with pytest.raises(ProcessError) as caught:
        client.run_json("Analyze this")
    error = caught.value
    assert len(calls) == 1  # Classification alone must never enable retrying.
    assert (error.stdout, error.stderr, error.returncode) == (stdout, stderr, returncode)
    assert error.command.argv[0] == provider
    return error


@pytest.mark.parametrize("status", [529, 500, 502, 503, 504])
@pytest.mark.parametrize("returncode", [0, 1])
def test_claude_native_status(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    returncode: int,
) -> None:
    stdout = json.dumps(
        {"type": "result", "is_error": True, "api_error_status": status, "result": "failed"}
    )
    error = fail_json(monkeypatch, "claude", stdout, returncode=returncode)
    assert isinstance(error, ClaudeProcessError)
    if status == 529:
        assert isinstance(error, ClaudeCapacityError)
        assert isinstance(error, CapacityError)
        assert not isinstance(error, ClaudeRateLimitError)
    else:
        assert isinstance(error, ClaudeServerError)
        assert isinstance(error, ServerError)
    failure = classify_provider_failure(raw_process_error("claude", stdout))
    assert failure is not None and failure.signal == str(status)
    assert failure.kind == ("capacity" if status == 529 else "server_error")
    assert failure.message == "failed"
    assert classify_provider_failure(error) == failure


@pytest.mark.parametrize("source", ["result", "stderr", "stdout"])
@pytest.mark.parametrize(
    "message,kind,signal",
    [
        ("API Error: 529 overloaded_error", CapacityError, "529"),
        ("Repeated 529 Overloaded errors", CapacityError, "529"),
        ("API Error: 500 Internal Server Error", ServerError, "500"),
        ("API Error: 502 Bad Gateway", ServerError, "502"),
        ("API Error: 503 Service Unavailable", ServerError, "503"),
        ("API Error: 504 Gateway Timeout", ServerError, "504"),
    ],
)
def test_claude_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    message: str,
    kind: type[ProcessError],
    signal: str,
) -> None:
    if source == "stdout" and not message.startswith("API Error:"):
        message = "API Error: " + message
    stdout = (
        json.dumps({"is_error": True, "result": message})
        if source == "result"
        else message if source == "stdout" else ""
    )
    stderr = message if source == "stderr" else ""
    error = fail_json(monkeypatch, "claude", stdout, stderr)
    assert isinstance(error, kind)
    failure = classify_provider_failure(raw_process_error("claude", stdout, stderr))
    assert failure is not None and failure.signal == signal
    assert failure.message == message


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
@pytest.mark.parametrize(
    "message,kind,signal",
    [
        (
            "Selected model is at capacity. Please try a different model.",
            CodexCapacityError,
            "server_overloaded",
        ),
        (
            "We're currently experiencing high demand. Please try again later.",
            CodexServerError,
            "internal_server_error",
        ),
        ("unexpected status 500 Internal Server Error", CodexServerError, "500"),
        ("unexpected status 502 Bad Gateway", CodexServerError, "502"),
        ("unexpected status 503 Service Unavailable", CodexServerError, "503"),
        ("unexpected status 504 Gateway Timeout", CodexServerError, "504"),
        ("exceeded retry limit, last status: 502 Bad Gateway", CodexServerError, "502"),
    ],
)
def test_codex_explicit_errors(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    message: str,
    kind: type[ProcessError],
    signal: str,
) -> None:
    event: dict[str, Any] = {"type": event_type}
    if event_type == "error":
        event["message"] = message
    else:
        event["error"] = {"message": message}
    stdout = json.dumps(event)
    error = fail_json(monkeypatch, "codex", stdout)
    assert isinstance(error, kind)
    assert isinstance(error, CodexProcessError)
    failure = classify_provider_failure(raw_process_error("codex", stdout))
    assert failure is not None and failure.signal == signal
    assert failure.kind == ("capacity" if kind is CodexCapacityError else "server_error")
    assert failure.message == message
    assert classify_provider_failure(error) == failure


def test_zero_exit_codex_failed_turn_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = json.dumps(
        {"type": "turn.failed", "error": {"message": "Selected model is at capacity"}}
    )
    assert isinstance(fail_json(monkeypatch, "codex", stdout, returncode=0), CodexCapacityError)


@pytest.mark.parametrize(
    "provider,stdout",
    [
        (
            "claude",
            json.dumps({"is_error": False, "result": "API Error: 503", "api_error_status": 529}),
        ),
        (
            "claude",
            json.dumps({"is_error": True, "structured_output": {"message": "API Error: 503"}}),
        ),
        ("claude", "Selected model is at capacity"),
        (
            "codex",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Selected model is at capacity"},
                }
            ),
        ),
        (
            "codex",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "aggregated_output": "unexpected status 503",
                    },
                }
            ),
        ),
        (
            "codex",
            json.dumps(
                {"type": "error", "structured_output": {"message": "unexpected status 503"}}
            ),
        ),
    ],
)
def test_model_and_tool_output_cannot_trigger_retry(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    stdout: str,
) -> None:
    error = fail_json(monkeypatch, provider, stdout, "unrelated process failure")
    assert not isinstance(error, (CapacityError, ServerError))
    assert classify_provider_failure(
        raw_process_error(provider, stdout, "unrelated process failure")
    ) is None


@pytest.mark.parametrize("message", ["Selected model is at capacity", "unexpected status 503"])
def test_codex_text_stderr_requires_error_prefix(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    for prefix, expected in [("", False), ("ERROR: ", True)]:

        def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 1, "", prefix + message)

        monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
        with pytest.raises(CodexProcessError) as caught:
            CodexClient().run_text("work")
        assert isinstance(caught.value, (CapacityError, ServerError)) is expected
        raw_error = raw_process_error("codex", stderr=prefix + message)
        assert (classify_provider_failure(raw_error) is not None) is expected
        failure = classify_provider_failure(raw_error, codex_output_protocol="jsonl")
        assert failure is not None and failure.message == prefix + message


@pytest.mark.parametrize(
    "provider,message,expected",
    [
        ("claude", "API Error: 429 rate limit", ClaudeRateLimitError),
        ("claude", "overloaded", ClaudeRateLimitError),
        ("claude", "API Error: 401 Unauthorized", ClaudeProcessError),
        ("claude", "API Error: 501 Not Implemented", ClaudeProcessError),
        ("claude", "Unable to connect to API", NetworkUnavailableError),
        ("codex", "unexpected status 429 rate limit", CodexRateLimitError),
        ("codex", "unexpected status 401 Unauthorized", CodexProcessError),
        ("codex", "unexpected status 501 Not Implemented", CodexProcessError),
        ("codex", "network is unreachable", NetworkUnavailableError),
    ],
)
def test_other_failures_keep_existing_classification(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    message: str,
    expected: type[ProcessError],
) -> None:
    error = fail_json(monkeypatch, provider, "", message)
    assert isinstance(error, expected)
    assert not isinstance(error, (CapacityError, ServerError))


def test_claude_structured_status_wins_over_message(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = json.dumps({"is_error": True, "api_error_status": 401, "result": "API Error: 503"})
    error = fail_json(monkeypatch, "claude", stdout)
    assert type(error) is ClaudeUnclassifiedProviderError
    assert not isinstance(error, (CapacityError, ServerError))
    assert classify_provider_failure(raw_process_error("claude", stdout)) is None


def test_classifier_native_status_overrides_conflicting_diagnostics() -> None:
    stdout = json.dumps({"is_error": True, "api_error_status": 529, "result": "failed"})
    diagnostic = "API Error: 503\nRetry-After: 37.5"
    failure = classify_provider_failure(raw_process_error("claude", stdout, diagnostic))
    assert failure is not None
    assert (failure.kind, failure.signal, failure.message, failure.retry_after_seconds) == (
        "capacity", "529", diagnostic, 37.5,
    )


@pytest.mark.parametrize(
    "hint,expected",
    [
        ("37", 37.0),
        ("37.5 seconds.", 37.5),
        ("0", 0.0),
        ("-1", None),
        ("NaN", None),
        ("invalid date", None),
    ],
)
def test_retry_after_hint_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    hint: str,
    expected: float | None,
) -> None:
    diagnostic = f"API Error: 529\nRetry-After: {hint}"
    error = fail_json(monkeypatch, "claude", "", diagnostic)
    assert isinstance(error, ClaudeCapacityError)
    assert error.retry_after_seconds == expected
    failure = classify_provider_failure(raw_process_error("claude", stderr=diagnostic))
    assert failure is not None and failure.retry_after_seconds == expected


def test_classifier_reads_retry_hint_from_raw_codex_error() -> None:
    diagnostic = "Selected model is at capacity\nRetry-After: 37.5"
    stdout = json.dumps({"type": "turn.failed", "error": {"message": diagnostic}})
    failure = classify_provider_failure(raw_process_error("codex", stdout))
    assert failure is not None
    assert (failure.kind, failure.signal, failure.message, failure.retry_after_seconds) == (
        "capacity", "server_overloaded", diagnostic, 37.5,
    )


def test_retry_after_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> FrozenDateTime:
            return cls(2026, 10, 31, 6, 56, tzinfo=timezone.utc)

    monkeypatch.setattr("stan_ai_client.provider_errors.datetime", FrozenDateTime)
    diagnostic = "API Error: 529\nRetry-After: Sat, 31 Oct 2026 06:57:00 GMT"
    error = fail_json(monkeypatch, "claude", "", diagnostic)
    assert isinstance(error, ClaudeCapacityError)
    assert error.retry_after_seconds == 60
    failure = classify_provider_failure(raw_process_error("claude", stderr=diagnostic))
    assert failure is not None and failure.retry_after_seconds == 60


def test_grok_overload_stays_a_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, "", "overloaded")

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
    with pytest.raises(GrokRateLimitError) as caught:
        GrokClient().run_json("work")
    assert classify_provider_failure(caught.value) is None
