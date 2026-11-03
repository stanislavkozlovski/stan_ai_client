from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from stan_ai_client import (
    ClaudeCodeClient,
    ClaudeCodeError,
    ClaudeProcessError,
    ClaudeUnclassifiedProviderError,
    CodexClient,
    CodexCodeError,
    CodexProcessError,
    CodexUnclassifiedProviderError,
    GrokClient,
    GrokCodeError,
    GrokProcessError,
    GrokUnclassifiedProviderError,
    NetworkUnavailableError,
    ProcessError,
    ProviderRetryPolicy,
    RateLimitError,
    RateLimitRetryPolicy,
    StructuredSchema,
    UnclassifiedProviderError,
    classify_provider_failure,
)

MESSAGE = "No inference workers available for this request. Reference: busy_v2."
SCHEMA: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
)
CLIENTS = {"claude": ClaudeCodeClient, "codex": CodexClient, "grok": GrokClient}
ERRORS = {
    "claude": ClaudeUnclassifiedProviderError,
    "codex": CodexUnclassifiedProviderError,
    "grok": GrokUnclassifiedProviderError,
}
PROCESS_ERRORS = {
    "claude": ClaudeProcessError,
    "codex": CodexProcessError,
    "grok": GrokProcessError,
}


def mock_process(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    stderr: str = "",
    returncode: int = 1,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert len(calls) == 1, "Unclassified errors must not retry"
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
    return calls


def error_payload(provider: str, message: str = MESSAGE) -> str:
    if provider == "claude":
        return json.dumps({"is_error": True, "result": message, "api_error_status": 599})
    if provider == "codex":
        return json.dumps({"type": "turn.failed", "error": {"message": message, "code": "busy_v2"}})
    return json.dumps({"type": "error", "message": message, "requestId": "request-123"})


@pytest.mark.parametrize(
    "provider,mode,returncode",
    [
        (provider, mode, returncode)
        for provider in CLIENTS
        for mode in ("text", "json", "structured")
        for returncode in (0, 1)
        if not (provider == "codex" and mode != "json" and returncode == 0)
    ]
    + [("codex", "usage", 0), ("codex", "usage", 1)],
)
def test_unknown_native_failure_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    mode: str,
    returncode: int,
) -> None:
    def unexpected_wait(delay: float | None = None) -> bool:
        pytest.fail("Unclassified errors must not enter either retry wait")

    monkeypatch.setattr("stan_ai_client._retry.time.sleep", unexpected_wait)
    if provider == "codex" and mode in {"text", "structured"}:
        stdout, stderr = "", "ERROR: " + MESSAGE
    else:
        stdout, stderr = error_payload(provider), ""
    calls = mock_process(monkeypatch, stdout, stderr, returncode)
    client = CLIENTS[provider]()
    options: dict[str, Any] = {"rate_limit_policy": RateLimitRetryPolicy(max_wait_seconds=1000)}
    if provider != "grok":
        policy = ProviderRetryPolicy()
        monkeypatch.setattr(policy.cancel, "wait", unexpected_wait)
        options["provider_retry_policy"] = policy
    if mode in {"structured", "usage"}:
        options["schema"] = SCHEMA
    if mode == "usage":
        options["capture_usage"] = True

    with pytest.raises(UnclassifiedProviderError) as caught:
        getattr(client, "run_structured" if mode == "usage" else f"run_{mode}")("work", **options)

    error = caught.value
    assert type(error) is ERRORS[provider]
    assert isinstance(error, PROCESS_ERRORS[provider])
    assert MESSAGE in str(error)
    assert (error.stdout, error.stderr, error.returncode) == (stdout, stderr, returncode)
    assert error.command.argv[0] == provider
    assert len(calls) == 1
    assert classify_provider_failure(error) is None
    if stdout:
        assert error.payload is not None
        if isinstance(error, ClaudeUnclassifiedProviderError):
            assert error.payload is not None and error.payload.extras["api_error_status"] == 599
        elif isinstance(error, CodexUnclassifiedProviderError):
            assert error.payload is not None and error.payload.error is not None
            assert error.payload.error["error"]["code"] == "busy_v2"
        elif isinstance(error, GrokUnclassifiedProviderError):
            assert error.payload is not None and error.payload.request_id == "request-123"
    records = [
        r for r in caplog.records if getattr(r, "error_kind", None) == "unclassified_provider_error"
    ]
    assert len(records) == 1
    assert records[0].provider == provider  # type: ignore[attr-defined]
    assert "unclassified_provider_error=true" in records[0].getMessage()


@pytest.mark.parametrize(
    "provider,source,prefix",
    [
        ("claude", "stdout", "  API Error:"),
        ("claude", "stderr", "API ERROR:"),
        ("grok", "stdout", "  Error:"),
        ("grok", "stderr", "ERROR:"),
        ("codex", "stderr", "  error:"),
    ],
)
def test_explicit_error_lines_are_unclassified(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    source: str,
    prefix: str,
) -> None:
    diagnostic = prefix + " " + MESSAGE
    mock_process(
        monkeypatch,
        diagnostic if source == "stdout" else "",
        diagnostic if source == "stderr" else "",
    )
    with pytest.raises(ERRORS[provider]):
        CLIENTS[provider]().run_text("work")


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "message": MESSAGE},
        {"type": "turn.failed", "error": {"future_field": "unrecognized error shape"}},
    ],
)
def test_codex_error_records_need_no_known_message_or_shape(
    monkeypatch: pytest.MonkeyPatch,
    event: dict[str, Any],
) -> None:
    mock_process(monkeypatch, json.dumps(event))
    with pytest.raises(CodexUnclassifiedProviderError) as caught:
        CodexClient().run_json("work")
    assert caught.value.payload is not None and event in caught.value.payload.events


@pytest.mark.parametrize(
    "provider,stdout",
    [
        ("claude", json.dumps({"is_error": False, "result": "API Error: " + MESSAGE})),
        ("grok", json.dumps({"text": "Error: " + MESSAGE, "stopReason": "EndTurn"})),
        (
            "codex",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "ERROR: " + MESSAGE},
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
                        "exit_code": 1,
                        "aggregated_output": "ERROR: " + MESSAGE,
                    },
                }
            ),
        ),
    ],
)
def test_model_and_tool_output_leaves_failed_process_unclassified(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    stdout: str,
) -> None:
    mock_process(monkeypatch, stdout, "process crashed")
    with pytest.raises(ProcessError) as caught:
        CLIENTS[provider]().run_json("work")
    assert type(caught.value) is ERRORS[provider]
    assert classify_provider_failure(caught.value) is None
    assert any(
        getattr(r, "error_kind", None) == "unclassified_provider_error" for r in caplog.records
    )


@pytest.mark.parametrize("provider", CLIENTS)
def test_unclassified_fallback_requires_no_known_error_format(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    mock_process(monkeypatch, "", "unrecognized command line option")
    with pytest.raises(ProcessError) as caught:
        CLIENTS[provider]().run_text("work")
    assert type(caught.value) is ERRORS[provider]


@pytest.mark.parametrize("provider", CLIENTS)
def test_failure_without_diagnostics_is_unclassified(
    monkeypatch: pytest.MonkeyPatch, provider: str,
) -> None:
    mock_process(monkeypatch, "", returncode=2)
    with pytest.raises(ERRORS[provider]) as caught:
        CLIENTS[provider]().run_text("work")
    assert caught.value.returncode == 2
    assert caught.value.stdout == caught.value.stderr == ""


@pytest.mark.parametrize("provider", CLIENTS)
def test_successful_error_prose_stays_output(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    text = "API Error: future failure\nError: another example"
    mock_process(monkeypatch, text, returncode=0)
    assert CLIENTS[provider]().run_text("quote error examples").text == text


@pytest.mark.parametrize("provider", CLIENTS)
@pytest.mark.parametrize(
    "message,expected",
    [
        ("rate limit exceeded", RateLimitError),
        ("network is unreachable", NetworkUnavailableError),
    ],
)
def test_known_classifiers_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    message: str,
    expected: type[ProcessError],
) -> None:
    mock_process(monkeypatch, error_payload(provider, message))
    with pytest.raises(expected) as caught:
        CLIENTS[provider]().run_json("work")
    assert not isinstance(caught.value, UnclassifiedProviderError)
    assert not any(
        getattr(r, "error_kind", None) == "unclassified_provider_error" for r in caplog.records
    )


def test_warning_excerpt_is_bounded_but_raw_evidence_is_intact(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    diagnostic = "future provider error: " + "x" * 1000
    stdout = error_payload("claude", diagnostic)
    mock_process(monkeypatch, stdout)
    with pytest.raises(ClaudeUnclassifiedProviderError) as caught:
        ClaudeCodeClient().run_json("work")
    assert str(caught.value) == diagnostic
    assert caught.value.stdout == stdout
    warning = next(r for r in caplog.records if getattr(r, "error_kind", None))
    assert repr(diagnostic[:200]) in warning.getMessage()
    assert diagnostic[:201] not in warning.getMessage()


def test_unclassified_types_preserve_public_hierarchies() -> None:
    for error, process_error, provider_error in [
        (ClaudeUnclassifiedProviderError, ClaudeProcessError, ClaudeCodeError),
        (CodexUnclassifiedProviderError, CodexProcessError, CodexCodeError),
        (GrokUnclassifiedProviderError, GrokProcessError, GrokCodeError),
    ]:
        assert issubclass(error, UnclassifiedProviderError)
        assert issubclass(error, process_error)
        assert issubclass(error, provider_error)
    from stan_ai_client.client import (
        ClaudeUnclassifiedProviderError as LegacyClaudeError,
        UnclassifiedProviderError as LegacyError,
    )

    assert LegacyClaudeError is ClaudeUnclassifiedProviderError
    assert LegacyError is UnclassifiedProviderError
