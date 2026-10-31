from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from stan_ai_client import (
    PROVIDER_RETRY_PREFIX,
    ClaudeCodeClient,
    CodexCapacityError,
    CodexClient,
    CodexProcessError,
    CodexRunOptions,
    CodexServerError,
    CommandMetadata,
    FailedAttempt,
    ProviderFailure,
    ProviderRetryCancelled,
    ProviderRetryPolicy,
    RateLimitRetryPolicy,
    RunOptions,
    ServerErrorWindowExceeded,
    StructuredSchema,
    render_failed_attempts,
    retry_delay,
    run_with_provider_retry,
)


class FakeWait(Event):
    def __init__(self, cancel_after: int | None = None) -> None:
        super().__init__()
        self.elapsed = 0.0
        self.waits: list[float] = []
        self.cancel_after = cancel_after

    def wait(self, timeout: float | None = None) -> bool:
        assert timeout is not None
        self.waits.append(timeout)
        self.elapsed += timeout
        if self.cancel_after == len(self.waits):
            self.set()
        return self.is_set()


def failure(
    kind: str = "capacity", hint: float | None = None
) -> CodexCapacityError | CodexServerError:
    evidence = (
        ProviderFailure("capacity", "server_overloaded", "Selected model is at capacity", hint)
        if kind == "capacity"
        else ProviderFailure("server_error", "502", "Bad Gateway", hint)
    )
    error_type = CodexCapacityError if kind == "capacity" else CodexServerError
    return error_type(
        evidence.message,
        command=CommandMetadata(("codex",), None, 0),
        returncode=1,
        stdout="",
        stderr=evidence.message,
        payload=None,
        failure=evidence,
    )


def test_backoff_caps_before_jitter_and_handles_long_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = ProviderRetryPolicy(jitter=0)
    assert [retry_delay(n, None, policy) for n in range(1, 10)] == [
        15,
        30,
        60,
        120,
        240,
        480,
        900,
        900,
        900,
    ]
    assert retry_delay(10**6, None, policy) == 900
    bounds = []

    def uniform(low: float, high: float) -> float:
        bounds.append((low, high))
        return high

    monkeypatch.setattr("stan_ai_client.provider_retry.random.uniform", uniform)
    policy = ProviderRetryPolicy()
    assert retry_delay(1, None, policy) == 18
    assert retry_delay(10, None, policy) == 1080
    assert retry_delay(1, 3600, policy) == 3600
    assert retry_delay(1, 0, policy) == 0
    assert bounds == [(0.8, 1.2), (0.8, 1.2)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_delay": 0},
        {"max_delay": -1},
        {"server_window": 0},
        {"max_delay": 1},
        {"jitter": 1},
        {"jitter": -0.1},
        {"first_delay": float("inf")},
        {"jitter": float("nan")},
    ],
)
def test_invalid_policy_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ProviderRetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_capacity_retries_past_server_window_and_reuses_original_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeWait()
    requests = []
    result = object()

    def call(prompt: str) -> object:
        requests.append(prompt)
        if len(requests) <= 100:
            raise failure()
        return result

    assert (
        run_with_provider_retry(
            call,
            "original",
            ProviderRetryPolicy(cancel=clock, jitter=0, server_window=1),
            provider="Codex",
            model="chosen-model",
        )
        is result
    )
    assert requests == ["original"] + [PROVIDER_RETRY_PREFIX + "original"] * 100
    assert clock.elapsed > 36000
    assert "model=chosen-model kind=capacity signal=server_overloaded" in caplog.text
    assert "retry_at=" in caplog.text


def test_server_window_includes_execution_time_and_keeps_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeWait()
    monkeypatch.setattr("stan_ai_client.provider_retry.time.monotonic", lambda: clock.elapsed)
    error = failure("server_error")

    def call(prompt: str) -> None:
        clock.elapsed += 10
        raise error

    with pytest.raises(ServerErrorWindowExceeded) as caught:
        run_with_provider_retry(
            call, "original", ProviderRetryPolicy(cancel=clock, jitter=0, server_window=60)
        )
    assert clock.waits == [15, 30]
    assert caught.value.__cause__ is error
    assert len(caught.value.attempts) == 3
    assert "| 1–3 |" in str(caught.value)


def test_capacity_does_not_reset_first_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeWait()
    monkeypatch.setattr("stan_ai_client.provider_retry.time.monotonic", lambda: clock.elapsed)
    errors = iter([failure("server_error"), failure(), failure("server_error")])

    def call(prompt: str) -> None:
        raise next(errors)

    with pytest.raises(ServerErrorWindowExceeded) as caught:
        run_with_provider_retry(
            call, "work", ProviderRetryPolicy(cancel=clock, jitter=0, server_window=60)
        )
    assert clock.waits == [15, 30]
    assert [row.failure.kind for row in caught.value.attempts] == [
        "server_error",
        "capacity",
        "server_error",
    ]


def test_capacity_time_before_first_server_error_is_not_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeWait()
    monkeypatch.setattr("stan_ai_client.provider_retry.time.monotonic", lambda: clock.elapsed)
    errors = iter([failure(hint=36000), failure("server_error", hint=60)])

    def call(prompt: str) -> str:
        error = next(errors, None)
        if error:
            raise error
        return "ok"

    assert (
        run_with_provider_retry(call, "work", ProviderRetryPolicy(cancel=clock, server_window=60))
        == "ok"
    )
    assert clock.waits == [36000, 60]  # A retry exactly at the deadline is allowed.


def test_server_hint_beyond_window_refuses_sleep() -> None:
    clock = FakeWait()

    def call(prompt: str) -> None:
        raise failure("server_error", hint=36001)

    with pytest.raises(ServerErrorWindowExceeded):
        run_with_provider_retry(call, "work", ProviderRetryPolicy(cancel=clock))
    assert clock.waits == []


class LateCancel(FakeWait):
    """Cancellation lands just after a wait has already timed out."""

    def wait(self, timeout: float | None = None) -> bool:
        super().wait(timeout)
        return False


@pytest.mark.parametrize("cancel", ["before_call", "during_wait", "after_wait"])
def test_cancellation_stops_without_another_call(cancel: str) -> None:
    clock = (LateCancel if cancel == "after_wait" else FakeWait)(cancel_after=1)
    if cancel == "before_call":
        clock.set()
    calls = []
    error = failure()

    def call(prompt: str) -> None:
        calls.append(prompt)
        raise error

    with pytest.raises(ProviderRetryCancelled) as caught:
        run_with_provider_retry(call, "work", ProviderRetryPolicy(cancel=clock))
    assert len(calls) == (0 if cancel == "before_call" else 1)
    assert len(caught.value.attempts) == len(calls)
    assert caught.value.__cause__ is (error if calls else None)


@pytest.mark.parametrize("policy", [None, ProviderRetryPolicy()])
def test_unrelated_error_escapes_unchanged(policy: ProviderRetryPolicy | None) -> None:
    error = ValueError("invalid input")

    def call(prompt: str) -> None:
        raise error

    with pytest.raises(ValueError) as caught:
        run_with_provider_retry(call, "work", policy)
    assert caught.value is error


def test_renderer_collapses_only_consecutive_identical_errors() -> None:
    when = datetime(2026, 10, 31, 6, 56, tzinfo=timezone.utc)
    capacity = failure().failure
    server = ProviderFailure(
        "server_error", "502", "<bad>|`gateway` " + "x" * 220 + "\nprivate detail"
    )
    rows = [FailedAttempt(when, x) for x in [capacity, capacity, server, capacity]]
    table = render_failed_attempts(rows)
    assert "| 1–2 |" in table
    assert "| 3 |" in table and "| 4 |" in table
    assert "2026-10-31 08:56:00 EET" in table
    assert "&lt;bad&gt;\\|\\`gateway\\`" in table
    assert "x" * 201 not in table and "private detail" not in table
    assert len(table.splitlines()) == 5


@pytest.mark.parametrize(
    "provider,mode",
    [
        ("claude", "text"),
        ("claude", "json"),
        ("claude", "structured"),
        ("codex", "text"),
        ("codex", "json"),
        ("codex", "structured"),
        ("codex", "usage"),
    ],
)
def test_client_retry_preserves_model_options_schema_and_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    mode: str,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    clock = FakeWait()
    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        if len(calls) < 3:
            if provider == "claude":
                return subprocess.CompletedProcess(
                    argv, 1, json.dumps({"is_error": True, "api_error_status": 529}), ""
                )
            if mode in {"json", "usage"}:
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    json.dumps(
                        {
                            "type": "turn.failed",
                            "error": {"message": "Selected model is at capacity"},
                        }
                    ),
                    "",
                )
            return subprocess.CompletedProcess(argv, 1, "", "ERROR: Selected model is at capacity")
        if "--output-schema" in argv:
            assert (
                json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text())
                == schema.schema
            )
        if mode == "usage":
            Path(argv[argv.index("--output-last-message") + 1]).write_text('{"answer":"ok"}')
            stdout = json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}
            )
        elif provider == "claude" and mode != "text":
            stdout = json.dumps({"result": "ok", "structured_output": {"answer": "ok"}})
        elif provider == "codex" and mode == "json":
            stdout = json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}
            )
        elif mode == "structured":
            stdout = '{"answer":"ok"}'
        else:
            stdout = "ok"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
    client: Any = ClaudeCodeClient() if provider == "claude" else CodexClient()
    options = (RunOptions if provider == "claude" else CodexRunOptions)(
        model="chosen-model",
        session_id="session-1",
        cwd=tmp_path,
        timeout_seconds=42,
    )
    kwargs: dict[str, Any] = {
        "options": options,
        "provider_retry_policy": ProviderRetryPolicy(cancel=clock, jitter=0),
    }
    if mode in {"structured", "usage"}:
        kwargs["schema"] = schema
    if mode == "usage":
        kwargs["capture_usage"] = True
    result = getattr(client, "run_structured" if mode == "usage" else f"run_{mode}")(
        "original", **kwargs
    )
    assert result.returncode == 0
    assert [call[1]["input"] for call in calls] == ["original"] + [
        PROVIDER_RETRY_PREFIX + "original"
    ] * 2
    for argv, options_passed in calls:
        assert "chosen-model" in argv and "session-1" in argv
        assert options_passed["cwd"] == str(tmp_path)
        assert options_passed["timeout"] == 42
    assert clock.waits == [15, 30]


def test_existing_quota_budget_is_not_reset_by_provider_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = iter(
        [
            "Selected model is at capacity",
            "rate limit retry-after: 20",
            "Selected model is at capacity",
            "rate limit retry-after: 20",
        ]
    )
    waits: list[float] = []

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [], 1, json.dumps({"type": "turn.failed", "error": {"message": next(errors)}}), ""
        )

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
    monkeypatch.setattr("stan_ai_client._retry.time.sleep", waits.append)
    with pytest.raises(CodexProcessError):
        CodexClient().run_json(
            "work",
            provider_retry_policy=ProviderRetryPolicy(cancel=FakeWait()),
            rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=100),
        )
    assert waits == [80]  # Existing quota parsing includes a 60-second buffer.


def test_quota_wait_cannot_reset_server_window_or_retry_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeWait()
    errors = iter(["unexpected status 502", "rate limit retry-after: 20", "unexpected status 503"])
    prompts = []

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        prompts.append(kwargs["input"])
        return subprocess.CompletedProcess(
            [], 1, json.dumps({"type": "turn.failed", "error": {"message": next(errors)}}), ""
        )

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", run)
    monkeypatch.setattr("stan_ai_client.provider_retry.time.monotonic", lambda: clock.elapsed)
    monkeypatch.setattr("stan_ai_client._retry.time.sleep", clock.wait)
    with pytest.raises(ServerErrorWindowExceeded) as caught:
        CodexClient().run_json(
            "work",
            provider_retry_policy=ProviderRetryPolicy(cancel=clock, jitter=0, server_window=100),
            rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=100),
        )
    assert clock.waits == [15, 80]
    assert prompts == ["work"] + [PROVIDER_RETRY_PREFIX + "work"] * 2
    assert [row.failure.signal for row in caught.value.attempts] == ["502", "503"]
