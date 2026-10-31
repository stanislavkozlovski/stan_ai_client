"""Opt-in, cancellable retries for classified capacity and server failures."""

from __future__ import annotations

import html
import logging
import math
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import groupby
from threading import Event
from typing import TypeVar
from zoneinfo import ZoneInfo

from .exceptions import (
    CapacityError,
    ProviderRetryCancelled,
    ServerError,
    ServerErrorWindowExceeded,
)
from .types import FailedAttempt

TRun = TypeVar("TRun")
PROVIDER_RETRY_PREFIX = (
    "We just ran this, but it failed on a provider capacity limit or server error. "
    "Retry and continue the work:\n\n"
)


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """Delays/window are seconds. Cancellation interrupts waits, not subprocesses."""

    first_delay: float = 15.0
    max_delay: float = 15 * 60.0
    jitter: float = 0.2
    server_window: float = 10 * 60 * 60.0
    cancel: Event = field(default_factory=Event, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("first_delay", "max_delay", "server_window"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.max_delay < self.first_delay:
            raise ValueError("max_delay must be >= first_delay")
        if not math.isfinite(self.jitter) or not 0 <= self.jitter < 1:
            raise ValueError("jitter must be finite and in [0, 1)")


def retry_delay(attempt: int, hint: float | None, policy: ProviderRetryPolicy) -> float:
    """Return capped exponential backoff with jitter, or an unchanged retry hint."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    if hint is not None and math.isfinite(hint) and hint >= 0:
        return hint
    delay = policy.first_delay
    # Stop multiplying at the cap, even after arbitrarily many capacity failures.
    for _ in range(attempt - 1):
        if delay >= policy.max_delay:
            break
        delay = min(delay * 2, policy.max_delay)
    return delay * random.uniform(1 - policy.jitter, 1 + policy.jitter)


def _table_cell(text: str) -> str:
    first_line = text.splitlines()[0] if text else ""
    return (
        html.escape(first_line[:200]).replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")
    )


def render_failed_attempts(attempts: Iterable[FailedAttempt]) -> str:
    """Render consecutive identical errors as one row, with Sofia local timestamps."""
    lines = [
        "| Attempts | When (Europe/Sofia) | Kind | Signal | Error |",
        "|---|---|---|---|---|",
    ]
    for _, group in groupby(
        enumerate(attempts, 1),
        key=lambda row: (row[1].failure.kind, row[1].failure.signal, row[1].failure.message),
    ):
        rows = list(group)
        first_number, first = rows[0]
        last_number, last = rows[-1]
        numbers = (
            str(first_number) if first_number == last_number else f"{first_number}–{last_number}"
        )
        start = first.when.astimezone(ZoneInfo("Europe/Sofia")).strftime("%Y-%m-%d %H:%M:%S %Z")
        end = last.when.astimezone(ZoneInfo("Europe/Sofia")).strftime("%Y-%m-%d %H:%M:%S %Z")
        when = start if first_number == last_number else f"{start} – {end}"
        failure = first.failure
        lines.append(
            f"| {numbers} | {when} | {failure.kind.replace('_', ' ')} | "
            f"{_table_cell(failure.signal)} | {_table_cell(failure.message)} |"
        )
    return "\n".join(lines)


def run_with_provider_retry(
    call: Callable[[str], TRun],
    prompt: str,
    policy: ProviderRetryPolicy | None,
    *,
    logger: logging.Logger | None = None,
    provider: str = "Provider",
    model: str | None = None,
) -> TRun:
    """Retry typed provider failures, resending the original prompt with one prefix.

    Capacity has no deadline. The server window starts at the first server
    error, includes time spent running calls, and is not reset by capacity
    failures. Other errors escape unchanged. ``None`` disables retries.
    """
    if policy is None:
        return call(prompt)
    logger = logger if logger is not None else logging.getLogger("stan_ai_client")
    attempts: list[FailedAttempt] = []
    server_since: float | None = None
    last_error: CapacityError | ServerError | None = None
    request = prompt
    while not policy.cancel.is_set():
        try:
            return call(request)
        except (CapacityError, ServerError) as error:
            last_error = error
            now = datetime.now(timezone.utc)
            attempts.append(FailedAttempt(now, error.failure))
            delay = retry_delay(len(attempts), error.retry_after_seconds, policy)
            if isinstance(error, ServerError):
                elapsed = time.monotonic()
                if server_since is None:
                    server_since = elapsed
                if elapsed + delay - server_since > policy.server_window:
                    raise ServerErrorWindowExceeded(
                        f"Server-error retry window ({policy.server_window:g}s) exceeded.\n\n"
                        + render_failed_attempts(attempts),
                        attempts=tuple(attempts),
                    ) from error
            logger.warning(
                "%s model=%s kind=%s signal=%s attempt=%d retry_at=%s wait_seconds=%.1f",
                provider,
                model,
                error.failure.kind,
                error.signal,
                len(attempts),
                (now + timedelta(seconds=delay)).isoformat(),
                delay,
            )
            if policy.cancel.wait(delay):
                break
            request = PROVIDER_RETRY_PREFIX + prompt
    # Every cancellation exits here, chained to the last provider error, if any.
    raise ProviderRetryCancelled(
        "Provider retry cancelled.\n\n" + render_failed_attempts(attempts),
        attempts=tuple(attempts),
    ) from last_error
