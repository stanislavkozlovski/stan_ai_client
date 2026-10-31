from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .exceptions import RateLimitError
from .provider_retry import ProviderRetryPolicy, run_with_provider_retry
from .types import RateLimitRetryPolicy

TRun = TypeVar("TRun")
TRateLimit = TypeVar("TRateLimit", bound=RateLimitError)


@dataclass
class _RateLimitRetryState:
    total_wait_seconds: float = 0.0
    attempt: int = 0


def run_with_retry_policies(
    operation: Callable[[str], TRun],
    *,
    prompt: str,
    rate_limit_policy: RateLimitRetryPolicy | None,
    provider_retry_policy: ProviderRetryPolicy | None,
    logger: logging.Logger,
    provider: str,
    model: str | None,
    rate_limit_error_type: type[TRateLimit],
) -> TRun:
    # Both budgets belong to the whole call, including alternating failure kinds.
    rate_state = _RateLimitRetryState()
    return run_with_provider_retry(
        lambda request: run_with_rate_limit_retry(
            lambda: operation(request),
            rate_limit_policy=rate_limit_policy,
            logger=logger,
            provider=provider,
            rate_limit_error_type=rate_limit_error_type,
            _state=rate_state,
        ),
        prompt, provider_retry_policy, logger=logger, provider=provider, model=model,
    )


def run_with_rate_limit_retry(
    operation: Callable[[], TRun],
    *,
    rate_limit_policy: RateLimitRetryPolicy | None,
    logger: logging.Logger,
    provider: str,
    rate_limit_error_type: type[TRateLimit],
    _state: _RateLimitRetryState | None = None,
) -> TRun:
    """Run ``operation``, retrying parseable rate limits within a wait budget.

    This is the single implementation of the sleep-budget retry loop shared by
    every backend client. Backends differ only in the ``provider`` log prefix
    and the concrete ``rate_limit_error_type`` they raise, so centralizing the
    budget accounting here keeps that safety-relevant logic from drifting.
    """

    if rate_limit_policy is None:
        return operation()

    state = _state if _state is not None else _RateLimitRetryState()

    while True:
        state.attempt += 1
        try:
            return operation()
        except rate_limit_error_type as exc:
            wait_seconds = exc.retry_after_seconds
            if wait_seconds is None:
                logger.warning(
                    "%s rate limited but no retry metadata was parsed attempt=%d total_wait_seconds=%.1f max_wait_seconds=%s reset_at=%s label=%s",
                    provider,
                    state.attempt,
                    state.total_wait_seconds,
                    rate_limit_policy.max_wait_seconds,
                    exc.reset_at,
                    rate_limit_policy.label,
                )
                raise

            wait_seconds_float = float(wait_seconds)
            if wait_seconds_float <= 0:
                logger.warning(
                    "%s rate limited with non-positive retry wait attempt=%d wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%s reset_at=%s label=%s",
                    provider,
                    state.attempt,
                    wait_seconds_float,
                    state.total_wait_seconds,
                    rate_limit_policy.max_wait_seconds,
                    exc.reset_at,
                    rate_limit_policy.label,
                )
                raise

            if rate_limit_policy.max_wait_seconds is not None:
                remaining_wait_seconds = rate_limit_policy.max_wait_seconds - state.total_wait_seconds
                if wait_seconds_float > remaining_wait_seconds:
                    logger.warning(
                        "%s rate limit exceeds wait budget attempt=%d wait_seconds=%.1f remaining_wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%.1f reset_at=%s label=%s",
                        provider,
                        state.attempt,
                        wait_seconds_float,
                        remaining_wait_seconds,
                        state.total_wait_seconds,
                        rate_limit_policy.max_wait_seconds,
                        exc.reset_at,
                        rate_limit_policy.label,
                    )
                    raise

            state.total_wait_seconds += wait_seconds_float
            logger.warning(
                "%s rate limited; retrying after reset attempt=%d wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%s retry_after_seconds=%s reset_at=%s label=%s",
                provider,
                state.attempt,
                wait_seconds_float,
                state.total_wait_seconds,
                rate_limit_policy.max_wait_seconds,
                exc.retry_after_seconds,
                exc.reset_at,
                rate_limit_policy.label,
            )
            time.sleep(wait_seconds_float)
