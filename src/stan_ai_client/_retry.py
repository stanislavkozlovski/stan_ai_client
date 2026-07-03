from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from .exceptions import RateLimitError
from .types import RateLimitRetryPolicy

TRun = TypeVar("TRun")
TRateLimit = TypeVar("TRateLimit", bound=RateLimitError)


def run_with_rate_limit_retry(
    operation: Callable[[], TRun],
    *,
    rate_limit_policy: RateLimitRetryPolicy | None,
    logger: logging.Logger,
    provider: str,
    rate_limit_error_type: type[TRateLimit],
) -> TRun:
    """Run ``operation``, retrying parseable rate limits within a wait budget.

    This is the single implementation of the sleep-budget retry loop shared by
    every backend client. Backends differ only in the ``provider`` log prefix
    and the concrete ``rate_limit_error_type`` they raise, so centralizing the
    budget accounting here keeps that safety-relevant logic from drifting.
    """

    if rate_limit_policy is None:
        return operation()

    total_wait_seconds = 0.0
    attempt = 0

    while True:
        attempt += 1
        try:
            return operation()
        except rate_limit_error_type as exc:
            wait_seconds = exc.retry_after_seconds
            if wait_seconds is None:
                logger.warning(
                    "%s rate limited but no retry metadata was parsed attempt=%d total_wait_seconds=%.1f max_wait_seconds=%s reset_at=%s label=%s",
                    provider,
                    attempt,
                    total_wait_seconds,
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
                    attempt,
                    wait_seconds_float,
                    total_wait_seconds,
                    rate_limit_policy.max_wait_seconds,
                    exc.reset_at,
                    rate_limit_policy.label,
                )
                raise

            if rate_limit_policy.max_wait_seconds is not None:
                remaining_wait_seconds = rate_limit_policy.max_wait_seconds - total_wait_seconds
                if wait_seconds_float > remaining_wait_seconds:
                    logger.warning(
                        "%s rate limit exceeds wait budget attempt=%d wait_seconds=%.1f remaining_wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%.1f reset_at=%s label=%s",
                        provider,
                        attempt,
                        wait_seconds_float,
                        remaining_wait_seconds,
                        total_wait_seconds,
                        rate_limit_policy.max_wait_seconds,
                        exc.reset_at,
                        rate_limit_policy.label,
                    )
                    raise

            total_wait_seconds += wait_seconds_float
            logger.warning(
                "%s rate limited; retrying after reset attempt=%d wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%s retry_after_seconds=%s reset_at=%s label=%s",
                provider,
                attempt,
                wait_seconds_float,
                total_wait_seconds,
                rate_limit_policy.max_wait_seconds,
                exc.retry_after_seconds,
                exc.reset_at,
                rate_limit_policy.label,
            )
            time.sleep(wait_seconds_float)
