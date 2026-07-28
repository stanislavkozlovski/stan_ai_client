from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from stan_ai_client import parse_rate_limit_info


def test_parse_retry_after_seconds() -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Rate limit exceeded. Retry after 3600", now=reference)

    assert info.retry_after_seconds == 3660
    assert info.reset_at is not None
    assert int((info.reset_at - reference).total_seconds()) == 3660


def test_parse_relative_hours_and_minutes() -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Resets in 2 hours 30 minutes", now=reference)

    assert info.retry_after_seconds == (2 * 3600) + (30 * 60) + 60


def test_parse_absolute_local_time() -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Resets at 15:00", now=reference)

    assert info.reset_at is not None
    assert info.reset_at.hour == 15
    assert info.reset_at.minute == 1


def test_parse_embedded_timezone_without_losing_case() -> None:
    reference = datetime(2026, 3, 19, 20, 30, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info(
        "You've hit your limit · resets 1am (Europe/Sofia)",
        now=reference,
    )

    assert info.reset_at is not None
    assert info.retry_after_seconds == 9060
    assert info.reset_at == datetime(2026, 3, 19, 23, 1, tzinfo=ZoneInfo("UTC"))


def test_parse_compact_absolute_time_with_embedded_timezone() -> None:
    timezone = ZoneInfo("Europe/Madrid")
    reference = datetime(2026, 7, 24, 17, 41, 11, tzinfo=timezone)
    info = parse_rate_limit_info(
        "You've hit your session limit · resets 5:50pm (Europe/Madrid)",
        now=reference,
    )

    assert info.retry_after_seconds == 589
    assert info.reset_at == datetime(2026, 7, 24, 17, 51, tzinfo=timezone)


@pytest.mark.parametrize(
    "message",
    [
        "Resets at 5:50 pm",
        "Resets at 5:50pm",
        "Resets at 05:50 PM.",
    ],
)
def test_parse_absolute_meridiem_time_variants(message: str) -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))

    info = parse_rate_limit_info(message, now=reference)

    assert info.retry_after_seconds == 28260
    assert info.reset_at == datetime(2026, 3, 19, 17, 51, tzinfo=ZoneInfo("UTC"))


@pytest.mark.parametrize(
    "message",
    [
        "Resets at 5:50 pmUTC",
        "Resets at 5:50 pm2",
    ],
)
def test_malformed_meridiem_suffix_does_not_fall_through_to_24_hour(
    message: str,
) -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))

    info = parse_rate_limit_info(message, now=reference)

    assert info.retry_after_seconds is None
    assert info.reset_at is None
