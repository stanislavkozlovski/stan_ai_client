from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from stan_ai_client import parse_rate_limit_info
from stan_ai_client.rate_limits import _normalize_time_str, _time_str_to_datetime


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


def test_parse_relative_minutes_only() -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Resets in 15 minutes", now=reference)

    assert info.retry_after_seconds == 15 * 60 + 60
    assert info.reset_at == datetime(2026, 3, 19, 10, 16, tzinfo=ZoneInfo("UTC"))


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


def test_parse_pr105_claude_usage_limit_11_40pm_europe_madrid() -> None:
    """Exact production shape from uzealot PR #105 final-delta audit."""
    timezone = ZoneInfo("Europe/Madrid")
    reference = datetime(2026, 7, 28, 23, 10, tzinfo=timezone)
    info = parse_rate_limit_info(
        "Claude AI usage limit reached · resets 11:40pm (Europe/Madrid)",
        now=reference,
    )

    # Advertised 23:40 plus the one-minute safety buffer.
    assert info.retry_after_seconds == 31 * 60
    assert info.reset_at == datetime(2026, 7, 28, 23, 41, tzinfo=timezone)


@pytest.mark.parametrize(
    "message",
    [
        "Resets at 5:50 pm",
        "Resets at 5:50pm",
        "Resets at 5:50PM",
        "Resets at 05:50 PM.",
        "Resets at 05:50pm",
        "usage limit · resets 5:50 pm",
        "usage limit · resets 5:50pm",
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
        "Resets at 11:40 pm",
        "Resets at 11:40pm",
        "Resets at 11:40PM",
        "Resets at 11:40 Pm",
    ],
)
def test_parse_late_evening_meridiem_variants(message: str) -> None:
    reference = datetime(2026, 7, 28, 22, 0, tzinfo=ZoneInfo("UTC"))

    info = parse_rate_limit_info(message, now=reference)

    assert info.retry_after_seconds == 101 * 60
    assert info.reset_at == datetime(2026, 7, 28, 23, 41, tzinfo=ZoneInfo("UTC"))


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5:50pm", "5:50 PM"),
        ("5:50 pm", "5:50 PM"),
        ("05:50PM", "05:50 PM"),
        ("11:40PM", "11:40 PM"),
        (" 11:40 pm ", "11:40 PM"),
        ("15:00", "15:00"),
        ("1am", "1 AM"),
        ("1 AM", "1 AM"),
    ],
)
def test_normalize_time_str_accepts_compact_and_spaced_forms(
    raw: str,
    expected: str,
) -> None:
    assert _normalize_time_str(raw) == expected


def test_time_str_to_datetime_accepts_legacy_compact_capture() -> None:
    """Older parsers passed the whole compact token (``5:50pm``) to strptime."""
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))

    parsed = _time_str_to_datetime(
        "5:50pm",
        reference=reference,
        timezone_to_use=None,
    )

    assert parsed == datetime(2026, 3, 19, 17, 50, tzinfo=ZoneInfo("UTC"))


def test_absolute_reset_rolls_to_next_day_when_time_already_passed() -> None:
    reference = datetime(2026, 3, 19, 18, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Resets at 5:50pm", now=reference)

    assert info.reset_at == datetime(2026, 3, 20, 17, 51, tzinfo=ZoneInfo("UTC"))
