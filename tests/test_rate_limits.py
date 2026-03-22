from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from stan_ai_client import parse_rate_limit_info


def test_parse_retry_after_seconds() -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Rate limit exceeded. Retry after 3600", now=reference)

    assert info.retry_after_seconds == 3630
    assert info.reset_at is not None
    assert int((info.reset_at - reference).total_seconds()) == 3630


def test_parse_relative_hours_and_minutes() -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Resets in 2 hours 30 minutes", now=reference)

    assert info.retry_after_seconds == (2 * 3600) + (30 * 60) + 60


def test_parse_absolute_local_time() -> None:
    reference = datetime(2026, 3, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    info = parse_rate_limit_info("Resets at 15:00", now=reference)

    assert info.reset_at is not None
    assert info.reset_at.hour == 15
    assert info.reset_at.minute == 0

