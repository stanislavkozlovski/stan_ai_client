from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "too many requests",
    "overloaded",
    "hit your limit",
    "usage limit",
    "limit reached",
)


@dataclass(frozen=True)
class RateLimitInfo:
    message: str
    retry_after_seconds: int | None
    reset_at: datetime | None


def is_rate_limit_text(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in RATE_LIMIT_MARKERS) or (
        "limit" in lower and "reset" in lower
    )


def parse_rate_limit_info(
    text: str,
    *,
    now: datetime | None = None,
    local_tz: tzinfo | None = None,
) -> RateLimitInfo:
    reference = now or datetime.now().astimezone()
    if local_tz is not None:
        reference = reference.astimezone(local_tz)

    lower = text.lower()
    tz = _extract_embedded_timezone(text)
    reset_at = _parse_absolute_reset(lower, reference, tz, local_tz)
    if reset_at is None:
        reset_at = _parse_relative_reset(lower, reference)

    retry_after_seconds: int | None = None
    if reset_at is not None:
        retry_after_seconds = max(0, int((reset_at - reference).total_seconds()))
    else:
        retry_after_seconds = _parse_retry_after_seconds(lower)
        if retry_after_seconds is not None:
            reset_at = reference + timedelta(seconds=retry_after_seconds)

    return RateLimitInfo(
        message=text,
        retry_after_seconds=retry_after_seconds,
        reset_at=reset_at,
    )


def _extract_embedded_timezone(text: str) -> ZoneInfo | None:
    match = re.search(r"\(([A-Za-z]+/[A-Za-z_]+)\)", text)
    if match is None:
        return None
    try:
        return ZoneInfo(match.group(1))
    except ZoneInfoNotFoundError:
        return None


def _parse_absolute_reset(
    text: str,
    reference: datetime,
    embedded_tz: ZoneInfo | None,
    local_tz: tzinfo | None,
) -> datetime | None:
    timezone_to_use = embedded_tz or local_tz

    match = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})\s*(am|pm)", text)
    if match is not None:
        return _time_str_to_datetime(
            f"{match.group(1)}:00 {match.group(2)}",
            reference=reference,
            timezone_to_use=timezone_to_use,
        )

    match = re.search(r"resets?\s+(?:at\s+)?(\d{1,2}:\d{2}\s*(?:am|pm))", text)
    if match is not None:
        return _time_str_to_datetime(
            match.group(1),
            reference=reference,
            timezone_to_use=timezone_to_use,
        )

    match = re.search(r"resets?\s+(?:at\s+)?(\d{1,2}:\d{2})(?:\s|$|\.)", text)
    if match is not None:
        return _time_str_to_datetime(
            match.group(1),
            reference=reference,
            timezone_to_use=timezone_to_use,
        )

    return None


def _parse_relative_reset(text: str, reference: datetime) -> datetime | None:
    match = re.search(
        r"resets?\s+in[:\s]+(\d+)\s*(?:hours?|h)\s*(?:(\d+)\s*(?:minutes?|m|min))?",
        text,
    )
    if match is not None:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) is not None else 0
        return reference + timedelta(hours=hours, minutes=minutes, seconds=60)

    match = re.search(r"resets?\s+in[:\s]+(\d+)\s*(?:minutes?|m|min)", text)
    if match is not None:
        minutes = int(match.group(1))
        return reference + timedelta(minutes=minutes, seconds=60)

    return None


def _parse_retry_after_seconds(text: str) -> int | None:
    match = re.search(r"retry[- ]after[:\s]+(\d+)", text)
    if match is None:
        return None
    return int(match.group(1)) + 30


def _time_str_to_datetime(
    time_str: str,
    *,
    reference: datetime,
    timezone_to_use: tzinfo | None,
) -> datetime | None:
    cleaned = time_str.strip().upper()
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

        if timezone_to_use is not None:
            reference_in_tz = reference.astimezone(timezone_to_use)
            target = reference_in_tz.replace(
                hour=parsed.hour,
                minute=parsed.minute,
                second=0,
                microsecond=0,
            )
            if target <= reference_in_tz:
                target += timedelta(days=1)
            return target.astimezone(reference.tzinfo)

        target = reference.replace(
            hour=parsed.hour,
            minute=parsed.minute,
            second=0,
            microsecond=0,
        )
        if target <= reference:
            target += timedelta(days=1)
        return target

    return None
