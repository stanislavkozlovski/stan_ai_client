from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .types import GrokJsonPayload


def is_grok_error_payload(payload: GrokJsonPayload) -> bool:
    return payload.extras.get("type") == "error"


def has_grok_result_envelope_evidence(payload: GrokJsonPayload) -> bool:
    """True when metadata beyond result fields identifies a likely envelope."""
    return is_grok_envelope_metadata(payload) or payload.cancellation_category is not None


def is_grok_cancelled_payload(payload: GrokJsonPayload) -> bool:
    """True for a Grok result envelope whose turn ended by cancellation.

    Structured schema values can legitimately contain a ``stopReason`` key, so
    require additional envelope evidence before treating that value as control
    metadata.
    """
    if not isinstance(payload.stop_reason, str) or payload.stop_reason.casefold() not in {
        "canceled",
        "cancelled",
    }:
        return False
    return bool(
        payload.session_id is not None
        or payload.request_id is not None
        or payload.cancellation_category is not None
    )


def is_grok_envelope_metadata(payload: GrokJsonPayload) -> bool:
    """True when the payload carries Grok envelope metadata (not schema data)."""
    return any(
        value is not None
        for value in (
            payload.stop_reason,
            payload.session_id,
            payload.request_id,
            payload.thought,
        )
    )


def is_grok_structured_envelope(payload: GrokJsonPayload) -> bool:
    """True when the payload is a Grok envelope wrapping a structuredOutput value."""
    return payload.has_structured_output and is_grok_envelope_metadata(payload)


def is_grok_structured_output_failure(payload: GrokJsonPayload) -> bool:
    """True when a Grok envelope reports/implies it produced no structuredOutput."""
    if payload.has_structured_output:
        return False
    if "structuredOutputError" in payload.extras or "structured_output_error" in payload.extras:
        return True
    return payload.text is not None and is_grok_envelope_metadata(payload)


def raw_grok_structured_payload(value: Any) -> GrokJsonPayload:
    """Wrap a raw JSON value from ``--json-schema`` as a structured-output payload.

    Newer Grok builds return the validated schema value directly instead of a
    Grok envelope, so the whole value is the structured output and no envelope
    metadata fields are populated.
    """
    return GrokJsonPayload(
        text=None,
        stop_reason=None,
        session_id=None,
        request_id=None,
        thought=None,
        structured_output=value,
        _structured_output_present=True,
    )


def parse_grok_json_payload(text: str) -> GrokJsonPayload:
    raw = text.strip()
    if not raw:
        raise ValueError("empty JSON output")

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")

    return GrokJsonPayload.from_dict(parsed)


def try_parse_grok_json_payload(text: str) -> GrokJsonPayload | None:
    try:
        return parse_grok_json_payload(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class GrokStructuredOutcome:
    """Single classification of Grok structured-mode stdout.

    ``kind`` decides how :class:`GrokClient` reacts:

    - ``"error"``: ``payload`` is a Grok ``{"type": "error"}`` envelope; raise a
      process error even when the CLI exited ``0``.
    - ``"cancelled"``: the envelope's stop reason says the turn was cancelled;
      raise a cancellation error unless an explicit raw candidate validates.
    - ``"malformed"``: Grok returned structured text that is not exactly one JSON
      value. ``json_value_count`` distinguishes concatenated roots when known.
    - ``"missing"``: ``payload`` is an envelope that produced no structuredOutput;
      raise the structured-output-missing error unless an explicit raw candidate
      validates against the caller schema.
    - ``"validate"``: ``payload`` is the raw-value payload used for error
      reporting, and ``candidates`` are the ``(payload, value)`` pairs to try
      against the caller schema in order. The first value that validates wins and
      its payload is returned.
    """

    kind: Literal["cancelled", "error", "malformed", "missing", "validate"]
    payload: GrokJsonPayload
    candidates: tuple[tuple[GrokJsonPayload, Any], ...] = ()
    detail: str | None = None
    json_value_count: int | None = None


def _decode_json_sequence(raw: str) -> tuple[tuple[Any, ...], json.JSONDecodeError | None]:
    """Decode every whitespace-separated top-level JSON value in ``raw``."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(raw):
        while index < len(raw) and raw[index].isspace():
            index += 1
        if index == len(raw):
            break
        try:
            value, index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError as exc:
            return tuple(values), exc
        values.append(value)
    return tuple(values), None


def _payload_for_json_value(value: Any) -> GrokJsonPayload:
    if isinstance(value, dict):
        envelope = GrokJsonPayload.from_dict(value)
        if is_grok_envelope_metadata(envelope) or is_grok_error_payload(envelope):
            return envelope
    return raw_grok_structured_payload(value)


def _looks_like_json_value_start(raw: str) -> bool:
    stripped = raw.lstrip()
    if not stripped:
        return False
    if stripped[0] in '{["-' or stripped[0].isdigit():
        return True
    return any(literal.startswith(stripped) for literal in ("true", "false", "null"))


def _malformed_json_sequence_outcome(
    raw: str,
    values: tuple[Any, ...],
    error: json.JSONDecodeError | None,
) -> GrokStructuredOutcome | None:
    if not values:
        if error is None or not _looks_like_json_value_start(raw):
            return None
        return GrokStructuredOutcome(
            "malformed",
            raw_grok_structured_payload(None),
            detail=(
                "Grok returned malformed JSON at character "
                f"{error.pos} before completing a top-level value"
            ),
            json_value_count=0,
        )
    count = len(values)
    if count == 1:
        if error is None:
            return None
        detail = "Grok returned one top-level JSON value"
    else:
        detail = f"Grok returned {count} concatenated top-level JSON values"
    if error is not None:
        detail += f" followed by malformed JSON at character {error.pos}"
    return GrokStructuredOutcome(
        "malformed",
        _payload_for_json_value(values[0]),
        detail=detail,
        json_value_count=count,
    )


def classify_grok_structured_stdout(stdout: str) -> GrokStructuredOutcome | None:
    """Classify structured-mode stdout with a single JSON parse.

    Returns ``None`` when stdout is not parseable JSON so the caller can raise a
    protocol error. Grok may return either a raw schema value or an envelope that
    wraps ``structuredOutput``; centralizing the raw/envelope decision here keeps
    the ambiguity in one place instead of spread across the client.
    """
    raw = stdout.strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        values, sequence_error = _decode_json_sequence(raw)
        return _malformed_json_sequence_outcome(raw, values, sequence_error)

    raw_payload = raw_grok_structured_payload(value)
    envelope = GrokJsonPayload.from_dict(value) if isinstance(value, dict) else None

    if envelope is not None and is_grok_error_payload(envelope):
        return GrokStructuredOutcome("error", envelope)
    if envelope is not None and is_grok_cancelled_payload(envelope):
        return GrokStructuredOutcome(
            "cancelled",
            envelope,
            ((raw_payload, raw_payload.structured_output),),
        )

    if (
        envelope is not None
        and has_grok_result_envelope_evidence(envelope)
        and envelope.has_structured_output
        and envelope.structured_output is None
    ):
        if envelope.text is None or not envelope.text.strip():
            return GrokStructuredOutcome(
                "missing",
                envelope,
                ((raw_payload, raw_payload.structured_output),),
            )

        text_values, text_error = _decode_json_sequence(envelope.text)
        malformed = _malformed_json_sequence_outcome(
            envelope.text,
            text_values,
            text_error,
        )
        if malformed is not None:
            return GrokStructuredOutcome(
                "malformed",
                envelope,
                detail=malformed.detail,
                json_value_count=malformed.json_value_count,
            )
        if text_error is not None or len(text_values) != 1:
            return GrokStructuredOutcome(
                "missing",
                envelope,
                ((raw_payload, raw_payload.structured_output),),
            )
        return GrokStructuredOutcome(
            "validate",
            envelope,
            (
                (envelope, text_values[0]),
                (raw_payload, raw_payload.structured_output),
            ),
        )

    if envelope is not None and is_grok_structured_output_failure(envelope):
        return GrokStructuredOutcome(
            "missing",
            envelope,
            ((raw_payload, raw_payload.structured_output),),
        )

    candidates: list[tuple[GrokJsonPayload, Any]] = []
    if envelope is not None and is_grok_structured_envelope(envelope):
        # Clear envelope: prefer its structuredOutput, fall back to the raw value.
        candidates.append((envelope, envelope.structured_output))
        candidates.append((raw_payload, raw_payload.structured_output))
    else:
        # Raw schema value; still accept an envelope's structuredOutput as a fallback.
        candidates.append((raw_payload, raw_payload.structured_output))
        if envelope is not None and envelope.has_structured_output:
            candidates.append((envelope, envelope.structured_output))

    return GrokStructuredOutcome("validate", raw_payload, tuple(candidates))


def summarize_grok_error_text(
    *,
    payload: GrokJsonPayload | None,
    stdout: str,
    stderr: str,
) -> str:
    if payload is not None and is_grok_error_payload(payload):
        message = payload.extras.get("message")
        if isinstance(message, str) and message:
            return message
    if payload is not None and payload.text:
        return payload.text
    if stderr.strip():
        return stderr.strip()[:500]
    return stdout.strip()[:500]
