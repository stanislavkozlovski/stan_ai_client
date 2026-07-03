from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .types import GrokJsonPayload


def is_grok_error_payload(payload: GrokJsonPayload) -> bool:
    return payload.extras.get("type") == "error"


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
    - ``"missing"``: ``payload`` is an envelope that produced no structuredOutput;
      raise the structured-output-missing error unless an explicit raw candidate
      validates against the caller schema.
    - ``"validate"``: ``payload`` is the raw-value payload used for error
      reporting, and ``candidates`` are the ``(payload, value)`` pairs to try
      against the caller schema in order. The first value that validates wins and
      its payload is returned.
    """

    kind: Literal["error", "missing", "validate"]
    payload: GrokJsonPayload
    candidates: tuple[tuple[GrokJsonPayload, Any], ...] = ()


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
    except (ValueError, json.JSONDecodeError):
        return None

    raw_payload = raw_grok_structured_payload(value)
    envelope = GrokJsonPayload.from_dict(value) if isinstance(value, dict) else None

    if envelope is not None and is_grok_error_payload(envelope):
        return GrokStructuredOutcome("error", envelope)
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
