from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .types import GrokJsonPayload

# A JSON value can only start with one of these characters (or a literal below),
# which is how truncated JSON is told apart from output that was never JSON.
_JSON_VALUE_START_CHARS = '{["-0123456789'
_JSON_LITERALS = ("true", "false", "null")
_CANCELLED_STOP_REASONS = frozenset({"canceled", "cancelled"})


def is_grok_error_payload(payload: GrokJsonPayload) -> bool:
    return payload.extras.get("type") == "error"


# Envelope evidence, grouped by how much a field proves.
#
# Grok structured mode returns either a control envelope or the caller's raw
# schema value, and a schema is free to use the envelope's field names for its
# own data. These three groups are the single mapping from field to evidence:
# the predicates below say which groups they trust rather than each re-deriving
# its own subset of fields.


def _has_envelope_turn_fields(payload: GrokJsonPayload) -> bool:
    """Envelope fields a domain value could plausibly reuse."""
    return payload.stop_reason is not None or payload.thought is not None


def _has_envelope_identifiers(payload: GrokJsonPayload) -> bool:
    """Envelope fields Grok mints per run, which a domain value has no reason to."""
    return payload.session_id is not None or payload.request_id is not None


def _has_cancellation_metadata(payload: GrokJsonPayload) -> bool:
    """Envelope field only a cancelled turn carries."""
    return payload.cancellation_category is not None


def is_grok_envelope_metadata(payload: GrokJsonPayload) -> bool:
    """True when the payload carries Grok envelope metadata (not schema data)."""
    return _has_envelope_turn_fields(payload) or _has_envelope_identifiers(payload)


def has_grok_result_envelope_evidence(payload: GrokJsonPayload) -> bool:
    """True when metadata beyond result fields identifies a likely envelope."""
    return is_grok_envelope_metadata(payload) or _has_cancellation_metadata(payload)


def is_grok_cancelled_payload(payload: GrokJsonPayload) -> bool:
    """True for a Grok result envelope whose turn ended by cancellation.

    Structured schema values can legitimately contain a ``stopReason`` key, so a
    cancelled-looking stop reason proves nothing on its own: require evidence
    that only Grok mints.
    """
    stop_reason = payload.stop_reason
    if not isinstance(stop_reason, str):
        return False
    if stop_reason.casefold() not in _CANCELLED_STOP_REASONS:
        return False
    return _has_envelope_identifiers(payload) or _has_cancellation_metadata(payload)


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
class GrokJsonText:
    """One decode of Grok JSON text under the "exactly one top-level value" rule.

    ``kind`` says how the text met or broke that rule:

    - ``"single"``: ``value`` is the one top-level JSON value.
    - ``"malformed"``: the text began as JSON but was truncated or held several
      concatenated roots. ``detail`` explains it without quoting model output,
      ``json_value_count`` counts the roots that completed, and ``value`` is the
      first of them (``None`` when none did).
    - ``"not_json"``: the text is empty or never looked like JSON, leaving the
      caller to decide whether that is a protocol error or a missing result.
    """

    kind: Literal["single", "malformed", "not_json"]
    value: Any = None
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


def _looks_like_json_value_start(raw: str) -> bool:
    stripped = raw.lstrip()
    if not stripped:
        return False
    if stripped[0] in _JSON_VALUE_START_CHARS:
        return True
    return any(literal.startswith(stripped) for literal in _JSON_LITERALS)


def decode_grok_json_text(text: str) -> GrokJsonText:
    """Decode ``text`` as exactly one top-level JSON value.

    Grok must emit a single JSON value in structured mode, both on stdout and
    inside an envelope's ``text``. This is the one place that applies the rule,
    so those two sources cannot drift apart on what counts as truncated,
    concatenated, or simply not JSON.
    """
    raw = text.strip()
    if not raw:
        return GrokJsonText("not_json")

    values, error = _decode_json_sequence(raw)
    if error is None and len(values) == 1:
        return GrokJsonText("single", values[0])

    if not values:
        if error is None or not _looks_like_json_value_start(raw):
            return GrokJsonText("not_json")
        return GrokJsonText(
            "malformed",
            detail=(
                "Grok returned malformed JSON at character "
                f"{error.pos} before completing a top-level value"
            ),
            json_value_count=0,
        )

    count = len(values)
    detail = (
        "Grok returned one top-level JSON value"
        if count == 1
        else f"Grok returned {count} concatenated top-level JSON values"
    )
    if error is not None:
        detail += f" followed by malformed JSON at character {error.pos}"
    return GrokJsonText("malformed", values[0], detail=detail, json_value_count=count)


@dataclass(frozen=True)
class GrokStructuredOutcome:
    """Single classification of Grok structured-mode stdout.

    ``kind`` decides how :class:`GrokClient` reacts:

    - ``"error"``: ``payload`` is a Grok ``{"type": "error"}`` envelope; raise a
      process error even when the CLI exited ``0``.
    - ``"cancelled"``: the envelope's stop reason says the turn was cancelled;
      raise a cancellation error unless an explicit raw candidate validates.
    - ``"malformed"``: Grok returned structured text that is not exactly one JSON
      value. ``json_value_count`` distinguishes concatenated roots when known. A
      complete outer raw value remains a candidate when only its envelope-like
      ``text`` field was malformed.
    - ``"missing"``: ``payload`` is an envelope that produced no structuredOutput;
      raise the structured-output-missing error unless an explicit raw candidate
      validates against the caller schema.
    - ``"validate"``: ``payload`` is the raw-value payload used for error
      reporting, and ``candidates`` are the ``(payload, value)`` pairs to try
      against the caller schema in order. The first value that validates wins and
      its payload is returned.

    Failure kinds carry ``candidates`` only when a complete outer value may still
    be the caller's raw schema object. Explicit errors and malformed stdout leave
    them empty because no eligible value survived intact.
    """

    kind: Literal["cancelled", "error", "malformed", "missing", "validate"]
    payload: GrokJsonPayload
    candidates: tuple[tuple[GrokJsonPayload, Any], ...] = ()
    detail: str | None = None
    json_value_count: int | None = None


def _payload_for_json_value(value: Any) -> GrokJsonPayload:
    if isinstance(value, dict):
        envelope = GrokJsonPayload.from_dict(value)
        if is_grok_envelope_metadata(envelope) or is_grok_error_payload(envelope):
            return envelope
    return raw_grok_structured_payload(value)


def classify_grok_structured_stdout(stdout: str) -> GrokStructuredOutcome | None:
    """Classify structured-mode stdout with a single JSON decode.

    Returns ``None`` when stdout is not JSON at all so the caller can raise a
    protocol error. Grok may return either a raw schema value or an envelope that
    wraps ``structuredOutput``; centralizing the raw/envelope decision here keeps
    the ambiguity in one place instead of spread across the client.
    """
    decoded = decode_grok_json_text(stdout)
    if decoded.kind == "not_json":
        return None
    if decoded.kind == "malformed":
        return GrokStructuredOutcome(
            "malformed",
            _payload_for_json_value(decoded.value),
            detail=decoded.detail,
            json_value_count=decoded.json_value_count,
        )

    value = decoded.value
    raw_payload = raw_grok_structured_payload(value)
    raw_candidates: tuple[tuple[GrokJsonPayload, Any], ...] = ((raw_payload, value),)
    if not isinstance(value, dict):
        return GrokStructuredOutcome("validate", raw_payload, raw_candidates)

    envelope = GrokJsonPayload.from_dict(value)

    if is_grok_error_payload(envelope):
        return GrokStructuredOutcome("error", envelope)
    if is_grok_cancelled_payload(envelope):
        return GrokStructuredOutcome("cancelled", envelope, raw_candidates)

    if (
        has_grok_result_envelope_evidence(envelope)
        and envelope.has_structured_output
        and envelope.structured_output is None
    ):
        # An envelope that declared structuredOutput but left it null still hands
        # the schema value back as text on some Grok builds.
        recovered = decode_grok_json_text(envelope.text or "")
        if recovered.kind == "malformed":
            return GrokStructuredOutcome(
                "malformed",
                envelope,
                raw_candidates,
                detail=recovered.detail,
                json_value_count=recovered.json_value_count,
            )
        if recovered.kind == "not_json":
            return GrokStructuredOutcome("missing", envelope, raw_candidates)
        return GrokStructuredOutcome(
            "validate",
            envelope,
            ((envelope, recovered.value), *raw_candidates),
        )

    if is_grok_structured_output_failure(envelope):
        return GrokStructuredOutcome("missing", envelope, raw_candidates)

    if is_grok_structured_envelope(envelope):
        # Clear envelope: prefer its structuredOutput, fall back to the raw value.
        return GrokStructuredOutcome(
            "validate",
            raw_payload,
            ((envelope, envelope.structured_output), *raw_candidates),
        )

    # Raw schema value; still accept an envelope's structuredOutput as a fallback.
    candidates = raw_candidates
    if envelope.has_structured_output:
        candidates = (*candidates, (envelope, envelope.structured_output))
    return GrokStructuredOutcome("validate", raw_payload, candidates)


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
