from __future__ import annotations

import json

from .types import GrokJsonPayload

def is_grok_error_payload(payload: GrokJsonPayload) -> bool:
    return payload.extras.get("type") == "error"


def _is_grok_error_envelope(data: dict[str, object]) -> bool:
    return data.get("type") == "error"


def parse_grok_json_payload(
    text: str,
    *,
    raw_structured_output: bool = False,
) -> GrokJsonPayload:
    raw = text.strip()
    if not raw:
        raise ValueError("empty JSON output")

    parsed = json.loads(raw)
    if raw_structured_output:
        if isinstance(parsed, dict) and _is_grok_error_envelope(parsed):
            return GrokJsonPayload.from_dict(parsed)

        return GrokJsonPayload(
            text=None,
            stop_reason=None,
            session_id=None,
            request_id=None,
            thought=None,
            structured_output=parsed,
            _structured_output_present=True,
        )

    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")

    return GrokJsonPayload.from_dict(parsed)


def try_parse_grok_json_payload(
    text: str,
    *,
    raw_structured_output: bool = False,
) -> GrokJsonPayload | None:
    try:
        return parse_grok_json_payload(
            text,
            raw_structured_output=raw_structured_output,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


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
