from __future__ import annotations

import json

from .types import GrokJsonPayload


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


def summarize_grok_error_text(
    *,
    payload: GrokJsonPayload | None,
    stdout: str,
    stderr: str,
) -> str:
    if payload is not None and payload.text:
        return payload.text
    if stderr.strip():
        return stderr.strip()[:500]
    return stdout.strip()[:500]
