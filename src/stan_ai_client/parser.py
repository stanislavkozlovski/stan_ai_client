from __future__ import annotations

import json

from .types import ClaudeJsonPayload


def parse_json_payload(text: str) -> ClaudeJsonPayload:
    raw = text.strip()
    if not raw:
        raise ValueError("empty JSON output")

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")

    return ClaudeJsonPayload.from_dict(parsed)


def try_parse_json_payload(text: str) -> ClaudeJsonPayload | None:
    try:
        return parse_json_payload(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def summarize_error_text(
    *,
    payload: ClaudeJsonPayload | None,
    stdout: str,
    stderr: str,
) -> str:
    if payload is not None and payload.result:
        return payload.result
    if stderr.strip():
        return stderr.strip()[:500]
    return stdout.strip()[:500]
