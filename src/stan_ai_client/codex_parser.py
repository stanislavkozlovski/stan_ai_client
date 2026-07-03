from __future__ import annotations

import json
from typing import Any

from .types import CodexJsonPayload


def parse_codex_jsonl_payload(text: str) -> CodexJsonPayload:
    raw = text.strip()
    if not raw:
        raise ValueError("empty Codex JSONL output")

    events: list[dict[str, Any]] = []
    thread_id: str | None = None
    result: str | None = None
    usage: dict[str, Any] = {}
    error: dict[str, Any] | None = None

    for line in raw.splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("expected each Codex JSONL line to be a JSON object")

        events.append(parsed)
        event_type = parsed.get("type")

        if event_type == "thread.started":
            raw_thread_id = parsed.get("thread_id")
            if isinstance(raw_thread_id, str):
                thread_id = raw_thread_id
        elif event_type == "turn.completed":
            raw_usage = parsed.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
        elif event_type == "item.completed":
            item = parsed.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    result = text_value
        elif event_type in {"turn.failed", "error"}:
            error = parsed

    return CodexJsonPayload(
        thread_id=thread_id,
        result=result,
        usage=usage,
        events=tuple(events),
        error=error,
        structured_output=None,
        _structured_output_present=False,
    )


def try_parse_codex_jsonl_payload(text: str) -> CodexJsonPayload | None:
    try:
        return parse_codex_jsonl_payload(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def make_codex_structured_payload(
    structured_output: object,
    *,
    structured_output_present: bool = True,
) -> CodexJsonPayload:
    return CodexJsonPayload(
        thread_id=None,
        result=None,
        usage={},
        events=(),
        error=None,
        structured_output=structured_output,
        _structured_output_present=structured_output_present,
    )


def summarize_codex_error_text(
    *,
    payload: CodexJsonPayload | None,
    stdout: str,
    stderr: str,
) -> str:
    if payload is not None and payload.error is not None:
        summarized = _summarize_error_event(payload.error)
        if summarized:
            return summarized
    if stderr.strip():
        return stderr.strip()[:500]
    return stdout.strip()[:500]


def _summarize_error_event(event: dict[str, Any]) -> str | None:
    for key in ("message", "error"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
        if isinstance(value, dict):
            nested = _summarize_error_event(value)
            if nested:
                return nested
    return None
