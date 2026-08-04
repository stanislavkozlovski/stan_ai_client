from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .types import CodexJsonPayload

CODEX_ERROR_EVENT_TYPES = frozenset({"error", "turn.failed"})
"""Event types Codex uses to declare a failure. Both the parsed payload's
``error`` field and network classification select events through this set."""

_CODEX_JSONL_PARSE_ERRORS = (TypeError, ValueError, json.JSONDecodeError)


def parse_codex_jsonl_payload(text: str) -> CodexJsonPayload:
    raw = text.strip()
    if not raw:
        raise ValueError("empty Codex JSONL output")

    return _make_codex_jsonl_payload(list(_iter_codex_jsonl_events(raw)))


def try_parse_codex_jsonl_payload(text: str) -> CodexJsonPayload | None:
    try:
        return parse_codex_jsonl_payload(text)
    except _CODEX_JSONL_PARSE_ERRORS:
        return None


def recover_codex_jsonl_prefix_payload(text: str) -> CodexJsonPayload | None:
    """Recover fully decoded leading events before a malformed JSONL tail."""
    events: list[dict[str, Any]] = []
    try:
        for event in _iter_codex_jsonl_events(text):
            events.append(event)
    except _CODEX_JSONL_PARSE_ERRORS:
        pass

    if not events:
        return None
    return _make_codex_jsonl_payload(events)


def _iter_codex_jsonl_events(text: str) -> Iterator[dict[str, Any]]:
    """The single line-scanning rule. Strict parsing and prefix recovery differ
    only in whether they stop at the first rejected line."""
    for line in text.splitlines():
        if line.strip():
            yield _parse_codex_jsonl_event(line)


def _parse_codex_jsonl_event(line: str) -> dict[str, Any]:
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        raise ValueError("expected each Codex JSONL line to be a JSON object")

    event_type = parsed.get("type")
    if not isinstance(event_type, str):
        raise ValueError("expected each Codex JSONL event to have a string type")
    return parsed


def _make_codex_jsonl_payload(events: list[dict[str, Any]]) -> CodexJsonPayload:
    thread_id: str | None = None
    result: str | None = None
    usage: dict[str, Any] = {}
    error: dict[str, Any] | None = None

    for parsed in events:
        event_type = parsed["type"]
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
        elif event_type in CODEX_ERROR_EVENT_TYPES:
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


_ERROR_TEXT_LIMIT = 500


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
    stripped_stderr = stderr.strip()
    if stripped_stderr:
        return stripped_stderr[-_ERROR_TEXT_LIMIT:]
    return stdout.strip()[:_ERROR_TEXT_LIMIT]


def _summarize_error_event(event: dict[str, Any]) -> str | None:
    for key in ("message", "error"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_ERROR_TEXT_LIMIT]
        if isinstance(value, dict):
            nested = _summarize_error_event(value)
            if nested:
                return nested
    return None
