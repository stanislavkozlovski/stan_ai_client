"""Parsing for ``agy`` headless output.

``agy`` (Google Antigravity CLI) writes newline-delimited ``stream-json``
events to stdout: one ``init`` event, ``step_update`` events for each step of
the turn, and a final ``result`` event whose body matches
``--output-format json``. Diagnostics go to stderr: ``error:`` lines, a
structured ``AGY_ERROR: {...}`` record when an agent or model call fails, and a
few stable notices (print timeout, headless permission denial, missing
conversation). Nothing in this module reads model output to classify errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .types import AntigravityAgentError, AntigravityJsonPayload

AGENT_ERROR_PREFIX = "AGY_ERROR:"
PRINT_TIMEOUT_NOTICE_PREFIX = "[agy] print timeout"
HEADLESS_DENIAL_MARKER = "headless mode cannot prompt for"
AUTO_MODE_UNAVAILABLE_MARKER = "--mode auto is not available"
_MISSING_CONVERSATION = re.compile(r'^warning: conversation ".*" not found$')
_DONE = "DONE"
_FINISH_STEP = "finish"


def stream_input_message(prompt: str) -> str:
    """Encode ``prompt`` as the single stream-json user message agy reads on stdin.

    ``ensure_ascii`` keeps the line encodable under any subprocess locale.
    """
    message = {"event": "user", "message": {"role": "user", "content": prompt}}
    return json.dumps(message, ensure_ascii=True) + "\n"


@dataclass(frozen=True)
class AntigravityStream:
    """One decode of stream-json stdout.

    ``result`` is the body of the last ``result`` event, if one arrived.
    ``invalid_lines`` counts non-empty lines that were not stream-json events.
    """

    events: tuple[dict[str, Any], ...]
    result: dict[str, Any] | None
    invalid_lines: int


def decode_antigravity_stream(stdout: str) -> AntigravityStream:
    events: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    invalid_lines = 0
    for line in stdout.split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            invalid_lines += 1
            continue
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            invalid_lines += 1
            continue
        events.append(event)
        body = event.get("result")
        if event["event"] == "result" and isinstance(body, dict):
            result = body
    return AntigravityStream(tuple(events), result, invalid_lines)


def try_parse_antigravity_payload(stdout: str) -> AntigravityJsonPayload | None:
    """Parse a complete stream-json run; ``None`` when no ``result`` event arrived."""
    stream = decode_antigravity_stream(stdout)
    if stream.result is None:
        return None
    return _payload_from_stream(stream, stream.result)


def recover_antigravity_partial_payload(stdout: str) -> AntigravityJsonPayload | None:
    """Best-effort payload for a run that was stopped before its ``result`` event.

    Keeps the conversation ID from the ``init`` event so a caller can resume,
    the response text streamed so far, and the usage of completed steps.
    """
    stream = decode_antigravity_stream(stdout)
    if stream.result is not None:
        return _payload_from_stream(stream, stream.result)
    if not stream.events:
        return None
    response = "".join(
        step.get("text_delta", "")
        for step in _steps(stream.events)
        if step.get("step_type") == "agent_response"
        and isinstance(step.get("text_delta"), str)
    )
    partial = {
        "conversation_id": _init_conversation_id(stream.events),
        "response": response,
    }
    return _payload_from_stream(
        stream, partial, extra_diagnostics=("stream ended before its result event",)
    )


def _payload_from_stream(
    stream: AntigravityStream,
    result: dict[str, Any],
    *,
    extra_diagnostics: tuple[str, ...] = (),
) -> AntigravityJsonPayload:
    diagnostics = extra_diagnostics
    if result.get("finished_reason") or result.get("status") not in (None, "SUCCESS"):
        diagnostics = (
            *diagnostics,
            "turn ended early; usage covers completed steps only",
        )
    step_usage: dict[str, int] | None = None
    if stream.invalid_lines:
        # A dropped step line would silently undercount, so leave usage unknown.
        diagnostics = (
            f"skipped {stream.invalid_lines} undecodable stream-json line(s)",
            *diagnostics,
        )
    else:
        conversation_id = result.get("conversation_id") or _init_conversation_id(
            stream.events
        )
        step_usage = sum_step_usage(
            stream.events,
            conversation_id=conversation_id if isinstance(conversation_id, str) else None,
        )
    return AntigravityJsonPayload.from_dict(
        result,
        events=stream.events,
        step_usage=step_usage,
        stream_diagnostics=diagnostics,
    )


def sum_step_usage(
    events: tuple[dict[str, Any], ...],
    *,
    conversation_id: str | None = None,
) -> dict[str, int] | None:
    """Sum the usage agy reported for each step this invocation completed.

    agy's ``result`` usage is cumulative across a resumed conversation, but each
    completed ``step_update`` carries that step's own counters, so their sum is
    exactly this invocation's usage. The last ``DONE`` report for a step index
    wins. Steps from another conversation (a subagent) are excluded, matching
    the ``result`` accounting. Returns ``None`` when a counter is unusable.
    """
    per_step: dict[object, dict[str, Any]] = {}
    for step in _steps(events):
        usage = step.get("usage")
        if step.get("state") != _DONE or not isinstance(usage, dict):
            continue
        step_conversation = step.get("conversation_id")
        if (
            conversation_id
            and isinstance(step_conversation, str)
            and step_conversation != conversation_id
        ):
            continue
        index = step.get("step_index")
        key: object = (
            index
            if isinstance(index, int) and not isinstance(index, bool)
            else ("unindexed", len(per_step))
        )
        per_step[key] = usage

    totals: dict[str, int] = {}
    for usage in per_step.values():
        for name, value in usage.items():
            if type(value) is not int or value < 0:
                return None
            totals[name] = totals.get(name, 0) + value
    return totals


def has_current_structured_output(payload: AntigravityJsonPayload) -> bool:
    """True when this invocation's events show agy accepting a structured result.

    agy validates the ``finish`` tool call against ``--json-schema`` and records
    a completed ``finish`` step only when it passes. A resumed conversation can
    otherwise report the ``structured_output`` of an earlier turn.
    """
    return any(
        step.get("step_type") == _FINISH_STEP and step.get("state") == _DONE
        for step in _steps(payload.events)
    )


def parse_antigravity_agent_error(stderr: str) -> AntigravityAgentError | None:
    """Return the last structured ``AGY_ERROR`` record on stderr, if any."""
    record: AntigravityAgentError | None = None
    for line in stderr.splitlines():
        candidate = line.strip()
        if not candidate.startswith(AGENT_ERROR_PREFIX):
            continue
        try:
            data = json.loads(candidate[len(AGENT_ERROR_PREFIX):])
        except (ValueError, RecursionError):
            continue
        if isinstance(data, dict):
            record = AntigravityAgentError.from_dict(data)
    return record


def antigravity_error_lines(stderr: str) -> tuple[str, ...]:
    """agy's fatal diagnostics: the stderr lines that start with ``error:``."""
    return tuple(
        candidate
        for candidate in (line.strip() for line in stderr.splitlines())
        if candidate.casefold().startswith("error:")
    )


def has_print_timeout_notice(stderr: str) -> bool:
    return any(
        line.strip().startswith(PRINT_TIMEOUT_NOTICE_PREFIX) for line in stderr.splitlines()
    )


def has_missing_conversation_warning(stderr: str) -> bool:
    """agy found no conversation for ``--conversation`` and started a new one."""
    return any(_MISSING_CONVERSATION.match(line.strip()) for line in stderr.splitlines())


def has_auto_mode_unavailable_warning(stderr: str) -> bool:
    return AUTO_MODE_UNAVAILABLE_MARKER in stderr


def headless_denial_notice(stderr: str) -> str | None:
    """agy's own explanation when a headless run auto-denied a tool and produced nothing."""
    for line in stderr.splitlines():
        if HEADLESS_DENIAL_MARKER in line:
            return line
    return None


def summarize_antigravity_error_text(
    *,
    payload: AntigravityJsonPayload | None,
    agent_error: AntigravityAgentError | None,
    stderr: str,
    returncode: int,
) -> str:
    if payload is not None and payload.error:
        return payload.error
    if agent_error is not None and agent_error.short_error:
        return agent_error.short_error
    error_lines = antigravity_error_lines(stderr)
    if error_lines:
        return error_lines[0]
    if stderr.strip():
        return stderr.strip()[:500]
    if payload is not None and payload.finished_reason:
        return f"Antigravity turn ended early (finished_reason={payload.finished_reason})"
    if payload is not None and payload.status:
        return f"Antigravity reported status {payload.status}"
    return f"Antigravity exited with code {returncode}"


def _steps(events: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for event in events:
        step = event.get("step_update") if event.get("event") == "step_update" else None
        if isinstance(step, dict):
            steps.append(step)
    return steps


def _init_conversation_id(events: tuple[dict[str, Any], ...]) -> str | None:
    for event in events:
        if event.get("event") == "init":
            value = event.get("conversation_id")
            if isinstance(value, str) and value:
                return value
    return None
