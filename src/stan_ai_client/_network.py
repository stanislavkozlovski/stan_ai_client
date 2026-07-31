from __future__ import annotations

from collections.abc import Mapping, Sequence

from .types import ClaudeJsonPayload, CodexJsonPayload, GrokJsonPayload

_NETWORK_UNAVAILABLE_MARKERS = (
    "failed to lookup address information",
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided",
    "could not resolve host",
    "couldn't resolve host",
    "unable to resolve host",
    "failed to resolve host",
    "failed to resolve address",
    "dns lookup failed",
    "dns lookup failure",
    "dns resolution failed",
    "dns resolution failure",
    "address resolution failed",
    "address resolution failure",
    "getaddrinfo eai_again",
    "getaddrinfo enotfound",
    "getaddrinfo eai_noname",
    "no address associated with hostname",
    "network is unreachable",
    "network unreachable",
    "host is unreachable",
    "no route to host",
)
_CLAUDE_NETWORK_UNAVAILABLE_MARKERS = (
    "unable to connect to api",
    "connection closed mid-response",
)
_TRUSTED_ERROR_KEYS = ("message", "error", "cause")
_CODEX_ERROR_EVENT_TYPES = frozenset({"error", "turn.failed"})


def is_network_unavailable_text(text: str) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in _NETWORK_UNAVAILABLE_MARKERS)


def _is_claude_network_unavailable_text(text: str) -> bool:
    normalized = text.casefold()
    return is_network_unavailable_text(text) or any(
        marker in normalized for marker in _CLAUDE_NETWORK_UNAVAILABLE_MARKERS
    )


def has_claude_network_unavailable_evidence(
    *,
    payload: ClaudeJsonPayload | None,
    stdout: str,
    stderr: str,
) -> bool:
    if _is_claude_network_unavailable_text(stderr):
        return True
    if (
        payload is not None
        and payload.is_error is True
        and payload.result is not None
        and _is_claude_network_unavailable_text(payload.result)
    ):
        return True
    return stdout.lstrip().casefold().startswith(
        "api error:"
    ) and _is_claude_network_unavailable_text(stdout)


def has_codex_network_unavailable_evidence(
    *,
    payload: CodexJsonPayload | None,
    stderr: str,
) -> bool:
    if is_network_unavailable_text(stderr):
        return True
    if payload is None:
        return False
    if (
        payload.error is not None
        and _trusted_error_record_has_network_unavailable_evidence(payload.error)
    ):
        return True
    return any(
        event.get("type") in _CODEX_ERROR_EVENT_TYPES
        and _trusted_error_record_has_network_unavailable_evidence(event)
        for event in payload.events
    )


def has_grok_network_unavailable_evidence(
    *,
    payload: GrokJsonPayload | None,
    stderr: str,
) -> bool:
    if is_network_unavailable_text(stderr):
        return True
    if payload is None or payload.extras.get("type") != "error":
        return False
    return any(
        _trusted_error_value_has_network_unavailable_evidence(value)
        for value in (
            payload.extras.get("message"),
            payload.extras.get("error"),
            payload.extras.get("cause"),
            payload.text,
        )
    )


def _trusted_error_record_has_network_unavailable_evidence(
    record: Mapping[str, object],
) -> bool:
    return any(
        _trusted_error_value_has_network_unavailable_evidence(record.get(key))
        for key in _TRUSTED_ERROR_KEYS
    )


def _trusted_error_value_has_network_unavailable_evidence(value: object) -> bool:
    if isinstance(value, str):
        return is_network_unavailable_text(value)
    if isinstance(value, Mapping):
        return _trusted_error_record_has_network_unavailable_evidence(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(
            _trusted_error_value_has_network_unavailable_evidence(item)
            for item in value
        )
    return False
