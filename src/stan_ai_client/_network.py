from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

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


def claude_trusted_error_texts(
    *,
    payload: ClaudeJsonPayload | None,
    stdout: str,
    stderr: str,
) -> tuple[str, ...]:
    texts: list[str] = []
    if stderr.strip():
        texts.append(stderr)
    if payload is not None and payload.is_error is True and payload.result is not None:
        texts.append(payload.result)
    texts.extend(_prefixed_error_lines(stdout, prefix="api error:"))
    return tuple(texts)


def has_claude_network_unavailable_evidence(
    *,
    payload: ClaudeJsonPayload | None,
    stdout: str,
    stderr: str,
) -> bool:
    return any(
        _is_claude_network_unavailable_text(text)
        for text in claude_trusted_error_texts(
            payload=payload,
            stdout=stdout,
            stderr=stderr,
        )
    )


def codex_trusted_error_texts(
    *,
    payload: CodexJsonPayload | None,
    stderr: str,
) -> tuple[str, ...]:
    texts: list[str] = []
    if stderr.strip():
        texts.append(stderr)
    if payload is None:
        return tuple(texts)
    if payload.error is not None:
        texts.extend(_iter_trusted_error_record_texts(payload.error))
    for event in payload.events:
        if event.get("type") in _CODEX_ERROR_EVENT_TYPES:
            texts.extend(_iter_trusted_error_record_texts(event))
    return tuple(texts)


def has_codex_network_unavailable_evidence(
    *,
    payload: CodexJsonPayload | None,
    stderr: str,
) -> bool:
    return any(
        is_network_unavailable_text(text)
        for text in codex_trusted_error_texts(payload=payload, stderr=stderr)
    )


def grok_trusted_error_texts(
    *,
    payload: GrokJsonPayload | None,
    stdout: str,
    stderr: str,
) -> tuple[str, ...]:
    texts: list[str] = []
    if stderr.strip():
        texts.append(stderr)
    if payload is not None and payload.extras.get("type") == "error":
        for value in (
            payload.extras.get("message"),
            payload.extras.get("error"),
            payload.extras.get("cause"),
            payload.text,
        ):
            texts.extend(_iter_trusted_error_value_texts(value))
    texts.extend(_prefixed_error_lines(stdout, prefix="error:"))
    return tuple(texts)


def has_grok_network_unavailable_evidence(
    *,
    payload: GrokJsonPayload | None,
    stdout: str,
    stderr: str,
) -> bool:
    return any(
        is_network_unavailable_text(text)
        for text in grok_trusted_error_texts(
            payload=payload,
            stdout=stdout,
            stderr=stderr,
        )
    )


def _prefixed_error_lines(text: str, *, prefix: str) -> tuple[str, ...]:
    lines: list[str] = []
    for line in text.splitlines():
        candidate = line.lstrip()
        if candidate.casefold().startswith(prefix):
            lines.append(candidate)
    return tuple(lines)


def _iter_trusted_error_record_texts(
    record: Mapping[str, object],
) -> Iterator[str]:
    for key in _TRUSTED_ERROR_KEYS:
        yield from _iter_trusted_error_value_texts(record.get(key))


def _iter_trusted_error_value_texts(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        yield from _iter_trusted_error_record_texts(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_trusted_error_value_texts(item)
