"""Capacity/server classification restricted to native error diagnostics."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from ._network import (
    CodexOutputProtocol,
    claude_trusted_error_texts,
    codex_trusted_error_texts,
)
from .exceptions import ClaudeProcessError, CodexProcessError, ProcessError, ProviderError
from .types import ProviderFailure

SERVER_STATUSES = {500, 502, 503, 504}
_CLAUDE_STATUS = re.compile(r"\bAPI Error:\s*(\d{3})\b", re.IGNORECASE)
_CODEX_STATUS = re.compile(
    r"\b(?:unexpected status(?: code)?|exceeded retry limit,\s*last status:)\s*(\d{3})\b",
    re.IGNORECASE,
)
_RETRY_AFTER = re.compile(r"\bretry[- ]after[:\s]+([^\r\n]+)", re.IGNORECASE)


def _retry_hint(texts: tuple[str, ...]) -> float | None:
    for text in texts:
        match = _RETRY_AFTER.search(text)
        if match is None:
            continue
        value = match[1].strip()
        numeric = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*(?:s|seconds?))?\.?", value, re.IGNORECASE)
        try:
            seconds = float(numeric[1] if numeric else value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    continue
                seconds = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def classify_provider_failure(
    error: ProcessError,
    *,
    codex_output_protocol: CodexOutputProtocol = "unstructured",
) -> ProviderFailure | None:
    """Classify Claude/Codex process evidence, never assistant or tool output.

    Codex stderr defaults to guarded ``ERROR:`` lines. Pass ``jsonl`` only for
    JSON exec mode, where stderr is diagnostic rather than model prose.
    Grok and every error outside the two provider families remain unchanged.
    """
    if isinstance(error, ProviderError):
        return error.failure
    if isinstance(error, ClaudeProcessError):
        texts = claude_trusted_error_texts(
            payload=error.payload,
            stdout=error.stdout,
            stderr=error.stderr,
        )
        status = None
        if error.payload is not None and error.payload.is_error is True:
            raw_status = error.payload.extras.get("api_error_status")
            if isinstance(raw_status, int) and not isinstance(raw_status, bool):
                status = raw_status
        if status is not None:
            # Machine-readable status wins over text, including for non-retryable errors.
            message = texts[0] if texts else f"Claude API Error: {status}"
            if status == 529 or status in SERVER_STATUSES:
                return ProviderFailure(
                    "capacity" if status == 529 else "server_error",
                    str(status),
                    message,
                    _retry_hint(texts),
                )
            return None
        for text in texts:
            match = _CLAUDE_STATUS.search(text)
            status = int(match[1]) if match else None
            if status == 529 or "repeated 529 overloaded errors" in text.casefold():
                return ProviderFailure("capacity", "529", text, _retry_hint(texts))
            if status in SERVER_STATUSES:
                return ProviderFailure("server_error", str(status), text, _retry_hint(texts))
    elif isinstance(error, CodexProcessError):
        texts = codex_trusted_error_texts(
            payload=error.payload,
            stderr=error.stderr,
            output_protocol=codex_output_protocol,
        )
        for text in texts:
            if "selected model is at capacity" in text.casefold():
                return ProviderFailure("capacity", "server_overloaded", text, _retry_hint(texts))
            if "currently experiencing high demand" in text.casefold():
                return ProviderFailure(
                    "server_error",
                    "internal_server_error",
                    text,
                    _retry_hint(texts),
                )
            match = _CODEX_STATUS.search(text)
            if match and int(match[1]) in SERVER_STATUSES:
                return ProviderFailure("server_error", match[1], text, _retry_hint(texts))
    return None
