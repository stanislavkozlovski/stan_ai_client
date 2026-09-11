"""Pure, conservative accounting for provider-owned usage envelopes.

None means unavailable; zero is a measured value. Reasoning is a subset of
output. The other components and ``unsplit_tokens`` are disjoint allocations.
No prices, persistence, session-file reads, or producer context live here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .types import ClaudeJsonPayload, CodexJsonPayload, GrokJsonPayload

_MAX_COUNT = 2**63 - 1
_CODEX_TOKEN_COUNTERS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
UsagePayload = (
    Mapping[str, Any] | ClaudeJsonPayload | CodexJsonPayload | GrokJsonPayload
)


@dataclass(frozen=True)
class TokenUsage:
    fresh_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    unsplit_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelUsage:
    model: str | None
    model_source: Literal["reported", "requested", "unknown"]
    tokens: TokenUsage
    reported_cost_usd: float | None = None


@dataclass(frozen=True)
class UsageFacts:
    provider: str
    session_id: str | None
    tokens: TokenUsage
    models: tuple[ModelUsage, ...]
    reported_cost_usd: float | None = None
    plan_type: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()


def normalize_ai_usage(
    provider: str,
    payload: UsagePayload,
    *,
    previous_snapshot: UsagePayload | None = None,
) -> UsageFacts:
    """Normalize one invocation, without summing repeated terminal events.

    Codex capture marks fresh calls as invocation scope and resumes as cumulative.
    Bare Codex counters need an explicit scope; Claude defaults to invocation.
    For a confirmed cumulative Codex source, ``previous_snapshot`` must be the
    immediately preceding snapshot from the same session. Missing or ambiguous
    baselines, changed counter shapes, and resets leave the delta unavailable.
    ``requested_model`` is an optional caller hint for single-model fallbacks.
    """
    raw = _raw_payload(payload)
    if provider != "codex":
        return _normalize(provider, raw)

    scope = raw.get("usage_scope", "unknown")
    if scope == "invocation":
        return _normalize(provider, raw)
    if scope != "cumulative":
        return _unavailable(_normalize(provider, raw), "unknown usage scope")

    previous = _raw_payload(previous_snapshot) if previous_snapshot is not None else {}
    if (
        previous.get("usage_scope") not in {"cumulative", "invocation"}
        or not _session_id(raw)
        or _session_id(raw) != _session_id(previous)
    ):
        return _unavailable(
            _normalize(provider, raw),
            "cumulative usage needs the immediately preceding same-session snapshot",
        )

    try:
        invocation_usage = _codex_usage_delta(
            _mapping(raw.get("usage")), _mapping(previous.get("usage"))
        )
    except ValueError:
        return _unavailable(
            _normalize(provider, raw), "cumulative counters reset or changed shape"
        )

    facts = _normalize(provider, {**raw, "usage": invocation_usage})
    return replace(
        facts,
        raw=raw,
        diagnostics=(*facts.diagnostics, "invocation delta from cumulative baseline"),
    )


def _raw_payload(payload: UsagePayload) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return deepcopy(dict(payload))
    raw = asdict(payload)
    raw.update(raw.pop("extras", {}))
    if isinstance(payload, ClaudeJsonPayload):
        raw["modelUsage"] = raw.pop("model_usage")
    return raw


def _normalize(provider: str, raw: dict[str, Any]) -> UsageFacts:
    raw_diagnostics = raw.get("usage_diagnostics")
    diagnostics = (
        [item for item in raw_diagnostics if isinstance(item, str)]
        if isinstance(raw_diagnostics, (tuple, list))
        else []
    )
    session = _session_id(raw)
    usage = _mapping(raw.get("usage"))
    model = _text(raw.get("model"))
    rows: list[ModelUsage] = []
    if provider == "codex":
        tokens = _codex_tokens(usage, diagnostics)
    elif provider == "claude":
        model_usage = _mapping(raw.get("modelUsage"))
        rows = [
            _claude_model_row(name, value, diagnostics)
            for name, value in sorted(
                model_usage.items(), key=lambda item: str(item[0])
            )
        ]
        if any(row.tokens != TokenUsage() for row in rows):
            tokens = _sum_rows(rows, diagnostics)
        else:
            tokens = _claude_tokens(usage or raw, diagnostics, camel=False)
            diagnostics.append("Claude model attribution incomplete")
            if rows and tokens != TokenUsage():
                rows.append(ModelUsage(None, "unknown", tokens))
    elif provider == "grok":
        tokens = TokenUsage()
        diagnostics.append("provider does not expose token usage")
    else:
        total = _count((usage or raw).get("total_tokens"), diagnostics, "total_tokens")
        tokens = TokenUsage(unsplit_tokens=total, total_tokens=total)
        diagnostics.append("only legacy aggregate usage is supported")
    if not rows:
        requested = _text(raw.get("requested_model"))
        rows = [
            ModelUsage(
                model or requested,
                "reported" if model else "requested" if requested else "unknown",
                tokens,
            )
        ]
    if tokens.total_tokens is None:
        diagnostics.append("total tokens unavailable")
    return UsageFacts(
        provider,
        session,
        tokens,
        tuple(rows),
        _cost(raw.get("total_cost_usd"), diagnostics, "total_cost_usd"),
        _text(raw.get("plan_type")),
        raw,
        tuple(dict.fromkeys(diagnostics)),
    )


def _codex_tokens(usage: Mapping[str, Any], diagnostics: list[str]) -> TokenUsage:
    def count(name: str) -> int | None:
        return _count(usage.get(name), diagnostics, name)

    input_tokens, output = count("input_tokens"), count("output_tokens")
    cached, written = count("cached_input_tokens"), count("cache_write_input_tokens")
    reasoning = count("reasoning_output_tokens")
    total = _sum_complete([input_tokens, output], diagnostics)
    if total is None:
        total = count("total_tokens")
    fresh = None
    if input_tokens is not None:
        if (cached or 0) + (written or 0) > input_tokens:
            diagnostics.append("cache split exceeds input; preserving unsplit input")
            cached = written = None
        elif cached is not None and (
            "cache_write_input_tokens" not in usage or written is not None
        ):
            fresh = input_tokens - cached - (written or 0)
    return _allocate(fresh, cached, written, output, reasoning, total, diagnostics)


def _codex_usage_delta(
    current: Mapping[str, Any], previous: Mapping[str, Any]
) -> dict[str, int]:
    delta: dict[str, int] = {}
    for name in _CODEX_TOKEN_COUNTERS:
        if (name in current) != (name in previous):
            raise ValueError("counter shape changed")
        if name not in current:
            continue
        now = _count(current[name], [], name)
        before = _count(previous[name], [], name)
        if now is None or before is None or now < before:
            raise ValueError("counter reset or invalid value")
        delta[name] = now - before
    return delta


def _claude_model_row(name: Any, value: Any, diagnostics: list[str]) -> ModelUsage:
    """Every ``modelUsage`` entry becomes exactly one row. A malformed entry
    keeps its identity with unavailable tokens, so it still blocks a complete
    invocation total, and a reported breakdown never falls back to a single
    ``requested_model`` hint."""
    model = name if isinstance(name, str) else None
    source: Literal["reported", "unknown"] = (
        "reported" if model is not None else "unknown"
    )
    if model is None or not isinstance(value, Mapping):
        diagnostics.append("invalid Claude model usage row")
    if not isinstance(value, Mapping):
        return ModelUsage(model, source, TokenUsage())
    return ModelUsage(
        model,
        source,
        _claude_tokens(value, diagnostics, camel=True),
        _cost(value.get("costUSD"), diagnostics, f"{name}.costUSD"),
    )


def _claude_tokens(
    usage: Mapping[str, Any], diagnostics: list[str], *, camel: bool
) -> TokenUsage:
    names = (
        (
            "inputTokens",
            "cacheReadInputTokens",
            "cacheCreationInputTokens",
            "outputTokens",
        )
        if camel
        else (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
        )
    )
    components = [_count(usage.get(name), diagnostics, name) for name in names]
    total = _sum_complete(components, diagnostics)
    return _allocate(
        components[0],
        components[1],
        components[2],
        components[3],
        None,
        total,
        diagnostics,
    )


def _allocate(
    fresh: int | None,
    cached: int | None,
    written: int | None,
    output: int | None,
    reasoning: int | None,
    total: int | None,
    diagnostics: list[str],
) -> TokenUsage:
    known = sum(
        value for value in (fresh, cached, written, output) if value is not None
    )
    if total is not None and known > total:
        diagnostics.append("token splits exceed total; preserving unassigned total")
        return TokenUsage(unsplit_tokens=total, total_tokens=total)
    if reasoning is not None and (output is None or reasoning > output):
        diagnostics.append("reasoning subset cannot be verified against output")
        reasoning = None
    if total is not None and known < total:
        diagnostics.append("token breakdown incomplete; preserving unsplit remainder")
    return TokenUsage(
        fresh,
        cached,
        written,
        output,
        reasoning,
        total - known if total is not None else None,
        total,
    )


def _sum_rows(rows: list[ModelUsage], diagnostics: list[str]) -> TokenUsage:
    """Count each model once. A component missing from one row cannot erase the
    others' known amounts, and the invocation total exists only when every row
    has one; the same allocator that built each row then keeps any remainder
    unsplit."""

    def known_sum(name: str) -> int | None:
        known = [
            getattr(row.tokens, name)
            for row in rows
            if getattr(row.tokens, name) is not None
        ]
        return _sum_complete(known, diagnostics) if known else None

    return _allocate(
        known_sum("fresh_input_tokens"),
        known_sum("cache_read_input_tokens"),
        known_sum("cache_write_input_tokens"),
        known_sum("output_tokens"),
        None,
        _sum_complete([row.tokens.total_tokens for row in rows], diagnostics),
        diagnostics,
    )


def _count(value: Any, diagnostics: list[str], name: str) -> int | None:
    if value is None:
        return None
    if type(value) is int and 0 <= value <= _MAX_COUNT:
        return int(value)
    diagnostics.append(f"invalid token count: {name}")
    return None


def _sum_complete(values: list[int | None], diagnostics: list[str]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return _count(
        sum(value for value in values if value is not None), diagnostics, "sum"
    )


def _cost(value: Any, diagnostics: list[str], name: str) -> float | None:
    if value is None:
        return None
    if type(value) in (int, float):
        try:
            cost = float(value)
            if math.isfinite(cost) and cost >= 0:
                return cost
        except OverflowError:
            pass
    diagnostics.append(f"invalid reported cost: {name}")
    return None


def _unavailable(facts: UsageFacts, diagnostic: str) -> UsageFacts:
    return replace(
        facts,
        tokens=TokenUsage(),
        reported_cost_usd=None,
        models=tuple(
            replace(row, tokens=TokenUsage(), reported_cost_usd=None)
            for row in facts.models
        ),
        diagnostics=(*facts.diagnostics, diagnostic),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _session_id(raw: Mapping[str, Any]) -> str | None:
    return (
        _text(raw.get("session_id"))
        or _text(raw.get("sessionId"))
        or _text(raw.get("thread_id"))
    )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
