from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pytest

from stan_ai_client import (
    ClaudeJsonPayload,
    GrokJsonPayload,
    TokenUsage,
    normalize_ai_usage,
)
from stan_ai_client.codex_parser import parse_codex_jsonl_payload


def codex(
    input_tokens: Any = 1000, cached: Any = 800, output: Any = 200
) -> dict[str, Any]:
    return {
        "thread_id": "session",
        "usage_scope": "invocation",
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "output_tokens": output,
        },
    }


def claude_model(
    input_tokens: int, cached: int, written: int, output: int
) -> dict[str, Any]:
    return {
        "inputTokens": input_tokens,
        "cacheReadInputTokens": cached,
        "cacheCreationInputTokens": written,
        "outputTokens": output,
    }


def test_codex_cache_and_reasoning_are_subsets_not_additional_tokens() -> None:
    payload = codex()
    payload["usage"]["reasoning_output_tokens"] = 150
    facts = normalize_ai_usage("codex", payload)
    assert facts.tokens == TokenUsage(200, 800, None, 200, 150, 0, 1200)
    assert facts.models[0].tokens == facts.tokens
    assert facts.models[0].model_source == "unknown"
    assert facts.reported_cost_usd is None


def test_codex_exposed_cache_writes_are_also_inside_input() -> None:
    payload = codex()
    payload["usage"]["cache_write_input_tokens"] = 100
    assert normalize_ai_usage("codex", payload).tokens == TokenUsage(
        100, 800, 100, 200, None, 0, 1200
    )


def test_impossible_cache_split_preserves_independent_total() -> None:
    facts = normalize_ai_usage("codex", codex(cached=1100))
    assert facts.tokens == TokenUsage(None, None, None, 200, None, 1000, 1200)
    assert facts.raw["usage"]["cached_input_tokens"] == 1100
    assert facts.diagnostics


@pytest.mark.parametrize(
    "value", [True, False, -1, 1.5, 1.0, "1000", 2**63, float("nan"), [], {}]
)
def test_invalid_counts_are_unavailable_and_preserved_raw(value: Any) -> None:
    facts = normalize_ai_usage("codex", codex(input_tokens=value))
    assert facts.tokens.fresh_input_tokens is None
    assert facts.tokens.total_tokens is None
    assert facts.tokens.output_tokens == 200
    assert facts.diagnostics
    assert "input_tokens" in facts.raw["usage"]


def test_zero_and_missing_are_different() -> None:
    zero = normalize_ai_usage("codex", codex(0, 0, 0))
    missing = normalize_ai_usage("codex", {})
    assert zero.tokens == TokenUsage(0, 0, None, 0, None, 0, 0)
    assert missing.tokens == TokenUsage()


def test_sum_overflow_is_unavailable() -> None:
    facts = normalize_ai_usage("codex", codex(2**63 - 1, 0, 1))
    assert facts.tokens.total_tokens is None
    assert facts.diagnostics


def test_reasoning_cannot_exceed_output() -> None:
    payload = codex()
    payload["usage"]["reasoning_output_tokens"] = 201
    facts = normalize_ai_usage("codex", payload)
    assert facts.tokens.reasoning_output_tokens is None
    assert facts.tokens.total_tokens == 1200


def test_claude_models_include_subagents_and_override_top_level_usage() -> None:
    raw = {
        "session_id": "session",
        "total_cost_usd": 0.25,
        "plan_type": "max",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "modelUsage": {
            "parent": {**claude_model(10, 100, 20, 30), "costUSD": 0.2},
            "subagent": {**claude_model(5, 50, 10, 15), "costUSD": 0.05},
        },
    }
    for payload in (raw, ClaudeJsonPayload.from_dict(raw)):
        facts = normalize_ai_usage("claude", payload)
        assert facts.tokens == TokenUsage(15, 150, 30, 45, None, 0, 240)
        assert sum(row.tokens.total_tokens or 0 for row in facts.models) == 240
        assert [row.model for row in facts.models] == ["parent", "subagent"]
        assert facts.reported_cost_usd == 0.25 and facts.plan_type == "max"
        assert facts.models[0].reported_cost_usd == 0.2


def test_claude_top_level_fallback_is_explicitly_incomplete() -> None:
    facts = normalize_ai_usage(
        "claude",
        {
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 0,
                "output_tokens": 30,
            },
            "requested_model": "claude",
        },
    )
    assert facts.tokens.total_tokens == 60
    assert facts.models[0].model_source == "requested"
    assert "Claude model attribution incomplete" in facts.diagnostics


def test_partial_model_breakdown_preserves_known_documented_components() -> None:
    facts = normalize_ai_usage(
        "claude",
        {
            "modelUsage": {
                "a": {
                    "outputTokens": 10,
                    "totalTokens": 100,
                    "reasoningOutputTokens": 9,
                    "future": {"value": 1},
                },
                "b": claude_model(10, 20, 30, 40),
            }
        },
    )
    assert facts.tokens == TokenUsage(10, 20, 30, 50)
    assert facts.models[0].tokens == TokenUsage(output_tokens=10)
    for name in (
        "fresh_input_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
    ):
        assert getattr(facts.tokens, name) == sum(
            getattr(row.tokens, name) or 0 for row in facts.models
        )
    assert facts.raw["modelUsage"]["a"]["totalTokens"] == 100
    assert facts.raw["modelUsage"]["a"]["reasoningOutputTokens"] == 9
    assert facts.raw["modelUsage"]["a"]["future"] == {"value": 1}


def test_undocumented_claude_totals_and_reasoning_remain_raw() -> None:
    facts = normalize_ai_usage(
        "claude",
        {
            "modelUsage": {
                "a": {
                    **claude_model(10, 20, 30, 40),
                    "totalTokens": 50,
                    "reasoningOutputTokens": 20,
                }
            }
        },
    )
    assert facts.tokens == TokenUsage(10, 20, 30, 40, None, 0, 100)
    assert facts.models[0].tokens == facts.tokens
    assert facts.raw["modelUsage"]["a"]["totalTokens"] == 50
    assert facts.raw["modelUsage"]["a"]["reasoningOutputTokens"] == 20


def test_partial_claude_top_level_usage_does_not_invent_a_total() -> None:
    facts = normalize_ai_usage(
        "claude",
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 999,
                "reasoning_output_tokens": 5,
            }
        },
    )
    assert facts.tokens == TokenUsage(fresh_input_tokens=10, output_tokens=20)
    assert facts.raw["usage"]["total_tokens"] == 999
    assert facts.raw["usage"]["reasoning_output_tokens"] == 5


def test_unusable_multi_model_breakdown_is_not_assigned_to_requested_model() -> None:
    facts = normalize_ai_usage(
        "claude",
        {
            "modelUsage": {"a": {}, "b": {}},
            "usage": {
                "input_tokens": 123,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 0,
            },
            "requested_model": "a",
        },
    )
    assert facts.tokens == TokenUsage(123, 0, 0, 0, None, 0, 123)
    assert facts.models[0].model is None


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), "1", 10**1000])
def test_cost_validation_never_prices_tokens(value: Any) -> None:
    facts = normalize_ai_usage("claude", {"total_cost_usd": value})
    assert facts.reported_cost_usd is None


def test_reported_zero_cost_is_retained() -> None:
    assert normalize_ai_usage("claude", {"total_cost_usd": 0}).reported_cost_usd == 0.0


def test_grok_keeps_identity_and_unknown_usage() -> None:
    facts = normalize_ai_usage(
        "grok", GrokJsonPayload.from_dict({"sessionId": "grok-1", "text": "ok"})
    )
    assert facts.session_id == "grok-1" and facts.tokens == TokenUsage()


def test_legacy_aggregate_stays_unsplit_and_normalization_does_not_mutate_payload() -> (
    None
):
    payload: dict[str, Any] = {"total_tokens": 123, "future": {"value": [1]}}
    original = deepcopy(payload)
    facts = normalize_ai_usage("legacy", payload)
    assert facts.tokens == TokenUsage(unsplit_tokens=123, total_tokens=123)
    assert payload == original
    facts.raw["future"]["value"].append(2)
    assert payload == original
    assert asdict(facts)["raw"]["total_tokens"] == 123


def test_fresh_resume_and_duplicate_terminals_count_current_invocation_once() -> None:
    fresh = parse_codex_jsonl_payload(
        '{"type":"turn.completed","usage":{"input_tokens":1000,"cached_input_tokens":800,"output_tokens":200}}'
    )
    resumed = parse_codex_jsonl_payload(
        '{"type":"turn.completed","usage":{"input_tokens":400,"cached_input_tokens":300,"output_tokens":10}}\n'
        * 2
    )
    # Explicit invocation scope (e.g. a future CLI) must never be subtracted.
    fresh = replace(fresh, usage_scope="invocation")
    resumed = replace(resumed, usage_scope="invocation")
    assert normalize_ai_usage("codex", fresh).tokens.total_tokens == 1200
    assert (
        normalize_ai_usage(
            "codex", resumed, previous_snapshot=fresh
        ).tokens.total_tokens
        == 410
    )


def test_codex_normalizer_uses_payload_fields_without_reinterpreting_events() -> None:
    payload = {
        **codex(10, 0, 5),
        "events": [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 1000,
                    "cached_input_tokens": 800,
                    "output_tokens": 200,
                },
            }
        ],
    }
    facts = normalize_ai_usage("codex", payload)
    assert facts.tokens.total_tokens == 15
    assert facts.raw["events"] == payload["events"]


def cumulative(input_tokens: Any = 1000) -> dict[str, Any]:
    return {
        **codex(input_tokens, 0, 200),
        "usage_scope": "cumulative",
    }


def test_confirmed_cumulative_counters_use_preceding_same_session_snapshot() -> None:
    facts = normalize_ai_usage(
        "codex", cumulative(1500), previous_snapshot=codex(1000, 0, 200)
    )
    assert facts.tokens.total_tokens == 500
    assert facts.tokens.output_tokens == 0
    assert facts.raw["usage"]["input_tokens"] == 1500


@pytest.mark.parametrize(
    "previous",
    [
        None,
        {**codex(), "thread_id": "other"},
        {**codex(), "usage_scope": "unknown"},
        cumulative(2000),
        {
            **codex(),
            "usage": {"input_tokens": 1000, "cached_input_tokens": 0},
        },
        codex(True, 0, 200),
    ],
)
def test_ambiguous_baseline_or_reset_leaves_delta_unavailable(
    previous: dict[str, Any] | None,
) -> None:
    facts = normalize_ai_usage("codex", cumulative(1500), previous_snapshot=previous)
    assert facts.tokens == TokenUsage()
    assert all(row.tokens == TokenUsage() for row in facts.models)
    assert facts.diagnostics


def test_cumulative_delta_allows_optional_counters_missing_from_both_snapshots() -> None:
    previous = {
        "thread_id": "session",
        "usage_scope": "invocation",
        "usage": {"input_tokens": 1000, "output_tokens": 100},
    }
    current = {
        "thread_id": "session",
        "usage_scope": "cumulative",
        "usage": {"input_tokens": 1500, "output_tokens": 120},
    }
    facts = normalize_ai_usage("codex", current, previous_snapshot=previous)
    assert facts.tokens == TokenUsage(
        output_tokens=20, unsplit_tokens=500, total_tokens=520
    )


def test_cumulative_delta_revalidates_reasoning_as_an_output_subset() -> None:
    previous = {
        **codex(1000, 0, 100),
        "usage": {
            **codex(1000, 0, 100)["usage"],
            "reasoning_output_tokens": 90,
        },
    }
    current = {
        **cumulative(1100),
        "usage": {
            **codex(1100, 0, 110)["usage"],
            "reasoning_output_tokens": 105,
        },
    }
    facts = normalize_ai_usage("codex", current, previous_snapshot=previous)
    assert facts.tokens.total_tokens == 110
    assert facts.tokens.output_tokens == 10
    assert facts.tokens.reasoning_output_tokens is None
    assert "reasoning subset cannot be verified against output" in facts.diagnostics


def test_unknown_counter_scope_is_not_guessed() -> None:
    assert (
        normalize_ai_usage("codex", {**codex(), "usage_scope": "unknown"}).tokens
        == TokenUsage()
    )


def test_live_codex_0_154_0_resume_fixture_uses_session_delta() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "codex_usage_0_154_0.json").read_text()
    )
    fresh = normalize_ai_usage("codex", fixture["fresh"])
    resumed = normalize_ai_usage(
        "codex", fixture["resume"], previous_snapshot=fixture["fresh"]
    )
    assert fresh.tokens.total_tokens == 17500
    assert resumed.tokens == TokenUsage(236, 17280, 0, 15, 0, 0, 17531)
    assert normalize_ai_usage("codex", fixture["resume"]).tokens.total_tokens is None
    assert resumed.raw["usage"]["output_tokens"] == 30
