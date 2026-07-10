from typing import get_args

from stan_ai_client import ClaudeEffort, CodexReasoningEffort, GrokEffort
from stan_ai_client.types import Effort, ReasoningEffort


def test_provider_specific_effort_types_are_public() -> None:
    assert get_args(ClaudeEffort) == ("low", "medium", "high", "max")
    assert get_args(CodexReasoningEffort) == (
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert get_args(GrokEffort) == ("low", "medium", "high", "max")


def test_legacy_effort_aliases_remain_compatible() -> None:
    assert Effort == ClaudeEffort
    assert ReasoningEffort == CodexReasoningEffort
