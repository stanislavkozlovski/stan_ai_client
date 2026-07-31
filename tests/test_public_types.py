from typing import get_args

from stan_ai_client import (
    ClaudeCodeError,
    ClaudeEffort,
    ClaudeNetworkUnavailableError,
    ClaudeProcessError,
    CodexCodeError,
    CodexNetworkUnavailableError,
    CodexProcessError,
    CodexReasoningEffort,
    GrokCodeError,
    GrokEffort,
    GrokNetworkUnavailableError,
    GrokProcessError,
    NetworkUnavailableError,
)
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


def test_network_exception_types_are_public_and_provider_compatible() -> None:
    assert issubclass(ClaudeNetworkUnavailableError, NetworkUnavailableError)
    assert issubclass(ClaudeNetworkUnavailableError, ClaudeProcessError)
    assert issubclass(ClaudeNetworkUnavailableError, ClaudeCodeError)
    assert issubclass(CodexNetworkUnavailableError, NetworkUnavailableError)
    assert issubclass(CodexNetworkUnavailableError, CodexProcessError)
    assert issubclass(CodexNetworkUnavailableError, CodexCodeError)
    assert issubclass(GrokNetworkUnavailableError, NetworkUnavailableError)
    assert issubclass(GrokNetworkUnavailableError, GrokProcessError)
    assert issubclass(GrokNetworkUnavailableError, GrokCodeError)
