from typing import get_args

from stan_ai_client import (
    AIClientTimeoutError,
    AntigravityApprovalRequiredError,
    AntigravityCapacityError,
    AntigravityCodeError,
    AntigravityEffort,
    AntigravityLimitError,
    AntigravityNetworkUnavailableError,
    AntigravityPermissionMode,
    AntigravityProcessError,
    AntigravityProtocolError,
    AntigravityRateLimitError,
    AntigravitySchemaValidationError,
    AntigravityServerError,
    AntigravitySessionNotFoundError,
    AntigravityStructuredOutputMissingError,
    AntigravityStructuredOutputValidationError,
    AntigravityTimeoutError,
    ApprovalRequiredError,
    CapacityError,
    ClaudeApprovalRequiredError,
    ClaudeCodeError,
    ClaudeEffort,
    ClaudeNetworkUnavailableError,
    ClaudeProcessError,
    ClaudeSessionNotFoundError,
    CodexApprovalRequiredError,
    CodexCodeError,
    CodexNetworkUnavailableError,
    CodexPermissionMode,
    CodexProcessError,
    CodexReasoningEffort,
    GrokApprovalRequiredError,
    GrokCodeError,
    GrokEffort,
    GrokNetworkUnavailableError,
    GrokPermissionMode,
    GrokProcessError,
    LimitError,
    NetworkUnavailableError,
    PermissionMode,
    ProtocolError,
    RateLimitError,
    ServerError,
    StructuredOutputMissingError,
    StructuredOutputValidationError,
    StructuredSchemaValidationError,
)
from stan_ai_client.types import Effort, ReasoningEffort


def test_provider_specific_effort_types_are_public() -> None:
    assert get_args(ClaudeEffort) == ("low", "medium", "high", "xhigh", "max")
    assert get_args(CodexReasoningEffort) == (
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert get_args(GrokEffort) == ("low", "medium", "high", "xhigh", "max")


def test_legacy_effort_aliases_remain_compatible() -> None:
    assert Effort == ClaudeEffort
    assert ReasoningEffort == CodexReasoningEffort


def test_provider_permission_modes_expose_auto() -> None:
    assert set(get_args(PermissionMode)) == {
        "acceptEdits",
        "auto",
        "bypassPermissions",
        "default",
        "dontAsk",
        "plan",
    }
    assert get_args(CodexPermissionMode) == ("default", "bypassPermissions", "auto")
    assert GrokPermissionMode is PermissionMode


def test_approval_exception_types_are_public_and_provider_compatible() -> None:
    assert issubclass(ClaudeApprovalRequiredError, ApprovalRequiredError)
    assert issubclass(ClaudeApprovalRequiredError, ClaudeProcessError)
    assert issubclass(CodexApprovalRequiredError, ApprovalRequiredError)
    assert issubclass(CodexApprovalRequiredError, CodexProcessError)
    assert issubclass(GrokApprovalRequiredError, ApprovalRequiredError)
    assert issubclass(GrokApprovalRequiredError, GrokProcessError)


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


def test_missing_session_error_is_public_and_process_compatible() -> None:
    assert issubclass(ClaudeSessionNotFoundError, ClaudeProcessError)
    assert issubclass(ClaudeSessionNotFoundError, ClaudeCodeError)


def test_antigravity_effort_and_permission_vocabularies() -> None:
    assert get_args(AntigravityEffort) == ("low", "medium", "high", "xhigh", "max")
    assert get_args(AntigravityPermissionMode) == (
        "default",
        "acceptEdits",
        "plan",
        "auto",
        "bypassPermissions",
    )


def test_antigravity_exception_hierarchy_matches_other_providers() -> None:
    for error_type in (
        AntigravityApprovalRequiredError,
        AntigravityCapacityError,
        AntigravityServerError,
        AntigravityNetworkUnavailableError,
        AntigravityRateLimitError,
        AntigravitySessionNotFoundError,
    ):
        assert issubclass(error_type, AntigravityProcessError)
        assert issubclass(error_type, AntigravityCodeError)
    assert issubclass(AntigravityApprovalRequiredError, ApprovalRequiredError)
    assert issubclass(AntigravityCapacityError, CapacityError)
    assert issubclass(AntigravityServerError, ServerError)
    assert issubclass(AntigravityNetworkUnavailableError, NetworkUnavailableError)
    assert issubclass(AntigravityRateLimitError, RateLimitError)
    assert issubclass(AntigravityRateLimitError, AntigravityLimitError)
    assert issubclass(AntigravityLimitError, LimitError)
    assert issubclass(AntigravityProtocolError, ProtocolError)
    assert issubclass(AntigravityStructuredOutputMissingError, StructuredOutputMissingError)
    assert issubclass(AntigravityStructuredOutputMissingError, AntigravityProtocolError)
    assert issubclass(
        AntigravityStructuredOutputValidationError, StructuredOutputValidationError
    )
    assert issubclass(AntigravityTimeoutError, AIClientTimeoutError)
    assert issubclass(AntigravityTimeoutError, AntigravityCodeError)
    assert issubclass(StructuredSchemaValidationError, AntigravitySchemaValidationError)
