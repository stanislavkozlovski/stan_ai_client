from __future__ import annotations

from .claude import (
    DEFAULT_LOGGER,
    JSON_SCHEMA_ARG_FLAG,
    REDACTED_ARG_FLAGS,
    ClaudeCodeClient,
    ResolvedRunOptions,
)
from .exceptions import (
    ClaudeExecutableNotFoundError,
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeRateLimitError,
    ClaudeStructuredOutputMissingError,
    ClaudeStructuredOutputValidationError,
    ClaudeTimeoutError,
)
from .parser import summarize_error_text, try_parse_json_payload
from .rate_limits import is_rate_limit_text, parse_rate_limit_info
from .schema import StructuredSchema
from .transport import PreparedCommand, execute_command
from .types import (
    ClaudeJsonPayload,
    CommandMetadata,
    Effort,
    JsonRunResult,
    RateLimitRetryPolicy,
    RunOptions,
    StructuredRunResult,
    TextRunResult,
)

__all__ = [
    "ClaudeExecutableNotFoundError",
    "ClaudeCodeClient",
    "ClaudeJsonPayload",
    "ClaudeProcessError",
    "ClaudeProtocolError",
    "ClaudeRateLimitError",
    "ClaudeStructuredOutputMissingError",
    "ClaudeStructuredOutputValidationError",
    "ClaudeTimeoutError",
    "CommandMetadata",
    "DEFAULT_LOGGER",
    "Effort",
    "JSON_SCHEMA_ARG_FLAG",
    "JsonRunResult",
    "PreparedCommand",
    "RateLimitRetryPolicy",
    "REDACTED_ARG_FLAGS",
    "ResolvedRunOptions",
    "RunOptions",
    "StructuredRunResult",
    "StructuredSchema",
    "TextRunResult",
    "execute_command",
    "is_rate_limit_text",
    "parse_rate_limit_info",
    "summarize_error_text",
    "try_parse_json_payload",
]
