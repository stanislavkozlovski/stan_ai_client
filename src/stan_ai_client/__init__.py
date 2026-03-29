from ._version import get_version
from .client import ClaudeCodeClient
from .exceptions import (
    ClaudeCodeError,
    ClaudeExecutableNotFoundError,
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeRateLimitError,
    ClaudeSchemaValidationError,
    ClaudeStructuredOutputMissingError,
    ClaudeStructuredOutputValidationError,
    ClaudeTimeoutError,
)
from .rate_limits import RateLimitInfo, parse_rate_limit_info
from .schema import StructuredSchema
from .types import (
    ClaudeJsonPayload,
    CommandMetadata,
    JsonRunResult,
    RunOptions,
    StructuredRunResult,
    TextRunResult,
)

__version__ = get_version()

__all__ = [
    "__version__",
    "ClaudeCodeClient",
    "ClaudeCodeError",
    "ClaudeExecutableNotFoundError",
    "ClaudeJsonPayload",
    "ClaudeProcessError",
    "ClaudeProtocolError",
    "ClaudeRateLimitError",
    "ClaudeSchemaValidationError",
    "ClaudeStructuredOutputMissingError",
    "ClaudeStructuredOutputValidationError",
    "ClaudeTimeoutError",
    "CommandMetadata",
    "JsonRunResult",
    "RateLimitInfo",
    "RunOptions",
    "StructuredRunResult",
    "StructuredSchema",
    "TextRunResult",
    "parse_rate_limit_info",
]
