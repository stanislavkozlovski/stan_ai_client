from __future__ import annotations

from datetime import datetime

from .rate_limits import RateLimitInfo
from .types import ClaudeJsonPayload, CodexJsonPayload, CommandMetadata, GrokJsonPayload

AIPayload = ClaudeJsonPayload | CodexJsonPayload | GrokJsonPayload


class AIClientError(RuntimeError):
    pass


class ExecutableNotFoundError(AIClientError):
    pass


class AIClientTimeoutError(AIClientError):
    pass


class ProcessError(AIClientError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: AIPayload | None,
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.payload = payload
        super().__init__(message)


class ProtocolError(AIClientError):
    def __init__(self, message: str, *, command: CommandMetadata, stdout: str, stderr: str) -> None:
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(message)


class SchemaValidationError(AIClientError):
    pass


class StructuredOutputMissingError(ProtocolError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        stdout: str,
        stderr: str,
        payload: AIPayload,
    ) -> None:
        self.payload = payload
        super().__init__(message, command=command, stdout=stdout, stderr=stderr)


class StructuredOutputValidationError(ProtocolError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        stdout: str,
        stderr: str,
        payload: AIPayload,
        structured_output: object,
    ) -> None:
        self.payload = payload
        self.structured_output = structured_output
        super().__init__(message, command=command, stdout=stdout, stderr=stderr)


class LimitError(ProcessError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: AIPayload | None,
        limit: RateLimitInfo,
    ) -> None:
        self.limit = limit
        self.rate_limit = limit
        super().__init__(
            message,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
        )

    @property
    def retry_after_seconds(self) -> int | None:
        return self.limit.retry_after_seconds

    @property
    def reset_at(self) -> datetime | None:
        return self.limit.reset_at


class RateLimitError(LimitError):
    pass


class ClaudeCodeError(AIClientError):
    pass


class ClaudeSchemaValidationError(SchemaValidationError, ClaudeCodeError):
    pass


class ClaudeExecutableNotFoundError(ExecutableNotFoundError, ClaudeCodeError):
    def __init__(self, executable: str) -> None:
        self.executable = executable
        super().__init__(f"Claude executable not found: {executable}")


class ClaudeTimeoutError(AIClientTimeoutError, ClaudeCodeError):
    def __init__(self, command: CommandMetadata, timeout_seconds: float) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Claude command timed out after {timeout_seconds}s")


class ClaudeProcessError(ProcessError, ClaudeCodeError):
    payload: ClaudeJsonPayload | None


class ClaudeProtocolError(ProtocolError, ClaudeCodeError):
    pass


class ClaudeStructuredOutputMissingError(
    StructuredOutputMissingError, ClaudeProtocolError
):
    payload: ClaudeJsonPayload


class ClaudeStructuredOutputValidationError(
    StructuredOutputValidationError, ClaudeProtocolError
):
    payload: ClaudeJsonPayload


class ClaudeLimitError(LimitError, ClaudeProcessError):
    pass


class ClaudeRateLimitError(RateLimitError, ClaudeLimitError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: ClaudeJsonPayload | None,
        rate_limit: RateLimitInfo,
    ) -> None:
        super().__init__(
            message,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
            limit=rate_limit,
        )


class CodexCodeError(AIClientError):
    pass


class CodexSchemaValidationError(SchemaValidationError, CodexCodeError):
    pass


class GrokCodeError(AIClientError):
    pass


class GrokSchemaValidationError(SchemaValidationError, GrokCodeError):
    pass


class StructuredSchemaValidationError(
    ClaudeSchemaValidationError, CodexSchemaValidationError, GrokSchemaValidationError
):
    pass


class CodexExecutableNotFoundError(ExecutableNotFoundError, CodexCodeError):
    def __init__(self, executable: str) -> None:
        self.executable = executable
        super().__init__(f"Codex executable not found: {executable}")


class CodexTimeoutError(AIClientTimeoutError, CodexCodeError):
    def __init__(self, command: CommandMetadata, timeout_seconds: float) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Codex command timed out after {timeout_seconds}s")


class CodexProcessError(ProcessError, CodexCodeError):
    payload: CodexJsonPayload | None


class CodexProtocolError(ProtocolError, CodexCodeError):
    pass


class CodexStructuredOutputMissingError(StructuredOutputMissingError, CodexProtocolError):
    payload: CodexJsonPayload


class CodexStructuredOutputValidationError(
    StructuredOutputValidationError, CodexProtocolError
):
    payload: CodexJsonPayload


class CodexLimitError(LimitError, CodexProcessError):
    pass


class CodexRateLimitError(RateLimitError, CodexLimitError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: CodexJsonPayload | None,
        rate_limit: RateLimitInfo,
    ) -> None:
        super().__init__(
            message,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
            limit=rate_limit,
        )


# --- Grok errors (mirror the pattern of Claude/Codex) ---


class GrokExecutableNotFoundError(ExecutableNotFoundError, GrokCodeError):
    def __init__(self, executable: str) -> None:
        self.executable = executable
        super().__init__(f"Grok executable not found: {executable}")


class GrokTimeoutError(AIClientTimeoutError, GrokCodeError):
    def __init__(self, command: CommandMetadata, timeout_seconds: float) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Grok command timed out after {timeout_seconds}s")


class GrokProcessError(ProcessError, GrokCodeError):
    payload: GrokJsonPayload | None


class GrokProtocolError(ProtocolError, GrokCodeError):
    pass


class GrokStructuredOutputMissingError(
    StructuredOutputMissingError, GrokProtocolError
):
    payload: GrokJsonPayload


class GrokStructuredOutputValidationError(
    StructuredOutputValidationError, GrokProtocolError
):
    payload: GrokJsonPayload


class GrokLimitError(LimitError, GrokProcessError):
    pass


class GrokRateLimitError(RateLimitError, GrokLimitError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: GrokJsonPayload | None,
        rate_limit: RateLimitInfo,
    ) -> None:
        super().__init__(
            message,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
            limit=rate_limit,
        )
