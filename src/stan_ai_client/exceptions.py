from __future__ import annotations

from datetime import datetime

from .rate_limits import RateLimitInfo
from .types import ClaudeJsonPayload, CommandMetadata


class ClaudeCodeError(RuntimeError):
    pass


class ClaudeSchemaValidationError(ClaudeCodeError):
    pass


class ClaudeExecutableNotFoundError(ClaudeCodeError):
    def __init__(self, executable: str) -> None:
        self.executable = executable
        super().__init__(f"Claude executable not found: {executable}")


class ClaudeTimeoutError(ClaudeCodeError):
    def __init__(self, command: CommandMetadata, timeout_seconds: float) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Claude command timed out after {timeout_seconds}s")


class ClaudeProcessError(ClaudeCodeError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: ClaudeJsonPayload | None,
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.payload = payload
        super().__init__(message)


class ClaudeProtocolError(ClaudeCodeError):
    def __init__(self, message: str, *, command: CommandMetadata, stdout: str, stderr: str) -> None:
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(message)


class ClaudeStructuredOutputMissingError(ClaudeProtocolError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        stdout: str,
        stderr: str,
        payload: ClaudeJsonPayload,
    ) -> None:
        self.payload = payload
        super().__init__(message, command=command, stdout=stdout, stderr=stderr)


class ClaudeStructuredOutputValidationError(ClaudeProtocolError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        stdout: str,
        stderr: str,
        payload: ClaudeJsonPayload,
        structured_output: object,
    ) -> None:
        self.payload = payload
        self.structured_output = structured_output
        super().__init__(message, command=command, stdout=stdout, stderr=stderr)


class ClaudeLimitError(ClaudeProcessError):
    def __init__(
        self,
        message: str,
        *,
        command: CommandMetadata,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: ClaudeJsonPayload | None,
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


class ClaudeRateLimitError(ClaudeLimitError):
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
