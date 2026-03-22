from __future__ import annotations

from .rate_limits import RateLimitInfo
from .types import ClaudeJsonPayload, CommandMetadata


class ClaudeCodeError(RuntimeError):
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


class ClaudeRateLimitError(ClaudeProcessError):
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
        self.rate_limit = rate_limit
        super().__init__(
            message,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
        )

