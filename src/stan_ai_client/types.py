from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Mapping, TypeVar

Effort = Literal["low", "medium", "high", "max"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
PermissionMode = Literal[
    "acceptEdits",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
    "auto",
]
CodexPermissionMode = Literal["default", "bypassPermissions"]
GrokPermissionMode = Literal[
    "acceptEdits",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
]
InputMode = Literal["stdin", "argv"]
TStructured = TypeVar("TStructured")


def _first_present(data: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


@dataclass(frozen=True)
class RunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    effort: Effort | None = None
    timeout_seconds: float | None = None
    input_mode: InputMode = "stdin"
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None
    add_dirs: tuple[str | Path, ...] | None = None
    permission_mode: PermissionMode | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    settings: str | None = None
    session_id: str | None = None
    continue_last_session: bool | None = None
    fork_session: bool | None = None
    extra_args: tuple[str, ...] | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CodexRunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    timeout_seconds: float | None = None
    input_mode: InputMode | None = None
    permission_mode: CodexPermissionMode | None = None
    session_id: str | None = None
    continue_last_session: bool | None = None
    skip_git_repo_check: bool | None = None
    ignore_user_config: bool | None = None
    ignore_rules: bool | None = None
    add_dirs: tuple[str | Path, ...] | None = None
    profile: str | None = None
    config_overrides: tuple[str, ...] | None = None
    extra_args: tuple[str, ...] | None = None
    resume_extra_args: tuple[str, ...] | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class GrokRunOptions:
    """Options for GrokClient.

    Prompt delivery (stdin vs argv) is handled transparently by GrokClient.
    No input_mode is exposed. Use --prompt-file automatically for very long prompts.
    """

    cwd: str | Path | None = None
    model: str | None = None
    effort: Effort | None = None
    timeout_seconds: float | None = None
    permission_mode: GrokPermissionMode | None = None
    session_id: str | None = None
    continue_last_session: bool | None = None
    fork_session: bool | None = None
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None
    system_prompt: str | None = None
    add_dirs: tuple[str | Path, ...] | None = None
    max_turns: int | None = None
    extra_args: tuple[str, ...] | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class RateLimitRetryPolicy:
    """Controls opt-in retry behavior for parseable rate-limit responses.

    Local AI CLIs often expose reset metadata in error text, so the policy is
    budget-based: callers decide how long an operation may wait.
    """

    max_wait_seconds: float | None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.max_wait_seconds is not None and self.max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be None or >= 0")


@dataclass(frozen=True)
class CommandMetadata:
    argv: tuple[str, ...]
    cwd: str | None
    elapsed_ms: float


@dataclass(frozen=True)
class ClaudeJsonPayload:
    type: str | None
    subtype: str | None
    is_error: bool | None
    duration_ms: int | None
    duration_api_ms: int | None
    num_turns: int | None
    result: str | None
    stop_reason: str | None
    session_id: str | None
    total_cost_usd: float | None
    structured_output: Any | None
    usage: dict[str, Any]
    model_usage: dict[str, dict[str, Any]]
    permission_denials: list[str]
    uuid: str | None
    extras: dict[str, Any]
    _structured_output_present: bool = field(default=False, repr=False)

    @property
    def has_structured_output(self) -> bool:
        return self._structured_output_present

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaudeJsonPayload":
        used = {
            "type",
            "subtype",
            "is_error",
            "duration_ms",
            "duration_api_ms",
            "num_turns",
            "result",
            "stop_reason",
            "session_id",
            "total_cost_usd",
            "structured_output",
            "usage",
            "modelUsage",
            "permission_denials",
            "uuid",
        }
        raw_usage = data.get("usage")
        raw_model_usage = data.get("modelUsage")
        raw_permission_denials = data.get("permission_denials")
        return cls(
            type=data.get("type"),
            subtype=data.get("subtype"),
            is_error=data.get("is_error"),
            duration_ms=data.get("duration_ms"),
            duration_api_ms=data.get("duration_api_ms"),
            num_turns=data.get("num_turns"),
            result=data.get("result"),
            stop_reason=data.get("stop_reason"),
            session_id=data.get("session_id"),
            total_cost_usd=data.get("total_cost_usd"),
            structured_output=data.get("structured_output"),
            usage=raw_usage if isinstance(raw_usage, dict) else {},
            model_usage=raw_model_usage if isinstance(raw_model_usage, dict) else {},
            permission_denials=(
                raw_permission_denials if isinstance(raw_permission_denials, list) else []
            ),
            uuid=data.get("uuid"),
            extras={key: value for key, value in data.items() if key not in used},
            _structured_output_present="structured_output" in data,
        )


@dataclass(frozen=True)
class CodexJsonPayload:
    thread_id: str | None
    result: str | None
    usage: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    error: dict[str, Any] | None
    structured_output: Any | None
    _structured_output_present: bool = field(default=False, repr=False)

    @property
    def has_structured_output(self) -> bool:
        return self._structured_output_present


@dataclass(frozen=True)
class GrokJsonPayload:
    """Payload returned by grok -p --output-format json.

    Much thinner than Claude's envelope. No usage/cost/num_turns.
    duration_ms is populated client-side by the wrapper.
    """

    text: str | None
    stop_reason: str | None
    session_id: str | None
    request_id: str | None
    thought: str | None
    structured_output: Any | None
    duration_ms: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    _structured_output_present: bool = field(default=False, repr=False)

    @property
    def has_structured_output(self) -> bool:
        return self._structured_output_present

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrokJsonPayload":
        # Grok uses camelCase in JSON: stopReason, sessionId, structuredOutput
        used = {
            "text",
            "stopReason",
            "stop_reason",
            "sessionId",
            "session_id",
            "requestId",
            "request_id",
            "thought",
            "structuredOutput",
            "structured_output",
            "duration_ms",
        }
        stop_reason = _first_present(data, "stopReason", "stop_reason")
        session_id = _first_present(data, "sessionId", "session_id")
        request_id = _first_present(data, "requestId", "request_id")
        structured_output = _first_present(data, "structuredOutput", "structured_output")
        return cls(
            text=data.get("text"),
            stop_reason=stop_reason,
            session_id=session_id,
            request_id=request_id,
            thought=data.get("thought"),
            structured_output=structured_output,
            duration_ms=data.get("duration_ms"),
            extras={key: value for key, value in data.items() if key not in used},
            _structured_output_present=bool(
                "structuredOutput" in data or "structured_output" in data
            ),
        )


@dataclass(frozen=True)
class TextRunResult:
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    text: str


@dataclass(frozen=True)
class JsonRunResult:
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: ClaudeJsonPayload


@dataclass(frozen=True)
class StructuredRunResult(Generic[TStructured]):
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: ClaudeJsonPayload
    structured_output: TStructured


@dataclass(frozen=True)
class CodexJsonRunResult:
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: CodexJsonPayload


@dataclass(frozen=True)
class CodexStructuredRunResult(Generic[TStructured]):
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: CodexJsonPayload
    structured_output: TStructured


@dataclass(frozen=True)
class GrokJsonRunResult:
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: GrokJsonPayload


@dataclass(frozen=True)
class GrokStructuredRunResult(Generic[TStructured]):
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: GrokJsonPayload
    structured_output: TStructured
