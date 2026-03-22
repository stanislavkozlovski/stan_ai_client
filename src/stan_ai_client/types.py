from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

Effort = Literal["low", "medium", "high", "max"]
PermissionMode = Literal[
    "acceptEdits",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
    "auto",
]
InputMode = Literal["stdin", "argv"]


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
    usage: dict[str, Any]
    model_usage: dict[str, dict[str, Any]]
    permission_denials: list[str]
    uuid: str | None
    extras: dict[str, Any]

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
            usage=raw_usage if isinstance(raw_usage, dict) else {},
            model_usage=raw_model_usage if isinstance(raw_model_usage, dict) else {},
            permission_denials=(
                raw_permission_denials if isinstance(raw_permission_denials, list) else []
            ),
            uuid=data.get("uuid"),
            extras={key: value for key, value in data.items() if key not in used},
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

