from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PreparedCommand:
    argv: tuple[str, ...]
    cwd: str | None
    timeout_seconds: float
    input_text: str | None
    env: Mapping[str, str] | None


def execute_command(command: PreparedCommand) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command.argv),
        cwd=command.cwd,
        text=True,
        capture_output=True,
        timeout=command.timeout_seconds,
        input=command.input_text,
        env=dict(command.env) if command.env is not None else None,
    )

