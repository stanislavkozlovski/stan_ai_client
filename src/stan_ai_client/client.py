from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from typing import Any, Mapping, TypeVar

from jsonschema.exceptions import ValidationError

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
    RunOptions,
    StructuredRunResult,
    TextRunResult,
)

DEFAULT_LOGGER = logging.getLogger("stan_ai_client")
REDACTED_ARG_FLAGS = {
    "--append-system-prompt",
    "--resume",
    "--settings",
    "--system-prompt",
}
TStructured = TypeVar("TStructured")


@dataclass(frozen=True)
class ResolvedRunOptions:
    cwd: str | Path | None
    model: str
    effort: Effort
    timeout_seconds: float
    input_mode: str
    allowed_tools: tuple[str, ...] | None
    disallowed_tools: tuple[str, ...] | None
    tools: tuple[str, ...] | None
    add_dirs: tuple[str | Path, ...] | None
    permission_mode: str | None
    system_prompt: str | None
    append_system_prompt: str | None
    settings: str | None
    session_id: str | None
    continue_last_session: bool
    fork_session: bool
    extra_args: tuple[str, ...] | None
    env: Mapping[str, str] | None


class ClaudeCodeClient:
    def __init__(
        self,
        *,
        executable: str = "claude",
        default_model: str = "claude-opus-4-6",
        default_effort: Effort = "max",
        default_timeout_seconds: float = 120.0,
        default_options: RunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None:
        self.executable = executable
        self.default_model = default_model
        self.default_effort = default_effort
        self.default_timeout_seconds = default_timeout_seconds
        self.default_options = default_options or RunOptions()
        self.logger = logger or DEFAULT_LOGGER
        self.log_prompts = log_prompts

    def run_text(self, prompt: str, *, options: RunOptions | None = None) -> TextRunResult:
        prepared, effective = self._prepare(prompt, output_format="text", options=options)
        self._log_start(prompt, output_format="text", prepared=prepared, effective=effective)
        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr
        parsed_payload = try_parse_json_payload(stdout)

        if completed.returncode != 0:
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=parsed_payload,
            )

        if parsed_payload is not None and parsed_payload.is_error:
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=parsed_payload,
            )

        result = TextRunResult(
            command=metadata,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            text=stdout.strip(),
        )
        self._log_finish(
            output_format="text",
            metadata=metadata,
            stdout=stdout,
            stderr=stderr,
            payload=parsed_payload,
        )
        return result

    def run_json(self, prompt: str, *, options: RunOptions | None = None) -> JsonRunResult:
        prepared, effective = self._prepare(prompt, output_format="json", options=options)
        self._log_start(prompt, output_format="json", prepared=prepared, effective=effective)
        completed, metadata, payload = self._execute_json(prepared, protocol_name="JSON mode")

        result = JsonRunResult(
            command=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            payload=payload,
        )
        self._log_finish(
            output_format="json",
            metadata=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )
        return result

    def run_structured(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        options: RunOptions | None = None,
    ) -> StructuredRunResult[TStructured]:
        prepared, effective = self._prepare(
            prompt,
            output_format="json",
            options=options,
            json_schema=schema,
        )
        self._log_start(prompt, output_format="json", prepared=prepared, effective=effective)
        self.logger.debug("Claude structured mode enabled schema_validated_locally=True")

        completed, metadata, payload = self._execute_json(
            prepared,
            protocol_name="structured mode",
        )

        if not payload.has_structured_output:
            self.logger.debug("Claude structured_output missing")
            missing_error = ClaudeStructuredOutputMissingError(
                "Claude did not return structured_output in structured mode",
                command=metadata,
                stdout=completed.stdout,
                stderr=completed.stderr,
                payload=payload,
            )
            self._log_protocol_error(missing_error)
            raise missing_error

        self.logger.debug("Claude structured_output present")

        try:
            structured_output = schema.validate_response(payload.structured_output)
        except ValidationError as exc:
            self.logger.debug("Claude structured_output validation failed error=%s", exc.message)
            validation_error = ClaudeStructuredOutputValidationError(
                f"Claude returned structured_output that does not match the schema: {exc.message}",
                command=metadata,
                stdout=completed.stdout,
                stderr=completed.stderr,
                payload=payload,
                structured_output=payload.structured_output,
            )
            self._log_protocol_error(validation_error)
            raise validation_error from exc

        self.logger.debug("Claude structured_output validation succeeded")

        result = StructuredRunResult(
            command=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            payload=payload,
            structured_output=structured_output,
        )
        self._log_finish(
            output_format="json",
            metadata=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )
        return result

    def _prepare(
        self,
        prompt: str,
        *,
        output_format: str,
        options: RunOptions | None,
        json_schema: StructuredSchema[Any] | None = None,
    ) -> tuple[PreparedCommand, ResolvedRunOptions]:
        effective = self._resolve_options(options)
        argv = [self.executable]

        if effective.session_id is not None and effective.continue_last_session:
            raise ValueError("RunOptions cannot set both session_id and continue_last_session")

        if effective.session_id is not None:
            argv.extend(["--resume", effective.session_id])
        elif effective.continue_last_session:
            argv.append("--continue")

        if effective.fork_session:
            argv.append("--fork-session")

        argv.append("-p")
        argv.extend(["--output-format", output_format])
        if json_schema is not None:
            argv.extend(["--json-schema", json_schema.cli_json])
        argv.extend(["--model", effective.model])
        argv.extend(["--effort", effective.effort])

        if effective.allowed_tools is not None:
            argv.extend(["--allowed-tools", ",".join(effective.allowed_tools)])
        if effective.disallowed_tools is not None:
            argv.extend(["--disallowed-tools", ",".join(effective.disallowed_tools)])
        if effective.tools is not None:
            argv.extend(["--tools", ",".join(effective.tools)])
        if effective.permission_mode is not None:
            argv.extend(["--permission-mode", effective.permission_mode])
        if effective.system_prompt is not None:
            argv.extend(["--system-prompt", effective.system_prompt])
        if effective.append_system_prompt is not None:
            argv.extend(["--append-system-prompt", effective.append_system_prompt])
        if effective.settings is not None:
            argv.extend(["--settings", effective.settings])
        if effective.add_dirs is not None:
            for directory in effective.add_dirs:
                argv.extend(["--add-dir", str(directory)])
        if effective.extra_args is not None:
            argv.extend(effective.extra_args)

        input_text: str | None = None
        if effective.input_mode == "stdin":
            input_text = prompt
        else:
            argv.append(prompt)

        merged_env = os.environ.copy()
        if effective.env is not None:
            merged_env.update(effective.env)

        cwd = None if effective.cwd is None else str(effective.cwd)
        prepared = PreparedCommand(
            argv=tuple(argv),
            cwd=cwd,
            timeout_seconds=effective.timeout_seconds,
            input_text=input_text,
            env=merged_env,
        )
        return prepared, effective

    def _execute_json(
        self,
        prepared: PreparedCommand,
        *,
        protocol_name: str,
    ) -> tuple[CompletedProcess[str], CommandMetadata, ClaudeJsonPayload]:
        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr
        payload = try_parse_json_payload(stdout)

        if completed.returncode != 0:
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )

        if payload is None:
            error = self._build_json_protocol_error(
                metadata,
                stdout=stdout,
                stderr=stderr,
                protocol_name=protocol_name,
            )
            self._log_protocol_error(error)
            raise error

        if payload.is_error:
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )

        return completed, metadata, payload

    def _execute(
        self, prepared: PreparedCommand
    ) -> tuple[CompletedProcess[str], CommandMetadata]:
        started_at = time.monotonic()
        try:
            completed = execute_command(prepared)
        except FileNotFoundError as exc:
            self.logger.error("Claude executable not found executable=%s", self.executable)
            raise ClaudeExecutableNotFoundError(self.executable) from exc
        except TimeoutExpired as exc:
            metadata = CommandMetadata(
                argv=prepared.argv,
                cwd=prepared.cwd,
                elapsed_ms=(time.monotonic() - started_at) * 1000,
            )
            self.logger.warning(
                "Claude run timed out timeout_seconds=%.1f cwd=%s argv=%s elapsed_ms=%.0f",
                prepared.timeout_seconds,
                prepared.cwd,
                _redact_argv(prepared.argv, prompt_in_argv=prepared.input_text is None),
                metadata.elapsed_ms,
            )
            raise ClaudeTimeoutError(metadata, prepared.timeout_seconds) from exc

        metadata = CommandMetadata(
            argv=prepared.argv,
            cwd=prepared.cwd,
            elapsed_ms=(time.monotonic() - started_at) * 1000,
        )
        return completed, metadata

    def _build_process_error(
        self,
        command: CommandMetadata,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: ClaudeJsonPayload | None,
    ) -> ClaudeProcessError:
        error_text = summarize_error_text(payload=payload, stdout=stdout, stderr=stderr)
        if is_rate_limit_text(error_text):
            rate_limit = parse_rate_limit_info(error_text)
            self.logger.warning(
                "Claude run failed returncode=%d elapsed_ms=%.0f error=%s retry_after_seconds=%s reset_at=%s",
                returncode,
                command.elapsed_ms,
                error_text,
                rate_limit.retry_after_seconds,
                rate_limit.reset_at,
            )
            return ClaudeRateLimitError(
                error_text,
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
                rate_limit=rate_limit,
            )

        self.logger.warning(
            "Claude run failed returncode=%d elapsed_ms=%.0f error=%s",
            returncode,
            command.elapsed_ms,
            error_text,
        )
        return ClaudeProcessError(
            error_text,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
        )

    def _resolve_options(self, options: RunOptions | None) -> ResolvedRunOptions:
        override = options or RunOptions()
        default = self.default_options
        cwd = override.cwd if override.cwd is not None else default.cwd
        model = override.model if override.model is not None else (
            default.model if default.model is not None else self.default_model
        )
        effort = override.effort if override.effort is not None else (
            default.effort if default.effort is not None else self.default_effort
        )
        timeout_seconds = (
            override.timeout_seconds
            if override.timeout_seconds is not None
            else (
                default.timeout_seconds
                if default.timeout_seconds is not None
                else self.default_timeout_seconds
            )
        )
        input_mode = override.input_mode if override.input_mode is not None else default.input_mode
        allowed_tools = (
            override.allowed_tools if override.allowed_tools is not None else default.allowed_tools
        )
        disallowed_tools = (
            override.disallowed_tools
            if override.disallowed_tools is not None
            else default.disallowed_tools
        )
        tools = override.tools if override.tools is not None else default.tools
        add_dirs = override.add_dirs if override.add_dirs is not None else default.add_dirs
        permission_mode = (
            override.permission_mode
            if override.permission_mode is not None
            else default.permission_mode
        )
        system_prompt = (
            override.system_prompt if override.system_prompt is not None else default.system_prompt
        )
        append_system_prompt = (
            override.append_system_prompt
            if override.append_system_prompt is not None
            else default.append_system_prompt
        )
        settings = override.settings if override.settings is not None else default.settings
        session_id = override.session_id if override.session_id is not None else default.session_id
        continue_last_session = (
            override.continue_last_session
            if override.continue_last_session is not None
            else (
                default.continue_last_session
                if default.continue_last_session is not None
                else False
            )
        )
        fork_session = (
            override.fork_session
            if override.fork_session is not None
            else (default.fork_session if default.fork_session is not None else False)
        )
        extra_args = (
            override.extra_args if override.extra_args is not None else default.extra_args
        )
        env = override.env if override.env is not None else default.env

        return ResolvedRunOptions(
            cwd=cwd,
            model=model,
            effort=effort,
            timeout_seconds=float(timeout_seconds),
            input_mode=input_mode,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            tools=tools,
            add_dirs=add_dirs,
            permission_mode=permission_mode,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            settings=settings,
            session_id=session_id,
            continue_last_session=continue_last_session,
            fork_session=fork_session,
            extra_args=extra_args,
            env=env,
        )

    def _log_start(
        self,
        prompt: str,
        *,
        output_format: str,
        prepared: PreparedCommand,
        effective: ResolvedRunOptions,
    ) -> None:
        self.logger.info(
            "Claude run starting output_format=%s model=%s effort=%s cwd=%s input_mode=%s timeout_seconds=%.1f prompt_chars=%d resume=%s continue=%s fork=%s",
            output_format,
            effective.model,
            effective.effort,
            prepared.cwd,
            effective.input_mode,
            effective.timeout_seconds,
            len(prompt),
            effective.session_id is not None,
            effective.continue_last_session,
            effective.fork_session,
        )
        self.logger.debug(
            "Claude argv=%s",
            _redact_argv(prepared.argv, prompt_in_argv=prepared.input_text is None),
        )
        if self.log_prompts:
            self.logger.debug("Claude prompt=%s", prompt)

    def _log_finish(
        self,
        *,
        output_format: str,
        metadata: CommandMetadata,
        stdout: str,
        stderr: str,
        payload: ClaudeJsonPayload | None,
    ) -> None:
        self.logger.info(
            "Claude run finished output_format=%s returncode=0 elapsed_ms=%.0f stdout_chars=%d stderr_chars=%d",
            output_format,
            metadata.elapsed_ms,
            len(stdout),
            len(stderr),
        )
        if payload is not None:
            self.logger.debug(
                "Claude payload session_id=%s total_cost_usd=%s duration_ms=%s stop_reason=%s is_error=%s",
                payload.session_id,
                payload.total_cost_usd,
                payload.duration_ms,
                payload.stop_reason,
                payload.is_error,
            )

    def _log_protocol_error(self, error: ClaudeProtocolError) -> None:
        self.logger.warning(
            "Claude protocol error elapsed_ms=%.0f error=%s",
            error.command.elapsed_ms,
            str(error),
        )

    def _build_json_protocol_error(
        self,
        command: CommandMetadata,
        *,
        stdout: str,
        stderr: str,
        protocol_name: str,
    ) -> ClaudeProtocolError:
        if not stdout.strip():
            return ClaudeProtocolError(
                f"Claude returned empty output in {protocol_name}",
                command=command,
                stdout=stdout,
                stderr=stderr,
            )

        return ClaudeProtocolError(
            f"Claude returned non-JSON output in {protocol_name}: {stdout.strip()[:500]}",
            command=command,
            stdout=stdout,
            stderr=stderr,
        )


def _redact_argv(argv: tuple[str, ...], *, prompt_in_argv: bool) -> tuple[str, ...]:
    redacted: list[str] = []
    redact_next = False
    for index, value in enumerate(argv):
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if value in REDACTED_ARG_FLAGS:
            redacted.append(value)
            redact_next = True
            continue
        if prompt_in_argv and index == len(argv) - 1:
            redacted.append("<prompt>")
            continue
        redacted.append(value)
    return tuple(redacted)
