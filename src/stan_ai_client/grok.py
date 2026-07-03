from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from typing import Any, Callable, Mapping, NoReturn, TypeVar

from jsonschema.exceptions import ValidationError

from ._options import first_set, first_set_or
from ._retry import run_with_rate_limit_retry
from .exceptions import (
    GrokExecutableNotFoundError,
    GrokProcessError,
    GrokProtocolError,
    GrokRateLimitError,
    GrokStructuredOutputMissingError,
    GrokStructuredOutputValidationError,
    GrokTimeoutError,
)
from .grok_parser import (
    is_grok_error_payload,
    summarize_grok_error_text,
    try_parse_grok_json_payload,
)
from .rate_limits import is_rate_limit_text, parse_rate_limit_info
from .schema import StructuredSchema
from .transport import PreparedCommand, execute_command
from .types import (
    CommandMetadata,
    Effort,
    GrokJsonPayload,
    GrokJsonRunResult,
    GrokPermissionMode,
    GrokRunOptions,
    GrokStructuredRunResult,
    RateLimitRetryPolicy,
    TextRunResult,
)

DEFAULT_LOGGER = logging.getLogger("stan_ai_client")
REDACTED_ARG_FLAGS = {
    "--system-prompt-override",
    "--resume",
    "-r",
    "--session-id",
    "-s",
}
JSON_SCHEMA_ARG_FLAG = "--json-schema"
PROMPT_FILE_THRESHOLD = 4096
TRun = TypeVar("TRun")
TStructured = TypeVar("TStructured")


@dataclass(frozen=True)
class ResolvedGrokRunOptions:
    cwd: str | Path | None
    model: str
    effort: Effort | None
    timeout_seconds: float
    permission_mode: GrokPermissionMode | None
    session_id: str | None
    continue_last_session: bool
    fork_session: bool
    allowed_tools: tuple[str, ...] | None
    disallowed_tools: tuple[str, ...] | None
    tools: tuple[str, ...] | None
    system_prompt: str | None
    add_dirs: tuple[str | Path, ...] | None
    max_turns: int | None
    extra_args: tuple[str, ...] | None
    env: Mapping[str, str] | None


@dataclass(frozen=True)
class PreparedGrokCommand(PreparedCommand):
    prompt_file_path: str | None = None


class GrokClient:
    def __init__(
        self,
        *,
        executable: str = "grok",
        default_model: str = "grok-build",
        default_effort: Effort | None = None,
        default_timeout_seconds: float = 120.0,
        default_options: GrokRunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None:
        self.executable = executable
        self.default_model = default_model
        self.default_effort = default_effort
        self.default_timeout_seconds = default_timeout_seconds
        self.default_options = default_options or GrokRunOptions()
        self.logger = logger or DEFAULT_LOGGER
        self.log_prompts = log_prompts

    def run_text(
        self,
        prompt: str,
        *,
        options: GrokRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
    ) -> TextRunResult:
        return self._run_with_rate_limit_policy(
            lambda: self._run_text_once(prompt, options=options),
            rate_limit_policy=rate_limit_policy,
        )

    def _run_text_once(self, prompt: str, *, options: GrokRunOptions | None = None) -> TextRunResult:
        prepared, effective = self._prepare(prompt, output_format="plain", options=options)
        self._log_start(prompt, output_format="plain", prepared=prepared, effective=effective)
        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr

        if completed.returncode != 0:
            payload = try_parse_grok_json_payload(stdout)
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )

        payload = try_parse_grok_json_payload(stdout)
        if payload is not None:
            payload = replace(payload, duration_ms=int(metadata.elapsed_ms))
        if payload is not None and is_grok_error_payload(payload):
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )

        result = TextRunResult(
            command=metadata,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            text=stdout.strip(),
        )
        self._log_finish(
            output_format="plain",
            metadata=metadata,
            stdout=stdout,
            stderr=stderr,
            payload=None,
        )
        return result

    def run_json(
        self,
        prompt: str,
        *,
        options: GrokRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
    ) -> GrokJsonRunResult:
        return self._run_with_rate_limit_policy(
            lambda: self._run_json_once(prompt, options=options),
            rate_limit_policy=rate_limit_policy,
        )

    def _run_json_once(self, prompt: str, *, options: GrokRunOptions | None = None) -> GrokJsonRunResult:
        prepared, effective = self._prepare(prompt, output_format="json", options=options)
        self._log_start(prompt, output_format="json", prepared=prepared, effective=effective)
        completed, metadata, payload = self._execute_json(prepared, protocol_name="JSON mode")

        result = GrokJsonRunResult(
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
        options: GrokRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
    ) -> GrokStructuredRunResult[TStructured]:
        return self._run_with_rate_limit_policy(
            lambda: self._run_structured_once(prompt, schema=schema, options=options),
            rate_limit_policy=rate_limit_policy,
        )

    def _run_structured_once(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        options: GrokRunOptions | None = None,
    ) -> GrokStructuredRunResult[TStructured]:
        prepared, effective = self._prepare(
            prompt,
            output_format="json",
            options=options,
            json_schema=schema,
        )
        self._log_start(prompt, output_format="json", prepared=prepared, effective=effective)
        self.logger.debug("Grok structured mode enabled schema_validated_locally=True")

        completed, metadata, payload = self._execute_json(
            prepared,
            protocol_name="structured mode",
            raw_structured_output=True,
        )

        envelope_payload = self._parse_envelope_payload(completed.stdout, metadata)
        if (
            envelope_payload is not None
            and _is_grok_envelope_metadata_payload(envelope_payload)
            and not envelope_payload.has_structured_output
        ):
            self.logger.debug("Grok structuredOutput missing")
            missing_error = GrokStructuredOutputMissingError(
                "Grok did not return structuredOutput in structured mode",
                command=metadata,
                stdout=completed.stdout,
                stderr=completed.stderr,
                payload=envelope_payload,
            )
            self._log_protocol_error(missing_error)
            raise missing_error

        if not payload.has_structured_output:
            self.logger.debug("Grok structuredOutput missing")
            missing_error = GrokStructuredOutputMissingError(
                "Grok did not return structuredOutput in structured mode",
                command=metadata,
                stdout=completed.stdout,
                stderr=completed.stderr,
                payload=payload,
            )
            self._log_protocol_error(missing_error)
            raise missing_error

        self.logger.debug("Grok structuredOutput present")

        envelope_validation_error: ValidationError | None = None
        if (
            envelope_payload is not None
            and _is_grok_structured_envelope_payload(envelope_payload)
        ):
            try:
                structured_output = schema.validate_response(envelope_payload.structured_output)
            except ValidationError as exc:
                envelope_validation_error = exc
                self.logger.debug(
                    "Grok envelope structuredOutput validation failed; trying raw JSON value"
                )
            else:
                self.logger.debug("Grok envelope structuredOutput validation succeeded")
                payload = envelope_payload
                return self._build_structured_result(
                    completed=completed,
                    metadata=metadata,
                    payload=payload,
                    structured_output=structured_output,
                )

        try:
            structured_output = schema.validate_response(payload.structured_output)
        except ValidationError as exc:
            if envelope_payload is not None and is_grok_error_payload(envelope_payload):
                raise self._build_process_error(
                    metadata,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    payload=envelope_payload,
                )
            payload, structured_output = self._validate_envelope_structured_output_or_raise(
                schema=schema,
                completed=completed,
                metadata=metadata,
                raw_payload=payload,
                envelope_payload=envelope_payload,
                validation_error=envelope_validation_error or exc,
            )

        self.logger.debug("Grok structuredOutput validation succeeded")

        return self._build_structured_result(
            completed=completed,
            metadata=metadata,
            payload=payload,
            structured_output=structured_output,
        )

    def _build_structured_result(
        self,
        *,
        completed: CompletedProcess[str],
        metadata: CommandMetadata,
        payload: GrokJsonPayload,
        structured_output: TStructured,
    ) -> GrokStructuredRunResult[TStructured]:
        result = GrokStructuredRunResult(
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

    def _parse_envelope_payload(
        self,
        stdout: str,
        metadata: CommandMetadata,
    ) -> GrokJsonPayload | None:
        envelope_payload = try_parse_grok_json_payload(stdout)
        if envelope_payload is None:
            return None
        return replace(envelope_payload, duration_ms=int(metadata.elapsed_ms))

    def _validate_envelope_structured_output_or_raise(
        self,
        *,
        schema: StructuredSchema[TStructured],
        completed: CompletedProcess[str],
        metadata: CommandMetadata,
        raw_payload: GrokJsonPayload,
        envelope_payload: GrokJsonPayload | None,
        validation_error: ValidationError,
    ) -> tuple[GrokJsonPayload, TStructured]:
        if envelope_payload is not None and envelope_payload.has_structured_output:
            try:
                structured_output = schema.validate_response(envelope_payload.structured_output)
            except ValidationError:
                pass
            else:
                self.logger.debug("Grok envelope structuredOutput validation succeeded")
                return envelope_payload, structured_output

        self._raise_structured_validation_error(
            schema_error=validation_error,
            completed=completed,
            metadata=metadata,
            payload=raw_payload,
        )

    def _raise_structured_validation_error(
        self,
        *,
        schema_error: ValidationError,
        completed: CompletedProcess[str],
        metadata: CommandMetadata,
        payload: GrokJsonPayload,
    ) -> NoReturn:
        self.logger.debug("Grok structuredOutput validation failed error=%s", schema_error.message)
        error = GrokStructuredOutputValidationError(
            f"Grok returned structured output that does not match the schema: {schema_error.message}",
            command=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
            structured_output=payload.structured_output,
        )
        self._log_protocol_error(error)
        raise error from schema_error

    def _run_with_rate_limit_policy(
        self,
        operation: Callable[[], TRun],
        *,
        rate_limit_policy: RateLimitRetryPolicy | None,
    ) -> TRun:
        return run_with_rate_limit_retry(
            operation,
            rate_limit_policy=rate_limit_policy,
            logger=self.logger,
            provider="Grok",
            rate_limit_error_type=GrokRateLimitError,
        )

    def _prepare(
        self,
        prompt: str,
        *,
        output_format: str,
        options: GrokRunOptions | None,
        json_schema: StructuredSchema[Any] | None = None,
    ) -> tuple[PreparedGrokCommand, ResolvedGrokRunOptions]:
        effective = self._resolve_options(options)
        argv: list[str] = [self.executable]

        if effective.session_id is not None and effective.continue_last_session:
            raise ValueError("GrokRunOptions cannot set both session_id and continue_last_session")

        if effective.session_id is not None:
            argv.extend(["--session-id", effective.session_id])
        elif effective.continue_last_session:
            argv.append("--continue")

        if effective.fork_session:
            argv.append("--fork-session")

        # Transparent prompt delivery
        prompt_file_path = None
        if len(prompt) > PROMPT_FILE_THRESHOLD:
            fd, tmp_path = tempfile.mkstemp(prefix="grok_prompt_", suffix=".txt", text=True)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(prompt)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise
            argv.extend(["--prompt-file", tmp_path])
            prompt_file_path = tmp_path
        else:
            argv.extend(["-p", prompt])

        argv.extend(["--output-format", output_format])
        if json_schema is not None:
            argv.extend(["--json-schema", json_schema.cli_json])

        argv.extend(["--model", effective.model])
        if effective.effort is not None:
            argv.extend(["--effort", effective.effort])

        if effective.allowed_tools is not None:
            for rule in effective.allowed_tools:
                argv.extend(["--allow", rule])
        if effective.disallowed_tools is not None:
            for rule in effective.disallowed_tools:
                argv.extend(["--deny", rule])
        if effective.tools is not None:
            argv.extend(["--tools", ",".join(effective.tools)])
        if effective.permission_mode is not None:
            argv.extend(["--permission-mode", effective.permission_mode])
        if effective.system_prompt is not None:
            argv.extend(["--system-prompt-override", effective.system_prompt])
        if effective.max_turns is not None:
            argv.extend(["--max-turns", str(effective.max_turns)])
        if effective.extra_args is not None:
            argv.extend(effective.extra_args)

        input_text: str | None = None

        merged_env = os.environ.copy()
        if effective.env is not None:
            merged_env.update(effective.env)

        cwd = None if effective.cwd is None else str(effective.cwd)
        prepared = PreparedGrokCommand(
            argv=tuple(argv),
            cwd=cwd,
            timeout_seconds=effective.timeout_seconds,
            input_text=input_text,
            env=merged_env,
            prompt_file_path=prompt_file_path,
        )
        return prepared, effective

    def _execute(self, prepared: PreparedGrokCommand) -> tuple[CompletedProcess[str], CommandMetadata]:
        started_at = time.monotonic()
        tmp_path = prepared.prompt_file_path
        try:
            completed = execute_command(prepared)
        except FileNotFoundError as exc:
            if prepared.cwd is not None and exc.filename == prepared.cwd:
                metadata = CommandMetadata(
                    argv=prepared.argv,
                    cwd=prepared.cwd,
                    elapsed_ms=(time.monotonic() - started_at) * 1000,
                )
                self.logger.error("Grok working directory not found cwd=%s", prepared.cwd)
                raise GrokProcessError(
                    f"Grok working directory not found: {prepared.cwd}",
                    command=metadata,
                    returncode=127,
                    stdout="",
                    stderr="",
                    payload=None,
                ) from exc

            self.logger.error("Grok executable not found executable=%s", self.executable)
            raise GrokExecutableNotFoundError(self.executable) from exc
        except TimeoutExpired as exc:
            metadata = CommandMetadata(
                argv=prepared.argv,
                cwd=prepared.cwd,
                elapsed_ms=(time.monotonic() - started_at) * 1000,
            )
            self.logger.warning(
                "Grok run timed out timeout_seconds=%.1f cwd=%s argv=%s elapsed_ms=%.0f",
                prepared.timeout_seconds,
                prepared.cwd,
                _redact_argv(prepared.argv),
                metadata.elapsed_ms,
            )
            raise GrokTimeoutError(metadata, prepared.timeout_seconds) from exc
        finally:
            self._cleanup_tmp(tmp_path)

        metadata = CommandMetadata(
            argv=prepared.argv,
            cwd=prepared.cwd,
            elapsed_ms=(time.monotonic() - started_at) * 1000,
        )
        return completed, metadata

    def _cleanup_tmp(self, tmp_path: str | None) -> None:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                self.logger.debug("Failed to cleanup grok prompt temp file %s: %s", tmp_path, exc)

    def _execute_json(
        self,
        prepared: PreparedGrokCommand,
        *,
        protocol_name: str,
        raw_structured_output: bool = False,
    ) -> tuple[CompletedProcess[str], CommandMetadata, GrokJsonPayload]:
        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr
        payload = try_parse_grok_json_payload(
            stdout,
            raw_structured_output=raw_structured_output,
        )
        if payload is not None:
            payload = replace(payload, duration_ms=int(metadata.elapsed_ms))

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

        if is_grok_error_payload(payload):
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )

        return completed, metadata, payload

    def _build_process_error(
        self,
        command: CommandMetadata,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        payload: GrokJsonPayload | None,
    ) -> GrokProcessError:
        error_text = summarize_grok_error_text(payload=payload, stdout=stdout, stderr=stderr)
        if (
            is_rate_limit_text(error_text)
            or "rate limit" in error_text.lower()
            or "429" in error_text
            or "resource-exhausted" in error_text
            or "You've hit the rate limit" in error_text
        ):
            rate_limit = parse_rate_limit_info(error_text)
            self.logger.warning(
                "Grok run failed returncode=%d elapsed_ms=%.0f error=%s retry_after_seconds=%s reset_at=%s",
                returncode,
                command.elapsed_ms,
                error_text,
                rate_limit.retry_after_seconds,
                rate_limit.reset_at,
            )
            return GrokRateLimitError(
                error_text,
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
                rate_limit=rate_limit,
            )

        self.logger.warning(
            "Grok run failed returncode=%d elapsed_ms=%.0f error=%s",
            returncode,
            command.elapsed_ms,
            error_text,
        )
        return GrokProcessError(
            error_text,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
        )

    def _resolve_options(self, options: GrokRunOptions | None) -> ResolvedGrokRunOptions:
        override = options or GrokRunOptions()
        default = self.default_options
        model = first_set_or(override.model, default.model, default=self.default_model)
        effort = first_set(override.effort, default.effort)
        if effort is None:
            effort = self.default_effort
        return ResolvedGrokRunOptions(
            cwd=first_set(override.cwd, default.cwd),
            model=model,
            effort=effort,
            timeout_seconds=float(
                first_set_or(
                    override.timeout_seconds,
                    default.timeout_seconds,
                    default=self.default_timeout_seconds,
                )
            ),
            permission_mode=first_set(override.permission_mode, default.permission_mode),
            session_id=first_set(override.session_id, default.session_id),
            continue_last_session=first_set_or(
                override.continue_last_session, default.continue_last_session, default=False
            ),
            fork_session=first_set_or(
                override.fork_session, default.fork_session, default=False
            ),
            allowed_tools=first_set(override.allowed_tools, default.allowed_tools),
            disallowed_tools=first_set(override.disallowed_tools, default.disallowed_tools),
            tools=first_set(override.tools, default.tools),
            system_prompt=first_set(override.system_prompt, default.system_prompt),
            add_dirs=first_set(override.add_dirs, default.add_dirs),
            max_turns=first_set(override.max_turns, default.max_turns),
            extra_args=first_set(override.extra_args, default.extra_args),
            env=first_set(override.env, default.env),
        )

    def _log_start(
        self,
        prompt: str,
        *,
        output_format: str,
        prepared: PreparedCommand,
        effective: ResolvedGrokRunOptions,
    ) -> None:
        self.logger.info(
            "Grok run starting output_format=%s model=%s effort=%s cwd=%s timeout_seconds=%.1f prompt_chars=%d resume=%s continue=%s fork=%s",
            output_format,
            effective.model,
            effective.effort,
            prepared.cwd,
            effective.timeout_seconds,
            len(prompt),
            effective.session_id is not None,
            effective.continue_last_session,
            effective.fork_session,
        )
        self.logger.debug(
            "Grok argv=%s",
            _redact_argv(prepared.argv),
        )
        if self.log_prompts:
            self.logger.debug("Grok prompt=%s", prompt)

    def _log_finish(
        self,
        *,
        output_format: str,
        metadata: CommandMetadata,
        stdout: str,
        stderr: str,
        payload: GrokJsonPayload | None,
    ) -> None:
        self.logger.info(
            "Grok run finished output_format=%s returncode=0 elapsed_ms=%.0f stdout_chars=%d stderr_chars=%d",
            output_format,
            metadata.elapsed_ms,
            len(stdout),
            len(stderr),
        )
        if payload is not None:
            self.logger.debug(
                "Grok payload session_id=%s stop_reason=%s has_structured=%s",
                payload.session_id,
                payload.stop_reason,
                payload.has_structured_output,
            )

    def _log_protocol_error(self, error: GrokProtocolError) -> None:
        self.logger.warning(
            "Grok protocol error elapsed_ms=%.0f error=%s",
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
    ) -> GrokProtocolError:
        if not stdout.strip():
            return GrokProtocolError(
                f"Grok returned empty output in {protocol_name}",
                command=command,
                stdout=stdout,
                stderr=stderr,
            )

        return GrokProtocolError(
            f"Grok returned non-JSON output in {protocol_name}: {stdout.strip()[:500]}",
            command=command,
            stdout=stdout,
            stderr=stderr,
        )


def _redact_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    redacted: list[str] = []
    replacement_for_next: str | None = None
    for index, value in enumerate(argv):
        if replacement_for_next is not None:
            redacted.append(replacement_for_next)
            replacement_for_next = None
            continue
        if value in REDACTED_ARG_FLAGS:
            redacted.append(value)
            replacement_for_next = "<redacted>"
            continue
        if value == JSON_SCHEMA_ARG_FLAG:
            redacted.append(value)
            replacement_for_next = "<json-schema>"
            continue
        if value in ("-p", "--prompt-file") and index + 1 < len(argv):
            redacted.append(value)
            replacement_for_next = "<prompt>"
            continue
        redacted.append(value)
    return tuple(redacted)


def _is_grok_structured_envelope_payload(payload: GrokJsonPayload) -> bool:
    return payload.has_structured_output and _is_grok_envelope_metadata_payload(payload)


def _is_grok_envelope_metadata_payload(payload: GrokJsonPayload) -> bool:
    return any(
        value is not None
        for value in (
            payload.stop_reason,
            payload.session_id,
            payload.request_id,
            payload.thought,
        )
    )
