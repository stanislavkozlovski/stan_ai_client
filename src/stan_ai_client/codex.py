from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from typing import Any, Callable, Mapping, TypeVar

from jsonschema.exceptions import ValidationError

from .codex_parser import (
    make_codex_structured_payload,
    summarize_codex_error_text,
    try_parse_codex_jsonl_payload,
)
from .exceptions import (
    CodexExecutableNotFoundError,
    CodexProcessError,
    CodexProtocolError,
    CodexRateLimitError,
    CodexStructuredOutputMissingError,
    CodexStructuredOutputValidationError,
    CodexTimeoutError,
)
from .rate_limits import is_rate_limit_text, parse_rate_limit_info
from .schema import StructuredSchema
from .transport import PreparedCommand, execute_command
from .types import (
    CodexJsonPayload,
    CodexJsonRunResult,
    CodexPermissionMode,
    CodexRunOptions,
    CodexStructuredRunResult,
    CommandMetadata,
    InputMode,
    RateLimitRetryPolicy,
    ReasoningEffort,
    TextRunResult,
)

DEFAULT_LOGGER = logging.getLogger("stan_ai_client")
BYPASS_APPROVALS_AND_SANDBOX_FLAG = "--dangerously-bypass-approvals-and-sandbox"
OUTPUT_SCHEMA_ARG_FLAG = "--output-schema"
REDACTED_ARG_FLAGS = {
    "-c",
    "--config",
    OUTPUT_SCHEMA_ARG_FLAG,
}
TRun = TypeVar("TRun")
TStructured = TypeVar("TStructured")


@dataclass(frozen=True)
class ResolvedCodexRunOptions:
    cwd: str | Path | None
    model: str
    reasoning_effort: ReasoningEffort
    timeout_seconds: float
    input_mode: InputMode
    permission_mode: CodexPermissionMode
    session_id: str | None
    continue_last_session: bool
    skip_git_repo_check: bool
    ignore_user_config: bool
    ignore_rules: bool
    add_dirs: tuple[str | Path, ...] | None
    profile: str | None
    config_overrides: tuple[str, ...] | None
    extra_args: tuple[str, ...] | None
    env: Mapping[str, str] | None


class CodexClient:
    def __init__(
        self,
        *,
        executable: str = "codex",
        default_model: str = "gpt-5.5",
        default_reasoning_effort: ReasoningEffort = "medium",
        default_permission_mode: CodexPermissionMode = "bypassPermissions",
        default_timeout_seconds: float = 120.0,
        default_options: CodexRunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None:
        self.executable = executable
        self.default_model = default_model
        self.default_reasoning_effort = default_reasoning_effort
        self.default_permission_mode = default_permission_mode
        self.default_timeout_seconds = default_timeout_seconds
        self.default_options = default_options or CodexRunOptions()
        self.logger = logger or DEFAULT_LOGGER
        self.log_prompts = log_prompts

    def run_text(
        self,
        prompt: str,
        *,
        options: CodexRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
    ) -> TextRunResult:
        return self._run_with_rate_limit_policy(
            lambda: self._run_text_once(prompt, options=options),
            rate_limit_policy=rate_limit_policy,
        )

    def _run_text_once(
        self, prompt: str, *, options: CodexRunOptions | None = None
    ) -> TextRunResult:
        prepared, effective = self._prepare(prompt, options=options)
        self._log_start(prompt, output_format="text", prepared=prepared, effective=effective)
        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr

        if completed.returncode != 0:
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=None,
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
            payload=None,
        )
        return result

    def run_json(
        self,
        prompt: str,
        *,
        options: CodexRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
    ) -> CodexJsonRunResult:
        return self._run_with_rate_limit_policy(
            lambda: self._run_json_once(prompt, options=options),
            rate_limit_policy=rate_limit_policy,
        )

    def _run_json_once(
        self, prompt: str, *, options: CodexRunOptions | None = None
    ) -> CodexJsonRunResult:
        prepared, effective = self._prepare(prompt, options=options, json_output=True)
        self._log_start(prompt, output_format="jsonl", prepared=prepared, effective=effective)
        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr
        payload = try_parse_codex_jsonl_payload(stdout)

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
                protocol_name="JSONL mode",
            )
            self._log_protocol_error(error)
            raise error

        if payload.error is not None:
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )

        result = CodexJsonRunResult(
            command=metadata,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            payload=payload,
        )
        self._log_finish(
            output_format="jsonl",
            metadata=metadata,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
        )
        return result

    def run_structured(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        options: CodexRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
    ) -> CodexStructuredRunResult[TStructured]:
        return self._run_with_rate_limit_policy(
            lambda: self._run_structured_once(prompt, schema=schema, options=options),
            rate_limit_policy=rate_limit_policy,
        )

    def _run_structured_once(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        options: CodexRunOptions | None = None,
    ) -> CodexStructuredRunResult[TStructured]:
        schema_path = self._write_schema_file(schema)
        try:
            return self._run_structured_with_schema_file(
                prompt,
                schema=schema,
                schema_path=schema_path,
                options=options,
            )
        finally:
            try:
                os.unlink(schema_path)
            except FileNotFoundError:
                pass

    def _run_structured_with_schema_file(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        schema_path: str,
        options: CodexRunOptions | None,
    ) -> CodexStructuredRunResult[TStructured]:
        prepared, effective = self._prepare(
            prompt,
            options=options,
            output_schema_path=schema_path,
        )
        self._log_start(prompt, output_format="structured", prepared=prepared, effective=effective)
        self.logger.debug("Codex structured mode enabled schema_validated_locally=True")

        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr

        if completed.returncode != 0:
            raise self._build_process_error(
                metadata,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                payload=None,
            )

        if not stdout.strip():
            payload = make_codex_structured_payload(None)
            missing_error = CodexStructuredOutputMissingError(
                "Codex returned empty output in structured mode",
                command=metadata,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )
            self._log_protocol_error(missing_error)
            raise missing_error

        try:
            raw_structured_output = json.loads(stdout)
        except json.JSONDecodeError as exc:
            error = CodexProtocolError(
                f"Codex returned non-JSON output in structured mode: {stdout.strip()[:500]}",
                command=metadata,
                stdout=stdout,
                stderr=stderr,
            )
            self._log_protocol_error(error)
            raise error from exc

        payload = make_codex_structured_payload(raw_structured_output)

        try:
            structured_output = schema.validate_response(raw_structured_output)
        except ValidationError as exc:
            self.logger.debug("Codex structured_output validation failed error=%s", exc.message)
            validation_error = CodexStructuredOutputValidationError(
                f"Codex returned structured output that does not match the schema: {exc.message}",
                command=metadata,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
                structured_output=raw_structured_output,
            )
            self._log_protocol_error(validation_error)
            raise validation_error from exc

        self.logger.debug("Codex structured_output validation succeeded")
        result = CodexStructuredRunResult(
            command=metadata,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            payload=payload,
            structured_output=structured_output,
        )
        self._log_finish(
            output_format="structured",
            metadata=metadata,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
        )
        return result

    def _run_with_rate_limit_policy(
        self,
        operation: Callable[[], TRun],
        *,
        rate_limit_policy: RateLimitRetryPolicy | None,
    ) -> TRun:
        if rate_limit_policy is None:
            return operation()

        total_wait_seconds = 0.0
        attempt = 0

        while True:
            attempt += 1
            try:
                return operation()
            except CodexRateLimitError as exc:
                wait_seconds = exc.retry_after_seconds
                if wait_seconds is None:
                    self.logger.warning(
                        "Codex rate limited but no retry metadata was parsed attempt=%d total_wait_seconds=%.1f max_wait_seconds=%s reset_at=%s label=%s",
                        attempt,
                        total_wait_seconds,
                        rate_limit_policy.max_wait_seconds,
                        exc.reset_at,
                        rate_limit_policy.label,
                    )
                    raise

                wait_seconds_float = float(wait_seconds)
                if wait_seconds_float <= 0:
                    self.logger.warning(
                        "Codex rate limited with non-positive retry wait attempt=%d wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%s reset_at=%s label=%s",
                        attempt,
                        wait_seconds_float,
                        total_wait_seconds,
                        rate_limit_policy.max_wait_seconds,
                        exc.reset_at,
                        rate_limit_policy.label,
                    )
                    raise

                if rate_limit_policy.max_wait_seconds is not None:
                    remaining_wait_seconds = (
                        rate_limit_policy.max_wait_seconds - total_wait_seconds
                    )
                    if wait_seconds_float > remaining_wait_seconds:
                        self.logger.warning(
                            "Codex rate limit exceeds wait budget attempt=%d wait_seconds=%.1f remaining_wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%.1f reset_at=%s label=%s",
                            attempt,
                            wait_seconds_float,
                            remaining_wait_seconds,
                            total_wait_seconds,
                            rate_limit_policy.max_wait_seconds,
                            exc.reset_at,
                            rate_limit_policy.label,
                        )
                        raise

                total_wait_seconds += wait_seconds_float
                self.logger.warning(
                    "Codex rate limited; retrying after reset attempt=%d wait_seconds=%.1f total_wait_seconds=%.1f max_wait_seconds=%s retry_after_seconds=%s reset_at=%s label=%s",
                    attempt,
                    wait_seconds_float,
                    total_wait_seconds,
                    rate_limit_policy.max_wait_seconds,
                    exc.retry_after_seconds,
                    exc.reset_at,
                    rate_limit_policy.label,
                )
                time.sleep(wait_seconds_float)

    def _prepare(
        self,
        prompt: str,
        *,
        options: CodexRunOptions | None,
        json_output: bool = False,
        output_schema_path: str | None = None,
    ) -> tuple[PreparedCommand, ResolvedCodexRunOptions]:
        effective = self._resolve_options(options)
        if effective.session_id is not None and effective.continue_last_session:
            raise ValueError("CodexRunOptions cannot set both session_id and continue_last_session")

        argv = [self.executable, "exec"]
        is_resume = effective.session_id is not None or effective.continue_last_session
        if is_resume:
            argv.append("resume")

        self._append_common_args(
            argv,
            effective=effective,
            json_output=json_output,
            output_schema_path=output_schema_path,
        )

        if effective.continue_last_session:
            argv.append("--last")
        elif effective.session_id is not None:
            argv.append(effective.session_id)

        input_text: str | None = None
        if effective.input_mode == "stdin":
            argv.append("-")
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

    def _append_common_args(
        self,
        argv: list[str],
        *,
        effective: ResolvedCodexRunOptions,
        json_output: bool,
        output_schema_path: str | None,
    ) -> None:
        if effective.model:
            argv.extend(["--model", effective.model])
        if effective.reasoning_effort:
            argv.extend(["-c", f'model_reasoning_effort="{effective.reasoning_effort}"'])
        if effective.permission_mode == "bypassPermissions":
            argv.append(BYPASS_APPROVALS_AND_SANDBOX_FLAG)
        if effective.cwd is not None:
            argv.extend(["--cd", str(effective.cwd)])
        if effective.profile is not None:
            argv.extend(["--profile", effective.profile])
        if effective.config_overrides is not None:
            for override in effective.config_overrides:
                argv.extend(["-c", override])
        if effective.skip_git_repo_check:
            argv.append("--skip-git-repo-check")
        if effective.ignore_user_config:
            argv.append("--ignore-user-config")
        if effective.ignore_rules:
            argv.append("--ignore-rules")
        if json_output:
            argv.append("--json")
        if output_schema_path is not None:
            argv.extend([OUTPUT_SCHEMA_ARG_FLAG, output_schema_path])
        if effective.add_dirs is not None:
            for directory in effective.add_dirs:
                argv.extend(["--add-dir", str(directory)])
        if effective.extra_args is not None:
            argv.extend(effective.extra_args)

    def _execute(
        self, prepared: PreparedCommand
    ) -> tuple[CompletedProcess[str], CommandMetadata]:
        started_at = time.monotonic()
        try:
            completed = execute_command(prepared)
        except FileNotFoundError as exc:
            self.logger.error("Codex executable not found executable=%s", self.executable)
            raise CodexExecutableNotFoundError(self.executable) from exc
        except TimeoutExpired as exc:
            metadata = CommandMetadata(
                argv=prepared.argv,
                cwd=prepared.cwd,
                elapsed_ms=(time.monotonic() - started_at) * 1000,
            )
            self.logger.warning(
                "Codex run timed out timeout_seconds=%.1f cwd=%s argv=%s elapsed_ms=%.0f",
                prepared.timeout_seconds,
                prepared.cwd,
                _redact_argv(prepared.argv, prompt_in_argv=prepared.input_text is None),
                metadata.elapsed_ms,
            )
            raise CodexTimeoutError(metadata, prepared.timeout_seconds) from exc

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
        payload: CodexJsonPayload | None,
    ) -> CodexProcessError:
        error_text = summarize_codex_error_text(payload=payload, stdout=stdout, stderr=stderr)
        if is_rate_limit_text(error_text):
            rate_limit = parse_rate_limit_info(error_text)
            self.logger.warning(
                "Codex run failed returncode=%d elapsed_ms=%.0f error=%s retry_after_seconds=%s reset_at=%s",
                returncode,
                command.elapsed_ms,
                error_text,
                rate_limit.retry_after_seconds,
                rate_limit.reset_at,
            )
            return CodexRateLimitError(
                error_text,
                command=command,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
                rate_limit=rate_limit,
            )

        self.logger.warning(
            "Codex run failed returncode=%d elapsed_ms=%.0f error=%s",
            returncode,
            command.elapsed_ms,
            error_text,
        )
        return CodexProcessError(
            error_text,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            payload=payload,
        )

    def _resolve_options(self, options: CodexRunOptions | None) -> ResolvedCodexRunOptions:
        override = options or CodexRunOptions()
        default = self.default_options
        cwd = override.cwd if override.cwd is not None else default.cwd
        model = override.model if override.model is not None else (
            default.model if default.model is not None else self.default_model
        )
        reasoning_effort = (
            override.reasoning_effort
            if override.reasoning_effort is not None
            else (
                default.reasoning_effort
                if default.reasoning_effort is not None
                else self.default_reasoning_effort
            )
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
        permission_mode = (
            override.permission_mode
            if override.permission_mode is not None
            else (
                default.permission_mode
                if default.permission_mode is not None
                else self.default_permission_mode
            )
        )
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
        skip_git_repo_check = (
            override.skip_git_repo_check
            if override.skip_git_repo_check is not None
            else (
                default.skip_git_repo_check
                if default.skip_git_repo_check is not None
                else False
            )
        )
        ignore_user_config = (
            override.ignore_user_config
            if override.ignore_user_config is not None
            else (default.ignore_user_config if default.ignore_user_config is not None else False)
        )
        ignore_rules = (
            override.ignore_rules
            if override.ignore_rules is not None
            else (default.ignore_rules if default.ignore_rules is not None else False)
        )
        input_mode = override.input_mode if override.input_mode is not None else default.input_mode
        add_dirs = override.add_dirs if override.add_dirs is not None else default.add_dirs
        profile = override.profile if override.profile is not None else default.profile
        config_overrides = (
            override.config_overrides
            if override.config_overrides is not None
            else default.config_overrides
        )
        extra_args = (
            override.extra_args if override.extra_args is not None else default.extra_args
        )
        env = override.env if override.env is not None else default.env

        return ResolvedCodexRunOptions(
            cwd=cwd,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=float(timeout_seconds),
            input_mode=input_mode,
            permission_mode=permission_mode,
            session_id=session_id,
            continue_last_session=continue_last_session,
            skip_git_repo_check=skip_git_repo_check,
            ignore_user_config=ignore_user_config,
            ignore_rules=ignore_rules,
            add_dirs=add_dirs,
            profile=profile,
            config_overrides=config_overrides,
            extra_args=extra_args,
            env=env,
        )

    def _log_start(
        self,
        prompt: str,
        *,
        output_format: str,
        prepared: PreparedCommand,
        effective: ResolvedCodexRunOptions,
    ) -> None:
        self.logger.info(
            "Codex run starting output_format=%s model=%s reasoning_effort=%s cwd=%s input_mode=%s timeout_seconds=%.1f prompt_chars=%d resume=%s continue=%s permission_mode=%s",
            output_format,
            effective.model,
            effective.reasoning_effort,
            prepared.cwd,
            effective.input_mode,
            effective.timeout_seconds,
            len(prompt),
            effective.session_id is not None,
            effective.continue_last_session,
            effective.permission_mode,
        )
        self.logger.debug(
            "Codex argv=%s",
            _redact_argv(prepared.argv, prompt_in_argv=prepared.input_text is None),
        )
        if self.log_prompts:
            self.logger.debug("Codex prompt=%s", prompt)

    def _log_finish(
        self,
        *,
        output_format: str,
        metadata: CommandMetadata,
        stdout: str,
        stderr: str,
        payload: CodexJsonPayload | None,
    ) -> None:
        self.logger.info(
            "Codex run finished output_format=%s returncode=0 elapsed_ms=%.0f stdout_chars=%d stderr_chars=%d",
            output_format,
            metadata.elapsed_ms,
            len(stdout),
            len(stderr),
        )
        if payload is not None:
            self.logger.debug(
                "Codex payload thread_id=%s usage=%s has_error=%s structured_output=%s",
                payload.thread_id,
                payload.usage,
                payload.error is not None,
                payload.has_structured_output,
            )

    def _log_protocol_error(self, error: CodexProtocolError) -> None:
        self.logger.warning(
            "Codex protocol error elapsed_ms=%.0f error=%s",
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
    ) -> CodexProtocolError:
        if not stdout.strip():
            return CodexProtocolError(
                f"Codex returned empty output in {protocol_name}",
                command=command,
                stdout=stdout,
                stderr=stderr,
            )

        return CodexProtocolError(
            f"Codex returned non-JSONL output in {protocol_name}: {stdout.strip()[:500]}",
            command=command,
            stdout=stdout,
            stderr=stderr,
        )

    def _write_schema_file(self, schema: StructuredSchema[Any]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="stan-ai-client-codex-schema-",
            delete=False,
        ) as schema_file:
            schema_file.write(schema.cli_json)
            schema_file.write("\n")
            return schema_file.name


def _redact_argv(argv: tuple[str, ...], *, prompt_in_argv: bool) -> tuple[str, ...]:
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
        if prompt_in_argv and index == len(argv) - 1:
            redacted.append("<prompt>")
            continue
        redacted.append(value)
    return tuple(redacted)
