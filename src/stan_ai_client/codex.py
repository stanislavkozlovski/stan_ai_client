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

from ._options import first_set, first_set_or
from ._retry import run_with_rate_limit_retry
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
    CodexSchemaValidationError,
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
DEFAULT_INPUT_MODE: InputMode = "stdin"
BYPASS_APPROVALS_AND_SANDBOX_FLAG = "--dangerously-bypass-approvals-and-sandbox"
OUTPUT_SCHEMA_ARG_FLAG = "--output-schema"
REDACTED_ARG_FLAGS = {
    "-c",
    "--config",
    OUTPUT_SCHEMA_ARG_FLAG,
}
UNSUPPORTED_CODEX_SCHEMA_KEYWORDS = (
    "allOf",
    "oneOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
)
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
    resume_extra_args: tuple[str, ...] | None
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

        if _payload_has_event_type(payload, "turn.failed"):
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
            payload = make_codex_structured_payload(
                None,
                structured_output_present=False,
            )
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
        return run_with_rate_limit_retry(
            operation,
            rate_limit_policy=rate_limit_policy,
            logger=self.logger,
            provider="Codex",
            rate_limit_error_type=CodexRateLimitError,
        )

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
        self._append_common_args(
            argv,
            effective=effective,
            json_output=json_output,
            output_schema_path=output_schema_path,
        )
        if effective.extra_args is not None:
            argv.extend(effective.extra_args)

        if is_resume:
            argv.append("resume")
            if effective.resume_extra_args is not None:
                argv.extend(effective.resume_extra_args)

        if effective.continue_last_session:
            argv.append("--last")
        elif effective.session_id is not None:
            argv.append(effective.session_id)

        input_text: str | None = None
        if effective.input_mode == "stdin":
            argv.append("-")
            input_text = prompt
        else:
            argv.append("--")
            argv.append(prompt)
            input_text = ""

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
            argv.extend(["--cd", str(Path(effective.cwd).resolve())])
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

    def _execute(
        self, prepared: PreparedCommand
    ) -> tuple[CompletedProcess[str], CommandMetadata]:
        started_at = time.monotonic()
        try:
            completed = execute_command(prepared)
        except FileNotFoundError as exc:
            if prepared.cwd is not None and exc.filename == prepared.cwd:
                metadata = CommandMetadata(
                    argv=prepared.argv,
                    cwd=prepared.cwd,
                    elapsed_ms=(time.monotonic() - started_at) * 1000,
                )
                self.logger.error("Codex working directory not found cwd=%s", prepared.cwd)
                raise CodexProcessError(
                    f"Codex working directory not found: {prepared.cwd}",
                    command=metadata,
                    returncode=127,
                    stdout="",
                    stderr="",
                    payload=None,
                ) from exc

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
                _redact_argv(prepared.argv, prompt_in_argv=_prompt_in_argv(prepared)),
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
        return ResolvedCodexRunOptions(
            cwd=first_set(override.cwd, default.cwd),
            model=first_set_or(override.model, default.model, default=self.default_model),
            reasoning_effort=first_set_or(
                override.reasoning_effort,
                default.reasoning_effort,
                default=self.default_reasoning_effort,
            ),
            timeout_seconds=float(
                first_set_or(
                    override.timeout_seconds,
                    default.timeout_seconds,
                    default=self.default_timeout_seconds,
                )
            ),
            input_mode=first_set_or(
                override.input_mode, default.input_mode, default=DEFAULT_INPUT_MODE
            ),
            permission_mode=first_set_or(
                override.permission_mode,
                default.permission_mode,
                default=self.default_permission_mode,
            ),
            session_id=first_set(override.session_id, default.session_id),
            continue_last_session=first_set_or(
                override.continue_last_session, default.continue_last_session, default=False
            ),
            skip_git_repo_check=first_set_or(
                override.skip_git_repo_check, default.skip_git_repo_check, default=False
            ),
            ignore_user_config=first_set_or(
                override.ignore_user_config, default.ignore_user_config, default=False
            ),
            ignore_rules=first_set_or(
                override.ignore_rules, default.ignore_rules, default=False
            ),
            add_dirs=first_set(override.add_dirs, default.add_dirs),
            profile=first_set(override.profile, default.profile),
            config_overrides=first_set(override.config_overrides, default.config_overrides),
            extra_args=first_set(override.extra_args, default.extra_args),
            resume_extra_args=first_set(
                override.resume_extra_args, default.resume_extra_args
            ),
            env=first_set(override.env, default.env),
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
            _redact_argv(prepared.argv, prompt_in_argv=effective.input_mode == "argv"),
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
        _validate_codex_output_schema(schema)
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
    resume_session_arg_index = _resume_session_arg_index(argv)
    for index, value in enumerate(argv):
        if replacement_for_next is not None:
            redacted.append(replacement_for_next)
            replacement_for_next = None
            continue
        if value in REDACTED_ARG_FLAGS:
            redacted.append(value)
            replacement_for_next = "<redacted>"
            continue
        if index == resume_session_arg_index:
            redacted.append("<redacted>")
            continue
        if prompt_in_argv and index == len(argv) - 1:
            redacted.append("<prompt>")
            continue
        redacted.append(value)
    return tuple(redacted)


def _payload_has_event_type(payload: CodexJsonPayload, event_type: str) -> bool:
    return any(event.get("type") == event_type for event in payload.events)


def _prompt_in_argv(prepared: PreparedCommand) -> bool:
    return bool(prepared.argv) and prepared.argv[-1] != "-"


def _resume_session_arg_index(argv: tuple[str, ...]) -> int | None:
    if "resume" not in argv or "--last" in argv or len(argv) < 2:
        return None

    session_index = len(argv) - 2
    if argv[session_index] == "--":
        session_index -= 1
    if argv[session_index] == "-":
        return None
    return session_index


def _validate_codex_output_schema(schema: StructuredSchema[Any]) -> None:
    errors = list(
        _iter_codex_output_schema_errors(
            schema.schema,
            path="$",
            require_root_object=True,
        )
    )
    if errors:
        detail = "; ".join(errors)
        raise CodexSchemaValidationError(
            f"Codex structured output schema is not supported: {detail}"
        )


def _iter_codex_output_schema_errors(
    node: object,
    *,
    path: str,
    require_root_object: bool = False,
) -> list[str]:
    if not isinstance(node, dict):
        return []

    errors: list[str] = []
    schema_type = node.get("type")
    if require_root_object and schema_type != "object":
        errors.append(f"{path} must be a root object schema")

    is_object_schema = (
        schema_type == "object"
        or (isinstance(schema_type, list) and "object" in schema_type)
        or "properties" in node
    )
    if is_object_schema:
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{path}.properties must be an object")
            properties = {}

        required = node.get("required")
        if not isinstance(required, list) or not all(
            isinstance(name, str) for name in required
        ):
            errors.append(f"{path}.required must list every property")
            required_names: set[str] = set()
        else:
            required_names = set(required)

        property_names = set(properties)
        missing = sorted(property_names - required_names)
        extra = sorted(required_names - property_names)
        if missing:
            errors.append(
                f"{path}.required must include every property: {', '.join(missing)}"
            )
        if extra:
            errors.append(
                f"{path}.required contains undefined properties: {', '.join(extra)}"
            )
        if node.get("additionalProperties") is not False:
            errors.append(f"{path}.additionalProperties must be false")

    errors.extend(_iter_schema_mapping_errors(node.get("properties"), f"{path}.properties"))
    errors.extend(_iter_schema_mapping_errors(node.get("$defs"), f"{path}.$defs"))
    errors.extend(_iter_schema_mapping_errors(node.get("definitions"), f"{path}.definitions"))

    items = node.get("items")
    if isinstance(items, dict):
        errors.extend(_iter_codex_output_schema_errors(items, path=f"{path}.items"))
    elif isinstance(items, list):
        for index, item in enumerate(items):
            errors.extend(
                _iter_codex_output_schema_errors(item, path=f"{path}.items[{index}]")
            )

    for keyword in UNSUPPORTED_CODEX_SCHEMA_KEYWORDS:
        if keyword in node:
            errors.append(f"{path}.{keyword} is not supported")

    value = node.get("anyOf")
    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(
                _iter_codex_output_schema_errors(
                    item,
                    path=f"{path}.anyOf[{index}]",
                )
            )

    return errors


def _iter_schema_mapping_errors(value: object, path: str) -> list[str]:
    if not isinstance(value, dict):
        return []

    errors: list[str] = []
    for key, child in value.items():
        errors.extend(_iter_codex_output_schema_errors(child, path=f"{path}.{key}"))
    return errors
