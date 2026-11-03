from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from typing import Any, Callable, Literal, Mapping, NoReturn, TypeVar

from jsonschema.exceptions import ValidationError

from ._network import (
    antigravity_trusted_error_texts,
    has_antigravity_network_unavailable_evidence,
)
from ._options import first_set, first_set_or
from ._retry import run_with_retry_policies
from .antigravity_parser import (
    has_auto_mode_unavailable_warning,
    has_current_structured_output,
    has_missing_conversation_warning,
    has_print_timeout_notice,
    headless_denial_notice,
    parse_antigravity_agent_error,
    recover_antigravity_partial_payload,
    stream_input_message,
    summarize_antigravity_error_text,
    try_parse_antigravity_payload,
)
from .exceptions import (
    AntigravityApprovalRequiredError,
    AntigravityCapacityError,
    AntigravityExecutableNotFoundError,
    AntigravityNetworkUnavailableError,
    AntigravityProcessError,
    AntigravityProtocolError,
    AntigravityRateLimitError,
    AntigravitySchemaValidationError,
    AntigravityServerError,
    AntigravitySessionNotFoundError,
    AntigravityStructuredOutputMissingError,
    AntigravityStructuredOutputValidationError,
    AntigravityTimeoutError,
)
from .provider_errors import _classify_antigravity_failure
from .provider_retry import ProviderRetryPolicy
from .rate_limits import is_antigravity_rate_limit_text, parse_rate_limit_info
from .schema import StructuredSchema
from .transport import PreparedCommand, execute_command
from .types import (
    AntigravityAgentError,
    AntigravityEffort,
    AntigravityJsonPayload,
    AntigravityJsonRunResult,
    AntigravityPermissionMode,
    AntigravityRunOptions,
    AntigravityStructuredRunResult,
    CommandMetadata,
    InputMode,
    RateLimitRetryPolicy,
    TextRunResult,
    UsageScope,
)

DEFAULT_LOGGER = logging.getLogger("stan_ai_client")
DEFAULT_INPUT_MODE: InputMode = "stdin"
DANGEROUSLY_SKIP_PERMISSIONS_FLAG = "--dangerously-skip-permissions"
JSON_SCHEMA_ARG_FLAG = "--json-schema"
CONVERSATION_ARG_FLAG = "--conversation"
PRINT_ARG_PREFIX = "--print="
REDACTED_ARG_FLAGS = {CONVERSATION_ARG_FLAG, "--settings"}
DISABLE_AUTO_UPDATE_ENV = "AGY_CLI_DISABLE_AUTO_UPDATE"
# Library permission vocabulary -> agy --mode values. "default" emits nothing
# and "bypassPermissions" emits --dangerously-skip-permissions.
AGENT_MODE_FLAG_VALUES: dict[str, str] = {
    "acceptEdits": "accept-edits",
    "plan": "plan",
    "auto": "auto",
}
_SUCCESS_STATUS = "SUCCESS"
_PRINT_TIMEOUT_REASON = "print_timeout"
AntigravityOutputProtocol = Literal["text", "stream-json"]
TRun = TypeVar("TRun")
TStructured = TypeVar("TStructured")


@dataclass(frozen=True)
class ResolvedAntigravityRunOptions:
    cwd: str | Path | None
    model: str | None
    effort: AntigravityEffort | None
    timeout_seconds: float
    input_mode: InputMode
    permission_mode: AntigravityPermissionMode
    session_id: str | None
    continue_last_session: bool
    add_dirs: tuple[str | Path, ...] | None
    agent: str | None
    project: str | None
    settings: str | None
    sandbox: bool | None
    disable_slash_commands: bool | None
    print_timeout_seconds: float | None
    log_file: str | Path | None
    extra_args: tuple[str, ...] | None
    env: Mapping[str, str] | None

    @property
    def resumes_conversation(self) -> bool:
        return self.session_id is not None or self.continue_last_session


@dataclass(frozen=True)
class PreparedAntigravityCommand(PreparedCommand):
    output_protocol: AntigravityOutputProtocol = "stream-json"


class AntigravityClient:
    """Run Google Antigravity through the local ``agy`` CLI in headless mode.

    JSON and structured calls read agy's ``stream-json`` events. In the default
    ``stdin`` input mode every call, including text, sends the prompt as one
    stream-json message on stdin, so prompts never appear in argv.
    """

    def __init__(
        self,
        *,
        executable: str = "agy",
        default_model: str | None = "gemini-4-argon",
        default_effort: AntigravityEffort | None = "high",
        default_permission_mode: AntigravityPermissionMode = "bypassPermissions",
        default_timeout_seconds: float = 120.0,
        default_options: AntigravityRunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None:
        self.executable = executable
        self.default_model = default_model
        self.default_effort = default_effort
        self.default_permission_mode = default_permission_mode
        self.default_timeout_seconds = default_timeout_seconds
        self.default_options = default_options or AntigravityRunOptions()
        self.logger = logger or DEFAULT_LOGGER
        self.log_prompts = log_prompts

    def run_text(
        self,
        prompt: str,
        *,
        options: AntigravityRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
        provider_retry_policy: ProviderRetryPolicy | None = None,
    ) -> TextRunResult:
        return self._run_with_retry_policies(
            lambda request: self._run_text_once(request, options=options),
            prompt=prompt,
            model=self._resolve_options(options).model,
            rate_limit_policy=rate_limit_policy,
            provider_retry_policy=provider_retry_policy,
        )

    def _run_text_once(
        self, prompt: str, *, options: AntigravityRunOptions | None = None
    ) -> TextRunResult:
        prepared, effective = self._prepare(prompt, options=options, json_output=False)
        self._log_start(prompt, output_format="text", prepared=prepared, effective=effective)
        payload: AntigravityJsonPayload | None = None
        if prepared.output_protocol == "stream-json":
            completed, metadata, payload = self._execute_stream(prepared, effective=effective)
            text = (payload.response or "").strip()
        else:
            completed, metadata = self._execute(prepared)
            self._raise_for_text_failure(completed, metadata, effective=effective)
            text = completed.stdout.strip()

        result = TextRunResult(
            command=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            text=text,
        )
        self._log_finish(
            completed,
            metadata,
            output_format="text",
            payload=payload,
        )
        return result

    def run_json(
        self,
        prompt: str,
        *,
        options: AntigravityRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
        provider_retry_policy: ProviderRetryPolicy | None = None,
    ) -> AntigravityJsonRunResult:
        return self._run_with_retry_policies(
            lambda request: self._run_json_once(request, options=options),
            prompt=prompt,
            model=self._resolve_options(options).model,
            rate_limit_policy=rate_limit_policy,
            provider_retry_policy=provider_retry_policy,
        )

    def _run_json_once(
        self, prompt: str, *, options: AntigravityRunOptions | None = None
    ) -> AntigravityJsonRunResult:
        prepared, effective = self._prepare(prompt, options=options, json_output=True)
        self._log_start(
            prompt, output_format="stream-json", prepared=prepared, effective=effective
        )
        completed, metadata, payload = self._execute_stream(prepared, effective=effective)
        result = AntigravityJsonRunResult(
            command=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            payload=payload,
        )
        self._log_finish(
            completed,
            metadata,
            output_format="stream-json",
            payload=payload,
        )
        return result

    def run_structured(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        options: AntigravityRunOptions | None = None,
        rate_limit_policy: RateLimitRetryPolicy | None = None,
        provider_retry_policy: ProviderRetryPolicy | None = None,
    ) -> AntigravityStructuredRunResult[TStructured]:
        return self._run_with_retry_policies(
            lambda request: self._run_structured_once(request, schema=schema, options=options),
            prompt=prompt,
            model=self._resolve_options(options).model,
            rate_limit_policy=rate_limit_policy,
            provider_retry_policy=provider_retry_policy,
        )

    def _run_structured_once(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        options: AntigravityRunOptions | None = None,
    ) -> AntigravityStructuredRunResult[TStructured]:
        validate_antigravity_output_schema(schema)
        prepared, effective = self._prepare(
            prompt, options=options, json_output=True, json_schema=schema
        )
        self._log_start(
            prompt, output_format="stream-json", prepared=prepared, effective=effective
        )
        self.logger.debug("Antigravity structured mode enabled schema_validated_locally=True")
        completed, metadata, payload = self._execute_stream(prepared, effective=effective)

        if not payload.has_structured_output:
            self._raise_structured_output_missing(
                completed,
                metadata,
                payload=payload,
                reason="Antigravity did not return structured_output in structured mode",
            )
        if effective.resumes_conversation and not has_current_structured_output(payload):
            # agy echoes the last structured_output of a resumed conversation even
            # when this turn's finish call failed schema validation.
            self._raise_structured_output_missing(
                completed,
                metadata,
                payload=payload,
                reason=(
                    "Antigravity returned structured_output from an earlier turn of the "
                    "resumed conversation; this turn produced none that satisfied the schema"
                ),
            )

        try:
            structured_output = schema.validate_response(payload.structured_output)
        except ValidationError as exc:
            self.logger.debug(
                "Antigravity structured_output validation failed error=%s", exc.message
            )
            validation_error = AntigravityStructuredOutputValidationError(
                "Antigravity returned structured_output that does not match the schema: "
                f"{exc.message}",
                command=metadata,
                stdout=completed.stdout,
                stderr=completed.stderr,
                payload=payload,
                structured_output=payload.structured_output,
            )
            self._log_protocol_error(validation_error)
            raise validation_error from exc

        self.logger.debug("Antigravity structured_output validation succeeded")
        result = AntigravityStructuredRunResult(
            command=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            payload=payload,
            structured_output=structured_output,
        )
        self._log_finish(
            completed,
            metadata,
            output_format="stream-json",
            payload=payload,
        )
        return result

    def _raise_structured_output_missing(
        self,
        completed: CompletedProcess[str],
        metadata: CommandMetadata,
        *,
        payload: AntigravityJsonPayload,
        reason: str,
    ) -> NoReturn:
        denied = _denied_action_names(payload)
        message = reason if not denied else f"{reason} (agy denied: {', '.join(denied)})"
        self.logger.debug("Antigravity structured_output missing")
        error = AntigravityStructuredOutputMissingError(
            message,
            command=metadata,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )
        self._log_protocol_error(error)
        raise error

    def _run_with_retry_policies(
        self,
        operation: Callable[[str], TRun],
        *,
        prompt: str,
        model: str | None,
        rate_limit_policy: RateLimitRetryPolicy | None,
        provider_retry_policy: ProviderRetryPolicy | None,
    ) -> TRun:
        return run_with_retry_policies(
            operation,
            prompt=prompt,
            model=model,
            provider_retry_policy=provider_retry_policy,
            rate_limit_policy=rate_limit_policy,
            logger=self.logger,
            provider="Antigravity",
            rate_limit_error_type=AntigravityRateLimitError,
        )

    def _prepare(
        self,
        prompt: str,
        *,
        options: AntigravityRunOptions | None,
        json_output: bool,
        json_schema: StructuredSchema[Any] | None = None,
    ) -> tuple[PreparedAntigravityCommand, ResolvedAntigravityRunOptions]:
        effective = self._resolve_options(options)
        if effective.session_id is not None and effective.continue_last_session:
            raise ValueError(
                "AntigravityRunOptions cannot set both session_id and continue_last_session"
            )

        argv = [self.executable]
        if effective.session_id is not None:
            argv.extend([CONVERSATION_ARG_FLAG, effective.session_id])
        elif effective.continue_last_session:
            argv.append("--continue")
        if effective.model is not None:
            argv.extend(["--model", effective.model])
        if effective.effort is not None:
            argv.extend(["--effort", effective.effort])
        if effective.permission_mode == "bypassPermissions":
            argv.append(DANGEROUSLY_SKIP_PERMISSIONS_FLAG)
        elif effective.permission_mode in AGENT_MODE_FLAG_VALUES:
            argv.extend(["--mode", AGENT_MODE_FLAG_VALUES[effective.permission_mode]])
        if effective.add_dirs is not None:
            for directory in effective.add_dirs:
                argv.extend(["--add-dir", str(directory)])
        if effective.agent is not None:
            argv.extend(["--agent", effective.agent])
        if effective.project is not None:
            argv.extend(["--project", effective.project])
        if effective.settings is not None:
            argv.extend(["--settings", effective.settings])
        _append_bool_flag(argv, "--sandbox", effective.sandbox)
        _append_bool_flag(argv, "--disable-slash-commands", effective.disable_slash_commands)
        if effective.print_timeout_seconds is not None:
            argv.extend(
                ["--print-timeout", _go_duration(effective.print_timeout_seconds)]
            )
        if effective.log_file is not None:
            argv.extend(["--log-file", str(effective.log_file)])
        if effective.extra_args is not None:
            argv.extend(effective.extra_args)

        # Protocol flags come last so extra_args cannot change the wire format.
        output_protocol: AntigravityOutputProtocol = (
            "stream-json" if json_output or effective.input_mode == "stdin" else "text"
        )
        if effective.input_mode == "stdin":
            argv.extend(["--input-format", "stream-json"])
        argv.extend(["--output-format", output_protocol])
        if json_schema is not None:
            argv.extend([JSON_SCHEMA_ARG_FLAG, json_schema.cli_json])

        if effective.input_mode == "stdin":
            input_text = stream_input_message(prompt)
        else:
            # The attached form keeps a prompt that starts with "-" from being
            # parsed as a flag. Empty stdin keeps inherited input out of the run.
            argv.append(PRINT_ARG_PREFIX + prompt)
            input_text = ""

        merged_env = os.environ.copy()
        if effective.env is not None:
            merged_env.update(effective.env)
        # Headless runs should not spawn agy's background self-updater; set the
        # variable to an empty string in env to keep agy's default behavior.
        merged_env.setdefault(DISABLE_AUTO_UPDATE_ENV, "1")

        cwd = None if effective.cwd is None else str(effective.cwd)
        prepared = PreparedAntigravityCommand(
            argv=tuple(argv),
            cwd=cwd,
            timeout_seconds=effective.timeout_seconds,
            input_text=input_text,
            env=merged_env,
            output_protocol=output_protocol,
        )
        return prepared, effective

    def _execute(
        self, prepared: PreparedAntigravityCommand
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
                self.logger.error("Antigravity working directory not found cwd=%s", prepared.cwd)
                raise AntigravityProcessError(
                    f"Antigravity working directory not found: {prepared.cwd}",
                    command=metadata,
                    returncode=127,
                    stdout="",
                    stderr="",
                    payload=None,
                ) from exc
            self.logger.error("Antigravity executable not found executable=%s", self.executable)
            raise AntigravityExecutableNotFoundError(self.executable) from exc
        except TimeoutExpired as exc:
            metadata = CommandMetadata(
                argv=prepared.argv,
                cwd=prepared.cwd,
                elapsed_ms=(time.monotonic() - started_at) * 1000,
            )
            self.logger.warning(
                "Antigravity run timed out timeout_seconds=%.1f cwd=%s argv=%s elapsed_ms=%.0f",
                prepared.timeout_seconds,
                prepared.cwd,
                _redact_argv(prepared.argv),
                metadata.elapsed_ms,
            )
            stdout = _timeout_text(exc.stdout)
            payload = (
                recover_antigravity_partial_payload(stdout)
                if prepared.output_protocol == "stream-json"
                else None
            )
            raise AntigravityTimeoutError(
                metadata,
                prepared.timeout_seconds,
                stdout=stdout,
                stderr=_timeout_text(exc.stderr),
                payload=None if payload is None else _stamp_duration(payload, metadata),
            ) from exc

        metadata = CommandMetadata(
            argv=prepared.argv,
            cwd=prepared.cwd,
            elapsed_ms=(time.monotonic() - started_at) * 1000,
        )
        return completed, metadata

    def _execute_stream(
        self,
        prepared: PreparedAntigravityCommand,
        *,
        effective: ResolvedAntigravityRunOptions,
    ) -> tuple[CompletedProcess[str], CommandMetadata, AntigravityJsonPayload]:
        completed, metadata = self._execute(prepared)
        stdout = completed.stdout
        stderr = completed.stderr
        payload = try_parse_antigravity_payload(stdout)
        if payload is not None:
            payload = replace(
                _stamp_duration(payload, metadata),
                usage_scope=_usage_scope(payload, effective),
                requested_model=effective.model,
            )
        self._warn_if_auto_mode_unavailable(stderr, effective=effective)

        if completed.returncode == 0:
            self._raise_if_conversation_missing(
                completed, metadata, payload=payload, effective=effective
            )
        self._raise_if_approval_required(
            completed, metadata, payload=payload, effective=effective
        )
        if completed.returncode != 0:
            raise self._build_process_error(
                completed, metadata, payload=payload, effective=effective
            )
        if payload is None:
            error = self._build_stream_protocol_error(metadata, stdout=stdout, stderr=stderr)
            self._log_protocol_error(error)
            raise error
        if payload.finished_reason == _PRINT_TIMEOUT_REASON:
            raise self._build_print_timeout_error(
                completed, metadata, payload=payload, effective=effective
            )
        if payload.status != _SUCCESS_STATUS or payload.finished_reason is not None:
            # agy omits finished_reason when a turn completes normally; any value
            # (interrupted, error, or a future reason) means the turn ended early.
            raise self._build_process_error(
                completed, metadata, payload=payload, effective=effective
            )
        return completed, metadata, payload

    def _raise_for_text_failure(
        self,
        completed: CompletedProcess[str],
        metadata: CommandMetadata,
        *,
        effective: ResolvedAntigravityRunOptions,
    ) -> None:
        """Plain-text output has no result envelope, so rely on agy's stderr notices."""
        self._warn_if_auto_mode_unavailable(completed.stderr, effective=effective)
        if completed.returncode != 0:
            raise self._build_process_error(
                completed, metadata, payload=None, effective=effective
            )
        self._raise_if_conversation_missing(
            completed, metadata, payload=None, effective=effective
        )
        if has_print_timeout_notice(completed.stderr):
            raise self._build_print_timeout_error(
                completed, metadata, payload=None, effective=effective
            )

    def _raise_if_conversation_missing(
        self,
        completed: CompletedProcess[str],
        command: CommandMetadata,
        *,
        payload: AntigravityJsonPayload | None,
        effective: ResolvedAntigravityRunOptions,
    ) -> None:
        if effective.session_id is None:
            return
        reported = None if payload is None else payload.conversation_id
        if not has_missing_conversation_warning(completed.stderr) and (
            reported is None or reported == effective.session_id
        ):
            return
        self.logger.warning(
            "Antigravity conversation not found returncode=%d elapsed_ms=%.0f "
            "new_conversation=%s",
            completed.returncode,
            command.elapsed_ms,
            reported is not None and reported != effective.session_id,
        )
        raise AntigravitySessionNotFoundError(
            "Antigravity could not find the requested conversation; agy ran the prompt "
            "in a new conversation instead",
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )

    def _raise_if_approval_required(
        self,
        completed: CompletedProcess[str],
        command: CommandMetadata,
        *,
        payload: AntigravityJsonPayload | None,
        effective: ResolvedAntigravityRunOptions,
    ) -> None:
        if (
            effective.permission_mode != "auto"
            or payload is None
            or not payload.denied_actions
        ):
            return

        # The model's own words when it produced any; otherwise agy's notice.
        approval_prompt = payload.response or headless_denial_notice(completed.stderr) or ""
        self.logger.warning(
            "Antigravity run requires approval returncode=%d elapsed_ms=%.0f "
            "approval_prompt_chars=%d denied_actions=%s",
            completed.returncode,
            command.elapsed_ms,
            len(approval_prompt),
            ",".join(_denied_action_names(payload)),
        )
        raise AntigravityApprovalRequiredError(
            "Antigravity auto mode stopped with an action awaiting approval",
            approval_prompt=approval_prompt,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )

    def _warn_if_auto_mode_unavailable(
        self, stderr: str, *, effective: ResolvedAntigravityRunOptions
    ) -> None:
        if effective.permission_mode == "auto" and has_auto_mode_unavailable_warning(stderr):
            self.logger.warning(
                "Antigravity auto mode is unavailable on this host (no terminal sandbox); "
                "agy continued in its default approval mode"
            )

    def _build_print_timeout_error(
        self,
        completed: CompletedProcess[str],
        command: CommandMetadata,
        *,
        payload: AntigravityJsonPayload | None,
        effective: ResolvedAntigravityRunOptions,
    ) -> AntigravityTimeoutError:
        timeout_seconds = (
            effective.print_timeout_seconds
            if effective.print_timeout_seconds is not None
            else effective.timeout_seconds
        )
        self.logger.warning(
            "Antigravity turn hit its print timeout print_timeout_seconds=%s elapsed_ms=%.0f",
            effective.print_timeout_seconds,
            command.elapsed_ms,
        )
        return AntigravityTimeoutError(
            command,
            timeout_seconds,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
            message=(
                "Antigravity turn stopped at its print timeout before completing; "
                "the partial output is on the exception"
            ),
        )

    def _build_process_error(
        self,
        completed: CompletedProcess[str],
        command: CommandMetadata,
        *,
        payload: AntigravityJsonPayload | None,
        effective: ResolvedAntigravityRunOptions,
    ) -> AntigravityProcessError:
        agent_error = parse_antigravity_agent_error(completed.stderr)
        trusted_texts = antigravity_trusted_error_texts(
            payload=payload, stderr=completed.stderr, agent_error=agent_error
        )
        error_text = summarize_antigravity_error_text(
            payload=payload,
            agent_error=agent_error,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

        failure = _classify_antigravity_failure(trusted_texts, agent_error=agent_error)
        if failure is not None:
            provider_error_cls = (
                AntigravityCapacityError
                if failure.kind == "capacity"
                else AntigravityServerError
            )
            self.logger.warning(
                "Antigravity run failed returncode=%d elapsed_ms=%.0f kind=%s signal=%s error=%s",
                completed.returncode,
                command.elapsed_ms,
                failure.kind,
                failure.signal,
                failure.message,
            )
            return provider_error_cls(
                failure.message,
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                payload=payload,
                failure=failure,
            )

        rate_limit_text = _rate_limit_text(trusted_texts, agent_error)
        if rate_limit_text is not None:
            rate_limit = parse_rate_limit_info(rate_limit_text)
            self.logger.warning(
                "Antigravity run failed returncode=%d elapsed_ms=%.0f error=%s "
                "retry_after_seconds=%s reset_at=%s",
                completed.returncode,
                command.elapsed_ms,
                rate_limit_text,
                rate_limit.retry_after_seconds,
                rate_limit.reset_at,
            )
            return AntigravityRateLimitError(
                rate_limit_text,
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                payload=payload,
                rate_limit=rate_limit,
            )

        error_cls: type[AntigravityProcessError]
        if has_antigravity_network_unavailable_evidence(trusted_texts):
            error_cls = AntigravityNetworkUnavailableError
            self.logger.warning(
                "Antigravity run failed returncode=%d elapsed_ms=%.0f "
                "network_unavailable=true error=%s",
                completed.returncode,
                command.elapsed_ms,
                error_text,
            )
        else:
            missing_conversation = (
                effective.session_id is not None
                and has_missing_conversation_warning(completed.stderr)
            )
            error_cls = (
                AntigravitySessionNotFoundError
                if missing_conversation
                else AntigravityProcessError
            )
            self.logger.warning(
                "Antigravity run failed returncode=%d elapsed_ms=%.0f error=%s",
                completed.returncode,
                command.elapsed_ms,
                error_text,
            )
        return error_cls(
            error_text,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
        )

    def _resolve_options(
        self, options: AntigravityRunOptions | None
    ) -> ResolvedAntigravityRunOptions:
        override = options or AntigravityRunOptions()
        default = self.default_options
        print_timeout_seconds = first_set(
            override.print_timeout_seconds, default.print_timeout_seconds
        )
        if print_timeout_seconds is not None and (
            not math.isfinite(print_timeout_seconds) or print_timeout_seconds < 0
        ):
            raise ValueError("print_timeout_seconds must be finite and >= 0")
        return ResolvedAntigravityRunOptions(
            cwd=first_set(override.cwd, default.cwd),
            model=first_set_or(override.model, default.model, default=self.default_model),
            effort=first_set_or(override.effort, default.effort, default=self.default_effort),
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
            add_dirs=first_set(override.add_dirs, default.add_dirs),
            agent=first_set(override.agent, default.agent),
            project=first_set(override.project, default.project),
            settings=first_set(override.settings, default.settings),
            sandbox=first_set(override.sandbox, default.sandbox),
            disable_slash_commands=first_set(
                override.disable_slash_commands, default.disable_slash_commands
            ),
            print_timeout_seconds=print_timeout_seconds,
            log_file=first_set(override.log_file, default.log_file),
            extra_args=first_set(override.extra_args, default.extra_args),
            env=first_set(override.env, default.env),
        )

    def _log_start(
        self,
        prompt: str,
        *,
        output_format: str,
        prepared: PreparedAntigravityCommand,
        effective: ResolvedAntigravityRunOptions,
    ) -> None:
        self.logger.info(
            "Antigravity run starting output_format=%s model=%s effort=%s cwd=%s "
            "input_mode=%s timeout_seconds=%.1f prompt_chars=%d resume=%s continue=%s "
            "permission_mode=%s",
            output_format,
            effective.model,
            effective.effort,
            prepared.cwd,
            effective.input_mode,
            effective.timeout_seconds,
            len(prompt),
            effective.session_id is not None,
            effective.continue_last_session,
            effective.permission_mode,
        )
        self.logger.debug("Antigravity argv=%s", _redact_argv(prepared.argv))
        if self.log_prompts:
            self.logger.debug("Antigravity prompt=%s", prompt)

    def _log_finish(
        self,
        completed: CompletedProcess[str],
        metadata: CommandMetadata,
        *,
        output_format: str,
        payload: AntigravityJsonPayload | None,
    ) -> None:
        self.logger.info(
            "Antigravity run finished output_format=%s returncode=0 elapsed_ms=%.0f "
            "stdout_chars=%d stderr_chars=%d",
            output_format,
            metadata.elapsed_ms,
            len(completed.stdout),
            len(completed.stderr),
        )
        if payload is not None:
            self.logger.debug(
                "Antigravity payload conversation_id=%s status=%s num_turns=%s "
                "usage_scope=%s invocation_usage=%s denied_actions=%d has_structured=%s",
                payload.conversation_id,
                payload.status,
                payload.num_turns,
                payload.usage_scope,
                payload.invocation_usage,
                len(payload.denied_actions),
                payload.has_structured_output,
            )

    def _log_protocol_error(self, error: AntigravityProtocolError) -> None:
        self.logger.warning(
            "Antigravity protocol error elapsed_ms=%.0f error=%s",
            error.command.elapsed_ms,
            str(error),
        )

    def _build_stream_protocol_error(
        self, command: CommandMetadata, *, stdout: str, stderr: str
    ) -> AntigravityProtocolError:
        if not stdout.strip():
            return AntigravityProtocolError(
                "Antigravity returned empty output in stream-json mode",
                command=command,
                stdout=stdout,
                stderr=stderr,
            )
        error = AntigravityProtocolError(
            "Antigravity stream-json output ended without a result event "
            f"({len(stdout)} stdout characters captured on the exception)",
            command=command,
            stdout=stdout,
            stderr=stderr,
        )
        error.payload = recover_antigravity_partial_payload(stdout)
        return error


def validate_antigravity_output_schema(schema: StructuredSchema[Any]) -> None:
    """Check the one structural rule agy enforces on ``--json-schema``.

    agy rejects any schema whose root ``type`` is not ``"object"`` at startup.
    Runs no subprocess. ``AntigravityClient.run_structured`` applies it first.
    """
    if schema.schema.get("type") != "object":
        raise AntigravitySchemaValidationError(
            'Antigravity structured output schema must have a root "type": "object"'
        )


def _rate_limit_text(
    texts: tuple[str, ...], agent_error: AntigravityAgentError | None
) -> str | None:
    text = next((text for text in texts if is_antigravity_rate_limit_text(text)), None)
    if text is not None:
        return text
    if agent_error is not None and (
        agent_error.status == "RESOURCE_EXHAUSTED"
        or (agent_error.code_kind == "http" and agent_error.error_code == 429)
    ):
        return agent_error.short_error or (texts[0] if texts else "RESOURCE_EXHAUSTED")
    return None


def _usage_scope(
    payload: AntigravityJsonPayload, effective: ResolvedAntigravityRunOptions
) -> UsageScope:
    """Scope of agy's result ``usage`` for this call.

    A resumed conversation reports conversation totals. A turn that ended
    early reports only its completed steps (zeros when none completed), which
    omits the in-flight model call, so it is not a measured invocation total.
    """
    if effective.resumes_conversation:
        return "cumulative"
    if payload.status == _SUCCESS_STATUS and payload.finished_reason is None:
        return "invocation"
    return "unknown"


def _denied_action_names(payload: AntigravityJsonPayload) -> list[str]:
    names: list[str] = []
    for action in payload.denied_actions:
        name = action.get("display_name") or action.get("action")
        names.append(name if isinstance(name, str) else "unknown")
    return names


def _stamp_duration(
    payload: AntigravityJsonPayload, metadata: CommandMetadata
) -> AntigravityJsonPayload:
    return replace(payload, duration_ms=int(metadata.elapsed_ms))


def _append_bool_flag(argv: list[str], flag: str, value: bool | None) -> None:
    if value is True:
        argv.append(flag)
    elif value is False:
        argv.append(f"{flag}=false")


def _go_duration(seconds: float) -> str:
    """Format seconds as a Go duration; agy rejects unitless values."""
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return f"{round(seconds * 1000)}ms"


def _timeout_text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _redact_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    redacted: list[str] = []
    replacement_for_next: str | None = None
    for value in argv:
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
        if value.startswith(PRINT_ARG_PREFIX):
            redacted.append(f"{PRINT_ARG_PREFIX}<prompt>")
            continue
        flag, separator, _ = value.partition("=")
        if separator and flag in REDACTED_ARG_FLAGS | {JSON_SCHEMA_ARG_FLAG}:
            redacted.append(f"{flag}=<redacted>")
            continue
        redacted.append(value)
    return tuple(redacted)
