from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema.validators import Draft202012Validator

from stan_ai_client import (
    AIClientTimeoutError,
    CodexClient,
    CodexCodeError,
    CodexExecutableNotFoundError,
    CodexNetworkUnavailableError,
    CodexProcessError,
    CodexProtocolError,
    CodexRateLimitError,
    CodexRunOptions,
    CodexSchemaValidationError,
    CodexStructuredOutputMissingError,
    CodexStructuredOutputValidationError,
    CodexTimeoutError,
    ExecutableNotFoundError,
    NetworkUnavailableError,
    RateLimitRetryPolicy,
    StructuredSchema,
    validate_codex_output_schema,
)
from stan_ai_client.codex import (
    UNSUPPORTED_CODEX_SCHEMA_KEYWORDS,
    _SCHEMA_MAPPING_KEYWORDS,
    _SCHEMA_VALUE_KEYWORDS,
)
from stan_ai_client.codex_parser import CODEX_ERROR_EVENT_TYPES

# Draft 2020-12 keywords that constrain a value directly instead of holding a
# subschema. Every other keyword the validator knows must be rejected or
# enumerated by the preflight, or a nested unsupported keyword can hide below it.
NON_SCHEMA_BEARING_SCHEMA_KEYWORDS = frozenset(
    {
        "$dynamicRef",
        "$ref",
        "const",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "required",
        "type",
    }
)


class RunRecorder:
    def __init__(
        self,
        completed: subprocess.CompletedProcess[str] | list[subprocess.CompletedProcess[str]],
    ) -> None:
        self.completed = completed if isinstance(completed, list) else [completed]
        self.calls: list[dict[str, Any]] = []
        self.schema_texts: list[str] = []
        self.schema_paths: list[str] = []

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: str | None,
        text: bool,
        capture_output: bool,
        timeout: float,
        input: str | None,
        env: Mapping[str, str] | None,
    ) -> subprocess.CompletedProcess[str]:
        call_index = len(self.calls)
        argv_tuple = tuple(argv)
        self.calls.append(
            {
                "argv": argv_tuple,
                "cwd": cwd,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
                "input": input,
                "env": env,
            }
        )
        if "--output-schema" in argv_tuple:
            schema_index = argv_tuple.index("--output-schema")
            schema_path = argv_tuple[schema_index + 1]
            self.schema_paths.append(schema_path)
            self.schema_texts.append(Path(schema_path).read_text(encoding="utf-8"))
        return self.completed[min(call_index, len(self.completed) - 1)]


def test_codex_run_text_uses_stdin_and_default_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    result = client.run_text("hello")

    argv = recorder.calls[0]["argv"]
    assert result.text == "done"
    assert argv[:2] == ("codex", "exec")
    assert argv[-1] == "-"
    assert recorder.calls[0]["input"] == "hello"
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert argv[argv.index("--model") + 1] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="medium"' in argv


def test_codex_client_init_defaults() -> None:
    client = CodexClient()

    assert client.executable == "codex"
    assert client.default_model == "gpt-5.6-sol"
    assert client.default_reasoning_effort == "medium"
    assert client.default_timeout_seconds == 120.0


def test_codex_run_text_accepts_max_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    client.run_text("hello", options=CodexRunOptions(reasoning_effort="max"))

    assert 'model_reasoning_effort="max"' in recorder.calls[0]["argv"]


def test_codex_run_text_accepts_minimal_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient(default_reasoning_effort="minimal")
    client.run_text("hello")

    assert 'model_reasoning_effort="minimal"' in recorder.calls[0]["argv"]


def test_codex_run_text_can_omit_bypass_and_use_argv_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    client.run_text(
        "tag this",
        options=CodexRunOptions(
            cwd="/tmp/article",
            input_mode="argv",
            permission_mode="default",
            skip_git_repo_check=True,
            add_dirs=("/tmp/more",),
            profile="ci",
            config_overrides=('web_search="disabled"',),
            extra_args=("--strict-config",),
        ),
    )

    argv = recorder.calls[0]["argv"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert recorder.calls[0]["cwd"] == "/tmp/article"
    assert recorder.calls[0]["input"] == ""
    assert argv[-2:] == ("--", "tag this")
    assert argv[argv.index("--cd") + 1] == "/tmp/article"
    assert argv[argv.index("--profile") + 1] == "ci"
    assert "--skip-git-repo-check" in argv
    assert "--add-dir" in argv
    assert "/tmp/more" in argv
    assert "--strict-config" in argv
    assert 'web_search="disabled"' in argv


def test_codex_run_text_uses_default_input_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient(default_options=CodexRunOptions(input_mode="argv"))
    client.run_text("tag this")

    assert recorder.calls[0]["input"] == ""
    assert recorder.calls[0]["argv"][-2:] == ("--", "tag this")


def test_codex_run_text_separates_argv_prompt_from_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    client.run_text("--not-an-option", options=CodexRunOptions(input_mode="argv"))

    argv = recorder.calls[0]["argv"]
    assert argv[-2:] == ("--", "--not-an-option")


def test_codex_run_text_normalizes_relative_cwd_for_cd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    client = CodexClient()
    client.run_text("hello", options=CodexRunOptions(cwd="repo"))

    argv = recorder.calls[0]["argv"]
    assert recorder.calls[0]["cwd"] == "repo"
    assert argv[argv.index("--cd") + 1] == str(repo_dir.resolve())


def test_codex_run_text_can_resume_session(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    client.run_text(
        "continue",
        options=CodexRunOptions(
            cwd="/tmp/repo",
            profile="ci",
            session_id="thread-1",
        ),
    )

    argv = recorder.calls[0]["argv"]
    resume_index = argv.index("resume")
    assert argv[:2] == ("codex", "exec")
    assert argv.index("--cd") < resume_index
    assert argv.index("--profile") < resume_index
    assert argv[resume_index + 1] == "thread-1"
    assert argv[-1] == "-"


def test_codex_run_text_can_continue_last_session(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    client.run_text("continue", options=CodexRunOptions(continue_last_session=True))

    argv = recorder.calls[0]["argv"]
    resume_index = argv.index("resume")
    assert argv[:2] == ("codex", "exec")
    assert argv[resume_index + 1] == "--last"
    assert argv[-1] == "-"


def test_codex_run_text_routes_extra_args_before_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    client.run_text(
        "continue",
        options=CodexRunOptions(
            continue_last_session=True,
            extra_args=("--color", "never"),
        ),
    )

    argv = recorder.calls[0]["argv"]
    resume_index = argv.index("resume")
    assert argv.index("--color") < resume_index
    assert argv.index("never") < resume_index
    assert argv[resume_index + 1] == "--last"
    assert argv[-1] == "-"


def test_codex_run_text_puts_resume_extra_args_after_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    client.run_text(
        "continue",
        options=CodexRunOptions(
            continue_last_session=True,
            resume_extra_args=("--all",),
        ),
    )

    argv = recorder.calls[0]["argv"]
    resume_index = argv.index("resume")
    assert argv.index("--all") > resume_index
    assert argv.index("--all") < argv.index("--last")
    assert argv[-1] == "-"


def test_codex_text_mode_ignores_network_prose_in_progress_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr = "\n".join(
        [
            "codex",
            "The documentation says network is unreachable.",
            "ERROR: permission denied",
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(CodexProcessError) as excinfo:
        CodexClient().run_text("quote the documentation")

    assert type(excinfo.value) is CodexProcessError
    assert not isinstance(excinfo.value, NetworkUnavailableError)
    assert excinfo.value.stderr == stderr


def test_codex_text_mode_does_not_retry_rate_limit_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr = "\n".join(
        [
            "codex",
            "The documentation says rate limit exceeded, retry after 60.",
        ]
    )
    recorder = RunRecorder(
        [
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr=stderr
            ),
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout="done\n", stderr=""
            ),
        ]
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.codex.time.sleep", sleeps.append)

    with pytest.raises(CodexProcessError) as excinfo:
        CodexClient().run_text(
            "quote the documentation",
            rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=120),
        )

    assert type(excinfo.value) is CodexProcessError
    assert not isinstance(excinfo.value, CodexRateLimitError)
    assert excinfo.value.stderr == stderr
    assert len(recorder.calls) == 1
    assert sleeps == []


def test_codex_text_mode_trusts_prefixed_stderr_errors(
    monkeypatch: pytest.MonkeyPatch,
    july_31_dns_diagnostic: str,
) -> None:
    causal_error = f"ERROR: {july_31_dns_diagnostic}"
    stderr = "\n".join(
        [
            "codex",
            "partial progress",
            causal_error,
            "ERROR: stream disconnected",
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(CodexNetworkUnavailableError) as excinfo:
        CodexClient().run_text("hello")

    assert str(excinfo.value) == causal_error
    assert excinfo.value.stderr == stderr


def test_codex_run_json_parses_jsonl_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
            '{"type":"turn.completed","usage":{"input_tokens":10}}',
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    result = client.run_json("hello")

    assert "--json" in recorder.calls[0]["argv"]
    assert result.payload.thread_id == "thread-1"
    assert result.payload.result == "done"
    assert result.payload.usage == {"input_tokens": 10}


def test_codex_run_json_raises_protocol_error_on_non_jsonl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="plain text", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexProtocolError):
        client.run_json("hello")


def test_codex_run_json_raises_protocol_error_on_non_event_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout='{"summary":"brief"}', stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexProtocolError):
        client.run_json("hello")


def test_codex_run_json_raises_process_error_from_turn_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"type":"turn.failed","message":"permission denied"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexProcessError) as excinfo:
        client.run_json("hello")

    assert "permission denied" in str(excinfo.value)
    assert excinfo.value.payload is not None
    assert excinfo.value.payload.error == {"type": "turn.failed", "message": "permission denied"}


def test_codex_dns_error_uses_common_and_provider_network_types(
    monkeypatch: pytest.MonkeyPatch,
    july_31_dns_diagnostic: str,
) -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "error", "message": july_31_dns_diagnostic}),
            json.dumps(
                {
                    "type": "turn.failed",
                    "error": {"message": "stream disconnected"},
                }
            ),
        ]
    )
    stderr = ""
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=stdout,
            stderr=stderr,
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(NetworkUnavailableError) as excinfo:
        CodexClient().run_json("hello")

    error = excinfo.value
    assert isinstance(error, CodexNetworkUnavailableError)
    assert isinstance(error, CodexProcessError)
    assert isinstance(error, CodexCodeError)
    assert error.command.argv[:2] == ("codex", "exec")
    assert error.returncode == 1
    assert error.stdout == stdout
    assert error.stderr == stderr
    assert error.payload is not None
    assert error.payload.events[0]["message"] == july_31_dns_diagnostic
    assert error.payload.error == {
        "type": "turn.failed",
        "error": {"message": "stream disconnected"},
    }


@pytest.mark.parametrize("event_type", sorted(CODEX_ERROR_EVENT_TYPES))
def test_codex_error_event_types_drive_payload_error_and_classification(
    monkeypatch: pytest.MonkeyPatch,
    july_31_dns_diagnostic: str,
    event_type: str,
) -> None:
    """The parser's error selection and network classification read one shared
    event-type set; narrowing either side leaves a parametrized case failing.
    The diagnostic sits in the earlier event so only the event scan can find it.
    """
    diagnostic_event = {"type": event_type, "message": july_31_dns_diagnostic}
    trailing_event = {"type": event_type, "message": "stream disconnected"}
    stdout = "\n".join([json.dumps(diagnostic_event), json.dumps(trailing_event)])
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout=stdout, stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(CodexNetworkUnavailableError) as excinfo:
        CodexClient().run_json("hello")

    assert excinfo.value.payload is not None
    assert excinfo.value.payload.error == trailing_event


def test_codex_dns_error_survives_a_truncated_jsonl_tail(
    monkeypatch: pytest.MonkeyPatch,
    july_31_dns_diagnostic: str,
) -> None:
    error_event = {"type": "error", "message": july_31_dns_diagnostic}
    stdout = "\n".join([json.dumps(error_event), '{"type":"turn.failed"'])
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout=stdout, stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(CodexNetworkUnavailableError) as excinfo:
        CodexClient().run_json("hello")

    assert excinfo.value.stdout == stdout
    assert excinfo.value.payload is not None
    assert excinfo.value.payload.events == (error_event,)
    assert excinfo.value.payload.error == error_event


def test_codex_stderr_rate_limit_precedes_network_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = json.dumps({"type": "error", "message": "Network is unreachable"})
    stderr = "429 rate limit exceeded, retry after 5"
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=stdout,
            stderr=stderr,
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(CodexRateLimitError) as excinfo:
        CodexClient().run_json("hello")

    assert excinfo.value.retry_after_seconds == 65
    assert not isinstance(excinfo.value, NetworkUnavailableError)


def test_codex_earlier_rate_limit_event_precedes_later_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "error",
                    "message": "Rate limit exceeded, retry after 5",
                }
            ),
            json.dumps(
                {
                    "type": "turn.failed",
                    "error": {"message": "Network is unreachable"},
                }
            ),
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout=stdout, stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(CodexRateLimitError) as excinfo:
        CodexClient().run_json("hello")

    assert excinfo.value.retry_after_seconds == 65
    assert not isinstance(excinfo.value, NetworkUnavailableError)


def test_codex_ignores_agent_message_network_prose_and_disconnect_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "diff --git a/doc.md b/doc.md\n+network is unreachable",
                    },
                }
            ),
            json.dumps({"type": "turn.failed", "message": "stream disconnected"}),
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout=stdout, stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    with pytest.raises(CodexProcessError) as excinfo:
        CodexClient().run_json("quote the diff")

    assert type(excinfo.value) is CodexProcessError
    assert not isinstance(excinfo.value, NetworkUnavailableError)


def test_codex_successful_network_prose_is_ordinary_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "The documentation says network is unreachable.",
            },
        }
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    result = CodexClient().run_json("quote the documentation")

    assert result.payload.result == "The documentation says network is unreachable."


def test_codex_run_json_preserves_recovered_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            '{"type":"error","message":"temporary failure"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
            '{"type":"turn.completed","usage":{"input_tokens":4}}',
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    result = client.run_json("hello")

    assert result.payload.result == "ok"
    assert result.payload.error == {"type": "error", "message": "temporary failure"}


def test_codex_rate_limit_policy_retries_json_after_parsed_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        [
            subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout='{"type":"error","message":"Rate limit exceeded, retry after 2"}',
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    sleeps: list[float] = []
    monkeypatch.setattr("stan_ai_client.codex.time.sleep", sleeps.append)

    result = client.run_json(
        "hello",
        rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=62),
    )

    assert result.payload.result == "ok"
    assert len(recorder.calls) == 2
    assert sleeps == [62.0]


def test_codex_rate_limit_error_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"type":"error","message":"Rate limit exceeded, retry after 5"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexRateLimitError) as excinfo:
        client.run_json("hello")

    assert excinfo.value.retry_after_seconds == 65
    assert not isinstance(excinfo.value, NetworkUnavailableError)


def test_codex_run_structured_passes_schema_file_and_validates_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"summary":"brief"}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
    )
    client = CodexClient()
    result = client.run_structured("summarize this", schema=schema)

    argv = recorder.calls[0]["argv"]
    assert result.structured_output == {"summary": "brief"}
    assert result.payload.structured_output == {"summary": "brief"}
    assert result.payload.has_structured_output is True
    assert "--output-schema" in argv
    assert recorder.schema_texts == [schema.cli_json + "\n"]
    assert recorder.schema_paths
    assert not Path(recorder.schema_paths[0]).exists()


def test_codex_structured_mode_ignores_network_prose_in_progress_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr = "\n".join(
        [
            "codex",
            "The documentation says network is unreachable.",
            "ERROR: permission denied",
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)
    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
    )

    with pytest.raises(CodexProcessError) as excinfo:
        CodexClient().run_structured("quote the documentation", schema=schema)

    assert type(excinfo.value) is CodexProcessError
    assert not isinstance(excinfo.value, NetworkUnavailableError)
    assert excinfo.value.stderr == stderr


def test_codex_run_structured_rejects_non_object_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="null", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexSchemaValidationError, match="root object"):
        client.run_structured("return null", schema=StructuredSchema.from_dict({"type": "null"}))

    assert recorder.calls == []


@pytest.mark.parametrize(
    "schema_dict, message",
    [
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "additionalProperties": False,
            },
            "required",
        ),
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            "additionalProperties",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "summary": {
                        "allOf": [
                            {"type": "string"},
                        ],
                    }
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            "allOf",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "summary": {
                        "oneOf": [
                            {"type": "string"},
                        ],
                    }
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            "oneOf",
        ),
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
                "not": {"required": ["summary"]},
            },
            "not",
        ),
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
                "dependentRequired": {"summary": ["detail"]},
            },
            "dependentRequired",
        ),
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
                "dependentSchemas": {"summary": {"required": ["summary"]}},
            },
            "dependentSchemas",
        ),
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
                "if": {"required": ["summary"]},
            },
            "if",
        ),
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
                "then": {"required": ["summary"]},
            },
            "then",
        ),
        (
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
                "else": {"required": ["summary"]},
            },
            "else",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "contentSchema": {"type": "string"},
                    }
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            "contentSchema",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "uniqueItems": True,
                                }
                            },
                            "required": ["evidence_ids"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["sections"],
                "additionalProperties": False,
            },
            "uniqueItems",
        ),
    ],
)
def test_codex_run_structured_rejects_unsupported_schema_subset(
    monkeypatch: pytest.MonkeyPatch,
    schema_dict: dict[str, object],
    message: str,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout='{"summary":"brief"}', stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexSchemaValidationError, match=message):
        client.run_structured("hello", schema=StructuredSchema.from_dict(schema_dict))

    assert recorder.calls == []


@pytest.mark.parametrize(
    ("array_schema", "expected_path"),
    [
        (
            {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "$.properties.evidence_ids.uniqueItems",
        ),
        (
            {
                "type": "array",
                "contains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
            "$.properties.evidence_ids.contains.uniqueItems",
        ),
        (
            {
                "type": "array",
                "prefixItems": [
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    }
                ],
            },
            "$.properties.evidence_ids.prefixItems[0].uniqueItems",
        ),
    ],
)
def test_validate_codex_output_schema_reports_nested_unique_items_path(
    array_schema: dict[str, object],
    expected_path: str,
) -> None:
    schema: StructuredSchema[dict[str, Any]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"evidence_ids": array_schema},
            "required": ["evidence_ids"],
            "additionalProperties": False,
        }
    )
    with pytest.raises(CodexSchemaValidationError) as excinfo:
        validate_codex_output_schema(schema)

    assert f"{expected_path} is not supported" in str(excinfo.value)


@pytest.mark.parametrize(
    ("schema_type", "keyword", "value"),
    [
        ("array", "contains", {"type": "string"}),
        ("string", "contentSchema", {"type": "string"}),
        ("array", "prefixItems", [{"type": "string"}]),
        ("object", "patternProperties", {"^x": {"type": "string"}}),
        ("object", "propertyNames", {"pattern": "^x"}),
        ("array", "unevaluatedItems", False),
        ("object", "unevaluatedProperties", False),
    ],
)
def test_validate_codex_output_schema_rejects_unsupported_schema_containers(
    schema_type: str,
    keyword: str,
    value: object,
) -> None:
    child_schema: dict[str, object] = {"type": schema_type, keyword: value}
    if schema_type == "array":
        child_schema["items"] = {"type": "string"}
    elif schema_type == "object":
        child_schema.update(
            properties={},
            required=[],
            additionalProperties=False,
        )

    schema: StructuredSchema[dict[str, Any]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"value": child_schema},
            "required": ["value"],
            "additionalProperties": False,
        }
    )

    with pytest.raises(CodexSchemaValidationError) as excinfo:
        validate_codex_output_schema(schema)

    assert f"$.properties.value.{keyword} is not supported" in str(excinfo.value)


@pytest.mark.parametrize(
    ("mapping_keyword", "mapping_key", "expected_path"),
    [
        ("properties", "a.b", '$.properties["a.b"].uniqueItems'),
        ("$defs", "items[0]", '$.$defs["items[0]"].uniqueItems'),
    ],
)
def test_validate_codex_output_schema_escapes_mapping_keys_in_paths(
    mapping_keyword: str,
    mapping_key: str,
    expected_path: str,
) -> None:
    child_schema: dict[str, object] = {
        "type": "array",
        "items": {"type": "string"},
        "uniqueItems": True,
    }
    schema_dict: dict[str, object] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
        mapping_keyword: {mapping_key: child_schema},
    }
    if mapping_keyword == "properties":
        schema_dict["required"] = [mapping_key]

    schema: StructuredSchema[dict[str, Any]] = StructuredSchema.from_dict(schema_dict)

    with pytest.raises(CodexSchemaValidationError) as excinfo:
        validate_codex_output_schema(schema)

    assert expected_path in str(excinfo.value)


def test_codex_schema_preflight_handles_every_draft_2020_12_keyword() -> None:
    handled = (
        set(UNSUPPORTED_CODEX_SCHEMA_KEYWORDS)
        | set(_SCHEMA_MAPPING_KEYWORDS)
        | set(_SCHEMA_VALUE_KEYWORDS)
        | NON_SCHEMA_BEARING_SCHEMA_KEYWORDS
    )

    assert set(Draft202012Validator.VALIDATORS) - handled == set()


def test_codex_structured_mode_surfaces_provider_error_from_stderr_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = "Generate the weekly Delta Growth report with evidence identifiers."
    provider_error = (
        "ERROR: unexpected status 400 Bad Request: "
        + "provider diagnostic context " * 30
        + '{"error":{"message":"Invalid schema for response_format \'output\': '
        "In context=('properties', 'evidence_ids'), 'uniqueItems' is not permitted.\","
        '"type":"invalid_request_error","param":"text.format.schema",'
        '"code":"invalid_json_schema"}}'
    )
    assert len(provider_error) > 500
    stderr = "\n".join(
        [
            ">_ You are using OpenAI Codex in ~/uzealot",
            "",
            "model: gpt-5.6-sol",
            "--------",
            "user",
            prompt,
            "",
            "thinking",
            *(f"progress line {index} while the model works" for index in range(30)),
            provider_error,
            "shutting down",
        ]
    )
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
    )
    with pytest.raises(CodexProcessError) as excinfo:
        CodexClient().run_structured(prompt, schema=schema)

    message = str(excinfo.value)
    assert type(excinfo.value) is CodexProcessError
    assert message == provider_error[-500:]
    assert "invalid_json_schema" in message
    assert "uniqueItems" in message
    assert prompt not in message
    assert excinfo.value.stderr == stderr
    assert excinfo.value.stdout == ""


@pytest.mark.parametrize(
    "options, expected_resume_arg",
    [
        (CodexRunOptions(session_id="thread-1"), "thread-1"),
        (CodexRunOptions(continue_last_session=True), "--last"),
    ],
)
def test_codex_run_structured_allows_resume_options(
    monkeypatch: pytest.MonkeyPatch,
    options: CodexRunOptions,
    expected_resume_arg: str,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout='{"summary":"brief"}', stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
    )
    result = client.run_structured("hello", schema=schema, options=options)

    argv = recorder.calls[0]["argv"]
    resume_index = argv.index("resume")
    assert result.structured_output == {"summary": "brief"}
    assert "--output-schema" in argv
    assert argv.index("--output-schema") < resume_index
    assert argv[resume_index + 1] == expected_resume_arg


def test_codex_run_structured_raises_when_output_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexStructuredOutputMissingError) as excinfo:
        client.run_structured(
            "hello",
            schema=StructuredSchema.from_dict(
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }
            ),
        )

    assert excinfo.value.payload.has_structured_output is False


def test_codex_run_structured_raises_when_output_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"summary":1}',
            stderr="",
        )
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexStructuredOutputValidationError) as excinfo:
        client.run_structured(
            "hello",
            schema=StructuredSchema.from_dict(
                {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                }
            ),
        )

    assert excinfo.value.structured_output == {"summary": 1}


def test_codex_missing_executable_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("codex not found")

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", raise_not_found)

    client = CodexClient(executable="codex")
    with pytest.raises(CodexExecutableNotFoundError) as excinfo:
        client.run_text("hello")

    assert isinstance(excinfo.value, ExecutableNotFoundError)


def test_codex_missing_cwd_is_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing_cwd(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", kwargs["cwd"])

    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", raise_missing_cwd)

    client = CodexClient()
    with pytest.raises(CodexProcessError) as excinfo:
        client.run_text("hello", options=CodexRunOptions(cwd="/tmp/missing-workspace"))

    assert "working directory not found" in str(excinfo.value)
    assert excinfo.value.command.cwd == "/tmp/missing-workspace"
    assert excinfo.value.returncode == 127


def test_codex_timeout_error_uses_provider_neutral_base() -> None:
    assert issubclass(CodexTimeoutError, AIClientTimeoutError)


def test_codex_logging_redacts_prompt_and_schema_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout='{"summary":"brief"}', stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.codex_logging")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
    )

    client = CodexClient(logger=logger)
    client.run_structured("super secret prompt", schema=schema)

    assert "Codex run starting" in caplog.text
    assert "--output-schema" in caplog.text
    assert "<redacted>" in caplog.text
    assert "super secret prompt" not in caplog.text
    assert recorder.schema_paths[0] not in caplog.text


def test_codex_logging_redacts_resume_session_id(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="done\n", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    logger = logging.getLogger("stan_ai_client.tests.codex_resume_logging")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    client = CodexClient(logger=logger)
    client.run_text(
        "super secret prompt",
        options=CodexRunOptions(input_mode="argv", session_id="thread-secret"),
    )

    assert "resume" in caplog.text
    assert "<redacted>" in caplog.text
    assert "thread-secret" not in caplog.text
    assert "super secret prompt" not in caplog.text
