from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from stan_ai_client import (
    AIClientTimeoutError,
    CodexClient,
    CodexExecutableNotFoundError,
    CodexProcessError,
    CodexProtocolError,
    CodexRateLimitError,
    CodexRunOptions,
    CodexStructuredRunResult,
    CodexStructuredOutputMissingError,
    CodexStructuredOutputValidationError,
    CodexTimeoutError,
    ExecutableNotFoundError,
    RateLimitRetryPolicy,
    StructuredSchema,
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

    client = CodexClient(default_model="gpt-5.5", default_reasoning_effort="xhigh")
    result = client.run_text("hello")

    argv = recorder.calls[0]["argv"]
    assert result.text == "done"
    assert argv[:2] == ("codex", "exec")
    assert argv[-1] == "-"
    assert recorder.calls[0]["input"] == "hello"
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert argv[argv.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="xhigh"' in argv


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
    assert recorder.calls[0]["input"] is None
    assert argv[-1] == "tag this"
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

    assert recorder.calls[0]["input"] is None
    assert recorder.calls[0]["argv"][-1] == "tag this"


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


def test_codex_run_structured_accepts_null_when_schema_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="null", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    result: CodexStructuredRunResult[None] = client.run_structured(
        "return null",
        schema=StructuredSchema.from_dict({"type": "null"}),
    )

    assert result.structured_output is None
    assert result.payload.has_structured_output is True


def test_codex_run_structured_raises_when_output_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr("stan_ai_client.transport.subprocess.run", recorder)

    client = CodexClient()
    with pytest.raises(CodexStructuredOutputMissingError) as excinfo:
        client.run_structured("hello", schema=StructuredSchema.from_dict({"type": "object"}))

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
