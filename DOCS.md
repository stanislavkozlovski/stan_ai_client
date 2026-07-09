# stan_ai_client Docs

## Overview

`stan_ai_client` is a thin Python wrapper around local AI coding CLI
executables.

It supports:

- `ClaudeCodeClient` over the local `claude` executable
- `CodexClient` over the local `codex exec` command
- `GrokClient` over the local `grok -p` headless mode

It is designed for scriptable local CLI usage where:

- the selected CLI is already installed and authenticated
- the caller wants a Python API instead of manual subprocess handling
- text mode, JSON mode, or structured mode is enough
- structured error handling matters
- stdlib logging is sufficient

It does not:

- call Anthropic or OpenAI APIs directly
- manage CLI authentication files
- implement streaming
- implement async execution
- implement a long-running job scheduler

## Installation

```bash
pip install stan-ai-client
```

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the CLI you want is available:

```bash
claude --version
codex --version
grok --version
```

## Versioning And Releases

Versioning is intentionally simple:

- `pyproject.toml` is the source of truth
- each non-bot push or merge to `main` bumps patch by `0.0.1`
- releases are tagged as `vX.Y.Z`
- the release workflow builds distributions and publishes them to PyPI

The package also exposes its installed version at runtime:

```python
from stan_ai_client import __version__

print(__version__)
```

## Main API

### `ClaudeCodeClient`

```python
class ClaudeCodeClient:
    def __init__(
        self,
        *,
        executable: str = "claude",
        default_model: str = "claude-opus-4-8",
        default_effort: Literal["low", "medium", "high", "max"] = "max",
        default_timeout_seconds: float = 120.0,
        default_options: RunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None: ...
```

Methods:

```python
def run_text(
    ...,
    options: RunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> TextRunResult: ...
def run_json(
    ...,
    options: RunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> JsonRunResult: ...
def run_structured(
    ...,
    schema: StructuredSchema[T],
    options: RunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> StructuredRunResult[T]: ...
```

### `CodexClient`

```python
class CodexClient:
    def __init__(
        self,
        *,
        executable: str = "codex",
        default_model: str = "gpt-5.5",
        default_reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "medium",
        default_permission_mode: Literal["default", "bypassPermissions"] = "bypassPermissions",
        default_timeout_seconds: float = 120.0,
        default_options: CodexRunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None: ...
```

Methods:

```python
def run_text(
    ...,
    options: CodexRunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> TextRunResult: ...
def run_json(
    ...,
    options: CodexRunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> CodexJsonRunResult: ...
def run_structured(
    ...,
    schema: StructuredSchema[T],
    options: CodexRunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> CodexStructuredRunResult[T]: ...
```

`CodexClient` defaults to `bypassPermissions`, which adds
`--dangerously-bypass-approvals-and-sandbox` to `codex exec`. Use
`CodexRunOptions(permission_mode="default")` or
`CodexClient(default_permission_mode="default")` to omit that flag.

### `GrokClient`

```python
class GrokClient:
    def __init__(
        self,
        *,
        executable: str = "grok",
        default_model: str = "grok-4.5",
        default_effort: Literal["low", "medium", "high", "max"] | None = None,
        default_timeout_seconds: float = 120.0,
        default_options: GrokRunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None: ...
```

GrokClient drives `grok --no-auto-update -p`. It always passes `--model`
(defaulting to xAI's current Grok model, `grok-4.5`). Prompt delivery is
handled transparently inside the client.

## StructuredSchema

`StructuredSchema` is shared by both clients.

```python
from stan_ai_client import StructuredSchema

schema = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
)
```

Behavior:

- accepts dict-backed JSON Schema objects only
- validates the schema locally before any subprocess starts
- stores the compact CLI JSON string as `cli_json`
- validates returned structured output against the same schema

Claude structured mode passes `cli_json` inline with `--json-schema`. Codex
structured mode writes `cli_json` to a temporary file and passes the path with
`--output-schema`.

## Options

### `RunOptions`

`RunOptions` controls one Claude invocation.

```python
@dataclass(frozen=True)
class RunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    effort: Effort | None = None
    timeout_seconds: float | None = None
    input_mode: Literal["stdin", "argv"] = "stdin"
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
```

Important mappings:

- `cwd`: subprocess working directory
- `model`: `--model`
- `effort`: `--effort`
- `input_mode="stdin"`: prompt over stdin while still using `claude -p`
- `input_mode="argv"`: prompt appended directly to argv
- `session_id`: `--resume <session_id>`
- `continue_last_session`: `--continue`
- `fork_session`: `--fork-session`
- `extra_args`: escape hatch for unsupported Claude flags

### `CodexRunOptions`

`CodexRunOptions` controls one Codex invocation.

```python
@dataclass(frozen=True)
class CodexRunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    timeout_seconds: float | None = None
    input_mode: Literal["stdin", "argv"] | None = None
    permission_mode: Literal["default", "bypassPermissions"] | None = None
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
```

Important mappings:

- `cwd`: subprocess working directory and `--cd <dir>`
- `model`: `--model`
- `reasoning_effort`: `-c model_reasoning_effort="<value>"`
- `permission_mode="bypassPermissions"`: `--dangerously-bypass-approvals-and-sandbox`
- `permission_mode="default"`: omit the bypass flag
- `input_mode="stdin"`: prompt sent through stdin with `codex exec -`
- `input_mode="argv"`: prompt appended to argv after an option separator
- `session_id`: `codex exec resume <session_id>`
- `continue_last_session`: `codex exec resume --last`
- `skip_git_repo_check`: `--skip-git-repo-check`
- `ignore_user_config`: `--ignore-user-config`
- `ignore_rules`: `--ignore-rules`
- `profile`: `--profile`
- `config_overrides`: repeated `-c <override>`
- `extra_args`: escape hatch for unsupported `codex exec` flags
- `resume_extra_args`: escape hatch for unsupported `codex exec resume` flags

`session_id` and `continue_last_session` are mutually exclusive.

### `GrokRunOptions`

`GrokRunOptions` controls one Grok invocation.

```python
@dataclass(frozen=True)
class GrokRunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    effort: Literal["low", "medium", "high", "max"] | None = None
    timeout_seconds: float | None = None
    permission_mode: Literal[
        "acceptEdits", "bypassPermissions", "default", "dontAsk", "plan"
    ] | None = None
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
```

Important mappings:

- `cwd`: subprocess working directory
- `model`: `--model`
- `effort`: `--effort`
- `permission_mode`: `--permission-mode`
- `session_id`: `--session-id <session_id>`
- `continue_last_session`: `--continue`
- `allowed_tools`: repeated `--allow <rule>`
- `disallowed_tools`: repeated `--deny <rule>`
- `tools`: `--tools <comma-separated-tools>`
- `system_prompt`: `--system-prompt-override`
- `add_dirs`: accepted for API symmetry but not emitted; Grok currently has no
  documented add-directory flag distinct from `--cwd`
- `max_turns`: `--max-turns`
- `extra_args`: escape hatch for unsupported Grok flags

Prompt delivery is automatic: short prompts use `-p <prompt>`, while long
prompts use `--prompt-file <tempfile>`. Generated invocations include
`--no-auto-update` by default for headless automation.

## Execution Modes

### Claude Text

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_text("Output YAML only.")
print(result.text)
```

Text mode requests `--output-format text`.

### Claude JSON

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_json("Reply with a short answer.")
print(result.payload.result)
print(result.payload.session_id)
```

JSON mode requests `--output-format json` and parses a single Claude JSON
envelope into `ClaudeJsonPayload`.

### Claude Structured

```python
from stan_ai_client import ClaudeCodeClient, StructuredSchema

client = ClaudeCodeClient()
schema = StructuredSchema.from_dict({"type": "object"})
result = client.run_structured("Return an object.", schema=schema)
print(result.structured_output)
```

Structured mode requests JSON mode, passes `--json-schema <compact-json>`,
requires `payload.structured_output`, and validates it locally.

### Codex Text

```python
from stan_ai_client import CodexClient

client = CodexClient()
result = client.run_text("Output YAML only.")
print(result.text)
```

Text mode runs `codex exec`.

### Codex JSONL

```python
from stan_ai_client import CodexClient

client = CodexClient()
result = client.run_json("Summarize this repository.")
print(result.payload.result)
print(result.payload.thread_id)
print(result.payload.usage)
```

JSON mode runs `codex exec --json` and parses stdout as JSONL. Raw events are
preserved on `CodexJsonPayload.events`.

### Codex Structured

```python
from stan_ai_client import CodexClient, StructuredSchema

client = CodexClient()
schema = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }
)
result = client.run_structured("Summarize this repository.", schema=schema)
print(result.structured_output["summary"])
```

Structured mode writes the schema to a temporary file, runs
`codex exec --output-schema <file>`, parses stdout as JSON, validates the result,
and deletes the temporary schema file.

Codex schemas are additionally checked against the OpenAI structured-output
subset before the temporary file is created. The root schema must be an object,
object properties must all be listed in `required`, and objects must set
`additionalProperties: false`. Unsupported schema keywords such as `allOf`,
`oneOf`, `not`, `dependentRequired`, `dependentSchemas`, `if`, `then`, and
`else` are rejected locally.

Codex structured mode supports `session_id` and `continue_last_session`;
`--output-schema` is passed to `codex exec` before the `resume` subcommand.

## Result Types

### `TextRunResult`

```python
@dataclass(frozen=True)
class TextRunResult:
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    text: str
```

### Claude results

```python
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
```

### Codex results

```python
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
```

### Payloads

`ClaudeJsonPayload` exposes Claude envelope fields such as `result`,
`session_id`, `total_cost_usd`, `usage`, `model_usage`, `structured_output`, and
`extras`.

`CodexJsonPayload` exposes:

- `thread_id`
- `result`
- `usage`
- `events`
- `error`
- `structured_output`

## Exception Model

Provider-specific exceptions remain available:

- `ClaudeCodeError`
- `ClaudeExecutableNotFoundError`
- `ClaudeTimeoutError`
- `ClaudeProcessError`
- `ClaudeProtocolError`
- `ClaudeRateLimitError`
- `ClaudeStructuredOutputMissingError`
- `ClaudeStructuredOutputValidationError`
- `CodexCodeError`
- `CodexExecutableNotFoundError`
- `CodexTimeoutError`
- `CodexProcessError`
- `CodexProtocolError`
- `CodexRateLimitError`
- `CodexStructuredOutputMissingError`
- `CodexStructuredOutputValidationError`

Provider-neutral base classes are also exported:

- `AIClientError`
- `AIClientTimeoutError`
- `ExecutableNotFoundError`
- `ProcessError`
- `ProtocolError`
- `SchemaValidationError`
- `StructuredSchemaValidationError`
- `StructuredOutputMissingError`
- `StructuredOutputValidationError`
- `LimitError`
- `RateLimitError`

Catch provider-specific exceptions when you care which CLI failed. Catch
provider-neutral exceptions when the caller should handle Claude and Codex the
same way.

## Rate Limits

`RateLimitRetryPolicy` controls opt-in retry behavior for parseable rate-limit
responses from both clients.

```python
@dataclass(frozen=True)
class RateLimitRetryPolicy:
    max_wait_seconds: float | None
    label: str | None = None
```

Behavior:

- without a policy, rate limits raise immediately
- with a policy, the client sleeps and retries only when `retry_after_seconds`
  was parsed
- if the next wait exceeds the remaining `max_wait_seconds` budget, the client
  re-raises the same typed rate-limit error
- each retry sleep is logged at `WARNING`

## Logging

Both clients use Python's stdlib `logging`.

At `INFO`:

- run start
- output mode
- model
- effort or reasoning effort
- cwd
- input mode
- timeout
- prompt length
- session/resume state
- run finish
- elapsed time
- stdout/stderr sizes

At `DEBUG`:

- redacted argv
- parsed payload metadata when available
- structured mode state
- full prompt text only if `log_prompts=True`

At `WARNING` or `ERROR`:

- missing executable
- timeout
- protocol errors
- process errors
- rate-limit details
- rate-limit retry waits and wait-budget refusals

Prompt text is not logged unless `log_prompts=True`.

## Internal Command Behavior

Claude:

- text mode requests `claude -p --output-format text`
- JSON mode requests `claude -p --output-format json`
- structured mode requests JSON mode and adds `--json-schema <compact-json>`
- stdin mode sends the prompt through stdin
- argv mode appends the prompt directly to argv

Codex:

- text mode runs `codex exec`
- JSON mode adds `--json`
- structured mode adds `--output-schema <tempfile>`
- stdin mode sends the prompt through stdin with `codex exec -`
- argv mode appends the prompt after `--`
- argv mode sends empty stdin so inherited piped input is not added as context
- `bypassPermissions` adds `--dangerously-bypass-approvals-and-sandbox`

Both clients copy `os.environ`, merge `options.env`, preserve raw stdout/stderr
on results and errors, and run synchronously through `subprocess.run`.

## Examples Included In The Repo

- [examples/smoke_test.py](./examples/smoke_test.py)
- [examples/codex_smoke_test.py](./examples/codex_smoke_test.py)
- [examples/summarize_article.py](./examples/summarize_article.py)
- [examples/tag_article.py](./examples/tag_article.py)
- [examples/logging_demo.py](./examples/logging_demo.py)
- [examples/rate_limit_retry.py](./examples/rate_limit_retry.py)

## Testing

Run tests:

```bash
pytest
```

Run type checks:

```bash
mypy src tests
```

Run lint:

```bash
ruff check .
```

## Current Limitations

- no streaming support
- no async API
- no background scheduler or persistent job queue
- no standalone CLI wrapper command
- no direct Anthropic or OpenAI API calls
- no first-class typed wrapper yet for every Claude Code or Codex flag
- shared `StructuredSchema` accepts dict-backed JSON Schema objects only
- Codex structured mode additionally enforces the OpenAI structured-output
  subset

For unsupported Claude flags, use `RunOptions(extra_args=...)`. For unsupported
Codex exec flags, use `CodexRunOptions(extra_args=...)`; for unsupported Codex
resume flags, use `CodexRunOptions(resume_extra_args=...)`.

## Suggested Usage Boundary

`stan_ai_client` should stay the thin process wrapper layer.

Keep these concerns outside the library:

- prompt templates specific to one application
- YAML or domain-object parsing
- business scheduling and persistence around long-running work
- app-specific logging policy beyond execution metadata
