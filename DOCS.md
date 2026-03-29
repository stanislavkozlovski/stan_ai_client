# stan_ai_client Docs

## Overview

`stan_ai_client` is a thin Python wrapper around the local `claude` executable.

It is designed for scriptable Claude Code usage where:

- Claude Code is already installed locally
- the caller wants a Python API instead of manual subprocess handling
- text mode, JSON mode, or structured mode is enough
- structured error handling matters
- stdlib logging is sufficient

It does not:

- call Anthropic APIs directly
- implement streaming
- implement async execution
- implement a retry loop or long-running job scheduler

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify Claude Code is available:

```bash
claude --version
```

## First Run

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_text("Reply with the single word: ok")
print(result.text)
```

## Main API

### `ClaudeCodeClient`

```python
class ClaudeCodeClient:
    def __init__(
        self,
        *,
        executable: str = "claude",
        default_model: str = "claude-opus-4-6",
        default_effort: Literal["low", "medium", "high", "max"] = "max",
        default_timeout_seconds: float = 120.0,
        default_options: RunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None: ...

    def run_text(self, prompt: str, *, options: RunOptions | None = None) -> TextRunResult: ...
    def run_json(self, prompt: str, *, options: RunOptions | None = None) -> JsonRunResult: ...
    def run_structured(
        self,
        prompt: str,
        *,
        schema: StructuredSchema[TStructured],
        options: RunOptions | None = None,
    ) -> StructuredRunResult[TStructured]: ...
```

### Constructor behavior

- `executable` selects the local binary to run. Default is `claude`.
- `default_model`, `default_effort`, and `default_timeout_seconds` provide process defaults.
- `default_options` lets you set a reusable baseline that per-call options can override.
- `logger` lets you provide a stdlib logger for client execution logs.
- `log_prompts` controls whether full prompt text is logged at debug level.

### `StructuredSchema`

`StructuredSchema` is the guided entry point for structured mode.

```python
from stan_ai_client import StructuredSchema

schema = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": ["summary"],
        "additionalProperties": False,
    }
)
```

Behavior:

- accepts dict-backed JSON Schema objects only
- validates the schema locally before any Claude subprocess starts
- stores the compact CLI JSON string once
- validates returned `structured_output` against the same schema

## Logging

Logging uses Python's stdlib `logging`.

If you do not configure logging in your application, the library still works normally; you just will not usually see info/debug output.

### Default logger behavior

- if you pass `logger=...`, that logger is used
- otherwise the client uses `logging.getLogger("stan_ai_client")`
- prompt text is not logged unless `log_prompts=True`

### What gets logged

At `INFO`:

- run start
- output mode
- model
- effort
- cwd
- input mode
- timeout
- prompt length
- whether the run is resuming, continuing, or forking
- run finish
- elapsed time
- stdout/stderr sizes

At `DEBUG`:

- redacted argv
- parsed JSON payload metadata when available
- structured mode enabled
- whether `structured_output` was present
- whether structured-output validation succeeded or failed
- full prompt text only if `log_prompts=True`

At `WARNING` or `ERROR`:

- missing executable
- timeout
- protocol errors
- process errors
- rate-limit details

### Logging example

```python
import logging

from stan_ai_client import ClaudeCodeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_app.claude")

client = ClaudeCodeClient(
    logger=logger,
    log_prompts=False,
)

client.run_text("Reply with the single word: ok")
```

### Logging safety

- prompts are redacted by default
- argv is logged in redacted form
- `--system-prompt`, `--append-system-prompt`, `--settings`, and resumed session IDs are redacted in argv logs
- use `log_prompts=True` only when you explicitly want prompt text in logs

## `RunOptions`

`RunOptions` controls one invocation.

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

### Field reference

`cwd`
- Working directory for the Claude process.
- Use this when Claude tools should read files from a specific directory.

`model`
- Overrides the default model for this call.

`effort`
- Sets Claude Code effort level.
- Supported values: `low`, `medium`, `high`, `max`.

`timeout_seconds`
- Subprocess timeout for the whole invocation.

`input_mode`
- `stdin` sends the prompt through stdin and still uses `claude -p`.
- `argv` appends the prompt directly to the command argv.
- Default is `stdin`.

`allowed_tools`
- Maps to `--allowed-tools`.
- Pass `()` to explicitly send an empty allowlist string.

`disallowed_tools`
- Maps to `--disallowed-tools`.

`tools`
- Maps to `--tools`.

`add_dirs`
- Repeats `--add-dir` for each provided path.

`permission_mode`
- Maps to `--permission-mode`.
- Current supported literal values are:
  - `acceptEdits`
  - `bypassPermissions`
  - `default`
  - `dontAsk`
  - `plan`
  - `auto`

`system_prompt`
- Maps to `--system-prompt`.

`append_system_prompt`
- Maps to `--append-system-prompt`.

`settings`
- Maps to `--settings`.

`session_id`
- Maps to `--resume <session_id>`.

`continue_last_session`
- Maps to `--continue`.
- Cannot be used together with `session_id`.

`fork_session`
- Maps to `--fork-session`.

`extra_args`
- Escape hatch for Claude CLI flags not yet modeled directly by `RunOptions`.
- Example:

```python
RunOptions(extra_args=("--debug", "--max-budget-usd", "1.50"))
```

`env`
- Additional environment variables merged on top of `os.environ`.

## Execution Modes

### Text mode

Use `run_text()` when you want Claude’s raw textual output.

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_text("Output YAML only.")
print(result.text)
```

Text mode still performs error normalization:

- non-zero exits raise exceptions
- if stdout happens to be JSON and reports `is_error=true`, that is treated as failure too

### JSON mode

Use `run_json()` when you want machine-readable Claude output.

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_json("Reply with a short answer.")
print(result.payload.result)
print(result.payload.total_cost_usd)
```

JSON mode requires valid JSON output:

- empty output raises `ClaudeProtocolError`
- non-JSON output raises `ClaudeProtocolError`
- JSON payloads with `is_error=true` raise `ClaudeProcessError` or `ClaudeRateLimitError`

### Structured mode

Use `run_structured()` when you want Claude’s JSON envelope plus a validated `structured_output`.

```python
from stan_ai_client import ClaudeCodeClient, StructuredSchema

client = ClaudeCodeClient()
schema = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "tags"],
        "additionalProperties": False,
    }
)

result = client.run_structured(
    "Summarize this article and return tags.",
    schema=schema,
)

print(result.structured_output)
print(result.payload.session_id)
print(result.payload.total_cost_usd)
```

Structured mode behavior:

- validates the input schema locally before Claude runs
- requests Claude JSON mode and passes `--json-schema`
- requires `structured_output` to be present in the JSON envelope
- does not fall back to `payload.result`
- validates the returned `structured_output` locally against the same schema
- preserves the full Claude envelope through `result.payload`

## Result Types

### `CommandMetadata`

```python
@dataclass(frozen=True)
class CommandMetadata:
    argv: tuple[str, ...]
    cwd: str | None
    elapsed_ms: float
```

Available on both text and JSON results.

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

Notes:

- `text` is just `stdout.strip()`
- raw `stdout` and `stderr` are preserved for debugging

### `JsonRunResult`

```python
@dataclass(frozen=True)
class JsonRunResult:
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: ClaudeJsonPayload
```

### `StructuredRunResult[TStructured]`

```python
@dataclass(frozen=True)
class StructuredRunResult(Generic[TStructured]):
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: ClaudeJsonPayload
    structured_output: TStructured
```

### `ClaudeJsonPayload`

Currently parsed fields:

- `type`
- `subtype`
- `is_error`
- `duration_ms`
- `duration_api_ms`
- `num_turns`
- `result`
- `stop_reason`
- `session_id`
- `total_cost_usd`
- `structured_output`
- `usage`
- `model_usage`
- `permission_denials`
- `uuid`
- `extras`

`extras` preserves unknown keys for forward compatibility.

## Exception Model

All library-specific exceptions inherit from `ClaudeCodeError`.

### `ClaudeExecutableNotFoundError`

Raised when the configured executable does not exist.

### `ClaudeTimeoutError`

Raised when the subprocess times out.

Carries:

- `command`
- `timeout_seconds`

### `ClaudeProcessError`

Raised when:

- the process exits non-zero
- or the JSON payload reports `is_error=true`

Carries:

- `command`
- `returncode`
- `stdout`
- `stderr`
- `payload`

### `ClaudeProtocolError`

Raised when JSON mode receives output that cannot be treated as valid JSON protocol output.

Carries:

- `command`
- `stdout`
- `stderr`

### `ClaudeSchemaValidationError`

Raised when `StructuredSchema.from_dict(...)` receives a non-dict input or an invalid JSON Schema.

### `ClaudeStructuredOutputMissingError`

Raised when structured mode succeeds at the process level but Claude does not return `structured_output`.

Carries:

- `command`
- `stdout`
- `stderr`
- `payload`

### `ClaudeStructuredOutputValidationError`

Raised when Claude returns `structured_output` but it does not validate against the provided schema.

Carries:

- `command`
- `stdout`
- `stderr`
- `payload`
- `structured_output`

### `ClaudeRateLimitError`

Specialized `ClaudeProcessError` for rate limits.

Carries everything from `ClaudeProcessError` plus:

- `rate_limit`

## Rate Limits

### `parse_rate_limit_info()`

```python
def parse_rate_limit_info(
    text: str,
    *,
    now: datetime | None = None,
    local_tz: tzinfo | None = None,
) -> RateLimitInfo: ...
```

Supported patterns:

- `retry after 3600`
- `retry-after: 3600`
- `resets in 2 hours 30 minutes`
- `resets in 4h 23m`
- `resets in 10 minutes`
- `resets at 15:00`
- `resets at 3:00 PM`
- `resets 3am`
- messages including timezone tags such as `(Europe/Madrid)`

### Example

```python
from stan_ai_client import ClaudeCodeClient, ClaudeRateLimitError

client = ClaudeCodeClient()

try:
    result = client.run_json("Summarize this article.")
except ClaudeRateLimitError as exc:
    print(exc.rate_limit.retry_after_seconds)
    print(exc.rate_limit.reset_at)
```

## Session Usage

### Resume a known session

```python
from stan_ai_client import ClaudeCodeClient, RunOptions

client = ClaudeCodeClient()
result = client.run_json(
    "Continue the task.",
    options=RunOptions(session_id="your-session-id"),
)
```

### Continue the most recent session

```python
from stan_ai_client import ClaudeCodeClient, RunOptions

client = ClaudeCodeClient()
result = client.run_json(
    "Continue the task.",
    options=RunOptions(continue_last_session=True),
)
```

### Fork a resumed or continued session

```python
from stan_ai_client import ClaudeCodeClient, RunOptions

client = ClaudeCodeClient()
result = client.run_json(
    "Continue but fork.",
    options=RunOptions(
        continue_last_session=True,
        fork_session=True,
    ),
)
```

Invalid combination:

```python
RunOptions(
    session_id="abc",
    continue_last_session=True,
)
```

That raises `ValueError`.

## Common Patterns

### Run inside a directory so Claude tools can read local files

```python
from pathlib import Path

from stan_ai_client import ClaudeCodeClient, RunOptions

client = ClaudeCodeClient()
result = client.run_json(
    "Summarize this repository.",
    options=RunOptions(
        cwd=Path("/path/to/repo"),
        allowed_tools=("Read", "Glob", "Grep", "Bash"),
    ),
)
```

### Use a shared baseline client with per-call overrides

```python
from stan_ai_client import ClaudeCodeClient, RunOptions

client = ClaudeCodeClient(
    default_model="claude-opus-4-6",
    default_effort="max",
    default_options=RunOptions(
        allowed_tools=("Read", "Glob"),
        timeout_seconds=180,
    ),
)

result = client.run_json(
    "Do the task.",
    options=RunOptions(
        cwd=".",
        extra_args=("--debug",),
    ),
)
```

### Plain-text generation

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_text("Output valid YAML only.")
print(result.text)
```

## Internal Command Behavior

Current command construction behavior:

- text mode always requests `--output-format text`
- JSON mode always requests `--output-format json`
- structured mode always requests `--output-format json`
- structured mode adds `--json-schema <compact-json>`
- both modes always use `claude -p`
- prompt goes over stdin by default
- `argv` mode appends the prompt directly to argv
- environment is copied from the parent process, then overridden by `RunOptions.env`

## Examples Included In The Repo

- [examples/smoke_test.py](./examples/smoke_test.py)
- [examples/summarize_article.py](./examples/summarize_article.py)
- [examples/tag_article.py](./examples/tag_article.py)
- [examples/logging_demo.py](./examples/logging_demo.py)

## Testing

Run tests:

```bash
pytest
```

Run type checks:

```bash
mypy src tests
```

## Current Limitations

- no streaming support
- no async API
- no built-in retry loop
- no standalone CLI wrapper command
- no first-class typed flags yet for every Claude Code option
- structured mode accepts dict-backed JSON Schema only

For unsupported Claude CLI flags, use `extra_args`.

## Suggested Usage Boundary

`stan_ai_client` should stay the thin process wrapper layer.

Keep these concerns outside the library:

- prompt templates specific to one application
- YAML or domain-object parsing
- business retries and backoff policy
- persistence
- app-specific logging policy beyond execution metadata
