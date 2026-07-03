# stan_ai_client

`stan_ai_client` is a thin Python wrapper around local AI coding CLIs.

It supports Claude Code through the local `claude` executable, Codex through
the local `codex exec` executable, and Grok through the local `grok` executable
(`grok -p`). It does not call Anthropic, OpenAI, or xAI APIs directly; the
relevant CLI must already be installed and authenticated on the machine.

The library is intentionally small and pragmatic:

- `run_text()` for plain-text output
- `run_json()` for Claude JSON envelopes or Codex JSONL events
- `run_structured()` for schema-validated structured output
- typed results
- structured exceptions
- local JSON Schema validation
- rate-limit parsing helpers
- opt-in rate-limit retry policy
- stdlib logging

## Why Use It

Use `stan_ai_client` when you want:

- a small Python API on top of Claude Code or Codex
- text mode and JSON mode without hand-rolling subprocess logic
- strongly guided structured output with local validation
- command metadata, typed JSON payloads, and normalized exceptions
- safe-by-default prompt logging behavior
- local automation that already depends on Claude Code or Codex being installed

Typical use cases:

- article summarization
- tagging or YAML generation
- one-shot repository or directory analysis
- local scripts that need session metadata, usage, cost, or duration

It is not a replacement for the Anthropic SDK, the OpenAI SDK, or the Codex SDK.
It intentionally stays at the local process-wrapper layer.

## Install

### From PyPI

```bash
pip install stan-ai-client
```

### From a local checkout

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### From GitHub

```bash
pip install "git+https://github.com/<your-user>/stan_ai_client.git"
```

## Releases

- the package version lives in `pyproject.toml`
- every non-bot push or merge to `main` bumps patch automatically
- tags use `vX.Y.Z`
- `main` releases build and publish to PyPI automatically
- release commits are created by GitHub Actions as `chore: release vX.Y.Z [skip ci]`

## Quickstart

### 1. Install Claude Code or Codex

Make sure the CLI you want is already available on your machine and authenticated:

```bash
claude --version
codex --version
```

### 2. Run the smoke test

```bash
python examples/smoke_test.py
python examples/codex_smoke_test.py
```

Those run text-mode and JSON-mode calls against the selected local CLI.

## Minimal Usage

### Claude text mode

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_text("Reply with the single word: ok")
print(result.text)
```

### Claude JSON mode

```python
from pathlib import Path

from stan_ai_client import ClaudeCodeClient, RunOptions

client = ClaudeCodeClient(
    default_model="claude-opus-4-8",
    default_effort="max",
    default_timeout_seconds=180,
)

result = client.run_json(
    "Summarize this article.",
    options=RunOptions(
        cwd=Path("."),
        allowed_tools=("Read", "Glob", "Grep", "Bash"),
    ),
)

print(result.payload.result)
print(result.payload.total_cost_usd)
print(result.payload.session_id)
```

### Claude structured mode

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

print(result.structured_output["summary"])
print(result.payload.session_id)
print(result.payload.total_cost_usd)
```

`run_structured()` validates the schema before Claude runs, requires `structured_output` in the response, and validates the returned object locally against the same schema.

### Codex text mode

```python
from stan_ai_client import CodexClient

client = CodexClient()
result = client.run_text("Reply with the single word: ok")
print(result.text)
```

`CodexClient` targets `codex exec`. By default it passes
`--dangerously-bypass-approvals-and-sandbox`, matching the current automation
preference for this package. Use `CodexRunOptions(permission_mode="default")`
to omit that flag and let Codex use its configured defaults.

### Codex JSONL mode

```python
from stan_ai_client import CodexClient

client = CodexClient()
result = client.run_json("Summarize this repository.")

print(result.payload.result)
print(result.payload.thread_id)
print(result.payload.usage)
```

Codex JSON mode uses `codex exec --json`, so the payload represents parsed JSONL
events instead of a Claude-style one-object envelope.

### Codex structured mode

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

Codex structured mode writes the validated schema to a temporary JSON file,
passes it with `--output-schema`, parses the final response as JSON, and
validates the returned object locally.

Codex additionally validates schemas against the OpenAI structured-output
subset before invoking the CLI. The root schema must be an object, every object
property must be required, and objects must set `additionalProperties: false`.
Unsupported schema keywords such as `allOf`, `oneOf`, `not`,
`dependentRequired`, `dependentSchemas`, `if`, `then`, and `else` are rejected
locally.
Structured Codex runs may also resume existing sessions with `session_id` or
`continue_last_session`.

### Grok text / JSON / structured

```python
from stan_ai_client import GrokClient, GrokRunOptions, StructuredSchema

client = GrokClient()  # defaults to model="grok-build"

result = client.run_text("Reply with the single word: ok")
print(result.text)

# JSON + stable named session
r = client.run_json("Summarize briefly.", options=GrokRunOptions(session_id="..."))
print(r.payload.text)
print(r.payload.session_id)

# Structured
schema = StructuredSchema.from_dict({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False})
res = client.run_structured("Return {ok: true}", schema=schema)
print(res.structured_output)
```

GrokClient drives `grok -p --output-format ...`. Prompt delivery is handled
transparently (direct arg or temp `--prompt-file` for long prompts). The JSON
payload is thinner than Claude's (no cost/usage fields); `duration_ms` is
provided client-side. Structured mode accepts either Grok's envelope
`structuredOutput` field or the raw validated JSON value returned by newer
Grok builds. Session support (`session_id`, `continue_last_session`) works well
for automation.

### Logging

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

By default, logging includes execution metadata, not full prompt text. Set `log_prompts=True` only if you explicitly want prompts written to logs.

### Error handling

If you automate Claude Code in batch jobs, pass a `RateLimitRetryPolicy` to let the client wait through parseable Claude rate limits up to your budget.

```python
from stan_ai_client import ClaudeCodeClient, RateLimitRetryPolicy

client = ClaudeCodeClient()

result = client.run_json(
    "Summarize this repository.",
    rate_limit_policy=RateLimitRetryPolicy(
        max_wait_seconds=5 * 60 * 60,
        label="repo summary",
    ),
)
```

For user-facing workflows, omit `rate_limit_policy` and catch `ClaudeRateLimitError` so you can return the reset time to the user.

```python
from stan_ai_client import ClaudeCodeClient, ClaudeRateLimitError

client = ClaudeCodeClient()

try:
    result = client.run_json("Summarize this repository.")
except ClaudeRateLimitError as exc:
    print(exc.reset_at or exc.retry_after_seconds)
```

## Public Surface

Top-level exports:

```python
from stan_ai_client import (
    __version__,
    ClaudeCodeClient,
    CodexClient,
    GrokClient,
    RunOptions,
    CodexRunOptions,
    GrokRunOptions,
    TextRunResult,
    JsonRunResult,
    StructuredRunResult,
    CodexJsonRunResult,
    CodexStructuredRunResult,
    ClaudeJsonPayload,
    CodexJsonPayload,
    GrokJsonPayload,
    GrokJsonRunResult,
    GrokPermissionMode,
    GrokStructuredRunResult,
    CommandMetadata,
    StructuredSchema,
    AIClientError,
    AIClientTimeoutError,
    ClaudeCodeError,
    CodexCodeError,
    ClaudeExecutableNotFoundError,
    CodexExecutableNotFoundError,
    ClaudeLimitError,
    CodexLimitError,
    ClaudeTimeoutError,
    CodexTimeoutError,
    ClaudeProcessError,
    CodexProcessError,
    ClaudeProtocolError,
    CodexProtocolError,
    ClaudeRateLimitError,
    CodexRateLimitError,
    StructuredSchemaValidationError,
    ClaudeSchemaValidationError,
    CodexSchemaValidationError,
    ClaudeStructuredOutputMissingError,
    CodexStructuredOutputMissingError,
    ClaudeStructuredOutputValidationError,
    CodexStructuredOutputValidationError,
    RateLimitRetryPolicy,
    RateLimitInfo,
    parse_rate_limit_info,
)
```

## Supported Features

- text mode via `run_text()`
- JSON mode via `run_json()`
- structured mode via `run_structured()`
- prompts sent over stdin by default
- optional argv prompt mode
- per-call working directory control
- model, effort/reasoning-effort, timeout, environment, and session controls
- support for Claude CLI flags via typed `RunOptions`
- support for Codex CLI flags via typed `CodexRunOptions`
- support for Grok CLI flags via typed `GrokRunOptions`
- raw stdout and stderr preserved on results and errors
- opt-in stdlib logging with safe default prompt handling
- typed JSON payload parsing with unknown fields preserved in `extras`
- typed Codex JSONL payload parsing with raw events preserved
- local input and output validation for structured mode
- rate-limit detection, reset-time parsing, and opt-in retry policy

## Examples

- [examples/smoke_test.py](./examples/smoke_test.py)
- [examples/codex_smoke_test.py](./examples/codex_smoke_test.py)
- [examples/grok_smoke_test.py](./examples/grok_smoke_test.py)
- [examples/summarize_article.py](./examples/summarize_article.py)
- [examples/tag_article.py](./examples/tag_article.py)
- [examples/logging_demo.py](./examples/logging_demo.py)
- [examples/rate_limit_retry.py](./examples/rate_limit_retry.py)

## Documentation

See [DOCS.md](./DOCS.md) for:

- full `RunOptions` reference
- logging behavior
- result types
- structured output usage
- exception model
- rate-limit handling
- session usage
- common patterns
- current limitations
- maintainer release flow

## Notes

- prompts default to stdin instead of argv
- Claude JSON mode always requests `--output-format json`
- Claude structured mode always requests `--output-format json` and `--json-schema`
- Claude text mode always requests `--output-format text`
- Codex JSON mode uses `codex exec --json`
- Codex structured mode uses `codex exec --output-schema <tempfile>`
- Codex defaults to `--dangerously-bypass-approvals-and-sandbox`
- Grok uses `grok -p --output-format plain|json` (prompt via arg or --prompt-file transparently)
- logging uses stdlib `logging`
- prompts are not written to logs unless `log_prompts=True`
- the library is sync-only in `0.1.x`
- streaming is intentionally out of scope right now

## Current Limitations

- no streaming support
- no async API
- no background scheduler or persistent job queue
- no standalone CLI wrapper command
- no first-class typed wrapper yet for every Claude Code flag
- no first-class typed wrapper yet for every Codex flag
- shared structured mode accepts dict-backed JSON Schema objects only
- Codex structured mode additionally enforces the OpenAI structured-output
  subset

For unsupported Claude Code flags, use `RunOptions(extra_args=...)`. For
unsupported Codex exec flags, use `CodexRunOptions(extra_args=...)`; for
unsupported Codex resume flags, use `CodexRunOptions(resume_extra_args=...)`.

## Development

```bash
pytest
mypy src tests
```
