# stan_ai_client

`stan_ai_client` is a thin Python wrapper around the local `claude` executable.

It does not call Anthropic APIs directly. Claude Code must already be installed and authenticated on the machine.

The library is intentionally small and pragmatic:

- `run_text()` for plain-text Claude output
- `run_json()` for `--output-format json`
- `run_structured()` for schema-validated structured output
- typed results
- structured exceptions
- local JSON Schema validation
- rate-limit parsing helpers
- stdlib logging

## Why Use It

Use `stan_ai_client` when you want:

- a small Python API on top of Claude Code
- text mode and JSON mode without hand-rolling subprocess logic
- strongly guided structured output with local validation
- command metadata, typed JSON payloads, and normalized exceptions
- safe-by-default prompt logging behavior
- local automation that already depends on Claude Code being installed

Typical use cases:

- article summarization
- tagging or YAML generation
- one-shot repository or directory analysis
- local scripts that need Claude session metadata, cost, or duration

It is not a replacement for the Anthropic API SDK, and it is not trying to abstract multiple providers.

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

### 1. Install Claude Code

Make sure `claude` is already available on your machine and authenticated:

```bash
claude --version
```

### 2. Run the smoke test

```bash
python examples/smoke_test.py
```

That runs one text-mode call and one JSON-mode call.

## Minimal Usage

### Text mode

```python
from stan_ai_client import ClaudeCodeClient

client = ClaudeCodeClient()
result = client.run_text("Reply with the single word: ok")
print(result.text)
```

### JSON mode

```python
from pathlib import Path

from stan_ai_client import ClaudeCodeClient, RunOptions

client = ClaudeCodeClient(
    default_model="claude-opus-4-6",
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

### Structured mode

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

If you automate Claude Code, treat `ClaudeLimitError` as a normal control-flow case: wait until `exc.reset_at` or sleep `exc.retry_after_seconds`, then retry.

```python
import time

from stan_ai_client import ClaudeCodeClient, ClaudeLimitError

client = ClaudeCodeClient()

for attempt in range(3):
    try:
        result = client.run_json("Summarize this repository.")
        break
    except ClaudeLimitError as exc:
        if attempt == 2 or exc.retry_after_seconds is None:
            raise
        time.sleep(exc.retry_after_seconds)
```

## Public Surface

Top-level exports:

```python
from stan_ai_client import (
    __version__,
    ClaudeCodeClient,
    RunOptions,
    TextRunResult,
    JsonRunResult,
    StructuredRunResult,
    ClaudeJsonPayload,
    CommandMetadata,
    StructuredSchema,
    ClaudeCodeError,
    ClaudeExecutableNotFoundError,
    ClaudeLimitError,
    ClaudeTimeoutError,
    ClaudeProcessError,
    ClaudeProtocolError,
    ClaudeRateLimitError,
    ClaudeSchemaValidationError,
    ClaudeStructuredOutputMissingError,
    ClaudeStructuredOutputValidationError,
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
- model, effort, timeout, environment, and session controls
- support for Claude CLI flags via typed `RunOptions`
- raw stdout and stderr preserved on results and errors
- opt-in stdlib logging with safe default prompt handling
- typed JSON payload parsing with unknown fields preserved in `extras`
- local input and output validation for structured mode
- rate-limit detection and reset-time parsing

## Examples

- [examples/smoke_test.py](./examples/smoke_test.py)
- [examples/summarize_article.py](./examples/summarize_article.py)
- [examples/tag_article.py](./examples/tag_article.py)
- [examples/logging_demo.py](./examples/logging_demo.py)

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
- JSON mode always requests `--output-format json`
- structured mode always requests `--output-format json` and `--json-schema`
- text mode always requests `--output-format text`
- logging uses stdlib `logging`
- prompts are not written to logs unless `log_prompts=True`
- the library is sync-only in `0.1.x`
- streaming is intentionally out of scope right now

## Current Limitations

- no streaming support
- no async API
- no built-in retry loop
- no standalone CLI wrapper command
- no first-class typed wrapper yet for every Claude Code flag
- structured mode accepts dict-backed JSON Schema objects only

For unsupported Claude Code flags, use `RunOptions(extra_args=...)`.

## Development

```bash
pytest
mypy src tests
```
