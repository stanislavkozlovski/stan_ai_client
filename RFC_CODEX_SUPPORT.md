# RFC: Codex Backend Support

## Status

Implemented.

## Context

`stan_ai_client` started as a thin, synchronous Python wrapper around the local
Claude Code CLI. The original Claude surface exposed:

- `ClaudeCodeClient.run_text()`
- `ClaudeCodeClient.run_json()`
- `ClaudeCodeClient.run_structured()`

The Claude implementation shells out to `claude -p`, captures stdout and stderr,
parses Claude's JSON envelope when requested, and normalizes failures into
Claude-specific exceptions.

Adding Codex support changes the original Claude-only product boundary. The goal
is not a vague all-provider abstraction; it is a pragmatic two-backend client
that can run either:

- Claude Code through the existing `claude` subprocess contract.
- Codex through Codex's non-interactive/programmatic surface.

Current OpenAI Codex docs describe `codex exec` as the stable non-interactive
CLI entry point. By default it streams progress to stderr and writes the final
agent message to stdout. It also supports JSONL event output through `--json`,
structured final responses through `--output-schema`, session resume through
`codex exec resume`, and a Python SDK through `openai-codex`. The closest fit
for this repository's existing shape is `codex exec`, not direct OpenAI API
calls.

Sources reviewed:

- Local code: `src/stan_ai_client/client.py`, `types.py`, `parser.py`,
  `schema.py`, `exceptions.py`, `transport.py`, and `tests/test_client.py`.
- Project docs: `README.md`, `DOCS.md`, and `ARCHITECTURE.md`.
- Official Codex docs: [non-interactive mode](https://developers.openai.com/codex/noninteractive),
  [CLI reference](https://developers.openai.com/codex/cli/reference),
  [Codex SDK](https://developers.openai.com/codex/sdk),
  [authentication](https://developers.openai.com/codex/auth), and
  [models](https://developers.openai.com/codex/models).

## Goals

- Add first-class Codex support without breaking existing Claude users.
- Keep the library synchronous and subprocess-oriented for the first Codex
  release.
- Preserve the existing high-level modes: text, JSON, and structured output.
- Make backend choice explicit at construction time, not hidden behind model
  string heuristics.
- Keep provider-specific options typed enough to avoid invalid flag mixtures.
- Share generic transport, schema validation, logging redaction, and retry
  mechanics where they are actually common.
- Keep the package/distribution name `stan-ai-client`.

## Non-goals

- Do not replace Claude support.
- Do not create a universal LLM SDK.
- Do not call the OpenAI Responses API directly in the first implementation.
- Do not add streaming or async behavior in the same change.
- Do not silently translate every Claude option into Codex. Some options are
  backend-specific and should remain so.

## Current Architecture Review

The original code was compact, but the Claude coupling was broad:

- `src/stan_ai_client/client.py` originally owned public methods, option
  resolution, CLI argv construction, execution, logging, retry, and error
  normalization.
- `src/stan_ai_client/types.py` exposed `RunOptions`, result types, and
  `ClaudeJsonPayload`.
- `src/stan_ai_client/parser.py` assumed Claude's JSON envelope shape.
- `src/stan_ai_client/exceptions.py` exposed only `Claude*` exceptions.
- `src/stan_ai_client/schema.py` validated JSON Schema locally, but raised
  `ClaudeSchemaValidationError`.
- `tests/test_client.py` asserted exact Claude argv behavior and Claude-specific
  logging text.

This means Codex support should not be added by conditionals inside
`ClaudeCodeClient._prepare()`. That would make `RunOptions` ambiguous and turn
Claude-specific result and exception names into accidental cross-provider API.

## Proposed Public API

Introduce a parallel Codex client and provider-neutral aliases while keeping the
existing Claude API stable.

```python
from stan_ai_client import CodexClient, CodexRunOptions, StructuredSchema

client = CodexClient(default_model="gpt-5.5")
result = client.run_text("Summarize this repository.")
print(result.text)

schema = StructuredSchema.from_dict({...})
structured = client.run_structured(
    "Extract project metadata.",
    schema=schema,
)
print(structured.structured_output)
```

Keep existing imports working:

```python
from stan_ai_client import ClaudeCodeClient, RunOptions
```

The implementation exposes explicit client classes only. A provider factory can
be added later if real callers need runtime backend selection from configuration.

## Proposed Internal Design

Split the current monolithic client into small backend-specific adapters around
a shared subprocess runner.

```text
src/stan_ai_client/
  client.py              # keep ClaudeCodeClient compatibility import/path
  claude.py              # ClaudeCodeClient implementation
  codex.py               # CodexClient implementation
  transport.py           # existing PreparedCommand + execute_command
  schema.py              # provider-neutral StructuredSchema
  common_types.py        # CommandMetadata, TextRunResult, retry policy
  claude_types.py        # ClaudeJsonPayload, ClaudeRunOptions if renamed
  codex_types.py         # CodexRunOptions, CodexJsonEvent, Codex payloads
  exceptions.py          # base + provider-specific compatibility classes
  parsers/
    claude.py
    codex.py
```

Do this in phases to reduce churn:

1. Add shared provider-neutral base types while re-exporting the existing names.
2. Move Claude implementation into `claude.py` and keep `client.py` as a
   compatibility shim.
3. Add `codex.py` with independent option resolution and command construction.
4. Add a small provider factory only if needed.

### Shared internal boundaries

Each backend keeps its own option dataclass and CLI-argv construction, but two
mechanics that must not drift between backends are centralized:

- `_options.py` (`first_set` / `first_set_or`) is the single definition of run
  option resolution. Every field in `_resolve_options` resolves as
  per-call override, then client `default_options`, then (when required) a
  client-level default. Because there is one rule, the only thing that decides
  whether `default_options` is honored for a field is whether that field is
  declared optional — the source of an earlier `input_mode` regression.
- `_retry.py` (`run_with_rate_limit_retry`) is the single implementation of the
  sleep-budget rate-limit retry loop. Backends supply only their `provider` log
  prefix and concrete `RateLimitError` subclass, so the wait-budget accounting
  cannot diverge between Claude and Codex.

A new backend should route option resolution and retry through these helpers
rather than re-deriving them.

## Codex CLI Mapping

`CodexClient` should target `codex exec`.

| Existing concept | Claude CLI | Codex CLI |
| --- | --- | --- |
| executable | `claude` | `codex` |
| text mode | `claude -p --output-format text` | `codex exec <prompt>` |
| JSON mode | `claude -p --output-format json` | `codex exec --json <prompt>` parsed from JSONL events |
| structured mode | `--json-schema <json>` | `--output-schema <schema-file>` |
| cwd | subprocess cwd | `--cd <dir>` and subprocess cwd |
| model | `--model <model>` | `--model <model>` |
| effort | `--effort <level>` | `-c model_reasoning_effort=<level>` if supported |
| resume session | `--resume <id>` | `codex exec resume <id>` |
| continue latest | `--continue` | `codex exec resume --last` |
| permission mode | `--permission-mode <mode>` | default to `--dangerously-bypass-approvals-and-sandbox` |
| extra flags | `extra_args` | `extra_args` |

The first Codex release should not expose fine-grained Codex sandbox options or
the `--ephemeral` persistence flag. Its effective permission default should be
`bypassPermissions`, implemented by adding
`--dangerously-bypass-approvals-and-sandbox` to `codex exec`. Callers can opt
out per run by setting `permission_mode="default"`, which omits that flag and
uses Codex's configured defaults.

Codex structured mode needs special handling because `--output-schema` takes a
file path, not an inline JSON string. The implementation should write the
already-validated `StructuredSchema.schema` to a temporary JSON file, pass that
path to `codex exec --output-schema`, then remove the file after execution.

For JSON mode, `--json` emits JSONL events rather than one JSON object. The
parser should collect:

- `thread.started.thread_id` as the session/thread id.
- the last completed `agent_message.text` as the result text.
- `turn.completed.usage` as usage metadata.
- `turn.failed` or `error` as failure metadata.
- all unknown events in an `events` list or `extras` field for forward
  compatibility.

## Codex Types

Add a Codex-specific options dataclass instead of overloading `RunOptions`.

```python
@dataclass(frozen=True)
class CodexRunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    timeout_seconds: float | None = None
    input_mode: InputMode | None = None
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
    env: Mapping[str, str] | None = None
```

`CodexClient` should also accept
`default_permission_mode: Literal["default", "bypassPermissions"] =
"bypassPermissions"` so the default behavior is explicit at construction time.
The supported `reasoning_effort` values are typed from the installed Codex model
catalog: `minimal`, `low`, `medium`, `high`, and `xhigh`. The option is
first-class, not only available through `config_overrides`.

Keep `StructuredSchema.cli_json` as-is for now. It already means "compact JSON
string suitable for a CLI argument" and renaming it would create churn without a
clear implementation benefit.

Add a Codex payload shape that is honest about JSONL:

```python
@dataclass(frozen=True)
class CodexJsonPayload:
    thread_id: str | None
    result: str | None
    usage: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    error: dict[str, Any] | None
    structured_output: Any | None
```

Do not genericize the existing `JsonRunResult` in the first Codex release. That
keeps compatibility risk lower for current Claude users. Add separate Codex
result classes first:

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

The existing Claude result classes can be revisited later if a provider-neutral
result abstraction becomes useful in real callers.

## Exceptions

Introduce provider-neutral base exceptions and keep existing Claude exception
classes as subclasses.

```python
class AIClientError(RuntimeError): ...
class ExecutableNotFoundError(AIClientError): ...
class AIClientTimeoutError(AIClientError): ...
class ProcessError(AIClientError): ...
class ProtocolError(AIClientError): ...
class SchemaValidationError(AIClientError): ...
class StructuredOutputMissingError(ProtocolError): ...
class StructuredOutputValidationError(ProtocolError): ...
class LimitError(ProcessError): ...
class RateLimitError(LimitError): ...
```

Then:

```python
class ClaudeCodeError(AIClientError): ...
class ClaudeTimeoutError(AIClientTimeoutError, ClaudeCodeError): ...
class ClaudeProcessError(ProcessError, ClaudeCodeError): ...
class ClaudeLimitError(LimitError, ClaudeProcessError): ...
class ClaudeRateLimitError(RateLimitError, ClaudeLimitError): ...
class ClaudeSchemaValidationError(SchemaValidationError, ClaudeCodeError): ...

class CodexCodeError(AIClientError): ...
class CodexTimeoutError(AIClientTimeoutError, CodexCodeError): ...
class CodexProcessError(ProcessError, CodexCodeError): ...
class CodexLimitError(LimitError, CodexProcessError): ...
class CodexRateLimitError(RateLimitError, CodexLimitError): ...
class CodexSchemaValidationError(SchemaValidationError, CodexCodeError): ...
class StructuredSchemaValidationError(
    ClaudeSchemaValidationError, CodexSchemaValidationError
): ...
```

The existing Claude exception names must remain import-compatible. New code can
catch the provider-neutral base classes. Because `StructuredSchema` is shared,
schema-construction failures use `StructuredSchemaValidationError`, which is
catchable as `SchemaValidationError`, `ClaudeSchemaValidationError`, and
`CodexSchemaValidationError`.

Rate-limit parsing should be shared only at the text-pattern level. Codex errors
may arrive as JSONL `error` events, stderr text, or process failures, so the
Codex error builder should summarize from those sources before calling the
existing `parse_rate_limit_info()`.

## Authentication And Security

Codex authentication should stay external to this library, matching the Claude
design. The client should assume the local `codex` CLI is installed and
authenticated, or that the caller provides per-run environment variables.

Recommended defaults:

- Do not read or manage `~/.codex/auth.json`.
- Do not expose `OPENAI_API_KEY` or `CODEX_API_KEY` helper parameters. Use
  `CodexRunOptions.env` for environment injection.
- Document that `CODEX_API_KEY` is for `codex exec` only and should be scoped to
  a single invocation in automation.
- Default permission mode should be `bypassPermissions`, which maps to
  `--dangerously-bypass-approvals-and-sandbox`.
- Do not add fine-grained Codex sandbox controls in the first release.
- Log whether the bypass flag is active, but do not log prompt text unless
  `log_prompts=True`.

## Structured Output Contract

Claude structured mode returns a Claude JSON envelope with a `structured_output`
field.

Codex structured mode returns the final response constrained by the schema. The
client should normalize this into `structured_output` in the Codex result, but
it should not pretend the raw protocol shape is the same as Claude's envelope.

Implementation detail:

1. Validate `StructuredSchema` locally.
2. Write schema JSON to a temporary file.
3. Run `codex exec --output-schema <tempfile>`.
4. Parse stdout as JSON for structured mode.
5. Validate the parsed object locally with the same `StructuredSchema`.
6. Return `CodexStructuredRunResult(payload=CodexJsonPayload(...), structured_output=...)`.

If `--json` and `--output-schema` are used together, prefer parsing JSONL events
and extracting the final agent message as JSON only after testing the exact CLI
behavior. Start without combining them.

Do not combine `--output-schema` with `codex exec resume`; resumed structured
mode should fail fast because the Codex resume subcommand does not support
schema-constrained output.

## Implementation Plan

### Phase 1: Compatibility Refactor

- Add provider-neutral base exceptions.
- Leave existing Claude result classes unchanged.
- Move Claude-specific parser code to a Claude parser module while preserving
  `stan_ai_client.parser` imports.
- Move `ClaudeCodeClient` to `claude.py`; keep `client.py` as an import shim.
- Ensure the current test suite passes unchanged.

### Phase 2: Codex Text Mode

- Add `CodexRunOptions`.
- Add `CodexClient.run_text()`.
- Build `codex exec` argv with model, cwd, permission mode, profile, config
  overrides, session resume, `--skip-git-repo-check`, and `extra_args`.
- Preserve prompt-over-stdin by using `codex exec -` when `input_mode="stdin"`.
- Normalize missing executable, timeout, non-zero exit, and logging.
- Add tests that assert argv and stdin behavior.

### Phase 3: Codex JSON Mode

- Add `CodexJsonPayload` and JSONL parser.
- Add `CodexClient.run_json()` with `--json`.
- Treat `error` and `turn.failed` events as process/protocol failures depending
  on process return code.
- Extract final agent message and usage metadata.
- Add tests for normal JSONL, malformed JSONL, error events, and non-zero exits.

### Phase 4: Codex Structured Mode

- Add temporary schema-file handling.
- Add `CodexClient.run_structured()`.
- Validate output locally.
- Add tests proving the temp file contains compact valid schema JSON and is not
  leaked in debug logs.
- Add tests for missing structured output, invalid structured output, and
  process errors.

### Phase 5: Docs And Examples

- Update README and DOCS with a backend overview.
- Add a Codex smoke example.
- Update ARCHITECTURE with the two-backend module map.
- Document which options are shared and which are backend-specific.

## Testing Strategy

Keep tests subprocess-free by monkeypatching `subprocess.run`, as the current
suite does.

Required tests:

- Current Claude tests remain green.
- Codex text mode constructs `("codex", "exec", "-")` for stdin mode.
- Codex text mode includes `--dangerously-bypass-approvals-and-sandbox` by
  default.
- Codex text mode omits the bypass flag when `permission_mode="default"`.
- Codex argv mode appends the prompt as the final argument.
- Codex resume by id constructs `("codex", "exec", "resume", "<id>", "-")`.
- Codex continue latest constructs `("codex", "exec", "resume", "--last", "-")`.
- Codex JSON mode parses JSONL events into `CodexJsonPayload`.
- Codex JSON mode surfaces `thread_id`, final message, usage, and raw events.
- Codex structured mode writes schema to a temp file and passes
  `--output-schema`.
- Codex structured mode validates the final object locally.
- Codex logs redact schema paths or contents where appropriate.
- Provider-neutral exceptions catch both Claude and Codex failures.
- Rate-limit retry policy retries Codex rate-limit errors when retry metadata is
  parseable.

Run before merge:

```bash
pytest
mypy src tests
ruff check .
```

## Implemented Outcome

- Added `CodexClient`, `CodexRunOptions`, `CodexJsonPayload`,
  `CodexJsonRunResult`, and `CodexStructuredRunResult`.
- Added provider-neutral base exceptions while keeping all existing Claude
  exception names import-compatible.
- Added Codex JSONL parsing, structured output via `--output-schema`, typed
  rate-limit handling, and safe logging redaction.
- Updated tests, docs, architecture notes, package metadata, and examples.

## Recommendation

Implement Codex support as a sibling backend, not as hidden branching inside
`ClaudeCodeClient`.

The first production-ready path should be `CodexClient` over `codex exec`
because it matches the current subprocess architecture and avoids adding a new
runtime dependency. Once that works, evaluate the `openai-codex` Python SDK for
longer-lived sessions, richer event handling, or async workflows.
