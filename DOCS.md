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

### Provider-specific effort types

Effort values are named by provider so signatures and editor hints do not imply
that values can be exchanged across CLIs:

```python
ClaudeEffort = Literal["low", "medium", "high", "max"]
CodexReasoningEffort = Literal[
    "minimal", "low", "medium", "high", "xhigh", "max"
]
GrokEffort = Literal["low", "medium", "high", "max"]
```

These types are exported from `stan_ai_client`. The generic `Effort` and
`ReasoningEffort` names remain available from `stan_ai_client.types` for
backward compatibility, but new code should use the provider-specific names.
The selected provider CLI remains responsible for validating whether a specific
model supports a value.

### `ClaudeCodeClient`

```python
class ClaudeCodeClient:
    def __init__(
        self,
        *,
        executable: str = "claude",
        default_model: str = "claude-opus-4-8",
        default_effort: ClaudeEffort = "max",
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
        default_model: str = "gpt-5.6-sol",
        default_reasoning_effort: CodexReasoningEffort = "medium",
        default_permission_mode: Literal[
            "default", "bypassPermissions", "auto"
        ] = "bypassPermissions",
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
    capture_usage: bool = False,
) -> CodexStructuredRunResult[T]: ...
```

`CodexClient` defaults to `bypassPermissions`, which adds
`--dangerously-bypass-approvals-and-sandbox` to `codex exec`. Use
`CodexRunOptions(permission_mode="default")` or
`CodexClient(default_permission_mode="default")` to omit that flag.
Use `permission_mode="auto"` to pass `--approve-for-me` instead of the bypass
flag.

### `GrokClient`

```python
class GrokClient:
    def __init__(
        self,
        *,
        executable: str = "grok",
        default_model: str = "grok-4.5",
        default_effort: GrokEffort | None = None,
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
    effort: ClaudeEffort | None = None
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
- `permission_mode="auto"`: `--permission-mode auto`
- `permission_mode="bypassPermissions"`:
  `--dangerously-skip-permissions`
- `permission_mode="dontAsk"`: `--permission-mode dontAsk`; this denies
  unapproved actions without prompting and is not bypass
- `extra_args`: escape hatch for unsupported Claude flags

Omitting `permission_mode` preserves the Claude CLI's configured/default
posture. The installed CLI calls automatic review `auto`; the library option
does not emit the proposed but nonexistent `--enable-auto-mode` spelling.

### `CodexRunOptions`

`CodexRunOptions` controls one Codex invocation.

```python
@dataclass(frozen=True)
class CodexRunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    reasoning_effort: CodexReasoningEffort | None = None
    timeout_seconds: float | None = None
    input_mode: Literal["stdin", "argv"] | None = None
    permission_mode: Literal["default", "bypassPermissions", "auto"] | None = None
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
- `permission_mode="auto"`: `--approve-for-me`
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
    effort: GrokEffort | None = None
    timeout_seconds: float | None = None
    permission_mode: Literal[
        "acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"
    ] | None = None
    session_id: str | None = None
    continue_last_session: bool | None = None
    fork_session: bool | None = None
    permission_allow_rules: tuple[str, ...] | None = None
    permission_deny_rules: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None
    excluded_tools: tuple[str, ...] | None = None
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
- `permission_allow_rules`: repeated `--allow <rule>` permission rules
- `permission_deny_rules`: repeated `--deny <rule>` permission rules
- `tools`: `--tools <comma-separated-tool-ids>`; this is the positive filter
  for the built-in tools visible to Grok
- `excluded_tools`: `--disallowed-tools <comma-separated-tool-ids>`; removes
  built-in tools from Grok's visible inventory
- `system_prompt`: `--system-prompt-override`
- `add_dirs`: accepted for API symmetry but not emitted; Grok currently has no
  documented add-directory flag distinct from `--cwd`
- `max_turns`: `--max-turns`
- `extra_args`: escape hatch for unsupported Grok flags

Prompt delivery is automatic: short prompts use `-p <prompt>`, while long
prompts use `--prompt-file <tempfile>`. Generated invocations include
`--no-auto-update` by default for headless automation.

Permission rules and tool filtering are deliberately separate. For a
read-only headless run, prefer a positive tool inventory:

```python
GrokRunOptions(tools=("read_file", "grep", "list_dir"))
```

Using `permission_deny_rules=("Bash",)` would deny shell calls if the
tool were visible, but would not prevent Grok from choosing it. In unattended
runs, a visible tool without a matching permission rule can still trigger a
permission cancellation.

### Automatic permission review

`permission_mode="auto"` selects each CLI's own automatic review mode:

| Client | Emitted flag | Existing default |
| --- | --- | --- |
| Claude | `--permission-mode auto` | omit the permission flag |
| Codex | `--approve-for-me` | `bypassPermissions` |
| Grok | `--permission-mode auto` | omit the permission flag |

Auto mode never emits Claude's `--dangerously-skip-permissions`, Codex's
`--dangerously-bypass-approvals-and-sandbox`, or Grok's `--always-approve`.
Other options—tools, cwd, prompts, session flags, output format, and timeouts—
are resolved and emitted exactly as in other permission modes.

Machine-readable approval stops raise the provider-neutral
`ApprovalRequiredError` through a concrete provider type:

- `ClaudeApprovalRequiredError`
- `CodexApprovalRequiredError`
- `GrokApprovalRequiredError`

Every approval error is also its provider's `ProcessError` subtype and exposes
`command`, `returncode`, `stdout`, `stderr`, and `payload`. Its
`approval_prompt` preserves the caller-facing text exactly as decoded from the
CLI response; it is not stripped or summarized.

```python
from stan_ai_client import ApprovalRequiredError, ClaudeCodeClient, RunOptions

try:
    ClaudeCodeClient().run_structured(
        "Apply the change.",
        schema=schema,
        options=RunOptions(permission_mode="auto"),
    )
except ApprovalRequiredError as exc:
    relay_to_user(exc.approval_prompt)
```

There is intentionally no stdin prompt, approval grant, automatic acceptance,
or retry loop. Claude classification uses JSON `permission_denials`; Grok uses
its explicit permission-cancellation category; Codex uses the native
auto-review denial record in `codex exec --json` events. This makes detection
available in Claude JSON/structured, Grok JSON/structured, and Codex JSONL
modes. Modes without a trustworthy machine-readable signal retain their normal
error behavior instead of classifying text heuristically.

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

To collect usage, opt in per call:

```python
from stan_ai_client import normalize_ai_usage

result = client.run_structured("Summarize this repository.", schema=schema, capture_usage=True)
facts = normalize_ai_usage("codex", result.payload)
print(result.structured_output["summary"], facts.tokens.total_tokens)
```

This adds `--json` and a fresh `--output-last-message` file. The file supplies
the validated structured result; `.stdout` contains raw JSONL and `.payload`
retains thread ID, usage, and decoded events. `usage_diagnostics` describes
missing or malformed event data. A valid final answer still succeeds when
accounting is unavailable. Terminal provider failures retain typed errors;
timeouts retain partial stdout/stderr, and in this mode also a recovered payload.
Result and schema files are cleaned up on every exit. Default structured,
text, and JSON calls keep their existing protocols.

Codex schemas are additionally checked against the OpenAI structured-output
subset before the temporary file is created. The root schema must be an object,
object properties must all be listed in `required`, and objects must set
`additionalProperties: false`. Root-level `anyOf`, boolean subschemas, external
references, and unresolved local references are also rejected. Unsupported
schema keywords such as `allOf`,
`oneOf`, `not`, `dependentRequired`, `dependentSchemas`, `if`, `then`, `else`,
and `uniqueItems`, plus unsupported reference, content, and schema-container
keywords such as `$dynamicRef`, `contentEncoding`, `contentSchema`, `contains`,
`prefixItems`, `patternProperties`, `propertyNames`, `unevaluatedItems`, and
`unevaluatedProperties`, are rejected locally with the offending JSON path.

The same check is exported as `validate_codex_output_schema(schema)`. It runs
no subprocess and makes no network request, so callers can validate schemas at
startup or in CI instead of discovering an incompatibility mid-run:

```python
from stan_ai_client import StructuredSchema, validate_codex_output_schema

schema = StructuredSchema.from_dict(
    {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            }
        },
        "required": ["tags"],
        "additionalProperties": False,
    }
)
validate_codex_output_schema(schema)  # raises CodexSchemaValidationError locally
```

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
- `usage_diagnostics` (best-effort structured capture)
- `usage_scope` (`invocation`, `cumulative`, or `unknown`)

### Normalized usage

`normalize_ai_usage(provider, payload, *, previous_snapshot=None)` accepts a
typed provider payload or a mapping. It returns `UsageFacts` containing
`tokens: TokenUsage`, `models: tuple[ModelUsage, ...]`, session/plan identity,
reported cost, copied raw facts, and diagnostics. It performs no I/O.

`TokenUsage` has `fresh_input_tokens`, `cache_read_input_tokens`,
`cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens`,
`unsplit_tokens`, and `total_tokens`. Reasoning is already inside output;
the other components and unsplit remainder are disjoint. Missing, unsupported,
and invalid counters are `None`; measured zero remains zero. Only nonnegative
integers fitting signed 64-bit storage are accepted. Reported finite,
nonnegative USD values are preserved without estimating prices.

Claude's `modelUsage` includes subagents and is counted once instead of adding
the smaller top-level `usage` envelope. Only its documented `inputTokens`,
`cacheReadInputTokens`, `cacheCreationInputTokens`, `outputTokens`, and
`costUSD` fields are normalized; other fields remain in `raw`. An incomplete
or malformed breakdown preserves known components but has no computed total.
Reported model names and costs remain available even when their token counters
are unusable. If no model has usable counters, whole-call facts fall back to
available top-level usage and assign those tokens to an additional unknown-model
row, with an attribution diagnostic. Codex input includes cached input and output
includes reasoning:
1,000 input, 800 cached, and 200 output yield 200 fresh + 800 cached + 200 output
= 1,200 total. Invalid cache splits preserve an independently valid total as
partly unsplit. Legacy `total_tokens` stays unsplit; Grok currently exposes
identity only.

Model names come from provider metadata (`reported`). A mapping may supply
`requested_model` for a single-model fallback (`requested`); otherwise the
row has no model (`unknown`). Multi-model totals are never assigned to that
hint. Unknown provider fields remain in `raw`.

Codex CLI 0.154.0 reports cumulative session counters, confirmed by the
fresh/resume smoke fixture and its
[JSONL emitter](https://github.com/openai/codex/blob/rust-v0.154.0/codex-rs/exec/src/event_processor_with_jsonl_output.rs).
Structured capture marks
fresh calls as `invocation` (a new session starts at zero) and resumed calls as
`cumulative`. A resumed call without a baseline has unavailable invocation
usage. Raw Codex events lack launch context and default to `unknown`; callers
must identify their scope. Claude result envelopes default to invocation scope.
Passing a previous snapshot alone never subtracts an invocation-scoped result.

For cumulative counters, the caller supplies the immediately preceding captured
snapshot from the same session. Compatible monotonic raw counters permit a
delta. If ordering is ambiguous, pass no baseline; a missing baseline, reset,
changed counter shape, or unknown scope leaves invocation tokens unavailable.
Raw cumulative facts remain intact.

```python
facts = normalize_ai_usage(
    "codex",
    resumed_result.payload,
    previous_snapshot=first_result.payload,
)
```

Run `python examples/codex_smoke_test.py --usage-only` for two small live
structured calls (fresh and resumed), with CLI/client versions and counters.

## Exception Model

Provider-specific exceptions remain available:

- `ClaudeCodeError`
- `ClaudeExecutableNotFoundError`
- `ClaudeTimeoutError`
- `ClaudeProcessError`
- `ClaudeApprovalRequiredError`
- `ClaudeNetworkUnavailableError`
- `ClaudeProtocolError`
- `ClaudeRateLimitError`
- `ClaudeStructuredOutputMissingError`
- `ClaudeStructuredOutputValidationError`
- `CodexCodeError`
- `CodexExecutableNotFoundError`
- `CodexTimeoutError`
- `CodexProcessError`
- `CodexApprovalRequiredError`
- `CodexNetworkUnavailableError`
- `CodexProtocolError`
- `CodexRateLimitError`
- `CodexStructuredOutputMissingError`
- `CodexStructuredOutputValidationError`
- `GrokCodeError`
- `GrokExecutableNotFoundError`
- `GrokTimeoutError`
- `GrokProcessError`
- `GrokApprovalRequiredError`
- `GrokNetworkUnavailableError`
- `GrokProtocolError`
- `GrokRateLimitError`
- `GrokCancelledError`
- `GrokMalformedStructuredOutputError`
- `GrokStructuredOutputMissingError`
- `GrokStructuredOutputValidationError`

Provider-neutral base classes are also exported:

- `AIClientError`
- `AIClientTimeoutError`
- `ExecutableNotFoundError`
- `ProcessError`
- `ApprovalRequiredError`
- `NetworkUnavailableError`
- `ProtocolError`
- `SchemaValidationError`
- `StructuredSchemaValidationError`
- `StructuredOutputMissingError`
- `StructuredOutputValidationError`
- `LimitError`
- `RateLimitError`

Catch provider-specific exceptions when you care which CLI failed. Catch
provider-neutral exceptions when the caller should handle Claude, Codex, and
Grok the same way.

Every `GrokProtocolError`, `GrokCancelledError`, and
`GrokApprovalRequiredError` exposes `session_id`,
`request_id`, `stop_reason`, and `cancellation_category` when Grok supplied
them, while the complete raw streams remain available as `stdout` and `stderr`.
`GrokMalformedStructuredOutputError` also exposes a safe `detail` and
`json_value_count`; callers can log those without copying raw model output into
routine diagnostics.

## Network Availability

`NetworkUnavailableError` is the provider-neutral contract for strong evidence
that a provider transport is unavailable. The concrete exceptions also
retain their provider contracts:

- `ClaudeNetworkUnavailableError` is a `ClaudeProcessError` and `ClaudeCodeError`
- `CodexNetworkUnavailableError` is a `CodexProcessError` and `CodexCodeError`
- `GrokNetworkUnavailableError` is a `GrokProcessError` and `GrokCodeError`

All four types preserve `command`, `returncode`, `stdout`, `stderr`, and
`payload`. Classification recognizes strong diagnostics such as DNS lookup or
temporary name-resolution failure, network unreachable, and no route to host.
Claude also recognizes its own `Unable to connect to API` and `Connection
closed mid-response` failures. Classification examines provider-declared error
payloads or events and guarded process diagnostics. Claude and Grok inspect
process stderr plus Claude `API Error:` and Grok `Error:` stdout lines. Codex
inspects JSONL errors and JSON-mode stderr (also for structured usage capture);
because text- and default structured-mode
stderr also carries progress, only `ERROR:`-prefixed lines are inspected in
those modes. Ordinary transcripts, prompts, documentation, diffs, and
successful output are not searched. Rate-limit evidence on any of those
diagnostic channels retains precedence. A bare stream disconnect remains the
provider's ordinary process error. The library does not sleep, retry, or impose
scheduling policy for this condition.

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
- `auto` adds `--approve-for-me` instead of the bypass flag

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
