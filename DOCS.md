# stan_ai_client Docs

## Overview

`stan_ai_client` is a thin Python wrapper around local AI coding CLI
executables.

It supports:

- `ClaudeCodeClient` over the local `claude` executable
- `CodexClient` over the local `codex exec` command
- `GrokClient` over the local `grok -p` headless mode
- `AntigravityClient` over the local `agy` (Google Antigravity) headless mode

It is designed for scriptable local CLI usage where:

- the selected CLI is already installed and authenticated
- the caller wants a Python API instead of manual subprocess handling
- text mode, JSON mode, or structured mode is enough
- structured error handling matters
- stdlib logging is sufficient

It does not:

- call Anthropic, OpenAI, xAI, or Google APIs directly
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
agy --version
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
ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]
CodexReasoningEffort = Literal[
    "minimal", "low", "medium", "high", "xhigh", "max"
]
GrokEffort = Literal["low", "medium", "high", "xhigh", "max"]
AntigravityEffort = Literal["low", "medium", "high", "xhigh", "max"]
```

These types are exported from `stan_ai_client`. The generic `Effort` and
`ReasoningEffort` names remain available from `stan_ai_client.types` for
backward compatibility, but new code should use the provider-specific names.
The selected provider CLI remains responsible for validating whether a specific
model supports a value.
For `grok-4.7`, use `low`, `medium`, `high`, or `xhigh`; `max` remains available
in the type for model overrides that support it. Antigravity validates effort
per model: agy 1.4.12 models accept `low`, `medium`, and `high`
(`gpt-oss-120b` only `medium`), and agy exits 1 for any other combination.

### `ClaudeCodeClient`

```python
class ClaudeCodeClient:
    def __init__(
        self,
        *,
        executable: str = "claude",
        default_model: str | None = "claude-opus-5-5",
        default_effort: ClaudeEffort | None = "max",
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
        default_model: str = "gpt-6.1-sol",
        default_reasoning_effort: CodexReasoningEffort = "max",
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
        default_model: str = "grok-4.7",
        default_effort: GrokEffort | None = None,
        default_timeout_seconds: float = 120.0,
        default_options: GrokRunOptions | None = None,
        logger: logging.Logger | None = None,
        log_prompts: bool = False,
    ) -> None: ...
```

GrokClient drives `grok --no-auto-update -p`. It always passes `--model`
(defaulting to `grok-4.7`). Prompt delivery is
handled transparently inside the client.

### `AntigravityClient`

```python
class AntigravityClient:
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
    ) -> None: ...
```

Methods:

```python
def run_text(
    ...,
    options: AntigravityRunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
    provider_retry_policy: ProviderRetryPolicy | None = None,
) -> TextRunResult: ...
def run_json(...) -> AntigravityJsonRunResult: ...
def run_structured(..., schema: StructuredSchema[T], ...) -> AntigravityStructuredRunResult[T]: ...
```

`AntigravityClient` drives Google Antigravity's `agy` CLI headlessly. By
default it sends the prompt as one `stream-json` message on stdin
(`--input-format stream-json --output-format stream-json`), which has no argv
size limit and keeps prompts out of process listings. JSON and structured
calls always parse `stream-json`, even with `input_mode="argv"`, so the
payload keeps the per-step events used for usage and freshness checks.

`default_model=None` or `default_effort=None` omits `--model`/`--effort` so
agy uses its configured model and that model's default thinking level. The
default permission mode, `bypassPermissions`, passes
`--dangerously-skip-permissions`; repeating the flag is harmless if `agy` on
`PATH` is a wrapper that already adds it, but such a wrapper also makes
narrower `permission_mode` values ineffective, so point `executable` at the
real binary for those. Every invocation also sets
`AGY_CLI_DISABLE_AUTO_UPDATE=1` unless `os.environ` or `options.env` already
set it (use an empty string to keep agy's self-updater).

## StructuredSchema

`StructuredSchema` is shared by all clients.

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

Claude, Grok, and Antigravity structured modes pass `cli_json` inline with
`--json-schema`. Codex structured mode writes `cli_json` to a temporary file and
passes the path with `--output-schema`. Antigravity rejects schemas whose root
`type` is not `"object"`; `validate_antigravity_output_schema(schema)` runs that
check without invoking agy, and `run_structured` applies it first.

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

### `AntigravityRunOptions`

`AntigravityRunOptions` controls one `agy` invocation.

```python
@dataclass(frozen=True)
class AntigravityRunOptions:
    cwd: str | Path | None = None
    model: str | None = None
    effort: AntigravityEffort | None = None
    timeout_seconds: float | None = None
    input_mode: Literal["stdin", "argv"] | None = None
    permission_mode: Literal[
        "default", "acceptEdits", "plan", "auto", "bypassPermissions"
    ] | None = None
    session_id: str | None = None
    continue_last_session: bool | None = None
    add_dirs: tuple[str | Path, ...] | None = None
    agent: str | None = None
    project: str | None = None
    settings: str | None = None
    sandbox: bool | None = None
    disable_slash_commands: bool | None = None
    print_timeout_seconds: float | None = None
    log_file: str | Path | None = None
    extra_args: tuple[str, ...] | None = None
    env: Mapping[str, str] | None = None
```

Important mappings:

- `cwd`: subprocess working directory (agy's workspace)
- `model`: `--model`; base names such as `gemini-4-argon` or `claude-opus-5-5`
  combine with `effort`, while variant slugs such as `gemini-4-argon-high`
  conflict with a different `effort`
- `effort`: `--effort`
- `input_mode="stdin"` (default): prompt as one stream-json stdin message
- `input_mode="argv"`: `--print=<prompt>`; text calls then read plain stdout.
  Linux caps one argument at 128 KiB
- `permission_mode="default"`: no flag; headless agy auto-denies tools that
  need approval and lists them in `payload.denied_actions`
- `permission_mode="acceptEdits"` / `"plan"` / `"auto"`: `--mode accept-edits`
  / `--mode plan` / `--mode auto`
- `permission_mode="bypassPermissions"` (client default):
  `--dangerously-skip-permissions`
- `session_id`: `--conversation <id>`; must be an exact conversation ID
- `continue_last_session`: `--continue` (most recent conversation in the
  workspace; prefer `session_id` when runs can overlap)
- `add_dirs`: repeated `--add-dir`
- `agent`: `--agent`; `project`: `--project`; `settings`: `--settings` (inline
  JSON or a file path, applied for this run only)
- `sandbox`, `disable_slash_commands`: `--sandbox` / `--disable-slash-commands`
  for `True`, `--flag=false` for `False`, nothing for `None`
- `print_timeout_seconds`: `--print-timeout` as a Go duration (`90s`, `1500ms`)
- `log_file`: `--log-file`
- `extra_args`: escape hatch for other `agy` flags, placed before the protocol
  flags so they cannot change the output format

`session_id` and `continue_last_session` are mutually exclusive.

### Automatic permission review

`permission_mode="auto"` selects each CLI's own automatic review mode:

| Client | Emitted flag | Existing default |
| --- | --- | --- |
| Claude | `--permission-mode auto` | omit the permission flag |
| Codex | `--approve-for-me` | `bypassPermissions` |
| Grok | `--permission-mode auto` | omit the permission flag |
| Antigravity | `--mode auto` | `bypassPermissions` |

Auto mode never emits Claude's `--dangerously-skip-permissions`, Codex's
`--dangerously-bypass-approvals-and-sandbox`, Grok's `--always-approve`, or
Antigravity's `--dangerously-skip-permissions`.
Other options—tools, cwd, prompts, session flags, output format, and timeouts—
are resolved and emitted exactly as in other permission modes.

Machine-readable approval stops raise the provider-neutral
`ApprovalRequiredError` through a concrete provider type:

- `ClaudeApprovalRequiredError`
- `CodexApprovalRequiredError`
- `GrokApprovalRequiredError`
- `AntigravityApprovalRequiredError`

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
auto-review denial record in `codex exec --json` events; Antigravity uses the
result's `denied_actions`. This makes detection available in Claude
JSON/structured, Grok JSON/structured, Codex JSONL, and every Antigravity
`stream-json` call (all calls in the default stdin input mode).
Antigravity's `approval_prompt` is the model's response when it wrote one,
otherwise agy's own "no output produced" denial notice;
`AntigravityApprovalRequiredError.denied_actions` lists what agy refused.
Modes without a trustworthy machine-readable signal retain their normal
error behavior instead of classifying text heuristically.

agy's auto mode needs its terminal sandbox. On hosts where unprivileged user
namespaces are blocked (Ubuntu 24.04's default AppArmor policy), agy warns and
continues in its default mode, which denies commands and file writes that
`settings.json` does not allow-list; the client logs that warning. The same
hosts run `sandbox=True` commands without isolation and without a warning.

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

### Antigravity Text

```python
from stan_ai_client import AntigravityClient

result = AntigravityClient().run_text("Output YAML only.")
print(result.text)
```

In the default stdin input mode, `result.text` is the result's `response`
and `result.stdout` is the raw `stream-json` stream. With
`input_mode="argv"`, agy prints plain text and `stdout` holds it.

### Antigravity JSON

```python
result = AntigravityClient().run_json("Summarize this repository.")
print(result.payload.response)
print(result.payload.conversation_id)
print(result.payload.invocation_usage)
```

JSON mode parses agy's `stream-json` events and returns the final `result`
event as `AntigravityJsonPayload`. A zero exit whose result has
`status != "SUCCESS"` or any `finished_reason` is raised as a process error,
except `finished_reason="print_timeout"`, which raises `AntigravityTimeoutError`
with the partial payload. In the default stdin mode this result is the only
failure signal: agy treats stream-json input as a multi-turn session and exits
0 after a failed turn, although it still prints its `error:` and `AGY_ERROR`
lines.

### Antigravity Structured

```python
result = AntigravityClient().run_structured("Summarize this repository.", schema=schema)
print(result.structured_output)
```

Structured mode adds `--json-schema <compact-json>`. agy validates the model's
answer and omits `structured_output` when it fails, which raises
`AntigravityStructuredOutputMissingError`; a returned value is also validated
locally against the same schema. On a resumed conversation agy re-reports the previous turn's
`structured_output` when the new answer fails validation, so resumed calls
also require a completed `finish` step in this invocation's events and
otherwise raise `AntigravityStructuredOutputMissingError`, keeping the stale
value on `error.payload`.

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

### Antigravity results

```python
@dataclass(frozen=True)
class AntigravityJsonRunResult:
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: AntigravityJsonPayload

@dataclass(frozen=True)
class AntigravityStructuredRunResult(Generic[TStructured]):
    command: CommandMetadata
    stdout: str
    stderr: str
    returncode: int
    payload: AntigravityJsonPayload
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

`AntigravityJsonPayload` exposes agy's result fields (`conversation_id`,
`status`, `response`, `error`, `finished_reason`, `duration_seconds`,
`num_turns`, `usage`, `denied_actions`, `structured_output`) plus:

- `events`: every decoded `stream-json` event, including `init`
- `extras`: unknown result fields such as `json_schema`
- `usage_scope`: `invocation` for a fresh call that finished normally,
  `cumulative` for resumed calls (agy's `usage`, `num_turns`, and
  `duration_seconds` are conversation totals), and `unknown` when the turn
  ended early (its `usage` then covers completed steps only)
- `step_usage`: summed usage of the steps this invocation completed (`None` if
  a stream line could not be decoded)
- `invocation_usage`: `usage` for `invocation` scope, `step_usage` otherwise
- `requested_model`: the `--model` value sent (agy does not report the
  resolved model)
- `duration_ms`: measured client-side
- `session_id` and `permission_mode` (from the `init` event) properties

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

Antigravity reports `input_tokens` without cache reads and `cache_read_tokens`
separately; its `output_tokens` include `thinking_tokens`, it reports no cache
writes, and its own `total_tokens` omits cache reads. So 2,883 input, 15,955
cache-read, and 157 output tokens normalize to 2,883 fresh + 15,955 cached +
157 output = 18,995 total. Its result counters are conversation totals as
well, and a turn that ended early reports only its completed steps. Because every
completed `stream-json` step reports its own usage, a resumed
`AntigravityJsonPayload` carries `step_usage`, and
`normalize_ai_usage("antigravity", payload)` uses it directly without a
previous snapshot; the same applies to `unknown` scope. A mapping without
`step_usage` follows the Codex rules above. Usage of subagents that agy spawns is not included in any count agy
reports. `python examples/antigravity_smoke_test.py --usage-only` runs a fresh
and a resumed call.

## Exception Model

Provider-specific exceptions remain available:

- `ClaudeCodeError`
- `ClaudeExecutableNotFoundError`
- `ClaudeTimeoutError`
- `ClaudeProcessError`
- `ClaudeUnclassifiedProviderError`
- `ClaudeSessionNotFoundError` (failed explicit resume with missing local history)
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
- `CodexUnclassifiedProviderError`
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
- `GrokUnclassifiedProviderError`
- `GrokApprovalRequiredError`
- `GrokNetworkUnavailableError`
- `GrokProtocolError`
- `GrokRateLimitError`
- `GrokCancelledError`
- `GrokMalformedStructuredOutputError`
- `GrokStructuredOutputMissingError`
- `GrokStructuredOutputValidationError`
- `AntigravityCodeError`
- `AntigravityExecutableNotFoundError`
- `AntigravityTimeoutError` (client timeout or `print_timeout_seconds`; keeps
  partial `stdout`, `stderr`, and a recovered `payload`)
- `AntigravityProcessError`
- `AntigravityCapacityError`
- `AntigravityServerError`
- `AntigravitySessionNotFoundError` (the requested conversation does not
  exist; agy ran the prompt in a new conversation, described by `payload`)
- `AntigravityApprovalRequiredError`
- `AntigravityNetworkUnavailableError`
- `AntigravityProtocolError`
- `AntigravityRateLimitError`
- `AntigravityStructuredOutputMissingError`
- `AntigravityStructuredOutputValidationError`

Every Antigravity process, protocol, and timeout error also exposes
`conversation_id` and `agent_error`, the parsed `AGY_ERROR` record
(`AntigravityAgentError`: `short_error`, `status`, `error_code`, `code_kind`,
`retryable`, `error_id`, `raw`) when agy printed one.

Provider-neutral base classes are also exported:

- `AIClientError`
- `AIClientTimeoutError`
- `ExecutableNotFoundError`
- `ProcessError`
- `UnclassifiedProviderError`
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
provider-neutral exceptions when the caller should handle Claude, Codex, Grok,
and Antigravity the same way.

Every `GrokProtocolError`, `GrokCancelledError`, and
`GrokApprovalRequiredError` exposes `session_id`,
`request_id`, `stop_reason`, and `cancellation_category` when Grok supplied
them, while the complete raw streams remain available as `stdout` and `stderr`.
`GrokMalformedStructuredOutputError` also exposes a safe `detail` and
`json_value_count`; callers can log those without copying raw model output into
routine diagnostics.

## Unclassified Provider Failures

When a completed Claude, Codex, or Grok invocation fails and no existing
specific classifier applies, it raises a provider-specific subclass of
`UnclassifiedProviderError` instead of a plain provider `ProcessError`:

| Provider | New fallback | Existing compatible catches |
|---|---|---|
| Claude | `ClaudeUnclassifiedProviderError` | `ClaudeProcessError`, `ClaudeCodeError` |
| Codex | `CodexUnclassifiedProviderError` | `CodexProcessError`, `CodexCodeError` |
| Grok | `GrokUnclassifiedProviderError` | `GrokProcessError`, `GrokCodeError` |

All three are also `ProcessError` and `AIClientError` subclasses. Catch
`UnclassifiedProviderError` before a broader `ProcessError` handler to route
these failures separately. The legacy `stan_ai_client.client` module also
exports the common and Claude-specific types.

The fallback needs no recognized wording or error-record format. It covers
nonzero CLI exits and native error results already treated as failures by the
clients, including Claude/Grok error envelopes and Codex failed turns with
exit code zero. A failed invocation without any diagnostic text is still
unclassified. Successful output containing error-like prose remains output.
Existing timeout, executable/working-directory, protocol, validation,
approval, cancellation, missing-session, rate-limit, capacity, and network
handling keeps its precedence and behavior.

"Unclassified" means the client has no more specific error type. It does not
establish whether the cause is local or remote, transient or permanent.
Neither `rate_limit_policy` nor `provider_retry_policy` retries it, and
`classify_provider_failure` still returns `None` for these errors. This change
does not add capacity heuristics or correct failures already matched by an
existing classifier.

The exception retains the original message, `command`, `returncode`, `stdout`,
`stderr`, and `payload`, including native statuses and unknown fields retained
by the existing parsers. It is propagated directly through the client retry
wrappers, so callers receive the new concrete type and its process evidence.

Each occurrence emits one `WARNING` on the client's logger with
`unclassified_provider_error=true`. The `LogRecord` carries
`error_kind="unclassified_provider_error"` and `provider="claude"`, `"codex"`,
or `"grok"`. The diagnostic is the first 200 characters of the existing error
summary, escaped with Python's `repr` so embedded newlines remain on one log
line. Truncating this excerpt does not alter the exception's diagnostics.
Applications own notification delivery, deduplication, and any redaction
needed before sending diagnostics outside their logs; the library sends no
alerts itself.

## Network Availability

`NetworkUnavailableError` is the provider-neutral contract for strong evidence
that a provider transport is unavailable. The concrete exceptions also
retain their provider contracts:

- `ClaudeNetworkUnavailableError` is a `ClaudeProcessError` and `ClaudeCodeError`
- `CodexNetworkUnavailableError` is a `CodexProcessError` and `CodexCodeError`
- `GrokNetworkUnavailableError` is a `GrokProcessError` and `GrokCodeError`
- `AntigravityNetworkUnavailableError` is an `AntigravityProcessError` and
  `AntigravityCodeError`

All five types preserve `command`, `returncode`, `stdout`, `stderr`, and
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
successful output are not searched. Antigravity inspects only its result
`error`, the `AGY_ERROR` short error, and `error:` stderr lines, and also
recognizes Go resolver wording (`no such host`, `server misbehaving`) and
`network is down`. agy masks connectivity loss at startup as a model-selection
error, which stays an ordinary `AntigravityProcessError`. Capacity/server
evidence takes precedence over rate limits, and rate limits take precedence
over network errors. A bare stream disconnect remains the
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

## Provider Capacity and Server Errors

Claude, Codex, and Antigravity classify provider failures before quota and
network checks.
`CapacityError` and `ServerError` inherit `ProviderError`, which inherits
`ProcessError`. Concrete types also retain their provider hierarchy:

| Provider | Capacity | Server |
|---|---|---|
| Claude | `ClaudeCapacityError` | `ClaudeServerError` |
| Codex | `CodexCapacityError` | `CodexServerError` |
| Antigravity | `AntigravityCapacityError` | `AntigravityServerError` |

All preserve `command`, `returncode`, `stdout`, `stderr`, and `payload`, and
expose `failure: ProviderFailure`, `signal`, and `retry_after_seconds`.
`ProviderFailure` contains `kind` (`capacity` or `server_error`), `signal`,
`message`, and optional `retry_after_seconds`.

Claude's failed result `api_error_status=529` means capacity; 500, 502, 503,
and 504 mean server failure. The native status wins over message matching.
Fallbacks recognize `API Error: <status>` and `Repeated 529 Overloaded errors`
in the same trusted diagnostics as network detection. Codex recognizes
`Selected model is at capacity`, `currently experiencing high demand`, and
the server statuses in `unexpected status <status>` or
`exceeded retry limit, last status: <status>` diagnostics. Its exec JSONL
error events carry messages, not rollout `codex_error_info` codes.
Antigravity reads its `AGY_ERROR` record: HTTP-kind code 529 means capacity and
500, 502, 503, and 504 mean server failure; gRPC-kind codes are not statuses.
`MODEL_CAPACITY_EXHAUSTED`, `PREFILL_QUEUE_OVERLOADED`, and agy's "model API
is currently overloaded" wording also mean capacity, ahead of quota. Quota
wording, `RESOURCE_EXHAUSTED`, and HTTP 429 are rate limits
(`AntigravityRateLimitError`); agy's quota messages carry no reset time, so
`RateLimitRetryPolicy` re-raises them unless the text includes a retry hint.

`classify_provider_failure(error, *, codex_output_protocol="unstructured")`
exports this classifier for controllers with process evidence. It returns
`ProviderFailure | None`. The default guards Codex stderr with `ERROR:`;
pass `"jsonl"` only for JSON exec mode, whose stderr is diagnostic. Model and
tool output are never classifier inputs. HTTP `Retry-After` seconds or dates
in trusted diagnostics supply retry hints, without the quota parser's buffer.
Grok classification is unchanged. In particular, bare `overloaded` remains a
rate-limit marker, while a Claude 529 is now a capacity error and is no longer
caught by `ClaudeRateLimitError` or `LimitError`.

`ProviderRetryPolicy` enables retries on Claude/Codex/Antigravity `run_text`,
`run_json`, and `run_structured` through the `provider_retry_policy` keyword. The default
is `None`, so existing calls never gain an implicit wait. Policy fields are:

| Field | Default | Meaning |
|---|---|---|
| `first_delay` | `15.0` | First backoff, in seconds |
| `max_delay` | `900.0` | Backoff cap before jitter, in seconds |
| `jitter` | `0.2` | Fractional random variation (±20%) |
| `server_window` | `36000.0` | Seconds from the first server error |
| `cancel` | A new `threading.Event` | Interrupts retry waits when set |

`retry_delay(attempt, hint, policy)` is shared with external schedulers.
Attempts start at 1, exponential backoff doubles to the cap, and a valid
provider hint wins unchanged over the cap and jitter. Capacity has no attempt
or elapsed-time limit. Server retries stop if their next retry would land
after the window, including call execution time and intervening waits;
capacity failures do not reset that clock. A retry exactly at the deadline
is allowed. An in-flight call is governed by its existing timeout.

`run_with_provider_retry(call, prompt, policy, *, logger=None,
provider="Provider", model=None)` runs a callable accepting the prompt.
It catches only the typed capacity/server errors; unrelated errors and
interrupts propagate. Each retry sends `PROVIDER_RETRY_PREFIX + prompt`,
using the original prompt so prefixes never accumulate:

```text
We just ran this, but it failed on a provider capacity limit or server error. Retry and continue the work:

```

The helper logs the provider, model, kind, signal, attempt, and next retry
time at `WARNING`. Set the policy's cancel event to stop a wait, or interrupt
the calling process as usual. Cancellation does not kill an in-flight CLI or
change the existing quota-wait policy. Combining both retry policies preserves
the cumulative quota wait budget and the server window across alternating
errors. Client call options, schema, and selected model stay unchanged.

`ServerErrorWindowExceeded` and `ProviderRetryCancelled` inherit
`AIClientError`. Each exposes `attempts: tuple[FailedAttempt, ...]` and includes
the error table in its message. `FailedAttempt` pairs `when` (an aware datetime)
with `failure`. `render_failed_attempts(attempts)` renders Sofia local times,
collapses consecutive identical errors into one row, and escapes each error's
first line truncated to 200 characters. On exhaustion, `__cause__` retains the
last provider process error. A controller should use these helpers or enable
the client's policy, with one retry owner per operation.

## Logging

All clients use Python's stdlib `logging`.

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
- capacity/server retry waits and their next retry times
- unclassified provider failures with `error_kind` and `provider` log fields

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

Antigravity:

- stdin mode (default) runs
  `agy ... --input-format stream-json --output-format stream-json` and writes
  `{"event":"user","message":{"role":"user","content":<prompt>}}` to stdin
- argv mode appends `--print=<prompt>` and sends empty stdin; text calls use
  `--output-format text`, JSON and structured calls `stream-json`
- structured mode adds `--json-schema <compact-json>`
- `bypassPermissions` adds `--dangerously-skip-permissions`; `auto`,
  `acceptEdits`, and `plan` add `--mode auto|accept-edits|plan`
- argv logs redact the prompt, `--conversation`, `--settings`, and the schema

All clients copy `os.environ`, merge `options.env`, preserve raw stdout/stderr
on results and errors, and run synchronously through `subprocess.run`.

## Examples Included In The Repo

- [examples/smoke_test.py](./examples/smoke_test.py)
- [examples/codex_smoke_test.py](./examples/codex_smoke_test.py)
- [examples/grok_smoke_test.py](./examples/grok_smoke_test.py)
- [examples/antigravity_smoke_test.py](./examples/antigravity_smoke_test.py)
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
- Antigravity reports no resolved model, cost, or cache writes, counts no
  subagent usage, and gives quota errors no reset time
- Antigravity approval stops are detectable only in `stream-json` calls, so
  `input_mode="argv"` text calls return normally after a denial

For unsupported Claude flags, use `RunOptions(extra_args=...)`. For unsupported
Codex exec flags, use `CodexRunOptions(extra_args=...)`; for unsupported Codex
resume flags, use `CodexRunOptions(resume_extra_args=...)`. For unsupported
`agy` flags, use `AntigravityRunOptions(extra_args=...)`.

## Suggested Usage Boundary

`stan_ai_client` should stay the thin process wrapper layer.

Keep these concerns outside the library:

- prompt templates specific to one application
- YAML or domain-object parsing
- business scheduling and persistence around long-running work
- app-specific logging policy beyond execution metadata

`ClaudeCodeClient(default_model=None, default_effort=None)` omits `--model` and
`--effort` so Claude chooses its configured defaults. Explicit `RunOptions`
values take precedence over constructor defaults.

`ClaudeSessionNotFoundError` preserves the command, return code, stdout, stderr,
and parsed payload of `ClaudeProcessError`. It is classified only for a failed
command containing `--resume`, when stderr, stdout, or the payload result reports
missing conversation history. Rate-limit and network errors are classified first.
