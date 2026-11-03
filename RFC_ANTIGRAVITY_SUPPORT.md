# Add Google Antigravity (`agy`) Backend Support

## Motivation

Add first-class support for Google Antigravity's local CLI, `agy`, as a fourth
backend next to `ClaudeCodeClient`, `CodexClient`, and `GrokClient`, with the
same surface: `run_text()`, `run_json()`, `run_structured()`, typed options,
typed payloads, normalized exceptions, usage normalization, and both retry
policies.

Everything below was observed on agy 1.4.12 (Linux, Google AI Pro sign-in) on
2026-11-03, from live runs, `agy changelog`, and the binary's own strings.
Later agy releases may change any of it; the unit tests pin the shapes the
client depends on.

## How `agy` Works Headlessly

### Invocation

- `agy -p/--print <prompt>` runs one turn and exits. `--prompt` is an alias.
  The prompt must be the flag's value; `-p` followed by another flag makes agy
  exit 2 ("-p took "--output-format" as its prompt"). The attached form
  `--print=<prompt>` is unambiguous for prompts that start with `-`.
- With a prompt flag, agy never reads stdin.
- `--input-format stream-json --output-format stream-json` (no `-p`) reads one
  NDJSON message per stdin line and runs one turn per message. The message is
  `{"event":"user","message":{"role":"user","content":"..."}}`. Prompts of
  300 KB work this way; argv is capped at 128 KiB per argument on Linux.
- Flags are Go `flag` flags: booleans accept `--flag=false`, durations need a
  unit (`--print-timeout 90s`; a bare `90` exits 2).
- Unknown `--output-format` values silently fall back to text, and an unknown
  `--mode` only warns, so the client owns those values.

### Output

`--output-format text` prints the response. `json` prints one envelope.
`stream-json` prints `init`, `step_update`, and a final `result` event whose
body is the `json` envelope:

```json
{"conversation_id":"…","status":"SUCCESS","response":"ok\n","duration_seconds":2.34,
 "num_turns":1,"usage":{"input_tokens":7503,"output_tokens":22,"thinking_tokens":21,
 "cache_read_tokens":0,"total_tokens":7525}}
```

Optional result fields: `error` (with `status: "ERROR"`), `finished_reason`
(`print_timeout`, `interrupted`, `error`; omitted for a normal finish),
`denied_actions` (`[{"action":"command","display_name":"RunCommand"}]`),
`structured_output`, `json_schema` (an echo), and `command` (slash commands).

The `init` event carries `permission_mode` (`request-review` by default,
`always-proceed` with `--dangerously-skip-permissions`). Each completed step
carries `step_type` (`user_input`, `agent_response`, `system_message`, `tool`,
`subagent`, `finish`, `error_message`), `state`, and, for model steps, its
own `usage`.

### Usage accounting

- `input_tokens` is the uncached prompt and `cache_read_tokens` the cached
  prompt, so they are disjoint: once caching started, one step reported 2,883
  input and 15,955 cache-read tokens for a context of about 18.8k.
  `output_tokens` includes `thinking_tokens` (a bare "ok" reply reported 25
  output with 24 thinking). `total_tokens` is `input + output` and omits cache
  reads. Cache writes are not reported.
- `usage`, `num_turns`, and `duration_seconds` in `result` are totals for the
  whole conversation, so they are cumulative after `--conversation`. Four
  resumes of one conversation reported 7,505 → 15,248 → 23,223 → 31,416 input
  tokens and `num_turns` 1 → 4.
- The per-step `usage` of the steps an invocation completed sums exactly to
  that invocation's share (checked against two resumed runs, including one
  with 74,292 cache-read tokens). A subagent's tokens appear in neither.
- A turn that ends early reports only its completed steps: an interrupted
  stream kept the 6,895 input tokens of its finished step, and a print timeout
  with no finished step reported all zeros.
- The result names no model. `--model gemini-4-argon` resolves to
  `gemini-4-argon-medium`; only the CLI log records the resolution.

### Exit codes and errors

| Exit | Meaning | Evidence |
|---|---|---|
| 0 | turn finished, or ended by `--print-timeout`, a headless denial, or a missing conversation; with `--input-format stream-json`, also an agent or model failure | result event (`status: "ERROR"`, `finished_reason: "error"` for a failure) |
| 1 | startup failure: model/effort selection, schema, empty prompt, unknown agent | `error: …` on stderr; JSON modes also print an `ERROR` envelope |
| 2 | flag usage error | Go usage text |
| 3 | agent or model failure under `-p` | `error: …` plus `AGY_ERROR: {...}` on stderr |

With stream-json input, agy treats the run as a multi-turn session and keeps
going after a failed turn, so it exits 0 even then. A network cut after 572
streamed characters produced exit 0, an `ERROR` result with the partial
response, and the same `error:` and `AGY_ERROR` stderr lines as `-p`.

The `AGY_ERROR` record has `short_error`, canonical `status`
(`RESOURCE_EXHAUSTED`, `UNKNOWN`, …), `error_code` (HTTP status when
`code_kind` is `http`, gRPC code when `grpc`), `retryable`, and `error_id`.
The binary also contains `You have exhausted your quota on this model.`,
`individual quota reached`, `MODEL_CAPACITY_EXHAUSTED`,
`PREFILL_QUEUE_OVERLOADED`, and `Your AI credits balance is too low to
continue.`. agy already retries transient 429/5xx calls in-process before
failing.

The `error_id` UUID can contain `429` as a substring (one observed ID was
`560d6d86-fbc6-429b-…`), and conversation-not-found warnings embed caller IDs,
so raw stderr cannot be matched against broad markers.

When the network is down at startup, agy reports a misleading model error
(`plan model not specified`, or `--effort is not supported for model …`); the
connection failure appears only in the CLI log.

### Sessions

- `--conversation <id>` resumes; `-c/--continue` resumes the workspace's most
  recent conversation. IDs must match exactly; prefixes and other casing fail.
- A missing ID is only a warning: `warning: conversation "<id>" not found`,
  exit 0, and the prompt runs in a brand-new conversation.
- **Stale structured output:** when a resumed turn's `finish` call fails
  schema validation, the result still reports the previous turn's
  `structured_output` with `status: "SUCCESS"`. Only a completed
  `step_type: "finish"` step in this invocation's events proves it is current.

### Structured output

`--json-schema` (inline or file path) needs `json` or `stream-json` output and
a root `"type": "object"`. agy validates the model's `finish` call against it
(it accepted `$defs`/`$ref`, `anyOf`, `const`, `pattern`, `uniqueItems`) and
omits `structured_output` when the final answer does not validate.

### Permissions

- Without `--dangerously-skip-permissions`, a headless run auto-denies tools
  that need approval, records them in `denied_actions`, and exits 0. When
  nothing was produced it prints `jetski: no output produced — a tool required
  the "command" permission that headless mode cannot prompt for…`. Text output
  has no denial signal once the model wrote anything.
- `--mode accept-edits|plan|auto` selects the agent mode. On this host (Ubuntu
  24.04 with `kernel.apparmor_restrict_unprivileged_userns=1`) the terminal
  sandbox cannot start, so `--mode auto` warns and falls back to the default
  mode, and `--sandbox` runs commands **unsandboxed without any stderr
  warning** (only the CLI log says so). A write outside the workspace
  succeeded under `--sandbox`.
- Repeating `--dangerously-skip-permissions` is harmless, so a wrapper such as
  `agy-all` can add it too.

### Other behavior

- `--print-timeout` returns partial output with `status: "SUCCESS"`,
  `finished_reason: "print_timeout"`, exit 0, and an `[agy] print timeout…`
  stderr line. The default is unlimited.
- Headless runs start agy's 15-minute self-update check; the binary honors
  `AGY_CLI_DISABLE_AUTO_UPDATE`.
- `-p /usage --output-format json` returns quota buckets (`remaining_fraction`,
  `reset_time`) without a model call. Quota errors themselves carry no reset
  time.
- A headless run is a single process.

## Proposal

`AntigravityClient` drives `agy` with these choices:

- **Protocol:** JSON and structured calls always read `stream-json`. The
  default `input_mode="stdin"` sends every prompt, text calls included, as one
  stream-json stdin message: no argv size limit, no flag ambiguity, and no
  prompt in `ps`. `input_mode="argv"` sends `--print=<prompt>`; text calls in
  that mode read plain stdout.
- **Defaults:** `default_model="gemini-4-argon"`, `default_effort="high"` (the
  highest level current agy models offer), `default_permission_mode=
  "bypassPermissions"` (`--dangerously-skip-permissions`, matching Codex's
  bypass default), 120 s timeout, `AGY_CLI_DISABLE_AUTO_UPDATE=1` unless the
  caller sets it.
- **Permission vocabulary:** `default` (no flag), `acceptEdits`, `plan`,
  `auto` (`--mode …`), `bypassPermissions`.
- **Failure handling:** a non-zero exit builds a typed process error; a zero
  exit with `status != "SUCCESS"` or any `finished_reason` is also a failure
  (the only signal in the default stdin mode), and
  `print_timeout` raises `AntigravityTimeoutError` with the partial payload.
  A missing conversation raises `AntigravitySessionNotFoundError` even though
  agy exited 0. In auto mode, `denied_actions` raise
  `AntigravityApprovalRequiredError`.
- **Trusted evidence:** classification reads only the result `error`, the
  `AGY_ERROR` short error and HTTP code, and `error:` stderr lines. Order:
  capacity/server (HTTP 529 and 500/502/503/504, capacity reasons), then
  quota (`RESOURCE_EXHAUSTED`, HTTP 429, quota wording other than Go's
  `disk quota exceeded`, standalone `429`),
  then network (shared markers plus Go's `no such host`, `server
  misbehaving`, `network is down`), then missing conversation.
- **Structured freshness:** a resumed structured call requires a completed
  `finish` step; otherwise it raises `AntigravityStructuredOutputMissingError`
  and keeps the stale value on the payload.
- **Usage:** payloads carry agy's `usage`, a `usage_scope` (`invocation` for a
  fresh call that finished, `cumulative` after a resume, `unknown` when the
  turn ended early), and `step_usage`. `normalize_ai_usage("antigravity",
  payload)` counts fresh input + cache reads + output, uses the step sum for
  any non-invocation scope, and falls back to the Codex-style
  previous-snapshot delta for bare mappings.

## What Cannot Be Matched 1:1

- No resolved model name, cost, or cache-write counts in the output, and agy's
  own `total_tokens` leaves out cache reads.
- Subagent tokens are not included in any usage agy reports.
- Network outages at startup surface as misleading model errors, so they stay
  ordinary `AntigravityProcessError`s.
- Quota errors carry no reset time, so `RateLimitRetryPolicy` usually re-raises
  immediately; `/usage` has the bucket reset times.
- Approval stops are detectable only from stream-json `denied_actions`;
  `input_mode="argv"` text runs return normally.
- Auto mode and `--sandbox` depend on the host's terminal sandbox.
- If `agy` on `PATH` is a wrapper that always adds
  `--dangerously-skip-permissions`, `permission_mode` cannot narrow it; point
  `executable` at the real binary instead.

## Rejected Alternatives

- **`--output-format json` for JSON mode.** It is the same result envelope but
  drops the per-step events that make resumed usage and structured freshness
  checkable.
- **argv-only prompt delivery (as Grok does).** It hits the 128 KiB argument
  limit and exposes prompts in process listings, while stdin delivery works.
- **Driving `agy` with the `--settings` permission allow-list.** That is
  user configuration, so it stays behind the `settings` option.

## Test Plan

- Unit tests with fixtures shaped like captured 1.4.12 output: argv and stdin
  construction, permission and option mapping, redaction, payload parsing,
  step-usage sums, stale structured output, missing conversations, approval
  stops, print timeouts, the `AGY_ERROR` classifier (including an `error_id`
  that spells `429`), network markers, partial payloads after a client
  timeout, and both retry policies.
- `examples/antigravity_smoke_test.py` for live text, JSON, structured,
  resume, and usage runs.
