# stan_ai_client Architecture

## Overview

`stan_ai_client` is a small synchronous Python wrapper around local AI coding
CLI executables.

It currently supports four backend clients:

- `ClaudeCodeClient` shells out to `claude -p`.
- `CodexClient` shells out to `codex exec`.
- `GrokClient` shells out to `grok -p` (with transparent prompt handling).
- `AntigravityClient` shells out to `agy` in headless `stream-json` mode.

The package does not call Anthropic, OpenAI, xAI, or Google APIs directly. Authentication,
account selection, and provider configuration stay in the underlying CLI. The
library builds commands, captures stdout and stderr, parses supported machine
output, validates structured responses locally, and normalizes failures into
typed exceptions.

All clients expose:

- `run_text()` for plain text
- `run_json()` for machine-readable output
- `run_structured()` for schema-guided structured output

All execution modes can optionally run through a `RateLimitRetryPolicy`. The
policy is a synchronous wrapper around one-attempt execution: it catches the
client's typed rate-limit exception, sleeps only when parsed reset metadata fits
the caller's wait budget, logs the wait, and retries the same operation.

Claude, Codex, and Antigravity raise typed capacity and server errors,
classified before quota and network checks from Claude's native API status,
Antigravity's `AGY_ERROR` record, or the trusted diagnostics that network
detection reads. Their modes can also opt into a
`ProviderRetryPolicy`, which runs outside the rate-limit wrapper: it waits with
cancellable backoff and resends the original prompt prefixed with
`PROVIDER_RETRY_PREFIX`. Neither wrapper resets the other's budget, so one call
keeps a single quota wait budget and a single server-error window.

## Execution Flow

### Automatic permission review

Automatic permission review is a non-interactive boundary. When a provider
exposes a trustworthy machine-readable approval stop, the client raises
`ApprovalRequiredError`; the wrapper never grants approval, prompts on stdin,
or retries the withheld action.

### Grok text / JSON / structured mode

Similar to Claude but using
`grok --no-auto-update -p --output-format plain|json [--json-schema ...]`.
Prompts are passed as argv value (or `--prompt-file` for long prompts).
Session resume via `--resume` / `--continue` / `--session-id`.
The returned `GrokJsonPayload` is intentionally minimal (text + ids + optional
structuredOutput). Structured mode also accepts newer Grok builds that return
the raw validated JSON value from `--json-schema`. Duration is measured
client-side.

### Antigravity text / JSON / structured mode

1. `AntigravityClient.run_*()` applies optional provider and rate-limit retries
2. single-attempt execution resolves `AntigravityRunOptions` and maps the
   permission vocabulary to `--dangerously-skip-permissions` or `--mode`
3. `antigravity.py` builds `agy ... --input-format stream-json
   --output-format stream-json` and writes one stream-json user message to
   stdin (default), or appends `--print=<prompt>` in argv mode
4. structured mode first checks the schema root is an object, then adds
   `--json-schema <compact-json>`
5. `transport.execute_command()` runs agy
6. `antigravity_parser.py` decodes the events into `AntigravityJsonPayload`
   (final `result`, events, and summed per-step usage) and parses the
   `AGY_ERROR` stderr record
7. the client raises for a non-zero exit, a non-`SUCCESS` result, any
   `finished_reason` (`print_timeout` as a timeout), a missing resumed
   conversation, auto-mode `denied_actions`, and, for resumed structured
   calls, `structured_output` without a completed `finish` step
8. structured output is validated locally; text calls return the result's
   `response` (or plain stdout in argv mode)

### Claude text mode

1. `ClaudeCodeClient.run_text()` applies optional provider and rate-limit retries
2. single-attempt text execution resolves `RunOptions`
3. `client._prepare()` builds `claude -p --output-format text`
4. `transport.execute_command()` runs the subprocess
5. `client` returns `TextRunResult` or raises a normalized exception

### Claude JSON mode

1. `ClaudeCodeClient.run_json()` applies optional provider and rate-limit retries
2. single-attempt JSON execution forces `--output-format json`
3. `transport.execute_command()` runs Claude
4. `parser.try_parse_json_payload()` parses the JSON envelope
5. `types.ClaudeJsonPayload` exposes normalized envelope fields
6. `client` returns `JsonRunResult` or raises a normalized exception

### Claude structured mode

1. The caller builds `StructuredSchema.from_dict(...)`
2. `schema.py` validates the JSON Schema locally with `jsonschema`
3. `ClaudeCodeClient.run_structured()` applies optional provider and rate-limit retries
4. single-attempt execution adds `--json-schema <schema.cli_json>`
5. Claude runs in JSON mode and returns the normal JSON envelope
6. `client` requires `payload.structured_output` to be present
7. `schema.py` validates the returned object against the same schema
8. `client` returns `StructuredRunResult`

### Codex text mode

1. `CodexClient.run_text()` applies optional provider and rate-limit retries
2. single-attempt text execution resolves `CodexRunOptions`
3. `codex.py` builds `codex exec`, using stdin by default
4. `transport.execute_command()` runs the subprocess
5. `client` returns `TextRunResult` or raises a normalized exception

By default, Codex runs with `--dangerously-bypass-approvals-and-sandbox`. Set
`CodexRunOptions(permission_mode="default")` to omit that flag.

### Codex JSON mode

1. `CodexClient.run_json()` applies optional provider and rate-limit retries
2. single-attempt JSON execution adds `--json`
3. `transport.execute_command()` runs Codex
4. `codex_parser.try_parse_codex_jsonl_payload()` parses JSONL events
5. `types.CodexJsonPayload` exposes thread ID, final message, usage, error, and
   raw events
6. `client` returns `CodexJsonRunResult` or raises a normalized exception

### Codex structured mode

1. The caller builds `StructuredSchema.from_dict(...)`
2. `schema.py` validates the JSON Schema locally with `jsonschema`
3. `CodexClient.run_structured()` writes the compact schema JSON to a temporary
   file
4. single-attempt execution adds `--output-schema <tempfile>`
5. Codex returns the final structured response as JSON
6. `codex.py` parses stdout as JSON
7. `schema.py` validates the returned object against the same schema
8. `client` deletes the temporary schema file and returns `CodexStructuredRunResult`

With `capture_usage=True`, execution also adds `--json` and a fresh
`--output-last-message` path. The final file supplies the validated object;
stdout remains JSONL, with best-effort usage/events in the payload. Provider
failures are classified independently from accounting. Both temporary files
are removed on success, failure, and timeout. `usage.py` converts provider
facts into disjoint token allocations without I/O or consumer context.

## Module Map

- `src/stan_ai_client/claude.py`: `ClaudeCodeClient`, Claude command
  preparation, execution, logging, and error normalization
- `src/stan_ai_client/grok.py`: `GrokClient`, Grok command preparation (using
  `-p`), execution, prompt file fallback, logging, and error normalization
- `src/stan_ai_client/grok_parser.py`: Grok JSON envelope parsing
- `src/stan_ai_client/antigravity.py`: `AntigravityClient`, `agy` command
  preparation, stream-json execution, approval/session/timeout guards,
  logging, and error normalization
- `src/stan_ai_client/antigravity_parser.py`: `agy` stream-json decoding,
  per-step usage sums, `AGY_ERROR` records, and stderr notices
- `src/stan_ai_client/client.py`: compatibility shim that re-exports
  `ClaudeCodeClient` for existing imports
- `src/stan_ai_client/codex.py`: `CodexClient`, Codex command preparation,
  execution, logging, structured schema-file handling, and error normalization
- `src/stan_ai_client/transport.py`: subprocess transport wrapper
- `src/stan_ai_client/parser.py`: Claude JSON envelope parsing helpers
- `src/stan_ai_client/codex_parser.py`: Codex JSONL parsing helpers
- `src/stan_ai_client/types.py`: request, retry-policy, payload, and result dataclasses
- `src/stan_ai_client/schema.py`: `StructuredSchema` and JSON Schema validation
- `src/stan_ai_client/exceptions.py`: provider-neutral and provider-specific
  exception hierarchy
- `src/stan_ai_client/_network.py`: provider-aware transport-connectivity
  detection over provider-declared error records and guarded process diagnostics
- `src/stan_ai_client/provider_errors.py`: Claude/Codex/Antigravity capacity
  and server-error classification over those trusted diagnostics, with retry
  hints
- `src/stan_ai_client/provider_retry.py`: `ProviderRetryPolicy`, backoff, the
  cancellable provider retry loop, and the failed-attempt table
- `src/stan_ai_client/_retry.py`: the rate-limit retry loop and its composition
  with provider retries
- `src/stan_ai_client/rate_limits.py`: rate-limit parsing helpers

## Logging

The clients use stdlib `logging`.

- `INFO`: run start and finish metadata
- `DEBUG`: redacted argv, payload metadata, and structured-output validation events
- `WARNING` / `ERROR`: protocol failures, process failures, network failures, rate limits,
  rate-limit retry waits, wait-budget refusals, capacity/server retry waits,
  missing executable, and timeouts

Prompt text is only logged when `log_prompts=True`.

Claude argv logs redact system prompts, settings, resumed session IDs, and inline
JSON schemas. Codex argv logs redact config override values, temporary schema
paths, and prompts passed through argv mode. Antigravity argv logs redact
`--print=` prompts, `--conversation` IDs, `--settings`, and inline JSON schemas.

## Database Schema

This project has no database, migrations, tables, or persistent application schema.

Full database schema:

- none
