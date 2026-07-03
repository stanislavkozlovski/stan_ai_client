# stan_ai_client Architecture

## Overview

`stan_ai_client` is a small synchronous Python wrapper around local AI coding
CLI executables.

It currently supports three backend clients:

- `ClaudeCodeClient` shells out to `claude -p`.
- `CodexClient` shells out to `codex exec`.
- `GrokClient` shells out to `grok -p` (with transparent prompt handling).

The package does not call Anthropic or OpenAI APIs directly. Authentication,
account selection, and provider configuration stay in the underlying CLI. The
library builds commands, captures stdout and stderr, parses supported machine
output, validates structured responses locally, and normalizes failures into
typed exceptions.

Both clients expose:

- `run_text()` for plain text
- `run_json()` for machine-readable output
- `run_structured()` for schema-guided structured output

All execution modes can optionally run through a `RateLimitRetryPolicy`. The
policy is a synchronous wrapper around one-attempt execution: it catches the
client's typed rate-limit exception, sleeps only when parsed reset metadata fits
the caller's wait budget, logs the wait, and retries the same operation.

## Execution Flow

### Grok text / JSON / structured mode

Similar to Claude but using `grok -p --output-format plain|json [--json-schema ...]`.
Prompts are passed as argv value (or `--prompt-file` for long prompts).
Session resume via `--resume` / `--continue` / `--session-id`.
The returned `GrokJsonPayload` is intentionally minimal (text + ids + optional
structuredOutput). Duration is measured client-side.

### Claude text mode

1. `ClaudeCodeClient.run_text()` applies an optional `RateLimitRetryPolicy`
2. single-attempt text execution resolves `RunOptions`
3. `client._prepare()` builds `claude -p --output-format text`
4. `transport.execute_command()` runs the subprocess
5. `client` returns `TextRunResult` or raises a normalized exception

### Claude JSON mode

1. `ClaudeCodeClient.run_json()` applies an optional `RateLimitRetryPolicy`
2. single-attempt JSON execution forces `--output-format json`
3. `transport.execute_command()` runs Claude
4. `parser.try_parse_json_payload()` parses the JSON envelope
5. `types.ClaudeJsonPayload` exposes normalized envelope fields
6. `client` returns `JsonRunResult` or raises a normalized exception

### Claude structured mode

1. The caller builds `StructuredSchema.from_dict(...)`
2. `schema.py` validates the JSON Schema locally with `jsonschema`
3. `ClaudeCodeClient.run_structured()` applies an optional `RateLimitRetryPolicy`
4. single-attempt execution adds `--json-schema <schema.cli_json>`
5. Claude runs in JSON mode and returns the normal JSON envelope
6. `client` requires `payload.structured_output` to be present
7. `schema.py` validates the returned object against the same schema
8. `client` returns `StructuredRunResult`

### Codex text mode

1. `CodexClient.run_text()` applies an optional `RateLimitRetryPolicy`
2. single-attempt text execution resolves `CodexRunOptions`
3. `codex.py` builds `codex exec`, using stdin by default
4. `transport.execute_command()` runs the subprocess
5. `client` returns `TextRunResult` or raises a normalized exception

By default, Codex runs with `--dangerously-bypass-approvals-and-sandbox`. Set
`CodexRunOptions(permission_mode="default")` to omit that flag.

### Codex JSON mode

1. `CodexClient.run_json()` applies an optional `RateLimitRetryPolicy`
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

## Module Map

- `src/stan_ai_client/claude.py`: `ClaudeCodeClient`, Claude command
  preparation, execution, logging, and error normalization
- `src/stan_ai_client/grok.py`: `GrokClient`, Grok command preparation (using
  `-p`), execution, prompt file fallback, logging, and error normalization
- `src/stan_ai_client/grok_parser.py`: Grok JSON envelope parsing
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
- `src/stan_ai_client/rate_limits.py`: rate-limit parsing helpers

## Logging

The clients use stdlib `logging`.

- `INFO`: run start and finish metadata
- `DEBUG`: redacted argv, payload metadata, and structured-output validation events
- `WARNING` / `ERROR`: protocol failures, process failures, rate limits,
  rate-limit retry waits, wait-budget refusals, missing executable, and timeouts

Prompt text is only logged when `log_prompts=True`.

Claude argv logs redact system prompts, settings, resumed session IDs, and inline
JSON schemas. Codex argv logs redact config override values, temporary schema
paths, and prompts passed through argv mode.

## Database Schema

This project has no database, migrations, tables, or persistent application schema.

Full database schema:

- none
