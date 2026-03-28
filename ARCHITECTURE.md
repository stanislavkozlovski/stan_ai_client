# stan_ai_client Architecture

## Overview

`stan_ai_client` is a small synchronous Python wrapper around the local `claude` executable.

The library exposes three execution modes:

- `run_text()` for plain text
- `run_json()` for the raw Claude JSON envelope
- `run_structured()` for schema-guided structured output on top of that JSON envelope

The package does not call Anthropic APIs directly. It shells out to the local Claude Code CLI, captures stdout and stderr, then normalizes the response into typed results and library-specific exceptions.

## Execution Flow

### Text mode

1. `ClaudeCodeClient.run_text()` resolves `RunOptions`
2. `client._prepare()` builds argv and the optional stdin payload
3. `transport.execute_command()` runs the subprocess
4. `client` returns `TextRunResult` or raises a normalized exception

### JSON mode

1. `ClaudeCodeClient.run_json()` forces `--output-format json`
2. `transport.execute_command()` runs Claude
3. `parser.try_parse_json_payload()` parses the JSON envelope
4. `types.ClaudeJsonPayload` exposes the normalized envelope fields
5. `client` returns `JsonRunResult` or raises a normalized exception

### Structured mode

1. The caller builds `StructuredSchema.from_dict(...)`
2. `schema.py` validates the JSON Schema locally with `jsonschema`
3. `ClaudeCodeClient.run_structured()` adds `--json-schema <schema.cli_json>`
4. Claude runs in JSON mode and returns the normal JSON envelope
5. `client` requires `payload.structured_output` to be present
6. `schema.py` validates the returned `structured_output` against the same schema
7. `client` returns `StructuredRunResult`

Structured mode preserves the full Claude envelope through `result.payload`, including session ID, cost, usage, duration, and raw output.

## Module Map

- `src/stan_ai_client/client.py`: public client, command preparation, execution, logging, and error normalization
- `src/stan_ai_client/transport.py`: subprocess transport wrapper
- `src/stan_ai_client/parser.py`: JSON payload parsing helpers
- `src/stan_ai_client/types.py`: request, payload, and result dataclasses
- `src/stan_ai_client/schema.py`: `StructuredSchema` and JSON Schema validation
- `src/stan_ai_client/exceptions.py`: library exception hierarchy
- `src/stan_ai_client/rate_limits.py`: rate-limit parsing helpers

## Logging

The client uses stdlib `logging`.

- `INFO`: run start and finish metadata
- `DEBUG`: redacted argv, payload metadata, and structured-output validation events
- `WARNING` / `ERROR`: protocol failures, process failures, rate limits, missing executable, and timeouts

Prompt text is only logged when `log_prompts=True`.

## Database Schema

This project has no database, migrations, tables, or persistent application schema.

Full database schema:

- none
