# RFC: First-Class Rate Limit Retry Policy

## Status

Implemented in branch `rfc-rate-limit-retry-policy`.

## Problem

`stan_ai_client` already treats Claude rate limits as structured failures:

- `ClaudeRateLimitError` is raised when Claude CLI output looks like a rate or usage limit.
- `ClaudeLimitError` exposes `retry_after_seconds` and `reset_at`.
- `parse_rate_limit_info()` parses common reset formats.

That is useful but still leaves application code with the operational burden: catch the exception, decide whether the reset is acceptable, log the wait, sleep, and retry. `justin` has already copied this missing behavior into a local `run_json_with_retries()` helper.

The library should provide an opt-in `rate_limit_policy` that handles rate-limit waits inside the client while keeping the caller in control of the only policy choice that matters for real Claude limits: "How long am I willing to wait?"

## Core Design

Rate limits usually last minutes or hours, not a few short retry attempts. A policy with `max_attempts`, manual fallback backoff, and separate buffer knobs makes the caller think in the wrong unit.

The first-class API should be budget-based:

```python
from stan_ai_client import ClaudeCodeClient, RateLimitRetryPolicy

client = ClaudeCodeClient()

rate_limit_policy = RateLimitRetryPolicy(
    max_wait_seconds=5 * 60 * 60,
    label="chapters pass1 chunk=12",
)

result = client.run_json(
    prompt,
    options=options,
    rate_limit_policy=rate_limit_policy,
)
```

Meaning:

- If Claude says the reset is within 5 hours, log it, sleep, and retry.
- If Claude says the reset is longer than 5 hours, re-raise the same `ClaudeRateLimitError`.
- If Claude does not provide parseable retry/reset metadata, re-raise the same `ClaudeRateLimitError`.
- If repeated rate limits cumulatively exceed 5 hours of sleep, re-raise the same `ClaudeRateLimitError`.

`rate_limit_policy=None` means no retry. That remains the default.

## Goals

- Make rate-limit retry behavior first-class in `stan_ai_client`.
- Keep default behavior backward-compatible: without `rate_limit_policy`, rate limits raise immediately.
- Make the caller think in one clear unit: maximum total wait.
- Use Claude's parsed reset metadata rather than arbitrary short retry loops.
- Log each wait in a consistent, useful format.
- Preserve the existing exception hierarchy and catch behavior.
- Support text, JSON, and structured modes with the same policy.
- Remove duplicated retry loops from application code such as `justin`.

## Non-Goals

- Do not retry non-rate-limit failures.
- Do not retry when no reset/retry metadata can be parsed.
- Do not add async support.
- Do not add streaming support.
- Do not call Anthropic APIs directly.
- Do not build a scheduler, queue, or persistence layer.
- Do not make long sleeps happen by default.

## Proposed API

Add `RateLimitRetryPolicy` and expose it from the package root.

```python
@dataclass(frozen=True)
class RateLimitRetryPolicy:
    """Controls opt-in retry behavior for Claude rate-limit responses.

    The policy is intentionally budget-based. Claude rate limits usually reset
    on a concrete schedule, so callers should decide how long the operation is
    allowed to wait rather than how many short retries to attempt.

    Args:
        max_wait_seconds: Maximum accumulated sleep time allowed for this call.
            `None` means wait for any parseable Claude reset.
        label: Optional human-readable context included in retry logs.
    """

    max_wait_seconds: float | None
    label: str | None = None
```

Field semantics:

- `max_wait_seconds`: Maximum accumulated sleep time allowed for this call. `None` means wait as long as Claude asks, as long as each rate-limit response includes parseable retry/reset metadata.
- `label`: Optional human-readable context included in retry logs. It should identify the operation, not include prompt text.

Validation:

- `max_wait_seconds` must be `None` or `>= 0`.

Recommended constructors:

```python
RateLimitRetryPolicy(max_wait_seconds=5 * 60 * 60)
RateLimitRetryPolicy(max_wait_seconds=None)  # explicit unbounded wait
RateLimitRetryPolicy(max_wait_seconds=3600, label="chapters pass2")
```

Optional convenience constructors:

```python
RateLimitRetryPolicy.max_wait(hours=5)
RateLimitRetryPolicy.unbounded()
RateLimitRetryPolicy.no_wait()
```

These are convenience only. The minimal API is the dataclass field.

## Public Method Signatures

Add one optional `rate_limit_policy` keyword argument:

```python
def run_text(
    self,
    prompt: str,
    *,
    options: RunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> TextRunResult: ...

def run_json(
    self,
    prompt: str,
    *,
    options: RunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> JsonRunResult: ...

def run_structured(
    self,
    prompt: str,
    *,
    schema: StructuredSchema[TStructured],
    options: RunOptions | None = None,
    rate_limit_policy: RateLimitRetryPolicy | None = None,
) -> StructuredRunResult[TStructured]: ...
```

Rationale:

- `rate_limit_policy` is explicit at call sites.
- The policy owns both retry behavior and retry log context, which prevents public method parameter sprawl.
- Keeping the policy per-call lets batch jobs and user-facing bots choose different wait budgets.

Rejected alternative:

```python
client.run_json_with_retries(...)
```

This duplicates the public method surface for text, JSON, and structured modes. A policy argument keeps the behavior attached to the existing operation.

## Retry Algorithm

Pseudocode:

```python
def run_with_rate_limit_policy(operation, *, rate_limit_policy, logger):
    if rate_limit_policy is None:
        return operation()

    label = rate_limit_policy.label
    total_wait_seconds = 0.0
    attempts = 0

    while True:
        attempts += 1

        try:
            return operation()
        except ClaudeRateLimitError as exc:
            wait_seconds = exc.retry_after_seconds

            if wait_seconds is None:
                logger.warning(
                    "Claude rate limited but no retry metadata was parsed "
                    "attempt=%d total_wait_seconds=%.1f max_wait_seconds=%s reset_at=%s label=%s",
                    attempts,
                    total_wait_seconds,
                    rate_limit_policy.max_wait_seconds,
                    exc.reset_at,
                    label,
                )
                raise

            if rate_limit_policy.max_wait_seconds is not None:
                remaining = rate_limit_policy.max_wait_seconds - total_wait_seconds
                if wait_seconds > remaining:
                    logger.warning(
                        "Claude rate limit exceeds wait budget "
                        "attempt=%d wait_seconds=%.1f remaining_wait_seconds=%.1f "
                        "total_wait_seconds=%.1f max_wait_seconds=%.1f reset_at=%s label=%s",
                        attempts,
                        wait_seconds,
                        remaining,
                        total_wait_seconds,
                        rate_limit_policy.max_wait_seconds,
                        exc.reset_at,
                        label,
                    )
                    raise

            total_wait_seconds += wait_seconds

            logger.warning(
                "Claude rate limited; retrying after reset "
                "attempt=%d wait_seconds=%.1f total_wait_seconds=%.1f "
                "max_wait_seconds=%s retry_after_seconds=%s reset_at=%s label=%s",
                attempts,
                wait_seconds,
                total_wait_seconds,
                rate_limit_policy.max_wait_seconds,
                exc.retry_after_seconds,
                exc.reset_at,
                label,
            )

            sleep(wait_seconds)
```

There is no separate `max_attempts`.

Repeated attempts are naturally bounded by `max_wait_seconds`. If Claude keeps returning short parseable reset windows and the caller configured `max_wait_seconds=None`, the library may wait indefinitely. That is acceptable only because the caller opted into an unbounded policy.

## Error Semantics

The retry wrapper should always re-raise the same exception type that single-attempt execution would raise.

Without a policy:

- A detected rate limit raises `ClaudeRateLimitError` immediately.

With a policy:

- If the retry eventually succeeds, return the normal result type.
- If no retry/reset metadata is parsed, re-raise `ClaudeRateLimitError`.
- If the next wait would exceed `max_wait_seconds`, re-raise `ClaudeRateLimitError`.
- If a later retry would exceed the remaining total wait budget, re-raise `ClaudeRateLimitError`.
- Do not wrap exhaustion in `ClaudeCodeError`.
- Do not introduce `ClaudeRateLimitRetryExhaustedError` in the first implementation.

Caller code should remain simple:

```python
try:
    result = client.run_json(
        prompt,
        rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=5 * 60 * 60),
    )
except ClaudeRateLimitError as exc:
    print(exc.retry_after_seconds)
    print(exc.reset_at)
```

## `retry_after_seconds` Semantics

`retry_after_seconds` should mean "the parsed amount of time until Claude says this limit resets."

It should not include hidden policy buffers.

Current code should be cleaned up as part of implementation:

- `retry after 3600` should produce `retry_after_seconds == 3600`, not `3630`.
- `resets in 2 hours 30 minutes` should produce exactly `9000`, not `9060`.
- Absolute reset parsing should continue to calculate seconds until the parsed reset time.

If we ever want a safety buffer, it should be explicit policy behavior and should not mutate `RateLimitInfo.retry_after_seconds`. This RFC does not propose a buffer knob for the first implementation.

## Logging

Each retry sleep should emit one `WARNING`.

Required fields:

- attempt number
- wait seconds
- total accumulated wait seconds
- max wait seconds
- `retry_after_seconds`
- `reset_at`
- operation label, if supplied

When the policy refuses to wait, log once before re-raising:

- no parsed retry metadata
- wait would exceed remaining wait budget

Example retry log:

```text
Claude rate limited; retrying after reset attempt=1 wait_seconds=14400.0 total_wait_seconds=14400.0 max_wait_seconds=18000 retry_after_seconds=14400 reset_at=2026-06-15T14:00:00+02:00 label=chapters pass1 chunk=12
```

Example over-budget log:

```text
Claude rate limit exceeds wait budget attempt=1 wait_seconds=21600.0 remaining_wait_seconds=18000.0 total_wait_seconds=0.0 max_wait_seconds=18000.0 reset_at=2026-06-15T16:00:00+02:00 label=chapters pass1 chunk=12
```

The logger should be the existing `ClaudeCodeClient.logger`. Prompt text must not be logged unless existing `log_prompts=True` behavior already permits it.

## Justin Before And After

Today, Justin owns this local loop:

```python
for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 2):
    try:
        return client.run_json(prompt, options=options)
    except ClaudeRateLimitError as exc:
        if exc.retry_after_seconds is not None:
            wait_seconds = exc.retry_after_seconds + RATE_LIMIT_BUFFER_SECONDS
        else:
            wait_seconds = fallback_backoff(attempt)

        if total_wait_seconds + wait_seconds > max_total_wait_seconds:
            raise ClaudeCodeError("rate-limit wait budget exceeded") from exc

        total_wait_seconds += wait_seconds
        logger.warning("Claude rate limited ...")
        time.sleep(wait_seconds)
```

With this RFC:

```python
rate_limit_policy = RateLimitRetryPolicy(
    max_wait_seconds=args.max_total_wait_seconds,
    label=f"chapters pass1 chunk={chunk_id}",
)

result = client.run_json(
    prompt,
    options=options,
    rate_limit_policy=rate_limit_policy,
)
```

Justin keeps the important decision, the maximum total wait, and removes the generic retry loop.

If Justin wants its current "wait indefinitely by default" CLI behavior:

```python
rate_limit_policy = RateLimitRetryPolicy(max_wait_seconds=None)
```

If the user passes `--max-total-wait-seconds 3600`:

```python
rate_limit_policy = RateLimitRetryPolicy(max_wait_seconds=3600)
```

## User-Facing Fail-Fast Example

Some user-facing applications should not sleep at all. They should let the typed error surface and turn it into a clear user response.

```python
try:
    result = client.run_json(
        prompt,
        options=options,
    )
except ClaudeRateLimitError as exc:
    retry_hint = exc.reset_at or exc.retry_after_seconds
    return f"Claude is rate limited. Try again later. Reset: {retry_hint}"
```

For this case, callers should omit `rate_limit_policy`. The library raises `ClaudeRateLimitError` immediately, preserving the parsed reset metadata for the application response.

## Implementation Plan

### 1. Add Policy Type

Edit `src/stan_ai_client/types.py`:

- Add frozen `RateLimitRetryPolicy`.
- Validate `max_wait_seconds`.
- Keep it separate from `RunOptions`, because retry policy is library control flow around Claude execution, not a Claude CLI flag.

### 2. Export Policy

Edit `src/stan_ai_client/__init__.py`:

- Import `RateLimitRetryPolicy`.
- Add it to `__all__`.

### 3. Normalize Rate-Limit Parsing

Edit `src/stan_ai_client/rate_limits.py`:

- Remove hidden additive buffers from parsed durations.
- `_parse_retry_after_seconds("retry after 3600")` should return `3600`.
- `_parse_relative_reset("resets in 2 hours 30 minutes")` should return exactly `now + 2h30m`.
- Keep `reset_at` and `retry_after_seconds` internally consistent.

Update `tests/test_rate_limits.py` accordingly.

### 4. Add Retry Wrapper Internals

Edit `src/stan_ai_client/client.py`:

- Import `RateLimitRetryPolicy`.
- Add `rate_limit_policy` to public run methods.
- Move each existing method body into a private single-attempt helper, or add a private wrapper that accepts a callable.
- Add `_run_with_rate_limit_policy(...)`.
- Add `_sleep(seconds: float)` as a test seam around `time.sleep()`.
- Catch only `ClaudeRateLimitError`.
- Re-raise the caught `ClaudeRateLimitError` directly for all refusal/exhaustion cases.

Suggested internal shape:

```python
def run_json(..., rate_limit_policy=None):
    return self._run_with_rate_limit_policy(
        lambda: self._run_json_once(prompt, options=options),
        rate_limit_policy=rate_limit_policy,
    )
```

### 5. Preserve Error Construction

No new exception type is required.

`ClaudeRateLimitError` should continue to be built in `_build_process_error()`. The retry wrapper should not manufacture a different error when a policy refuses to wait.

### 6. Tests

Add or extend tests:

- No policy: a rate limit raises immediately.
- Policy succeeds after one parsed wait and retry.
- Policy uses parsed `retry_after_seconds` exactly.
- Policy refuses to retry when `retry_after_seconds is None`.
- Policy refuses to retry when wait exceeds `max_wait_seconds`.
- Policy refuses to retry when cumulative waits exceed `max_wait_seconds`.
- Policy with `max_wait_seconds=0` only retries zero-second waits.
- Policy with `max_wait_seconds=None` allows repeated parseable waits.
- Text mode retries.
- JSON mode retries.
- Structured mode retries.
- Retry logs include wait and reset metadata.
- Refusal logs include the reason.
- Policy validation rejects invalid values.

Likely files:

- `tests/test_rate_limits.py`
- `tests/test_client.py`
- possible new `tests/test_rate_limit_retries.py`

### 7. Docs

Edit `README.md`:

- Replace the manual retry example with `RateLimitRetryPolicy(max_wait_seconds=...)`.
- Update the public surface list.
- Remove or revise "no built-in retry loop".

Edit `DOCS.md`:

- Add `RateLimitRetryPolicy` reference section.
- Document method parameters.
- Document error semantics.
- Document exact `retry_after_seconds` semantics.

Edit `ARCHITECTURE.md`:

- Note that client execution has an optional reset-aware rate-limit retry wrapper around single-attempt Claude calls.

### 8. Examples

Optionally add:

- `examples/rate_limit_retry.py`

Keep it focused on configuring a maximum wait budget.

## Backward Compatibility

This is backward-compatible if:

- `rate_limit_policy` defaults to `None`.
- Existing exceptions keep their current class hierarchy and fields.
- Existing run methods still raise immediately without a policy.
- Existing return types remain unchanged.

The only behavior change outside explicit retry policy is the cleanup of `retry_after_seconds` values to remove hidden buffers. That is technically observable, but it makes the field match its name and is worth doing before the retry policy depends on it.

## Open Questions

1. Should `max_wait_seconds=None` be allowed?

   Recommendation: yes. It is useful for batch jobs that already run unattended. The caller must explicitly opt in.

2. Should `max_wait_seconds=0` mean "do not wait"?

   Recommendation: yes. This keeps the model simple: the policy can only retry when the computed wait fits the remaining budget.

3. Should unknown reset metadata use fallback backoff?

   Recommendation: no for the first implementation. If Claude does not tell us when the limit resets, the library should not guess. Re-raise `ClaudeRateLimitError`.

4. Should exhausted retries raise a new error type?

   Recommendation: no. Re-raise `ClaudeRateLimitError` so callers have one limit error to catch.

5. Should policy live in `RunOptions`?

   Recommendation: no. `RunOptions` maps mostly to Claude CLI execution options. Retry policy is client control flow around execution.

6. Should the client have a default policy in `__init__`?

   Recommendation: not initially. Per-call policy is clearer and avoids accidental long sleeps.

## Acceptance Criteria

- Callers can pass `rate_limit_policy=RateLimitRetryPolicy(max_wait_seconds=...)` to `run_text()`, `run_json()`, and `run_structured()`.
- Without a policy, current immediate raise behavior is unchanged.
- Rate-limit retries sleep exactly until the parsed retry/reset duration.
- Missing retry/reset metadata is not retried.
- Waits longer than the caller's remaining wait budget are not retried.
- Retry attempts are logged once per sleep.
- Refused retries are logged once before re-raising.
- All refusal/exhaustion cases re-raise `ClaudeRateLimitError`.
- `retry_after_seconds` no longer includes hidden additive buffers.
- Justin's local retry helper can be replaced with policy configuration only.
