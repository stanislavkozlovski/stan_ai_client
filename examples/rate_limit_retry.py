from __future__ import annotations

from stan_ai_client import ClaudeCodeClient, ClaudeRateLimitError, RateLimitRetryPolicy


RATE_LIMIT_MAX = 5 * 60 * 60


def main() -> None:
    client = ClaudeCodeClient()
    try:
        result = client.run_json(
            "Summarize this repository in one paragraph.",
            rate_limit_policy=RateLimitRetryPolicy(
                max_wait_seconds=RATE_LIMIT_MAX,
                label="repo summary",
            ),
        )
    except ClaudeRateLimitError as exc:
        # With this policy, the client only raises here when the reset cannot
        # be parsed or the wait would exceed RATE_LIMIT_MAX.
        print(
            f"rate limit wait is above RATE_LIMIT_MAX={RATE_LIMIT_MAX} seconds:",
            exc.reset_at or exc.retry_after_seconds,
        )
        raise

    print(result.payload.result)


if __name__ == "__main__":
    main()
