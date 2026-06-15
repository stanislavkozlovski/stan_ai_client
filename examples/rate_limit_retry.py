from __future__ import annotations

from stan_ai_client import ClaudeCodeClient, ClaudeRateLimitError, RateLimitRetryPolicy


def main() -> None:
    client = ClaudeCodeClient()
    try:
        result = client.run_json(
            "Summarize this repository in one paragraph.",
            rate_limit_policy=RateLimitRetryPolicy(
                max_wait_seconds=5 * 60 * 60,
                label="repo summary",
            ),
        )
    except ClaudeRateLimitError as exc:
        print("rate limited until:", exc.reset_at or exc.retry_after_seconds)
        raise

    print(result.payload.result)


if __name__ == "__main__":
    main()
