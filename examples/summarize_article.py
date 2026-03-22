from __future__ import annotations

from pathlib import Path

from stan_ai_client import ClaudeCodeClient, RunOptions


def main() -> None:
    article_dir = Path(".")
    prompt = "Summarize this article in one paragraph."
    client = ClaudeCodeClient(
        default_model="claude-opus-4-6",
        default_effort="max",
        default_timeout_seconds=180,
    )
    result = client.run_json(
        prompt,
        options=RunOptions(
            cwd=article_dir,
            allowed_tools=("Read", "Glob", "Grep", "Bash"),
        ),
    )
    print(result.payload.result)


if __name__ == "__main__":
    main()

