from __future__ import annotations

from pathlib import Path

from stan_ai_client import ClaudeCodeClient, RunOptions


def main() -> None:
    article_dir = Path(".")
    prompt = "Output YAML tags for this article."
    client = ClaudeCodeClient()
    result = client.run_text(
        prompt,
        options=RunOptions(
            cwd=article_dir,
            input_mode="stdin",
        ),
    )
    print(result.text)


if __name__ == "__main__":
    main()

