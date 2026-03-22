from __future__ import annotations

import logging

from stan_ai_client import ClaudeCodeClient


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("stan_ai_client.demo")

    client = ClaudeCodeClient(
        logger=logger,
        log_prompts=False,
    )
    result = client.run_text("Reply with the single word: ok")
    print(result.text)


if __name__ == "__main__":
    main()
