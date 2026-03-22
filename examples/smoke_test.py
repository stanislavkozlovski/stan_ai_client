from __future__ import annotations

from stan_ai_client import ClaudeCodeClient


def main() -> None:
    client = ClaudeCodeClient()
    text_result = client.run_text("Reply with the single word: ok")
    print("text:", text_result.text)

    json_result = client.run_json("Reply with the single word: ok")
    print("json:", json_result.payload.result)


if __name__ == "__main__":
    main()
