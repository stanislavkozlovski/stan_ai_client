from __future__ import annotations

from stan_ai_client import CodexClient


def main() -> None:
    client = CodexClient()
    text_result = client.run_text("Reply with the single word: ok")
    print("text:", text_result.text)

    json_result = client.run_json("Reply with the single word: ok")
    print("json:", json_result.payload.result)


if __name__ == "__main__":
    main()
