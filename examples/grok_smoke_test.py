#!/usr/bin/env python3
"""Smoke test for GrokClient.

Run with: python examples/grok_smoke_test.py
Skips gracefully if grok is not available.
"""

from __future__ import annotations

import shutil

from stan_ai_client import GrokClient, GrokRunOptions, StructuredSchema


def main() -> None:
    if not shutil.which("grok"):
        print("grok not found in PATH, skipping Grok smoke test.")
        return

    client = GrokClient(
        default_timeout_seconds=60.0,
    )

    print("=== Grok text mode ===")
    res = client.run_text("Reply with the single word: OK")
    print("text:", res.text[:100])
    print("returncode:", res.returncode)

    print("\n=== Grok JSON mode ===")
    resj = client.run_json("Return a short greeting as JSON with key 'msg'.")
    print("payload keys:", list(resj.payload.extras.keys()) if resj.payload.extras else "basic")
    print("text:", resj.payload.text[:80] if resj.payload.text else None)

    print("\n=== Grok structured ===")
    schema = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        }
    )
    ress = client.run_structured(
        "Return an object with status set to 'green'.",
        schema=schema,
    )
    print("structured:", ress.structured_output)

    print("\n=== Grok session resume (simple) ===")
    sid = None
    r1 = client.run_json("Remember the number 4242. Reply only with STORED.")
    sid = r1.payload.session_id
    print("session1:", sid)
    if sid:
        r2 = client.run_json(
            "What number did I ask you to remember? Reply with only the digits.",
            options=GrokRunOptions(session_id=sid),
        )
        print("resume text:", r2.payload.text)

    print("\nGrok smoke test complete.")


if __name__ == "__main__":
    main()
