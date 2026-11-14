#!/usr/bin/env python3
"""Smoke test for AntigravityClient against the local ``agy`` CLI.

Run with: python examples/antigravity_smoke_test.py [--usage-only] [--model M] [--effort E]
Skips gracefully if agy is not available. Every call runs in a fresh temporary
workspace with permission_mode="default", so the agent cannot run commands or
write files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from typing import Any

from stan_ai_client import (
    AntigravityClient,
    AntigravityRunOptions,
    StructuredSchema,
    __version__,
    normalize_ai_usage,
)


def usage_smoke(client: AntigravityClient) -> None:
    """A fresh and a resumed call; prints the counters normalization uses."""
    conversation_id = None
    for label in ("fresh", "resume"):
        result = client.run_json(
            "Do not use tools. Reply with the single word: ok",
            options=AntigravityRunOptions(session_id=conversation_id),
        )
        payload = result.payload
        if conversation_id is not None:
            assert payload.conversation_id == conversation_id
        conversation_id = payload.conversation_id
        facts = normalize_ai_usage("antigravity", payload)
        assert facts.tokens.total_tokens is not None, facts.diagnostics
        print(
            json.dumps(
                {
                    "call": label,
                    "conversation_id": conversation_id,
                    "scope": payload.usage_scope,
                    "reported_usage": payload.usage,
                    "invocation_usage": payload.invocation_usage,
                    "tokens": asdict(facts.tokens),
                    "diagnostics": facts.diagnostics,
                }
            ),
            flush=True,
        )


def full_smoke(client: AntigravityClient) -> None:
    text = client.run_text("Reply with the single word: ok")
    print("text (stdin):", text.text)

    argv_text = client.run_text(
        "Reply with the single word: ok", options=AntigravityRunOptions(input_mode="argv")
    )
    print("text (argv):", argv_text.text)

    first = client.run_json("Remember the number 4242. Reply only with STORED.")
    print("json:", (first.payload.response or "").strip(), first.payload.conversation_id)
    resumed = client.run_json(
        "What number did I ask you to remember? Reply with only the digits.",
        options=AntigravityRunOptions(session_id=first.payload.conversation_id),
    )
    print("resume:", (resumed.payload.response or "").strip(), resumed.payload.invocation_usage)

    schema: StructuredSchema[dict[str, Any]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["green"]}},
            "required": ["status"],
            "additionalProperties": False,
        }
    )
    structured = client.run_structured(
        "Return an object with status set to 'green'.", schema=schema
    )
    print("structured:", structured.structured_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage-only", action="store_true", help="Only run the usage smoke")
    parser.add_argument("--model", default="gemini-3.8-flash", help="agy model (base name)")
    parser.add_argument("--effort", default="low", help="effort level the model supports")
    args = parser.parse_args()

    if not shutil.which("agy"):
        print("agy not found in PATH, skipping Antigravity smoke test.")
        return
    print("client:", __version__)
    print("cli:", subprocess.check_output(["agy", "--version"], text=True).strip())

    with tempfile.TemporaryDirectory(prefix="stan-ai-client-agy-smoke-") as workspace:
        client = AntigravityClient(
            default_model=args.model,
            default_effort=args.effort,
            default_timeout_seconds=180.0,
            default_options=AntigravityRunOptions(cwd=workspace, permission_mode="default"),
        )
        if args.usage_only:
            usage_smoke(client)
        else:
            full_smoke(client)
    print("Antigravity smoke test complete.")


if __name__ == "__main__":
    main()
