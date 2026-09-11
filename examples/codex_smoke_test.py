from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict

from stan_ai_client import (
    CodexClient,
    CodexRunOptions,
    StructuredSchema,
    __version__,
    normalize_ai_usage,
)


def usage_smoke() -> None:
    """Two tiny real calls; prints versions and provider counters for inspection.

    Session identity and final output are assertions. Counter scope must also
    be checked against the supported CLI implementation, not inferred merely
    because the second count happens to be larger or smaller.
    """
    print("client:", __version__, flush=True)
    print(
        "cli:",
        subprocess.check_output(["codex", "--version"], text=True).strip(),
        flush=True,
    )
    schema: StructuredSchema[dict[str, str]] = StructuredSchema.from_dict(
        {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["ok"]}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    with tempfile.TemporaryDirectory(prefix="stan-ai-client-smoke-") as directory:
        client = CodexClient(
            default_options=CodexRunOptions(
                cwd=directory,
                ignore_user_config=True,
                ignore_rules=True,
                skip_git_repo_check=True,
                permission_mode="default",
            )
        )
        session_id = None
        previous_snapshot = None
        for label in ("fresh", "resume"):
            result = client.run_structured(
                'Do not use tools. Reply with the JSON object {"answer":"ok"}.',
                schema=schema,
                capture_usage=True,
                options=CodexRunOptions(session_id=session_id),
            )
            assert result.structured_output == {"answer": "ok"}
            assert result.payload.thread_id
            if session_id is not None:
                assert result.payload.thread_id == session_id
            session_id = result.payload.thread_id
            facts = normalize_ai_usage(
                "codex", result.payload, previous_snapshot=previous_snapshot
            )
            assert facts.tokens.total_tokens is not None, facts.diagnostics
            print(
                json.dumps(
                    {
                        "call": label,
                        "session_id": session_id,
                        "scope": result.payload.usage_scope,
                        "usage": result.payload.usage,
                        "tokens": asdict(facts.tokens),
                        "diagnostics": facts.diagnostics,
                    }
                ),
                flush=True,
            )
            previous_snapshot = result.payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usage-only",
        action="store_true",
        help="Only run structured fresh/resume usage smoke",
    )
    if parser.parse_args().usage_only:
        usage_smoke()
        return
    client = CodexClient()
    text_result = client.run_text("Reply with the single word: ok")
    print("text:", text_result.text)

    json_result = client.run_json("Reply with the single word: ok")
    print("json:", json_result.payload.result)


if __name__ == "__main__":
    main()
