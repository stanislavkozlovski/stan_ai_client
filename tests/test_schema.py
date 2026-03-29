from __future__ import annotations

from typing import Any

import pytest

from stan_ai_client import ClaudeSchemaValidationError, StructuredSchema


def test_structured_schema_rejects_non_dict_input() -> None:
    with pytest.raises(ClaudeSchemaValidationError):
        StructuredSchema.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_structured_schema_rejects_invalid_json_schema() -> None:
    invalid_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": "summary",
    }

    with pytest.raises(ClaudeSchemaValidationError):
        StructuredSchema.from_dict(invalid_schema)
