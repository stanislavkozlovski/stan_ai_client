from __future__ import annotations

from typing import Any

import pytest

from stan_ai_client import (
    ClaudeSchemaValidationError,
    CodexSchemaValidationError,
    SchemaValidationError,
    StructuredSchema,
    StructuredSchemaValidationError,
)


def test_structured_schema_rejects_non_dict_input() -> None:
    with pytest.raises(StructuredSchemaValidationError) as excinfo:
        StructuredSchema.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]
    assert isinstance(excinfo.value, SchemaValidationError)
    assert isinstance(excinfo.value, ClaudeSchemaValidationError)
    assert isinstance(excinfo.value, CodexSchemaValidationError)


def test_structured_schema_rejects_invalid_json_schema() -> None:
    invalid_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": "summary",
    }

    with pytest.raises(StructuredSchemaValidationError):
        StructuredSchema.from_dict(invalid_schema)
