from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .exceptions import ClaudeSchemaValidationError

TStructured = TypeVar("TStructured")


@dataclass(frozen=True)
class StructuredSchema(Generic[TStructured]):
    schema: dict[str, Any]
    cli_json: str
    _validator: Draft202012Validator = field(repr=False, compare=False)

    @classmethod
    def from_dict(
        cls: type["StructuredSchema[TStructured]"], schema: dict[str, Any]
    ) -> "StructuredSchema[TStructured]":
        if not isinstance(schema, dict):
            raise ClaudeSchemaValidationError(
                "Structured schema must be a dict-backed JSON Schema object"
            )

        schema_copy = deepcopy(schema)
        try:
            Draft202012Validator.check_schema(schema_copy)
            cli_json = json.dumps(schema_copy, separators=(",", ":"))
        except (SchemaError, TypeError, ValueError) as exc:
            raise ClaudeSchemaValidationError(f"Invalid structured schema: {exc}") from exc

        return cls(
            schema=schema_copy,
            cli_json=cli_json,
            _validator=Draft202012Validator(schema_copy),
        )

    def validate_response(self, structured_output: object) -> TStructured:
        self._validator.validate(structured_output)
        return cast(TStructured, structured_output)
