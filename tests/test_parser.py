from __future__ import annotations

from stan_ai_client.parser import parse_json_payload, try_parse_json_payload


def test_parse_json_payload_preserves_extras() -> None:
    payload = parse_json_payload(
        '{"result":"ok","total_cost_usd":0.2,"modelUsage":{"a":{"tokens":1}},"extra_field":1}'
    )

    assert payload.result == "ok"
    assert payload.total_cost_usd == 0.2
    assert payload.model_usage == {"a": {"tokens": 1}}
    assert payload.extras == {"extra_field": 1}


def test_parse_json_payload_extracts_structured_output() -> None:
    payload = parse_json_payload(
        '{"result":"ok","structured_output":{"summary":"brief"},"extra_field":1}'
    )

    assert payload.structured_output == {"summary": "brief"}
    assert payload.has_structured_output is True
    assert payload.extras == {"extra_field": 1}


def test_try_parse_json_payload_returns_none_for_invalid_json() -> None:
    assert try_parse_json_payload("not json") is None
