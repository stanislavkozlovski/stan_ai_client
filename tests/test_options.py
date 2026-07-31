from __future__ import annotations

from stan_ai_client._options import first_set, first_set_or


def test_first_set_prefers_override() -> None:
    assert first_set("override", "fallback") == "override"


def test_first_set_uses_fallback_when_override_is_none() -> None:
    assert first_set(None, "fallback") == "fallback"


def test_first_set_returns_none_when_both_unset() -> None:
    assert first_set(None, None) is None


def test_first_set_keeps_falsy_override() -> None:
    # An explicit falsy value is a real choice and must win over the fallback;
    # this is the property every option field depends on for override-vs-default
    # resolution, so it lives in one place instead of 30+ hand-written checks.
    assert first_set(False, True) is False
    assert first_set((), ("x",)) == ()
    assert first_set(0, 5) == 0


def test_first_set_or_falls_back_to_default_when_all_unset() -> None:
    assert first_set_or(None, None, default="client-default") == "client-default"


def test_first_set_or_prefers_first_set_value_over_default() -> None:
    assert first_set_or(None, "from-default-options", default="client-default") == (
        "from-default-options"
    )
    assert first_set_or("from-call", "from-default-options", default="client-default") == (
        "from-call"
    )


def test_first_set_or_keeps_falsy_resolved_value() -> None:
    assert first_set_or(False, None, default=True) is False
    assert first_set_or(0, None, default=120) == 0
