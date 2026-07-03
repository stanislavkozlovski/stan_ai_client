from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def first_set(override: _T, fallback: _T) -> _T:
    """Return ``override`` when it is set, otherwise ``fallback``.

    This is the single definition of how a run option is resolved from its
    layered sources: a per-call override, then the client's ``default_options``.
    Routing every field through this helper keeps the resolution rule uniform so
    a field cannot silently stop honoring ``default_options`` just because it was
    typed as non-optional.

    ``override`` and ``fallback`` share a declared type that already includes
    ``None`` for optional fields, so the resolved value stays optional exactly
    when the field is.
    """

    return override if override is not None else fallback


def first_set_or(override: _T | None, fallback: _T | None, *, default: _T) -> _T:
    """Resolve ``override`` then ``fallback``, then a guaranteed ``default``.

    Use for options that must always resolve to a concrete value (model, effort,
    timeouts, permission mode, booleans) where the final fallback is a
    client-level default that is always present.
    """

    resolved = override if override is not None else fallback
    return default if resolved is None else resolved
