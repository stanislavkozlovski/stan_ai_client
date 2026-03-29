from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import NoReturn

import pytest

from stan_ai_client import __version__
from stan_ai_client._version import (
    UNKNOWN_VERSION,
    bump_patch_version,
    bump_patch_version_in_text,
    get_version,
    read_local_version,
)


def test_exported_version_is_non_empty() -> None:
    assert __version__


def test_get_version_falls_back_to_pyproject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "stan-ai-client"\nversion = "1.2.3"\n')

    monkeypatch.setattr("stan_ai_client._version.version", _raise_package_not_found)
    monkeypatch.setattr("stan_ai_client._version._PYPROJECT_PATH", pyproject)

    assert get_version() == "1.2.3"


def test_read_local_version_returns_unknown_when_missing(tmp_path: Path) -> None:
    assert read_local_version(tmp_path / "missing.toml") == UNKNOWN_VERSION


def test_bump_patch_version_in_text_updates_project_version() -> None:
    text = '[project]\nname = "stan-ai-client"\nversion = "0.1.0"\n'

    updated_text, new_version = bump_patch_version_in_text(text)

    assert new_version == "0.1.1"
    assert 'version = "0.1.1"' in updated_text


def test_bump_patch_version_rejects_non_semver() -> None:
    with pytest.raises(ValueError, match="simple X.Y.Z semver"):
        bump_patch_version("0.1.0.dev1")


def _raise_package_not_found(_: str) -> NoReturn:
    raise PackageNotFoundError("stan-ai-client")
