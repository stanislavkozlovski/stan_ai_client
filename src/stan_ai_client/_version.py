from __future__ import annotations

import re
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGE_NAME = "stan-ai-client"
UNKNOWN_VERSION = "0.0.0"
_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
_PROJECT_VERSION_RE = re.compile(r'^(?P<prefix>\s*version\s*=\s*")(?P<version>[^"]+)(?P<suffix>".*)$')
_SEMVER_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


def get_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return read_local_version()


def read_local_version(path: Path | None = None) -> str:
    pyproject_path = path or _PYPROJECT_PATH

    try:
        with pyproject_path.open("rb") as handle:
            pyproject = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return UNKNOWN_VERSION

    project = pyproject.get("project")
    if not isinstance(project, dict):
        return UNKNOWN_VERSION

    raw_version = project.get("version")
    if isinstance(raw_version, str) and raw_version:
        return raw_version

    return UNKNOWN_VERSION


def bump_patch_version_in_text(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    in_project_section = False
    version_line_index: int | None = None
    new_version: str | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project_section = stripped == "[project]"
            continue

        if not in_project_section:
            continue

        match = _PROJECT_VERSION_RE.match(line)
        if match is None:
            continue

        if version_line_index is not None:
            raise ValueError("Found multiple version entries in the [project] section")

        version_line_index = index
        current_version = match.group("version")
        new_version = bump_patch_version(current_version)
        lines[index] = f'{match.group("prefix")}{new_version}{match.group("suffix")}\n'

    if version_line_index is None or new_version is None:
        raise ValueError("Could not find a version entry in the [project] section")

    return "".join(lines), new_version


def bump_patch_version_in_pyproject(path: Path) -> str:
    text = path.read_text()
    updated_text, new_version = bump_patch_version_in_text(text)
    path.write_text(updated_text)
    return new_version


def bump_patch_version(raw_version: str) -> str:
    match = _SEMVER_RE.fullmatch(raw_version)
    if match is None:
        raise ValueError(f"Version must be simple X.Y.Z semver, got {raw_version!r}")

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch")) + 1
    return f"{major}.{minor}.{patch}"
