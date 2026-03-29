from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bump the patch version in pyproject.toml and print the new version."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to the pyproject.toml file to update.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from stan_ai_client._version import bump_patch_version_in_pyproject

    new_version = bump_patch_version_in_pyproject(args.path)
    print(new_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
