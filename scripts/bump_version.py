#!/usr/bin/env python3
"""Bump and synchronize the app version across project metadata files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = ROOT / "core" / "constants.py"
PACKAGE_PATH = ROOT / "package.json"
PACKAGE_LOCK_PATH = ROOT / "package-lock.json"
APP_VERSION_PATTERN = re.compile(r'^(APP_VERSION\s*=\s*)"([^"]+)"', re.MULTILINE)
SEMVER_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class VersionState:
    constants: str
    package: str
    package_lock: str
    package_lock_root: str

    def unique(self) -> set[str]:
        return {
            self.constants,
            self.package,
            self.package_lock,
            self.package_lock_root,
        }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_constants_version() -> str:
    constants_text = CONSTANTS_PATH.read_text(encoding="utf-8")
    match = APP_VERSION_PATTERN.search(constants_text)
    if not match:
        raise SystemExit(f"Could not find APP_VERSION in {CONSTANTS_PATH}")
    return match.group(2)


def read_version_state() -> VersionState:
    package = read_json(PACKAGE_PATH)
    package_lock = read_json(PACKAGE_LOCK_PATH)
    root_package = package_lock.get("packages", {}).get("", {})

    try:
        return VersionState(
            constants=read_constants_version(),
            package=package["version"],
            package_lock=package_lock["version"],
            package_lock_root=root_package["version"],
        )
    except KeyError as exc:
        raise SystemExit(f"Missing version field in package metadata: {exc}") from exc


def parse_semver(version: str) -> tuple[int, int, int]:
    match = SEMVER_PATTERN.fullmatch(version)
    if not match:
        raise SystemExit(f"Only numeric X.Y.Z versions are supported, got {version!r}")
    return tuple(int(part) for part in match.groups())


def bump_version(version: str, part: str) -> str:
    major, minor, patch = parse_semver(version)

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise SystemExit(f"Unsupported bump part: {part}")

    return f"{major}.{minor}.{patch}"


def write_constants_version(version: str) -> None:
    constants_text = CONSTANTS_PATH.read_text(encoding="utf-8")
    updated_text, replacements = APP_VERSION_PATTERN.subn(rf'\1"{version}"', constants_text, count=1)
    if replacements != 1:
        raise SystemExit(f"Could not update APP_VERSION in {CONSTANTS_PATH}")
    CONSTANTS_PATH.write_text(updated_text, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_package_versions(version: str) -> None:
    package = read_json(PACKAGE_PATH)
    package["version"] = version
    write_json(PACKAGE_PATH, package)

    package_lock = read_json(PACKAGE_LOCK_PATH)
    package_lock["version"] = version
    package_lock["packages"][""]["version"] = version
    write_json(PACKAGE_LOCK_PATH, package_lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "part",
        nargs="?",
        default="patch",
        choices=("major", "minor", "patch"),
        help="Semantic version part to bump. Defaults to patch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the next version without modifying files.",
    )
    args = parser.parse_args()

    version_state = read_version_state()
    versions = version_state.unique()
    if len(versions) != 1:
        formatted = ", ".join(f"{field}={value}" for field, value in vars(version_state).items())
        raise SystemExit(f"Version fields are out of sync: {formatted}")

    current_version = next(iter(versions))
    next_version = bump_version(current_version, args.part)

    if args.dry_run:
        print(f"{current_version} -> {next_version}")
        return 0

    write_constants_version(next_version)
    write_package_versions(next_version)
    print(f"Bumped version: {current_version} -> {next_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
