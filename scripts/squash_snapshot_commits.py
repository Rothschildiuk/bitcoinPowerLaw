#!/usr/bin/env python3
"""Squash consecutive trailing data snapshot commits into one commit."""

from __future__ import annotations

import argparse
import subprocess


DEFAULT_SNAPSHOT_COMMIT_MESSAGE = "Update daily data snapshots"


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_stdout(*args: str) -> str:
    return run_git(*args).stdout.strip()


def ensure_clean_worktree() -> None:
    if run_git("diff", "--quiet", check=False).returncode != 0:
        raise SystemExit("Working tree has unstaged changes; refusing to rewrite history.")
    if run_git("diff", "--cached", "--quiet", check=False).returncode != 0:
        raise SystemExit("Index has staged changes; refusing to rewrite history.")


def commit_subject(commit_ref: str) -> str:
    return git_stdout("show", "-s", "--format=%s", commit_ref)


def parent_commit(commit_ref: str) -> str | None:
    parents = git_stdout("show", "-s", "--format=%P", commit_ref).split()
    if not parents:
        return None
    return parents[0]


def collect_trailing_snapshot_commits(message: str) -> list[str]:
    commits = []
    current_ref = "HEAD"
    while commit_subject(current_ref) == message:
        current_sha = git_stdout("rev-parse", current_ref)
        commits.append(current_sha)
        parent = parent_commit(current_ref)
        if parent is None:
            break
        current_ref = parent
    return commits


def squash_trailing_snapshot_commits(message: str) -> bool:
    ensure_clean_worktree()
    snapshot_commits = collect_trailing_snapshot_commits(message)
    if len(snapshot_commits) <= 1:
        print(f"Found {len(snapshot_commits)} trailing snapshot commit(s); nothing to squash.")
        return False

    oldest_snapshot_parent = parent_commit(snapshot_commits[-1])
    if oldest_snapshot_parent is None:
        raise SystemExit("Refusing to squash snapshot commits without a parent commit.")

    old_head = git_stdout("rev-parse", "HEAD")
    new_head = git_stdout(
        "commit-tree",
        "HEAD^{tree}",
        "-p",
        oldest_snapshot_parent,
        "-m",
        message,
    )
    run_git("update-ref", "HEAD", new_head, old_head)
    print(
        f"Squashed {len(snapshot_commits)} trailing snapshot commits into {new_head[:7]}."
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--message",
        default=DEFAULT_SNAPSHOT_COMMIT_MESSAGE,
        help="Commit subject used to identify data snapshot commits.",
    )
    args = parser.parse_args()
    squash_trailing_snapshot_commits(args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
