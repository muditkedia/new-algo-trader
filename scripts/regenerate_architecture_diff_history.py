"""Rebuild the local cumulative architecture review artifact from Git evidence."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "architecture_diff_history.txt"
RELEVANT_PATHS = (".gitignore", "pyproject.toml", "src", "tests", "docs", "scripts")
COMMITTED_HEADER = (
    "============================================================\n"
    "COMMITTED ARCHITECTURE HISTORY\n"
    "============================================================\n"
)
CURRENT_HEADER = (
    "\n============================================================\n"
    "CURRENT UNCOMMITTED TASK\n"
    "============================================================\n"
)
STATUS_HEADER = (
    "\n============================================================\n"
    "CURRENT GIT STATUS\n"
    "============================================================\n"
)


def git(*arguments: str, accepted: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="strict",
        capture_output=True,
        check=False,
    )
    if result.returncode not in accepted:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def main() -> None:
    committed = git(
        "log",
        "--reverse",
        "--format=fuller",
        "--patch",
        "--",
        *RELEVANT_PATHS,
    )
    current = git("diff", "--no-ext-diff", "HEAD", "--", *RELEVANT_PATHS)
    untracked = tuple(
        sorted(
            filter(
                None,
                git(
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    *RELEVANT_PATHS,
                ).splitlines(),
            )
        )
    )
    untracked_patches = []
    for path in untracked:
        untracked_patches.append(
            git(
                "diff",
                "--no-index",
                "--no-ext-diff",
                "--",
                "/dev/null",
                path,
                accepted=(0, 1),
            )
        )
    status = git("status", "--short")
    content = (
        COMMITTED_HEADER
        + committed
        + CURRENT_HEADER
        + current
        + "".join(untracked_patches)
        + STATUS_HEADER
        + status
    )
    temporary = OUTPUT.with_suffix(".txt.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
