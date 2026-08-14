"""Network-free local project validation entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = (
        (sys.executable, "-m", "pytest", "-q"),
        (sys.executable, "-m", "ruff", "check", "."),
        ("git", "diff", "--check"),
    )
    for command in commands:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
