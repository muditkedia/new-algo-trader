"""Create a deterministic source-review ZIP without secrets or bulky local evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".secrets",
    ".venv",
    "__pycache__",
    "data",
    "logs",
    "results",
}


def review_bundle_files(root: Path = ROOT) -> tuple[Path, ...]:
    """Return deterministic safe source paths without opening excluded content."""
    selected = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file():
            selected.append(path)
    return tuple(sorted(selected, key=lambda path: path.relative_to(root).as_posix()))


def create_review_bundle(output_path: Path, root: Path = ROOT) -> Path:
    """Write a byte-reproducible ZIP and never include or print file contents."""
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"review bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in review_bundle_files(root):
            if path.resolve() == destination:
                continue
            relative = path.relative_to(root).as_posix()
            info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = create_review_bundle(args.output)
    print(f"Review bundle created: {path}")


if __name__ == "__main__":
    main()
