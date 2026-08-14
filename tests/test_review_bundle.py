from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from scripts.create_review_bundle import create_review_bundle, review_bundle_files


def test_review_bundle_excludes_sensitive_and_bulky_directories(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("pass\n", encoding="utf-8")
    for directory in (".secrets", ".venv", ".git", "data", "logs", "results"):
        target = root / directory
        target.mkdir()
        (target / "forbidden.txt").write_text("secret", encoding="utf-8")
    assert [path.relative_to(root).as_posix() for path in review_bundle_files(root)] == [
        "src/app.py"
    ]
    bundle = create_review_bundle(tmp_path / "review.zip", root)
    with ZipFile(bundle) as archive:
        assert archive.namelist() == ["src/app.py"]
