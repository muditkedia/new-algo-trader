"""Transparent checksum-verified Trade Meta-Model persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb

from algo_trader.ml.meta_model import TrainedTradeMetaModel
from algo_trader.ml.models import MetaModelMetadata

ARTIFACT_FILES = ("metadata.json", "classifier.txt", "regressor.txt", "calibrator.json")


class ModelArtifactIntegrityError(ValueError):
    """Raised when persisted model files are missing or fail their checksums."""


def save_trade_meta_model(model: TrainedTradeMetaModel, output_directory: Path) -> Path:
    """Write a new immutable native-model artifact without overwriting a path."""
    if not isinstance(model, TrainedTradeMetaModel):
        raise TypeError("model must be a TrainedTradeMetaModel")
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError(f"Trade Meta-Model artifact path already exists: {directory}")
    directory.mkdir(parents=True)
    metadata = model.metadata.model_dump(
        mode="json",
        exclude={"calibration_coefficient", "calibration_intercept"},
    )
    payloads = {
        "metadata.json": _canonical_json(metadata),
        "classifier.txt": model.classifier.model_to_string(),
        "regressor.txt": model.regressor.model_to_string(),
        "calibrator.json": _canonical_json(
            {
                "coefficient": model.metadata.calibration_coefficient,
                "intercept": model.metadata.calibration_intercept,
            }
        ),
    }
    for filename, payload in payloads.items():
        (directory / filename).write_text(payload, encoding="utf-8", newline="\n")
    manifest = {
        "algorithm": "sha256",
        "files": {
            filename: _sha256((directory / filename).read_bytes())
            for filename in ARTIFACT_FILES
        },
    }
    (directory / "manifest.json").write_text(
        _canonical_json(manifest), encoding="utf-8", newline="\n"
    )
    return directory


def load_trade_meta_model(input_directory: Path) -> TrainedTradeMetaModel:
    """Verify all component hashes before reconstructing native boosters."""
    directory = Path(input_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ModelArtifactIntegrityError("model artifact manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "sha256" or set(manifest.get("files", {})) != set(
        ARTIFACT_FILES
    ):
        raise ModelArtifactIntegrityError("model artifact manifest is invalid")
    for filename in ARTIFACT_FILES:
        path = directory / filename
        if not path.is_file():
            raise ModelArtifactIntegrityError(f"model artifact component is missing: {filename}")
        if _sha256(path.read_bytes()) != manifest["files"][filename]:
            raise ModelArtifactIntegrityError(f"model artifact checksum failed: {filename}")

    metadata_values = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((directory / "calibrator.json").read_text(encoding="utf-8"))
    metadata = MetaModelMetadata(
        **metadata_values,
        calibration_coefficient=calibration["coefficient"],
        calibration_intercept=calibration["intercept"],
    )
    return TrainedTradeMetaModel(
        metadata=metadata,
        classifier=lgb.Booster(model_file=str(directory / "classifier.txt")),
        regressor=lgb.Booster(model_file=str(directory / "regressor.txt")),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
