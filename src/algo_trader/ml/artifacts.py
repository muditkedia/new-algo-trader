"""Transparent checksum-verified Trade Meta-Model persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
from pydantic import ValidationError

from algo_trader.ml.meta_model import TrainedTradeMetaModel
from algo_trader.ml.models import MetaModelArtifactIdentity, MetaModelMetadata

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
    _, metadata, _ = _verify_artifact(directory)
    return TrainedTradeMetaModel(
        metadata=metadata,
        classifier=lgb.Booster(model_file=str(directory / "classifier.txt")),
        regressor=lgb.Booster(model_file=str(directory / "regressor.txt")),
    )


def inspect_trade_meta_model_artifact(input_directory: Path) -> MetaModelArtifactIdentity:
    """Return the composite identity of a fully checksum-verified artifact."""
    manifest, metadata, _ = _verify_artifact(Path(input_directory))
    return MetaModelArtifactIdentity(
        research_scope_id=metadata.research_scope_id,
        plan_id=metadata.plan_id,
        model_version=metadata.model_version,
        allowed_strategy_ids=metadata.allowed_strategy_ids,
        artifact_fingerprint=_sha256(_canonical_json(manifest).encode("utf-8")),
    )


def _verify_artifact(
    directory: Path,
) -> tuple[dict[str, object], MetaModelMetadata, dict[str, object]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ModelArtifactIntegrityError("model artifact manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelArtifactIntegrityError("model artifact manifest is invalid") from error
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        manifest.get("algorithm") != "sha256"
        or not isinstance(files, dict)
        or set(files) != set(ARTIFACT_FILES)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in files.values()
        )
    ):
        raise ModelArtifactIntegrityError("model artifact manifest is invalid")
    for filename in ARTIFACT_FILES:
        path = directory / filename
        if not path.is_file():
            raise ModelArtifactIntegrityError(f"model artifact component is missing: {filename}")
        if _sha256(path.read_bytes()) != files[filename]:
            raise ModelArtifactIntegrityError(f"model artifact checksum failed: {filename}")

    try:
        metadata_values = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        calibration = json.loads(
            (directory / "calibrator.json").read_text(encoding="utf-8")
        )
        metadata = MetaModelMetadata(
            **metadata_values,
            calibration_coefficient=calibration["coefficient"],
            calibration_intercept=calibration["intercept"],
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValidationError,
    ) as error:
        raise ModelArtifactIntegrityError("model artifact metadata is invalid") from error
    return manifest, metadata, calibration


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
