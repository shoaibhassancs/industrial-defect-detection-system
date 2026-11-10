"""Save and load a fitted PatchCore model: safetensors plus a manifest (Phase 2 step 3).

Format (decision 014, confirmed by the owner on 2026-11-10, after step F2):

* ``model.safetensors``: only tensors, no pickle. It holds the backbone parameters and batch-norm
  buffers (names starting with ``feature_extractor.``) and the ``memory_bank``. F2 showed these are
  what is needed to reproduce the outputs; leaving out the batch-norm buffers changes them.
* ``manifest.json``: the constructor settings (backbone, layers, neighbours, coreset ratio), the
  preprocessing settings, the name, shape and dtype of every saved tensor, the SHA-256 and size of
  the weights file, and the library versions.

The Gaussian blur kernel of the anomaly map is *not* saved: anomalib rebuilds it in ``__init__``.
Its shape and hash are recorded, and ``load_artifact`` refuses a model whose rebuilt kernel differs.

Stages (Phase 3 steps 2-3). A fitted artifact has no ``stage`` key. ``freeze_artifact`` copies
it to a new folder with ``stage: "frozen"`` and a ``freeze`` block (version, the settings fixed
in advance, and the hash of the weights it came from). ``calibrate_artifact`` copies a frozen
artifact to another new folder with ``stage: "calibrated"`` and a ``calibration`` block. No stage
changes the weights, and no stage overwrites an earlier one. ``load_artifact`` checks the stage
fields (see ``_check_stage``).

Loading checks, in this order: the manifest is well formed; the preprocessing version is the one
this code implements; the weights file has the recorded size and SHA-256 (before it is opened as
tensors); the tensor names, shapes and dtypes match the manifest; the rebuilt model accepts the
tensors with strict name checking. Any failure raises ``ArtifactError`` with a plain message.

This is the local file layout. Choosing where artifacts are stored and published is Phase 4.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import anomalib
import safetensors
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from safetensors.torch import load_file, save_file

from defect_core.calibration import Calibration, CalibrationError
from defect_core.model import ModelConfig, build_model
from defect_core.preprocessing import PREPROCESSING_VERSION, PreprocessConfig

SCHEMA_VERSION = 1
FORMAT = "safetensors"
MODEL_FILE = "model.safetensors"
MANIFEST_FILE = "manifest.json"

_BACKBONE_PREFIX = "feature_extractor."
_BANK_KEY = "memory_bank"
_REBUILT_KEYS = ("anomaly_map_generator.blur.kernel",)  # rebuilt by anomalib, not saved

STAGE_FROZEN = "frozen"
STAGE_CALIBRATED = "calibrated"
SETTING_KEYS = ("alpha", "lower_percentile", "highlight_percentile", "upper_multiple")


class ArtifactError(Exception):
    """The artifact is missing, damaged, or does not match its manifest."""


@dataclass(frozen=True)
class LoadedArtifact:
    model: PatchcoreModel  # eval mode, memory bank filled
    model_config: ModelConfig
    preprocess_config: PreprocessConfig
    manifest: dict[str, object]
    calibration: Calibration | None = None  # present only for a calibrated artifact


# --- helpers -------------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _describe(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in sorted(tensors.items())
    }


def _saved_tensors(model: PatchcoreModel) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    saved = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items()
        if name.startswith(_BACKBONE_PREFIX) or name == _BANK_KEY
    }
    left_out = set(state) - set(saved)
    if left_out != set(_REBUILT_KEYS):
        # anomalib changed what its state holds: refuse rather than save an incomplete model
        raise ArtifactError(f"Unexpected model state; not saved: {sorted(left_out)}")
    return saved


def _rebuilt_info(model: PatchcoreModel) -> dict[str, dict[str, object]]:
    state = model.state_dict()
    return {
        key: {"shape": list(state[key].shape), "sha256": _tensor_sha256(state[key])}
        for key in _REBUILT_KEYS
    }


def _versions() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "anomalib": anomalib.__version__,
        "safetensors": safetensors.__version__,
    }


# --- save ----------------------------------------------------------------------------------


def save_artifact(
    model: PatchcoreModel,
    model_config: ModelConfig,
    preprocess_config: PreprocessConfig,
    out_dir: str | Path,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write ``model.safetensors`` and ``manifest.json`` into a new directory; return the manifest.

    The model must be fitted (non-empty memory bank). An artifact is never overwritten: ``out_dir``
    must not already contain either file. ``provenance`` is free-form JSON-compatible metadata the
    caller wants recorded (for example the seed and the split hash).
    """
    if model.memory_bank.numel() == 0:
        raise ArtifactError("Cannot save a model with an empty memory bank")
    if model.memory_bank.ndim != 2:
        raise ArtifactError(f"Memory bank must be 2-D, got shape {tuple(model.memory_bank.shape)}")

    out = Path(out_dir)
    if (out / MODEL_FILE).exists() or (out / MANIFEST_FILE).exists():
        raise ArtifactError(f"{out} already holds an artifact; artifacts are never overwritten")
    out.mkdir(parents=True, exist_ok=True)

    tensors = _saved_tensors(model)
    weights_path = out / MODEL_FILE
    save_file(tensors, str(weights_path), metadata={"format": "pt"})

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "model": model_config.to_dict(),
        "preprocessing": preprocess_config.to_dict(),
        "memory_bank": {
            "shape": list(model.memory_bank.shape),
            "dtype": str(model.memory_bank.dtype),
        },
        "tensors": _describe(tensors),
        "rebuilt_tensors": _rebuilt_info(model),
        "files": {
            MODEL_FILE: {"sha256": _file_sha256(weights_path), "bytes": weights_path.stat().st_size}
        },
        "libraries": _versions(),
        "provenance": provenance or {},
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# --- load ----------------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _read_manifest(directory: Path) -> dict[str, object]:
    path = directory / MANIFEST_FILE
    _require(path.is_file(), f"Manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ArtifactError(f"Manifest is not valid JSON: {error}") from error
    _require(isinstance(manifest, dict), "Manifest must be a JSON object")
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        f"Unsupported manifest schema_version {manifest.get('schema_version')!r}; "
        f"this code reads version {SCHEMA_VERSION}",
    )
    _require(manifest.get("format") == FORMAT, f"Unsupported format {manifest.get('format')!r}")
    for key in ("model", "preprocessing", "memory_bank", "tensors", "rebuilt_tensors", "files"):
        _require(isinstance(manifest.get(key), dict), f"Manifest field {key!r} is missing")
    return manifest


def _model_config(data: dict[str, object]) -> ModelConfig:
    try:
        layers = data["layers"]
        _require(isinstance(layers, list) and bool(layers), "model.layers must be a list")
        return ModelConfig(
            backbone=str(data["backbone"]),
            layers=tuple(str(layer) for layer in layers),
            num_neighbors=int(data["num_neighbors"]),
            coreset_ratio=float(data["coreset_ratio"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(f"Manifest 'model' section is invalid: {error}") from error


def _preprocess_config(data: dict[str, object]) -> PreprocessConfig:
    try:
        config = PreprocessConfig(
            version=str(data["version"]),
            size=int(data["size"]),
            mean=tuple(float(x) for x in data["mean"]),  # type: ignore[arg-type]
            std=tuple(float(x) for x in data["std"]),  # type: ignore[arg-type]
            interpolation=str(data["interpolation"]),
            antialias=bool(data["antialias"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(f"Manifest 'preprocessing' section is invalid: {error}") from error
    _require(
        config.version == PREPROCESSING_VERSION,
        f"Artifact was made with preprocessing version {config.version!r}, but this code "
        f"implements version {PREPROCESSING_VERSION!r}",
    )
    return config


def load_artifact(directory: str | Path) -> LoadedArtifact:
    """Load and verify an artifact; return the fitted model (eval mode) and its settings."""
    directory = Path(directory)
    manifest = _read_manifest(directory)
    model_config = _model_config(manifest["model"])  # type: ignore[arg-type]
    preprocess_config = _preprocess_config(manifest["preprocessing"])  # type: ignore[arg-type]

    # 1. integrity of the weights file, before it is opened as tensors
    files = manifest["files"]
    _require(
        isinstance(files, dict) and set(files) == {MODEL_FILE},
        f"Manifest 'files' must list exactly {MODEL_FILE}",
    )
    entry = files[MODEL_FILE]
    weights_path = directory / MODEL_FILE
    _require(weights_path.is_file(), f"Weights file not found: {weights_path}")
    _require(isinstance(entry, dict) and isinstance(entry.get("sha256"), str), "Bad 'files' entry")
    _require(
        weights_path.stat().st_size == entry.get("bytes"),
        f"{MODEL_FILE} has {weights_path.stat().st_size} bytes, manifest says {entry.get('bytes')}",
    )
    _require(
        _file_sha256(weights_path) == entry["sha256"],
        f"{MODEL_FILE} does not match its recorded SHA-256 (file damaged or replaced)",
    )

    # 2. tensor names, shapes and dtypes against the manifest
    try:
        tensors = load_file(str(weights_path))
    except Exception as error:  # safetensors raises several types for malformed files
        raise ArtifactError(f"Could not read {MODEL_FILE}: {error}") from error
    _require(
        _describe(tensors) == manifest["tensors"],
        "Tensor names, shapes or dtypes in the weights file differ from the manifest",
    )
    bank = tensors.get(_BANK_KEY)
    _require(bank is not None, "Weights file has no memory_bank")
    _require(
        {"shape": list(bank.shape), "dtype": str(bank.dtype)} == manifest["memory_bank"],
        "Memory bank shape or dtype differs from the manifest",
    )
    _require(bank.ndim == 2 and bank.shape[0] > 0, "Memory bank must be a non-empty 2-D tensor")
    _require(bool(torch.isfinite(bank).all()), "Memory bank contains NaN or infinite values")

    # 3. rebuild the model and load strictly
    try:
        model = build_model(model_config, pre_trained=False)
    except Exception as error:
        raise ArtifactError(f"Could not build the model from the manifest: {error}") from error
    backbone = {k: v for k, v in tensors.items() if k != _BANK_KEY}
    expected = {
        k for k in model.state_dict() if k.startswith(_BACKBONE_PREFIX)
    }  # what this model needs
    _require(
        set(backbone) == expected,
        "Backbone tensor names do not match the model built from the manifest "
        f"(missing {sorted(expected - set(backbone))[:3]}, "
        f"unexpected {sorted(set(backbone) - expected)[:3]})",
    )
    try:
        model.load_state_dict(backbone, strict=False)  # the bank and blur kernel are not in it
    except RuntimeError as error:
        raise ArtifactError(f"Backbone tensors do not fit the model: {error}") from error
    model.memory_bank = bank.clone()
    _require(
        bank.shape[1] == _embedding_width(model),
        f"Memory bank width {bank.shape[1]} does not match the model's feature size",
    )

    # 4. the rebuilt (unsaved) tensors must be what the artifact was made with
    _require(
        _rebuilt_info(model) == manifest["rebuilt_tensors"],
        "The anomaly-map blur kernel rebuilt by this anomalib differs from the one the artifact "
        "was made with (anomalib version change?)",
    )

    # 5. the stage fields (frozen / calibrated) must be consistent with the weights and each other
    calibration = _check_stage(manifest, entry["sha256"])
    return LoadedArtifact(model.eval(), model_config, preprocess_config, manifest, calibration)


def _embedding_width(model: PatchcoreModel) -> int:
    """Feature size of one patch: the summed channel count of the selected layers."""
    return int(sum(model.feature_extractor.out_dims))


# --- stages: frozen and calibrated ---------------------------------------------------------


def canonical_sha256(data: object) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no spaces) of plain data."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _settings_ok(settings: object) -> bool:
    return (
        isinstance(settings, dict)
        and set(SETTING_KEYS) <= set(settings)
        and all(
            isinstance(settings[k], int | float) and not isinstance(settings[k], bool)
            for k in SETTING_KEYS
        )
    )


def _check_stage(manifest: dict[str, object], weights_sha256: str) -> Calibration | None:
    """Check the ``stage``, ``freeze`` and ``calibration`` fields; return the calibration if any.

    A calibration block that does not match the weights it sits next to, the settings fixed at
    freeze time, or itself is refused. This catches a block copied from another model, an edited
    threshold or score, and settings changed after the freeze. A person who edits the block and
    also recomputes every hash can not be detected by hashes stored in the same folder.
    """
    stage = manifest.get("stage")
    has_freeze, has_calibration = "freeze" in manifest, "calibration" in manifest
    if stage is None:
        _require(not has_freeze and not has_calibration, "Manifest has stage data but no 'stage'")
        return None
    _require(stage in (STAGE_FROZEN, STAGE_CALIBRATED), f"Unknown artifact stage {stage!r}")

    freeze = manifest.get("freeze")
    _require(isinstance(freeze, dict), "A frozen artifact needs a 'freeze' block")
    _require(
        isinstance(freeze.get("version"), str) and bool(freeze["version"]),
        "The 'freeze' block has no version",
    )
    _require(_settings_ok(freeze.get("frozen_settings")), "The frozen settings are incomplete")
    source = freeze.get("source")
    _require(
        isinstance(source, dict) and source.get("weights_sha256") == weights_sha256,
        "The weights are not the ones that were frozen (SHA-256 differs from the freeze block)",
    )

    if stage == STAGE_FROZEN:
        _require(not has_calibration, "A frozen (not calibrated) artifact has a calibration block")
        return None

    entry = manifest.get("calibration")
    _require(isinstance(entry, dict), "A calibrated artifact needs a 'calibration' block")
    block = entry.get("block")
    _require(isinstance(block, dict), "The calibration block is missing")
    try:
        calibration = Calibration.from_dict(block)
    except CalibrationError as error:
        raise ArtifactError(str(error)) from error
    _require(
        entry.get("block_sha256") == canonical_sha256(block),
        "The calibration block was changed after it was written (its SHA-256 differs)",
    )
    _require(
        entry.get("weights_sha256") == weights_sha256,
        "The calibration was made for different weights than this artifact holds",
    )
    frozen = freeze["frozen_settings"]
    used = {
        "alpha": calibration.alpha,
        "lower_percentile": calibration.pixel.lower_percentile,
        "highlight_percentile": calibration.pixel.highlight_percentile,
        "upper_multiple": calibration.pixel.upper_multiple,
    }
    _require(
        all(float(frozen[k]) == used[k] for k in SETTING_KEYS),
        "The calibration used settings that differ from the ones fixed at freeze time",
    )
    inputs = entry.get("inputs")
    _require(
        isinstance(inputs, dict)
        and inputs.get("group") == "calib"
        and inputs.get("n") == calibration.n,
        "The calibration inputs do not describe the stored scores (group 'calib', same n)",
    )
    _require(
        inputs.get("preprocessing_version") == manifest["preprocessing"].get("version"),  # type: ignore[union-attr]
        "The calibration images were prepared with a different preprocessing version",
    )
    return calibration


def _new_folder(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ArtifactError(f"{path} already exists; artifacts are never overwritten")
    path.mkdir(parents=True, exist_ok=True)


def _copy_weights(source: Path, target: Path, expected_sha256: str) -> None:
    shutil.copyfile(source / MODEL_FILE, target / MODEL_FILE)
    if _file_sha256(target / MODEL_FILE) != expected_sha256:
        raise ArtifactError("The copied weights do not match the original (SHA-256 differs)")


def _write_manifest(directory: Path, manifest: dict[str, object]) -> None:
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def freeze_artifact(
    source_dir: str | Path, out_dir: str | Path, version: str, frozen_settings: dict[str, object],
    details: dict[str, object] | None = None,
) -> dict[str, object]:  # fmt: skip
    """Copy a verified, fitted artifact to a new folder and mark it frozen. Returns the manifest.

    ``frozen_settings`` are the calibration settings fixed in advance (``SETTING_KEYS`` plus any
    notes); calibration later has to use exactly these. The weights are copied byte for byte and
    their hash is checked. The source folder is not changed. Only a fitted artifact (no stage) can
    be frozen.
    """
    source, out = Path(source_dir), Path(out_dir)
    loaded = load_artifact(source)  # verifies the source completely
    _require("stage" not in loaded.manifest, "Only a fitted artifact can be frozen")
    _require(bool(version), "A version is needed")
    _require(_settings_ok(frozen_settings), f"frozen_settings needs numbers for {SETTING_KEYS}")
    _new_folder(out)
    weights_sha256 = loaded.manifest["files"][MODEL_FILE]["sha256"]  # type: ignore[index]
    _copy_weights(source, out, weights_sha256)
    manifest = copy.deepcopy(loaded.manifest)
    manifest["stage"] = STAGE_FROZEN
    manifest["freeze"] = {
        "version": version,
        "frozen_settings": frozen_settings,
        "source": {
            "weights_sha256": weights_sha256,
            "manifest_sha256": _file_sha256(source / MANIFEST_FILE),
        },
        **(details or {}),
    }
    _write_manifest(out, manifest)
    load_artifact(out)  # the new folder must load cleanly
    return manifest


def calibrate_artifact(
    frozen_dir: str | Path, out_dir: str | Path, calibration: Calibration,
    inputs: dict[str, object],
) -> dict[str, object]:  # fmt: skip
    """Copy a frozen artifact to a new folder with a calibration block. Returns the manifest.

    ``calibration`` must have been made with the settings fixed at freeze time. ``inputs`` says
    what was scored: ``group`` ("calib"), ``n``, ``preprocessing_version`` and whatever the caller
    wants recorded (list hashes). The weights are copied byte for byte; the frozen folder is not
    changed.
    """
    frozen, out = Path(frozen_dir), Path(out_dir)
    loaded = load_artifact(frozen)
    _require(
        loaded.manifest.get("stage") == STAGE_FROZEN, "Only a frozen artifact can be calibrated"
    )
    _require(isinstance(inputs, dict), "inputs must be a dict")
    _new_folder(out)
    weights_sha256 = loaded.manifest["files"][MODEL_FILE]["sha256"]  # type: ignore[index]
    block = calibration.to_dict()
    manifest = copy.deepcopy(loaded.manifest)
    manifest["stage"] = STAGE_CALIBRATED
    manifest["calibration"] = {
        "block": block,
        "block_sha256": canonical_sha256(block),
        "weights_sha256": weights_sha256,
        "frozen_manifest_sha256": _file_sha256(frozen / MANIFEST_FILE),
        "inputs": inputs,
    }
    try:
        _check_stage(manifest, weights_sha256)  # refuse before anything is written
    except ArtifactError:
        out.rmdir()
        raise
    _copy_weights(frozen, out, weights_sha256)
    _write_manifest(out, manifest)
    load_artifact(out)
    return manifest
