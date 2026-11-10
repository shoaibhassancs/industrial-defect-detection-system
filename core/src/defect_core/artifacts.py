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

Loading checks, in this order: the manifest is well formed; the preprocessing version is the one
this code implements; the weights file has the recorded size and SHA-256 (before it is opened as
tensors); the tensor names, shapes and dtypes match the manifest; the rebuilt model accepts the
tensors with strict name checking. Any failure raises ``ArtifactError`` with a plain message.

This is the local file layout. Choosing where artifacts are stored and published is Phase 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import anomalib
import safetensors
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from safetensors.torch import load_file, save_file

from defect_core.model import ModelConfig, build_model
from defect_core.preprocessing import PREPROCESSING_VERSION, PreprocessConfig

SCHEMA_VERSION = 1
FORMAT = "safetensors"
MODEL_FILE = "model.safetensors"
MANIFEST_FILE = "manifest.json"

_BACKBONE_PREFIX = "feature_extractor."
_BANK_KEY = "memory_bank"
_REBUILT_KEYS = ("anomaly_map_generator.blur.kernel",)  # rebuilt by anomalib, not saved


class ArtifactError(Exception):
    """The artifact is missing, damaged, or does not match its manifest."""


@dataclass(frozen=True)
class LoadedArtifact:
    model: PatchcoreModel  # eval mode, memory bank filled
    model_config: ModelConfig
    preprocess_config: PreprocessConfig
    manifest: dict[str, object]


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
    return LoadedArtifact(model.eval(), model_config, preprocess_config, manifest)


def _embedding_width(model: PatchcoreModel) -> int:
    """Feature size of one patch: the summed channel count of the selected layers."""
    return int(sum(model.feature_extractor.out_dims))
