"""Tests for defect_core.artifacts (Phase 2, step 3).

Synthetic data only: a small random ``resnet18`` and random images. The point is the file format and
its checks, not detection quality. Real pretrained weights and full-size memory banks are not used.
"""

from __future__ import annotations

import copy
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from defect_core.artifacts import (
    MANIFEST_FILE,
    MODEL_FILE,
    SCHEMA_VERSION,
    ArtifactError,
    load_artifact,
    save_artifact,
)
from defect_core.model import ModelConfig, build_model, fit_memory_bank, infer
from defect_core.preprocessing import DEFAULT_CONFIG, PreprocessConfig
from safetensors.torch import load_file, save_file

SIZE = 128
CONFIG = ModelConfig(backbone="resnet18")


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted during test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


def _images(n: int, seed: int) -> torch.Tensor:
    return torch.rand(n, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(seed))


def _fitted_model():
    torch.manual_seed(7)
    model = build_model(CONFIG, pre_trained=False)
    # give the batch-norm buffers non-default values, like pretrained weights would have
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.copy_(torch.randn_like(module.running_mean) * 0.1)
            module.running_var.copy_(torch.rand_like(module.running_var) + 0.5)
    fit_memory_bank(model, [_images(4, 1), _images(4, 2)], CONFIG.coreset_ratio, seed=0)
    return model


@pytest.fixture(scope="module")
def fitted():
    return _fitted_model()


@pytest.fixture
def saved(tmp_path: Path, fitted) -> Path:
    save_artifact(fitted, CONFIG, DEFAULT_CONFIG, tmp_path / "art", provenance={"seed": 0})
    return tmp_path / "art"


def _manifest(directory: Path) -> dict:
    return json.loads((directory / MANIFEST_FILE).read_text())


def _write_manifest(directory: Path, manifest: dict) -> None:
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest))


def _replace_weights(directory: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Swap the weights file and make the manifest's hash and size agree, so that only the shape
    and name checks (not the integrity check) can catch the change."""
    import hashlib

    path = directory / MODEL_FILE
    save_file(tensors, str(path), metadata={"format": "pt"})
    manifest = _manifest(directory)
    manifest["files"][MODEL_FILE] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    _write_manifest(directory, manifest)


# --- the round trip ------------------------------------------------------------------------


def test_reload_gives_bit_identical_outputs(saved: Path, fitted) -> None:
    images = _images(3, 99)
    ref_score, ref_map = infer(fitted, images)
    loaded = load_artifact(saved)
    score, anomaly_map = infer(loaded.model, images)
    assert torch.equal(score, ref_score)
    assert torch.equal(anomaly_map, ref_map)
    assert not loaded.model.training


def test_reload_in_a_fresh_process_gives_identical_outputs(saved: Path, fitted, tmp_path) -> None:
    images = _images(3, 99)
    ref_score, ref_map = infer(fitted, images)
    torch.save(images, tmp_path / "in.pt")
    script = (
        "import sys, torch\n"
        "from defect_core.artifacts import load_artifact\n"
        "from defect_core.model import infer\n"
        "loaded = load_artifact(sys.argv[1])\n"
        "s, m = infer(loaded.model, torch.load(sys.argv[2]))\n"
        "torch.save({'s': s, 'm': m}, sys.argv[3])\n"
    )
    out = tmp_path / "out.pt"
    subprocess.run(
        [sys.executable, "-c", script, str(saved), str(tmp_path / "in.pt"), str(out)],
        check=True,
        capture_output=True,
    )
    result = torch.load(out)
    assert torch.equal(result["s"], ref_score)
    assert torch.equal(result["m"], ref_map)


def test_the_manifest_records_settings_shapes_hash_and_versions(saved: Path, fitted) -> None:
    manifest = _manifest(saved)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["format"] == "safetensors"
    assert manifest["model"] == CONFIG.to_dict()
    assert manifest["preprocessing"] == DEFAULT_CONFIG.to_dict()
    assert manifest["memory_bank"] == {
        "shape": list(fitted.memory_bank.shape),
        "dtype": "torch.float32",
    }
    assert manifest["provenance"] == {"seed": 0}
    assert set(manifest["libraries"]) == {"torch", "anomalib", "safetensors"}
    assert manifest["files"][MODEL_FILE]["bytes"] == (saved / MODEL_FILE).stat().st_size
    assert len(manifest["files"][MODEL_FILE]["sha256"]) == 64
    assert manifest["tensors"]["memory_bank"]["shape"] == list(fitted.memory_bank.shape)


def test_only_the_required_tensors_are_saved(saved: Path) -> None:
    names = set(load_file(str(saved / MODEL_FILE)))
    assert "memory_bank" in names
    assert all(n == "memory_bank" or n.startswith("feature_extractor.") for n in names)
    assert not any("blur" in n or "anomaly_map_generator" in n for n in names)
    assert any(n.endswith("running_mean") for n in names)  # batch-norm buffers are included


def test_saving_is_deterministic(tmp_path: Path, fitted) -> None:
    save_artifact(fitted, CONFIG, DEFAULT_CONFIG, tmp_path / "a")
    save_artifact(fitted, CONFIG, DEFAULT_CONFIG, tmp_path / "b")
    for name in (MODEL_FILE, MANIFEST_FILE):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_load_makes_no_network_call_and_returns_the_settings(saved: Path) -> None:
    loaded = load_artifact(saved)  # the network is blocked by the autouse fixture
    assert loaded.model_config == CONFIG
    assert loaded.preprocess_config == DEFAULT_CONFIG


# --- save refusals -------------------------------------------------------------------------


def test_saving_an_unfitted_model_is_refused(tmp_path: Path) -> None:
    model = build_model(CONFIG, pre_trained=False)
    with pytest.raises(ArtifactError, match="empty memory bank"):
        save_artifact(model, CONFIG, DEFAULT_CONFIG, tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_an_existing_artifact_is_never_overwritten(saved: Path, fitted) -> None:
    before = (saved / MODEL_FILE).read_bytes()
    with pytest.raises(ArtifactError, match="never overwritten"):
        save_artifact(fitted, CONFIG, DEFAULT_CONFIG, saved)
    assert (saved / MODEL_FILE).read_bytes() == before


# --- integrity and consistency checks on load ----------------------------------------------


def test_a_changed_byte_is_detected(saved: Path) -> None:
    path = saved / MODEL_FILE
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="SHA-256"):
        load_artifact(saved)


def test_a_truncated_file_is_detected(saved: Path) -> None:
    path = saved / MODEL_FILE
    path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(ArtifactError, match="bytes"):
        load_artifact(saved)


def test_missing_files_give_clear_errors(saved: Path, tmp_path: Path) -> None:
    (saved / MODEL_FILE).unlink()
    with pytest.raises(ArtifactError, match="Weights file not found"):
        load_artifact(saved)
    with pytest.raises(ArtifactError, match="Manifest not found"):
        load_artifact(tmp_path / "does-not-exist")


def test_bad_manifests_are_refused(saved: Path) -> None:
    good = _manifest(saved)

    (saved / MANIFEST_FILE).write_text("{ not json")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        load_artifact(saved)

    _write_manifest(saved, {**good, "schema_version": 99})
    with pytest.raises(ArtifactError, match="schema_version"):
        load_artifact(saved)

    _write_manifest(saved, {**good, "format": "pickle"})
    with pytest.raises(ArtifactError, match="format"):
        load_artifact(saved)

    broken = copy.deepcopy(good)
    del broken["model"]["backbone"]
    _write_manifest(saved, broken)
    with pytest.raises(ArtifactError, match="'model' section"):
        load_artifact(saved)

    other_file = copy.deepcopy(good)
    other_file["files"]["../evil.safetensors"] = other_file["files"].pop(MODEL_FILE)
    _write_manifest(saved, other_file)
    with pytest.raises(ArtifactError, match="exactly"):
        load_artifact(saved)


def test_a_different_preprocessing_version_is_refused(tmp_path: Path, fitted) -> None:
    old = PreprocessConfig(version="0")
    save_artifact(fitted, CONFIG, old, tmp_path / "old")
    with pytest.raises(ArtifactError, match="preprocessing version"):
        load_artifact(tmp_path / "old")


def test_manifest_shape_that_disagrees_with_the_file_is_refused(saved: Path) -> None:
    manifest = _manifest(saved)
    manifest["memory_bank"]["shape"][0] += 1
    _write_manifest(saved, manifest)
    with pytest.raises(ArtifactError, match="Memory bank shape"):
        load_artifact(saved)


def test_a_missing_backbone_tensor_is_refused(saved: Path) -> None:
    tensors = load_file(str(saved / MODEL_FILE))
    victim = next(n for n in tensors if n.endswith("running_mean"))
    del tensors[victim]
    manifest = _manifest(saved)
    del manifest["tensors"][victim]
    _write_manifest(saved, manifest)
    _replace_weights(saved, tensors)
    with pytest.raises(ArtifactError, match="Backbone tensor names"):
        load_artifact(saved)


def test_a_backbone_tensor_of_the_wrong_shape_is_refused(saved: Path) -> None:
    tensors = load_file(str(saved / MODEL_FILE))
    victim = next(n for n in tensors if n.endswith("conv1.weight"))
    tensors[victim] = torch.zeros(3, 3)
    manifest = _manifest(saved)
    manifest["tensors"][victim]["shape"] = [3, 3]
    _write_manifest(saved, manifest)
    _replace_weights(saved, tensors)
    with pytest.raises(ArtifactError, match="do not fit the model"):
        load_artifact(saved)


def test_a_memory_bank_of_the_wrong_width_is_refused(saved: Path) -> None:
    tensors = load_file(str(saved / MODEL_FILE))
    tensors["memory_bank"] = tensors["memory_bank"][:, :-1].contiguous()
    manifest = _manifest(saved)
    manifest["tensors"]["memory_bank"]["shape"] = list(tensors["memory_bank"].shape)
    manifest["memory_bank"]["shape"] = list(tensors["memory_bank"].shape)
    _write_manifest(saved, manifest)
    _replace_weights(saved, tensors)
    with pytest.raises(ArtifactError, match="feature size"):
        load_artifact(saved)


def test_a_memory_bank_with_nan_is_refused(saved: Path) -> None:
    tensors = load_file(str(saved / MODEL_FILE))
    tensors["memory_bank"][0, 0] = float("nan")
    _replace_weights(saved, tensors)
    with pytest.raises(ArtifactError, match="NaN"):
        load_artifact(saved)


def test_a_wrong_dtype_is_refused(saved: Path) -> None:
    tensors = load_file(str(saved / MODEL_FILE))
    tensors["memory_bank"] = tensors["memory_bank"].half()
    _replace_weights(saved, tensors)  # manifest still says float32
    with pytest.raises(ArtifactError, match="differ from the manifest"):
        load_artifact(saved)


def test_a_changed_blur_kernel_record_is_detected(saved: Path) -> None:
    manifest = _manifest(saved)
    manifest["rebuilt_tensors"]["anomaly_map_generator.blur.kernel"]["sha256"] = "0" * 64
    _write_manifest(saved, manifest)
    with pytest.raises(ArtifactError, match="blur kernel"):
        load_artifact(saved)


def test_an_unknown_backbone_in_the_manifest_gives_a_clear_error(saved: Path) -> None:
    manifest = _manifest(saved)
    manifest["model"]["backbone"] = "no_such_backbone"
    _write_manifest(saved, manifest)
    with pytest.raises(ArtifactError, match="Could not build the model"):
        load_artifact(saved)
