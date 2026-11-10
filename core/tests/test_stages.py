"""Tests for the frozen and calibrated stages of an artifact (Phase 3 steps 2-3).

Synthetic data only (a small random ``resnet18`` and random scores): the point is that stages are
recorded, that nothing is overwritten, and that a calibration block which does not match its
weights, its settings or itself is refused on load.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from defect_core.artifacts import (
    MANIFEST_FILE,
    MODEL_FILE,
    ArtifactError,
    calibrate_artifact,
    canonical_sha256,
    freeze_artifact,
    load_artifact,
    save_artifact,
)
from defect_core.calibration import build_calibration
from defect_core.model import ModelConfig, build_model, fit_memory_bank
from defect_core.preprocessing import DEFAULT_CONFIG

CONFIG = ModelConfig(backbone="resnet18")
SETTINGS = {
    "alpha": 0.1, "lower_percentile": 1.0, "highlight_percentile": 99.5, "upper_multiple": 1.5,
    "upper_multiple_status": "unvalidated display choice",
}  # fmt: skip
INPUTS = {"group": "calib", "n": 20, "preprocessing_version": DEFAULT_CONFIG.version}


@pytest.fixture(scope="module")
def fitted_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(3)
    model = build_model(CONFIG, pre_trained=False)
    gen = torch.Generator().manual_seed(1)
    batches = [torch.rand(4, 3, 96, 96, generator=gen) for _ in range(2)]
    fit_memory_bank(model, batches, CONFIG.coreset_ratio, seed=0)
    path = tmp_path_factory.mktemp("fitted") / "fit"
    save_artifact(model, CONFIG, DEFAULT_CONFIG, path, provenance={"seed": 0})
    return path


def _calibration(n: int = 20, **override: float):
    rng = np.random.default_rng(0)
    settings = {k: SETTINGS[k] for k in SETTINGS if k != "upper_multiple_status"} | override
    maps = [rng.random((8, 8)) + 0.1 for _ in range(n)]
    return build_calibration(rng.random(n) + 1.0, maps, **settings)


@pytest.fixture
def frozen(tmp_path: Path, fitted_dir: Path) -> Path:
    freeze_artifact(fitted_dir, tmp_path / "frozen", "1.0.0", SETTINGS)
    return tmp_path / "frozen"


@pytest.fixture
def calibrated(tmp_path: Path, frozen: Path) -> Path:
    calibrate_artifact(frozen, tmp_path / "cal", _calibration(), INPUTS)
    return tmp_path / "cal"


def _read(directory: Path) -> dict:
    return json.loads((directory / MANIFEST_FILE).read_text())


def _write(directory: Path, manifest: dict) -> None:
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest))


# --- freezing -------------------------------------------------------------------------------


def test_freezing_copies_the_weights_exactly_and_leaves_the_fitted_folder_alone(
    tmp_path: Path, fitted_dir: Path
) -> None:
    before = {p.name: p.read_bytes() for p in fitted_dir.iterdir()}
    manifest = freeze_artifact(fitted_dir, tmp_path / "f", "1.0.0", SETTINGS, {"note": "x"})
    assert {p.name: p.read_bytes() for p in fitted_dir.iterdir()} == before
    assert (tmp_path / "f" / MODEL_FILE).read_bytes() == (fitted_dir / MODEL_FILE).read_bytes()
    assert manifest["stage"] == "frozen" and manifest["freeze"]["version"] == "1.0.0"
    assert manifest["freeze"]["note"] == "x"
    assert manifest["freeze"]["source"]["weights_sha256"] == manifest["files"][MODEL_FILE]["sha256"]
    loaded = load_artifact(tmp_path / "f")
    assert loaded.calibration is None and loaded.manifest["stage"] == "frozen"


def test_a_frozen_folder_is_never_overwritten(tmp_path: Path, fitted_dir: Path) -> None:
    freeze_artifact(fitted_dir, tmp_path / "f", "1.0.0", SETTINGS)
    with pytest.raises(ArtifactError, match="never overwritten"):
        freeze_artifact(fitted_dir, tmp_path / "f", "1.0.0", SETTINGS)


def test_only_a_fitted_artifact_can_be_frozen(tmp_path: Path, frozen: Path) -> None:
    with pytest.raises(ArtifactError, match="Only a fitted"):
        freeze_artifact(frozen, tmp_path / "again", "1.0.1", SETTINGS)


def test_freezing_needs_a_version_and_complete_settings(tmp_path: Path, fitted_dir: Path) -> None:
    with pytest.raises(ArtifactError, match="version"):
        freeze_artifact(fitted_dir, tmp_path / "a", "", SETTINGS)
    with pytest.raises(ArtifactError, match="frozen_settings"):
        freeze_artifact(fitted_dir, tmp_path / "b", "1.0.0", {"alpha": 0.1})
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_a_damaged_source_is_not_frozen(tmp_path: Path, fitted_dir: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(fitted_dir, broken)
    data = bytearray((broken / MODEL_FILE).read_bytes())
    data[-1] ^= 1
    (broken / MODEL_FILE).write_bytes(bytes(data))
    with pytest.raises(ArtifactError):
        freeze_artifact(broken, tmp_path / "out", "1.0.0", SETTINGS)
    assert not (tmp_path / "out").exists()


# --- calibrating ----------------------------------------------------------------------------


def test_calibration_adds_a_checked_block_and_changes_nothing_else(
    tmp_path: Path, frozen: Path
) -> None:
    frozen_files = {p.name: p.read_bytes() for p in frozen.iterdir()}
    calibration = _calibration()
    calibrate_artifact(frozen, tmp_path / "cal", calibration, INPUTS)
    assert {p.name: p.read_bytes() for p in frozen.iterdir()} == frozen_files
    assert (tmp_path / "cal" / MODEL_FILE).read_bytes() == (frozen / MODEL_FILE).read_bytes()
    loaded = load_artifact(tmp_path / "cal")
    assert loaded.calibration == calibration and loaded.manifest["stage"] == "calibrated"
    assert loaded.manifest["freeze"] == _read(frozen)["freeze"]


def test_only_a_frozen_artifact_can_be_calibrated_and_never_twice_into_one_folder(
    tmp_path: Path, fitted_dir: Path, calibrated: Path
) -> None:
    with pytest.raises(ArtifactError, match="Only a frozen"):
        calibrate_artifact(fitted_dir, tmp_path / "x", _calibration(), INPUTS)
    with pytest.raises(ArtifactError, match="Only a frozen"):
        calibrate_artifact(calibrated, tmp_path / "y", _calibration(), INPUTS)
    with pytest.raises(ArtifactError, match="never overwritten"):
        calibrate_artifact(calibrated.parent / "frozen", calibrated, _calibration(), INPUTS)


def test_calibration_with_other_settings_than_the_frozen_ones_is_refused(
    tmp_path: Path, frozen: Path
) -> None:
    for override in ({"alpha": 0.2}, {"upper_multiple": 2.0}, {"lower_percentile": 5.0}):
        with pytest.raises(ArtifactError, match="differ from the ones fixed at freeze"):
            calibrate_artifact(frozen, tmp_path / "c", _calibration(**override), INPUTS)
    assert not (tmp_path / "c").exists()  # nothing was left behind


def test_calibration_inputs_must_describe_the_scores(tmp_path: Path, frozen: Path) -> None:
    for inputs in ({**INPUTS, "n": 19}, {**INPUTS, "group": "fit"}):
        with pytest.raises(ArtifactError, match="calibration inputs"):
            calibrate_artifact(frozen, tmp_path / "c", _calibration(), inputs)
    with pytest.raises(ArtifactError, match="preprocessing"):
        bad = {**INPUTS, "preprocessing_version": "0"}
        calibrate_artifact(frozen, tmp_path / "c", _calibration(), bad)


# --- tampering and mismatches are rejected on load -----------------------------------------


def _edit(directory: Path, change) -> None:
    manifest = _read(directory)
    change(manifest)
    _write(directory, manifest)


def _rehash(manifest: dict) -> None:
    """What an editor who also fixes the stored block hash would do."""
    manifest["calibration"]["block_sha256"] = canonical_sha256(manifest["calibration"]["block"])


def test_an_edited_threshold_is_rejected(calibrated: Path) -> None:
    _edit(calibrated, lambda m: m["calibration"]["block"].update(threshold=0.0))
    with pytest.raises(ArtifactError, match="threshold|changed after"):
        load_artifact(calibrated)


def test_an_edited_threshold_with_a_fixed_block_hash_is_still_rejected(calibrated: Path) -> None:
    def change(m: dict) -> None:
        m["calibration"]["block"]["threshold"] = m["calibration"]["block"]["calibration_scores"][0]
        _rehash(m)

    _edit(calibrated, change)
    with pytest.raises(ArtifactError, match="threshold is not the k-th"):
        load_artifact(calibrated)


def test_edited_scores_with_a_fixed_hash_are_rejected_when_inconsistent(calibrated: Path) -> None:
    def change(m: dict) -> None:
        scores = m["calibration"]["block"]["calibration_scores"]
        scores[0], scores[1] = scores[1], scores[0] - 1  # not sorted
        _rehash(m)

    _edit(calibrated, change)
    with pytest.raises(ArtifactError, match="not sorted"):
        load_artifact(calibrated)


def test_a_wrong_alpha_or_k_is_rejected(calibrated: Path) -> None:
    def change(m: dict) -> None:
        m["calibration"]["block"]["alpha"] = 0.5
        _rehash(m)

    _edit(calibrated, change)
    with pytest.raises(ArtifactError):
        load_artifact(calibrated)


def test_a_block_whose_hash_was_not_updated_is_rejected(calibrated: Path) -> None:
    load_artifact(calibrated)  # untouched, it loads
    _edit(calibrated, lambda m: m["calibration"]["block"]["pixel"].update(lower=-5.0))
    with pytest.raises(ArtifactError):
        load_artifact(calibrated)


def test_settings_changed_after_the_freeze_are_rejected(calibrated: Path) -> None:
    def change(m: dict) -> None:
        m["freeze"]["frozen_settings"]["alpha"] = 0.2

    _edit(calibrated, change)
    with pytest.raises(ArtifactError, match="differ from the ones fixed at freeze"):
        load_artifact(calibrated)


def test_a_calibration_copied_to_other_weights_is_rejected(
    tmp_path: Path, calibrated: Path, fitted_dir: Path
) -> None:
    torch.manual_seed(99)
    other = build_model(CONFIG, pre_trained=False)
    fit_memory_bank(other, [torch.rand(4, 3, 96, 96)], CONFIG.coreset_ratio, seed=0)
    save_artifact(other, CONFIG, DEFAULT_CONFIG, tmp_path / "other")
    freeze_artifact(tmp_path / "other", tmp_path / "other_frozen", "1.0.0", SETTINGS)
    # put the first model's calibration block into the other model's frozen manifest
    manifest = _read(tmp_path / "other_frozen")
    manifest["stage"] = "calibrated"
    manifest["calibration"] = copy.deepcopy(_read(calibrated)["calibration"])
    _write(tmp_path / "other_frozen", manifest)
    with pytest.raises(ArtifactError, match="different weights"):
        load_artifact(tmp_path / "other_frozen")


def test_weights_swapped_after_freezing_are_rejected(tmp_path: Path, calibrated: Path) -> None:
    # same hash bookkeeping as an honest fitted artifact, but different weights
    torch.manual_seed(5)
    other = build_model(CONFIG, pre_trained=False)
    fit_memory_bank(other, [torch.rand(4, 3, 96, 96)], CONFIG.coreset_ratio, seed=1)
    save_artifact(other, CONFIG, DEFAULT_CONFIG, tmp_path / "o")
    swapped = _read(tmp_path / "o")["files"][MODEL_FILE]
    shutil.copyfile(tmp_path / "o" / MODEL_FILE, calibrated / MODEL_FILE)
    manifest = _read(calibrated)
    manifest["files"][MODEL_FILE] = swapped
    manifest["tensors"] = _read(tmp_path / "o")["tensors"]
    manifest["memory_bank"] = _read(tmp_path / "o")["memory_bank"]
    _write(calibrated, manifest)
    with pytest.raises(ArtifactError, match="not the ones that were frozen"):
        load_artifact(calibrated)


def test_stage_fields_must_agree(calibrated: Path, frozen: Path, fitted_dir: Path) -> None:
    _edit(calibrated, lambda m: m.pop("calibration"))
    with pytest.raises(ArtifactError, match="needs a 'calibration' block"):
        load_artifact(calibrated)
    _edit(calibrated, lambda m: m.update(stage="finished"))
    with pytest.raises(ArtifactError, match="Unknown artifact stage"):
        load_artifact(calibrated)
    _edit(frozen, lambda m: m.update(calibration={"block": {}}))
    with pytest.raises(ArtifactError, match="has a calibration block"):
        load_artifact(frozen)
    _edit(frozen, lambda m: (m.pop("calibration"), m.pop("stage")))
    with pytest.raises(ArtifactError, match="no 'stage'"):
        load_artifact(frozen)


def test_calibrated_inputs_must_match_the_stored_scores(calibrated: Path) -> None:
    _edit(calibrated, lambda m: m["calibration"]["inputs"].update(n=3))
    with pytest.raises(ArtifactError, match="calibration inputs"):
        load_artifact(calibrated)
