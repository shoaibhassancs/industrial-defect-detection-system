"""Tests for defect_ml.freeze and defect_ml.calibrate (Phase 3 steps 2-3).

A tiny fake dataset and RANDOM weights are used, as in test_fit.py: they test the wiring, the
refusals, the records and the data access, not the quality of a real model. The real runs are
separate. In the fake dataset the official test and mask files are empty, so opening one could not
succeed silently; the audit checks are the main evidence.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest
import yaml
from defect_core.artifacts import ArtifactError, load_artifact
from defect_core.model import infer
from defect_ml import calibrate, fit, freeze
from defect_ml.data.mvtec import TrainingImages, sha256_of
from defect_ml.fit import FitConfig, FitError

_FAKE = Path(__file__).resolve().parent / "test_mvtec.py"
_spec = importlib.util.spec_from_file_location("ml_fake_dataset_freeze_calibrate", _FAKE)
fake = importlib.util.module_from_spec(_spec)
sys.modules["ml_fake_dataset_freeze_calibrate"] = fake
_spec.loader.exec_module(fake)

REPO = Path(__file__).resolve().parents[2]
CALIBRATION = {  # alpha 0.4 because the fake dataset has only 2 calibration images
    "alpha": 0.4, "lower_percentile": 1, "highlight_percentile": 99.5, "upper_multiple": 1.5,
    "upper_multiple_status": "unvalidated display choice",
}  # fmt: skip


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted during test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


def _settings(root: Path) -> dict:
    data, split_dir, manifest = fake.make_dataset(root)
    return {
        "run_name": "test_fit-v0",
        "category": "bottle",
        "model": {
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "num_neighbors": 9,
            "coreset_ratio": 0.1,
        },  # fmt: skip
        "weights": {"pretrained": False},
        "fit": {"seed": 0, "batch_size": 3, "torch_threads": 2},
        "paths": {
            "data_dir": str(data),
            "split_dir": str(split_dir),
            "manifest": str(manifest),
            "artifact_dir": str(root / "artifacts"),
            "record_dir": str(root / "records"),
        },  # fmt: skip
        "guard": {"max_swap_used_mb": 1e9, "min_free_disk_gib": 0},
        "calibration": CALIBRATION,
    }


@pytest.fixture
def fitted(tmp_path: Path) -> tuple[FitConfig, Path]:
    raw = _settings(tmp_path)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = FitConfig.from_yaml(path)
    assert fit.run(config, allow_random_weights=True, repo=REPO) == 0
    return config, path


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_of(p) for p in sorted(root.rglob("*")) if p.is_file()}


# --- freeze ---------------------------------------------------------------------------------


def test_freeze_makes_a_new_folder_opens_no_dataset_file_and_changes_nothing_else(
    fitted: tuple[FitConfig, Path],
) -> None:
    config, path = fitted
    before = _tree_hashes(Path(config.artifact_dir)) | _tree_hashes(Path(config.record_dir))
    assert freeze.run(config, path) == 0
    paths = freeze.stage_paths(config, "1.0.0")
    after = _tree_hashes(Path(config.artifact_dir)) | _tree_hashes(Path(config.record_dir))
    assert {k: v for k, v in after.items() if k in before} == before  # nothing old changed
    assert paths["frozen"].name == "test_frozen-v1.0.0"

    loaded = load_artifact(paths["frozen"])
    assert loaded.manifest["stage"] == "frozen" and loaded.calibration is None
    freeze_block = loaded.manifest["freeze"]
    assert freeze_block["frozen_settings"]["alpha"] == 0.4
    assert freeze_block["frozen_settings"]["config_sha256"] == sha256_of(path)
    assert freeze_block["source_fit_record_sha256"] == sha256_of(paths["fit_record"])

    record = json.loads(paths["freeze_record"].read_text())
    assert record["data_access"]["dataset_files_opened"] == 0
    assert record["data_access"]["directory_listings_inside_dataset_folder"] == 0
    assert record["weights_identical_to_fitted"] is True
    assert "NOT calibrated" in record["status"] and "no test image" in record["status"]


def test_freeze_never_overwrites(fitted: tuple[FitConfig, Path]) -> None:
    config, path = fitted
    assert freeze.run(config, path) == 0
    assert freeze.run(config, path) == 1


def test_freeze_refuses_when_the_fit_record_does_not_belong_to_the_artifact(
    fitted: tuple[FitConfig, Path],
) -> None:
    config, path = fitted
    record_path = freeze.stage_paths(config, "1.0.0")["fit_record"]
    record = json.loads(record_path.read_text())
    record["manifest_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(FitError, match="manifest differs"):
        freeze.run(config, path)


def test_freeze_refuses_when_the_model_settings_changed_since_the_fit(
    fitted: tuple[FitConfig, Path],
) -> None:
    config, path = fitted
    raw = copy.deepcopy(config.raw)
    raw["model"]["num_neighbors"] = 5
    with pytest.raises(FitError, match="model settings"):
        freeze.run(FitConfig.from_dict(raw), path)


def test_freeze_needs_a_fit_run_name() -> None:
    with pytest.raises(FitError, match="must end with"):
        freeze.base_name("something_else")


# --- calibrate ------------------------------------------------------------------------------


def test_calibrate_scores_only_the_calib_files_and_matches_an_independent_scoring(
    fitted: tuple[FitConfig, Path],
) -> None:
    config, path = fitted
    assert freeze.run(config, path) == 0
    paths = freeze.stage_paths(config, "1.0.0")
    before = _tree_hashes(paths["frozen"]) | _tree_hashes(paths["fitted"])

    assert calibrate.run(config, path) == 0
    assert _tree_hashes(paths["frozen"]) | _tree_hashes(paths["fitted"]) == before

    record = json.loads(paths["calibration_record"].read_text())
    access = record["data_access"]
    assert access["group_read"] == "calib" and access["images_loaded"] == 2
    assert access["files_opened_inside_dataset_folder"] == 2
    assert access["exactly_the_listed_calib_files_each_once"] is True
    assert access["directory_listings_inside_dataset_folder"] == 0
    assert "NOT evaluated" in record["status"] and "no test image" in record["status"]
    assert record["calibrated_manifest_reloads_and_matches"] is True
    assert record["inputs"]["batch_size"] == 1

    loaded = load_artifact(paths["calibrated"])
    calibration = loaded.calibration
    assert calibration is not None and calibration.n == 2 and calibration.k == 2
    assert record["calibration"]["threshold"] == calibration.threshold

    # score the same two images again, independently, with the frozen model
    frozen = load_artifact(paths["frozen"])
    images = TrainingImages.open(
        config.data_dir, "bottle", "calib",
        split_dir=config.split_dir, manifest_path=config.manifest,
    )  # fmt: skip
    again = sorted(float(infer(frozen.model, b)[0][0]) for b in images.batches(1))
    assert list(calibration.calibration_scores) == again
    assert calibration.threshold == again[calibration.k - 1]


def test_calibrate_refuses_an_unfrozen_run_and_a_second_calibration(
    fitted: tuple[FitConfig, Path],
) -> None:
    config, path = fitted
    with pytest.raises(ArtifactError, match="not found|Manifest"):
        calibrate.run(config, path)  # nothing frozen yet
    assert freeze.run(config, path) == 0
    assert calibrate.run(config, path) == 0
    assert calibrate.run(config, path) == 1  # never overwritten


def test_calibrate_refuses_settings_that_differ_from_the_freeze(
    fitted: tuple[FitConfig, Path], tmp_path: Path
) -> None:
    config, path = fitted
    assert freeze.run(config, path) == 0
    raw = copy.deepcopy(config.raw)
    raw["calibration"]["upper_multiple"] = 3.0
    other = FitConfig.from_dict(raw)
    with pytest.raises(FitError, match="differ from the ones fixed at freeze"):
        calibrate.run(other, path)
    assert not freeze.stage_paths(config, "1.0.0")["calibrated"].exists()


def test_calibrate_refuses_when_the_frozen_weights_were_tampered_with(
    fitted: tuple[FitConfig, Path],
) -> None:
    config, path = fitted
    assert freeze.run(config, path) == 0
    weights = freeze.stage_paths(config, "1.0.0")["frozen"] / "model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    weights.write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="SHA-256"):
        calibrate.run(config, path)


def test_a_calibration_block_naming_other_weights_is_rejected_on_load(
    fitted: tuple[FitConfig, Path],
) -> None:
    config, path = fitted
    assert freeze.run(config, path) == 0
    assert calibrate.run(config, path) == 0
    folder = freeze.stage_paths(config, "1.0.0")["calibrated"]
    load_artifact(folder)  # the honest folder loads
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["calibration"]["weights_sha256"] = "0" * 64
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="different weights"):
        load_artifact(folder)
