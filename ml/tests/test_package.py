"""Tests for defect_ml.package (Phase 4): packaging, the notice, the parity check on the calibration
images, and the refusals. Fake dataset and random weights, as in test_evaluate.py; the sealed test
files are never opened here (only the evaluation fixture reads them, as in its own tests)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from defect_core.artifacts import ArtifactError, load_artifact
from defect_ml import evaluate, freeze, package
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import sha256_of
from defect_ml.evaluate import EvaluationError

_SRC = Path(__file__).resolve().parent / "test_evaluate.py"
_spec = importlib.util.spec_from_file_location("ml_test_evaluate_for_package", _SRC)
ev = importlib.util.module_from_spec(_spec)
sys.modules["ml_test_evaluate_for_package"] = ev
_spec.loader.exec_module(ev)
calibrated = ev.calibrated


def _evaluated(calibrated):
    config, path, root, _ = calibrated
    (result,) = evaluate.run_batch([path], eval_root=root)
    return config, path, root, result


def test_packaging_makes_a_verified_release_and_proves_parity_without_the_test_set(
    calibrated,
) -> None:
    config, path, root, result = _evaluated(calibrated)
    paths = freeze.stage_paths(config, "1.0.0")
    calibrated_before = {p.name: sha256_of(p) for p in paths["calibrated"].iterdir()}
    log_before = (root / evaluate.LOG_NAME).read_bytes()

    with OpenAudit() as audit:
        assert package.run(config, eval_root=root) == 0
    opened, _ = audit.inside(Path(config.data_dir))
    assert opened and all("/train/good/" in str(p) for p in opened)  # only calibration images
    assert (root / evaluate.LOG_NAME).read_bytes() == log_before  # the sealed log was not touched
    assert {p.name: sha256_of(p) for p in paths["calibrated"].iterdir()} == calibrated_before

    release = package.release_path(config, "1.0.0")
    assert release.name == "test_v1.0.0_release"
    loaded = load_artifact(release)
    manifest = loaded.manifest
    assert manifest["stage"] == "packaged" and loaded.calibration is not None
    block = manifest["package"]
    assert block["evaluation"]["metrics_sha256"] == sha256_of(result / "metrics.json")
    saved = json.loads((result / "metrics.json").read_text())
    assert block["evaluation"]["results"] == saved["results"]
    assert block["evaluation"]["weights_sha256"] == manifest["files"]["model.safetensors"]["sha256"]
    assert set(block["data"]) == {"data_manifest_sha256", "split_file_sha256", "dataset"}
    assert block["created_utc"].endswith("Z") and block["category"] == "bottle"
    assert set(block["git"]) == {"commit", "uncommitted_changes"}

    notice = (release / "NOTICE.md").read_text()
    for must_say in (
        "CC BY-NC-SA 4.0", "MVTec Software GmbH", "non-commercial", "not a legal finding",
        "unconfirmed", "not verified", "No endorsement", "bottle", config.model.backbone,
    ):  # fmt: skip
        assert must_say.lower() in notice.lower()

    record = json.loads((Path(config.record_dir) / "package_test_v1.0.0.json").read_text())
    parity = record["parity_on_calib_images"]
    assert parity["sorted_scores_equal_stored_calibration_scores"] is True
    assert parity["only_listed_calib_files_opened"] is True and parity["test_set_opened"] is False
    assert record["weights_identical_to_calibrated"] is True and "NOT uploaded" in record["status"]


def test_a_second_package_is_refused(calibrated) -> None:
    config, path, root, _ = _evaluated(calibrated)
    assert package.run(config, eval_root=root) == 0
    assert package.run(config, eval_root=root) == 1


def test_no_evaluation_or_a_mismatched_one_is_refused_and_nothing_is_written(calibrated) -> None:
    config, path, root, _ = calibrated
    with pytest.raises(EvaluationError, match="no finished evaluation"):
        package.run(config, eval_root=root)
    assert not package.release_path(config, "1.0.0").exists()

    (result,) = evaluate.run_batch([path], eval_root=root)
    record = json.loads((result / "metrics.json").read_text())
    record["weights_sha256"] = "0" * 64  # an evaluation of other weights
    (result / "metrics.json").write_text(json.dumps(record))
    with pytest.raises(ArtifactError, match="not for this exact calibrated"):
        package.run(config, eval_root=root)
    assert not package.release_path(config, "1.0.0").exists()


def test_a_tight_disk_stops_before_anything_is_written(calibrated, monkeypatch) -> None:
    config, path, root, _ = _evaluated(calibrated)
    monkeypatch.setattr(package.runinfo, "free_disk_gib", lambda: 7.0)
    raw = dict(config.raw)
    raw["guard"] = {"max_swap_used_mb": 1e9, "min_free_disk_gib": 7}
    tight = package.FitConfig.from_dict(raw)
    assert package.run(tight, eval_root=root) == 1
    assert not package.release_path(config, "1.0.0").exists()
