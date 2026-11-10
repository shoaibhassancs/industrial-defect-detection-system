"""Tests for defect_ml.evaluate (Phase 3 step 6): the sealed evaluation.

A tiny fake dataset with REAL (synthetic) test images and masks, and RANDOM weights: these tests
check the gate, the access control, the log, the records and the wiring, not the quality of a
model. The real evaluation of the real artifacts is a separate, one-time run.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from defect_core.artifacts import ArtifactError, load_artifact
from defect_core.calibration import Calibration, build_calibration
from defect_ml import calibrate, evaluate, fit, freeze, metrics
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import TrainingImages, assert_training_path
from defect_ml.evaluate import (
    AccessLog,
    EvaluationError,
    assert_evaluation_path,
    verify_access_log,
)
from defect_ml.fit import FitConfig
from PIL import Image

_FAKE = Path(__file__).resolve().parent / "test_freeze_calibrate.py"
_spec = importlib.util.spec_from_file_location("ml_fake_freeze_for_evaluate", _FAKE)
stages = importlib.util.module_from_spec(_spec)
sys.modules["ml_fake_freeze_for_evaluate"] = stages
_spec.loader.exec_module(stages)

TYPES = {"crack": 3, "good": 3, "hole": 2}  # images per folder; sorted, as the split file needs


def _png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def add_test_set(root: Path) -> dict[str, list[str]]:
    """Replace the empty placeholder test files with small real images and masks."""
    data, split_dir, manifest = root / "mvtec", root / "splits", root / "data_manifest.json"
    bottle = data / "bottle"
    for folder in ("test", "ground_truth"):
        for child in list((bottle / folder).iterdir()):
            for f in child.iterdir():
                f.unlink()
            child.rmdir()
    listing: dict[str, list[str]] = {}
    rng = np.random.default_rng(1)
    for defect, count in TYPES.items():
        listing[defect] = []
        for i in range(count):
            array = (rng.random((48, 64, 3)) * 80 + 60).astype(np.uint8)
            name = f"{i:03d}.png"
            if defect != "good":
                array[10:24, 20:40] = 250
                mask = np.zeros((48, 64), dtype=np.uint8)
                mask[10:24, 20:40] = 255
                _png(bottle / "ground_truth" / defect / f"{i:03d}_mask.png", Image.fromarray(mask))
            _png(bottle / "test" / defect / name, Image.fromarray(array))
            listing[defect].append(f"test/{defect}/{name}")
    doc = json.loads((split_dir / "bottle.json").read_text())
    doc["test_files_not_opened"] = listing
    doc["counts"]["test_listed_by_name_only"] = sum(len(v) for v in listing.values())
    (split_dir / "bottle.json").write_text(json.dumps(doc))
    m = json.loads(manifest.read_text())
    m["categories"] = {"bottle": {"test_good": TYPES["good"], "test_defective": 5, "masks": 5}}
    manifest.write_text(json.dumps(m))
    return listing


@pytest.fixture
def calibrated(tmp_path: Path):
    """A calibrated random-weight artifact, its settings file and a fake dataset with test files."""
    raw = stages._settings(tmp_path)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = FitConfig.from_yaml(path)
    assert fit.run(config, allow_random_weights=True, repo=Path.cwd()) == 0
    assert freeze.run(config, path) == 0
    assert calibrate.run(config, path) == 0
    listing = add_test_set(tmp_path)
    return config, path, tmp_path / "evaluation", listing


def _log_lines(root: Path) -> list[dict]:
    path = root / evaluate.LOG_NAME
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# --- a complete evaluation ------------------------------------------------------------------


def test_an_evaluation_reads_only_the_listed_files_logs_each_and_writes_the_results(
    calibrated,
) -> None:
    config, path, root, listing = calibrated
    (folder,) = evaluate.run_batch([path], eval_root=root)
    assert folder == root / "test" / "v1.0.0" / "bottle"  # <model>/v<version>/<category>

    record = json.loads((folder / "metrics.json").read_text())
    access = record["data_access"]
    assert access["images_read"] == 8 and access["masks_read"] == 5
    assert access["files_opened_inside_dataset_folder"] == 13
    assert access["exactly_the_listed_files_each_once"] is True
    assert access["only_test_and_ground_truth_opened"] is True
    assert access["directory_listings_inside_dataset_folder"] == 0
    assert "nothing tuned" in record["status"] and record["headline"] is False
    assert "256" in record["results"]["pixel"]["resolution"]

    # the log: start, 13 opens, finish; chain intact; paths are the listed files only
    lines = _log_lines(root)
    assert [x["event"] for x in lines] == ["start"] + ["open"] * 13 + ["finish"]
    assert verify_access_log(root / evaluate.LOG_NAME) == []
    opened = {x["path"] for x in lines if x["event"] == "open"}
    expected = {f"bottle/{p}" for v in listing.values() for p in v}
    for kind in ("crack", "hole"):
        expected |= {f"bottle/ground_truth/{kind}/{i:03d}_mask.png" for i in range(TYPES[kind])}
    assert opened == expected and not any("train" in p for p in opened)
    assert all(len(x["sha256"]) == 64 for x in lines if x["event"] == "open")
    start = lines[0]
    assert start["frozen_at_utc"] <= start["time_utc"] <= lines[1]["time_utc"]
    assert lines[0]["weights_sha256"] == record["weights_sha256"]

    # predictions: the verdict is the shared strict rule; ranks come from the calibration
    calibration = load_artifact(freeze.stage_paths(config, "1.0.0")["calibrated"]).calibration
    rows = list(csv.DictReader((folder / "predictions.csv").open()))
    assert len(rows) == 8 and rows[0]["defect_type"] == "good"
    for row in rows:
        score = float(row["score"])
        assert int(row["flagged"]) == int(score > calibration.threshold)
        assert int(row["normal_rank"]) == calibration.normal_rank(score)
        assert int(row["label"]) == int(row["defect_type"] != "good")

    # metrics.json agrees with the metrics module run directly on the saved predictions
    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    assert record["results"]["image"]["auroc"]["estimate"] == metrics.image_auroc(labels, scores)
    at = record["results"]["image"]["at_calibrated_threshold"]
    assert (at["tp"] + at["fn"], at["fp"] + at["tn"]) == (5, 3)
    alarms = record["results"]["image"]["false_alarm_on_normal_test_images"]
    assert alarms["alpha"] == calibration.alpha and alarms["n_normal"] == 3
    assert set(record["results"]["image"]["per_defect_recall"]) == {"crack", "hole"}
    assert set(json.loads((folder / "curves.json").read_text())) >= {"image_roc", "pixel_pro"}
    assert len(json.loads((folder / "histograms.json").read_text())["bin_edges"]) == 21
    assert len(record["mask_pixel_counts"]) == 5


def test_the_headline_flag_follows_the_adr(calibrated) -> None:
    config, path, root, _ = calibrated
    assert evaluate.HEADLINE_BACKBONE == "wide_resnet50_2.tv_in1k"
    assert config.model.backbone != evaluate.HEADLINE_BACKBONE


# --- the gate: nothing is opened unless every condition holds --------------------------------


def _assert_nothing_was_opened(root: Path, data_dir: Path, audit: OpenAudit) -> None:
    opened, listed = audit.inside(data_dir)
    assert not [p for p in opened if "/test/" in str(p) or "/ground_truth/" in str(p)]
    assert not listed
    assert not (root / evaluate.LOG_NAME).exists() or _log_lines(root) == []


def test_a_fitted_or_frozen_artifact_is_refused_before_any_test_file_is_opened(
    tmp_path: Path,
) -> None:
    raw = stages._settings(tmp_path)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = FitConfig.from_yaml(path)
    assert fit.run(config, allow_random_weights=True, repo=Path.cwd()) == 0
    assert freeze.run(config, path) == 0  # frozen, not calibrated
    add_test_set(tmp_path)
    root = tmp_path / "evaluation"
    with OpenAudit() as audit:
        with pytest.raises(ArtifactError):  # there is no calibrated folder at all
            evaluate.run_batch([path], eval_root=root)
    _assert_nothing_was_opened(root, Path(config.data_dir), audit)


def test_an_existing_result_is_never_overwritten_and_the_log_is_unchanged(calibrated) -> None:
    _, path, root, _ = calibrated
    evaluate.run_batch([path], eval_root=root)
    before = (root / evaluate.LOG_NAME).read_bytes()
    with pytest.raises(EvaluationError, match="one evaluation per version"):
        evaluate.run_batch([path], eval_root=root)
    assert (root / evaluate.LOG_NAME).read_bytes() == before


def test_a_version_that_does_not_match_the_artifact_is_refused(calibrated) -> None:
    config, path, root, _ = calibrated
    calibrated_dir = freeze.stage_paths(config, "1.0.0")["calibrated"]
    other = freeze.stage_paths(config, "2.0.0")["calibrated"]
    calibrated_dir.rename(other)  # same files under another version name
    with pytest.raises(EvaluationError, match="not 2.0.0"):
        evaluate.run_batch([path], version="2.0.0", eval_root=root)
    assert _log_lines(root) == []


def test_a_tight_disk_stops_the_batch_before_any_test_file_is_opened(
    calibrated, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path, root, _ = calibrated
    monkeypatch.setattr(evaluate.runinfo, "free_disk_gib", lambda: 6.9)
    raw = dict(config.raw)
    raw["guard"] = {"max_swap_used_mb": 1e9, "min_free_disk_gib": 7}
    tight = path.with_name("tight.yaml")
    tight.write_text(yaml.safe_dump(raw))
    with OpenAudit() as audit:
        with pytest.raises(EvaluationError, match="free disk"):
            evaluate.run_batch([tight], eval_root=root)
    _assert_nothing_was_opened(root, Path(config.data_dir), audit)
    monkeypatch.setattr(evaluate.runinfo, "free_disk_gib", lambda: 7.0)  # at the guard: also no
    with pytest.raises(EvaluationError, match="free disk"):
        evaluate.run_batch([tight], eval_root=root)


def test_one_bad_settings_file_stops_the_whole_batch_before_the_first_model_runs(
    calibrated,
) -> None:
    config, path, root, _ = calibrated
    raw = dict(config.raw)
    raw["run_name"] = "other_fit-v0"  # no artifact exists for it
    bad = path.with_name("bad.yaml")
    bad.write_text(yaml.safe_dump(raw))
    with OpenAudit() as audit:
        with pytest.raises(ArtifactError):
            evaluate.run_batch([path, bad], eval_root=root)
    _assert_nothing_was_opened(root, Path(config.data_dir), audit)
    assert not (root / "test").exists()
    with pytest.raises(EvaluationError, match="listed twice"):
        evaluate.run_batch([path, path], eval_root=root)


def test_changed_settings_or_counts_are_refused(calibrated) -> None:
    config, path, root, _ = calibrated
    raw = dict(config.raw)
    raw["calibration"] = {**raw["calibration"], "upper_multiple": 2.0}
    other = path.with_name("other.yaml")
    other.write_text(yaml.safe_dump(raw))
    with pytest.raises(EvaluationError, match="differs from the frozen settings"):
        evaluate.run_batch([other], eval_root=root)
    manifest = Path(config.manifest)
    doc = json.loads(manifest.read_text())
    doc["categories"]["bottle"]["test_good"] = 99
    manifest.write_text(json.dumps(doc))
    with pytest.raises(EvaluationError, match="verified counts"):
        evaluate.run_batch([path], eval_root=root)
    assert _log_lines(root) == []


def test_a_mask_that_does_not_fit_its_image_stops_the_run_and_writes_no_results(
    calibrated,
) -> None:
    config, path, root, _ = calibrated
    mask = Path(config.data_dir) / "bottle" / "ground_truth" / "crack" / "000_mask.png"
    Image.new("L", (10, 10)).save(mask)
    with pytest.raises(EvaluationError, match="mask size"):
        evaluate.run_batch([path], eval_root=root)
    assert not (root / "test").exists()  # no result folder
    assert _log_lines(root)[-1]["event"] != "finish"  # the log shows an unfinished run


# --- the evaluation's own path check, and the training guard -------------------------------


def test_the_evaluation_path_check_allows_only_test_and_ground_truth_files(tmp_path: Path) -> None:
    data = tmp_path / "mvtec"
    bottle = data / "bottle"
    assert_evaluation_path(data, bottle / "test" / "crack" / "000.png")
    assert_evaluation_path(data, bottle / "ground_truth" / "crack" / "000_mask.png")
    for forbidden in (
        bottle / "train" / "good" / "000.png",
        bottle / "test" / "000.png",
        bottle / "test" / "crack" / "sub" / "000.png",
        bottle / "test" / ".." / "train" / "good" / "000.png",
        bottle / "readme.txt",
        tmp_path / "elsewhere" / "bottle" / "test" / "crack" / "000.png",
    ):
        with pytest.raises(PermissionError):
            assert_evaluation_path(data, forbidden)


def test_the_training_guard_and_loader_still_refuse_the_test_files(calibrated) -> None:
    config, _, _, _ = calibrated
    data = Path(config.data_dir)
    with pytest.raises(PermissionError):
        assert_training_path(data, data / "bottle" / "test" / "crack" / "000.png")
    with pytest.raises(PermissionError):
        assert_training_path(data, data / "bottle" / "ground_truth" / "crack" / "000_mask.png")
    images = TrainingImages.open(
        data, "bottle", "calib", split_dir=config.split_dir, manifest_path=config.manifest
    )
    with pytest.raises(KeyError):
        images.load("test/crack/000.png")


def test_the_reader_refuses_unlisted_files(calibrated) -> None:
    config, _, root, listing = calibrated
    reader = evaluate.SealedReader(
        Path(config.data_dir).resolve(), "bottle", AccessLog(root / evaluate.LOG_NAME), {}
    )
    reader.allow(listing["good"])
    reader.read(listing["good"][0], "image")
    for refused in ("train/good/000.png", listing["crack"][0]):
        with pytest.raises(PermissionError, match="not one of the listed"):
            reader.read(refused, "image")
    assert len(_log_lines(root)) == 1  # only the permitted read was logged


# --- the log --------------------------------------------------------------------------------


def test_a_changed_removed_or_reordered_log_line_is_detected(tmp_path: Path) -> None:
    log = AccessLog(tmp_path / "log.jsonl")
    for i in range(4):
        log.append("open", path=f"bottle/test/good/{i}.png")
    path = tmp_path / "log.jsonl"
    assert verify_access_log(path) == []
    original = path.read_text().splitlines()

    edited = original[1].replace("1.png", "9.png")
    path.write_text("\n".join([original[0], edited, *original[2:]]) + "\n")
    assert verify_access_log(path)  # changed content breaks the next line's link
    path.write_text("\n".join([original[0], *original[2:]]) + "\n")
    assert "does not follow" in verify_access_log(path)[0]  # removed line
    path.write_text("\n".join([original[0], original[2], original[1], original[3]]) + "\n")
    assert verify_access_log(path)  # reordered
    path.write_text("\n".join(original[:3]) + "\n")
    assert verify_access_log(path) == []  # a shortened tail is not detectable by the chain alone
    path.write_text("not json\n")
    assert "not valid JSON" in verify_access_log(path)[0]
    with pytest.raises(EvaluationError, match="damaged"):
        AccessLog(path)


def test_appending_continues_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    AccessLog(path).append("open", path="a")
    reopened = AccessLog(path)
    assert reopened.lines == 1
    reopened.append("open", path="b")
    assert verify_access_log(path) == [] and len(path.read_text().splitlines()) == 2


# --- the pure metrics assembly --------------------------------------------------------------


def test_compute_metrics_is_pure_and_agrees_with_the_metrics_module() -> None:
    rng = np.random.default_rng(3)
    labels = np.array([0] * 6 + [1] * 6)
    types = ["good"] * 6 + ["a"] * 3 + ["b"] * 3
    scores = np.concatenate([rng.normal(1, 0.1, 6), rng.normal(1.5, 0.2, 6)])
    masks = np.zeros((12, 16, 16), dtype=bool)
    masks[6:, 4:9, 4:9] = True
    maps = rng.random((12, 16, 16)) * 0.3 + masks * 0.5
    built = build_calibration(
        rng.normal(1, 0.1, 19), [rng.random((16, 16)) for _ in range(19)],
        alpha=0.1, lower_percentile=1, highlight_percentile=99.5, upper_multiple=1.5,
    )  # fmt: skip
    calibration = Calibration.from_dict(json.loads(json.dumps(built.to_dict())))
    results, curves, hist = evaluate.compute_metrics(
        labels, types, scores, masks, maps, calibration
    )
    assert results["image"]["auroc"]["estimate"] == metrics.image_auroc(labels, scores)
    assert results["pixel"]["auroc"] == metrics.pixel_auroc(masks, maps)
    assert results["pixel"]["aupro_fpr_0.3"] == metrics.aupro(masks, maps, 0.3)
    assert results["pixel"]["n_anomalous_pixels"] == int(masks.sum())
    assert hist["calibrated_threshold"] == calibration.threshold
    assert sum(hist["normal"]) == 6 and sum(hist["anomalous"]) == 6
    assert len(curves["pixel_pro"]["fpr"]) <= evaluate.CURVE_POINTS
