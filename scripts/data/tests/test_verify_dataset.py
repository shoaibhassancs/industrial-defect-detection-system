"""Tests for scripts/data/verify_dataset.py, using a tiny fake dataset (no real data needed)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
from defect_ml.data.mvtec import assert_training_path, sha256_of
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "verify_dataset.py"
spec = importlib.util.spec_from_file_location("verify_dataset", SCRIPT)
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)


def make_png(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (color, 0, 0)).save(path)


def make_dataset(root: Path) -> Path:
    """bottle: 3 train images, 2 good test files, 2 'crack' test images with masks.

    The test and mask files are EMPTY (not valid images): if the script ever opened one, the
    run would report an error.
    """
    data = root / "mvtec"
    for i in range(3):
        make_png(data / "bottle" / "train" / "good" / f"{i:03d}.png", 10 * (i + 1))
    for name in ("000.png", "001.png"):
        (data / "bottle" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (data / "bottle" / "test" / "good" / name).write_bytes(b"")
    for i in range(2):
        (data / "bottle" / "test" / "crack").mkdir(parents=True, exist_ok=True)
        (data / "bottle" / "test" / "crack" / f"{i:03d}.png").write_bytes(b"")
        (data / "bottle" / "ground_truth" / "crack").mkdir(parents=True, exist_ok=True)
        (data / "bottle" / "ground_truth" / "crack" / f"{i:03d}_mask.png").write_bytes(b"")
    (data / "bottle" / ".DS_Store").write_bytes(b"x")  # hidden file must be ignored
    # The script expects all 15 categories, so the other 14 exist here as empty folder skeletons.
    for other in vd.CATEGORIES[1:]:
        for sub in ("train/good", "test", "ground_truth"):
            (data / other / sub).mkdir(parents=True, exist_ok=True)
    return data


def args_for(data: Path, tmp: Path, **extra) -> argparse.Namespace:
    values = {
        "data_dir": str(data),
        "category": "bottle",
        "manifest_out": str(tmp / "manifest.json"),
        "summary_out": str(tmp / "summary.json"),
        "expected": None,
        "source": "test",
    }
    values.update(extra)
    return argparse.Namespace(**values)


def test_counts_and_deep_check_on_clean_dataset(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    assert vd.run(args_for(data, tmp_path)) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    bottle = summary["per_category_counts"]["bottle"]
    assert bottle == {"train_good": 3, "test_good": 2, "test_defective": 2, "masks": 2}
    assert summary["deep_check"]["bottle"]["images_ok"] == 3
    assert summary["deep_check"]["bottle"]["sizes"] == {"8x6": 3}
    assert summary["files_opened"]["test"] == 0 and summary["files_opened"]["ground_truth"] == 0
    assert summary["official_comparison"]["status"] == "NOT RUN"


def test_missing_mask_is_reported_from_filenames_only(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    (data / "bottle" / "ground_truth" / "crack" / "001_mask.png").unlink()
    assert vd.run(args_for(data, tmp_path)) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert any("without a mask" in i for i in summary["issues"]["bottle"])


def test_corrupt_training_image_is_reported(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    (data / "bottle" / "train" / "good" / "001.png").write_bytes(b"not an image")
    assert vd.run(args_for(data, tmp_path)) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["deep_check"]["bottle"]["images_failed"] == 1


def test_guard_refuses_files_outside_train_good(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    assert_training_path(data, data / "bottle" / "train" / "good" / "000.png")  # allowed
    for forbidden in (
        data / "bottle" / "test" / "good" / "000.png",
        data / "bottle" / "ground_truth" / "crack" / "000_mask.png",
    ):
        with pytest.raises(PermissionError):
            assert_training_path(data, forbidden)


def test_the_script_uses_the_single_shared_guard() -> None:
    """The guard is defined once, in defect_ml, and the script uses that same function."""
    assert vd.assert_training_path is assert_training_path
    assert vd.sha256_of is sha256_of
    assert "def assert_training_path" not in SCRIPT.read_text()


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    vd.run(args_for(data, tmp_path))
    first = (tmp_path / "manifest.json").read_bytes()
    vd.run(args_for(data, tmp_path))
    assert (tmp_path / "manifest.json").read_bytes() == first


def run_with_expected(data: Path, tmp: Path, expected: dict) -> tuple[int, dict]:
    path = tmp / "expected.json"
    path.write_text(json.dumps(expected))
    code = vd.run(args_for(data, tmp, expected=str(path)))
    return code, json.loads((tmp / "summary.json").read_text())["official_comparison"]


def test_comparison_passes_and_detects_mismatches(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    good = {"train_good": 3, "test_good": 2, "test_defective": 2}
    code, result = run_with_expected(
        data, tmp_path, {"source": "t", "categories": {"bottle": good}, "totals": good}
    )
    assert code == 0 and result["status"] == "PASS" and result["totals_compared"]

    wrong = {**good, "train_good": 4}
    code, result = run_with_expected(
        data, tmp_path, {"source": "t", "categories": {"bottle": wrong}}
    )
    assert code == 1 and result["status"] == "FAIL"
    assert result["mismatches"] == ["bottle.train_good: expected 4, found 3"]

    code, result = run_with_expected(
        data, tmp_path, {"source": "t", "categories": {"bottle": good}, "totals": wrong}
    )
    assert code == 1 and any(m.startswith("TOTAL.") for m in result["mismatches"])


def test_comparison_rejects_mask_fields(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    bad = {"source": "t", "categories": {"bottle": {"masks": 2}}}
    (tmp_path / "expected.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="cannot compare"):
        vd.run(args_for(data, tmp_path, expected=str(tmp_path / "expected.json")))


def test_shipped_expected_counts_file_is_consistent() -> None:
    """The committed table must cover all 15 categories and add up to its own totals."""
    expected = json.loads((SCRIPT.parent / "expected_counts.json").read_text())
    assert set(expected["categories"]) == set(vd.CATEGORIES)
    assert expected["source_is_official_mvtec"] is True
    assert "mvtec_ad.pdf" in expected["source"]
    # The counts' source (official paper) and the download's source (Kaggle) are separate facts.
    assert "Kaggle" in expected["download_source_of_local_folder"]
    assert "Kaggle" not in expected["source"]
    for field in vd.COMPARABLE_FIELDS:
        assert sum(c[field] for c in expected["categories"].values()) == expected["totals"][field]
    assert all(set(c) == set(vd.COMPARABLE_FIELDS) for c in expected["categories"].values())
