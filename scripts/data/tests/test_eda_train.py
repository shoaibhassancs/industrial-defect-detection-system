"""Tests for scripts/data/eda_train.py, using a tiny fake dataset (no real data needed)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from defect_ml.data.mvtec import assert_training_path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))  # the scripts import each other by name

import eda_train as eda  # noqa: E402
import make_splits  # noqa: E402
import verify_dataset  # noqa: E402


def make_dataset(root: Path, n_train: int = 20, bright_index: int | None = None) -> Path:
    """Tiny PNG training images with slightly different brightness; EMPTY test files.

    The test files are not valid images, so any attempt to open one would fail the run.
    """
    data = root / "mvtec"
    train = data / "bottle" / "train" / "good"
    train.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_train):
        base = 250 if i == bright_index else 90 + i
        pixels = np.clip(rng.normal(base, 6, size=(12, 16, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(train / f"{i:03d}.png")
    for defect, count in (("good", 3), ("crack", 2)):
        folder = data / "bottle" / "test" / defect
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f"{i:03d}.png").write_bytes(b"")
    return data


def write_split(data: Path, tmp: Path) -> Path:
    split_dir = tmp / "splits"
    split_dir.mkdir(exist_ok=True)
    doc = make_splits.build_split(data, "bottle", 0, tmp / "none.json")
    (split_dir / "bottle.json").write_text(json.dumps(doc, indent=1) + "\n")
    return split_dir


def args_for(data: Path, tmp: Path, **extra) -> argparse.Namespace:
    values = {
        "data_dir": str(data),
        "category": "bottle",
        "split_dir": str(tmp / "splits"),
        "manifest": str(tmp / "no_manifest.json"),
        "out": str(tmp / "eda.json"),
        "figures_dir": str(tmp / "figs"),
    }
    values.update(extra)
    return argparse.Namespace(**values)


def test_eda_runs_and_writes_outputs(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    write_split(data, tmp_path)
    assert eda.run(args_for(data, tmp_path)) == 0
    out = json.loads((tmp_path / "eda.json").read_text())
    assert out["files_opened"] == {"train_good": 20, "any_other_folder": 0}
    assert out["images"]["fit"] + out["images"]["calib"] == 20
    assert out["images"]["sizes"] == {"16x12": 20}
    assert out["inputs_match_verified_manifest"] is None  # no manifest was given
    for metric in eda.METRICS:
        assert {"fit", "calib", "ks_statistic", "ks_p_value"} <= set(out["metrics"][metric])
    assert "not a proof" in out["fit_vs_calib_summary"]["caution"]
    for name in ("fit_sample_sheet", "calib_sample_sheet", "fit_vs_calib_histograms"):
        assert (tmp_path / "figs" / f"bottle_{name}.png").is_file()


def test_output_is_deterministic(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    write_split(data, tmp_path)
    eda.run(args_for(data, tmp_path))
    first = (tmp_path / "eda.json").read_bytes()
    eda.run(args_for(data, tmp_path))
    assert (tmp_path / "eda.json").read_bytes() == first


def test_unusually_bright_image_is_flagged(tmp_path: Path) -> None:
    data = make_dataset(tmp_path, bright_index=7)
    write_split(data, tmp_path)
    eda.run(args_for(data, tmp_path))
    out = json.loads((tmp_path / "eda.json").read_text())
    flagged = [o["file"] for o in out["metrics"]["luma_mean"]["outliers_all_209"]]
    assert "train/good/007.png" in flagged


def test_a_split_that_contains_a_test_file_is_refused(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    split_dir = write_split(data, tmp_path)
    doc = json.loads((split_dir / "bottle.json").read_text())
    doc["fit"] = sorted(doc["fit"] + ["test/good/000.png"])
    (split_dir / "bottle.json").write_text(json.dumps(doc))
    assert eda.run(args_for(data, tmp_path)) == 1
    assert not (tmp_path / "eda.json").exists()  # nothing was analysed or written


def test_the_guard_blocks_paths_outside_train_good(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    for forbidden in ("test/good/000.png", "ground_truth/crack/000_mask.png"):
        try:
            assert_training_path(data, data / "bottle" / forbidden)
        except PermissionError:
            continue
        raise AssertionError(f"guard allowed {forbidden}")


def test_files_that_differ_from_the_verified_manifest_are_refused(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    write_split(data, tmp_path)
    train = data / "bottle" / "train" / "good"
    images = [
        {"path": f"bottle/train/good/{p.name}", "sha256": verify_dataset.sha256_of(p)}
        for p in sorted(train.iterdir())
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"deep_check": {"bottle": {"images": images}}}))
    assert eda.run(args_for(data, tmp_path, manifest=str(manifest))) == 0
    assert json.loads((tmp_path / "eda.json").read_text())["inputs_match_verified_manifest"] is True

    images[3]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"deep_check": {"bottle": {"images": images}}}))
    assert eda.run(args_for(data, tmp_path, manifest=str(manifest))) == 1


def test_script_source_never_names_the_sealed_folders() -> None:
    """The plan's grep check: the exploration code must not mention test or mask folders."""
    source = (SCRIPT_DIR / "eda_train.py").read_text()
    assert "ground_truth" not in source
    assert not re.search(r"[\"'/]test[\"'/]", source)
