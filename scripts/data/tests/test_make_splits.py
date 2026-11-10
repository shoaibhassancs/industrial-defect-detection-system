"""Tests for scripts/data/make_splits.py, using a tiny fake dataset (no real data needed)."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from defect_ml.data.splits import n_calib_for

SCRIPT = Path(__file__).resolve().parents[1] / "make_splits.py"
spec = importlib.util.spec_from_file_location("make_splits", SCRIPT)
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_dataset(root: Path, n_train: int = 20) -> Path:
    """Empty files only: the script lists names and must never open anything."""
    data = root / "mvtec"
    train = data / "bottle" / "train" / "good"
    train.mkdir(parents=True)
    for i in range(n_train):
        (train / f"{i:03d}.png").write_bytes(b"")
    (train / ".DS_Store").write_bytes(b"x")  # hidden file must be ignored
    # Test files deliberately share names with training files (as in the real dataset).
    for defect, count in (("good", 3), ("crack", 2)):
        folder = data / "bottle" / "test" / defect
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f"{i:03d}.png").write_bytes(b"")
    return data


def args_for(data: Path, tmp: Path, **extra) -> argparse.Namespace:
    values = {
        "data_dir": str(data),
        "category": "bottle",
        "seed": 0,
        "out_dir": str(tmp / "splits"),
        "manifest": str(tmp / "no_manifest.json"),
        "force": False,
    }
    values.update(extra)
    return argparse.Namespace(**values)


def test_rounding_matches_the_plan() -> None:
    assert n_calib_for(209) == 42  # bottle: 167 fit + 42 calib
    assert n_calib_for(10) == 2
    assert n_calib_for(60) == 12
    assert n_calib_for(0) == 0


def test_split_is_a_valid_partition_and_deterministic(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    first = ms.build_split(data, "bottle", 0, tmp_path / "none.json")
    second = ms.build_split(data, "bottle", 0, tmp_path / "none.json")
    assert first == second
    assert ms.validate_split(first) == []
    assert first["counts"] == {
        "train_good_total": 20,
        "fit": 16,
        "calib": 4,
        "test_listed_by_name_only": 5,
    }
    assert not any(p.endswith(".DS_Store") for p in first["fit"] + first["calib"])


def test_seed_and_category_change_the_split() -> None:
    names = [f"{i:03d}.png" for i in range(60)]
    base = ms.assign(names, 0, "bottle")
    assert ms.assign(names, 1, "bottle") != base
    assert ms.assign(names, 0, "cable") != base


def test_same_file_names_in_train_and_test_do_not_count_as_overlap(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    doc = ms.build_split(data, "bottle", 0, tmp_path / "none.json")
    names_in_both = {Path(p).name for p in doc["fit"] + doc["calib"]} & {
        Path(p).name for paths in doc["test_files_not_opened"].values() for p in paths
    }
    assert names_in_both  # the fake data really does reuse names, as MVTec does
    assert ms.validate_split(doc) == []


def test_validation_detects_problems(tmp_path: Path) -> None:
    doc = ms.build_split(make_dataset(tmp_path), "bottle", 0, tmp_path / "none.json")

    overlap = copy.deepcopy(doc)
    overlap["calib"] = sorted(overlap["calib"] + overlap["fit"][:1])
    assert any("overlap" in p for p in ms.validate_split(overlap))

    leaked = copy.deepcopy(doc)
    leaked["fit"] = sorted(leaked["fit"] + ["test/good/000.png"])
    assert any("test file appears" in p for p in ms.validate_split(leaked))

    tampered = copy.deepcopy(doc)
    tampered["train_files_sha256"] = "0" * 64
    assert any("sha256" in p for p in ms.validate_split(tampered))

    swapped = copy.deepcopy(doc)  # a valid-looking partition that is not the stated method's
    swapped["fit"][0], swapped["calib"][0] = swapped["calib"][0], swapped["fit"][0]
    swapped["fit"], swapped["calib"] = sorted(swapped["fit"]), sorted(swapped["calib"])
    assert any("stated method" in p for p in ms.validate_split(swapped))


def test_existing_split_is_never_changed_silently(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    args = args_for(data, tmp_path)
    assert ms.run(args) == 0
    out = tmp_path / "splits" / "bottle.json"
    original = out.read_text()
    assert ms.run(args) == 0  # same input, same file
    assert out.read_text() == original

    (data / "bottle" / "train" / "good" / "999.png").write_bytes(b"")  # dataset changed
    assert ms.run(args) == 1  # refused
    assert out.read_text() == original
    assert ms.run(args_for(data, tmp_path, force=True)) == 0
    assert out.read_text() != original


def test_manifest_cross_check(tmp_path: Path) -> None:
    data = make_dataset(tmp_path)
    names = [f"{i:03d}.png" for i in range(20)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "deep_check": {
                    "bottle": {"images": [{"path": f"bottle/train/good/{n}"} for n in names]}
                }
            }
        )
    )
    doc = ms.build_split(data, "bottle", 0, manifest)
    assert doc["data_manifest_sha256"] is not None

    manifest.write_text(
        json.dumps({"deep_check": {"bottle": {"images": [{"path": "bottle/train/good/000.png"}]}}})
    )
    with pytest.raises(ValueError, match="differs from the verified manifest"):
        ms.build_split(data, "bottle", 0, manifest)


def test_committed_bottle_split_is_consistent() -> None:
    """The committed split list must be self-consistent (no dataset needed)."""
    doc = json.loads((REPO_ROOT / "splits" / "bottle.json").read_text())
    assert ms.validate_split(doc) == []
    assert doc["category"] == "bottle" and doc["seed"] == 0
    assert doc["counts"]["train_good_total"] == 209
    assert (doc["counts"]["fit"], doc["counts"]["calib"]) == (167, 42)
    assert doc["counts"]["test_listed_by_name_only"] == 83  # 20 good + 63 defective
