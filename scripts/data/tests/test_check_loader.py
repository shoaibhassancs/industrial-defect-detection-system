"""Tests for scripts/data/check_loader.py on a tiny fake dataset (test and mask files are empty)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_loader.py"
spec = importlib.util.spec_from_file_location("check_loader", SCRIPT)
cl = importlib.util.module_from_spec(spec)
sys.modules["check_loader"] = cl
spec.loader.exec_module(cl)

# Reuse the fake-dataset builder of the ml tests (loaded by path, like the scripts above).
_FAKE = Path(__file__).resolve().parents[3] / "ml" / "tests" / "test_mvtec.py"
_fake_spec = importlib.util.spec_from_file_location("ml_fake_dataset", _FAKE)
fake = importlib.util.module_from_spec(_fake_spec)
sys.modules["ml_fake_dataset"] = fake
_fake_spec.loader.exec_module(fake)


def args_for(root: Path, **extra) -> argparse.Namespace:
    data, split_dir, manifest = fake.make_dataset(root)
    values = {
        "data_dir": str(data), "category": "bottle", "split_dir": str(split_dir),
        "manifest": str(manifest), "out_dir": str(root / "out"), "batch_size": 4,
    }  # fmt: skip
    values.update(extra)
    return argparse.Namespace(**values)


def test_the_check_passes_and_reports_what_the_process_opened(tmp_path: Path) -> None:
    assert cl.run(args_for(tmp_path)) == 0
    report = json.loads((tmp_path / "out" / "loader_check_bottle.json").read_text())
    assert report["result"] == "PASS"
    assert (report["groups"]["fit"]["images"], report["groups"]["calib"]["images"]) == (8, 2)
    audit = report["audit"]
    assert audit["files_opened_inside_dataset_folder"] == fake.N_TRAIN
    assert audit["each_listed_training_file_opened_exactly_once"] is True
    assert audit["files_opened_outside_the_listed_training_files"] == []
    assert audit["directory_listings_inside_dataset_folder"] == 0
    assert report["groups"]["fit"]["tensor_shape_per_image"] == [[3, 256, 256]]


def test_the_report_is_deterministic(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    cl.run(args)
    first = (tmp_path / "out" / "loader_check_bottle.json").read_bytes()
    cl.run(args)
    assert (tmp_path / "out" / "loader_check_bottle.json").read_bytes() == first


def test_the_audit_would_notice_an_extra_file_or_listing(tmp_path: Path) -> None:
    """Self-check of the check: if something else under the dataset folder is opened, it FAILS."""
    args = args_for(tmp_path)
    data = Path(args.data_dir)
    original = cl.TrainingImages.batches

    def sneaky(self, batch_size):  # a loader that also reads a file and lists a folder
        (data / "bottle" / "readme.txt").write_text("x")
        (data / "bottle" / "readme.txt").read_bytes()
        list((data / "bottle" / "train").iterdir())
        yield from original(self, batch_size)

    cl.TrainingImages.batches = sneaky
    try:
        assert cl.run(args) == 1
    finally:
        cl.TrainingImages.batches = original
    report = json.loads((tmp_path / "out" / "loader_check_bottle.json").read_text())
    assert report["result"] == "FAIL"
    assert report["audit"]["files_opened_outside_the_listed_training_files"] == [
        "bottle/readme.txt"
    ]
    assert report["audit"]["directory_listings_inside_dataset_folder"] >= 1


def test_the_script_source_never_names_the_sealed_folders() -> None:
    source = SCRIPT.read_text()
    assert "ground_truth" not in source
    assert not re.search(r"[\"'/]test[\"'/]", source)
