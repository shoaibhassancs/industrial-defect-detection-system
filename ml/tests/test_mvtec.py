"""Tests for defect_ml.data (the training guard, the split rules, and the training loader).

A tiny fake dataset is used. Its test and mask files are EMPTY, and an audit hook records every file
the process opens, so the tests can show that the loader never opens a test image or a mask.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch
from defect_core.preprocessing import DEFAULT_CONFIG, load_image, to_tensor
from defect_ml.data import mvtec, splits
from defect_ml.data.mvtec import (
    DataIntegrityError,
    TrainingImages,
    assert_training_path,
    sha256_of,
)
from defect_ml.data.splits import SplitError, read_split, validate_split
from PIL import Image

# --- an audit hook that records the files opened while a test asks it to ---------------------
_RECORDING: list[str] | None = None


def _audit(event: str, args: tuple) -> None:
    if _RECORDING is not None and event in ("open", "os.listdir", "os.scandir"):
        if args and isinstance(args[0], str | bytes | Path):
            _RECORDING.append(str(args[0]))


sys.addaudithook(_audit)


@pytest.fixture
def recorded():
    global _RECORDING
    _RECORDING = []
    yield _RECORDING
    _RECORDING = None


# --- fake dataset -----------------------------------------------------------------------------
N_TRAIN = 10  # -> calib = round(0.2 * 10) = 2, fit = 8


def make_dataset(root: Path) -> tuple[Path, Path, Path]:
    """Return (data_dir, split_dir, manifest_path) for a bottle dataset with 10 training images."""
    data = root / "mvtec"
    train = data / "bottle" / "train" / "good"
    train.mkdir(parents=True)
    for i in range(N_TRAIN):
        Image.new("RGB", (40 + i, 30), (20 * i % 256, 90, 160)).save(train / f"{i:03d}.png")
    for folder, name in (
        ("test/good", "000.png"),
        ("test/crack", "000.png"),
        ("ground_truth/crack", "000_mask.png"),
    ):
        (data / "bottle" / folder).mkdir(parents=True, exist_ok=True)
        (data / "bottle" / folder / name).write_bytes(b"")  # empty: opening one cannot succeed

    names = sorted(p.name for p in train.iterdir())
    fit, calib = splits.assign(names, 0, "bottle")
    fit_paths, calib_paths = ([f"train/good/{n}" for n in x] for x in (fit, calib))
    doc = {
        "split_version": splits.SPLIT_VERSION,
        "category": "bottle",
        "method": splits.METHOD,
        "seed": 0,
        "calib_fraction": splits.CALIB_FRACTION,
        "counts": {
            "train_good_total": len(names),
            "fit": len(fit),
            "calib": len(calib),
            "test_listed_by_name_only": 2,
        },
        "train_files_sha256": splits.list_sha256(fit_paths + calib_paths),
        "data_manifest_sha256": None,
        "paths_are_relative_to": "<dataset folder>/bottle/",
        "fit": fit_paths,
        "calib": calib_paths,
        "test_files_not_opened": {"crack": ["test/crack/000.png"], "good": ["test/good/000.png"]},
    }
    split_dir = root / "splits"
    split_dir.mkdir()
    (split_dir / "bottle.json").write_text(json.dumps(doc))
    manifest = {
        "deep_check": {
            "bottle": {
                "images": [
                    {"path": f"bottle/train/good/{n}", "sha256": sha256_of(train / n)}
                    for n in names
                ]
            }
        }
    }
    manifest_path = root / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return data, split_dir, manifest_path


def open_group(root: Path, group: str) -> TrainingImages:
    data, split_dir, manifest = make_dataset(root) if not (root / "mvtec").exists() else (
        root / "mvtec", root / "splits", root / "data_manifest.json")  # fmt: skip
    return TrainingImages.open(data, "bottle", group, split_dir=split_dir, manifest_path=manifest)


# --- the guard ------------------------------------------------------------------------------


def test_the_guard_allows_only_files_directly_in_train_good(tmp_path: Path) -> None:
    data, _, _ = make_dataset(tmp_path)
    assert_training_path(data, data / "bottle" / "train" / "good" / "000.png")  # allowed
    for forbidden in (
        data / "bottle" / "test" / "good" / "000.png",
        data / "bottle" / "test" / "crack" / "000.png",
        data / "bottle" / "ground_truth" / "crack" / "000_mask.png",
        data / "bottle" / "train" / "other" / "000.png",
        data / "bottle" / "train" / "good" / "sub" / "000.png",
        data / "bottle" / "readme.txt",
        data / "bottle" / "train" / "good",
        data / "bottle" / "train" / "good" / ".." / ".." / "test" / "good" / "000.png",
    ):
        with pytest.raises(PermissionError):
            assert_training_path(data, forbidden)


def test_the_guard_refuses_paths_outside_the_dataset_and_symlinks_that_lead_out(
    tmp_path: Path,
) -> None:
    data, _, _ = make_dataset(tmp_path)
    with pytest.raises(PermissionError, match="outside the dataset"):
        assert_training_path(data, tmp_path / "elsewhere.png")
    link = data / "bottle" / "train" / "good" / "link.png"
    link.symlink_to(data / "bottle" / "test" / "good" / "000.png")
    with pytest.raises(PermissionError):
        assert_training_path(data, link)


def test_the_guard_has_no_switch_or_bypass() -> None:
    assert list(inspect.signature(assert_training_path).parameters) == ["data_dir", "path"]
    for func in (TrainingImages.open, TrainingImages.load, TrainingImages.batches):
        assert not {"allow", "unsafe", "bypass", "force", "include_test"} & set(
            inspect.signature(func).parameters
        )
    source = inspect.getsource(mvtec)
    assert "os.environ" not in source and "getenv(" not in source


# --- the split rules (single definition) ----------------------------------------------------


def test_read_split_accepts_a_consistent_file_and_refuses_the_rest(tmp_path: Path) -> None:
    _, split_dir, _ = make_dataset(tmp_path)
    path = split_dir / "bottle.json"
    doc = read_split(path)
    assert (len(doc["fit"]), len(doc["calib"])) == (8, 2)
    assert validate_split(doc) == []

    bad = json.loads(path.read_text())
    bad["fit"][0] = "test/good/000.png"  # a test path smuggled into the fit list
    path.write_text(json.dumps(bad))
    with pytest.raises(SplitError):
        read_split(path)
    path.write_text("{not json")
    with pytest.raises(SplitError):
        read_split(path)
    with pytest.raises(SplitError):
        read_split(split_dir / "missing.json")


def test_the_committed_bottle_split_is_valid() -> None:
    doc = read_split(Path(__file__).resolve().parents[2] / "splits" / "bottle.json")
    assert (len(doc["fit"]), len(doc["calib"])) == (167, 42)


# --- the loader -----------------------------------------------------------------------------


def test_each_group_loads_only_its_listed_files_in_order(tmp_path: Path) -> None:
    fit, calib = open_group(tmp_path, "fit"), open_group(tmp_path, "calib")
    assert (len(fit), len(calib)) == (8, 2)
    assert not set(fit.names) & set(calib.names)
    assert len(set(fit.names) | set(calib.names)) == N_TRAIN
    batches = list(calib.batches(batch_size=5))
    assert [b.shape for b in batches] == [(2, 3, 256, 256)]
    assert calib.opened == [f"bottle/{n}" for n in calib.names]


def test_batches_are_prepared_exactly_as_defect_core_prepares_them(tmp_path: Path) -> None:
    images = open_group(tmp_path, "fit")
    batches = list(images.batches(batch_size=3))
    assert [b.shape[0] for b in batches] == [3, 3, 2]  # the last batch is smaller
    assert all(b.dtype == torch.float32 and b.shape[1:] == (3, 256, 256) for b in batches)
    for name, tensor in zip(images.names, torch.cat(batches), strict=True):
        expected = to_tensor(load_image(images.data_dir / "bottle" / name), DEFAULT_CONFIG)
        assert torch.equal(tensor, expected)  # input images differ in size; all become 256 x 256


def test_the_loader_never_opens_a_test_image_or_a_mask(tmp_path: Path, recorded) -> None:
    open_group(tmp_path, "fit")  # builds the dataset too
    recorded.clear()
    for group in ("fit", "calib"):
        images = open_group(tmp_path, group)
        for _ in images.batches(4):
            pass
    data = tmp_path / "mvtec"
    inside = [Path(p).relative_to(data).parts for p in recorded if str(data) in p]
    assert len(inside) == N_TRAIN  # each training file once, nothing else
    assert all(parts[:3] == ("bottle", "train", "good") and len(parts) == 4 for parts in inside)
    assert not [parts for parts in inside if parts[1] == "test" or parts[1] == "ground_truth"]


def test_a_changed_file_is_refused_before_it_is_used(tmp_path: Path) -> None:
    images = open_group(tmp_path, "fit")
    victim = images.data_dir / "bottle" / images.names[2]
    Image.new("RGB", (40, 30), (1, 2, 3)).save(victim)  # a valid image, but not the verified one
    batches = images.batches(4)  # the first batch contains the changed file
    with pytest.raises(DataIntegrityError, match="differs from the verified manifest"):
        next(batches)


def test_a_file_missing_from_the_manifest_is_refused_before_any_image_is_opened(
    tmp_path: Path, recorded
) -> None:
    data, split_dir, manifest = make_dataset(tmp_path)
    listed = json.loads((split_dir / "bottle.json").read_text())["fit"][0]  # in the fit group
    doc = json.loads(manifest.read_text())
    images = doc["deep_check"]["bottle"]["images"]
    doc["deep_check"]["bottle"]["images"] = [r for r in images if r["path"] != f"bottle/{listed}"]
    manifest.write_text(json.dumps(doc))
    recorded.clear()
    with pytest.raises(DataIntegrityError, match="no verified hash"):
        TrainingImages.open(data, "bottle", "fit", split_dir=split_dir, manifest_path=manifest)
    assert not [p for p in recorded if p.endswith(".png")]  # no image was touched


def test_missing_or_incomplete_manifests_are_refused(tmp_path: Path) -> None:
    data, split_dir, manifest = make_dataset(tmp_path)
    for content in (None, "{}", '{"deep_check": {}}', "not json"):
        if content is None:
            manifest.unlink()
        else:
            manifest.write_text(content)
        with pytest.raises(DataIntegrityError, match="verify_dataset"):
            TrainingImages.open(data, "bottle", "fit", split_dir=split_dir, manifest_path=manifest)


def test_other_misuse_is_refused(tmp_path: Path) -> None:
    data, split_dir, manifest = make_dataset(tmp_path)
    args = {"split_dir": split_dir, "manifest_path": manifest}
    with pytest.raises(ValueError, match="group"):
        TrainingImages.open(data, "bottle", "test", **args)
    with pytest.raises(ValueError, match="category"):
        TrainingImages.open(data, "bottles", "fit", **args)
    with pytest.raises(DataIntegrityError, match="split file is for"):
        shutil_copy = split_dir / "cable.json"
        shutil_copy.write_text((split_dir / "bottle.json").read_text())
        TrainingImages.open(data, "cable", "fit", **args)
    images = TrainingImages.open(data, "bottle", "fit", **args)
    with pytest.raises(KeyError):
        images.load("test/good/000.png")  # not a listed file: refused before any path is built
    with pytest.raises(ValueError, match="batch_size"):
        next(images.batches(0))
    assert images.opened == []


def test_a_symlink_in_train_good_that_leads_to_a_test_file_is_refused(tmp_path: Path) -> None:
    images = open_group(tmp_path, "fit")
    name = images.names[0]
    link = images.data_dir / "bottle" / name
    link.unlink()
    link.symlink_to(images.data_dir / "bottle" / "test" / "good" / "000.png")
    with pytest.raises(PermissionError):
        images.load(name)
    assert images.opened == []  # refused before the file was read
