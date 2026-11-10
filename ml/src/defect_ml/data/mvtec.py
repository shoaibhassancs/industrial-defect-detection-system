"""MVTec AD data access for fitting and calibration: the training guard and the training loader.

Two things live here, each defined once:

1. ``assert_training_path``: the guard of the **training loader**. It allows only files that sit
   directly in ``<category>/train/good``.
2. ``TrainingImages``: reads the ``fit`` or ``calib`` group listed in ``splits/<category>.json``,
   checks every file against the verified manifest, and prepares it with
   ``defect_core.preprocessing`` (the only definition of preprocessing).

Sealed test data, and how it is opened later
--------------------------------------------
The official ``test/`` images and the ``ground_truth/`` masks are not reachable through this
module: the guard refuses them, and the loader has no switch, argument or environment variable that
changes that. This is a statement about the *training* path, not a permanent ban on the data.

After the model is frozen and calibrated (PLAN.md Phase 3), the sealed evaluation reads those files
through a **separate, explicit path**: ``defect_ml/evaluate.py`` (not written yet). By design that
code:

* refuses to run unless the artifact's manifest is frozen and calibrated;
* opens test files only through its own reader, not through this module, and has its own check of
  which paths it may open;
* appends every file it opens to ``evaluation/test_access_log.jsonl`` (append-only);
* is the only code that reads ``test/`` and ``ground_truth/``.

So opening the test set is a deliberate, logged act of a different component, and fitting and
calibration code can not do it by accident. See ``docs/data.md`` and ``PLAN.md`` section 5.6.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from defect_core.preprocessing import DEFAULT_CONFIG, PreprocessConfig, load_image, to_tensor

from defect_ml.data.splits import read_split

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]  # fmt: skip
GROUPS = ("fit", "calib")


class DataIntegrityError(Exception):
    """A training file or the manifest does not match what was verified."""


def assert_training_path(data_dir: Path, path: Path) -> None:
    """Raise ``PermissionError`` unless ``path`` is a file directly in ``<category>/train/good``.

    This is the guard of the training loader and of the dataset tools that read training images.
    The path is resolved first, so ``..`` and symbolic links that lead elsewhere are refused too.
    """
    try:
        parts = path.resolve().relative_to(data_dir.resolve()).parts
    except ValueError:
        msg = f"refusing to open a file outside the dataset folder: {path}"
        raise PermissionError(msg) from None
    if len(parts) != 4 or parts[1:3] != ("train", "good"):
        raise PermissionError(f"refusing to open a file outside train/good: {'/'.join(parts)}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest_hashes(manifest_path: str | Path, category: str) -> dict[str, str]:
    """The verified SHA-256 of each training image: ``{"bottle/train/good/000.png": "..."}``.

    Raises ``DataIntegrityError`` if the manifest is missing or has no deep check for the category.
    """
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
        images = manifest["deep_check"][category]["images"]
        return {record["path"]: record["sha256"] for record in images}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise DataIntegrityError(
            f"no verified manifest for '{category}' in {manifest_path} ({error!r}); "
            "run scripts/data/verify_dataset.py first"
        ) from error


@dataclass(frozen=True)
class TrainingImages:
    """The ``fit`` or ``calib`` images of one category, loaded and checked file by file.

    Only the files listed in the split file for this group are ever opened. Each is read once, its
    SHA-256 is compared with the verified manifest, and only then is it decoded (from the same
    bytes that were hashed). ``opened`` records the relative path of every file read, in order.
    """

    data_dir: Path
    category: str
    group: str
    names: tuple[str, ...]  # relative to <data_dir>/<category>/, e.g. "train/good/000.png"
    hashes: dict[str, str]  # "<category>/<name>" -> verified SHA-256
    config: PreprocessConfig
    opened: list[str]

    @classmethod
    def open(
        cls,
        data_dir: str | Path,
        category: str,
        group: str,
        *,
        split_dir: str | Path,
        manifest_path: str | Path,
        config: PreprocessConfig = DEFAULT_CONFIG,
    ) -> TrainingImages:
        """Prepare a loader. Reads the split file and the manifest; opens no image."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        if group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
        split = read_split(Path(split_dir) / f"{category}.json")
        if split["category"] != category:
            raise DataIntegrityError(f"split file is for '{split['category']}', not '{category}'")
        names = tuple(split[group])
        hashes = read_manifest_hashes(manifest_path, category)
        missing = [n for n in names if f"{category}/{n}" not in hashes]
        if missing:  # refuse before any image is opened
            raise DataIntegrityError(
                f"{len(missing)} listed file(s) have no verified hash in the manifest, "
                f"for example {missing[:3]}"
            )
        return cls(Path(data_dir), category, group, names, hashes, config, [])

    def __len__(self) -> int:
        return len(self.names)

    def load(self, name: str) -> torch.Tensor:
        """Read, check and prepare one listed file; returns a float32 tensor (3, size, size)."""
        if name not in self.names:
            raise KeyError(f"{name!r} is not in the '{self.group}' group of {self.category}")
        path = self.data_dir / self.category / name
        assert_training_path(self.data_dir, path)
        data = path.read_bytes()
        self.opened.append(f"{self.category}/{name}")
        if sha256_bytes(data) != self.hashes[f"{self.category}/{name}"]:
            raise DataIntegrityError(f"{self.category}/{name} differs from the verified manifest")
        return to_tensor(load_image(data), self.config)

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Yield tensors (B, 3, size, size) in the listed order; the last batch may be smaller."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        for start in range(0, len(self.names), batch_size):
            yield torch.stack([self.load(n) for n in self.names[start : start + batch_size]])
