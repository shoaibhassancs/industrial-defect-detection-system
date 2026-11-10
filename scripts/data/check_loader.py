"""Check the fit/calibration loader on the real training images (Phase 2 step 5 groundwork).

Runs `defect_ml.data.mvtec.TrainingImages` over the `fit` and `calib` groups of one category and
reports, as evidence rather than as a promise:

* how many images were loaded per group, and the shape, dtype and value statistics of the batches;
* which files the process opened inside the dataset folder. A Python audit hook records every
  `open`, `listdir` and `scandir` event made while the loader runs, so the list comes from the
  running process, not from the loader's own bookkeeping. The check passes only if the files opened
  are exactly the listed training files (each once) and nothing else under the dataset folder was
  opened or listed.

It reads the split file, the manifest and the training images. It writes
`experiments/loader_check_<category>.json` (small, deterministic: no timestamp).

Usage
-----
    uv run python scripts/data/check_loader.py --category bottle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from defect_ml.audit import OpenAudit
from defect_ml.data.mvtec import CATEGORIES, TrainingImages
from defect_ml.data.splits import read_split


class Stats:
    """Running per-channel statistics over normalised batches (float64)."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = torch.zeros(3, dtype=torch.float64)
        self.sumsq = torch.zeros(3, dtype=torch.float64)
        self.minimum, self.maximum = float("inf"), float("-inf")
        self.shapes: set[tuple[int, ...]] = set()
        self.dtypes: set[str] = set()
        self.finite = True

    def add(self, batch: torch.Tensor) -> None:
        self.shapes.add(tuple(batch.shape[1:]))
        self.dtypes.add(str(batch.dtype))
        self.finite &= bool(torch.isfinite(batch).all())
        values = batch.to(torch.float64)
        self.count += batch.shape[0]
        self.sum += values.sum(dim=(0, 2, 3))
        self.sumsq += (values**2).sum(dim=(0, 2, 3))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.pixels = batch.shape[2] * batch.shape[3]

    def summary(self) -> dict:
        n = self.count * self.pixels
        mean = self.sum / n
        std = (self.sumsq / n - mean**2).clamp_min(0).sqrt()
        return {
            "images": self.count,
            "tensor_shape_per_image": [list(s) for s in sorted(self.shapes)],
            "dtypes": sorted(self.dtypes),
            "all_finite": self.finite,
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "channel_mean": [round(float(x), 4) for x in mean],
            "channel_std": [round(float(x), 4) for x in std],
        }


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    split = read_split(Path(args.split_dir) / f"{args.category}.json")
    listed = {f"{args.category}/{p}" for p in split["fit"] + split["calib"]}

    results: dict[str, dict] = {}
    opened_by_loader: list[str] = []
    with OpenAudit() as audit:
        for group in ("fit", "calib"):
            images = TrainingImages.open(
                data_dir, args.category, group,
                split_dir=args.split_dir, manifest_path=args.manifest,
            )  # fmt: skip
            stats = Stats()
            for batch in images.batches(args.batch_size):
                stats.add(batch)
            results[group] = stats.summary()
            opened_by_loader += images.opened

    # What the running process itself did inside the dataset folder
    file_opens, listings = audit.inside(data_dir)
    rel_opened = sorted(str(p.resolve().relative_to(data_dir)) for p in file_opens)
    expected_pattern = sorted(listed)
    ok = (
        rel_opened == expected_pattern
        and not listings
        and sorted(opened_by_loader) == expected_pattern
        and results["fit"]["images"] == len(split["fit"])
        and results["calib"]["images"] == len(split["calib"])
        and all(r["all_finite"] for r in results.values())
    )
    report = {
        "check": "fit/calib loader on real training images",
        "category": args.category,
        "result": "PASS" if ok else "FAIL",
        "split_file_sha256_of_lists": split["train_files_sha256"],
        "groups": results,
        "audit": {
            "method": "sys.addaudithook: open, os.listdir and os.scandir events during the run",
            "files_opened_inside_dataset_folder": len(file_opens),
            "each_listed_training_file_opened_exactly_once": rel_opened == expected_pattern,
            "files_opened_outside_the_listed_training_files": sorted(
                set(rel_opened) - set(expected_pattern)
            ),
            "directory_listings_inside_dataset_folder": len(listings),
        },
    }
    out = Path(args.out_dir) / f"loader_check_{args.category}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("category", "result", "groups", "audit")}, indent=1))
    print(f"wrote {out}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/raw/mvtec-ad")
    parser.add_argument("--category", default="bottle", choices=CATEGORIES)
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--manifest", default="data/data_manifest.json")
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--batch-size", type=int, default=8)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
