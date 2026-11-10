"""Phase 1, step 5: a seeded 80/20 `fit` / `calib` split of one category's training images.

* `fit`   (80%): the images the PatchCore memory bank will be built from.
* `calib` (20%): held-out normal TRAINING images, used only to set the threshold (PLAN.md 5.6).
* The official test files are listed by NAME only, as a record. They are never opened, and they
  never appear in `fit` or `calib`.

How the split is chosen (no random-number generator, so it cannot change between library versions):
the images are ranked by SHA-256 of "<seed>:<category>:<file name>"; the first n_calib in that
ranking form `calib`, the rest form `fit`; n_calib = round(0.2 * n). For `bottle` (n = 209) that is
42 calib and 167 fit.

The script lists file names only. It never opens an image.

Output: `splits/<category>.json` (meant to be committed). An existing split is never changed
silently: running again must reproduce the same file, otherwise the script stops unless
`--force` is given (a different split means a new model version, see PLAN.md 4.5).

Usage
-----
    uv run python scripts/data/make_splits.py --category bottle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SPLIT_VERSION = 1
CALIB_FRACTION = 0.2
DEFAULT_SEED = 0
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]  # fmt: skip
METHOD = (
    "Rank the training files by SHA-256 of '<seed>:<category>:<file name>'. The first n_calib in "
    "that ranking are `calib`, all others are `fit`. n_calib = round(0.2 * n). File names only; "
    "no image is opened."
)


def list_visible(path: Path) -> list[str]:
    """Sorted names in a folder, without hidden files such as .DS_Store."""
    if not path.is_dir():
        return []
    return sorted(n for n in os.listdir(path) if not n.startswith("."))


def n_calib_for(n: int) -> int:
    return int(n * CALIB_FRACTION + 0.5)  # round half up


def list_sha256(paths: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(paths)) + "\n").encode()).hexdigest()


def assign(names: list[str], seed: int, category: str) -> tuple[list[str], list[str]]:
    """Return (fit, calib) as sorted lists of file names."""

    def rank(name: str) -> tuple[str, str]:
        return hashlib.sha256(f"{seed}:{category}:{name}".encode()).hexdigest(), name

    ranked = sorted(names, key=rank)
    k = n_calib_for(len(names))
    return sorted(ranked[k:]), sorted(ranked[:k])


def cross_check_manifest(manifest_path: Path, category: str, names: list[str]) -> str | None:
    """If the verification manifest covers this category, its file set must equal ours."""
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    deep = manifest.get("deep_check", {}).get(category)
    if deep is None:
        return None
    verified = sorted(Path(r["path"]).name for r in deep["images"])
    if verified != sorted(names):
        raise ValueError(
            f"{category}/train/good on disk differs from the verified manifest "
            f"({len(names)} files vs {len(verified)}); run verify_dataset.py again first"
        )
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def build_split(data_dir: Path, category: str, seed: int, manifest_path: Path) -> dict:
    cat_dir = data_dir / category
    names = list_visible(cat_dir / "train" / "good")
    if not names:
        raise ValueError(f"no training images found in {cat_dir / 'train' / 'good'}")
    manifest_sha = cross_check_manifest(manifest_path, category, names)
    fit, calib = assign(names, seed, category)
    test_files = {
        defect: [f"test/{defect}/{n}" for n in list_visible(cat_dir / "test" / defect)]
        for defect in list_visible(cat_dir / "test")
    }
    fit_paths = [f"train/good/{n}" for n in fit]
    calib_paths = [f"train/good/{n}" for n in calib]
    return {
        "split_version": SPLIT_VERSION,
        "category": category,
        "method": METHOD,
        "seed": seed,
        "calib_fraction": CALIB_FRACTION,
        "counts": {
            "train_good_total": len(names),
            "fit": len(fit),
            "calib": len(calib),
            "test_listed_by_name_only": sum(len(v) for v in test_files.values()),
        },
        "train_files_sha256": list_sha256(fit_paths + calib_paths),
        "data_manifest_sha256": manifest_sha,
        "paths_are_relative_to": f"<dataset folder>/{category}/",
        "fit": fit_paths,
        "calib": calib_paths,
        "test_files_not_opened": test_files,
    }


def validate_split(doc: dict) -> list[str]:
    """Return a list of problems (empty means the split is consistent)."""
    problems: list[str] = []
    fit, calib = doc["fit"], doc["calib"]
    test = [p for paths in doc["test_files_not_opened"].values() for p in paths]
    for label, paths in (("fit", fit), ("calib", calib), ("test", test)):
        if paths != sorted(paths):
            problems.append(f"{label} is not sorted")
        if len(set(paths)) != len(paths):
            problems.append(f"{label} has duplicates")
    if set(fit) & set(calib):
        problems.append("fit and calib overlap")
    if (set(fit) | set(calib)) & set(test):
        problems.append("a test file appears in fit or calib")
    bad = [p for p in fit + calib if not p.startswith("train/good/")]
    if bad:
        problems.append(f"{len(bad)} fit/calib path(s) are not under train/good/")
    if any(not p.startswith("test/") for p in test):
        problems.append("a test path is not under test/")
    total = len(fit) + len(calib)
    counts = doc["counts"]
    if (counts["train_good_total"], counts["fit"], counts["calib"]) != (
        total,
        len(fit),
        len(calib),
    ):
        problems.append("counts do not match the lists")
    if len(calib) != n_calib_for(total):
        problems.append(f"calib size {len(calib)} is not round(0.2 * {total})")
    if doc["train_files_sha256"] != list_sha256(fit + calib):
        problems.append("train_files_sha256 does not match the lists")
    names = [Path(p).name for p in fit + calib]
    expected_fit, expected_calib = assign(names, doc["seed"], doc["category"])
    if [Path(p).name for p in calib] != expected_calib or [
        Path(p).name for p in fit
    ] != expected_fit:
        problems.append("the lists are not what the stated method gives for this seed")
    return problems


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    doc = build_split(data_dir, args.category, args.seed, Path(args.manifest))
    problems = validate_split(doc)
    if problems:
        print("INTERNAL ERROR: the new split is inconsistent:", problems, file=sys.stderr)
        return 1
    text = json.dumps(doc, indent=1) + "\n"
    out = Path(args.out_dir) / f"{args.category}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() != text and not args.force:
        print(
            f"REFUSING to change the existing split {out}; use --force only on purpose.",
            file=sys.stderr,
        )
        return 1
    status = "unchanged" if out.exists() and out.read_text() == text else "written"
    out.write_text(text)
    c = doc["counts"]
    total, fit, calib = c["train_good_total"], c["fit"], c["calib"]
    print(f"{args.category}: {total} training images -> fit {fit}, calib {calib}")
    print(f"test files listed by name only (not opened): {c['test_listed_by_name_only']}")
    cross_checked = doc["data_manifest_sha256"] is not None
    print(f"seed {doc['seed']}; cross-checked with the verified manifest: {cross_checked}")
    print(f"{out}: {status}; sha256 of the lists: {doc['train_files_sha256']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/raw/mvtec-ad")
    parser.add_argument("--category", default="bottle", choices=CATEGORIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--manifest", default="data/data_manifest.json")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing, different split"
    )
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
