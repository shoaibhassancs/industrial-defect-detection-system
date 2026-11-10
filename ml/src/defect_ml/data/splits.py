"""The `fit` / `calib` split of one category's training images: rules, checks and reading.

`scripts/data/make_splits.py` writes `splits/<category>.json` with these functions; the training
loader (`mvtec.py`) reads it with `read_split`, which refuses an inconsistent file. There is one
definition of the rules, here.

How the split is chosen (no random-number generator, so it cannot change between library
versions): the images are ranked by SHA-256 of "<seed>:<category>:<file name>"; the first n_calib
in that ranking form `calib`, the rest form `fit`; n_calib = round(0.2 * n). For `bottle`
(n = 209) that is 42 calib and 167 fit. A split file also lists the official test files by NAME
only, as a record; nothing in this module opens a file other than the split file itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_VERSION = 1
CALIB_FRACTION = 0.2
DEFAULT_SEED = 0
METHOD = (
    "Rank the training files by SHA-256 of '<seed>:<category>:<file name>'. The first n_calib in "
    "that ranking are `calib`, all others are `fit`. n_calib = round(0.2 * n). File names only; "
    "no image is opened."
)


class SplitError(ValueError):
    """A split file is missing, malformed, or inconsistent."""


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


def read_split(path: str | Path) -> dict:
    """Read a split file and check it; raises ``SplitError`` if it is unusable.

    This reads the split file only. It opens no image.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
        problems = validate_split(doc)
        if doc["split_version"] != SPLIT_VERSION:
            problems.append(f"unsupported split_version {doc['split_version']!r}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise SplitError(f"cannot use split file {path}: {error!r}") from error
    if problems:
        raise SplitError(f"split file {path} is inconsistent: {problems}")
    return doc
