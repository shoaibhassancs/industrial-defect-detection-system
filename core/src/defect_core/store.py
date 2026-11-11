"""Verified storage of model packages (open decision O3, ADR-0003).

The packages (`artifacts/*_release/`) are not kept in git. They are planned to live in a **private**
Hugging Face model repository, pinned by commit. This module is the part that needs no network and
no Hugging Face library: it describes a package, writes and checks the lock file
(`artifacts.lock.json`), and installs a downloaded package only after every file matched the lock.
The scripts `scripts/upload_artifacts.py` and `scripts/download_artifacts.py` add the network calls.

What the checks mean. The lock records the size and SHA-256 of every file of every package, the
repository and the commit the files were uploaded in. A download is accepted only if each file has
exactly the recorded size and SHA-256 **and** the package is internally consistent (the manifest's
own record of the weights and the notice agrees with the files). Anything else is refused, nothing
is
installed, and nothing half-downloaded is left behind. This detects a changed, truncated, swapped or
missing file; it does not prove the files are good models, and it does not decide whether uploading
them is permitted (see `LICENCE_STATUS`).

No file here imports torch or anomalib. Fully loading a package
(`defect_core.artifacts.load_artifact`)
remains the final check and is done by the download script.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

LOCK_FILE_NAME = "artifacts.lock.json"
SCHEMA_VERSION = 1
STORE_TYPE = "huggingface-model-repo"
VISIBILITY = "private"
PACKAGE_FILES = ("model.safetensors", "manifest.json", "NOTICE.md")
STATUS_PLANNED = "planned"
STATUS_UPLOADED = "uploaded"

LICENCE_STATUS = (
    "UNVERIFIED. Choosing a private store does not establish that uploading these weights is "
    "permitted. The weights are treated as material derived from MVTec AD (CC BY-NC-SA 4.0, "
    "non-commercial) as a precaution; the terms of the ImageNet data behind the pretrained "
    "backbone "
    "are not verified. This is the project's cautious policy, not a legal finding."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHUNK = 1024 * 1024

# fetch(repo_id, revision, path_in_repo, staging_dir) -> path of the downloaded file
Fetch = Callable[[str, str, str, Path], Path]


class StoreError(Exception):
    """A package, the lock file or a download is not what it should be (plain message)."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(data: object) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _fail(message: str) -> StoreError:
    return StoreError(message)


# --- describing a local package ---------------------------------------------------------------


def describe_package(folder: Path) -> dict[str, dict[str, Any]]:
    """Size and SHA-256 of the three files of a packaged artifact, after consistency checks.

    The folder must hold exactly `PACKAGE_FILES`, the manifest must say `stage: packaged`, and the
    manifest's record of the weights and of the notice must match the files. Raises `StoreError`.
    """
    if not folder.is_dir():
        raise _fail(f"{folder.name}: the package folder does not exist")
    present = sorted(p.name for p in folder.iterdir())
    if present != sorted(PACKAGE_FILES):
        raise _fail(f"{folder.name}: expected exactly {sorted(PACKAGE_FILES)}, found {present}")
    files: dict[str, dict[str, Any]] = {}
    for name in PACKAGE_FILES:
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise _fail(f"{folder.name}/{name}: must be a regular file")
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if files["NOTICE.md"]["bytes"] == 0:
        raise _fail(f"{folder.name}/NOTICE.md: the notice is empty")
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        recorded = manifest["files"]["model.safetensors"]
        package = manifest["package"]
        stage = manifest["stage"]
        notice = package["notice"]
        weights_sha = package["weights_sha256"]
    except (ValueError, KeyError, TypeError) as error:
        raise _fail(f"{folder.name}/manifest.json: unreadable or incomplete ({error!r})") from error
    if stage != "packaged":
        raise _fail(f"{folder.name}: stage is {stage!r}, only a packaged artifact may be stored")
    weights = files["model.safetensors"]
    if recorded.get("sha256") != weights["sha256"] or recorded.get("bytes") != weights["bytes"]:
        raise _fail(f"{folder.name}: model.safetensors does not match the manifest's record of it")
    if weights_sha != weights["sha256"]:
        raise _fail(f"{folder.name}: model.safetensors does not match package.weights_sha256")
    if notice.get("file") != "NOTICE.md" or notice.get("sha256") != files["NOTICE.md"]["sha256"]:
        raise _fail(f"{folder.name}: NOTICE.md does not match the manifest's record of it")
    return files


# --- the lock file -----------------------------------------------------------------------------


def build_lock(artifacts_dir: Path, folders: Iterable[str], repo_id: str) -> dict[str, Any]:
    """A new lock (status `planned`, no revision) from local package folders."""
    names = sorted(set(folders))
    if not names:
        raise _fail("no package folders were given")
    lock: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": STATUS_PLANNED,
        "store": {
            "type": STORE_TYPE,
            "repo_id": repo_id,
            "visibility": VISIBILITY,
            "revision": None,
        },
        "licence_status": LICENCE_STATUS,
        "packages": {name: {"files": describe_package(artifacts_dir / name)} for name in names},
    }
    validate_lock(lock)
    return lock


def validate_lock(lock: object) -> None:
    """Raise `StoreError` unless `lock` is well formed (shape, hashes, names, status, revision)."""
    if not isinstance(lock, dict) or lock.get("schema") != SCHEMA_VERSION:
        raise _fail("the lock file is not a schema-1 lock")
    store = lock.get("store")
    if not isinstance(store, dict) or store.get("type") != STORE_TYPE:
        raise _fail(f"the lock's store type must be {STORE_TYPE!r}")
    if store.get("visibility") != VISIBILITY:
        raise _fail("the lock must say the repository is private")
    if not isinstance(store.get("repo_id"), str) or not _REPO_ID.match(store["repo_id"]):
        raise _fail("the lock's repo_id must look like 'owner/name'")
    status, revision = lock.get("status"), store.get("revision")
    if status == STATUS_PLANNED:
        if revision is not None:
            raise _fail("a planned lock must not have a revision")
    elif status == STATUS_UPLOADED:
        if not isinstance(revision, str) or not _COMMIT.match(revision):
            raise _fail("an uploaded lock needs the full 40-character commit hash as its revision")
    else:
        raise _fail("the lock's status must be 'planned' or 'uploaded'")
    if not isinstance(lock.get("licence_status"), str) or not lock["licence_status"].startswith(
        "UNVERIFIED"
    ):
        raise _fail("the lock must carry the cautious, unverified licence status")
    packages = lock.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise _fail("the lock lists no packages")
    for name, package in packages.items():
        if not isinstance(name, str) or not _FOLDER.match(name) or name in {".", ".."}:
            raise _fail(f"unsafe package folder name {name!r}")
        files = package.get("files") if isinstance(package, dict) else None
        if not isinstance(files, dict) or sorted(files) != sorted(PACKAGE_FILES):
            raise _fail(f"{name}: the lock must list exactly {sorted(PACKAGE_FILES)}")
        for file_name, entry in files.items():
            ok = (
                isinstance(entry, dict)
                and isinstance(entry.get("bytes"), int)
                and not isinstance(entry.get("bytes"), bool)
                and entry["bytes"] > 0
                and isinstance(entry.get("sha256"), str)
                and _SHA256.match(entry["sha256"])
            )
            if not ok:
                raise _fail(f"{name}/{file_name}: a size and a lowercase SHA-256 are required")


def read_lock(path: Path) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _fail(f"cannot read the lock file {path}: {error}") from error
    validate_lock(lock)
    return lock


def write_lock(path: Path, lock: dict[str, Any]) -> None:
    """Validate, then write atomically (a failed write leaves the old file in place)."""
    validate_lock(lock)
    text = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    handle, temp = tempfile.mkstemp(prefix=".lock-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def plan_fingerprint(lock: dict[str, Any]) -> str:
    """SHA-256 of what an upload would send: the repository, its visibility and every file.

    It does not include the revision or the status, so the owner's approval of a plan stays valid
    after the upload writes the commit hash into the lock. Any changed byte, added or removed file,
    other repository or other visibility gives another fingerprint.
    """
    validate_lock(lock)
    store = lock["store"]
    return _canonical_sha256(
        {
            "schema": lock["schema"],
            "store": {k: store[k] for k in ("type", "repo_id", "visibility")},
            "packages": lock["packages"],
        }
    )


def remote_paths(lock: dict[str, Any], folders: Iterable[str] | None = None) -> list[str]:
    """Paths inside the repository, `<package folder>/<file>`, in a fixed order."""
    chosen = _choose(lock, folders)
    return [f"{name}/{file}" for name in chosen for file in PACKAGE_FILES]


def _choose(lock: dict[str, Any], folders: Iterable[str] | None) -> list[str]:
    available = sorted(lock["packages"])
    if folders is None:
        return available
    wanted = sorted(set(folders))
    unknown = [name for name in wanted if name not in lock["packages"]]
    if unknown or not wanted:
        raise _fail(f"not in the lock: {unknown or 'nothing selected'}; the lock has {available}")
    return wanted


def check_local(
    artifacts_dir: Path, lock: dict[str, Any], folders: Iterable[str] | None = None
) -> None:
    """Refuse unless the local package folders still match the lock exactly."""
    for name in _choose(lock, folders):
        found = describe_package(artifacts_dir / name)
        for file_name in PACKAGE_FILES:
            if found[file_name] != lock["packages"][name]["files"][file_name]:
                raise _fail(
                    f"{name}/{file_name} no longer matches the lock (size or SHA-256 differs); "
                    "rebuild the lock deliberately if the package was meant to change"
                )


def plan_lines(lock: dict[str, Any], folders: Iterable[str] | None = None) -> list[str]:
    """What an upload would send, as plain lines for the owner to read and approve."""
    store = lock["store"]
    lines = [
        f"Store:        {store['type']}",
        f"Repository:   {store['repo_id']}",
        f"Visibility:   {store['visibility']} (an upload refuses a repository that is not private)",
        "Target:       branch 'main'; the commit hash (revision) exists only after the upload and "
        "is then written into the lock"
        if lock["status"] == STATUS_PLANNED
        else f"Revision:     {store['revision']} (already uploaded)",
        f"Status:       {lock['status']}",
        f"Licence:      {lock['licence_status']}",
        "Files:",
    ]
    total = 0
    for name in _choose(lock, folders):
        for file_name in PACKAGE_FILES:
            entry = lock["packages"][name]["files"][file_name]
            total += entry["bytes"]
            lines.append(
                f"  {name}/{file_name}  {entry['bytes']:,} bytes  sha256 {entry['sha256']}"
            )
    lines.append(f"Total:        {total:,} bytes")
    lines.append(f"Plan fingerprint: {plan_fingerprint(lock)}")
    return lines


# --- installing a download ---------------------------------------------------------------------


def install_verified(
    lock: dict[str, Any],
    dest_root: Path,
    fetch: Fetch,
    folders: Iterable[str] | None = None,
) -> list[Path]:
    """Download, verify and install packages. All of them are installed, or none.

    `fetch(repo_id, revision, path_in_repo, staging_dir)` returns the path of a downloaded file; it
    is always called with the lock's pinned commit. Each file is copied into a temporary package
    folder under `dest_root`, and the **copy** is checked for size and SHA-256 against the lock; the
    package must then pass `describe_package` and equal the lock's record. Only then are the folders
    moved into place. Existing destinations are never overwritten. On any failure the temporary
    folder is removed and `StoreError` is raised.
    """
    validate_lock(lock)
    if lock["status"] != STATUS_UPLOADED:
        raise _fail("the lock has no revision yet (status 'planned'): nothing has been uploaded")
    repo_id, revision = lock["store"]["repo_id"], lock["store"]["revision"]
    chosen = _choose(lock, folders)
    dest_root.mkdir(parents=True, exist_ok=True)
    taken = [name for name in chosen if (dest_root / name).exists()]
    if taken:
        raise _fail(f"already exists, nothing was changed (move it away first): {taken}")

    temp = Path(tempfile.mkdtemp(prefix=".download-", dir=dest_root))
    try:
        staging = temp / "_staging"
        staging.mkdir()
        for name in chosen:
            package_dir = temp / name
            package_dir.mkdir()
            for file_name in PACKAGE_FILES:
                expected = lock["packages"][name]["files"][file_name]
                remote = f"{name}/{file_name}"
                try:
                    got = Path(fetch(repo_id, revision, remote, staging))
                except Exception as error:  # network, authorisation, missing file: all refusals
                    raise _fail(f"could not download {remote}: {type(error).__name__}") from error
                if not got.is_file():
                    raise _fail(f"{remote}: the download did not produce a file")
                target = package_dir / file_name
                shutil.copyfile(got, target)
                if target.stat().st_size != expected["bytes"]:
                    raise _fail(f"{remote}: wrong size (changed or incomplete); refused")
                if file_sha256(target) != expected["sha256"]:
                    raise _fail(f"{remote}: SHA-256 differs from the lock (changed file); refused")
            if describe_package(package_dir) != lock["packages"][name]["files"]:
                raise _fail(f"{name}: the package does not match the lock; refused")
        installed = []
        for name in chosen:
            final = dest_root / name
            if final.exists():  # appeared while we were downloading
                raise _fail(f"{final.name} appeared during the download; nothing was installed")
            installed.append(final)
        for name in chosen:
            os.replace(temp / name, dest_root / name)
        return installed
    finally:
        shutil.rmtree(temp, ignore_errors=True)
