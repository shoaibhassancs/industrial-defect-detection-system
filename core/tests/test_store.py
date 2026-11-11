"""Tests for defect_core.store: package description, the lock file, and verified installation.

Everything uses tiny fake packages built in a temporary folder; no network, no model, no dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from defect_core import store

NAME_A = "bottle_a_v1.0.0_release"
NAME_B = "bottle_b_v1.0.0_release"
REPO_ID = "someone/defectlens-test"
COMMIT = "a" * 40


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_package(root: Path, name: str, weights: bytes = b"weights-A" * 50) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    notice = f"Notice for {name}: non-commercial demo.\n".encode()
    (folder / "model.safetensors").write_bytes(weights)
    (folder / "NOTICE.md").write_bytes(notice)
    manifest = {
        "stage": "packaged",
        "files": {"model.safetensors": {"bytes": len(weights), "sha256": sha(weights)}},
        "package": {
            "weights_sha256": sha(weights),
            "notice": {"file": "NOTICE.md", "sha256": sha(notice)},
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


@pytest.fixture
def local(tmp_path: Path) -> Path:
    root = tmp_path / "local"
    make_package(root, NAME_A, b"A" * 300)
    make_package(root, NAME_B, b"B" * 200)
    return root


def uploaded_lock(local: Path) -> dict[str, Any]:
    lock = store.build_lock(local, [NAME_A, NAME_B], REPO_ID)
    lock["status"] = store.STATUS_UPLOADED
    lock["store"]["revision"] = COMMIT
    return lock


def fake_remote(local: Path, calls: list[tuple[str, str, str]] | None = None) -> store.Fetch:
    """A 'remote' that serves the files of `local` by their path in the repository."""

    def fetch(repo_id: str, revision: str, path: str, staging: Path) -> Path:
        if calls is not None:
            calls.append((repo_id, revision, path))
        target = staging / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local / path, target)
        return target

    return fetch


# --- describing a package ----------------------------------------------------------------------


def test_a_consistent_package_is_described(local: Path) -> None:
    files = store.describe_package(local / NAME_A)
    assert sorted(files) == sorted(store.PACKAGE_FILES)
    assert files["model.safetensors"] == {"bytes": 300, "sha256": sha(b"A" * 300)}


def test_describe_refuses_inconsistent_packages(local: Path) -> None:
    folder = local / NAME_A
    (folder / "extra.txt").write_text("x")
    with pytest.raises(store.StoreError, match="expected exactly"):
        store.describe_package(folder)
    (folder / "extra.txt").unlink()
    (folder / "NOTICE.md").unlink()
    with pytest.raises(store.StoreError, match="expected exactly"):
        store.describe_package(folder)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m.update(stage="calibrated"), "only a packaged"),
        (lambda m: m["files"]["model.safetensors"].update(sha256="0" * 64), "manifest's record"),
        (lambda m: m["files"]["model.safetensors"].update(bytes=1), "manifest's record"),
        (lambda m: m["package"].update(weights_sha256="1" * 64), "weights_sha256"),
        (lambda m: m["package"]["notice"].update(sha256="2" * 64), "NOTICE.md does not match"),
        (lambda m: m["package"].pop("notice"), "unreadable or incomplete"),
    ],
)
def test_describe_refuses_a_manifest_that_disagrees_with_the_files(
    local: Path, edit: Any, message: str
) -> None:
    path = local / NAME_A / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(store.StoreError, match=message):
        store.describe_package(local / NAME_A)


def test_describe_refuses_changed_weights_notice_and_empty_notice(local: Path) -> None:
    folder = local / NAME_A
    (folder / "model.safetensors").write_bytes(b"A" * 299 + b"X")  # same size, other content
    with pytest.raises(store.StoreError, match="manifest's record"):
        store.describe_package(folder)
    other = make_package(local, "other_release")
    (other / "NOTICE.md").write_bytes(b"")
    with pytest.raises(store.StoreError, match="empty"):
        store.describe_package(other)


def test_describe_refuses_a_symlinked_file(local: Path, tmp_path: Path) -> None:
    folder = local / NAME_A
    real = tmp_path / "real.md"
    shutil.move(folder / "NOTICE.md", real)
    (folder / "NOTICE.md").symlink_to(real)
    with pytest.raises(store.StoreError, match="regular file"):
        store.describe_package(folder)


# --- the lock ----------------------------------------------------------------------------------


def test_build_lock_is_planned_private_and_unverified(local: Path) -> None:
    lock = store.build_lock(local, [NAME_B, NAME_A], REPO_ID)
    assert lock["status"] == "planned" and lock["store"]["revision"] is None
    assert lock["store"]["visibility"] == "private"
    assert lock["licence_status"].startswith("UNVERIFIED")
    assert sorted(lock["packages"]) == [NAME_A, NAME_B]


def test_lock_round_trips_through_the_file(local: Path, tmp_path: Path) -> None:
    path = tmp_path / store.LOCK_FILE_NAME
    lock = uploaded_lock(local)
    store.write_lock(path, lock)
    assert store.read_lock(path) == lock
    assert path.read_text().endswith("\n")
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".lock-")] == []


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lk: lk.update(schema=2), "schema-1"),
        (lambda lk: lk["store"].update(visibility="public"), "private"),
        (lambda lk: lk["store"].update(repo_id="no-slash"), "owner/name"),
        (lambda lk: lk["store"].update(repo_id="../x/y"), "owner/name"),
        (lambda lk: lk["store"].update(type="s3"), "store type"),
        (lambda lk: lk.update(status="uploaded"), "40-character"),
        (lambda lk: lk["store"].update(revision="main"), "must not have a revision"),
        (lambda lk: lk.update(status="done"), "status"),
        (lambda lk: lk.update(licence_status="Permitted."), "unverified licence"),
        (lambda lk: lk.update(packages={}), "no packages"),
        (
            lambda lk: lk["packages"].update({"../escape": copy.deepcopy(lk["packages"][NAME_A])}),
            "unsafe",
        ),
        (lambda lk: lk["packages"][NAME_A]["files"].pop("NOTICE.md"), "exactly"),
        (
            lambda lk: lk["packages"][NAME_A]["files"]["manifest.json"].update(sha256="ABC"),
            "SHA-256",
        ),
        (lambda lk: lk["packages"][NAME_A]["files"]["manifest.json"].update(bytes=0), "SHA-256"),
        (lambda lk: lk["packages"][NAME_A]["files"]["manifest.json"].update(bytes=True), "SHA-256"),
    ],
)
def test_validate_lock_refuses_bad_locks(local: Path, edit: Any, message: str) -> None:
    lock = store.build_lock(local, [NAME_A], REPO_ID)
    edit(lock)
    with pytest.raises(store.StoreError, match=message):
        store.validate_lock(lock)


def test_a_failed_write_leaves_the_old_lock(local: Path, tmp_path: Path) -> None:
    path = tmp_path / store.LOCK_FILE_NAME
    store.write_lock(path, store.build_lock(local, [NAME_A], REPO_ID))
    before = path.read_text()
    broken = store.build_lock(local, [NAME_A], REPO_ID)
    broken["store"]["visibility"] = "public"
    with pytest.raises(store.StoreError):
        store.write_lock(path, broken)
    assert path.read_text() == before


def test_the_fingerprint_changes_with_anything_that_would_be_uploaded(local: Path) -> None:
    base = store.build_lock(local, [NAME_A, NAME_B], REPO_ID)
    reference = store.plan_fingerprint(base)
    assert reference == store.plan_fingerprint(copy.deepcopy(base))
    # not affected by the revision or status: an approval stays valid after the upload
    assert reference == store.plan_fingerprint(uploaded_lock(local))
    other_repo = store.build_lock(local, [NAME_A, NAME_B], "someone/elsewhere")
    assert store.plan_fingerprint(other_repo) != reference
    assert store.plan_fingerprint(store.build_lock(local, [NAME_A], REPO_ID)) != reference
    (local / NAME_A / "model.safetensors").write_bytes(b"Z" * 300)
    changed = copy.deepcopy(base)
    changed["packages"][NAME_A]["files"]["model.safetensors"]["sha256"] = sha(b"Z" * 300)
    assert store.plan_fingerprint(changed) != reference


def test_plan_lines_show_every_file_size_hash_visibility_and_licence_status(local: Path) -> None:
    lock = store.build_lock(local, [NAME_A, NAME_B], REPO_ID)
    text = "\n".join(store.plan_lines(lock))
    assert REPO_ID in text and "private" in text and "UNVERIFIED" in text
    for name in (NAME_A, NAME_B):
        for file_name, entry in lock["packages"][name]["files"].items():
            assert f"{name}/{file_name}" in text and entry["sha256"] in text
    assert store.plan_fingerprint(lock) in text
    assert "Total:" in text


def test_check_local_detects_a_change_after_the_lock_was_made(local: Path) -> None:
    lock = store.build_lock(local, [NAME_A, NAME_B], REPO_ID)
    store.check_local(local, lock)
    (local / NAME_B / "NOTICE.md").write_bytes(b"edited notice\n")
    manifest_path = local / NAME_B / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["package"]["notice"]["sha256"] = sha(b"edited notice\n")  # a consistent edit
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(store.StoreError, match="no longer matches the lock"):
        store.check_local(local, lock)
    store.check_local(local, lock, [NAME_A])  # the untouched package alone still matches


# --- installing a download ---------------------------------------------------------------------


def test_a_good_download_is_installed_from_the_pinned_commit(local: Path, tmp_path: Path) -> None:
    lock, calls, dest = uploaded_lock(local), [], tmp_path / "dest"
    installed = store.install_verified(lock, dest, fake_remote(local, calls))
    assert sorted(p.name for p in installed) == [NAME_A, NAME_B]
    for name in (NAME_A, NAME_B):
        assert store.describe_package(dest / name) == lock["packages"][name]["files"]
    assert {c[:2] for c in calls} == {(REPO_ID, COMMIT)}  # always the commit, never a branch
    assert sorted(c[2] for c in calls) == sorted(store.remote_paths(lock))
    assert sorted(p.name for p in dest.iterdir()) == [NAME_A, NAME_B]  # no temp or staging left


def test_one_package_can_be_selected(local: Path, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    store.install_verified(uploaded_lock(local), dest, fake_remote(local), [NAME_B])
    assert [p.name for p in dest.iterdir()] == [NAME_B]
    with pytest.raises(store.StoreError, match="not in the lock"):
        store.install_verified(uploaded_lock(local), dest, fake_remote(local), ["nope"])


def test_a_planned_lock_cannot_be_installed(local: Path, tmp_path: Path) -> None:
    lock = store.build_lock(local, [NAME_A], REPO_ID)
    called: list[Any] = []
    with pytest.raises(store.StoreError, match="nothing has been uploaded"):
        store.install_verified(lock, tmp_path / "d", lambda *a: called.append(a))
    assert called == []


def tamper_same_size(root: Path) -> None:
    path = root / NAME_B / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[10] ^= 0xFF
    path.write_bytes(bytes(data))


def tamper_truncate(root: Path) -> None:
    path = root / NAME_B / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])


def tamper_notice(root: Path) -> None:
    (root / NAME_B / "NOTICE.md").write_bytes(b"A softer notice.\n")


def tamper_manifest(root: Path) -> None:
    path = root / NAME_B / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stage"] = "calibrated"
    path.write_text(json.dumps(manifest))


def tamper_consistent_swap(root: Path) -> None:
    """A whole, internally consistent package of different content: only the lock can catch it."""
    shutil.rmtree(root / NAME_B)
    make_package(root, NAME_B, b"B" * 200 + b"!")


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (tamper_same_size, "SHA-256 differs"),
        (tamper_truncate, "wrong size"),
        (tamper_notice, "wrong size|SHA-256 differs"),
        (tamper_manifest, "wrong size|SHA-256 differs"),
        (tamper_consistent_swap, "wrong size|SHA-256 differs"),
    ],
)
def test_a_tampered_download_is_refused_and_nothing_is_installed(
    local: Path, tmp_path: Path, tamper: Any, message: str
) -> None:
    lock = uploaded_lock(local)
    remote = tmp_path / "remote"
    shutil.copytree(local, remote)
    tamper(remote)
    dest = tmp_path / "dest"
    with pytest.raises(store.StoreError, match=message):
        store.install_verified(lock, dest, fake_remote(remote))
    assert list(dest.iterdir()) == []  # not even the good package A, and no temp folder


def test_a_missing_remote_file_or_a_failing_fetch_is_refused(local: Path, tmp_path: Path) -> None:
    lock, dest = uploaded_lock(local), tmp_path / "dest"
    remote = tmp_path / "remote"
    shutil.copytree(local, remote)
    (remote / NAME_B / "NOTICE.md").unlink()
    with pytest.raises(store.StoreError, match="could not download"):
        store.install_verified(lock, dest, fake_remote(remote))

    def broken(*_: Any) -> Path:
        raise ConnectionError("secret-looking detail that must not be echoed")

    with pytest.raises(store.StoreError) as raised:
        store.install_verified(lock, dest, broken)
    assert "ConnectionError" in str(raised.value) and "secret-looking" not in str(raised.value)
    assert list(dest.iterdir()) == []


def test_a_fetch_that_returns_no_file_is_refused(local: Path, tmp_path: Path) -> None:
    with pytest.raises(store.StoreError, match="did not produce a file"):
        store.install_verified(
            uploaded_lock(local), tmp_path / "d", lambda _r, _v, _p, staging: staging / "nothing"
        )


def test_an_existing_destination_is_never_overwritten(local: Path, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    (dest / NAME_A).mkdir(parents=True)
    (dest / NAME_A / "keep.txt").write_text("mine")
    with pytest.raises(store.StoreError, match="already exists"):
        store.install_verified(uploaded_lock(local), dest, fake_remote(local))
    assert (dest / NAME_A / "keep.txt").read_text() == "mine"
    assert [p.name for p in dest.iterdir()] == [NAME_A]


def test_a_tampered_lock_hash_is_refused(local: Path, tmp_path: Path) -> None:
    """The lock is what a download trusts; a wrong hash in it refuses the (good) files."""
    lock = uploaded_lock(local)
    lock["packages"][NAME_A]["files"]["model.safetensors"]["sha256"] = "f" * 64
    with pytest.raises(store.StoreError, match="SHA-256 differs"):
        store.install_verified(lock, tmp_path / "d", fake_remote(local))
