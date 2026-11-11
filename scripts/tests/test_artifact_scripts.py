"""Tests for scripts/upload_artifacts.py and scripts/download_artifacts.py.

No network, no token and no upload: the Hugging Face side is replaced by small fakes. What these
tests cannot show (a real upload, a real private repository, a real fine-grained token) is NOT RUN
and is listed in ADR-0003 and decisions.md.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from defect_core import store

REPO = Path(__file__).resolve().parents[2]


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


up = load("upload_artifacts")
down = load("download_artifacts")

NAME_A, NAME_B = "bottle_a_v1.0.0_release", "bottle_b_v1.0.0_release"
WRITE_TOKEN = "hf_" + "W" * 34  # a made-up, token-shaped string
READ_TOKEN = "hf_" + "R" * 34
COMMIT = "c" * 40


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_package(root: Path, name: str, weights: bytes) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    notice = f"Notice {name}\n".encode()
    (folder / "model.safetensors").write_bytes(weights)
    (folder / "NOTICE.md").write_bytes(notice)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "packaged",
                "files": {"model.safetensors": {"bytes": len(weights), "sha256": sha(weights)}},
                "package": {
                    "weights_sha256": sha(weights),
                    "notice": {"file": "NOTICE.md", "sha256": sha(notice)},
                },
            }
        )
    )


class NotFound(Exception):
    pass


class FakeApi:
    def __init__(self, *, exists: bool = True, private: bool = True, oid: str = COMMIT) -> None:
        self.exists, self.private, self.oid = exists, private, oid
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def repo_info(self, repo_id: str, **kwargs: Any) -> Any:
        self.calls.append(("repo_info", {"repo_id": repo_id, **kwargs}))
        if not self.exists:
            raise NotFound
        return SimpleNamespace(private=self.private)

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        self.exists = True

    def create_commit(self, **kwargs: Any) -> Any:
        self.calls.append(("create_commit", kwargs))
        return SimpleNamespace(oid=self.oid)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def setup(tmp_path: Path) -> SimpleNamespace:
    artifacts = tmp_path / "artifacts"
    make_package(artifacts, NAME_A, b"A" * 300)
    make_package(artifacts, NAME_B, b"B" * 200)
    lock_path = tmp_path / "artifacts.lock.json"
    return SimpleNamespace(artifacts=artifacts, lock=lock_path, tmp=tmp_path)


def common(setup: SimpleNamespace) -> list[str]:
    return ["--lock", str(setup.lock), "--artifacts-dir", str(setup.artifacts)]


def write_lock(setup: SimpleNamespace, repo_id: str = "someone/defectlens-test") -> str:
    code = up.main(
        [
            *common(setup),
            "--write-lock",
            "--repo-id",
            repo_id,
            "--package",
            NAME_A,
            "--package",
            NAME_B,
        ],
        env={},
        api_factory=lambda _t: pytest.fail("no API in a dry run"),
    )
    assert code == 0
    return store.plan_fingerprint(store.read_lock(setup.lock))


def run_upload(
    setup: SimpleNamespace, *flags: str, api: FakeApi | None = None, token: str | None = WRITE_TOKEN
) -> int:
    env = {up.TOKEN_VARIABLE: token} if token else {}
    return up.main(
        [*common(setup), *flags],
        env=env,
        api_factory=lambda _t: api if api is not None else pytest.fail("API must not be used"),
        not_found=lambda: (NotFound,),
    )


# --- upload: the default is a dry run ----------------------------------------------------------


def test_default_mode_prints_the_plan_and_touches_nothing(
    setup: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    fingerprint = write_lock(setup)
    capsys.readouterr()
    assert run_upload(setup, token=None) == 0  # no token, no API
    out = capsys.readouterr().out
    assert fingerprint in out and "private" in out and "UNVERIFIED" in out
    assert "Dry run" in out
    for name in (NAME_A, NAME_B):
        assert f"{name}/model.safetensors" in out
    assert store.read_lock(setup.lock)["status"] == "planned"


def test_write_lock_is_local_and_refused_after_an_upload(setup: SimpleNamespace) -> None:
    fingerprint = write_lock(setup)
    api = FakeApi()
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=api) == 0
    assert run_upload(setup, "--write-lock") == 2  # refuses to rewrite an uploaded lock


@pytest.mark.parametrize("approval", [None, "0" * 64, "not-a-hash"])
def test_upload_needs_the_exact_plan_fingerprint(
    setup: SimpleNamespace, approval: str | None
) -> None:
    write_lock(setup)
    api = FakeApi()
    flags = ["--upload", *(["--approve-plan", approval] if approval else [])]
    assert run_upload(setup, *flags, api=api) == 2
    assert api.calls == [] and store.read_lock(setup.lock)["status"] == "planned"


def test_upload_needs_a_token_and_never_reads_a_generic_one(setup: SimpleNamespace) -> None:
    fingerprint = write_lock(setup)
    api = FakeApi()
    env = {"HF_TOKEN": WRITE_TOKEN, up.TOKEN_VARIABLE: ""}
    code = up.main(
        [*common(setup), "--upload", "--approve-plan", fingerprint],
        env=env,
        api_factory=lambda _t: api,
        not_found=lambda: (NotFound,),
    )
    assert code == 2 and api.calls == []


def test_a_changed_local_file_is_refused_before_any_network_call(setup: SimpleNamespace) -> None:
    fingerprint = write_lock(setup)
    (setup.artifacts / NAME_A / "model.safetensors").write_bytes(b"A" * 299 + b"!")
    api = FakeApi()
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=api) == 2
    assert api.calls == []


def test_a_changed_repository_or_file_needs_a_new_approval(setup: SimpleNamespace) -> None:
    old = write_lock(setup, "someone/one")
    new = write_lock(setup, "someone/two")
    assert old != new
    api = FakeApi()
    assert run_upload(setup, "--upload", "--approve-plan", old, api=api) == 2
    assert api.calls == []


# --- upload: the repository checks and the commit ----------------------------------------------


def test_an_approved_upload_is_one_commit_of_exactly_the_locked_files(
    setup: SimpleNamespace,
) -> None:
    fingerprint = write_lock(setup)
    api = FakeApi()
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=api) == 0
    assert api.names() == ["repo_info", "create_commit"]  # no repository was created
    commit = api.calls[1][1]
    assert commit["repo_id"] == "someone/defectlens-test" and commit["repo_type"] == "model"
    assert commit["token"] == WRITE_TOKEN
    paths = sorted(op.path_in_repo for op in commit["operations"])
    assert paths == sorted(store.remote_paths(store.read_lock(setup.lock)))
    lock = store.read_lock(setup.lock)
    assert lock["status"] == "uploaded" and lock["store"]["revision"] == COMMIT
    assert store.plan_fingerprint(lock) == fingerprint  # the approval still describes the lock


def test_a_public_repository_is_refused(setup: SimpleNamespace) -> None:
    fingerprint = write_lock(setup)
    api = FakeApi(private=False)
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=api) == 2
    assert "create_commit" not in api.names()
    assert store.read_lock(setup.lock)["status"] == "planned"


def test_a_missing_repository_is_not_created_unless_asked(setup: SimpleNamespace) -> None:
    fingerprint = write_lock(setup)
    api = FakeApi(exists=False)
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=api) == 2
    assert api.names() == ["repo_info"]
    assert (
        run_upload(
            setup, "--upload", "--approve-plan", fingerprint, "--create-private-repo", api=api
        )
        == 0
    )
    create = dict(api.calls)["create_repo"]
    assert create["private"] is True and create["token"] == WRITE_TOKEN


def test_a_bad_commit_hash_from_the_api_is_not_recorded(setup: SimpleNamespace) -> None:
    fingerprint = write_lock(setup)
    api = FakeApi(oid="main")
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=api) == 2
    assert store.read_lock(setup.lock)["status"] == "planned"


def test_an_uploaded_lock_cannot_be_uploaded_again(setup: SimpleNamespace) -> None:
    fingerprint = write_lock(setup)
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=FakeApi()) == 0
    api = FakeApi()
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=api) == 2
    assert api.calls == []


def test_an_api_failure_is_reported_without_the_token(
    setup: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    fingerprint = write_lock(setup)

    class Failing(FakeApi):
        def create_commit(self, **kwargs: Any) -> Any:
            raise RuntimeError(f"401 for token {WRITE_TOKEN}")

    capsys.readouterr()
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=Failing()) == 1
    captured = capsys.readouterr()
    assert WRITE_TOKEN not in captured.out + captured.err and "[redacted]" in captured.err
    assert store.read_lock(setup.lock)["status"] == "planned"


# --- download ----------------------------------------------------------------------------------


def uploaded(setup: SimpleNamespace) -> Path:
    fingerprint = write_lock(setup)
    assert run_upload(setup, "--upload", "--approve-plan", fingerprint, api=FakeApi()) == 0
    remote = setup.tmp / "remote"
    shutil.copytree(setup.artifacts, remote)
    return remote


def remote_fetch(remote: Path, seen: list[Any] | None = None) -> Any:
    def factory(token: str | None) -> store.Fetch:
        def fetch(repo_id: str, revision: str, path: str, staging: Path) -> Path:
            if seen is not None:
                seen.append((token, revision, path))
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote / path, target)
            return target

        return fetch

    return factory


def run_download(
    setup: SimpleNamespace, factory: Any, *flags: str, token: str | None = None
) -> int:
    loaded: list[Path] = []
    env = (
        {down.TOKEN_VARIABLE: token, "HF_TOKEN": WRITE_TOKEN}
        if token
        else {"HF_TOKEN": WRITE_TOKEN}
    )
    code = down.main(
        ["--lock", str(setup.lock), "--dest", str(setup.tmp / "dest"), *flags],
        env=env,
        fetch_factory=factory,
        load_check=loaded.append,
    )
    setup.loaded = loaded
    return code


def test_download_installs_verifies_loads_and_uses_only_the_read_token(
    setup: SimpleNamespace,
) -> None:
    remote, seen = uploaded(setup), []
    assert run_download(setup, remote_fetch(remote, seen), token=READ_TOKEN) == 0
    assert {t for t, _, _ in seen} == {READ_TOKEN}  # never the generic HF_TOKEN
    assert {r for _, r, _ in seen} == {COMMIT}
    assert sorted(p.name for p in setup.loaded) == [NAME_A, NAME_B]
    assert sorted(p.name for p in (setup.tmp / "dest").iterdir()) == [NAME_A, NAME_B]


def test_download_without_a_token_asks_for_no_authorisation(setup: SimpleNamespace) -> None:
    remote, seen = uploaded(setup), []
    assert run_download(setup, remote_fetch(remote, seen)) == 0
    assert {t for t, _, _ in seen} == {None}  # the real fetch turns None into token=False


def test_download_refuses_a_tampered_file_and_installs_nothing(
    setup: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    remote = uploaded(setup)
    (remote / NAME_B / "model.safetensors").write_bytes(b"B" * 199 + b"!")
    assert run_download(setup, remote_fetch(remote), token=READ_TOKEN) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert list((setup.tmp / "dest").iterdir()) == [] and setup.loaded == []


def test_download_refuses_a_lock_without_a_revision(setup: SimpleNamespace) -> None:
    write_lock(setup)  # planned: nothing was uploaded
    assert run_download(setup, remote_fetch(setup.artifacts)) == 2
    assert not (setup.tmp / "dest").exists() or list((setup.tmp / "dest").iterdir()) == []


def test_a_failing_load_check_is_reported(setup: SimpleNamespace, capsys: Any) -> None:
    remote = uploaded(setup)

    def bad(_: Path) -> None:
        raise ValueError("not a loadable artifact")

    code = down.main(
        ["--lock", str(setup.lock), "--dest", str(setup.tmp / "dest")],
        env={},
        fetch_factory=remote_fetch(remote),
        load_check=bad,
    )
    assert code == 1 and "FAILED: ValueError" in capsys.readouterr().err


def test_the_real_fetch_never_falls_back_to_a_cached_or_generic_token(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_download(**kwargs: Any) -> str:
        captured.update(kwargs)
        return str(Path("/nonexistent"))

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    down.hub_fetch(None)("a/b", COMMIT, "p/f", Path("/stage"))
    assert captured["token"] is False and captured["revision"] == COMMIT
    down.hub_fetch(READ_TOKEN)("a/b", COMMIT, "p/f", Path("/stage"))
    assert captured["token"] == READ_TOKEN


# --- tokens and secrets ------------------------------------------------------------------------


@pytest.mark.parametrize("module", [up, down])
def test_no_command_line_option_takes_a_token(module: Any) -> None:
    options = {o for action in module.build_parser()._actions for o in action.option_strings}
    assert not [o for o in options if "token" in o.lower() or "secret" in o.lower()]


def test_the_two_scripts_use_different_token_variables() -> None:
    assert up.TOKEN_VARIABLE != down.TOKEN_VARIABLE
    source = (REPO / "scripts" / "download_artifacts.py").read_text()
    for write_path in ("create_commit", "HfApi", "upload_file", "create_repo", "delete_"):
        assert write_path not in source


def test_no_hugging_face_token_is_in_the_repository_sources() -> None:
    pattern = re.compile(r"hf_[A-Za-z0-9]{30,}")
    roots = ["core/src", "scripts", "backend", "docs", "README.md", "artifacts.lock.json"]
    offenders = []
    for root in roots:
        base = REPO / root
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".toml", ".yml", ""}:
                if "tests" in path.parts and path.name.startswith("test_artifact_scripts"):
                    continue  # this file builds made-up token-shaped strings on purpose
                if pattern.search(path.read_text(errors="ignore")):
                    offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


def test_credential_files_stay_out_of_git() -> None:
    ignored = (REPO / ".gitignore").read_text()
    assert ".env" in ignored and "artifacts/" in ignored and "*.safetensors" in ignored


# --- the real packages (skipped where they do not exist, for example in CI) --------------------


def real_folders() -> list[str]:
    return [n for n in up.DEFAULT_PACKAGES if (REPO / "artifacts" / n).is_dir()]


@pytest.mark.skipif(not (REPO / "artifacts").is_dir(), reason="model packages are not in git")
def test_the_committed_lock_matches_the_local_packages() -> None:
    lock = store.read_lock(REPO / store.LOCK_FILE_NAME)
    names = [n for n in sorted(lock["packages"]) if (REPO / "artifacts" / n).is_dir()]
    assert names, "no package folder found"
    store.check_local(REPO / "artifacts", lock, names)


def test_the_committed_lock_is_valid_private_and_unverified() -> None:
    lock = store.read_lock(REPO / store.LOCK_FILE_NAME)
    assert lock["store"]["visibility"] == "private"
    assert lock["licence_status"].startswith("UNVERIFIED")
    assert sorted(lock["packages"]) == sorted(up.DEFAULT_PACKAGES)


def test_upload_itself_rechecks_the_local_files_before_any_network_call(
    setup: SimpleNamespace,
) -> None:
    """`upload()` is safe even when called without `main()`'s own check."""
    write_lock(setup)
    lock = store.read_lock(setup.lock)
    (setup.artifacts / NAME_B / "NOTICE.md").write_bytes(b"changed\n")
    api = FakeApi()
    with pytest.raises(store.StoreError):
        up.upload(
            lock, setup.artifacts, api, WRITE_TOKEN,
            packages=None, create_private_repo=False, not_found=(NotFound,),
        )  # fmt: skip
    assert api.calls == []
