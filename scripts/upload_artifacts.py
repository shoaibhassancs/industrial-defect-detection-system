"""Plan, and (only on explicit approval) upload, the model packages to the private store.

Decision O3 / ADR-0003: a private Hugging Face model repository, pinned by commit. This script
**does nothing on the network by default**. Its three modes:

  (no flag)       Read `artifacts.lock.json`, check the local package folders still match it, and
                  print the plan: repository, visibility, every file with size and SHA-256, the
                  licence status and a *plan fingerprint*. No token, no network.
  --write-lock    Build or refresh the lock from the local `*_release` folders (local only). Refused
                  once the lock says `uploaded`.
  --upload        The only mode that touches the network. It needs `--approve-plan <fingerprint>`
                  (the fingerprint the owner approved), the write token in the environment variable
                  DEFECTLENS_HF_WRITE_TOKEN, and a repository that is private. It never creates a
                  repository unless `--create-private-repo` is also given. After the commit it
                  writes
                  the commit hash into the lock. Verify with `download_artifacts.py`.

The token is read from the environment only: there is no token option (command lines end up in shell
history), it is never printed or written to a file. Use a fine-grained token limited to this one
repository (see docs/adr/ADR-0003-artifact-store.md).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from defect_core import store

REPO = Path(__file__).resolve().parents[1]
TOKEN_VARIABLE = "DEFECTLENS_HF_WRITE_TOKEN"
DEFAULT_REPO_ID = "shoaibxhassan/defectlens-bottle-models"  # a proposal; the owner may change it
DEFAULT_PACKAGES = (
    "bottle_wide_resnet50_2_tv_in1k_v1.0.0_release",
    "bottle_resnet18_tv_in1k_v1.0.0_release",
)
COMMIT_MESSAGE = "Add DefectLens bottle model packages (v1.0.0): weights, manifest, NOTICE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lock", type=Path, default=REPO / store.LOCK_FILE_NAME)
    parser.add_argument("--artifacts-dir", type=Path, default=REPO / "artifacts")
    parser.add_argument("--write-lock", action="store_true", help="build or refresh the lock")
    parser.add_argument("--repo-id", help="with --write-lock: 'owner/name' (default: the lock's)")
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        help="package folder name (repeatable; default: all in the lock)",
    )
    parser.add_argument("--upload", action="store_true", help="UPLOAD (needs --approve-plan)")
    parser.add_argument("--approve-plan", help="the plan fingerprint the owner approved")
    parser.add_argument(
        "--create-private-repo",
        action="store_true",
        help="with --upload: create the repository as private if it does not exist",
    )
    return parser


def _redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "[redacted]") if secret else text


def _default_api(token: str) -> Any:
    from huggingface_hub import HfApi  # imported only for a real upload

    return HfApi(token=token)


def _not_found_errors() -> tuple[type[BaseException], ...]:
    from huggingface_hub.errors import RepositoryNotFoundError

    return (RepositoryNotFoundError,)


def upload(
    lock: dict[str, Any],
    artifacts_dir: Path,
    api: Any,
    token: str,
    *,
    packages: Sequence[str] | None,
    create_private_repo: bool,
    not_found: tuple[type[BaseException], ...],
) -> str:
    """Upload the locked files in one commit and return its hash. Raises `StoreError`."""
    if lock["status"] != store.STATUS_PLANNED:
        raise store.StoreError("the lock already has a revision: these files were uploaded")
    store.check_local(artifacts_dir, lock, packages)  # before any network call
    repo_id = lock["store"]["repo_id"]
    try:
        info = api.repo_info(repo_id, repo_type="model", token=token)
    except not_found:
        if not create_private_repo:
            raise store.StoreError(
                f"the repository {repo_id} does not exist; nothing was created "
                "(create it as PRIVATE yourself, or pass --create-private-repo)"
            ) from None
        api.create_repo(repo_id, repo_type="model", private=True, token=token)
        info = api.repo_info(repo_id, repo_type="model", token=token)
    if getattr(info, "private", None) is not True:
        raise store.StoreError(f"the repository {repo_id} is not private; nothing was uploaded")

    from huggingface_hub import CommitOperationAdd

    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(artifacts_dir / path))
        for path in store.remote_paths(lock, packages)
    ]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=COMMIT_MESSAGE,
        token=token,
    )
    revision = str(getattr(commit, "oid", ""))
    if not store._COMMIT.match(revision):
        raise store.StoreError("the upload returned no commit hash; check the repository")
    return revision


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    api_factory: Callable[[str], Any] = _default_api,
    not_found: Callable[[], tuple[type[BaseException], ...]] = _not_found_errors,
) -> int:
    args = build_parser().parse_args(argv)
    environment = os.environ if env is None else env
    token = environment.get(TOKEN_VARIABLE) or None
    try:
        if args.write_lock:
            if args.upload:
                raise store.StoreError("use --write-lock and --upload in separate runs")
            existing = store.read_lock(args.lock) if args.lock.exists() else None
            if existing and existing["status"] == store.STATUS_UPLOADED:
                raise store.StoreError("the lock already says 'uploaded'; it is not rewritten")
            repo_id = args.repo_id or (
                existing["store"]["repo_id"] if existing else DEFAULT_REPO_ID
            )
            names = args.packages or (
                sorted(existing["packages"]) if existing else DEFAULT_PACKAGES
            )
            lock = store.build_lock(args.artifacts_dir, names, repo_id)
            store.write_lock(args.lock, lock)
            print(f"Wrote {args.lock} (status planned). Nothing was uploaded.")
        else:
            lock = store.read_lock(args.lock)
        store.check_local(args.artifacts_dir, lock, args.packages)
        for line in store.plan_lines(lock, args.packages):
            print(line)
        if not args.upload:
            print("Dry run: no token was read and nothing was sent.")
            return 0
        fingerprint = store.plan_fingerprint(lock)
        if args.approve_plan != fingerprint:
            raise store.StoreError(
                "refusing to upload: --approve-plan must equal the plan fingerprint shown above"
            )
        if args.packages and sorted(set(args.packages)) != sorted(lock["packages"]):
            raise store.StoreError("upload all packages in the lock (the approval covers all)")
        if not token:
            raise store.StoreError(f"refusing to upload: {TOKEN_VARIABLE} is not set")
        revision = upload(
            lock,
            args.artifacts_dir,
            api_factory(token),
            token,
            packages=args.packages,
            create_private_repo=args.create_private_repo,
            not_found=not_found(),
        )
        lock["status"] = store.STATUS_UPLOADED
        lock["store"]["revision"] = revision
        store.write_lock(args.lock, lock)
        print(f"Uploaded in commit {revision}; the lock was updated.")
        print("Next: run scripts/download_artifacts.py into an empty folder to verify the upload.")
        return 0
    except store.StoreError as error:
        print(f"REFUSED: {_redact(str(error), token)}", file=sys.stderr)
        return 2
    except Exception as error:  # network or API failure: report the kind, never the token
        print(f"FAILED: {type(error).__name__}: {_redact(str(error), token)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
