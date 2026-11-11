"""Download the model packages from the private store, verify them, and install them.

Read-only: this script has no upload or delete path. It fetches the files of the commit pinned in
`artifacts.lock.json` (never a branch name), checks every file's size and SHA-256 against the lock,
checks that each package is internally consistent, installs all packages or none, and then loads
each
package with `defect_core.artifacts.load_artifact` (the full artifact check) unless told not to.

The private repository needs a token. Put a **read-only, fine-grained token limited to this one
repository** in the environment variable DEFECTLENS_HF_READ_TOKEN. There is no token option, and the
generic HF_TOKEN variable and any cached login are deliberately not used, so a write token cannot
slip in by accident. The token is never printed or written. (That a given token really is read-only
is set on the Hugging Face side and cannot be checked here.)

Existing package folders are never overwritten: move them away first.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from defect_core import store

REPO = Path(__file__).resolve().parents[1]
TOKEN_VARIABLE = "DEFECTLENS_HF_READ_TOKEN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lock", type=Path, default=REPO / store.LOCK_FILE_NAME)
    parser.add_argument("--dest", type=Path, default=REPO / "artifacts")
    parser.add_argument(
        "--package", action="append", dest="packages", help="package folder (repeatable)"
    )
    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="skip loading each package with defect_core.artifacts.load_artifact",
    )
    return parser


def hub_fetch(token: str | None) -> store.Fetch:
    """The real download: one file of one commit, read-only. No token means no authorisation."""

    def fetch(repo_id: str, revision: str, path_in_repo: str, staging: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                revision=revision,
                repo_type="model",
                local_dir=staging,
                token=token or False,  # False: never fall back to HF_TOKEN or a cached login
            )
        )

    return fetch


def _load_check(folder: Path) -> None:
    from defect_core.artifacts import load_artifact

    load_artifact(folder)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    fetch_factory: Callable[[str | None], store.Fetch] = hub_fetch,
    load_check: Callable[[Path], None] = _load_check,
) -> int:
    args = build_parser().parse_args(argv)
    environment = os.environ if env is None else env
    token = environment.get(TOKEN_VARIABLE) or None
    try:
        lock = store.read_lock(args.lock)
        installed = store.install_verified(lock, args.dest, fetch_factory(token), args.packages)
        for folder in installed:
            print(f"Verified against the lock and installed: {folder}")
        if not args.skip_load_check:
            for folder in installed:
                load_check(folder)
                print(f"Loaded and checked as an artifact: {folder.name}")
        revision = lock["store"]["revision"]
        print(f"Done. Commit {revision} of {lock['store']['repo_id']}.")
        return 0
    except store.StoreError as error:
        print(
            f"REFUSED: {str(error).replace(token, '[redacted]') if token else error}",
            file=sys.stderr,
        )
        return 2
    except Exception as error:  # e.g. ArtifactError from the load check
        message = str(error).replace(token, "[redacted]") if token else str(error)
        print(f"FAILED: {type(error).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
