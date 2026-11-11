# ADR-0003: Artifact store for the model packages (open decision O3)

- **Status:** Decided by the owner on 2026-11-11: a **private Hugging Face model repository** is the
  planned store for both `bottle` model packages. **Nothing has been created or uploaded.** The
  repository does not exist yet, no token exists, and no upload has been approved.
- **Licence status: UNVERIFIED.** Choosing a private store does **not** establish that uploading the
  weights is permitted. See "Licence" below. This ADR is not legal advice.

## What is stored

Two packages, each exactly three files (`model.safetensors`, `manifest.json`, `NOTICE.md`), built by
Phase 4 and kept locally in `artifacts/` (not in git):

| Package folder | Weights | Manifest | NOTICE |
|---|---|---|---|
| `bottle_wide_resnet50_2_tv_in1k_v1.0.0_release` (headline) | 204,703,520 bytes | 48,019 bytes | 2,852 bytes |
| `bottle_resnet18_tv_in1k_v1.0.0_release` | 37,425,984 bytes | 23,101 bytes | 2,845 bytes |

Total 242,206,321 bytes. Sizes and SHA-256 fingerprints of every file are in
[`artifacts.lock.json`](../../artifacts.lock.json) (the lock) and were also checked with `shasum -a 256`.

## Decision

- Store: a **private** Hugging Face model repository. Proposed name `shoaibxhassan/defectlens-bottle-models`
  (taken from the Hugging Face account name seen in this environment; **not verified**, and the owner may
  change it before the upload). Layout: one folder per package, the same names as locally.
- Packages are **pinned by commit**: after an upload the 40-character commit hash is written into the lock,
  and downloads fetch that commit, never a branch name.
- Every download is checked against the lock (size and SHA-256 of each file, the package's internal
  consistency, then a full `load_artifact`). Any difference refuses the whole download and installs
  nothing. The check detects a changed, truncated, swapped or missing file. It does not prove the model is
  good, and it does not protect against the lock file itself being replaced (the lock is in git).
- Code: `defect_core.store` (no network, no Hugging Face import), `scripts/upload_artifacts.py`,
  `scripts/download_artifacts.py`. `huggingface-hub` was already installed through anomalib; it is now also
  declared for the scripts (no new package in `uv.lock`).

## Why this store (from the project's measured sizes; store terms are [Unverified], read from memory)

- 205 MB and 37 MB are small enough for a model repository; Docker builds on Render and local clones can
  fetch them by commit.
- A private repository keeps the weights out of public view while the licence question is open. A GitHub
  Release on the public repository would be public; Git LFS free quotas could be used up by repeated builds
  (these two points are from memory and not checked against current terms).

## Credentials (never in source control)

- Two separate environment variables, no command-line token option (shell history), never printed or
  written: `DEFECTLENS_HF_WRITE_TOKEN` (upload script only) and `DEFECTLENS_HF_READ_TOKEN` (download script).
  The generic `HF_TOKEN` and any cached login are deliberately not used, so a write token cannot slip into
  a download or a deployment.
- The owner should create **fine-grained** tokens limited to this one repository: one with write access for
  the upload, one **read-only** for downloads and Render. Whether a token is really that narrow is set on
  Hugging Face and **cannot be verified by the scripts**.
- The scripts refuse to upload to a repository that is not private, and never create one unless
  `--create-private-repo` is passed.

## How an upload is approved

1. `python scripts/upload_artifacts.py` (no flags) prints the plan: repository, visibility, every file with
   size and SHA-256, the licence status and a **plan fingerprint**. It needs no token and sends nothing.
2. The owner reviews the plan and approves it separately.
3. Only then `--upload --approve-plan <fingerprint>` runs; any change to a file, the repository or the
   visibility changes the fingerprint and needs a new approval.
4. Afterwards `scripts/download_artifacts.py` into an empty folder verifies what was uploaded.

## Licence

- The weights are treated, as a precaution, as material derived from MVTec AD, which its own `readme.txt`
  licenses under CC BY-NC-SA 4.0 (non-commercial, share-alike). Whether those terms attach to a fitted model,
  and whether storing it in a private repository is permitted, has **not** been assessed. The terms of the
  ImageNet data behind the pretrained backbone (`bsd-3-clause` weights card) are **not verified**.
- Each package contains `NOTICE.md` with the attribution, the terms as recorded, the non-endorsement
  statement and "not legal advice". The Hugging Face repository's own terms were not read.
- Any step that would make the weights public (a public repository, a public release, a Docker image that
  contains them and is published) needs its own decision.

## Not verified (NOT RUN)

A real upload; a real private repository and its visibility setting; fine-grained token scoping and
read-only behaviour; a real download from Hugging Face; the repository's current size limits and terms; that
`huggingface_hub` 2.4.0's `create_commit` and `hf_hub_download` behave as the scripts assume against the
live service (their signatures were read from the installed package, and the scripts are tested against
fakes only).

## Consequences

- Render and Docker builds need the read-only token as a secret and a pinned lock; that wiring is Phase 5/7
  work and is not done.
- A change to any package means a new version, a new lock and a new approval.
