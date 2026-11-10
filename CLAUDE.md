# CLAUDE.md

Guidance for Claude Code when working in this repository. Project background: [SCOPE.md](SCOPE.md). Roadmap: [project-notes/PLAN.md](project-notes/PLAN.md).

## Working style: the owner drives, Claude guides
- Work **one step at a time**.
- When the owner says **"next"**: explain exactly one step (what, tools, recommended approach and reasons, alternatives, expected outcome, verification) and **stop**. Do not implement.
- When the owner says **"go"**: implement only that approved step. Then report what was created and modified, the commands run and their real results, what could not be verified, and stop.
- Never move to the next step on your own. Silence or a general acknowledgement is not approval.
- If something is ambiguous between discussing and implementing, ask.
- Explain in plain, simple English. Explain technical terms when they first appear.

## Hard rules
- **No git actions** unless the owner explicitly asks: no `git init`, commit, push, or creating a remote. The GitHub repository is intentionally public (owner's statement, 2026-11-10) and the project is noncommercial. Never change the repository's visibility.
- **Do not edit `project-notes/PLAN.md` without approval.** When a change is agreed, update it and summarise the change.
- Do not add dependencies without explaining why.
- Do not implement future steps early, and do not create empty packages in advance.

## Records to keep up to date
- **`decisions.md`** (project root) is the running log of every decision and significant action. At the end of each step, append entries: what was decided or done, why, files created or modified, commands run and their real results, and anything unverified. Never delete past entries; if a decision changes, add a new entry and mark the old one `Superseded`.
- **`project-notes/each-step-notes.md`** (private, gitignored) holds short plain-language summaries for the owner. After each step is executed, add a summary of that step: what it was for, how it was done, and what was found, in simple, easy English (see the existing entries for the style). When a phase is completed, write that phase's summary in the same file. Never mark a step or phase as done in it unless it was actually run and verified.
- **ADRs** (`docs/adr/`, tracked) are written only for real checkpoint decisions. `project-notes/` is gitignored and holds private working notes only.
- **`.gitignore`**: whenever a new file or folder should stay out of git (generated output, secrets, datasets, model files, caches, local tool files), add it and note it in `decisions.md`. Tell the owner what was added.

## Truthfulness
- Never claim something works unless it was run. Report failures and warnings as they are.
- Label claims as verified, assumed, or recommended. Mark unverified items **[Unverified]**.
- Prefer official documentation for technical claims.
- Synthetic-data resource measurements (Phase 0F) are never presented as real-data quality results, and never used alone to choose a backbone or hosting plan.

## Environment
- Machine: MacBook Air M1, 8 GB RAM. Use CPU by default and run heavy experiments sequentially, one process at a time.
- Python is managed with **uv** (Python 3.12, see `.python-version`). Use `uv sync --all-packages`, `uv run ...`, and `uv lock`. Do not use `pip install` directly.
- Workspace layout: `core/` (`defect_core`, shared logic) exists now. `ml/` is created when Phase 2 begins and `backend/` in Phase 5. Phase 1 tooling lives in `scripts/data/` and the committed split lists in `splits/` (open item O1 is resolved).

## Project rules to keep
- `defect_core` is the only definition of preprocessing, model loading, inference, calibration and rendering; training, evaluation, the CLI, and the API all use it.
- The MVTec test set stays sealed until the model artifact is frozen and calibrated. The threshold comes from held-out *training* normals, never from test images.
- Do not use anomalib's `Engine` or `MVTecAD` datamodule defaults for the deployed artifact (their validation and threshold defaults use test data).
- Anomaly scores are not probabilities or confidence. Verdicts are "demo decisions", never "pass" or "fail".
- Never commit the dataset (`data/`), model artifacts, or secrets.

## Tooling status
- Ruff and pytest are set up (step 0.3). Use `make lint`, `make format`, `make test`, `make setup`. Run `make lint` and `make test` after code changes. `make test` runs `core/tests` and `scripts/data/tests`. Strict Mypy and pre-commit are deferred.
- CI is introduced gradually. Stage 1 (Python lint and test on Linux) is in `.github/workflows/ci.yml` (step 0.5). Its first run on GitHub (commit `90624b0`, 2026-11-09) concluded `success` on Ubuntu x86_64, with every step passing (verified from GitHub's run and step results; the raw logs were not readable without a login). Later stages arrive with their components (backend and Docker in Phase 5, frontend in Phase 6, end-to-end in Phase 11).
- The GitHub remote exists (`origin`). Do not push, or change anything on GitHub, unless the owner asks.
