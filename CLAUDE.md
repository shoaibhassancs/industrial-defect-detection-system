# CLAUDE.md

Guidance for Claude Code when working in this repository. Project background: [SCOPE.md](SCOPE.md). Roadmap: [project-notes/PLAN.md](project-notes/PLAN.md).

## Working style: the owner drives, Claude guides
- Work **one step at a time**, or one small, focused group of closely related steps when doing them together is logical (owner's instruction, 2026-11-11). Never bundle unrelated work, and never propose a whole phase at once.
- When the owner says **"next"**: explain the proposed step, or each step of the proposed small group (what, tools, recommended approach and reasons, why the steps fit together, dependencies, risks, alternatives, expected outcome, verification) and **stop**. Do not implement. "Next" is never approval to start; wait for the owner's approval.
- At the end of every completed-step report, show **only a one-line recommendation** for the next step or related group, and invite the owner to say "next" for the full explanation. Do not include the detailed explanation in that report.
- When the owner says **"go"**: implement only that approved step. Then report what was created and modified, the commands run and their real results, what could not be verified, and stop.
- Never move to the next step on your own. Silence or a general acknowledgement is not approval.
- If something is ambiguous between discussing and implementing, ask.
- Explain in plain, simple English. Explain technical terms when they first appear.
- **`PLAN.md` is a working plan, not a rulebook** (owner's instruction, 2026-11-10). If implementation shows that a planned step no longer fits, explain the mismatch and recommend a change only when it would genuinely improve the project. Do not change the plan for the sake of changing it, and do not follow outdated plan text blindly. The approval rule below still applies: propose first; when a change is agreed, update the plan and related records.

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
- Workspace layout: `core/` (`defect_core`, shared logic) exists now. `ml/` (`defect_ml`, offline training and evaluation tooling) exists since 2026-11-10 and holds the MVTec data guard, the split rules and the fit/calibration loader (`ml/src/defect_ml/data/`); `backend/` (the FastAPI service, package `app`) exists since 2026-11-10 (Phase 5, local only; no Docker yet). `frontend/` (the React web app, a standalone npm project outside the uv workspace, with its own `package-lock.json`) exists since 2026-11-11 (Phase 6, local only). Frontend commands: `make frontend-install`, `make frontend-check`, `make api-contract`, `make evidence` (regenerates `frontend/src/generated/evidence.json` from the saved `metrics.json`; a test fails if it is stale); never use `npm install` without `--save-exact`, and install with `npm ci`. Phase 1 scripts live in `scripts/data/` and the committed split lists in `splits/` (open item O1 is resolved); the scripts import the guard and the split rules from `defect_ml` and must not copy them.

## Project rules to keep
- `defect_core` is the only definition of preprocessing, model loading, inference, calibration and rendering; training, evaluation, the CLI, and the API all use it.
- The MVTec test set stays sealed until the model artifact is frozen and calibrated. The threshold comes from held-out *training* normals, never from test images. The one definition of the training-data guard is `defect_ml.data.mvtec.assert_training_path`; the training loader (`TrainingImages`) has no switch or argument that reaches `test/` or `ground_truth/`. The sealed evaluation (Phase 3, `defect_ml/evaluate.py`, written and run once for `bottle` on 2026-11-10) is a separate, explicit, logged path that runs only on a frozen and calibrated artifact (see `docs/data.md`); do not add a bypass to the training loader.
- Do not use anomalib's `Engine` or `MVTecAD` datamodule defaults for the deployed artifact (their validation and threshold defaults use test data).
- Anomaly scores are not probabilities or confidence. Verdicts are "demo decisions", never "pass" or "fail".
- Never commit the dataset (`data/`), model artifacts, or secrets.

## Tooling status
- Ruff and pytest are set up (step 0.3). Use `make lint`, `make format`, `make test`, `make setup`. Run `make lint` and `make test` after code changes. `make test` runs `core/tests`, `ml/tests`, `backend/tests`, `scripts/data/tests` and `scripts/tests`. `make frontend-check` runs the frontend's format check, lint, type-check, Vitest and build. The frontend's TypeScript types are generated from `frontend/openapi.json` (made by `backend/app/openapi_export.py`); edit none of the generated files (`schema.d.ts`, `openapi.json`, `evidence.json`) by hand. Strict Mypy and pre-commit are deferred.
- **Declare what you import.** Every third-party package that Python code in this repository imports directly must be declared: in `core/pyproject.toml` (or `ml/pyproject.toml`) when `defect_core`'s (or `defect_ml`'s) own code uses it, otherwise in the root `[dependency-groups]` group `scripts` (installed by default). Do not rely on a package arriving indirectly through another. `scripts/tests/test_declared_imports.py` enforces this. After changing dependencies, run `uv lock` and commit `pyproject.toml` and `uv.lock` together, because CI runs with `UV_LOCKED=1`.
- **CI status after commit `8d47a2e` (owner's standing instruction, 2026-11-10):** assume CI ran successfully until the owner says it did not. This is an assumption, not something Claude verified: say so in `decisions.md` when it matters, and do not list "CI not run" as an open caveat in every report.
  - **Exception, 2026-11-10 (resolved):** the owner reported a failing run (Linux: the macOS-only path `/System/Volumes/Data` in `defect_ml.runinfo`; 16 tests), fixed in `decisions.md` entry 063. The owner then reported that CI passed on the fix (owner's report; Claude did not verify it). The standing assumption applies again. Code must still work on Linux: no macOS-only path or command without a fallback.
- CI is introduced gradually. Stage 1 (Python lint and test on Linux) is in `.github/workflows/ci.yml` (step 0.5). Its first run on GitHub (commit `90624b0`, 2026-11-09) concluded `success` on Ubuntu x86_64, with every step passing (verified from GitHub's run and step results; the raw logs were not readable without a login). Stage 3 (frontend format check, lint, type-check, tests and build, a parallel job `frontend`) was written on 2026-11-11 and has not been run on GitHub by Claude (assume it passes, as an assumption, until the owner says otherwise). Later stages arrive with their components (Docker in Phase 5/7, end-to-end in Phase 11).
- The GitHub remote exists (`origin`). Do not push, or change anything on GitHub, unless the owner asks.
