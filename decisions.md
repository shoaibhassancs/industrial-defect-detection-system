# decisions.md — DefectLens decision and action log

A chronological record of every decision and significant action in this project, so that everything is traceable. Newest entries go at the bottom. Claude appends to this file at the end of each step (see CLAUDE.md).

**Entry format**
- **Type:** `Decision` (a choice the owner or Claude+owner made), `Action` (something that was done), or `Finding` (a verified fact that affects the project).
- **Where:** the phase and step it belongs to.
- **Status:** `Final`, `Candidate` (to be confirmed by a later measurement), `Open`, `Superseded`, or `To confirm`.
- Unverified claims are marked **[Unverified]**.

**Related files:** [project-notes/PLAN.md](project-notes/PLAN.md) (roadmap), [SCOPE.md](SCOPE.md) (scope), [CLAUDE.md](CLAUDE.md) (working rules). Large technical decisions get an ADR in the tracked folder `docs/adr/`, starting with ADR-0002 (model and serving route); there is no ADR-0001 (see entry 030).

---

## Log

### 001 — Working style: owner drives, Claude guides
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Project-wide · **Status:** Final
- **What:** Work proceeds one step at a time. "next" means explain exactly one step. "go" means implement only that step. Claude reports what changed and what was actually verified, then stops. Nothing moves forward automatically.
- **Why:** The owner wants to understand each stage, not have everything implemented at once.

### 002 — PLAN.md is a guide, not a contract
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Project-wide · **Status:** Final
- **What:** `PLAN.md` may be changed with a technical reason. Significant changes are proposed with advantages, disadvantages and impact, and need the owner's approval. `PLAN.md` is updated when a change is agreed.

### 003 — Truthfulness rules
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Project-wide · **Status:** Final
- **What:** Claims are labelled as verified, assumed, or recommended. Nothing is reported as working unless it was run. Failures are reported as they are.

### 004 — No git actions unless asked; repository private until later
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Project-wide · **Status:** Final for the git rules; the "private during development" part is superseded by entry 039
- **What:** Claude does not run `git init`, commit, push, or create remotes unless explicitly asked. The repository stays private during development and goes public later. CI configuration is prepared only after the owner configures the GitHub remote.

### 005 — Code license: MIT, added when the repo goes public
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 13 (or earlier on request) · **Status:** Final
- **What:** Original project code is MIT. Third-party material (dataset, pretrained weights, libraries) keeps its own terms and attribution and is described separately. The `LICENSE` file is added when the repository is about to go public.

### 006 — Initial repository inspection
- **Date:** 2026-11-09 · **Type:** Finding · **Where:** Start-up · **Status:** Final
- **What:** The repository contained only `project-notes/PLAN.md`, `project-notes/project_specifications.md` and a `.code-workspace` file. No code, no data, no git history. `PLAN.md` is in `project-notes/`, not the project root.

### 007 — Split Phase 0 into small steps
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 0 · **Status:** Final
- **What:** Phase 0 becomes steps 0.1 (version preflight and minimal uv project), 0.2 (PatchCore compatibility smoke test), 0.3 (Ruff and pytest), 0.4 (housekeeping), 0.5 (CI stage 1).
- **Why:** The original Phase 0 bundled too much. Splitting lets each piece be understood and verified on its own.

### 008 — Minimal workspace: `core/` first, other packages when needed
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 0 onward · **Status:** Final; the "Phase 1 or 2" timing and open item O1 are superseded by entry 040 (`ml/` is created when Phase 2 begins; O1 resolved)
- **What:** Start with `core/` only. `ml/` is created when first needed (Phase 1 or 2), `backend/` in Phase 5. No empty packages in advance.

### 009 — Ruff and pytest early; strict Mypy and pre-commit deferred; CI gradual
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 0 step 0.3 and later · **Status:** Final
- **What:** Ruff and pytest come in step 0.3. Strict Mypy and pre-commit wait until they add practical value. CI is staged: Python lint and test (step 0.5, after the remote exists), backend and Docker (Phase 5), frontend (Phase 6), end-to-end (Phase 11).

### 010 — Frontend scaffold moved to Phase 6
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 6 · **Status:** Final
- **Why:** There is no need for Node tooling until there is a UI to build.

### 011 — Licensing workflow removed; keep basic attribution only
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 1, `PLAN.md` §5.2 · **Status:** Final
- **What:** No licensing phase, gate, dataset-derived-material register, or contact with MVTec. Dataset download and implementation do not wait on any licensing review. MVTec AD stays the dataset. Basic attribution (dataset, creators, license name and link, citations, a note on processed images) and a non-commercial notice go in the README, About page, `docs/data.md` and `DATA_LICENSE.md`.
- **Supersedes:** the earlier proposal for steps L1–L3 and the sample/artifact publication gates (assumptions A8, A9, A13 withdrawn).

### 012 — Three separate activities: compatibility, resource feasibility, real-data quality
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phases 0, 0F, 3 · **Status:** Final
- **What:** (1) Compatibility smoke test (step 0.2) shows the stack runs. (2) Phase 0F measures resource cost on synthetic data. (3) Phase 3 measures detection quality on the sealed real test set. Their results are never mixed or substituted for one another.

### 013 — New Phase 0F: synthetic resource feasibility (F1–F3)
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 0F · **Status:** Final
- **What:** F1 measures fit time, peak memory, memory-bank size and per-image latency. F2 verifies save and reload. F3 repeats measurements in a memory- and CPU-limited Linux container. Runs are CPU-only and sequential because the Mac has 8 GB RAM. Scripts go in `scripts/feasibility/`, results in `experiments/feasibility/`. ARM64 container results are a proxy for Render, not proof.
- **Limits:** synthetic runs say nothing about detection quality. The assumption that memory-bank size and fit time do not depend on image content is **[Unverified]** until Phase 2.

### 014 — `safetensors` is a candidate, not a decision
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** F2, Phase 2 and 4 · **Status:** Candidate
- **What:** The artifact format is confirmed or replaced by the simplest reliable alternative after F2 tests it against the actual anomalib model state.

### 015 — Backbone rule and memory target
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Checkpoint A (`PLAN.md` §5.4) · **Status:** Final as a rule; the choice itself is pending measurements
- **What:** Synthetic measurements never select a backbone or hosting plan. `wide_resnet50_2` is the primary candidate and `resnet18` the secondary. Both are evaluated on the sealed test set and both are reported. The rule naming the headline model is written into ADR-0002 before test access. The ~400 MB serving figure is a measurement objective, not a requirement, and a model is not downgraded solely to fit a hosting plan.
- **Note:** Wording of this rule was proposed by Claude and approved by the owner's general approval of the revised roadmap.

### 016 — Home page makes no API calls; cold-start message at inspect time
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 6, `PLAN.md` §1.3, §7, §9.4 · **Status:** Final
- **Why:** Polling `/ready` from the Home page would wake the sleeping free-tier API for every visitor. The backend is now contacted only when an inspection is requested, with a message explaining the wait.

### 017 — Heatmap scaling rule fixed before sealed test access
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 3 step 1, `PLAN.md` §5.6 · **Status:** Final
- **What:** The lower percentile and the numeric upper-bound multiple are written into the config before any test-set access.

### 018 — Proxy-aware client-IP handling for rate limiting
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 5 · **Status:** Final
- **Why:** Behind Render's proxy, all requests could otherwise share one IP and one rate limit. The exact configuration is verified against Render's documentation at that time.

### 019 — Add a local `infer` CLI (Phase 4.5)
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 4.5, requirement C10 · **Status:** Final
- **Why:** The project specification (§8) asks for a clear explanation of how to run inference locally.

### 020 — PLAN.md updated to the approved roadmap
- **Date:** 2026-11-09 · **Type:** Action · **Where:** `project-notes/PLAN.md` · **Status:** Done
- **What:** Applied decisions 007–019 to `PLAN.md` using a checked script (each replacement had to match exactly once). Added Phase 0F and Phase 4.5, rewrote Phase 0–2, §5.2, the §5.4 backbone rule, and §14 (new assumptions A14–A16 and open items O1–O3). Added item 12 to §15. A backup of the previous version was kept in the Claude scratchpad, outside the project.
- **Checked:** searched for stale references (license gates, status chip, "private by default") and fixed those found. The whole file was not re-read line by line.

### 021 — Step 0.1 version preflight and minimal uv project
- **Date:** 2026-11-09 · **Type:** Action + Finding · **Where:** Phase 0 step 0.1 · **Status:** Done
- **Verified facts (this Mac, macOS arm64):**
  - Python **3.12.16** (found by uv in Homebrew), anomalib **2.7.0**, torch **2.15.0**, torchvision **0.29.2**.
  - anomalib 2.7.0 lists Python 3.10–3.12, so Python 3.12 is the newest it supports.
  - anomalib does not install torch by default; torch and torchvision come from its `cpu` extra, so `core/` depends on `anomalib[cpu]`.
  - `uv lock` resolved 98 packages; `uv sync --all-packages` installed them; `defect_core` and `PatchcoreModel` import. `.venv` is 1.1 GB on disk.
  - macOS download is about 280 MB by Claude's sum of lockfile wheel sizes (an estimate, not a measured download).
  - `freia` is source-only and was built locally without problems.
- **Not verified:** installation on Linux (linux/amd64) was not tested, only resolved in the lock. Torch on Linux likely needs the CPU-only index for Docker **[Unverified]**.
- **Files created:** `.python-version` (`3.12`), root `pyproject.toml` (uv workspace, member `core`), `core/pyproject.toml` (package `defect-core`, build backend hatchling), `core/src/defect_core/__init__.py`, `uv.lock`, `.venv/`.
- **Decision recorded:** Python is pinned to 3.12, replacing the earlier "to be determined" status in `PLAN.md` (the `PLAN.md` text itself has not yet been updated with the verified versions).

### 022 — SCOPE.md and CLAUDE.md created
- **Date:** 2026-11-09 · **Type:** Action · **Where:** project root · **Status:** Done
- **What:** `SCOPE.md` summarises scope from `PLAN.md` and the spec. `CLAUDE.md` records the working rules (decisions 001–004, 009, 012, 015), environment (uv, 8 GB Mac), and project rules (shared `defect_core`, sealed test set, no anomalib Engine defaults, no probabilities wording). No new project decisions were made in them.

### 023 — `.gitignore` created; `project-notes/` ignored
- **Date:** 2026-11-09 · **Type:** Action + Decision · **Where:** project root, Phase 0 step 0.4 (done early at owner's request) · **Status:** To confirm
- **What:** `.gitignore` covers the dataset (`data/`), artifacts, model weight files, Python caches and `.venv/`, secrets (`.env*`, with `.env.example` kept), Node/frontend build output, editor and OS files (including `.vscode/`, `*.code-workspace`, `.DS_Store`), local Claude settings, and logs. `uv.lock`, `pyproject.toml`, `.python-version`, `CLAUDE.md`, `SCOPE.md` and `decisions.md` are deliberately **not** ignored.
- **`project-notes/` ignored:** the owner asked for "python-notes". No such folder exists, so Claude read this as `project-notes/` (contains `PLAN.md`, the spec, and later ADRs and the experiment log). **To confirm with the owner.** Consequence: once git is used, those notes will not be tracked or published unless this line is removed.
- **Verified:** the patterns were tested in a throwaway git repo in the scratchpad (not in the project). Ignored and tracked paths behaved as intended.
- **Note:** `*.code-workspace` and `.vscode/` are ignored as editor-local files; this is Claude's choice and can be reversed.
- **Ongoing rule:** Claude updates `.gitignore` whenever a new file or folder should stay out of git.

### 024 — `decisions.md` created as the running log
- **Date:** 2026-11-09 · **Type:** Action · **Where:** project root · **Status:** Done
- **What:** This file. Located in the project root (not in `project-notes/`) so it stays visible if `project-notes/` is ignored. Entries 001–023 were back-filled from the planning conversation and step 0.1.

### 025 — Step 0.2 PatchCore compatibility smoke test
- **Date:** 2026-11-09 · **Type:** Action + Finding · **Where:** Phase 0 step 0.2 · **Status:** Done (passed)
- **What:** Created `scripts/smoke_patchcore.py`. It blocks network access, builds a `resnet18` `PatchcoreModel` with `pre_trained=False`, runs a train-mode pass on 4 random 128×128 images, builds the memory bank with `subsample_embedding(0.1)`, runs an eval-mode pass, checks shapes and finiteness, and constructs (without running) a `wide_resnet50_2` model.
- **Command:** `uv run python scripts/smoke_patchcore.py`. Run twice. Both runs: 10 of 10 checks passed, exit code 0 (exit code confirmed on the second run).
- **Verified facts:**
  - `pre_trained=False` made no network attempts (network was blocked and nothing failed).
  - An eval-mode pass with an empty memory bank raises `ValueError`, as `PLAN.md` §9.1 expects. So the real call order is: train-mode pass, then `subsample_embedding`, then eval-mode pass.
  - resnet18 with layers `layer2` and `layer3` gives 384-dimensional patch features; 4 images at 128×128 gave 1024 patches, and a 10% coreset gave a bank of shape (102, 384), float32.
  - A `wide_resnet50_2` `PatchcoreModel` built through layer3 has 24,862,528 parameters (about 99 MB in float32). The feature extractor appears to keep only the layers up to those requested, rather than the full 68.9M-parameter network **[Unverified: inferred from parameter count]**. This is relevant to Phase 0F memory estimates.
  - **Reproducibility finding:** the coreset step is not made deterministic by `torch.manual_seed` alone. `KCenterGreedy` uses a `SparseRandomProjection(eps=0.9)` with `random_state=None`. In a scratch test (outside the project), two identical runs differed (max score difference 0.0108) with only torch seeded, and were exactly identical once `numpy.random.seed` was also set. Fits must therefore seed both torch and numpy (relevant to `PLAN.md` §9.3 "repeatability" and Phase 2). The sklearn mechanism is **[Unverified]**; only the observed effect was tested.
- **Warnings seen (harmless):** timm and anomalib `FutureWarning`s about deprecated import paths, and a `DeprecationWarning` on `subsample_embedding` about its `embeddings` argument (we do not pass it).
- **Not verified:** anything about detection quality, real-size memory or time, save/reload, Linux, or Render. MPS reported available but was not used.
- **Files created:** `scripts/smoke_patchcore.py`.
- **Next planned:** step 0.3 turns this into a pytest test, adds Ruff and a minimal Makefile. The seeding finding should be reflected in `PLAN.md` when the owner approves a plan edit.

### 026 — PLAN.md updated with Step 0.1–0.2 findings
- **Date:** 2026-11-09 · **Type:** Action · **Where:** `project-notes/PLAN.md` · **Status:** Done
- **What:** On the owner's instruction ("do the recommended changes"), edited `PLAN.md` to record: the verified versions (Python 3.12.16, anomalib 2.7.0, torch 2.15.0, torchvision 0.29.2) and that torch arrives via `anomalib[cpu]` (§3.1); the verified fit call order and empty-bank `ValueError` (Phase 2 step 2); the requirement to seed both `torch` and `numpy` for repeatable fits (§9.3); updated assumptions A10 and A11 to their verified-so-far status; new item 13 in §15. Edits were applied with a script that required each target line to match exactly once.
- **Still unverified and kept marked as such:** Linux installation, and the exact mechanism behind the seeding effect.

### 027 — Step 0.3 Ruff, pytest, and Makefile
- **Date:** 2026-11-09 · **Type:** Action + Decision · **Where:** Phase 0 step 0.3 · **Status:** Done (passed)
- **What:** Added a `dev` dependency group (`ruff` 0.17.4, `pytest` 9.2.0, plus pluggy and iniconfig) to the root `pyproject.toml`, with Ruff settings (Python 3.12, line length 100, rules E, W, F, I, UP, B) and pytest settings (`testpaths = core/tests`). Created `core/tests/test_patchcore_smoke.py` with 5 tests (no network use when building without pretrained weights; primary backbone name accepted; eval with an empty bank is rejected; fit-then-score shapes and finiteness; same torch+numpy seeds give identical scores). Created a `Makefile` with `setup`, `lint`, `format`, `test` only.
- **Decisions:**
  - `scripts/smoke_patchcore.py` is **kept** (recommended default; the owner approved without objecting). It duplicates the tests but prints readable output for manual runs. Revisit later.
  - Modest Ruff rule set for now; strict Mypy and pre-commit stay deferred (entry 009).
- **Results (real):** first `make lint` found 7 style issues in the existing script (one import order, six lines over 100 characters; no logic bugs). Fixed them. After that: `make lint` passes; `make test` gives 5 passed in about 8–11 s, run twice with exit code 0. The edited smoke script still passes (exit 0).
- **Tool checks (scratchpad throwaway files, outside the project):** Ruff flagged an unused import, an unsorted import and an undefined name (exit 1); pytest failed on a wrong assertion.
- **Warnings seen (harmless):** the timm and anomalib `FutureWarning`s and the `subsample_embedding` `DeprecationWarning`, now shown in the pytest summary.
- **Not verified:** CI (step 0.5), Linux.
- **Files created:** `Makefile`, `core/tests/test_patchcore_smoke.py`. **Modified:** `pyproject.toml`, `uv.lock`, `scripts/smoke_patchcore.py` (import order and line wrapping only), `CLAUDE.md` (tooling status), `decisions.md`.
- **`.gitignore`:** no change needed; `.ruff_cache/` and `.pytest_cache/` were already listed.

### 028 — Deleted `scripts/smoke_patchcore.py`
- **Date:** 2026-11-09 · **Type:** Decision + Action · **Where:** Phase 0 (after step 0.3) · **Status:** Done
- **What:** On the owner's approval, deleted `scripts/smoke_patchcore.py` and the then-empty `scripts/` folder. **Supersedes** the "keep" decision in entry 027.
- **Why:** The 5 pytest tests in `core/tests/test_patchcore_smoke.py` cover every check in the script, and they run through `make test` (and later CI). Keeping both meant two copies of the same logic that could drift apart. The two things only the script offered (printed versions and MPS availability, and the `wide_resnet50_2` parameter count) are recorded in this log and move to `docs/dev-environment.md` in step 0.4.
- **Checked:** after deletion, `make lint` passes and `make test` gives 5 passed. The earlier entries 025 and 027 still name the script because they describe what happened at the time.
- **Observation:** a `data/` folder with empty `raw/` and `processed/` subfolders appeared in the project root (created 10:49, not by Claude, presumably by the owner). It is ignored by `.gitignore`. `PLAN.md` expects the dataset at `data/mvtec_ad/`; the layout will be settled in Phase 1.

### 029 — Confirmed: "python-notes" meant `project-notes/`, which stays gitignored
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** `.gitignore` · **Status:** Final
- **What:** The owner confirmed the folder meant in entry 023 is `project-notes/`. It stays ignored by git. **Resolves** the "To confirm" status of entry 023 and open item O4.
- **Consequence:** `PLAN.md`, the specification, and anything else placed in `project-notes/` will not be tracked or published. Documents meant for reviewers (for example ADRs) need a different, tracked location; see the discussion before step 0.4.

### 030 — Step 0.4 housekeeping (revised): README, environment doc; no ADR-0001; `.env.example` deferred
- **Date:** 2026-11-09 · **Type:** Decision + Action · **Where:** Phase 0 step 0.4 · **Status:** Done
- **Decisions (owner approved the recommendation):**
  - **ADR-0001 dropped.** `decisions.md` already records the stack decisions, so an ADR would repeat it.
  - **ADRs from ADR-0002 onward** are written only for real checkpoint decisions (model and serving route, artifact store, compute plan) and live in the **tracked** folder `docs/adr/`, because they are for reviewers and `project-notes/` is gitignored. This replaces the `project-notes/decisions/` location in the earlier plan.
  - **`.env.example` deferred to Phase 5**, since no environment variables exist yet.
  - **`decisions.md` stays in the project root and is not gitignored** (owner's instruction). It will be public if the repository is made public.
- **Files created:** `README.md` (short stub: description, early-development status, setup commands, document links, dataset attribution placeholder, MIT-later note); `docs/dev-environment.md` (machine, tools, library versions, findings so far, commands).
- **Files modified:** `project-notes/PLAN.md` (step 0.4 row, deferred list, deliverables, ADR location in §11 and Phase 2 and Phase 13, `.env.example` in Phase 5 step 1, new §15 item 14); `decisions.md`.
- **Facts recorded (read from the machine, 2026-11-09):** MacBook Air with Apple M1, 8 cores (4 performance, 4 efficiency), 8 GB RAM, macOS 15.2 (24C101), uv 0.13.6, Docker 29.9.2, Node 22.16.0, GNU Make 3.81. Lockfile: 102 packages (81 installed on this Mac).
- **Finding to watch:** free space on the data volume is **11 GiB (95% used)**, down from 19 GiB earlier today. The `.venv` is about 1.2 GB. The MVTec AD download and Docker images will need space, so free space should be checked before Phase 1 (dataset size **[Unverified]**).
- **Checked:** `make setup`, `make lint` (all checks passed) and `make test` (5 passed) all work as the README describes; all relative links in README, docs, SCOPE, CLAUDE and decisions resolve; in a throwaway git repo `README.md`, `docs/` and `decisions.md` are tracked and `project-notes/` is ignored.
- **Corrections made while writing:** the README first named the dataset's publisher from memory; removed as unverified (Phase 1 will take the exact wording from the official page). The package count was first written from an earlier lock and was corrected to 102 / 81 after counting.

### 031 — Plain-language step and phase summaries in `project-notes/each-step-notes.md`
- **Date:** 2026-11-10 · **Type:** Decision + Action · **Where:** `project-notes/each-step-notes.md`, `CLAUDE.md` · **Status:** Done
- **What:** On the owner's request, created `project-notes/each-step-notes.md` with short plain-language summaries (purpose, how it was done, what was found) of steps 0.1–0.4. Steps 0.5 and F1 and the Phase 0 / Phase 0F summaries are placeholders. Added a rule to `CLAUDE.md` ("Records to keep up to date") so every later session adds a step summary after each step and a phase summary when a phase completes.
- **`.gitignore`:** no change needed; `project-notes/` is already ignored (entry 029), so these notes stay private.
- **Files created:** `project-notes/each-step-notes.md`. **Modified:** `CLAUDE.md`, `decisions.md`.

### 032 — Step F2 save and reload verification (synthetic)
- **Date:** 2026-11-10 · **Type:** Action + Finding + Recommendation · **Where:** Phase 0F step F2 · **Status:** Done; format choice **To confirm** (entry 014 stays `Candidate` until the owner confirms)
- **What:** `scripts/feasibility/f2_save_reload.py` builds `PatchcoreModel` (`pre_trained=False`), gives the batch-norm buffers non-default values (stand-in for pretrained weights), fits on 8 random 256×256 images (layers `layer2`/`layer3`, coreset 0.1, torch and numpy seeded), saves in three formats, and reloads each in a **separate Python process** (all process IDs differ), comparing scores and anomaly maps with `torch.equal`. Run for `resnet18` and `wide_resnet50_2`, one process at a time. Synthetic data only: it says nothing about detection quality or real-data memory.
- **Command:** `uv run python scripts/feasibility/f2_save_reload.py run --work-dir <scratch dir>`. Run twice; the second run (kept in the JSON) added an output comparison to the "plain strict load" test. Both runs: every step exit 0.
- **Verified results (both backbones):**
  - **Reload is bit-identical** (max absolute difference 0.0 for scores and maps) from `safetensors` and from a plain state dict (`torch.save`, loaded with `weights_only=True`). The whole-module pickle also reproduces outputs but cannot be loaded with `weights_only=True` (`UnpicklingError`); it needs `weights_only=False`, i.e. pickle execution. This is the kind of file anomalib's `TorchInferencer` reads, and that loader refuses to run unless `TRUST_REMOTE_CODE=1` (read from its source). The whole-module file is a stand-in; `TorchInferencer` itself was **not** run.
  - **What the state is:** `resnet18`: 92 tensors; `wide_resnet50_2`: 260 tensors. They are the backbone parameters, the batch-norm buffers, the `memory_bank`, and one blur kernel.
  - **What is needed to reproduce outputs:** backbone parameters, batch-norm buffers, and the memory bank. Dropping the batch-norm buffers changed the outputs (max score difference 0.85 for `resnet18`, 1.41 for `wide_resnet50_2`). A fresh random model with only the bank assigned gave different outputs. The blur kernel is **not** needed: it is rebuilt in `__init__` and the outputs were identical without it. With no bank, scoring raises `ValueError: Memory bank is empty`.
  - **Bank loading:** a plain `load_state_dict(state, strict=True)` of the whole state, bank included, into a freshly built model worked and gave identical outputs. Plain PyTorch would reject the size mismatch of the empty bank buffer; anomalib resizes it on load (`DynamicBufferMixin`, `anomalib/models/components/base/dynamic_buffer.py`). The explicit "load everything else, then assign the bank" path also gave identical outputs.
  - **The fit pass did not change the batch-norm buffers** in this setup.
  - **Sizes (bytes):** `resnet18` file sizes: safetensors 12,422,836, state dict 12,447,861, whole module 12,458,841. `wide_resnet50_2`: safetensors 104,677,524, state dict 104,745,301, whole module 104,764,387. The bank in this tiny test was 819 rows (384 or 1536 dimensions, float32); a real fit will have far more rows, so sizes here are not an estimate of the real artifact.
- **Recommendation (for the owner to confirm):** `safetensors` for the weights and bank, plus `manifest.json`. Reasons: bit-identical reload (verified), no pickle (safetensors holds only tensors), file size essentially the same as the alternatives. A plain state dict loaded with `weights_only=True` also met the criteria, so the difference is small; `safetensors` is the plan's candidate and avoids depending on `torch.load` behaviour. Whole-module pickle is not recommended (needs `weights_only=False`).
- **Things the artifact code must still handle (from these findings):** save the three groups above; record backbone name, layers and `num_neighbors` in the manifest (they are constructor arguments, not tensors); do not rely on anomalib's resize-on-load alone, so add an explicit check that the bank shape and checksum match the manifest.
- **Dependency note:** `safetensors` (0.8.0) is installed through `uv.lock`, but `core/pyproject.toml` does not list it. It should be added there explicitly when `defect_core` uses it (Phase 2). Not done now.
- **Not verified:** Linux/amd64 reload, or bit-identity across platforms (§9.3); real pretrained weights (batch-norm values were simulated); a float16 bank; anomalib's Lightning `.ckpt` path and `TorchInferencer` itself; realistic bank sizes; behaviour with a different torch thread count (both processes used 4 threads).
- **Environment during the run:** battery power; swap use rose from about 340 MB to 765 MB in the first run and from 750 MB to 1,456 MB in the second (swap total grew from 1 GB to 2 GB), so the Mac was under memory pressure even for this light job. `make test` afterwards: 5 passed in 7.7 s (it took 74 s before the restart, which is consistent with the earlier slowdown being swap-related; the cause was not proven).
- **Files created:** `scripts/feasibility/f2_save_reload.py`, `experiments/feasibility/f2_save_reload.json` (the `scripts/feasibility/` and `experiments/feasibility/` folders). **Modified:** `decisions.md`, `project-notes/each-step-notes.md`. Model files from the runs were written to the Claude scratchpad (outside the project, about 343 MB per run), not into the repository.
- **`.gitignore`:** no change needed. The JSON is small and meant to be committed (`PLAN.md` §11); weight files were kept outside the project.
- **Checks:** `make lint` passes; `make test` 5 passed. `PLAN.md` was **not** edited; assumption A16 (safetensors can represent the model state) is now verified for this stack and should be updated there with the owner's approval.

### 033 — Git initialised, first local commit; GitHub push blocked
- **Date:** 2026-11-10 · **Type:** Action · **Status:** Superseded in part by entry 034 (the commit hash, file count, and the "no remote exists" statement below are out of date)
- **What:** On the owner's explicit request, ran `git init -b main` and made one commit (`58318ae`, 16 files). `project-notes/`, `data/`, `.venv/` and caches are ignored and were not committed. Tried `gh repo create industrial-defect-detection-system --private`; it failed because `gh` is not logged in.
- **Still to do:** owner runs `gh auth login`, then the private repo (same name as the folder) is created and pushed. **[Unverified]** until then: no remote exists.

### 034 — Correction: actual Git state (commit amended, remote exists and is in sync)
- **Date:** 2026-11-10 · **Type:** Finding (correction of entry 033) · **Where:** repository · **Status:** Final
- **Verified facts (read-only checks on 2026-11-10; Claude ran no git write commands):**
  - The reflog shows the first commit `58318ae` was **amended by the owner** into `f9fad09` ("Initial commit: Phase 0 environment setup and Phase 0F step F2"). The commit has **15 files**, not 16. I did not find out which file differs.
  - The tracked files include no dataset, no `project-notes/`, no `.venv/`, and no model weight files. The largest tracked file is `uv.lock` (164 KB).
  - A remote `origin` exists: `https://github.com/shoaibhassancs/industrial-defect-detection-system.git`. `git ls-remote origin` returned `f9fad09…` for `HEAD` and `refs/heads/main`, the same commit as the local `main`, which tracks `origin/main`. The remote and the push were done by the owner outside this session, not by Claude.
- **Privacy of the GitHub repository:** **not confirmed.** An unauthenticated request to `https://api.github.com/repos/shoaibhassancs/industrial-defect-detection-system` returned HTTP 404. That is what GitHub returns for a private repository to someone who is not signed in, so it is consistent with "private", but it would also be the answer for a repository that does not exist. Since `git ls-remote` found the repository with the owner's stored credentials, the "does not exist" reading is ruled out. Still, the owner should confirm it on GitHub (see the step 0.5 report).
- **Supersedes:** the commit hash, file count and "no remote exists" parts of entry 033.

### 035 — Step 0.5 CI stage 1: workflow written, not yet run
- **Date:** 2026-11-10 · **Type:** Action + Finding · **Where:** Phase 0 step 0.5 · **Status:** Written; later verified by a green GitHub run (see entry 037)
- **What:** Created `.github/workflows/ci.yml`. On pushes to `main`, pull requests, and manual runs it checks out the code, installs uv, runs `uv sync --all-packages`, then `make lint` and `make test` on `ubuntu-latest` (x86_64, like Render). Details: `UV_LOCKED=1` so a stale `uv.lock` fails the run; uv pinned to 0.13.6 (same as the development Mac); Python 3.12 comes from `.python-version`; uv cache kept with `prune-cache: false`; read-only token permissions; newer pushes cancel older runs; 45-minute timeout; two disk-space printouts and a torch version printout to help diagnose the download concern below.
- **Actions pinned to commit hashes:** `actions/checkout` v7.0.1 (`3d3c42e5…`) and `astral-sh/setup-uv` v10.3.3 (`68559e99…`). The latest versions and the commits behind their tags were read from the GitHub API on 2026-11-10, and the inputs used (`version`, `enable-cache`, `cache-dependency-glob`, `prune-cache`, `persist-credentials`) were checked in each action's `action.yml` at those commits.
- **Checks done:** the workflow YAML parses and has the expected structure (job, steps, triggers, pinned `uses`). `make lint` still passes. **Not done:** the workflow has not run anywhere. No linter for workflows (`actionlint`) is installed, so GitHub may still reject something. Nothing was pushed.
- **Dependency and download concern (from `uv.lock`, not measured):** the lock resolves torch for Linux from PyPI. For x86_64 Linux it lists 19 NVIDIA/CUDA-related packages (plus `triton`) whose largest wheels sum to about **2.45 GB**, and a torch wheel of about **539 MB**. So the first Linux install may download roughly **3 GB** and unpack to more. The Mac gets none of these. `uv sync` has no `--torch-backend` option in uv 0.13.6 (checked with `--help`), so this cannot be fixed inside the workflow alone. The runner's free disk space is unknown **[Unverified]**; the workflow prints it. If the run is too slow or runs out of space, the fix is a Linux-only CPU torch index in `pyproject.toml` plus `uv lock` (planned for the Docker build in Phase 5 anyway). That changes dependency configuration, so it needs the owner's approval first.
- **Still unverified:** that dependencies install on Linux at all (assumption A11), that the 5 tests pass there, and the real size and time of the install.
- **Files created:** `.github/workflows/ci.yml`. **Modified:** `decisions.md`, `project-notes/PLAN.md` (step 0.5 row, assumption A11, new §15 item 15), `CLAUDE.md` (Tooling status), `project-notes/each-step-notes.md`.
- **`.gitignore`:** no change needed.

### 036 — Local verification of the CI workflow (what could and could not be checked)
- **Date:** 2026-11-10 · **Type:** Action + Finding · **Where:** Phase 0 step 0.5 · **Status:** Local checks done; superseded in part by entry 037 (the workflow has since run on GitHub)
- **PASS (local, 2026-11-10):**
  - YAML structure (triggers, `contents: read`, `UV_LOCKED=1`, `ubuntu-latest`, 45-minute timeout, the expected commands).
  - Validation against the SchemaStore `github-workflow` JSON schema (Draft 7): 0 errors. A first attempt reported 2 errors that came from PyYAML reading the bare key `on` as `True` (a YAML 1.1 quirk, not a workflow problem); the re-run with the key restored passed. A deliberately broken copy was rejected, so the validator does catch mistakes. `actionlint` is not installed and was not used.
  - Both pinned action hashes still equal their release tags on GitHub.
  - `make lint` and `make test` targets exist and pass with `UV_LOCKED=1`: lint all passed; 5 tests passed in 7.7 s.
  - `uv sync --all-packages --offline --dry-run` with `UV_LOCKED=1`: lock matches `pyproject.toml`; "would make no changes".
  - The workflow's version printout command runs on the Mac (Python 3.12.16, torch 2.15.0, CUDA build None, anomalib 2.7.0).
  - The local uv version (0.13.6) equals the version pinned in the workflow.
- **NOT RUN:** a real Linux install (not attempted: wrong CPU type for Linux x86_64 wheels, and about 3 GB of downloads), and the workflow itself.
- **Estimate (from `uv.lock` only; no download):** walking the lock's dependency graph with Linux x86_64 markers gives **99 packages** and about **3.25 GB** of wheels, of which 19 NVIDIA/CUDA-related packages plus `triton` are 2.45 GB; `freia` is source-only and would be built on the runner. This refines the earlier rough figure (entry 035) and is consistent with it. The runner's free disk space is still unknown.
- **Free disk on the Mac before and during the checks:** 16 GiB.
- **Only a run on GitHub can show:** that the workflow is accepted, that Python 3.12 and the dependencies install on the runner, the real download size and time, whether the disk is enough, and that the 5 tests pass on Linux.
- **Files changed:** `decisions.md` only. Scratch files (schema copy, estimate script) are in the Claude scratchpad, outside the project. No commit or push was made.

### 037 — Step 0.5 result: the first GitHub Actions run succeeded; repository is now public
- **Date:** 2026-11-10 · **Type:** Finding + Action · **Where:** Phase 0 step 0.5, repository · **Status:** CI stage 1 verified (with limits below); the visibility question is resolved by entry 039 (the owner says public is intentional)
- **How it was checked:** read-only calls to GitHub's public REST API (the repository answered without a login). `gh` is not logged in, so the raw job logs (HTTP 403 without authentication) were **not** readable. Claude pushed nothing and changed nothing on GitHub.
- **Verified (from GitHub's run, job and step records):**
  - Run `42724707634`, workflow "CI", event `push`, branch `main`, commit `90624b0` (the current local `HEAD`), attempt 1: status `completed`, conclusion **`success`**. Started 2026-11-09 23:58:39 UTC. Link: <https://github.com/shoaibhassancs/industrial-defect-detection-system/actions/runs/42724707634>.
  - One job, `lint-and-test`, on a GitHub-hosted `ubuntu-latest` runner: `success`, 138 s in total. Every step is `success`: checkout 1 s, install uv 1 s, disk check 1 s, **install dependencies 61 s**, versions printout 3 s, **lint 0 s, test 16 s**, uv cache save 50 s.
  - The one annotation is a notice that the `ubuntu-latest` label "will migrate to Ubuntu 26 beginning October 19, 2026" (issue 14748 in `actions/runner-images`). It is not a failure.
- **What this proves:** the locked dependencies install on a GitHub Ubuntu x86_64 runner, `make lint` exits 0, and `make test` exits 0 (pytest exits non-zero on any failing test). Assumption A11 (Linux install) is therefore verified for that runner.
- **Not verified (needs the logs):** the exact number of tests passed (the local run has 5), the runner's operating-system version (given the notice above, `ubuntu-latest` may now mean Ubuntu 26.04 or still 24.04), the Python 3.12 patch version used, the real download size, free disk space before and after, and whether torch there reports a CUDA build. The install taking only 61 s on a cold cache suggests the 3.25 GB estimate (entry 036) did not cause trouble, but that is an inference.
- **Repository visibility changed:** on 2026-11-10 an unauthenticated API call returned `"private": false, "visibility": "public"`. Earlier the same kind of call returned 404 (entry 034). So the repository **is public now**. Claude did not change it and does not know who did or why. `CLAUDE.md` and `decisions.md` say the repository stays private until the owner decides. **To confirm with the owner** whether this is intended. What is exposed: the committed files only (`project-notes/`, `data/`, `.venv/`, weights and caches are ignored and were not committed). There is no `LICENSE` file yet, so by default others have no permission to reuse the code; the MIT license is planned for when the repository goes public (entry 005).
- **Supersedes:** the "not verified" statements in entries 035 and 036, and the privacy note in entry 034.
- **Files modified (for entry 037):** `decisions.md`, `README.md` (the Linux sentence), `docs/dev-environment.md` (the Linux paragraph), `CLAUDE.md` (Tooling status), `project-notes/PLAN.md` (step 0.5 row, assumption A11, §15 item 16), `project-notes/each-step-notes.md` (step 0.5 and the Phase 0 summary). No git commands that change anything were run.

### 038 — Phase 1 step 4: dataset verification script (sealed-test-friendly scope)
- **Date:** 2026-11-10 · **Type:** Decision + Action + Finding · **Where:** Phase 1 step 4 · **Status:** Done; the "comparison NOT RUN" part is superseded by entry 039 (it now passes against an owner-supplied table); the "download source unconfirmed" part is superseded by entry 042 (Kaggle, owner-stated)
- **Decisions (owner-approved in this session):** (1) test-side checks are limited to **file names and counts**; the same-size mask check and any hashing of `test/` and `ground_truth/` files move to Phase 3, after the freeze; (2) §14 **O1 stays open**: `scripts/data/verify_dataset.py` is the location for this script only and says nothing about where the split manifests and the EDA notebook go.
- **What was built:** `scripts/data/verify_dataset.py`.
  - For all 15 categories it lists names only and counts `train/good`, `test/good`, each defect type and the masks. It checks by name that every defective test image has a mask called `<name>_mask.png`, ignores hidden files, and flags unexpected folders or non-PNG files.
  - For one category (default `bottle`) it decodes every `train/good` image (`Image.verify()` and a full load) and records size, mode, format, byte size and SHA-256.
  - A guard (`assert_allowed`) raises `PermissionError` for any path outside `<category>/train/good`.
  - It writes `data/data_manifest.json` (gitignored, deterministic: no timestamps or absolute paths) and `experiments/dataset_verification.json` (small; includes the manifest's SHA-256 and the counts of files opened).
- **Results (real run on `data/raw/mvtec-ad/`, exit 0, "RESULT: OK", no issues in any category):**
  - `bottle`: `train/good` 209, `test/good` 20, defective 63 (`broken_large` 20, `broken_small` 22, `contamination` 21), masks 63.
  - All 209 `bottle` training images decode: all 900×900, all RGB, all PNG, 111,806,381 bytes in total, 0 duplicate files.
  - Totals over all 15 categories by file-name count: 3,629 training images, 467 good test images, 1,258 defective test images, 1,258 masks (1,725 test images, 5,354 in all). The totals equal the figures I remember for MVTec AD, but that is memory, not a checked source, so the comparison with the official numbers stays **NOT RUN**.
  - Files opened: 209 training images, **0** files under `test/` or `ground_truth/`.
  - The manifest is identical on a re-run (SHA-256 `db95f8235aa5c81ab74ec6abbacb3ec427b4b0a27555286aad52f98ed52e26e2`, 55,021 bytes). The run took about 3 s.
- **Official counts could not be read:** two attempts through the web-fetch agent failed. `www.mvtec.com` did not resolve (DNS lookup failed, also from the shell), and the CVPR 2019 paper on `openaccess.thecvf.com` returned HTTP 403. Nothing was filled in from memory. To finish the comparison, the owner can paste the official per-category table, or run the check from a network that can reach the page; the script accepts a JSON file via `--expected` (format in its docstring).
- **Download source:** not recorded anywhere. The owner, when asked, gave the folder path (`data/raw/mvtec-ad/`), not the website. Recorded as **unconfirmed** in the summary file. Clues only: the top-level `readme.txt` and `license.txt` carry MVTec's attribution text and are dated 20 Feb 2022. Plan step 3 (compare with the official archive if another source was used) stays open.
- **Tests:** `scripts/data/tests/test_verify_dataset.py` (6 tests on a tiny fake dataset: counts and deep check; a missing mask is reported from names alone; a corrupt training image is reported; the guard refuses `test/` and `ground_truth/` paths; the manifest is deterministic; the official-comparison logic detects a mismatch). The test and mask files in the fake dataset are empty, so any attempt to open them would fail the run. `pyproject.toml` `testpaths` now includes `scripts/data/tests`. A first run of two tests failed because the fake dataset had only one category while the script (correctly) requires all 15; the test data was fixed, not the script.
- **`.gitignore` change (found and fixed during this step):** the pattern `data/` also matched `scripts/data/`, so git and Ruff would have silently skipped the new script and tests (Ruff still reported 8 files after they were added; CI would have lacked them). It is now `/data/`, which ignores only the top-level dataset folder. Checked with `git check-ignore`: `data/raw` and `data/data_manifest.json` are ignored; the two new files are tracked. The owner should know the pattern changed.
- **Dependency note:** the script imports Pillow (13.0.0), which is installed through torchvision and anomalib but is not declared in `core/pyproject.toml`. It should be declared explicitly when `defect_core` gets its preprocessing (Phase 2). No dependency was added now.
- **Checks:** `make lint` passes (after fixing five over-long lines); `make test` 11 passed (5 existing, 6 new). **Not verified:** the new tests on GitHub Actions, since nothing was pushed.
- **Files created:** `scripts/data/verify_dataset.py`, `scripts/data/tests/test_verify_dataset.py`, `experiments/dataset_verification.json`, `data/data_manifest.json` (gitignored). **Modified:** `.gitignore`, `pyproject.toml`, `decisions.md`, `README.md` ("What exists so far"), `CLAUDE.md` (Tooling status), `project-notes/PLAN.md` (Phase 1 step 4 and acceptance criteria, new §15 item 17), `project-notes/each-step-notes.md`. No commit, push or visibility change was made.
- **Not done on purpose:** the documentation corrections the owner deferred (for example the `data/mvtec_ad/` path in Phase 1 step 1 and the `ml/` wording) are still pending.

### 039 — Repository is intentionally public; project is noncommercial; count comparison completed
- **Date:** 2026-11-10 · **Type:** Decision + Action + Finding · **Where:** repository, Phase 1 step 4 · **Status:** Final for visibility and licensing facts; the "unofficial / Kaggle" description of the count table is **superseded by entry 040**; the "download source still unconfirmed" part is superseded by entry 042
- **Owner statements (recorded as given; Claude makes no legal determination):** the GitHub repository is **intentionally public**, and the project is **noncommercial**. Claude does not change the visibility and will not ask for this to be confirmed again. **Supersedes** the "private during development, public later" parts of entry 004 and `CLAUDE.md`, `SCOPE.md`, `PLAN.md` §5.2 item 7, and resolves the visibility question in entries 034 and 037.
- **Facts that stay true:** only committed files are public (`project-notes/`, `data/`, `.venv/`, weights and caches are not committed). The dataset is not in the repository and keeps its own license (CC BY-NC-SA 4.0) and attribution. Entry 005 said the MIT `LICENSE` file would be added when the repository goes public; the repository is public now and **no `LICENSE` file exists yet**. Adding it is the owner's decision and was not done.
- **Download source:** still **unconfirmed**. The owner supplied the count table below with a Kaggle link as its source, but did not say the dataset was downloaded from there, so no download source is inferred.
- **Count comparison (Phase 1 step 4):** `scripts/data/expected_counts.json` holds the owner's table (train, normal test, defective test per category, plus totals), with `source_is_official_mvtec: false` and the Kaggle address as the stated source. The table's 15 rows sum to its own totals (3,629 / 467 / 1,258; checked). The script compares **only** `train_good`, `test_good` and `test_defective`, i.e. training images, normal test images and defective test images. It does **not** compare mask files or defect regions, and it now raises an error if a "masks" field is put into the file (our observed mask-file counts equal the defective-image counts in every category, but that is a separate fact, not part of the comparison). Totals are compared too.
- **Result: PASS.** On the real run (exit 0, "RESULT: OK"), all 15 categories and the totals match, with no mismatches. `bottle`: 209 / 20 / 63. Totals: 3,629 / 467 / 1,258. The data manifest is unchanged (SHA-256 `db95f8235aa5c81ab74ec6abbacb3ec427b4b0a27555286aad52f98ed52e26e2`), and 0 files under `test/` or `ground_truth/` were opened.
- **What PASS means and does not mean:** the local folder matches the counts the owner supplied from a Kaggle listing. Claude did not read that page, and it is **not** the MVTec website, so this is not a check against the official documentation. It also says nothing about image contents or about where the folder came from.
- **Tests:** `make lint` passes; `make test` 13 passed (5 in `core/tests` and 8 in `scripts/data/tests`, including: pass and mismatch detection, a totals mismatch, rejection of a "masks" field, and a check that the shipped table covers all 15 categories and adds up).
- **Files created:** `scripts/data/expected_counts.json`. **Modified:** `scripts/data/verify_dataset.py` (comparison logic, default expected file), `scripts/data/tests/test_verify_dataset.py`, `experiments/dataset_verification.json` (regenerated), `decisions.md`, `CLAUDE.md`, `SCOPE.md`, `README.md`, `project-notes/PLAN.md`, `project-notes/each-step-notes.md`. No commit, push or visibility change.

### 040 — Correction: the count table comes from the official MVTec AD paper; O1 resolved (option C)
- **Date:** 2026-11-10 · **Type:** Decision + Correction · **Where:** Phase 1 (`PLAN.md` §14 O1, steps 4–6) · **Status:** Final; the statements that the download source is "still unconfirmed" are superseded by entry 042
- **Correction of entry 039:** the owner corrected the source of the expected-count table. It comes from the **official MVTec AD paper** (<https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf>), not from a Kaggle page. `scripts/data/expected_counts.json` now says so (`source_is_official_mvtec: true`).
- **What Claude has and has not checked:** the table's rows sum to its own totals (checked). Claude could **not** open the PDF: on 2026-11-10 the first attempts failed with a DNS error, and a later attempt reached the server but it closed the connection. So the table is **owner-stated and unchecked by Claude against the PDF**; the file records this as `verified_by_claude_against_source: false`. The comparison therefore means "the local folder matches the official paper's counts as the owner transcribed them".
- **Two separate facts, kept separate:** (1) where the counts come from: the official paper (owner-stated); (2) which website the local dataset folder was downloaded from: **still unconfirmed**. The Kaggle link in the owner's first message was a source claim for the table, never a statement about the download, and no download source is inferred. The expected-counts file also carries the field `download_source_of_local_folder: "unconfirmed"`, and a test checks both facts.
- **Supersedes:** the Kaggle / "unofficial table" wording in entry 039 and in `PLAN.md` (Phase 1 step 4, its acceptance bullet, §15 item 18).
- **O1 resolved, option C (owner-approved):** `ml/` is created when Phase 2 begins (nothing created early). Phase 1 tooling lives in `scripts/data/`. The committed split lists live in a new top-level `splits/` folder (not `ml/splits/`), so Phase 2 only reads them and nothing moves. The train-only exploration is a **guarded script** (`scripts/data/eda_train.py`), not a notebook: it reads `train/good` images only (train and calibration), through the same guard as `verify_dataset.py`; it writes a small statistics file `experiments/eda_bottle.json` (committed) and sample sheets to `data/processed/eda/` (gitignored, so dataset images are not republished). A Jupyter notebook would need Jupyter, which is not installed (numpy, Pillow, matplotlib, pandas are); one can still be added under `ml/notebooks/` in Phase 2. **The EDA script is not built yet** (that is Phase 1 step 6).
- **`PLAN.md` changes made to match:** §14 O1 marked resolved; "`ml/` in Phase 1 or 2" changed to "when Phase 2 begins" (§3.1, §8 roadmap diagram, Phase 0 deferred list, Phase 2 dependencies, §11 boundaries); Phase 1 steps 5 and 6 and deliverables; §11 tree (`splits/` at the top level, `scripts/data/`, and the three Phase 1 items removed from under `ml/`); §15 item 19. **Still deferred (not touched):** the `data/mvtec_ad/` path in Phase 1 step 1, the repository description in §11, the licensing leftovers in §15 items 6 and 11, the old ADR path in this file's header, and the stale disk figure in `docs/dev-environment.md`.
- **Files modified:** `scripts/data/expected_counts.json`, `scripts/data/tests/test_verify_dataset.py`, `experiments/dataset_verification.json` (regenerated), `decisions.md`, `CLAUDE.md`, `project-notes/PLAN.md`, `project-notes/each-step-notes.md`.

### 041 — Phase 1 step 5: split lists for `bottle`
- **Date:** 2026-11-10 · **Type:** Action + Finding · **Where:** Phase 1 step 5 · **Status:** Done for `bottle`
- **What was built:** `scripts/data/make_splits.py` and the committed output `splits/bottle.json`.
  - **Method:** the 209 `train/good` images are ranked by SHA-256 of `<seed>:<category>:<file name>`; the first n_calib = round(0.2 × 209) = 42 are `calib`, the other 167 are `fit`. Seed 0 (a fixed default chosen before any result existed; it has no link to data or results). This uses no random-number generator, so a different numpy or Python version cannot change the split.
  - **File names only.** The script never opens an image. The 83 test files are listed by name in the file (`test_files_not_opened`: `good` 20, `broken_large` 20, `broken_small` 22, `contamination` 21) as a record; they are not in `fit` or `calib`. File paths include the folder (`train/good/000.png`, `test/good/000.png`), because training and test files share names such as `000.png`.
  - **Cross-check:** if `data/data_manifest.json` covers the category, the training file set on disk must equal the verified one (it did: the split records that manifest's SHA-256 `db95f8235aa5c81ab74ec6abbacb3ec427b4b0a27555286aad52f98ed52e26e2`).
  - **Safety:** an existing split is never changed silently. Re-running reproduces the identical file; a different result (other seed, changed dataset) is refused unless `--force` is given, because a different split means a new model version (`PLAN.md` §4.5).
- **Results (real run, exit 0):** 209 → fit 167, calib 42; no overlap; fit + calib = 209 distinct files; lists SHA-256 `9e077545997de454131f8060af32936e469aba9a949b6ae93d3967bce9502291`; `splits/bottle.json` is 8,325 bytes. A second run left the file byte-identical. A run with `--seed 1` was refused (exit 1) and left the file untouched. `splits/bottle.json` is not git-ignored.
- **Tests (8 new, `scripts/data/tests/test_make_splits.py`):** rounding (209 → 42, matching `PLAN.md` §5.6), valid partition and determinism, seed and category change the split, same file names in train and test are not confused, validation detects overlap / a test file in `fit` / a tampered hash / a partition that is valid but not the stated method's, existing splits are never changed silently, the manifest cross-check, and the committed `splits/bottle.json` is self-consistent (167 / 42 / 83, seed 0). The test files in the fake dataset are empty, so any attempt to open one would fail.
- **Checks:** `make lint` passes; `make test` **21 passed** (5 in `core/tests`, 16 in `scripts/data/tests`). Not run on GitHub (nothing was pushed).
- **Not done / open:** the exact seed policy for the other categories (Phase 8 says "the same rule and seed policy"; the script takes `--category` and `--seed`, and only `bottle` was generated). Step 6 (the EDA script) is not built. The split is not yet used by anything.
- **Files created:** `scripts/data/make_splits.py`, `scripts/data/tests/test_make_splits.py`, `splits/bottle.json`. **Modified:** see entry 040 for the shared record changes; also `README.md` and `project-notes/each-step-notes.md`. No commit, push or visibility change.

### 042 — Download source recorded (Kaggle); documentation corrections no longer deferred
- **Date:** 2026-11-10 · **Type:** Decision + Action · **Where:** Phase 1 steps 1 and 3, project documents · **Status:** Final for the source statement; contents comparison with the official archive **not done**
- **Owner statement:** the local dataset folder (`data/raw/mvtec-ad/`) came from **Kaggle**. The specific Kaggle dataset page was not stated, and Claude does not assume it is the page the owner linked earlier (that link was given as a source for the count table, which the owner then corrected to the official paper, entry 040). Recorded in `scripts/data/verify_dataset.py` (default `--source`), `scripts/data/expected_counts.json`, `experiments/dataset_verification.json` (regenerated), `README.md`, and `PLAN.md`. **Supersedes** every "download source unconfirmed" statement in entries 038, 039 and 040 (their status lines now say so).
- **What this does and does not mean:** `PLAN.md` Phase 1 step 3 says that if a Kaggle copy is used, its contents and hashes should be compared with the official archive. That is **only partly done**: the per-category counts (training, normal test, defective test) match the official paper's table, but file contents and hashes were not compared. The official site could not be reached from Claude's environment and the official archive has not been downloaded. "The Kaggle copy has the same content as the official archive" is therefore **[Unverified]**; it is new assumption A17 in `PLAN.md` §14. Separately, the dataset's own `readme.txt` and `license.txt` (dated 20 Feb 2022) carry MVTec's attribution and the CC BY-NC-SA 4.0 text.
- **Owner instruction:** documentation corrections are no longer deferred; they are made as they are found, and each is reported with its reason. Corrections made in this step, and why:
  - `PLAN.md` Phase 1 step 1: path `data/mvtec_ad/` → `data/raw/mvtec-ad/`, status "done by the owner, from Kaggle". *Why: the old path did not exist; the real folder is `data/raw/mvtec-ad/`.*
  - `PLAN.md` Phase 1 step 3: status added (what was and was not compared). *Why: the Kaggle copy is now confirmed as the one in use, so the step applies.*
  - `PLAN.md` §11: the sentence "the repository holds only the spec and this plan" replaced by a description of what exists today; the `data/` tree comment now lists `raw/`, `processed/` and the manifest. *Why: both were false.*
  - `PLAN.md` §3.1 (two places): "Linux not yet tested" replaced by a pointer to the CI run. *Why: CI installed and tested on Linux on 2026-11-09 (entry 037).*
  - `PLAN.md` §15 items 6 and 11: "[Superseded 2026-11-10 …]" markers added; the old text is kept. *Why: they still said artifacts are "private by default", samples wait for a "license review" and publishing is an "open legal question", which contradicts decision 011, §5.2 and entry 039.*
  - `PLAN.md` §14: new row A17; §15: new item 20.
  - `decisions.md` header: the ADR location changed from `project-notes/decisions/` to `docs/adr/`, and the ADR-0001 mention removed. *Why: entry 030 moved ADRs to a tracked folder and dropped ADR-0001; the header still showed the old plan.*
  - `docs/dev-environment.md`: disk figure and a swap finding (below). *Why: the 11 GiB figure was stale and the swap behaviour explained slow runs and a shrinking disk.*
  - `README.md`: download source sentence added to "Dataset and attribution". *Why: the repository is public, so readers should know where the local copy came from and that it was not compared with the official archive.*
  - `project-notes/each-step-notes.md`: the "still open: which website" line replaced. *Why: it is answered.*
- **Left unchanged on purpose:** `PLAN.md` line 6 ("Repository state at planning time: only the spec exists"): it is labelled as the state at planning time and is true as history. Past entries in this file keep their original wording; later entries and status lines supersede them.
- **Files modified:** `scripts/data/verify_dataset.py`, `scripts/data/expected_counts.json`, `scripts/data/tests/test_verify_dataset.py` (the test now checks that the counts' source and the download source are different facts), `experiments/dataset_verification.json`, `decisions.md`, `README.md`, `docs/dev-environment.md`, `project-notes/PLAN.md`, `project-notes/each-step-notes.md`.
- **Checks:** `make lint` passes; `make test` 21 passed; the verification script re-run: comparison PASS, manifest hash unchanged (`db95f8235aa5…`), 0 files under `test/` or `ground_truth/` opened.
- **Not verified:** that the Kaggle page is the one the owner linked; Kaggle's own description of the data; contents or hashes against the official archive.

### 043 — Phase 1 step 6 (train-only EDA) and declared dependencies
- **Date:** 2026-11-10 · **Type:** Decision + Action + Finding · **Where:** Phase 1 step 6; `pyproject.toml`, `uv.lock` · **Status:** Done for `bottle`. The step 6 code and the dependency change have **not** run in CI yet (CI last ran on the earlier commit `8d47a2e`).
- **Owner instruction:** the directly imported packages must be declared in a suitable dependency group, not left to arrive indirectly.
- **Audit (read-only script over `core/` and `scripts/`, before any change):** four packages were already imported directly without any declaration: `numpy` (core tests, feasibility script, dataset scripts), `torch` (core tests, feasibility script), `safetensors` (feasibility script), `Pillow` (dataset tests). Declared before: `anomalib` (core), `pytest` (root group `dev`). The new EDA script adds `scipy` and `matplotlib`. So the gap was wider than the new script.
- **Change (why each part):**
  - New root group `scripts` in `pyproject.toml`: `numpy`, `pillow`, `scipy`, `matplotlib`, `safetensors`, `torch`. *Why a group in the root:* all current direct importers are scripts and tests, and `defect_core`'s own code (`core/src`) imports nothing third-party yet. *Why not `dev`:* these are runtime needs of the scripts, not developer tooling. *Why `torch` too:* it is imported directly (and was only present through the `anomalib[cpu]` extra); declaring it also makes a future Linux CPU-only index setting possible, since uv applies such settings only to direct dependencies. No version limits were added; `uv.lock` pins the versions.
  - `[tool.uv] default-groups = ["dev", "scripts"]`. *Why:* `make setup` and CI run `uv sync --all-packages` with no group options, so the group must install by default.
  - `uv lock --offline` updated `uv.lock`: 102 packages as before, **no package version changed**; only the group declarations and `default-groups` were added. `UV_LOCKED=1 uv sync --all-packages --offline --dry-run` then reported "would make no changes".
  - **Phase 2:** when `defect_core`'s own code imports torch, numpy, Pillow or safetensors, declare them in `core/pyproject.toml` too (`PLAN.md` §3.1). This supersedes the dependency notes in entries 032 (safetensors) and 038 (Pillow).
  - **Guard against regressions:** `scripts/tests/test_declared_imports.py` parses every `.py` file under `core/` and `scripts/`, maps each third-party import to its installed package (`importlib.metadata.packages_distributions`), and fails if none of `pyproject.toml`'s dependency lists declares it. Negative control: run against the committed (old) declarations it fails and names numpy, torch, safetensors and Pillow. `pyproject.toml` `testpaths` now has three entries.
  - **Consequence for git:** `pyproject.toml` and `uv.lock` changed together and must be committed together, because CI sets `UV_LOCKED=1`. Not pushed by Claude.
- **EDA script:** `scripts/data/eda_train.py` reads `splits/bottle.json`, refuses to run if the split is inconsistent (it reuses `make_splits.validate_split`), and opens **only** the 209 listed training images, each through `verify_dataset.assert_allowed`. It checks each file's SHA-256 against the verified manifest and refuses to report on any mismatch. Per image: luma mean and spread, mean R/G/B, a sharpness number (mean absolute neighbouring-pixel difference), and file size. Per metric it compares fit and calib with a two-sample Kolmogorov–Smirnov (KS) test, and flags robust-z outliers (above 4). Outputs: `experiments/eda_bottle.json` (deterministic, no timestamp) and, outside git, sample sheets and histograms in `data/processed/eda/`. It imports `make_splits` and `verify_dataset` by name (a single definition of the guard and of the split validation).
- **Results (real run, exit 0, about 6 s; second run byte-identical, SHA-256 of the JSON `5a6253130f87dcb15c33bab5d6e8eb9b8303b5ff1f381e3fbc026e700d5d4eec`):**
  - Files opened: 209 training images, **0** in any other folder; all 209 match the verified manifest; all 900×900, RGB.
  - fit vs calib means: luma mean 136.31 vs 136.42, luma spread 92.90 vs 92.88, mean R/G/B 136.51/135.55/139.71 vs 136.61/135.67/139.79, sharpness 1.757 vs 1.758.
  - KS p-values: luma mean 0.846, luma spread 0.751, R 0.849, G 0.892, B 0.978, sharpness 0.487 (file size 0.379, shown for information only and not in the six-test family). Bonferroni level 0.05 / 6 = 0.0083: **no metric below it**. Outliers: 0 in every metric. Reading: no strong evidence that fit and calib differ on these metrics.
  - The sample sheets (checked by eye) show bottle openings seen from above on a white background; the histograms of fit and calib overlap closely.
- **How to read it:** a rough warning check, not proof that the groups are equivalent. With 42 calib images the KS test has little power; the p-values are not corrected beyond the stated Bonferroni level; the metrics are simple global numbers, not what the model sees; the sheets are thumbnails of the originals, **not** the model's preprocessing (which does not exist yet). Nothing was tuned; no finding changes any setting.
- **Tests:** `make test` **29 passed** = 5 (`core/tests`) + 8 (`test_verify_dataset`) + 8 (`test_make_splits`) + 7 (`test_eda_train`: runs and writes outputs, deterministic, a bright image is flagged, a split containing a test file is refused, the guard blocks paths outside `train/good`, files differing from the verified manifest are refused, and a grep check that the script source never names the test or mask folders) + 1 (declared imports). `make lint` passes (after fixing import order and two over-long lines). The EDA test data keeps empty test files, so any attempt to open one would fail the run.
- **CI status (read-only, GitHub's public API):** run `42731046817` on the owner's commit `8d47a2e` ("Phase 1: add dataset verification script and bottle split lists"), `completed / success`, every step `success` (install 16 s, tests 20 s, job 66 s). That covers Phase 1 steps 4 and 5 on Linux; the raw logs were not readable, so the test count there is unverified. It does **not** cover the EDA script, the `scripts` group or the new tests. Supersedes the "not run on GitHub" statement in entries 038 and 041 for steps 4 and 5.
- **Download source:** still recorded as Kaggle (owner-stated), specific page unconfirmed, not compared with the official archive (entry 042). The EDA does not depend on it.
- **Corrections made along the way (reported as asked):**
  - Root `pyproject.toml` comment "ml/ in Phase 1 or 2" changed to "ml/ when Phase 2 begins". *Why: stale since the O1 decision (entry 040).*
  - `project-notes/each-step-notes.md`: the step 5 line "Nothing has been run on GitHub yet" updated. *Why: CI has since passed on the commit that contains those scripts.*
  - `docs/dev-environment.md`: added the declared-dependencies paragraph and the second CI run. *Why: the file lists the environment; it should show how dependencies are declared.*
  - `PLAN.md`: step 6 status, the §11 `scripts/` line (now mentions `scripts/tests/`), §15 item 21. `CLAUDE.md`: Tooling status (three test folders, the new "declare what you import" rule). `README.md`: the EDA script in "What exists so far".
- **Not verified:** the EDA script, the new dependency group and the new tests under Linux CI; Matplotlib's non-interactive output on a machine without a display (the script selects the file-only backend, but this was only run on the Mac); that fit and calib are exchangeable beyond these six simple metrics.
- **Files created:** `scripts/data/eda_train.py`, `scripts/data/tests/test_eda_train.py`, `scripts/tests/test_declared_imports.py`, `experiments/eda_bottle.json`, and (not committed) `data/processed/eda/*.png`. **Modified:** `pyproject.toml`, `uv.lock`, `decisions.md`, `CLAUDE.md`, `README.md`, `docs/dev-environment.md`, `project-notes/PLAN.md`, `project-notes/each-step-notes.md`. No commit, push or visibility change was made by Claude.

---

## Open decisions

| ID | Question | Needed by |
|---|---|---|
| ~~O1~~ | Resolved in entry 040 (option C): tooling in `scripts/data/`, split lists in `splits/`, exploration as a guarded script; `ml/` is created when Phase 2 begins. | — |
| O2 | Is a paid Render plan acceptable if the preferred model does not fit the free plan? | Checkpoint B |
| O3 | Artifact store: private Hugging Face repo, private GitHub Release, or Git LFS? | Phase 4 |
| ~~O4~~ | Resolved in entry 029: `project-notes/` stays ignored. | — |

### 044 — Phase 1 step 2: `docs/data.md` and `DATA_LICENSE.md`
- **Date:** 2026-11-10 · **Type:** Decision + Action + Finding · **Where:** Phase 1 step 2 · **Status:** Done; the "IJCV 2021 citation omitted" part is **superseded by entry 046** (the fresh reading of the official page is still not done)
- **Owner instruction:** write the two documents; take the CVPR 2019 citation from the dataset's `readme.txt` and the license from MVTec's official page; keep the download source as Kaggle with the page and its terms unconfirmed; do not invent the IJCV 2021 citation; record that the project is public and noncommercial and that the dataset is not committed; no commit, push or visibility change.
- **Sources actually read:** `data/raw/mvtec-ad/readme.txt` and `license.txt` (verified by reading; the `bottle` copies of both are byte-identical to the top-level ones, checked with `diff`). The license text is the full CC BY-NC-SA 4.0 legal code; the readme gives "Copyright 2019 MVTec Software GmbH", the license name and link, and the CVPR 2019 citation (authors, title, venue, year; no pages, no DOI).
- **Sources that failed (so nothing was copied from them):** the official MVTec page (connection closed twice), the Springer IJCV article page (HTTP 303 to a login page; the redirect was not followed), the CVF Open Access page for the 2019 paper (HTTP 403). The fetches were done by helper agents and they reported plainly that they could not read the pages.
- **Decisions and why:**
  - The license statement in both documents is sourced to the dataset's own files, and says so, because the official page could not be re-read. Earlier readings of the page (2026-11-09, `PLAN.md` §5.2) are mentioned as such, not repeated.
  - **IJCV 2021 citation omitted, marked [Unverified], listed as an open item.** *Why:* the owner said not to invent it and no authoritative source was reachable. No title, volume, pages or DOI were written from memory.
  - The CVPR citation is quoted exactly as in the readme, with no added pages or DOI.
  - The Kaggle source is stated as owner-stated, page and terms unconfirmed, copy not compared with the official archive (A17).
  - Both files say the repository is public and noncommercial, the dataset is not committed (`/data/` is ignored), derived material keeps the attribution and ShareAlike terms, and that no `LICENSE` file exists yet (MIT planned).
  - The documents refer to "the project notes", not to the private `project-notes/` folder by path, because the repository is public and that folder is git-ignored.
- **Files created:** `docs/data.md`, `DATA_LICENSE.md`. **Modified:** `project-notes/PLAN.md` (Phase 1 step 2 status line and §15 item 22; no other change), `README.md` (list of what exists and the project-documents list), `decisions.md`, `project-notes/each-step-notes.md`. No code changed.
- **Not verified:** the current wording of the official page; the IJCV 2021 citation; the Kaggle page and its terms; that the Kaggle copy equals the official archive. This is documentation, not legal advice.

### 045 — `PLAN.md` is a working plan (owner instruction)
- **Date:** 2026-11-10 · **Type:** Decision · **Where:** `CLAUDE.md` working style · **Status:** Final
- **Owner instruction:** treat `PLAN.md` as a working plan. If implementation shows a planned step no longer fits, explain the mismatch and recommend a change only when it would genuinely improve the project; keep the plan and related records updated once a change is agreed; do not follow outdated plan text blindly.
- **Done:** one bullet added to "Working style" in `CLAUDE.md`. It keeps the existing hard rule (no `PLAN.md` edit without approval): Claude proposes, the owner agrees, then the plan is updated. `PLAN.md` itself was not changed.
- **Files modified:** `CLAUDE.md`, `decisions.md`. No code, commit, push or visibility change.

### 046 — IJCV 2021 citation added (owner-supplied)
- **Date:** 2026-11-10 · **Type:** Decision + Correction · **Where:** Phase 1 step 2 documents · **Status:** The citation text stands; the description of its source is **corrected by entry 047**. Supersedes the omission of the IJCV citation in entry 044
- **Source and how far it is checked:** the project owner says they read the citation on Springer's publisher page and supplied it: Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." *International Journal of Computer Vision*, 129, 1038–1059. DOI 10.1007/s11263-020-01400-4. Claude could not open that page (Springer redirected to a login on 2026-11-10), so the text was copied as given and **not independently checked by Claude**. The documents say this. The DOI is the one Claude had already tried to fetch.
- **Changes:** `docs/data.md` (the IJCV `[Unverified]` section replaced by the citation and its source note; the open item reworded), `DATA_LICENSE.md` (citation added), `README.md` (sentence updated), `project-notes/PLAN.md` (Phase 1 step 2 status and §15 item 22; "one gap" removed), `decisions.md` (entry 044 status), `project-notes/each-step-notes.md`.
- **Left as it was, on purpose:** Kaggle page and its terms unconfirmed; the file-by-file comparison with the official archive not done (A17); the official MVTec page's license wording not re-read.
- **Checks:** see the report for this step. No commit, push or visibility change.

### 047 — Correction: who found the IJCV citation and the license page
- **Date:** 2026-11-10 · **Type:** Correction · **Where:** `docs/data.md`, `DATA_LICENSE.md`, `PLAN.md`, `each-step-notes.md` · **Status:** Final; corrects the source description in entry 046
- **What was wrong:** entry 046 and the documents said the owner read the citation on Springer's page. The owner clarified they did not; an AI assistant (ChatGPT) looked up Springer's publisher page and the official MVTec page, the owner relayed the result, and Claude copied it. *Why corrected:* the source of a citation should be stated as it happened.
- **Now stated:** the IJCV citation text (unchanged) and the statement that MVTec's page gives the license terms are reported by that assistant and relayed by the owner; Claude could not open either page (Springer login redirect, MVTec connection failure) and has not checked them first-hand. The official page is now cited in `docs/data.md` and `DATA_LICENSE.md`; the license wording in the repository still comes from the dataset's own `readme.txt` and `license.txt`.
- **Unchanged on purpose:** Kaggle page and terms unconfirmed; no file-by-file comparison with the official archive (A17).
- **Files modified:** `docs/data.md`, `DATA_LICENSE.md`, `project-notes/PLAN.md` (status line of Phase 1 step 2 and §15 item 22 only), `project-notes/each-step-notes.md`, `decisions.md`. No commit, push or visibility change.

### 048 — Phase 2 step 1: shared preprocessing; Phase 2 dependency wording
- **Date:** 2026-11-10 · **Type:** Decision + Action + Correction · **Where:** Phase 2 step 1; `core/`; `PLAN.md` Phase 2 header · **Status:** Done. The module has not run in CI or on real dataset images.
- **Owner instruction:** implement the shared preprocessing module with synthetic-image tests; read the plan's preprocessing rules first; declare direct dependencies in `core/pyproject.toml` and update the lock; correct the Phase 2 dependency wording (steps 1–4 need F2, real-data steps need F1); no commit, push or visibility change.
- **Plan reference note:** the instruction named §9; the preprocessing rules are in **§5.5** (also §5.4 item 3, and the tests in §9.1). Those were read and followed.
- **What was built (`core/src/defect_core/preprocessing.py`):**
  - `PreprocessConfig`: frozen record with `version="1"`, `size=256`, ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), `interpolation="bilinear"`, `antialias=True`, and `to_dict()` for the manifest. *Why the two extra fields:* §5.5 says "bilinear, antialias"; storing them makes the manifest state the whole recipe.
  - `load_image(path | bytes)`: decode, `ImageOps.exif_transpose`, RGB; transparent pixels composited on white; grayscale copied to 3 channels. Corrupt or non-image data raises `ImageLoadError` (a `ValueError`); a missing file raises `FileNotFoundError`. **Unsupported modes (for example 16-bit grayscale) are refused**, because Pillow would silently clip them. This refusal is my addition, not in the plan; a 16-bit input is unlikely for MVTec or web uploads and a clear error is safer than wrong pixels. Upload limits (size cap, magic bytes, min/max side) are API concerns and were not built.
  - `to_tensor(image, config)`: float32 in [0, 1], bilinear antialiased resize to 256×256 with **no crop**, ImageNet normalise; output shape (3, 256, 256).
- **Check against anomalib (verified by reading the installed 2.7.0 source):** its PatchCore pre-processor uses `Resize(image_size, antialias=True)` and `Normalize` with the same mean and std. A test confirms our output equals that pipeline on a float tensor (tolerance 1e-6). The same anomalib file also imports a `CenterCrop`; what it does in the default pre-processor was **not** examined, because the plan says no crop and our test compares only the resize and normalise steps.
- **Tests (18 new, `core/tests/test_preprocessing.py`, synthetic images only):** config values and frozen/serialisation; path, string and bytes inputs agree; EXIF rotation applied; grayscale to 3 channels; transparent pixels become white and half-transparent pixels blend; corrupt, truncated and unsupported-mode inputs raise clear errors; missing file; shape and dtype for non-square input; no crop (both halves kept); normalisation equals the formula; determinism; antialiasing (checkerboard averages to grey); equality with anomalib-style resize+normalise; configurable size and refusal of non-RGB input or an unsupported interpolation.
- **Dependencies (why):** `core/src` now imports `torch`, `torchvision` and `pillow` directly, so they are declared in `core/pyproject.toml` (rule in entry 043 and `CLAUDE.md`). `uv lock --offline`: 102 packages as before; only the `defect-core` entry gained the three requirements; **no package version changed**. `UV_LOCKED=1 uv sync --all-packages --offline --dry-run` then reports no changes after `uv sync --all-packages --offline` re-installed `defect-core` (81 packages checked). `numpy` is not declared in core because core's own code does not import it (the tests do, and the `scripts` group declares it). `pyproject.toml`-level files changed: `core/pyproject.toml` and `uv.lock` — **they must be committed together** (CI uses `UV_LOCKED=1`).
- **Plan correction (owner-approved wording):** the Phase 2 "Dependencies" line now says steps 1–4 need F2 only and steps 5–8 need F1 (F3 for the container comparison in step 6); `ml/` is created when step 5 begins. Step 1 got a status line; `PLAN.md` §15 item 23 added. No other plan text changed.
- **Checks:** `make lint` PASS; `make test` **47 passed** (29 before + 18 new); the declared-imports test passes with the new declarations.
- **Not verified:** behaviour on real MVTec images (not opened); Linux/CI (not pushed); equality with anomalib's *dataset loader* path (only its transform pipeline was compared); decompression-bomb handling beyond Pillow's defaults; the Kaggle page and the file-by-file comparison with the official archive remain unconfirmed (entry 042).
- **Files created:** `core/src/defect_core/preprocessing.py`, `core/tests/test_preprocessing.py`. **Modified:** `core/pyproject.toml`, `uv.lock`, `project-notes/PLAN.md`, `README.md`, `decisions.md`, `project-notes/each-step-notes.md`. No commit, push or visibility change.

### 049 — Anomalib pre-processing check, and Phase 2 step 2 (`model.py`)
- **Date:** 2026-11-10 · **Type:** Finding + Decision + Action · **Where:** Phase 2 step 2; `core/` · **Status:** Done. Not run on real data, real pretrained weights or CI.
- **Owner instruction:** before `model.py`, inspect the installed anomalib training and inference pipelines (especially `CenterCrop` or any other transform) and compare with the no-crop rule in `PLAN.md` §5.5 and `defect_core.preprocessing`. If they conflict, explain and recommend the smallest change; do not change preprocessing silently.
- **Finding: no conflict (read from the installed anomalib 2.7.0 source and confirmed by running it):**
  - `PatchcoreModel` (the torch module we use) applies **no** image transform. Its `forward` only casts the input to the memory bank's dtype and extracts features.
  - `Patchcore.configure_pre_processor(image_size=None, center_crop_size=None)` builds `Resize(size, antialias=True)` then `Normalize(ImageNet mean/std)`. A `CenterCrop` is added **only** if `center_crop_size` is given, and its default is `None`. Instantiating `Patchcore(...)` with defaults gave exactly `[Resize(256×256, bilinear, antialias=True), Normalize(...)]`.
  - Anomalib's default image loader (`read_image`: Pillow `convert("RGB")`, then float in [0, 1]) differs from ours by at most 6e-8 (float rounding), and the full transform output of the two paths differed by at most 0.0 on a random 173×311 image.
  - The `MVTecAD` data module's augmentations default to `None`. The PreProcessor is applied by Lightning hooks (`on_*_batch_start`); our deployed path does not use Lightning, so only `defect_core.preprocessing` applies.
  - The other `CenterCrop` imports found in anomalib belong to other models (`super_add`, `ai_vad`), not PatchCore.
  - **Result:** the plan's statement "mirrors anomalib's default pre-processor, no centre-crop" holds. Preprocessing was **not** changed. This closes the "not examined" note in entry 048. A regression test now compares our output with `Patchcore.configure_pre_processor().transform`, asserts its steps are exactly `[Resize, Normalize]`, and so fails if a future anomalib version adds a crop to the default.
  - Also checked: a train-mode pass does **not** change the backbone's batch-norm statistics (anomalib's feature extractor switches itself to eval mode; confirmed by running it and by a test). So fitting leaves the backbone weights untouched, as F2 assumed.
- **What was built (`core/src/defect_core/model.py`):**
  - `ModelConfig` (frozen): `backbone="wide_resnet50_2"`, `layers=("layer2","layer3")`, `num_neighbors=9`, `coreset_ratio=0.1`; `to_dict()` for the manifest.
  - `build_model(config, *, pre_trained)`: returns anomalib's `PatchcoreModel` in eval mode with an empty bank. `pre_trained` is **required** (no default): `True` downloads ImageNet weights (fitting), `False` does not (loading an artifact, tests).
  - `fit_memory_bank(model, batches, coreset_ratio, seed)`: seeds **both torch and numpy**, clears leftover embeddings, runs the train-mode pass under `no_grad`, calls `subsample_embedding`, and always returns the model to eval mode. Refuses an empty input, a `coreset_ratio` outside (0, 1], and batches that are not floating-point tensors of shape (B, 3, H, W). A failure leaves no stored embeddings. Returns the number of patch features collected.
  - `infer(model, batch)`: returns raw scores (B,) and anomaly maps (B, 1, H, W); refuses train mode; an empty bank raises anomalib's `ValueError`. No anomalib post-processor, no threshold.
  - **Choices of mine, not in the plan:** the required `pre_trained` argument, the train-mode refusal in `infer`, and the input-shape checks. *Why:* each prevents a silent mistake (an accidental download, scores from a model that is still collecting features, a wrongly shaped tensor).
- **Tests (11 new in `core/tests/test_model.py`, 1 new in `test_preprocessing.py`):** config defaults and frozen; `pre_trained` must be given; build makes no network call (network blocked) and starts empty in eval mode; empty bank rejected; fit gives a subset bank of the expected size and leaves eval mode with no leftovers; output shapes; an image that is in the bank scores lower than a very different image; same seed → identical bank and scores; backbone weights and batch-norm statistics unchanged by fitting; a second fit replaces the first; bad inputs refused and a failed fit leaves nothing behind; `infer` refuses train mode. These use a small random `resnet18` and random images: they test wiring, not detection quality.
- **Checks:** `make lint` PASS; `make test` **60 passed** (47 + 13).
- **Warnings:** each fit prints anomalib's `DeprecationWarning` about the `embeddings` argument of `subsample_embedding` even though we do not pass it (already listed as harmless in `docs/dev-environment.md`).
- **Not verified:** behaviour with real pretrained weights or real images; memory and time for `wide_resnet50_2` at full size (F1 and Phase 2 step 5); Linux and CI; anomalib versions other than 2.7.0. The Kaggle page and the file-by-file comparison with the official archive remain unconfirmed.
- **Files created:** `core/src/defect_core/model.py`, `core/tests/test_model.py`. **Modified:** `core/tests/test_preprocessing.py`, `project-notes/PLAN.md` (step 2 status line, §15 item 24), `README.md`, `decisions.md`, `project-notes/each-step-notes.md`. `core/pyproject.toml` and `uv.lock`: `numpy` added to core's declared dependencies, because `model.py` imports it directly (it was already installed and declared in the root `scripts` group, which is why the declared-imports test did not catch it; the rule in `CLAUDE.md` says core's own imports belong in `core/pyproject.toml`). `uv lock --offline`: 102 packages, no version changed. No commit or push; `core/pyproject.toml` and `uv.lock` must be committed together.
