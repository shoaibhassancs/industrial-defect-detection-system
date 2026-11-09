# decisions.md — DefectLens decision and action log

A chronological record of every decision and significant action in this project, so that everything is traceable. Newest entries go at the bottom. Claude appends to this file at the end of each step (see CLAUDE.md).

**Entry format**
- **Type:** `Decision` (a choice the owner or Claude+owner made), `Action` (something that was done), or `Finding` (a verified fact that affects the project).
- **Where:** the phase and step it belongs to.
- **Status:** `Final`, `Candidate` (to be confirmed by a later measurement), `Open`, `Superseded`, or `To confirm`.
- Unverified claims are marked **[Unverified]**.

**Related files:** [project-notes/PLAN.md](project-notes/PLAN.md) (roadmap), [SCOPE.md](SCOPE.md) (scope), [CLAUDE.md](CLAUDE.md) (working rules). Large technical decisions will also get an ADR in `project-notes/decisions/` (ADR-0001 stack, ADR-0002 model and serving route, and so on).

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
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Project-wide · **Status:** Final
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
- **Date:** 2026-11-09 · **Type:** Decision · **Where:** Phase 0 onward · **Status:** Final, with one open item (O1)
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
- **Date:** 2026-11-10 · **Type:** Action + Finding · **Where:** Phase 0 step 0.5 · **Status:** Written; **not verified on GitHub** (needs the owner's push)
- **What:** Created `.github/workflows/ci.yml`. On pushes to `main`, pull requests, and manual runs it checks out the code, installs uv, runs `uv sync --all-packages`, then `make lint` and `make test` on `ubuntu-latest` (x86_64, like Render). Details: `UV_LOCKED=1` so a stale `uv.lock` fails the run; uv pinned to 0.13.6 (same as the development Mac); Python 3.12 comes from `.python-version`; uv cache kept with `prune-cache: false`; read-only token permissions; newer pushes cancel older runs; 45-minute timeout; two disk-space printouts and a torch version printout to help diagnose the download concern below.
- **Actions pinned to commit hashes:** `actions/checkout` v7.0.1 (`3d3c42e5…`) and `astral-sh/setup-uv` v10.3.3 (`68559e99…`). The latest versions and the commits behind their tags were read from the GitHub API on 2026-11-10, and the inputs used (`version`, `enable-cache`, `cache-dependency-glob`, `prune-cache`, `persist-credentials`) were checked in each action's `action.yml` at those commits.
- **Checks done:** the workflow YAML parses and has the expected structure (job, steps, triggers, pinned `uses`). `make lint` still passes. **Not done:** the workflow has not run anywhere. No linter for workflows (`actionlint`) is installed, so GitHub may still reject something. Nothing was pushed.
- **Dependency and download concern (from `uv.lock`, not measured):** the lock resolves torch for Linux from PyPI. For x86_64 Linux it lists 19 NVIDIA/CUDA-related packages (plus `triton`) whose largest wheels sum to about **2.45 GB**, and a torch wheel of about **539 MB**. So the first Linux install may download roughly **3 GB** and unpack to more. The Mac gets none of these. `uv sync` has no `--torch-backend` option in uv 0.13.6 (checked with `--help`), so this cannot be fixed inside the workflow alone. The runner's free disk space is unknown **[Unverified]**; the workflow prints it. If the run is too slow or runs out of space, the fix is a Linux-only CPU torch index in `pyproject.toml` plus `uv lock` (planned for the Docker build in Phase 5 anyway). That changes dependency configuration, so it needs the owner's approval first.
- **Still unverified:** that dependencies install on Linux at all (assumption A11), that the 5 tests pass there, and the real size and time of the install.
- **Files created:** `.github/workflows/ci.yml`. **Modified:** `decisions.md`, `project-notes/PLAN.md` (step 0.5 row, assumption A11, new §15 item 15), `CLAUDE.md` (Tooling status), `project-notes/each-step-notes.md`.
- **`.gitignore`:** no change needed.

### 036 — Local verification of the CI workflow (what could and could not be checked)
- **Date:** 2026-11-10 · **Type:** Action + Finding · **Where:** Phase 0 step 0.5 · **Status:** Local checks done; **the workflow has not run on GitHub**
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

---

## Open decisions

| ID | Question | Needed by |
|---|---|---|
| O1 | Where Phase 1 dataset tooling lives: create `ml/` at the start of Phase 1, or keep temporary scripts under `scripts/`? | Before Phase 1 |
| O2 | Is a paid Render plan acceptable if the preferred model does not fit the free plan? | Checkpoint B |
| O3 | Artifact store: private Hugging Face repo, private GitHub Release, or Git LFS? | Phase 4 |
| ~~O4~~ | Resolved in entry 029: `project-notes/` stays ignored. | — |
