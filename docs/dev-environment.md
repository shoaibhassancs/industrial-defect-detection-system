# Development environment

Where the numbers in this project come from. Speed and memory results mean little without the machine that produced them, so this file records it. Values below were read from the machine and from `uv.lock` on **2026-11-09** unless marked otherwise.

## Machine

| Item | Value |
|---|---|
| Model | MacBook Air (2020), Apple M1 |
| CPU cores | 8 (4 performance, 4 efficiency) |
| Memory | 8 GB (8,589,934,592 bytes), shared by the OS, apps, and any model we run |
| OS | macOS 15.2 (build 24C101), arm64 |
| Disk (data volume) | 228 GiB total. Free space varied a lot: 19 GiB at the start of 2026-11-09, 11 GiB later that day, as low as 5.0 GiB on 2026-11-10 while the Mac was swapping heavily, 17 GiB after closing apps and restarting, about 16 GiB afterwards. Check it before downloads and measurements. |

Consequences for the project:
- The Mac has no fan, so long runs may slow down from heat. Record wall time, and plug the Mac in.
- Memory is tight. Run heavy experiments one at a time and close other apps.
- Free disk space is limited. The `.venv` alone is about 1.2 GB. The MVTec AD download and Docker images will need more; check free space before downloading. The dataset size is **[Unverified]** until Phase 1.

## Tools

| Tool | Version | Used for |
|---|---|---|
| uv | 0.13.6 | Python versions, dependencies, running commands |
| Python | 3.12.16 (pinned by `.python-version`, `requires-python >=3.12,<3.13`) | Everything |
| make | GNU Make 3.81 | Shortcuts in the `Makefile` |
| Docker | 29.9.2 | Installed. Not used yet (Phase 0F container test, Phase 5 backend image). |
| Node | v22.16.0 | Installed; used by `frontend/` (Phase 6). npm 10.9.2. |

## Key libraries (from `uv.lock`)

| Library | Version | Notes |
|---|---|---|
| anomalib | 2.7.0 | Provides `PatchcoreModel`. Torch is **not** a default dependency; it comes from the `cpu` extra, so `core/` depends on `anomalib[cpu]`. |
| torch | 2.15.0 | CPU/MPS build for macOS arm64 |
| torchvision | 0.29.2 | |
| timm | 1.0.31 | Backbone networks |
| lightning | 2.6.6 | Pulled in by anomalib |
| torchmetrics | 1.9.0 | |
| kornia | 0.9.0 | |
| numpy | 2.5.4 | |
| ruff / pytest | 0.17.4 / 9.2.0 | Development only |

The lockfile lists 102 packages (our own `defect-core`, 4 development tools, and the rest third-party). 81 are installed on this Mac; the others are only needed on other platforms. The macOS download was about 280 MB by a sum of lockfile wheel sizes (an estimate, not a measured download).

**Declared dependencies (since 2026-11-10):** the root `pyproject.toml` has two dependency groups, both installed by default (`[tool.uv] default-groups`): `dev` (ruff, pytest) and `scripts` (numpy, pillow, scipy, matplotlib, safetensors, torch), the packages imported directly by the repository's scripts and tests. `core/pyproject.toml` declares `anomalib[cpu]` and what `defect_core` imports; `ml/pyproject.toml` declares `defect-core` and `torch`. `scripts/tests/test_declared_imports.py` fails if Python code imports a third-party package that none of these declare. Adding the `scripts` group changed no package version in `uv.lock`.

**Linux (verified once, 2026-11-09):** a GitHub Actions run on `ubuntu-latest` (x86_64) installed the locked dependencies (61 s for the install step) and passed lint and tests (commit `90624b0`; the job took 138 s in total). This comes from GitHub's step results; the raw logs were not readable without a login, so the download size, free disk space and test count on the runner are **[Unverified]**. The lock still resolves Linux torch from PyPI with the large NVIDIA/CUDA packages (about 3.25 GB of wheels by our estimate from the lock). A CPU-only torch index for Linux has not been configured; the Docker image in Phase 5 will need one. A second run (commit `8d47a2e`, which added the Phase 1 dataset scripts and tests) also passed: install step 16 s, test step 20 s, job 66 s (the install was faster, probably because of the cache; not checked in the logs). Other Linux setups (for example the Render runtime) have not been tested.

## Findings that affect how we work

- **Swap can fill the disk and slow everything (observed 2026-11-10):** with many apps open, macOS had 7.9 GB of 9.2 GB swap in use, free disk fell to about 5 GiB, and `make test` took 74 s instead of about 8 s. After closing apps and restarting, swap fell to about 0.4 GB and tests took 7.7 s again. The cause of the slowdown is consistent with swapping but was not proven. Before timed runs, check `sysctl vm.swapusage` and `df -h`.
- **Dataset:** the local copy is at `data/raw/mvtec-ad/` (about 4.9 GB, gitignored). The owner says it was downloaded from Kaggle. See `decisions.md` entry 042.

- **Apple GPU backend (MPS):** reported available by torch, but not used. Fitting, calibration, and evaluation run on CPU so the reported numbers match the deployment target.
- **Seeding:** to get repeatable results, seed **both** `torch` and `numpy`. With only torch seeded, two identical runs gave different scores (max difference 0.0108 in a small test); with numpy also seeded they were identical. The anomalib coreset sampler uses a random projection with no fixed seed. The exact mechanism is **[Unverified]**; the effect was tested.
- **`wide_resnet50_2` size:** as built by `PatchcoreModel` (layers `layer2` and `layer3`, no pretrained weights), it has 24,862,528 parameters, about 99 MB in float32. This is lower than the full network because only the layers up to the ones PatchCore uses are kept **[Unverified: inferred from the parameter count]**. Real memory use is measured in Phase 0F.
- **Fit and score cost on this Mac (F1, synthetic, 2026-11-10):** random weights and 167 random 256×256 images, CPU, 4 threads, other apps open, Mac on AC power, swap about 2.1–2.8 GB. `resnet18`: fit 329 s (features 3.9 s, coreset 325 s), memory bank 17,100 × 384 (25 MB), peak resident memory 1,348 MB, one image in 84 ms (median) / 92 ms (95th percentile), artifact 37.4 MB. `wide_resnet50_2`: fit 354 s (features 16.7 s, coreset 337 s), bank 17,100 × 1,536 (100 MB), peak resident memory 1,927 MB, one image in 212 ms / 243 ms, artifact 204.7 MB. Both artifacts reloaded with bit-identical outputs. Limits: timings include swapping and other apps; peak resident memory ignores memory the OS compressed or swapped; random data; not Render. Details: `experiments/feasibility/f1_*.json`, `decisions.md` entry 052.
- **Real `bottle` fits (2026-11-10, torchvision ImageNet weights, 167 `fit` images, CPU, 4 threads, Mac on AC power with other apps open):** `resnet18.tv_in1k`: fit 320 s (features 6.7 s, coreset 313.6 s), bank 17,100 × 384 (25 MB), peak resident memory 1,062 MB, artifact 37.4 MB. `wide_resnet50_2.tv_in1k`: fit 355 s (features 20.0 s, coreset 334.6 s), bank 17,100 × 1,536 (100 MB), peak resident memory 1,907 MB, artifact 204.7 MB; swap rose from 2.3 to about 3.6 GB (4.0 GB at its highest) and free disk dipped to about 11.3 GiB during that run. Compared with the synthetic F1 runs, the bank shapes are identical and the coreset times are within about 4% (F1: 325 s and 337 s), which fits the assumption that these costs depend on image count, resolution and feature width, not on image content or weight values. That is two runs on a busy Mac, not a proof. Details: `experiments/fit_*.json`, `decisions.md` entry 057.
- **Serving-style benchmark of the fitted `bottle` artifacts (2026-11-10, NATIVE macOS PROXY: arm64 macOS on this Mac, not a Linux container; 1 and 2 threads are rough settings, not CPU quotas; Mac on AC power, swap about 3.0 GB, other apps open):** fresh process, import, load, 3 warm-up and 50 timed single-image inferences on `fit` images (read, decode, preprocess, score). Importing torch costs about 190 MB of resident memory, and importing `defect_core` (which pulls in anomalib) brings it to about 505-520 MB, before any model is loaded. `resnet18`: load 0.35 s, steady about 711-722 MB, peak 988-1,003 MB, median 131 ms (1 thread) / 109 ms (2 threads) per image, 95th percentile 144 / 118 ms. `wide_resnet50_2`: load 2.0 s, steady about 650-770 MB (resident readings fluctuate; the OS compresses pages), peak 1,042-1,102 MB, median 272 ms / 237 ms, 95th percentile 297 / 256 ms. Scoring is about 85-92% of the time; preprocessing about 17-19 ms. Details: `experiments/benchmark_*.json`, `decisions.md` entry 058. The container measurements (F3) are still open.
- **Order of calls:** a train-mode pass collects patch features, `subsample_embedding` builds the memory bank, and only then does an eval-mode pass work. Scoring with an empty bank raises `ValueError`.
- **Harmless warnings:** timm and anomalib print deprecation warnings (`FutureWarning`, and a `DeprecationWarning` about an unused `embeddings` argument of `subsample_embedding`).

## Web stack (added 2026-11-10 for Phase 5, installed locally from PyPI; versions are pinned by `uv.lock`)

`backend/` declares: FastAPI 0.162.2, Starlette 1.10.0, Pydantic 2.14.3, pydantic-settings 2.16.2,
Uvicorn 0.54.0, python-multipart 0.0.33, structlog 26.1.0, slowapi 0.1.10 and `limits` 5.8.0 (used by
slowapi and the settings check). httpx 0.28.1 is a development dependency (FastAPI's test client).
Adding them added 306 lines to `uv.lock` and changed no existing pin. **Native macOS measurement
(proxy, not a container):** the real server with the `wide_resnet50_2` package used about 752 MB
resident after load and warm-up and about 898 MB after three inspections of a 400 x 300 image (about
0.27-0.31 s each over HTTP, one thread).

## Frontend stack (added 2026-11-11 for Phase 6; installed locally with npm, versions pinned by `frontend/package-lock.json`)

`frontend/` is a standalone npm project (not part of the uv workspace). Runtime: React 19.3.1 and
react-dom 19.3.1. Build and style: Vite 8.3.7, `@vitejs/plugin-react` 6.3.0, Tailwind CSS 4.3.5 with
`@tailwindcss/vite`, TypeScript **6.0.3**. Checks: ESLint 10.14.0, `typescript-eslint` 8.72.2,
`eslint-plugin-react-hooks` 7.1.1, Prettier 3.9.10. Tests: Vitest 5.1.0 (with `@vitest/coverage-v8`),
jsdom **29.1.1**, React Testing Library 16.3.3 (+ `@testing-library/dom` 10, `user-event` 14.6.9,
`jest-dom` 7.0.1), React Router 7.18.5 (added 2026-11-11 for the page navigation), MSW 3.0.4 (mock API), `axe-core` 4.14.0 (automatic accessibility rules).
Types from the API: `openapi-typescript` 7.13.0. 319 packages installed by `npm ci` (225 top-level folders), about 213 MB in `node_modules/`.

Why these pins (checked against the npm registry on 2026-11-11):
- TypeScript is pinned to 6.0.3 because the latest `typescript-eslint` accepts TypeScript below 6.1
  (the newest TypeScript is 7.0.2).
- jsdom is pinned to 29.x because jsdom 30 needs Node 22.22.2 or newer and this Mac has 22.16.0.
- `react-router` is pinned to 7.18.5 because every 8.x release needs Node 22.22 or newer (checked on the registry); 7.x needs Node 20 or newer and accepts React 18 or newer.
- `@types/node` is on the 22 line to match the Node runtime.
- `openapi-typescript` 7.13.0 declares a peer of `typescript ^5.x`. A package.json `overrides` entry
  lets it use TypeScript 6.0.3. That is a declared-range mismatch we worked around, not a tested
  guarantee; the generated `schema.d.ts` compiles with `tsc` and the frontend tests and build pass.
- `eslint-plugin-jsx-a11y` was not added: its latest version declares support for ESLint up to 9, and
  this project uses ESLint 10. Accessibility is covered by labelled markup, `axe-core` in the tests
  and a manual keyboard check instead.
- Installing used the npm cache in a scratch folder, deleted afterwards (the cache was 230 MB per
  stage and would have pushed free disk close to the 7 GiB guard).

## Commands

```
make setup    # install everything into .venv (uv sync --all-packages)
make lint     # Ruff checks, no changes
make format   # Ruff auto-fix and format
make test     # run the test suite (about 5-6 minutes on the M1 with the model tests)
make frontend-install  # npm ci in frontend/
make frontend-check    # frontend format check, lint, type-check, tests, build
make evidence          # re-make the Home page's evidence snapshot from the saved metrics.json
make api-contract      # re-export frontend/openapi.json and regenerate the TypeScript types
```

## Related files

- [README.md](../README.md), [SCOPE.md](../SCOPE.md), [decisions.md](../decisions.md)
