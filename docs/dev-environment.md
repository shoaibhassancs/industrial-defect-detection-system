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
| Node | v22.16.0 | Installed. Not used until Phase 6 (frontend). |

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

**Linux (verified once, 2026-11-09):** a GitHub Actions run on `ubuntu-latest` (x86_64) installed the locked dependencies (61 s for the install step) and passed lint and tests (commit `90624b0`; the job took 138 s in total). This comes from GitHub's step results; the raw logs were not readable without a login, so the download size, free disk space and test count on the runner are **[Unverified]**. The lock still resolves Linux torch from PyPI with the large NVIDIA/CUDA packages (about 3.25 GB of wheels by our estimate from the lock). A CPU-only torch index for Linux has not been configured; the Docker image in Phase 5 will need one. Other Linux setups (for example the Render runtime) have not been tested.

## Findings that affect how we work

- **Swap can fill the disk and slow everything (observed 2026-11-10):** with many apps open, macOS had 7.9 GB of 9.2 GB swap in use, free disk fell to about 5 GiB, and `make test` took 74 s instead of about 8 s. After closing apps and restarting, swap fell to about 0.4 GB and tests took 7.7 s again. The cause of the slowdown is consistent with swapping but was not proven. Before timed runs, check `sysctl vm.swapusage` and `df -h`.
- **Dataset:** the local copy is at `data/raw/mvtec-ad/` (about 4.9 GB, gitignored). The owner says it was downloaded from Kaggle. See `decisions.md` entry 042.

- **Apple GPU backend (MPS):** reported available by torch, but not used. Fitting, calibration, and evaluation run on CPU so the reported numbers match the deployment target.
- **Seeding:** to get repeatable results, seed **both** `torch` and `numpy`. With only torch seeded, two identical runs gave different scores (max difference 0.0108 in a small test); with numpy also seeded they were identical. The anomalib coreset sampler uses a random projection with no fixed seed. The exact mechanism is **[Unverified]**; the effect was tested.
- **`wide_resnet50_2` size:** as built by `PatchcoreModel` (layers `layer2` and `layer3`, no pretrained weights), it has 24,862,528 parameters, about 99 MB in float32. This is lower than the full network because only the layers up to the ones PatchCore uses are kept **[Unverified: inferred from the parameter count]**. Real memory use is measured in Phase 0F.
- **Order of calls:** a train-mode pass collects patch features, `subsample_embedding` builds the memory bank, and only then does an eval-mode pass work. Scoring with an empty bank raises `ValueError`.
- **Harmless warnings:** timm and anomalib print deprecation warnings (`FutureWarning`, and a `DeprecationWarning` about an unused `embeddings` argument of `subsample_embedding`).

## Commands

```
make setup    # install everything into .venv (uv sync --all-packages)
make lint     # Ruff checks, no changes
make format   # Ruff auto-fix and format
make test     # run the test suite (about 10 seconds)
```

## Related files

- [README.md](../README.md), [SCOPE.md](../SCOPE.md), [decisions.md](../decisions.md)
