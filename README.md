# DefectLens — Industrial Anomaly Inspection Demo

A portfolio project that demonstrates **visual anomaly detection** on industrial products. A user picks a product category, uploads an image, and the app reports whether it looks **normal** or **anomalous**, with the anomaly score, the threshold, and a heatmap of where the image differs from normal.

The model is **PatchCore** (PyTorch, via the anomalib library), fitted only on defect-free images from the **MVTec AD** dataset. It never needs examples of defects.

> **Status: early development.** Release 1 (one category, `bottle`) has not been built yet. There is no live demo, no trained model, and no results to report. See [SCOPE.md](SCOPE.md) for what is planned and [decisions.md](decisions.md) for the running log of decisions and progress.

> **Educational demo.** This is not a certified quality-control or inspection system. Results are only meaningful for supported categories photographed like the MVTec AD images, and the anomaly score is not a probability or a confidence.

## What exists so far

- A Python workspace (`core/`) with PatchCore compatibility tests.
- Ruff (lint and format) and pytest, run automatically on GitHub Actions (`.github/workflows/ci.yml`).
- Experiments on saving and reloading a PatchCore model (`scripts/feasibility/`), using synthetic data only.
- A script that checks a local copy of the MVTec AD dataset (`scripts/data/verify_dataset.py`). The dataset itself is not part of this repository.
- A fixed 80/20 split of the `bottle` training images into a "fit" group and a "calibration" group (`scripts/data/make_splits.py`, `splits/bottle.json`). It lists file names only.
- A train-only look at those images (`scripts/data/eda_train.py`): simple brightness, colour and sharpness numbers, comparing the fit and calibration groups (a rough check, not a proof). It reads training images only.
- Shared image preprocessing (`core/src/defect_core/preprocessing.py`) with tests on synthetic images. The PatchCore wrapper (`model.py`: build, fit, score) with tests on random data.
- Data notes and licensing: [docs/data.md](docs/data.md) and [DATA_LICENSE.md](DATA_LICENSE.md).

Nothing else yet. The roadmap is private for now.

## Set up and check

Requires [uv](https://docs.astral.sh/uv/) and `make`. The environment was set up on an Apple M1 Mac with 8 GB of memory; see [docs/dev-environment.md](docs/dev-environment.md).

```
make setup    # install dependencies (Python 3.12, PyTorch, anomalib) into .venv
make lint     # style and mistake checks
make test     # run the tests (about 10 seconds)
```

On Linux, a GitHub Actions run (Ubuntu, x86_64, 2026-11-09, commit `90624b0`) installed the locked dependencies and passed lint and tests. For Linux, `uv.lock` lists the large GPU-enabled PyTorch build; a CPU-only setup has not been configured yet.

## Project documents

- [SCOPE.md](SCOPE.md): what the project is and is not.
- [decisions.md](decisions.md): dated log of decisions and actions.
- [docs/dev-environment.md](docs/dev-environment.md): machine, tools, and library versions.
- [docs/data.md](docs/data.md): the dataset, its layout, citation, and what has been checked.
- [DATA_LICENSE.md](DATA_LICENSE.md): dataset license and attribution, and what is and is not redistributed.

## Dataset and attribution

The model will be fitted on the **MVTec Anomaly Detection (MVTec AD)** dataset, released by MVTec under the **CC BY-NC-SA 4.0** license, which prohibits commercial use. This project is a non-commercial educational demo. The local copy used for development is not included in this repository; the project owner says it was downloaded from Kaggle, and it has not been compared file by file with the official MVTec archive (only the per-category image counts match the official paper). Attribution, the citation, and notes on processed images are in [DATA_LICENSE.md](DATA_LICENSE.md) and [docs/data.md](docs/data.md); both the CVPR 2019 and IJCV 2021 citations are included. Official page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>.

## License

This repository is public and the project is noncommercial. The plan is to license the original project code under MIT, but no `LICENSE` file has been added yet. The dataset is not part of this repository and keeps its own license (see above).
