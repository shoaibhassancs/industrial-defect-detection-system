# DefectLens — Industrial Anomaly Inspection Demo

A portfolio project that demonstrates **visual anomaly detection** on industrial products. A user picks a product category, uploads an image, and the app reports whether it looks **normal** or **anomalous**, with the anomaly score, the threshold, and a heatmap of where the image differs from normal.

The model is **PatchCore** (PyTorch, via the anomalib library), fitted only on defect-free images from the **MVTec AD** dataset. It never needs examples of defects.

> **Status: early development.** Release 1 (one category, `bottle`) has not been built yet. There is no live demo, no trained model, and no results to report. See [SCOPE.md](SCOPE.md) for what is planned and [decisions.md](decisions.md) for the running log of decisions and progress.

> **Educational demo.** This is not a certified quality-control or inspection system. Results are only meaningful for supported categories photographed like the MVTec AD images, and the anomaly score is not a probability or a confidence.

## What exists so far

- A Python workspace (`core/`) with PatchCore compatibility tests.
- Ruff (lint and format) and pytest.

Nothing else yet. The roadmap is private for now.

## Set up and check

Requires [uv](https://docs.astral.sh/uv/) and `make`. The environment was set up on an Apple M1 Mac with 8 GB of memory; see [docs/dev-environment.md](docs/dev-environment.md).

```
make setup    # install dependencies (Python 3.12, PyTorch, anomalib) into .venv
make lint     # style and mistake checks
make test     # run the tests (about 10 seconds)
```

Installation on Linux has not been tested yet.

## Project documents

- [SCOPE.md](SCOPE.md): what the project is and is not.
- [decisions.md](decisions.md): dated log of decisions and actions.
- [docs/dev-environment.md](docs/dev-environment.md): machine, tools, and library versions.

## Dataset and attribution

The model will be fitted on the **MVTec Anomaly Detection (MVTec AD)** dataset, released by MVTec under the **CC BY-NC-SA 4.0** license, which prohibits commercial use. This project is a non-commercial educational demo. Full attribution, citations, and notes on processed images will be added in `docs/data.md` when the dataset is added (Phase 1). Official page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>.

## License

The license for original project code (MIT) will be added when the repository is made public.
