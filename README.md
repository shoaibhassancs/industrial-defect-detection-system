# DefectLens — Industrial Anomaly Inspection Demo

DefectLens is a local demo for checking bottle images for unusual patterns. Upload an image to see an anomaly score, a threshold-based result, and a heatmap highlighting areas that differ from defect-free examples.

It uses PatchCore with PyTorch and anomalib. The models learn from defect-free images, so they do not need examples of defects during training.

> **This is an educational portfolio project, not a certified inspection or quality-control system.** It was evaluated on the MVTec AD `bottle` category only. Anomaly scores are not probabilities or confidence estimates.

## Status

Release 1 runs locally and supports one category: `bottle`. It includes two fitted and calibrated models, a FastAPI service, and a React web app. There is no hosted demo; deployment is deferred for this version.

The model packages and dataset are not included in this repository. See [Run it locally](#run-it-locally) for what you need before starting the app.

## Screenshots

These screenshots show the local app workflow; they are not evidence of model accuracy. Evaluation results are in [docs/evaluation.md](docs/evaluation.md).

![Home page of the local app](screenshots/a.png)

![Demo page showing a calibration example](screenshots/c.png)

The image on the Demo page is a calibration example from the MVTec AD training data, adapted with the app’s overlay. It illustrates the interface, not detection performance. MVTec AD is licensed under CC BY-NC-SA 4.0. This project is not endorsed by MVTec; see [DATA_LICENSE.md](DATA_LICENSE.md).

![Results page showing saved test-set results](screenshots/e.png)

More screens are in [screenshots/](screenshots/). The local setup walkthrough is in [docs/local-walkthrough.md](docs/local-walkthrough.md).

## Evaluation results

The two models were evaluated once on the held-back MVTec AD `bottle` test set. Both achieved an image-level AUROC of **1.00** and flagged **63 of 63** defective test images.

That result does not mean the models are ready for real inspections. At the chosen thresholds:

| Model | Defective test images flagged | Normal test images flagged by mistake |
| --- | ---: | ---: |
| `wide_resnet50_2` | 63 / 63 | 3 / 20 (15%) |
| `resnet18` | 63 / 63 | 5 / 20 (25%) |

The false-alarm rate for both models was above the project’s 5% target. The normal-image sample was small, and these results apply only to the evaluated `bottle` images. See [docs/evaluation.md](docs/evaluation.md) for the full methodology, figures, and limitations.

## How to read a result

- **Anomaly score:** higher means the image looks more unusual compared with defect-free examples. It is not a probability.
- **Threshold:** the stored cutoff used for the demo’s “Normal” or “Anomalous” result.
- **Rank:** compares the score with calibration images. It is a rank, not confidence.
- **Heatmap and overlay:** highlight areas that differ from normal examples. They do not measure defect size or prove that a visible area is defective.

The model is intended for images similar to the supported MVTec AD bottle images: centred bottles photographed from above in controlled lighting. It cannot reliably recognize an image that is outside this setting; an unrelated image may simply receive a high anomaly score.

## Run it locally

You need Python 3.12, [uv](https://docs.astral.sh/uv/), `make`, and Node 22.12 or newer for the web app. The project was developed on an Apple M1 Mac with 8 GB RAM and has been tested locally on that machine. See [docs/dev-environment.md](docs/dev-environment.md).

### 1. Set up and run checks

```bash
make setup
make lint
make test
```

### 2. Prepare the data and model package

You need a local copy of the MVTec AD dataset and a packaged model. Neither is stored in this repository. The project’s local data copy came from Kaggle and has not been compared file by file with the official archive; image counts were compared with the published paper.

Check the dataset terms before obtaining or reusing it. Follow [docs/data.md](docs/data.md) for the data layout, then follow [docs/evaluation.md](docs/evaluation.md) §8 to fit, calibrate, evaluate, and package the models. The evaluation steps that open the test set require explicit confirmation.

### 3. Run inference on one image

After creating a model package, run:

```bash
uv run python -m defect_core.infer \
  --artifact artifacts/bottle_wide_resnet50_2_tv_in1k_v1.0.0_release \
  path/to/image.png
```

To save a heatmap and overlay, add `--out results/`. To print machine-readable output, add `--json`. The command verifies the package before scoring and does not overwrite existing output files.

### 4. Run the API and web app

Start the API in one terminal:

```bash
export MODEL_DIR=artifacts/bottle_wide_resnet50_2_tv_in1k_v1.0.0_release
export ALLOWED_ORIGINS=http://localhost:5173
uv run uvicorn --factory app.main:create_app --app-dir backend --port 8000 --no-access-log
```

Then start the web app in another terminal:

```bash
make frontend-install
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Choose the `bottle` category, select a JPEG or PNG image, and press **Inspect image**.

The API accepts JPEG and PNG uploads up to 10 MB, with each image side between 64 and 8,000 pixels. It does not store inspection results. Settings are listed in [.env.example](.env.example); the frontend API address is configurable with `VITE_API_BASE_URL`.

For a walkthrough of the pages and results, see [docs/local-walkthrough.md](docs/local-walkthrough.md). Frontend checks can be run with `make frontend-check`.

## Model files

The packaged models are not in Git. They are built by the project’s fit, freeze, calibration, evaluation, and packaging steps described in [docs/evaluation.md](docs/evaluation.md) §8.

A private Hugging Face repository was considered as a future store, but none has been created and no model files have been uploaded. Whether uploading the model weights is permitted has not been verified. See [docs/adr/ADR-0003-artifact-store.md](docs/adr/ADR-0003-artifact-store.md).

## Known limits

- Only the `bottle` category was evaluated.
- The test set was evaluated once; its results were not used to tune the models.
- The test set contained 20 normal images, so the false-alarm estimates are uncertain.
- The anomaly score and rank are not probabilities or confidence.
- The model may label an out-of-scope image as anomalous.
- The local browser review was not recorded item by item. Phone layouts, other browsers, screen readers, and measured colour contrast are not verified.
- The API and web app have not been deployed. Render was not tested, and there is no Docker image.

## Data and attribution

The models use the MVTec Anomaly Detection (MVTec AD) dataset. MVTec provides it under the **CC BY-NC-SA 4.0** license, which does not allow commercial use. This project is a non-commercial educational demo, and the local dataset copy is not included here.

The copy used for development was downloaded from Kaggle and was not checked file by file against the official archive. Its per-category image counts match the published paper. Read [DATA_LICENSE.md](DATA_LICENSE.md) and [docs/data.md](docs/data.md) for attribution, citations, and details. The [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) has the dataset information.

## License

The project code currently has no license granting reuse rights. Although the repository is public, that does not grant permission to copy, modify, or redistribute the code. The MVTec AD data, screenshots, and model files have separate terms; see [DATA_LICENSE.md](DATA_LICENSE.md).

## Project documents

- [Scope](SCOPE.md): what this release includes and defers.
- [Evaluation](docs/evaluation.md): methods, results, and limits.
- [Model card](docs/model-card.md): intended use, model details, and limitations.
- [Data notes](docs/data.md) and [data license](DATA_LICENSE.md): dataset layout, terms, and attribution.
- [Local walkthrough](docs/local-walkthrough.md): starting the app and exploring it.
- [Deployment readiness](docs/deployment-readiness.md): deployment constraints and items not yet verified.
- [Decision log](decisions.md): project decisions and progress.
