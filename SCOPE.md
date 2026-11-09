# DefectLens — Scope

A short statement of what this project is and is not. The detailed roadmap is in [project-notes/PLAN.md](project-notes/PLAN.md); the original requirements are in [project-notes/project_specifications.md](project-notes/project_specifications.md). If this file and PLAN.md disagree, PLAN.md is the one to update and then this file follows.

## What it is
A personal portfolio web app that demonstrates **industrial visual anomaly detection**. A visitor picks a supported MVTec AD category, uploads an image, and gets:

- a verdict, **Normal** or **Anomalous**, always labelled a *demo decision*;
- the anomaly score, the threshold, and the decision rule;
- a heatmap and overlay showing where the image differs from normal.

A PyTorch **PatchCore** model (via anomalib), fitted only on defect-free training images, produces every result. Nothing is hardcoded or simulated.

## Release 1 (the first milestone)
- One category: `bottle`.
- Real PatchCore inference from a versioned artifact.
- A threshold calibrated on held-out normal training images; the test set stays sealed until the model is frozen.
- Held-out metrics on a Model page (image AUROC, AUPR, pixel AUROC, AUPRO, per-defect recall, failure cases).
- Local `infer` command-line tool.
- FastAPI backend and React frontend, deployed on Render, stateless.
- README, evaluation doc, short model card, dataset attribution.

## Release 2 (only after Release 1)
Several categories, private per-browser history, deterministic PDF reports, an analytics dashboard, end-to-end tests.

## Optional experiments
Custom PatchCore, PaDiM baseline, out-of-distribution warning, other models, all 15 categories. Never required.

## Out of scope
- User accounts.
- Defect-type classification (the model says *whether* and *where*, not *what kind*).
- LLM-written explanations or reports.
- Live camera or video inspection.
- Training from the web UI.
- Claims of certified quality control, calibrated probabilities, or performance on arbitrary products.

## Limits to state honestly
- Educational demo only; not a certified inspection system.
- Meaningful only for supported categories in MVTec-like conditions.
- The score is not a probability or a confidence.
- All metrics describe the MVTec AD test split only.
- The free hosting tier sleeps; the first request after idle can take about a minute.

## Constraints
- Development machine: MacBook Air M1, 8 GB RAM. CPU by default; experiments run one at a time.
- Dataset: MVTec AD (CC BY-NC-SA 4.0). Basic attribution and a non-commercial notice are kept in the docs.
- Code license: MIT for original code, added when the repository goes public. The repository stays private until then.
