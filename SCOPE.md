# DefectLens — Scope

A short statement of what this project is and is not. The detailed roadmap is in [project-notes/PLAN.md](project-notes/PLAN.md); the original requirements are in [project-notes/project_specifications.md](project-notes/project_specifications.md). If this file and PLAN.md disagree, PLAN.md is the one to update and then this file follows.

## What it is
A personal portfolio web app that demonstrates **industrial visual anomaly detection**. A visitor picks a supported MVTec AD category, uploads an image, and gets:

- a verdict, **Normal** or **Anomalous**, always labelled a *demo decision*;
- the anomaly score, the threshold, and the decision rule;
- a heatmap and overlay showing where the image differs from normal.

A PyTorch **PatchCore** model (via anomalib), fitted only on defect-free training images, produces every result. Nothing is hardcoded or simulated.

## Scope of this portfolio version (owner decision, 2026-11-11)
This version is a **focused, local `bottle` demo**: a fitted and calibrated PatchCore model, a documented sealed evaluation, a local API and web app, and screenshots. **Only the `bottle` category was evaluated, and no hosted demo exists.** Phases 7 to 14 of the roadmap are **deferred for this version**: not done, and not permanently cancelled. They can be revisited if a concrete need arises.

| Phases | Deferred because |
|---|---|
| 7 (hosted demo) and 12 (production deploy) | Paid hosting is ruled out; the local app, screenshots and saved evaluation already show the project. A production deploy is not needed for a local portfolio version. |
| 8 (more categories), 9 (saved history), 10 (PDF reports, dashboard), 14 (extra model studies) | Beyond the focused `bottle` demo. |
| 11 (hardening, accessibility, end-to-end tests, observability) | Not needed for a local portfolio version; the basic checks in the README stay. |
| 13 (final polish) | Revisit if a concrete need arises. |

The sections "Release 1" and "Release 2" below are the original plan, kept for reference; they are not all met.

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
- Repository: intentionally public (owner's statement, 2026-11-10). The project is noncommercial.
- Code license: none. No `LICENSE` file exists and the owner has decided not to add one (2026-11-11); the code is public and viewable, but no reuse licence is granted. This does not change the terms for the dataset, figures or weights.
