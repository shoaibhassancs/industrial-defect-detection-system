# Data

What data this project uses, where it came from, how it is organised on disk, and what has been checked. License and attribution are in [DATA_LICENSE.md](../DATA_LICENSE.md). Labels: **verified** means a script or a command was run and its output read; **[Unverified]** means it was not.

## Dataset

**MVTec AD** (MVTec Anomaly Detection): photographs of 15 kinds of industrial objects and textures. The training images show only normal ("good") items. The test images contain normal items and items with defects, with pixel masks for the defects. The model is fitted on training normals only.

- Creators: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger (MVTec Software GmbH).
- License: CC BY-NC-SA 4.0 (non-commercial). See [DATA_LICENSE.md](../DATA_LICENSE.md).
- Official page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>. It states the dataset's license terms: reported as checked by the same AI assistant and relayed by the owner on 2026-11-10. Claude could not open the page (the connection failed), so Claude has not checked it; the license wording in this repository comes from the `readme.txt` and `license.txt` inside the local copy.

## Citation

Copied verbatim from the dataset's `readme.txt` (verified by reading the file in the local copy):

> Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern Recognition, 2019

The readme gives no pages or DOI, and none is added here.

**Journal version (IJCV, 2021):**

> Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." *International Journal of Computer Vision*, 129, 1038–1059. <https://doi.org/10.1007/s11263-020-01400-4>

Source: the citation was found by an AI assistant (ChatGPT) looking up Springer's publisher page (<https://link.springer.com/article/10.1007/s11263-020-01400-4>); the project owner passed it on, and Claude copied it from the owner's message on 2026-11-10. The owner did not read the Springer page personally. The assistant reported it as matching the publisher page. Claude could not open that page (it redirected to a login), so Claude has **not** checked it itself. The DOI is the one Claude had already tried to fetch.

## Local copy

- Location: `data/raw/mvtec-ad/` (about 4.9 GB). It is git-ignored and is never committed.
- Download source: **Kaggle**, stated by the project owner. The specific Kaggle page and its terms are **unconfirmed**.
- Not compared with MVTec's official archive (contents and hashes). Only the per-category counts of training, normal test and defective test images were compared, with a table from the official paper that the owner supplied; Claude could not open the paper itself. Treat "same as the official archive" as **[Unverified]** (recorded as assumption A17 in the project notes).
- The folder contains a `readme.txt` and `license.txt` at the top and in each category (the readme and license were checked to be present and identical for `bottle`).

## Layout on disk

```
data/raw/mvtec-ad/
  readme.txt, license.txt
  <category>/            15 categories (below)
    train/good/          normal images only (used for fitting and calibration)
    test/<type>/         normal and defective test images (sealed, see below)
    ground_truth/<type>/ pixel masks for defective test images (sealed)
```

Categories: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.

Counts per category (`train_good` / `test_good` / `test_defective`) are in `scripts/data/expected_counts.json`, with totals 3,629 / 467 / 1,258. They come from the owner-supplied table and the local copy matched them in the verification run (`experiments/dataset_verification.json`, verified by `scripts/data/verify_dataset.py`). `bottle` has 209 training images, 20 normal test images and 63 defective test images. The bottle training images checked in step 6 are all 900×900 RGB.

## Rules for using the data

- **The test set is sealed.** Until the model is frozen and calibrated, the project reads only the names and counts of test files, never their pixels or masks. The check `verify_dataset.assert_allowed` lets scripts open only `<category>/train/good`.
- **The threshold comes from held-out training normals**, never from test images.
- Anomaly scores are not probabilities, and a verdict is a "demo decision", not a pass or fail.

## Splits and derived files

| File | What it is | In git? |
|---|---|---|
| `splits/bottle.json` | The 209 bottle training files split into `fit` (167) and `calib` (42) by a fixed rule (SHA-256 ranking, seed 0), plus the names of the 83 test files. File names only. | Yes |
| `experiments/dataset_verification.json` | Counts, comparison with the expected table, and a hash of the file listing. | Yes |
| `experiments/eda_bottle.json` | Summary numbers (brightness, colour, sharpness) for the fit and calibration groups, and rough comparisons. | Yes |
| `data/processed/eda/*.png` | Sample sheets and histograms made from the training images. These are resized thumbnails of dataset images. | No (git-ignored) |

## Processed and derived images

Anything made from dataset images (thumbnails, heatmaps, overlays, resized or re-encoded copies) is modified material and keeps the attribution in [DATA_LICENSE.md](../DATA_LICENSE.md). So far, nothing of that kind has been published; the only such files are local and git-ignored.

## Pretrained weights

Not used yet. The source and terms of the backbone weights will be recorded here when they are first used (planned for Phase 2).

## Open items

- Confirm the Kaggle page and its terms, or replace the local copy with the official archive and compare.
- Have Claude (or the owner personally) read the official page and the Springer page to confirm the license statement and the IJCV citation first-hand.
- Add the `LICENSE` file for original code (owner's decision).
