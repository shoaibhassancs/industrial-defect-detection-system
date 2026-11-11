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

- **The test set is sealed.** Until the model is frozen and calibrated, the project reads only the names and counts of test files, never their pixels or masks. The guard `defect_ml.data.mvtec.assert_training_path` (defined once; the dataset scripts and the training loader all use it) lets them open only files directly in `<category>/train/good`.
- **The threshold comes from held-out training normals**, never from test images.
- Anomaly scores are not probabilities, and a verdict is a "demo decision", not a pass or fail.

## How the sealed test set is opened later

The training and calibration loader (`defect_ml.data.mvtec.TrainingImages`) reads only the files listed in `splits/<category>.json` for its group (`fit` or `calib`), checks each against the verified manifest, and prepares it with `defect_core.preprocessing`. It cannot reach `test/` or `ground_truth/`, and it has no flag, argument or environment variable that changes that. This is a rule about the **training path**, not a permanent ban on the data.

After the model is frozen and calibrated (Phase 3), the held-out test images and masks are read by a **separate, explicit component**, `defect_ml/evaluate.py` (written; run once per version, first run for `bottle` on 2026-11-10). By design that component:

1. refuses to run unless the artifact's manifest is frozen and calibrated;
2. reads test files through its own reader, with its own check of which paths it may open, and not through the training loader;
3. appends every file it opens to `evaluation/test_access_log.jsonl`, an append-only log;
4. together with `defect_ml/visualize.py` (figures; it imports `evaluate.py`'s reader, path check and log, tags its reads `purpose: "visualize"`, and requires its recomputed scores to equal the saved `predictions.csv` exactly), is the only code that reads `test/` and `ground_truth/`.

So opening the test set is a deliberate, logged act of a different component, and fitting or calibration code cannot do it by accident. If the model or its settings change afterwards, that is a new version and a new sealed evaluation (see `PLAN.md` section 5.6).

## Splits and derived files

| File | What it is | In git? |
|---|---|---|
| `splits/bottle.json` | The 209 bottle training files split into `fit` (167) and `calib` (42) by a fixed rule (SHA-256 ranking, seed 0), plus the names of the 83 test files. File names only. | Yes |
| `experiments/dataset_verification.json` | Counts, comparison with the expected table, and a hash of the file listing. | Yes |
| `experiments/fit_bottle_resnet18_tv_in1k_fit-v0.json`, `experiments/fit_bottle_wide_resnet50_2_tv_in1k_fit-v0.json` | Run records of the two real `bottle` fits on the `fit` images: timings, memory, bank shape, weight hashes, the audit of which dataset files were opened, system state. | Yes |
| `artifacts/<run_name>/` | The fitted, not yet frozen or calibrated artifacts (`model.safetensors`, `manifest.json`). 37 MB and 205 MB. | No (git-ignored) |
| `experiments/loader_check_bottle.json` | Evidence that the fit/calibration loader works on the real `bottle` training images: shapes, value statistics, and the list of files the running process opened (from a Python audit hook). | Yes |
| `experiments/eda_bottle.json` | Summary numbers (brightness, colour, sharpness) for the fit and calibration groups, and rough comparisons. | Yes |
| `data/processed/eda/*.png` | Sample sheets and histograms made from the training images. These are resized thumbnails of dataset images. | No (git-ignored) |

## Processed and derived images

Anything made from dataset images (thumbnails, heatmaps, overlays, resized or re-encoded copies) is modified material and keeps the attribution in [DATA_LICENSE.md](../DATA_LICENSE.md). Some of it is already public: the four evaluation figures (`evaluation/*/v1.0.0/bottle/figures/*.png`) are tracked in Git and have been on the public GitHub repository since 2026-11-10 (checked 2026-11-11: the repository is public and all four files could be fetched without logging in). They contain MVTec AD images, resized and overlaid with heatmaps and outlines, and they carry [evaluation/NOTICE.md](../evaluation/NOTICE.md). Whether that notice is sufficient, and whether their publication needs MVTec's confirmation, is **unresolved** (see [licensing-review.md](licensing-review.md)). Other local sample sheets and histograms are written to `data/processed/`, which is git-ignored.

## Pretrained weights

The backbones are initialised with ImageNet-pretrained weights, pinned **by tag** in the backbone name, so a run cannot silently use a library default. The tags are torchvision's original ImageNet weights as re-hosted by timm (the pinning choice follows the PatchCore paper's use of torchvision models; it does not guarantee reproducing the paper's numbers). Downloaded on 2026-11-10 into the Hugging Face cache (`~/.cache/huggingface`, outside the repository, never committed). Each record below was made by the fit run itself, which also checked that every tensor of the file that the model keeps is equal to the model's tensor (so the hash describes the weights actually used).

| | `resnet18.tv_in1k` | `wide_resnet50_2.tv_in1k` |
|---|---|---|
| Hub repository | `timm/resnet18.tv_in1k` | `timm/wide_resnet50_2.tv_in1k` |
| File | `model.safetensors` | `model.safetensors` |
| Hub revision (commit) | `bbd144b3e5565108aad885f145491d11bc6ce807` | `343c5fa59ca8a131e0cdf42144a641b36a0387c2` |
| Size | 46,807,446 bytes | 275,835,296 bytes |
| SHA-256 | `694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81` | `df6fb6c4824769769de18e14088475fd6ee94236849aa4e5d8022ba9d9a9a16c` |
| Tensors compared with the model / differing | 90 / 0 | 258 / 0 |
| Original source named by timm | `https://download.pytorch.org/models/resnet18-f37072fd.pth` | `https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth` |

**Terms of the weight files (what is verified, what is not):**
- **Verified (read 2026-11-10 through the Hugging Face connector, repository metadata of both repositories):** the model cards declare `license: bsd-3-clause`. That is a statement made by the model cards; this project did not check it against any other source.
- **Not verified, and not inferred from the card:** the terms of the ImageNet data the weights were trained on (no statement of them appeared in the repository metadata that was read); whether the original torchvision files carry any additional terms. This is a noncommercial educational demo, but nothing here is a legal conclusion. The card's code-style license is **not** treated as the terms of the training data.
- The exact weights are also recorded in each fitted artifact's manifest (`model.backbone` carries the tag; `provenance.weights` the revision and hash) and in `experiments/fit_*.json`.

## Open items

- Confirm the Kaggle page and its terms, or replace the local copy with the official archive and compare.
- Have Claude (or the owner personally) read the official page and the Springer page to confirm the license statement and the IJCV citation first-hand.
- Add the `LICENSE` file for original code (owner's decision).
