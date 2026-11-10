# Evaluation

How the DefectLens models were evaluated on the MVTec AD `bottle` category, and exactly what the
saved results show. Every number below comes from a committed file in `evaluation/`,
`experiments/`, `splits/` or `docs/adr/`; the file is named next to it. The numbers describe the
MVTec AD test split of one category and nothing else.

**Status (2026-11-10):** two versions evaluated, both `v1.0.0` of `bottle`: `wide_resnet50_2`
(the headline model, fixed by rule before any test access) and `resnet18`. Both are reported.
No other category has been fitted or evaluated.

## 1. Versions evaluated

All evaluated versions are listed here, as PLAN §5.6 requires. None was left out.

| Version | Backbone (weights tag) | Role | Result folder |
|---|---|---|---|
| `bottle` v1.0.0 | `wide_resnet50_2.tv_in1k` | **Headline** (ADR-0002) | `evaluation/wide_resnet50_2_tv_in1k/v1.0.0/bottle/` |
| `bottle` v1.0.0 | `resnet18.tv_in1k` | Secondary candidate, also reported | `evaluation/resnet18_tv_in1k/v1.0.0/bottle/` |

Each version was evaluated **once**. The evaluation code refuses to overwrite a result folder. If a
setting or the method is ever changed, that makes a new version with its own sealed evaluation,
listed in this table next to these two.

## 2. Protocol

### 2.1 Data

The 209 normal training images of `bottle` were split once, by a seeded rule written in
`defect_ml/data/splits.py` (SHA-256 ranking, seed 0), recorded in `splits/bottle.json`:

| Group | Images | Used for |
|---|---|---|
| `fit` | 167 | Building the memory bank |
| `calib` | 42 | Calibration only (threshold, rank table, heatmap scale, highlight level) |
| `test` | 83: 20 normal, 63 defective | **Sealed**; reporting only |

The 63 defective test images are of three types, with a pixel mask each: `broken_large` (20),
`broken_small` (22), `contamination` (21). The training set contains no anomalies, so there is no
validation set with defects: nothing about defects could be measured before the test set was
opened (PLAN §5.4). The data and its terms are described in [data.md](data.md).

### 2.2 Order of events and what was kept sealed

1. **Fit** on the 167 `fit` images only (`experiments/fit_*.json`; the data-access audit shows
   exactly those files were opened).
2. **Freeze** the fitted artifact (`experiments/freeze_*.json`). The settings of the next step were
   fixed in `ml/configs/` and in ADR-0002 before this point.
3. **Calibrate** on the 42 `calib` images only (`experiments/calibration_*.json`; audit: exactly
   those 42 files).
4. **Decide the headline rule** (ADR-0002, accepted before any test access; addendum: F3 skipped
   for the headline, so `wide_resnet50_2` is the headline and cannot change).
5. **Sealed evaluation** of both models in one planned batch, one after the other, identical code
   and settings, nothing changed between them (`evaluation/test_access_log.jsonl`: each model's
   `start` line comes after its freeze time).
6. **Figures** (`visualize.py`), descriptive only, through the same logged path.

The threshold comes from held-out *training* normals, never from test images. The test set was
opened only by `defect_ml/evaluate.py` and `defect_ml/visualize.py`, which share one reader, one
path check and one log. The training loader cannot reach those files. Every file opened is in
`evaluation/test_access_log.jsonl` (592 lines: the evaluation and the figures step each read the
same 146 files per model, 83 images and 63 masks, none from `train/`; each line carries the hash of the line before it, so a changed or removed line is
detectable with `verify_access_log`). An independent audit hook recorded the same file set.
The data manifest holds no hashes of test files, so the log records each file's SHA-256 as read.

### 2.3 Settings (frozen before test access)

| Setting | Value |
|---|---|
| Layers / neighbours / coreset ratio | `layer2`, `layer3` / 9 / 0.1 |
| Preprocessing | RGB, 256×256 bilinear with antialiasing, ImageNet normalisation, no crop |
| Error level α | 0.05 |
| Heatmap lower limit / highlight level | 1st / 99.5th percentile of the calibration pixels |
| Heatmap upper limit | 1.5 × the highlight level. **1.5 is an unvalidated display choice** |
| Bootstrap | 1,000 resamples, seed 0, 95% percentile interval |

Source: `ml/configs/bottle_*.yaml` and each artifact's manifest.

## 3. Calibration: what the threshold means and what it does not

The verdict rule is `score > threshold`, strictly. The threshold is the *k*-th smallest of the *n*
image scores of the held-out calibration images, with *k* = ceil((*n* + 1)(1 − α)). Here *n* = 42,
so *k* = 41: the threshold is the 41st smallest calibration score (**1.3682** for
`wide_resnet50_2`, **1.2961** for `resnet18`; `experiments/calibration_*.json`).

**The guarantee, stated carefully.** This is split-conformal calibration. If a new normal image is
exchangeable with the calibration images (drawn the same way), the chance that it is flagged is
(*n* + 1 − *k*)/(*n* + 1) = 2/43 ≈ **4.65%**, which is at most α. Three limits apply:

1. It is about **normal** images only. It says nothing about how many defects are found.
2. It is an **average over possible calibration sets**. For one fixed set of 42 images, the true
   false-alarm rate of the resulting threshold is itself uncertain: it follows a Beta(2, 41)
   distribution with mean 4.65% and standard deviation about 3.2% (standard result; my
   calculation of the numbers). With *n* = 42 the threshold is coarse.
3. It requires the images to be exchangeable. Whether the official test normals behave like the
   training normals is **not known**, and cannot be checked from these data alone.

The score is an anomaly score, not a probability or a confidence. The two verdicts are "demo
decisions"; the project does not use "pass" or "fail".

## 4. Metrics

All metrics are computed by `defect_ml/metrics.py`, tested on synthetic inputs with hand-worked
answers, and (for AUPRO) checked against anomalib's own algorithm on a small fixture.

| Metric | Definition |
|---|---|
| Image AUROC, AUPR | Threshold-free. Anomalous is the positive class. AUPR is average precision (scikit-learn). Interval: percentile bootstrap that resamples **images**, separately within the normal and the defective ones |
| At the threshold | TP, FP, TN, FN with the strict rule above; TPR, FPR, precision, F1. A rate with a zero denominator would be reported as undefined, not zero |
| Recall per defect type | Detected / total for each folder name (folder names are used for evaluation only) |
| Pixel AUROC | Over every pixel of every image. Inflated by the many easy background pixels, so it is reported next to AUPRO |
| AUPRO (FPR ≤ 0.3) | MVTec's region-overlap metric: per-region overlap averaged over the 8-connected ground-truth regions (each region counts equally), against the false-positive rate over all normal pixels, area up to 0.3 divided by 0.3 |
| Highlight pixels | Precision and recall of the pixels strictly above the calibrated highlight level, which draws the outline in the app |

**Resolution.** Pixel metrics are computed at **256 × 256**, the model's map size: the ground-truth
masks were resized to 256 × 256 with nearest-neighbour. They were **not** computed at the original
900 × 900 size, where the test set is tens of millions of pixels. Resizing can change localisation
numbers; here no mask became empty and the resized areas stayed within 1% of the original scaled
(`mask_pixel_counts` in each `metrics.json`). The PatchCore paper reports that segmentation
quality depends on input resolution, so these numbers should not be compared directly with
published ones.

## 5. Results

Source: `evaluation/<model>/v1.0.0/bottle/metrics.json` (also `predictions.csv`, `curves.json`,
`histograms.json`).

### 5.1 Image level (83 test images: 20 normal, 63 defective)

| | `wide_resnet50_2` (headline) | `resnet18` |
|---|---|---|
| AUROC (95% bootstrap interval) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) |
| AUPR | 1.000 | 1.000 |
| Threshold | 1.3682 | 1.2961 |
| Defective images flagged (TPR) | 63 of 63 (1.00) | 63 of 63 (1.00) |
| **Normal images flagged (FPR), target α = 0.05** | **3 of 20 (0.15)** | **5 of 20 (0.25)** |
| Precision / F1 | 0.955 / 0.977 | 0.926 / 0.962 |
| Recall: `broken_large` / `broken_small` / `contamination` | 20/20, 22/22, 21/21 | 20/20, 22/22, 21/21 |

The highest normal score is below the lowest defective score for both models (1.488 against 1.811
for `wide_resnet50_2`; 1.349 against 1.608 for `resnet18`), so the two groups separate completely
at the image level. The 1.000–1.000 interval reflects that: when a sample separates perfectly, every
resample does too. It is **not** a proof that errors cannot occur on new images; it shows only that
no error appeared among these 83.

### 5.2 Pixel level (at 256 × 256, see §4)

| | `wide_resnet50_2` | `resnet18` |
|---|---|---|
| Pixel AUROC | 0.987 | 0.983 |
| AUPRO (FPR ≤ 0.3) | 0.948 | 0.936 |
| Highlight level: pixel precision / recall | 0.296 / 0.9997 | 0.302 / 0.9987 |

The highlight level catches almost every defect pixel but also many normal ones, so the outline in
the app **overstates** the defect extent. It is a display aid, not part of the verdict.

### 5.3 The false alarms

The false-alarm rates on normal test images (15% and 25%) are higher than the 5% level aimed for.
What can and cannot be said:

- **Size of the effect.** With only 20 normal test images, one image is 5 percentage points, and
  the estimates are very coarse. The numbers of false alarms are 3 and 5.
- **How surprising, if the guarantee held.** Using the Beta(2, 41) uncertainty of §3 (so allowing for
  the coarse threshold), the chance of at least 3 false alarms among 20 is about **0.09**
  (`wide_resnet50_2`) and of at least 5 about **0.011** (`resnet18`) (beta-binomial; my
  calculation, assuming exchangeable normal images). So the headline model's result is not
  unusual for a threshold this coarse, and the `resnet18` result is harder to reconcile with the
  nominal rate. This does not settle whether the test normals differ from the training normals.
- **Margins.** The flagged normal images are just above the threshold: `wide_resnet50_2` 1.407,
  1.488 and 1.382 against 1.368; `resnet18` 1.349, 1.300, 1.317, 1.342 and 1.328 against 1.296
  (one image by 0.004). Two of the three flagged `wide_resnet50_2` images and three of the five for
  `resnet18` score above all 42 calibration scores (their normal rank is 42 in `predictions.csv`).
  The middle of the test-normal scores matches the calibration scores closely (`wide_resnet50_2`
  median 1.209 against 1.205), so the difference is in the upper tail, which 20 images cannot pin
  down.
- **Which images.** The same three normal images (`test/good/001`, `015`, `019`) are flagged by both
  models; `resnet18` also flags `002` and `009`.
- **Observation, not a proven cause.** In the figures (`figures/false_positives.png` in each result
  folder) these bottle mouths show bright orange streaks on the glass rim, and the highlighted patches lie
  on that rim. This is **what the pictures look like to a reader**. It was not tested; reflections
  may or may not be the reason, and no experiment here separates them from other differences.
- **No change was made.** A threshold that gave zero false alarms on these 20 images exists, because
  the groups separate. Choosing it from test data would be tuning on the test set, so it was not
  adopted. A different α, more calibration images, or another rule would be a new version, proposed
  by the owner, with its own sealed evaluation, listed in §1.

### 5.4 Defect size

`figures/mask_size_analysis.json`: no relationship between the ground-truth area and the score was
seen (rank correlation 0.08 for `wide_resnet50_2`, −0.08 for `resnet18`); every defect in the smallest
third of areas (0.6%–3.2% of the image) was detected; median pixel recall at the highlight level is
1.0 in every group, and lowest for the largest mask (`broken_large/003`, 27% of the image: 0.995 and
0.979). `contamination` images have the lowest defective scores (minimum 1.811 and 1.608). This is
descriptive and based on 63 images.

### 5.5 Comparing the two models

`wide_resnet50_2` has fewer false alarms and slightly higher pixel scores than `resnet18` here. This
is one category, one split seed, one calibration set and one run, with no interval that separates
them at the image level. **No claim is made that one backbone is better in general.** The headline
was fixed by rule before test access (ADR-0002) and does not depend on these results.

## 6. Limitations

- **Small test set.** 20 normal and 63 defective images. Rates move in steps of 5% (normal) and
  1.6% (defective); intervals at perfect separation are degenerate.
- **Small calibration set.** 42 images give a coarse threshold (§3).
- **One category, one seed, one run.** Nothing here says how the method behaves on the other 14
  categories, or with another split.
- **Exchangeability is assumed, not shown** (§3).
- **Pixel metrics at 256 × 256 only.**
- **MVTec AD only.** The images are controlled, centred and well lit. The models cannot reliably
  detect out-of-scope images and do not identify defect types.
- **Scores are not probabilities.** Verdicts are demo decisions.
- **Display choices.** The heatmap multiple 1.5 is unvalidated; the highlight outline overstates
  defect extent.
- **Not measured here:** serving memory, latency and hosting. The only serving numbers so far are the
  native macOS proxy measurements of `decisions.md` entry 058 and `dev-environment.md`; they are not
  container or hosting measurements (F3 is not done).

## 7. Not done

- **The optional anomalib reference check** (PLAN Phase 3 step 8): not run. See the recommendation
  in `decisions.md` entry 067.
- **Other categories**, F3 (container measurement), and a clean-checkout reproduction test (the
  Makefile targets of §8 exist but have not been run from a clean checkout).

## 8. Reproducing the results

With the dataset at `data/raw/mvtec-ad/` and the committed `splits/` and `data/data_manifest.json`,
the steps run in this order, one model at a time (the targets are in the `Makefile`):

```
make fit freeze calibrate        # fit on `fit`, freeze, calibrate on `calib`; no test file is opened
make evaluate CONFIRM_OPEN_TEST_SET=yes   # the sealed test set, logged, once per version
make figures  CONFIRM_OPEN_TEST_SET=yes   # example figures (re-reads the test files, logged)
make package                     # package for publishing; reads only `calib` images; no upload
```

`make evaluate` and `make figures` stop with a message unless the confirmation is given in the same
command, because opening the test set is a deliberate act. They are thin wrappers: the commands are
`python -m defect_ml.<step> <settings files> --version 1.0.0` with the two files
`ml/configs/bottle_wrn50.yaml` and `ml/configs/bottle_resnet18.yaml` (override with `CONFIGS=` and
`VERSION=`). Every step refuses to overwrite an earlier result, so a repeat needs a new `VERSION` or
the earlier outputs moved aside; the refusals were run on the existing results and changed nothing. The
fits download the pinned ImageNet weights (`docs/data.md`). **Not verified:** a run from a clean
checkout, and that a fresh run reproduces the numbers bit for bit on another machine; the scores were
identical when recomputed twice on the same Mac (the figures step checks this), and the coreset is
seeded.

## 9. Files

| File | Content |
|---|---|
| `evaluation/test_access_log.jsonl` | Every test file opened, with SHA-256 and the hash chain |
| `evaluation/<model>/v1.0.0/bottle/metrics.json` | All metrics, settings, access audit, system state, limitations |
| `.../predictions.csv` | One row per test image: score, verdict, normal rank |
| `.../curves.json`, `.../histograms.json` | ROC, precision-recall, PRO curve (thinned), score histograms |
| `.../figures/` | Example sheets, `mask_size_analysis.json`, `index.json` |
| `experiments/{fit,freeze,calibration}_*.json` | Records of the earlier stages |
| `docs/adr/ADR-0002-model-and-serving.md` | The headline rule and the F3 timing decision |

The figures contain adapted MVTec AD images (resized, with overlays). They are not covered by the
project's code license; see [../DATA_LICENSE.md](../DATA_LICENSE.md).
