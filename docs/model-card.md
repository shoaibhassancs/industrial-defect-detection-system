# Model card (short form): DefectLens PatchCore, `bottle`

A short, honest description of the models in this repository. Results and method details are in
[evaluation.md](evaluation.md); the data and its terms in [data.md](data.md) and
[../DATA_LICENSE.md](../DATA_LICENSE.md). A fuller card is planned for a later phase.

**Status (2026-11-10):** two models, category `bottle` only, version 1.0.0. Educational demo, not a
certified inspection system.

## Model details

- **Method:** PatchCore (Roth et al., CVPR 2022), anomaly detection with a memory bank of normal
  patch features, using anomalib 2.7.0's `PatchcoreModel` as a plain PyTorch module. No anomalous
  image is used for fitting.
- **Versions:** `wide_resnet50_2.tv_in1k` (**headline**, chosen by a rule fixed before test access,
  ADR-0002) and `resnet18.tv_in1k` (also reported). Layers `layer2` and `layer3`, 9 neighbours,
  coreset ratio 0.1.
- **Backbone weights:** torchvision's ImageNet weights as re-hosted by timm, pinned by tag and by
  file hash (`docs/data.md`). The Hugging Face model cards of the two weight files state
  `license: bsd-3-clause` (read on 2026-11-10; not checked against another source). The terms of the
  ImageNet training data are **not verified** and are not inferred from that statement.
- **Input:** one RGB image, resized to 256 × 256 (bilinear, antialiased), ImageNet-normalised.
- **Output:** an image-level anomaly score, a 256 × 256 anomaly map, and, from the calibration
  stored with the model, a verdict (`score > threshold`), the score's rank among normal calibration
  scores, a fixed-scale heatmap, and a highlight outline.
- **Artifact:** `model.safetensors` plus `manifest.json` (settings, hashes, freeze and calibration
  blocks). Model files are not committed.

## Intended use

- Learning, demonstration and portfolio use: showing how an unsupervised anomaly detector can flag
  and localise surface defects on MVTec AD `bottle` images.
- Images like the MVTec AD `bottle` images: a bottle seen from above, centred, with controlled
  lighting and a plain background.

## Out-of-scope use

- Any real quality-control, safety or production decision. The system is not certified and does not
  replace human inspection.
- Other products, cameras, lighting or backgrounds. The model **cannot reliably detect** that an
  image is out of scope; it may give an arbitrary score for it.
- Identifying the *type* of a defect. It reports whether and where an image looks unlike normal.
- Treating the score as a probability, a confidence or a defect severity. It is none of these.
- Commercial use: the dataset license forbids it (CC BY-NC-SA 4.0).

## Data

- **Training and calibration:** the 209 normal training images of MVTec AD `bottle` (167 to build the
  memory bank, 42 held out for calibration), split once by a seeded rule (`splits/bottle.json`).
- **Evaluation:** the official `bottle` test set, 83 images (20 normal, 63 defective with masks),
  kept sealed until the models were frozen and calibrated.
- The dataset is not in this repository. Its source for this project is a Kaggle copy whose page and
  comparison with the official archive are **unconfirmed**.

## Calibration and its guarantee

The threshold is the 41st smallest of the 42 calibration scores. If a new normal image is drawn
like the calibration images, the chance it is flagged is about 2/43 (4.65%) on average over possible
calibration sets. This concerns **normal** images only, assumes the new images are exchangeable with the
calibration images, and is coarse with 42 images (details: [evaluation.md](evaluation.md) §3).

## Results (sealed test set, `bottle`, v1.0.0)

| | `wide_resnet50_2` (headline) | `resnet18` |
|---|---|---|
| Image AUROC | 1.000 | 1.000 |
| Defective images flagged | 63 of 63 | 63 of 63 |
| **Normal images flagged (target 5%)** | **3 of 20 (15%)** | **5 of 20 (25%)** |
| Pixel AUROC / AUPRO (at 256 × 256) | 0.987 / 0.948 | 0.983 / 0.936 |

Interpretation, in short:
- Defective and normal test images separate completely by score, for both models.
- The false-alarm rates on the normal test images are above the 5% level aimed for. With 20 normal
  test images and a 42-image calibration set, these rates are very coarse estimates; they are
  not unusual for the headline model under the stated assumptions and less consistent with the
  nominal rate for `resnet18`.
- Intervals around perfect separation are degenerate (1.000–1.000) and do not show that errors cannot
  occur on new images.

## Failure modes seen

- **False alarms** on a few normal bottles, with scores only just above the threshold. In the figures
  these images show bright orange streaks on the glass rim and the flagged patches lie on the rim.
  That is an **observation from the pictures, not a proven cause**.
- **Over-wide outlines:** the highlight outline covers almost all defect pixels but also many normal
  ones (pixel precision about 0.30), so it overstates the defect size.
- **No missed defects** among the 63 test defects, so no false-negative examples exist; this says
  little about defect types or sizes not present in the test set.

## Limitations

One category, one split seed, one run, a small test set, a small calibration set; exchangeability is
assumed, not shown; pixel metrics at 256 × 256 only; the heatmap multiple 1.5 is an unvalidated
display choice; no serving, hosting or container measurement exists yet (only native macOS proxy
timings in `decisions.md` entry 058). Full list: [evaluation.md](evaluation.md) §6.

## Ethical and legal notes

Trained on, and evaluated with, MVTec AD (CC BY-NC-SA 4.0, MVTec Software GmbH); the figures in
`evaluation/` contain adapted dataset images and carry that license. No endorsement by MVTec or the
dataset authors is implied. The project code has no licence granting reuse rights: no `LICENSE` file
exists and the owner has decided not to add one (2026-11-11). This does not change the terms above.

## References

- Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P. (2022). *Towards Total Recall
  in Industrial Anomaly Detection.* CVPR 2022.
- Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C. (2021). *The MVTec Anomaly
  Detection Dataset.* International Journal of Computer Vision 129, 1038–1059.
  (Citation relayed by the project owner; see [data.md](data.md) for its source.)
