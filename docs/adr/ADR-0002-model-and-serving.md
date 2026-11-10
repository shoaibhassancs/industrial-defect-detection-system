# ADR-0002: Headline model rule and serving route (Checkpoint A)

- **Status:** Accepted by the owner on 2026-11-10, with the timing clarification in Decision 1
  (point 2). Drafted as provisional the same day. Accepted before any test image or mask was opened.
- **Scope:** which model is the *headline* model, how the serving route is chosen, and what the
  measurements so far do and do not support. The hosting plan is **not** decided here (Checkpoint B).

## Evidence available today

All measurements use the `bottle` category, `fit` images only, on a MacBook Air M1 (8 GB, AC power).

| | `resnet18.tv_in1k` | `wide_resnet50_2.tv_in1k` |
|---|---|---|
| Fit on 167 images (4 threads) | 320 s | 355 s |
| Fit peak memory | 1,062 MB | 1,907 MB |
| Memory bank | 17,100 x 384 | 17,100 x 1,536 |
| Artifact weights file | 37.4 MB | 204.7 MB |
| Serving proxy, peak memory (whole process) | 988-1,003 MB | 1,042-1,102 MB |
| Serving proxy, median per image (1 / 2 threads) | 131 / 109 ms | 272 / 237 ms |

Sources: `decisions.md` entries 057 and 058; `experiments/fit_*.json`, `experiments/benchmark_*.json`.

### What this supports
- Both backbones can be **fitted on this Mac** (swap rose to about 3.6-4.0 GB for the larger one, and the
  safety guard never triggered). The Phase 2 condition "cannot be fitted on the Mac" is not met.
- On this Mac, natively, one process that imports the code (torch plus anomalib) uses about 0.5 GB
  before any model is loaded, so the per-process peak is about 1.0-1.1 GB for either backbone.
- Scoring dominates the time per image, and the larger backbone takes roughly twice as long.

### What this does not support
- **Detection quality.** No anomalous image has been seen. Nothing here says either backbone detects
  defects better.
- **A Linux container, a CPU quota, Render, or a full web service.** The numbers are a *native macOS
  proxy*. 1-2 threads are rough settings, not equivalents to a CPU quota. The container test (F3) and
  its measurements are still open.
- **A hosting claim.** Whether any plan can hold either model is not known yet.
- The roughly 400 MB figure is a **measurement target, not a requirement**. It is not used to reject a
  route or a backbone.

## Decision 1 - rule that names the headline model (proposed, fixed before test access)

1. **Both** `wide_resnet50_2.tv_in1k` and `resnet18.tv_in1k` are frozen, calibrated, evaluated on the
   sealed test set, and reported in full. Neither is dropped from the report.
2. The **headline model is `wide_resnet50_2.tv_in1k`** (anomalib's default and the PatchCore paper
   configuration), **unless** at Checkpoint B it is found that it cannot be served on any hosting plan
   the owner is willing to use, judged by **container measurements** (not by the native proxy and not by
   test results). In that case the headline model is `resnet18.tv_in1k`, and the report says why.
   **Timing (owner's clarification):** F3 can change the headline only if F3 is **completed before any
   test image or mask is opened**. If F3 has not been completed by then, `wide_resnet50_2.tv_in1k` stays
   the headline, as declared here, and the rule is not changed afterwards.
3. **Test metrics never choose the headline model.** The choice depends only on the hosting
   condition in point 2, which can be decided without opening the test set. This avoids picking the
   better-looking result of two tries on the test data.
4. If the owner prefers to see how large the quality difference is before deciding, that is reported
   alongside, as a result. It does not change the headline.
5. Any later change to this rule needs a new ADR that supersedes this one and is dated before the test
   set is opened. After the first test access, the rule cannot be changed (see the timing in point 2).
6. The headline model's comparison with the other is described as measured on `bottle` only, with the
   confidence intervals from Phase 3 step 5. No claim that one backbone is "better in general" is made.

## Decision 2 - serving route (provisional)

- **S1 (anomalib's `PatchcoreModel` in the serving process) stays the working route.** It is what the
  fits, the benchmark and Phase 3 evaluation use, and it is the reference for any later route.
- **S2 (a thin PyTorch module without anomalib) is not built and not chosen.** The measurement that
  motivates considering it: importing the code costs about 0.5 GB before any model, whereas importing
  torch alone cost about 190 MB. How much S2 would save in a real container is **[Unverified]**.
- The final S1-or-S2 choice, and any hosting claim, wait for the container measurement (F3) and
  Checkpoint B. S2 would need the parity test in PLAN.md section 5.4 (identical verdicts on calibration
  and test images) before use. S3 needs its own ADR.

## Frozen settings (Phase 3 step 1, fixed 2026-11-10 before any test access)

Kept from the fits: backbone tags `wide_resnet50_2.tv_in1k` and `resnet18.tv_in1k`, layers
`layer2`/`layer3`, `num_neighbors` 9, coreset ratio 0.1, split seed 0. New, in the `calibration`
section of `ml/configs/bottle_*.yaml`: alpha 0.05, heatmap lower percentile 1, pixel highlight
percentile 99.5, heatmap upper limit = 1.5 x the highlight level. The multiple **1.5 is an unvalidated
display choice**; it only changes how the heatmap looks, never the verdict, and it is not tuned on
test images. A change to any of these produces a new version, and every evaluated version is listed.

## What stays open
- F3: container measurement with CPU quotas and a memory limit (needs a CPU-only torch index decision).
  It can still change the headline, but only if it is completed before the first test access.
- Checkpoint B: hosting plan.
- Whether the Mac can fit the other 14 categories. Only `bottle` was fitted.
- ImageNet training-data terms (A7) and the Kaggle page / archive comparison (A17) remain unconfirmed.

## Consequences
- Phase 3 can proceed with a frozen configuration for both backbones without waiting for F3.
- If F3 later shows that `wide_resnet50_2` cannot be served, the headline switches by rule, with no
  test-based choice involved, and the evaluation of both is already in hand.
- The risk of this ordering: serving constraints discovered late cannot alter the frozen artifacts
  without producing a new version, and every evaluated version is listed.
