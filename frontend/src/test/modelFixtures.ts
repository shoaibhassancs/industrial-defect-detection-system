import { readFileSync } from "node:fs";
import { join } from "node:path";

/** The REAL saved evaluation of the headline model (tracked in the repository). The Model page
 * tests feed these values through the page and check what is shown. */
export const METRICS_FILE = join(
  process.cwd(),
  "..",
  "evaluation",
  "wide_resnet50_2_tv_in1k",
  "v1.0.0",
  "bottle",
  "metrics.json",
);

interface SavedInterval {
  estimate: number;
  low: number;
  high: number;
  confidence: number;
}

/** The parts of the saved results that the page shows (the file has more). */
export interface SavedResults {
  image: {
    n_normal: number;
    n_anomalous: number;
    auroc: SavedInterval;
    aupr: SavedInterval;
    at_calibrated_threshold: {
      tp: number;
      fp: number;
      tn: number;
      fn: number;
      precision: number;
      f1: number;
      threshold: number;
    };
    false_alarm_on_normal_test_images: {
      false_alarms: number;
      n_normal: number;
      fpr: number;
      alpha: number;
      bound_for_new_normal_images: number;
    };
    per_defect_recall: Record<string, { detected: number; n: number; recall: number }>;
  };
  pixel: {
    auroc: number;
    "aupro_fpr_0.3": number;
    n_images: number;
    resolution: string;
  };
}

export function savedResults(): SavedResults {
  return JSON.parse(readFileSync(METRICS_FILE, "utf8")).results as SavedResults;
}

/** What GET /api/v1/metrics/bottle returns: the package's `results`, which a backend check proves
 * equal to metrics.json (scripts/tests/test_evaluation_matches_package.py). */
export function metricsResponse(results: SavedResults = savedResults()) {
  return {
    category: "bottle",
    version: "1.0.0",
    status: "evaluated on the official test set; nothing tuned on it",
    results,
    curves_available: false,
    limitations: "docs/evaluation.md describes the protocol and its limits.",
  };
}

/** What GET /api/v1/model returns (field values as the service builds them from the package). */
export function modelResponse(overrides: Record<string, unknown> = {}) {
  return {
    algorithm: "PatchCore (memory bank of normal patch features)",
    name: "bottle_wide_resnet50_2_tv_in1k",
    backbone: "wide_resnet50_2.tv_in1k",
    layers: ["layer2", "layer3"],
    preprocessing: {},
    calibration: {
      method: "split-conformal on held-out normal training images",
      alpha: 0.05,
      n: 42,
      k: 41,
      threshold: 1.3681522607803345,
      false_alarm_bound: 0.046511627906976744,
      meaning:
        "Chance that a new normal image is flagged, on average over calibration sets, if it is exchangeable with the calibration images. Says nothing about defects.",
    },
    version: "1.0.0",
    headline_model: true,
    limitations: [
      "Educational demo trained on MVTec AD; not a certified inspection system.",
      "Only meaningful for bottle images captured like the MVTec AD dataset.",
      "It can not reliably tell that an image is out of scope.",
      "The score is an anomaly score, not a probability or a confidence.",
      "Metrics describe one category, one split and a small test set.",
    ],
    ...overrides,
  };
}
