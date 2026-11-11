// Shapes of GET /api/v1/model and GET /api/v1/metrics/{category}.
//
// These two endpoints are not described field by field in the generated contract (the backend
// returns free-form dictionaries), so the shapes are written here by hand and every answer is
// checked by the parsers below. A missing or non-numeric field makes the call fail with
// BAD_RESPONSE instead of showing a wrong or empty number. The page only DISPLAYS these values;
// nothing here (or on the page) recomputes a metric. A test feeds the real, tracked
// evaluation/**/metrics.json through this code and checks every number on screen.
import { ApiError } from "./errors";

export interface Interval {
  estimate: number;
  low: number;
  high: number;
  confidence: number | null;
}

export interface ThresholdCounts {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  tpr: number;
  fpr: number;
  precision: number;
  f1: number;
  threshold: number;
}

export interface FalseAlarms {
  false_alarms: number;
  n_normal: number;
  fpr: number;
  alpha: number;
  bound_for_new_normal_images: number;
}

export interface DefectRecall {
  type: string;
  detected: number;
  n: number;
  recall: number;
}

export interface ImageResults {
  n_normal: number;
  n_anomalous: number;
  auroc: Interval;
  aupr: Interval;
  at_calibrated_threshold: ThresholdCounts;
  false_alarm_on_normal_test_images: FalseAlarms;
  per_defect_recall: DefectRecall[];
}

export interface PixelResults {
  auroc: number;
  aupro_fpr_0_3: number;
  n_images: number;
  resolution: string;
}

export interface MetricsInfo {
  category: string;
  version: string;
  status: string;
  image: ImageResults;
  pixel: PixelResults;
}

export interface CalibrationInfo {
  method: string;
  alpha: number;
  n: number;
  k: number;
  threshold: number;
  false_alarm_bound: number;
  meaning: string;
}

export interface ModelInfo {
  algorithm: string;
  name: string;
  backbone: string;
  layers: string[];
  version: string;
  headline_model: boolean | null;
  calibration: CalibrationInfo;
  limitations: string[];
}

function bad(what: string): never {
  throw new ApiError({ code: "BAD_RESPONSE", message: `Unexpected answer: ${what}` });
}

function rec(value: unknown, what: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) bad(what);
  return value as Record<string, unknown>;
}

function num(obj: Record<string, unknown>, key: string, what: string): number {
  const v = obj[key];
  if (typeof v !== "number" || !Number.isFinite(v)) bad(`${what}.${key} is not a number`);
  return v as number;
}

function str(obj: Record<string, unknown>, key: string, what: string): string {
  const v = obj[key];
  if (typeof v !== "string") bad(`${what}.${key} is not text`);
  return v as string;
}

function interval(value: unknown, what: string): Interval {
  const o = rec(value, what);
  const confidence = o["confidence"];
  return {
    estimate: num(o, "estimate", what),
    low: num(o, "low", what),
    high: num(o, "high", what),
    confidence: typeof confidence === "number" ? confidence : null,
  };
}

export function parseModelInfo(value: unknown): ModelInfo {
  const o = rec(value, "model");
  const c = rec(o["calibration"], "model.calibration");
  const layers = o["layers"];
  const limitations = o["limitations"];
  if (!Array.isArray(layers) || !layers.every((x) => typeof x === "string")) bad("model.layers");
  if (!Array.isArray(limitations) || !limitations.every((x) => typeof x === "string")) {
    bad("model.limitations");
  }
  const headline = o["headline_model"];
  return {
    algorithm: str(o, "algorithm", "model"),
    name: str(o, "name", "model"),
    backbone: str(o, "backbone", "model"),
    layers: layers as string[],
    version: str(o, "version", "model"),
    headline_model: typeof headline === "boolean" ? headline : null,
    calibration: {
      method: str(c, "method", "model.calibration"),
      alpha: num(c, "alpha", "model.calibration"),
      n: num(c, "n", "model.calibration"),
      k: num(c, "k", "model.calibration"),
      threshold: num(c, "threshold", "model.calibration"),
      false_alarm_bound: num(c, "false_alarm_bound", "model.calibration"),
      meaning: str(c, "meaning", "model.calibration"),
    },
    limitations: limitations as string[],
  };
}

export function parseMetrics(value: unknown): MetricsInfo {
  const o = rec(value, "metrics");
  const results = rec(o["results"], "metrics.results");
  const image = rec(results["image"], "metrics.results.image");
  const pixel = rec(results["pixel"], "metrics.results.pixel");
  const at = rec(image["at_calibrated_threshold"], "image.at_calibrated_threshold");
  const fa = rec(image["false_alarm_on_normal_test_images"], "image.false_alarm");
  const recalls = rec(image["per_defect_recall"], "image.per_defect_recall");
  return {
    category: str(o, "category", "metrics"),
    version: str(o, "version", "metrics"),
    status: str(o, "status", "metrics"),
    image: {
      n_normal: num(image, "n_normal", "image"),
      n_anomalous: num(image, "n_anomalous", "image"),
      auroc: interval(image["auroc"], "image.auroc"),
      aupr: interval(image["aupr"], "image.aupr"),
      at_calibrated_threshold: {
        tp: num(at, "tp", "at_calibrated_threshold"),
        fp: num(at, "fp", "at_calibrated_threshold"),
        tn: num(at, "tn", "at_calibrated_threshold"),
        fn: num(at, "fn", "at_calibrated_threshold"),
        tpr: num(at, "tpr", "at_calibrated_threshold"),
        fpr: num(at, "fpr", "at_calibrated_threshold"),
        precision: num(at, "precision", "at_calibrated_threshold"),
        f1: num(at, "f1", "at_calibrated_threshold"),
        threshold: num(at, "threshold", "at_calibrated_threshold"),
      },
      false_alarm_on_normal_test_images: {
        false_alarms: num(fa, "false_alarms", "false_alarm"),
        n_normal: num(fa, "n_normal", "false_alarm"),
        fpr: num(fa, "fpr", "false_alarm"),
        alpha: num(fa, "alpha", "false_alarm"),
        bound_for_new_normal_images: num(fa, "bound_for_new_normal_images", "false_alarm"),
      },
      per_defect_recall: Object.entries(recalls).map(([type, v]) => {
        const r = rec(v, `per_defect_recall.${type}`);
        return {
          type,
          detected: num(r, "detected", type),
          n: num(r, "n", type),
          recall: num(r, "recall", type),
        };
      }),
    },
    pixel: {
      auroc: num(pixel, "auroc", "pixel"),
      aupro_fpr_0_3: num(pixel, "aupro_fpr_0.3", "pixel"),
      n_images: num(pixel, "n_images", "pixel"),
      resolution: str(pixel, "resolution", "pixel"),
    },
  };
}
