// Makes src/generated/evidence.json: a small snapshot of the SAVED headline evaluation, so the
// Home page can show the headline evidence without calling the API (the Home page makes no API
// calls). Nothing is calculated here: every value is copied from metrics.json. A test fails when
// the committed snapshot differs from what this script would write now.
//
//   npm run gen:evidence        (also: make evidence)
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const METRICS_RELATIVE = "evaluation/wide_resnet50_2_tv_in1k/v1.0.0/bottle/metrics.json";
export const METRICS_PATH = fileURLToPath(new URL(`../../${METRICS_RELATIVE}`, import.meta.url));
export const SNAPSHOT_PATH = fileURLToPath(
  new URL("../src/generated/evidence.json", import.meta.url),
);

/** @param {string} metricsText the text of metrics.json */
export function buildEvidence(metricsText) {
  const m = JSON.parse(metricsText);
  const image = m.results.image;
  const at = image.at_calibrated_threshold;
  const alarms = image.false_alarm_on_normal_test_images;
  return {
    source: METRICS_RELATIVE,
    source_sha256: createHash("sha256").update(metricsText).digest("hex"),
    category: m.category,
    version: m.version,
    backbone: m.model.backbone,
    n_anomalous: image.n_anomalous,
    n_normal: image.n_normal,
    defects_flagged: at.tp,
    defects_missed: at.fn,
    false_alarms: alarms.false_alarms,
    false_alarm_rate: alarms.fpr,
    target_rate: alarms.alpha,
  };
}

export function snapshotText(metricsText) {
  return `${JSON.stringify(buildEvidence(metricsText), null, 2)}\n`;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  writeFileSync(SNAPSHOT_PATH, snapshotText(readFileSync(METRICS_PATH, "utf8")));
  console.log(`wrote ${SNAPSHOT_PATH}`);
}
