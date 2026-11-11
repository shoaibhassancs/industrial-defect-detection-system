// @vitest-environment node
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { METRICS_PATH, snapshotText } from "../../scripts/gen-evidence.mjs";
import evidence from "./evidence.json";

describe("the Home page's evidence snapshot", () => {
  it("is exactly what the generator makes from the saved metrics.json (not stale)", () => {
    const committed = readFileSync(new URL("./evidence.json", import.meta.url), "utf8");
    expect(
      committed,
      "src/generated/evidence.json is out of date: run `make evidence` (npm run gen:evidence)",
    ).toBe(snapshotText(readFileSync(METRICS_PATH, "utf8")));
  });

  it("holds the headline values the Home page states, copied and not calculated", () => {
    const saved = JSON.parse(readFileSync(METRICS_PATH, "utf8")).results.image;
    expect(evidence.defects_flagged).toBe(saved.at_calibrated_threshold.tp);
    expect(evidence.defects_missed).toBe(saved.at_calibrated_threshold.fn);
    expect(evidence.false_alarms).toBe(saved.false_alarm_on_normal_test_images.false_alarms);
    expect(evidence.false_alarm_rate).toBe(saved.false_alarm_on_normal_test_images.fpr);
    expect(evidence.target_rate).toBe(saved.false_alarm_on_normal_test_images.alpha);
    expect([evidence.defects_flagged, evidence.n_anomalous]).toEqual([63, 63]);
    expect([evidence.false_alarms, evidence.n_normal, evidence.false_alarm_rate]).toEqual([3, 20, 0.15]);
  });

  it("records which file it came from and that file's hash", () => {
    const raw = readFileSync(METRICS_PATH, "utf8");
    expect(evidence.source).toMatch(/wide_resnet50_2_tv_in1k\/v1\.0\.0\/bottle\/metrics\.json$/);
    expect(evidence.source_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(snapshotText(raw)).toContain(evidence.source_sha256);
  });
});
