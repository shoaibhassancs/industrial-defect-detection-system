import type { CategoryInfo, InspectionResponse } from "../api/types";

/** A 1 x 1 PNG, so tests never need a real image. */
export const TINY_PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

export const CATEGORY: CategoryInfo = {
  id: "bottle",
  display_name: "Bottle",
  kind: "object",
  threshold: 0.4321,
  alpha: 0.05,
  calib_n: 42,
  metrics_summary: null,
};

/** Typed against the generated contract: a change in the backend's description that breaks this
 * shape stops `npm run typecheck`. Values are made up for tests; they are not model results. */
export function inspection(overrides: Partial<InspectionResponse> = {}): InspectionResponse {
  return {
    id: "00000000-0000-4000-8000-000000000001",
    created_at: "2026-01-01T00:00:00Z",
    category: "bottle",
    model: { name: "test-model", version: "0.0.0-test" },
    prediction: "normal",
    decision_rule: "anomalous if anomaly_score > threshold; threshold = test rule text",
    anomaly_score: 0.2,
    threshold: 0.4321,
    calibration_rank: { score_higher_than: 7, of_calibration_normals: 42 },
    highlighted_area_fraction: 0.0123,
    region_bbox: null,
    images: {
      original: { url: TINY_PNG_DATA_URL, width: 1, height: 1 },
      heatmap: { url: TINY_PNG_DATA_URL },
      overlay: { url: TINY_PNG_DATA_URL },
    },
    timing_ms: { inference: 123.4, total: 150.2 },
    persisted: false,
    warnings: [],
    limitation: "Results are only meaningful for bottle images captured like the MVTec AD dataset.",
    disclaimer: "Demo decision based only on this model and category. Not a quality certification.",
    ...overrides,
  };
}

export function envelope(
  code: string,
  message: string,
  details: Record<string, unknown> = {},
  requestId = "req-test-123",
) {
  return { error: { code, message, details, request_id: requestId } };
}

/** Synthetic images made in the test itself (no dataset files). */
export function pngFile(name = "synthetic.png", bytes = 64): File {
  const head = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  const body = new Uint8Array(Math.max(bytes, head.length));
  body.set(head);
  return new File([body], name, { type: "image/png" });
}

export function jpegFile(name = "synthetic.jpg"): File {
  const body = new Uint8Array(64);
  body.set([0xff, 0xd8, 0xff, 0xe0]);
  return new File([body], name, { type: "image/jpeg" });
}

export function textFile(name = "notes.png"): File {
  return new File(["this is not an image"], name, { type: "image/png" });
}
