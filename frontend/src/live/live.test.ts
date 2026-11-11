// @vitest-environment node
// A check of the real client code against a REAL running backend. It is skipped in `npm test`.
// Run it with the backend up and an image of your own (never a dataset image):
//   LIVE_IMAGE=/path/to/synthetic.png npm run test:live
import { readFile } from "node:fs/promises";
import { http, passthrough } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import { getCategories, inspectImage } from "../api/client";
import { ApiError } from "../api/errors";
import { server } from "../test/server";

const imagePath = process.env["LIVE_IMAGE"];

describe.skipIf(!imagePath)("the client against the real backend", () => {
  // registered before every test, because the shared setup resets the mock handlers after each one
  beforeEach(() => {
    // the mock server must let these requests through to the real backend
    server.use(http.all("*", () => passthrough()));
  });

  it("lists the categories", async () => {
    const categories = await getCategories();
    expect(categories.map((c) => c.id)).toEqual(["bottle"]);
    expect(categories[0]?.calib_n).toBeGreaterThan(0);
  });

  it("inspects a synthetic image and the answer has the documented shape", async () => {
    const bytes = await readFile(imagePath as string);
    const file = new File([bytes], "synthetic.png", { type: "image/png" });
    const result = await inspectImage("bottle", file);
    console.log(
      JSON.stringify({
        prediction: result.prediction,
        score: result.anomaly_score,
        threshold: result.threshold,
        rank: result.calibration_rank,
        area: result.highlighted_area_fraction,
        bbox: result.region_bbox,
        timing_ms: result.timing_ms,
        original: [result.images.original.width, result.images.original.height],
        png_data_urls_bytes: [
          result.images.original.url.length,
          result.images.heatmap.url.length,
          result.images.overlay.url.length,
        ],
      }),
    );
    expect(["normal", "anomalous"]).toContain(result.prediction);
    // the server's verdict follows its own stated rule: anomalous only if score > threshold
    expect(result.prediction === "anomalous").toBe(result.anomaly_score > result.threshold);
    for (const image of [result.images.original, result.images.heatmap, result.images.overlay]) {
      expect(image.url.startsWith("data:image/png;base64,")).toBe(true);
    }
    expect(result.persisted).toBe(false);
  });

  it("gets the real error envelope for a text file sent as an image", async () => {
    const file = new File(["not an image"], "x.png", { type: "image/png" });
    const error = await inspectImage("bottle", file).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("UNSUPPORTED_MEDIA_TYPE");
    expect((error as ApiError).status).toBe(415);
    expect((error as ApiError).requestId).toBeTruthy();
  });

  it("gets UNSUPPORTED_CATEGORY for a category that is not served", async () => {
    const bytes = await readFile(imagePath as string);
    const file = new File([bytes], "synthetic.png", { type: "image/png" });
    const error = await inspectImage("carpet", file).catch((e: unknown) => e);
    expect((error as ApiError).code).toBe("UNSUPPORTED_CATEGORY");
  });
});
