import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { BASE } from "../test/handlers";
import { envelope, inspection, pngFile } from "../test/fixtures";
import { server } from "../test/server";
import { getCategories, inspectImage } from "./client";
import { ApiError } from "./errors";

async function failure(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error("expected the call to fail");
}

describe("inspectImage", () => {
  it("sends the category and the file as multipart form data and returns the result", async () => {
    let seen: { category: unknown; file: unknown } | null = null;
    server.use(
      http.post(`${BASE}/inspections`, async ({ request }) => {
        const form = await request.formData();
        seen = { category: form.get("category"), file: form.get("file") };
        return HttpResponse.json(inspection({ prediction: "anomalous" }));
      }),
    );
    const result = await inspectImage("bottle", pngFile("a.png"));
    expect(result.prediction).toBe("anomalous");
    expect(seen).not.toBeNull();
    const form = seen as unknown as { category: unknown; file: File };
    expect(form.category).toBe("bottle");
    // the test environment mixes jsdom's and Node's File classes, so only the bytes and the type are
    // checked here (the server ignores the file name anyway)
    expect(form.file.size).toBe(64);
    expect(form.file.type).toBe("image/png");
  });

  it.each([
    [415, "UNSUPPORTED_MEDIA_TYPE"],
    [413, "FILE_TOO_LARGE"],
    [422, "INVALID_IMAGE"],
    [503, "BUSY"],
    [500, "INFERENCE_FAILED"],
  ])(
    "turns the documented error envelope with status %i into an ApiError",
    async (status, code) => {
      server.use(
        http.post(`${BASE}/inspections`, () =>
          HttpResponse.json(envelope(code, "server words", { max_mb: 10 }, "rid-9"), { status }),
        ),
      );
      const error = await failure(inspectImage("bottle", pngFile()));
      expect(error.code).toBe(code);
      expect(error.status).toBe(status);
      expect(error.requestId).toBe("rid-9");
      expect(error.details).toEqual({ max_mb: 10 });
    },
  );

  it("reads the Retry-After header of a 429", async () => {
    server.use(
      http.post(`${BASE}/inspections`, () =>
        HttpResponse.json(envelope("RATE_LIMITED", "slow down"), {
          status: 429,
          headers: { "Retry-After": "60" },
        }),
      ),
    );
    const error = await failure(inspectImage("bottle", pngFile()));
    expect(error.code).toBe("RATE_LIMITED");
    expect(error.retryAfterSeconds).toBe(60);
  });

  it("copes with an error that is not the envelope (for example a proxy's HTML page)", async () => {
    server.use(
      http.post(
        `${BASE}/inspections`,
        () =>
          new HttpResponse("<html>Bad gateway</html>", {
            status: 502,
            headers: { "Content-Type": "text/html", "X-Request-ID": "proxy-1" },
          }),
      ),
    );
    const error = await failure(inspectImage("bottle", pngFile()));
    expect(error.code).toBe("HTTP_ERROR");
    expect(error.status).toBe(502);
    expect(error.requestId).toBe("proxy-1");
  });

  it("reports an unreachable server as NETWORK", async () => {
    server.use(http.post(`${BASE}/inspections`, () => HttpResponse.error()));
    const error = await failure(inspectImage("bottle", pngFile()));
    expect(error.code).toBe("NETWORK");
    expect(error.status).toBeNull();
  });

  it("rejects a 200 answer that is not the documented shape", async () => {
    server.use(http.post(`${BASE}/inspections`, () => HttpResponse.json({ hello: "world" })));
    const error = await failure(inspectImage("bottle", pngFile()));
    expect(error.code).toBe("BAD_RESPONSE");
  });

  it("reports a cancelled request as ABORTED", async () => {
    server.use(
      http.post(`${BASE}/inspections`, async () => {
        await new Promise((resolve) => setTimeout(resolve, 200));
        return HttpResponse.json(inspection());
      }),
    );
    const controller = new AbortController();
    const pending = inspectImage("bottle", pngFile(), controller.signal);
    controller.abort();
    expect((await failure(pending)).code).toBe("ABORTED");
  });
});

describe("getCategories", () => {
  it("returns the list", async () => {
    const list = await getCategories();
    expect(list.map((c) => c.id)).toEqual(["bottle"]);
  });

  it("keeps the 503 code and Retry-After while the model is loading", async () => {
    server.use(
      http.get(`${BASE}/categories`, () =>
        HttpResponse.json(envelope("MODEL_NOT_READY", "later"), {
          status: 503,
          headers: { "Retry-After": "10" },
        }),
      ),
    );
    const error = await failure(getCategories());
    expect(error.code).toBe("MODEL_NOT_READY");
    expect(error.retryAfterSeconds).toBe(10);
  });
});
