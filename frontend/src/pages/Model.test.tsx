import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { getMetrics, getModelInfo } from "../api/client";
import { BASE } from "../test/handlers";
import { envelope } from "../test/fixtures";
import { metricsResponse, modelResponse, savedResults } from "../test/modelFixtures";
import { server } from "../test/server";

afterEach(() => vi.restoreAllMocks());

function useModelHandlers(metrics: unknown = metricsResponse(), model: unknown = modelResponse()) {
  server.use(
    http.get(`${BASE}/model`, () => HttpResponse.json(model as object)),
    http.get(`${BASE}/metrics/bottle`, () => HttpResponse.json(metrics as object)),
  );
}

async function renderModelPage() {
  render(
    <MemoryRouter initialEntries={["/model"]}>
      <App />
    </MemoryRouter>,
  );
  await screen.findByRole("heading", { name: "What the test showed" });
  return screen.getByRole("main");
}

// Independent writing of the saved values (the page's own helpers are not used here).
const d3 = (x: number) => x.toFixed(3);
const d4 = (x: number) => x.toFixed(4);
const pct = (x: number) => `${Number((x * 100).toFixed(2))}%`;

describe("the Model page shows the saved evaluation, and only that", () => {
  it("shows every saved number exactly as stored (real metrics.json through the page)", async () => {
    const r = savedResults();
    useModelHandlers();
    const main = await renderModelPage();
    const text = main.textContent ?? "";
    const img = r.image;
    const at = img.at_calibrated_threshold;
    const fa = img.false_alarm_on_normal_test_images;
    const expected: [string, string][] = [
      ["defective flagged", `${at.tp} of ${img.n_anomalous}`],
      ["false alarms", `${fa.false_alarms} of ${fa.n_normal} (${pct(fa.fpr)})`],
      ["alpha", pct(fa.alpha)],
      ["bound", pct(fa.bound_for_new_normal_images)],
      ["threshold", d4(at.threshold)],
      ["precision", d3(at.precision)],
      ["f1", d3(at.f1)],
      [
        "auroc",
        `${d3(img.auroc.estimate)} (${pct(img.auroc.confidence)} interval: ${d3(img.auroc.low)} to ${d3(img.auroc.high)})`,
      ],
      [
        "aupr",
        `${d3(img.aupr.estimate)} (${pct(img.aupr.confidence)} interval: ${d3(img.aupr.low)} to ${d3(img.aupr.high)})`,
      ],
      ["pixel auroc", d3(r.pixel.auroc)],
      ["aupro", d3(r.pixel["aupro_fpr_0.3"])],
      ["pixel images", `${r.pixel.n_images} test images against the defect masks`],
      [
        "total images",
        `${img.n_normal + img.n_anomalous} images: ${img.n_anomalous} defective and ${img.n_normal} normal`,
      ],
    ];
    for (const [name, value] of expected) expect(text, name).toContain(value);
    for (const [type, v] of Object.entries(img.per_defect_recall)) {
      expect(text, type).toContain(`${type.replace(/_/g, " ")}${v.detected} of ${v.n}`);
    }
    // the counts table: rows are defective / normal, columns flagged / called normal
    const counts = within(main).getByRole("table", { name: /what the demo decided/i });
    const rows = within(counts).getAllByRole("row").slice(1);
    const cells = rows.map((row) =>
      within(row)
        .getAllByRole("cell")
        .map((c) => c.textContent),
    );
    expect(cells).toEqual([
      [String(at.tp), String(at.fn)],
      [String(at.fp), String(at.tn)],
    ]);
  });

  it("keeps the owner's headline values visible: 63 of 63 and 3 of 20 (15%)", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    expect(main.textContent).toContain("63 of 63");
    expect(main.textContent).toContain("3 of 20 (15%)");
  });

  it("contains no number that did not come from the saved data or the service's own text", async () => {
    const results = savedResults();
    const model = modelResponse();
    useModelHandlers(metricsResponse(results), model);
    const main = await renderModelPage();
    const token = /\b\d+(?:\.\d+)?%?/g;
    const allowed = new Set<string>();
    const img = results.image;
    const at = img.at_calibrated_threshold;
    const fa = img.false_alarm_on_normal_test_images;
    const add = (...xs: (string | number)[]) => xs.forEach((x) => allowed.add(String(x)));
    add(at.tp, at.fp, at.tn, at.fn, img.n_normal, img.n_anomalous, img.n_normal + img.n_anomalous);
    add(
      d3(at.precision),
      d3(at.f1),
      d4(at.threshold),
      pct(fa.fpr),
      pct(fa.alpha),
      pct(fa.bound_for_new_normal_images),
    );
    for (const k of ["auroc", "aupr"] as const) {
      add(d3(img[k].estimate), d3(img[k].low), d3(img[k].high), pct(img[k].confidence));
    }
    add(d3(results.pixel.auroc), d3(results.pixel["aupro_fpr_0.3"]), results.pixel.n_images);
    for (const v of Object.values(img.per_defect_recall)) add(v.detected, v.n);
    add(model.calibration.n, model.calibration.k, d4(model.calibration.threshold));
    add(pct(model.calibration.alpha), pct(model.calibration.false_alarm_bound), model.version);
    // numbers inside the service's own sentences (resolution note, limitations, meaning)
    // the version (for example 1.0.0) is split by the number pattern into 1.0 and 0
    for (const s of [
      model.version,
      results.pixel.resolution,
      model.calibration.meaning,
      ...model.limitations,
    ]) {
      for (const m of s.match(token) ?? []) allowed.add(m);
    }
    // read the page with a space after every element, so table cells do not run together
    const copy = main.cloneNode(true) as HTMLElement;
    copy.querySelectorAll("*").forEach((el) => el.append(" "));
    const stray = (copy.textContent ?? "").match(token)?.filter((t) => !allowed.has(t)) ?? [];
    expect(stray).toEqual([]);
  });

  it("states that both models were evaluated and links to docs/evaluation.md", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    expect(main.textContent).toMatch(/Two models were evaluated on the same test images/);
    expect(main.textContent).toMatch(/headline model/);
    expect(main.textContent).toMatch(/other model \(resnet18\)/);
    const link = within(main).getByRole("link", { name: /docs\/evaluation\.md/ });
    expect(link.getAttribute("href")).toMatch(/\/docs\/evaluation\.md$/);
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("shows the false-alarm result prominently, before the detailed tables, with the small-test-set limit", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    const text = main.textContent ?? "";
    expect(text.indexOf("3 of 20 (15%)")).toBeGreaterThan(-1);
    expect(text.indexOf("3 of 20 (15%)")).toBeLessThan(text.indexOf("Image-level results: counts"));
    expect(text.indexOf("63 of 63")).toBeLessThan(text.indexOf("Image-level results: counts"));
    expect(text).toMatch(/higher than the 5% the threshold aimed for/);
    expect(text).toMatch(/The test set is small: only 20 normal images/);
    expect(text).toMatch(/coarse estimate/);
    expect(text).toMatch(/not a promise that the model finds every\s*defect elsewhere/);
    expect(text).toMatch(/does not recalculate or adjust anything/);
  });

  it("says 'not higher than' when the saved false-alarm rate is not above the target", async () => {
    const r = savedResults();
    r.image.false_alarm_on_normal_test_images.fpr = 0.04;
    useModelHandlers(metricsResponse(r));
    const main = await renderModelPage();
    expect(main.textContent).toMatch(/not higher than the 5% the threshold aimed for/);
  });

  it("keeps the key evidence and caveats outside any collapsed section", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    for (const phrase of [
      /63 of 63/,
      /3 of 20 \(15%\)/,
      /higher than the 5% the threshold aimed for/,
      /The test set is small: only 20 normal images/,
      /Two models were evaluated on the same test images/,
      /A demo decision, not a quality certification/,
      /not a probability or a confidence/,
    ]) {
      const hit = within(main).getAllByText(phrase)[0]!;
      expect(hit.closest("details"), String(phrase)).toBeNull();
    }
    expect(
      within(main)
        .getByRole("link", { name: /docs\/evaluation\.md/ })
        .closest("details"),
    ).toBeNull();
  });

  it("puts the supporting detail in collapsed sections that still hold all of it", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    const titles = [...main.querySelectorAll("details > summary")].map((x) => x.textContent);
    expect(titles).toEqual([
      "Image-level results: counts",
      "Image-level results: scores and intervals",
      "Defective images flagged, by defect type",
      "Pixel-level results: does the heatmap point at the defect?",
      "How the threshold was set",
      "Model details",
      "All limitations",
    ]);
    for (const d of main.querySelectorAll("details")) expect(d).not.toHaveAttribute("open");
  });

  it("shows the at-a-glance table with only four values, before the detail", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    const glance = within(main).getByRole("heading", { name: "At a glance" }).closest("section")!;
    const labels = within(glance)
      .getAllByRole("row")
      .map((row) => within(row).getByRole("rowheader").textContent?.split(" (")[0]);
    expect(labels).toEqual(["Precision", "F1", "AUROC", "Pixel AUROC"]);
  });

  it("adds no chart and no image (no dataset figures yet)", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    expect(main.querySelectorAll("img, svg, canvas, figure")).toHaveLength(0);
  });

  it("never says pass or fail, and keeps the score-is-not-a-probability limitation", async () => {
    useModelHandlers();
    const main = await renderModelPage();
    expect(main.textContent).not.toMatch(/\b(pass|passed|fail|failed)\b/i);
    expect(main.textContent).toMatch(/not a probability or a confidence/);
    expect(main.textContent).toMatch(/not a certified inspection system/);
  });

  it("does not call a model 'headline' unless the package says so", async () => {
    useModelHandlers(metricsResponse(), modelResponse({ headline_model: false }));
    const main = await renderModelPage();
    expect(main.textContent).not.toMatch(/Headline model wide_resnet50_2/);
    expect(main.textContent).toMatch(/Model\s+wide_resnet50_2\.tv_in1k/);
  });

  it("has no automatically detectable accessibility problems", async () => {
    useModelHandlers();
    await renderModelPage();
    const result = await axe.run(document.body, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(result.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
  });
});

describe("the Model page when the service is not ready or answers badly", () => {
  it("waits for a starting model and then shows the results by itself", async () => {
    let calls = 0;
    server.use(
      http.get(`${BASE}/categories`, () => {
        calls += 1;
        return calls < 2
          ? HttpResponse.json(envelope("MODEL_NOT_READY", "later"), {
              status: 503,
              headers: { "Retry-After": "0" },
            })
          : HttpResponse.json([
              {
                id: "bottle",
                display_name: "Bottle",
                kind: "object",
                threshold: 1,
                alpha: 0.05,
                calib_n: 42,
                metrics_summary: null,
              },
            ]);
      }),
    );
    useModelHandlers();
    render(
      <MemoryRouter initialEntries={["/model"]}>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByText(/model is starting up/i)).toBeInTheDocument();
    expect(
      await screen.findByRole("heading", { name: "What the test showed" }),
    ).toBeInTheDocument();
  });

  it("shows an error with a retry button when the service cannot be reached", async () => {
    server.use(http.get(`${BASE}/categories`, () => HttpResponse.error()));
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/model"]}>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be reached/i);
    server.resetHandlers();
    useModelHandlers();
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(
      await screen.findByRole("heading", { name: "What the test showed" }),
    ).toBeInTheDocument();
  });

  it("refuses an answer with a missing number instead of showing a wrong page", async () => {
    const broken = savedResults();
    Reflect.deleteProperty(broken.image.at_calibrated_threshold, "tp");
    useModelHandlers(metricsResponse(broken));
    render(
      <MemoryRouter initialEntries={["/model"]}>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(/unexpected way/i);
    expect(screen.queryByText(/of 63/)).not.toBeInTheDocument();
  });

  it("the parsers reject a non-number where a number is expected", async () => {
    const broken = savedResults();
    Object.assign(broken.pixel, { auroc: "0.98" });
    server.use(
      http.get(`${BASE}/metrics/bottle`, () => HttpResponse.json(metricsResponse(broken))),
    );
    await expect(getMetrics("bottle")).rejects.toMatchObject({ code: "BAD_RESPONSE" });
    server.use(http.get(`${BASE}/model`, () => HttpResponse.json({ algorithm: "x" })));
    await expect(getModelInfo()).rejects.toMatchObject({ code: "BAD_RESPONSE" });
  });
});
