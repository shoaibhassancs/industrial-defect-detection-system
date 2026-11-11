import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { BASE } from "../../test/handlers";
import { CATEGORY, envelope, inspection, jpegFile, pngFile, textFile } from "../../test/fixtures";
import { server } from "../../test/server";
import { InspectPage } from "./InspectPage";

async function chooseAndInspect(file: File) {
  const user = userEvent.setup();
  render(<InspectPage slowAfterMs={60} startupRetryMs={30} />);
  const input = await screen.findByLabelText(/^image/i);
  await user.upload(input, file);
  return { user, input };
}

async function clickInspect(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: /inspect image/i }));
}

describe("the Inspect flow", () => {
  it("loads the categories and keeps the button off until a valid image is chosen", async () => {
    render(<InspectPage />);
    expect(await screen.findByLabelText(/category/i)).toHaveValue(CATEGORY.id);
    expect(screen.getByRole("button", { name: /inspect image/i })).toBeDisabled();
  });

  it("shows a normal result: decision, score next to threshold, and the overlay first", async () => {
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    const heading = await screen.findByRole("heading", { name: "Result" });
    await waitFor(() => expect(heading).toHaveFocus());
    expect(screen.getByText("Demo decision: Normal")).toBeInTheDocument();
    const section = heading.closest("section")!;
    // the score and the threshold are in the visible summary, not behind a collapsed section
    const score = within(section).getAllByText("0.2000")[0]!;
    const threshold = within(section).getAllByText("0.4321")[0]!;
    expect(score.closest("details")).toBeNull();
    expect(threshold.closest("details")).toBeNull();
    // the rank sentence is in the details
    expect(section.textContent).toContain("Score higher than 7 of 42 calibration normal images.");
    // the overlay is the image shown first
    expect(within(section).getAllByRole("img")).toHaveLength(1);
    expect(screen.getByRole("img", { name: /^Overlay:/ })).toBeInTheDocument();
  });

  it("shows the heatmap and the original in tabs, with the overlay selected by default", async () => {
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    const tabs = await screen.findAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Overlay", "Heatmap", "Original"]);
    expect(tabs.map((t) => t.getAttribute("aria-selected"))).toEqual(["true", "false", "false"]);
    expect(screen.getByRole("tabpanel")).toHaveAccessibleName("Overlay");
    await user.click(screen.getByRole("tab", { name: "Heatmap" }));
    expect(screen.getByRole("tab", { name: "Heatmap" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("img", { name: /^Heatmap:/ })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /^Overlay:/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Original" }));
    expect(screen.getByRole("img", { name: /^Original:/ })).toBeInTheDocument();
  });

  it("moves between the tabs with the arrow keys, Home and End", async () => {
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    const overlay = await screen.findByRole("tab", { name: "Overlay" });
    overlay.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Heatmap" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "Heatmap" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "Original" })).toHaveFocus();
    await user.keyboard("{ArrowRight}"); // wraps round
    expect(screen.getByRole("tab", { name: "Overlay" })).toHaveFocus();
    await user.keyboard("{ArrowLeft}"); // wraps the other way
    expect(screen.getByRole("tab", { name: "Original" })).toHaveFocus();
    await user.keyboard("{Home}");
    expect(screen.getByRole("tab", { name: "Overlay" })).toHaveFocus();
    // only the selected tab is in the tab order
    expect(screen.getByRole("tab", { name: "Heatmap" })).toHaveAttribute("tabindex", "-1");
  });

  it("keeps the decision, the demo-not-certification message and the limits visible, and the explanations in closed sections", async () => {
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    const section = (await screen.findByRole("heading", { name: "Result" })).closest("section")!;
    for (const phrase of [
      /Demo decision: Normal/,
      /not a quality certification/,
      /not a probability or a confidence/,
      /cannot reliably tell that an image is out of scope/,
      /only meaningful for bottle images/i,
    ]) {
      const hit = within(section).getAllByText(phrase)[0]!;
      expect(hit.closest("details"), String(phrase)).toBeNull();
    }
    const titles = [...section.querySelectorAll("details > summary")].map((x) => x.textContent);
    expect(titles).toEqual([
      "Show details",
      "How to read these results",
      "How this decision was made",
    ]);
    for (const d of section.querySelectorAll("details")) expect(d).not.toHaveAttribute("open");
    // every explanation is still in the page, inside the first section
    const details = section.querySelector("details")!;
    for (const phrase of [
      /higher means more unusual/,
      /Threshold \(the cutoff\)/,
      /not a measure of confidence/,
      /A display aid/,
      /not a measured defect size/,
    ]) {
      expect(details.textContent).toMatch(phrase);
    }
  });

  it("shows the server's verdict as is: an anomalous result, with the region", async () => {
    server.use(
      http.post(`${BASE}/inspections`, () =>
        HttpResponse.json(
          inspection({
            prediction: "anomalous",
            anomaly_score: 0.9,
            region_bbox: [10, 20, 30, 40],
            calibration_rank: { score_higher_than: 42, of_calibration_normals: 42 },
          }),
        ),
      ),
    );
    const { user } = await chooseAndInspect(jpegFile());
    await clickInspect(user);
    expect(await screen.findByText("Demo decision: Anomalous")).toBeInTheDocument();
    expect(screen.getByText(/x 10, y 20, 30 × 40 px/)).toBeInTheDocument();
  });

  it("never calls the verdict pass or fail, and says the score is not a probability", async () => {
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    const panel = (await screen.findByRole("heading", { name: "Result" })).closest("section")!;
    expect(panel.textContent).not.toMatch(/\b(pass|passed|fail|failed|defective)\b/i);
    expect(within(panel).getAllByText(/not a probability or a confidence/i).length).toBeGreaterThan(
      0,
    );
    expect(within(panel).getAllByText(/not a quality certification/i).length).toBeGreaterThan(0);
  });

  it("explains each number in plain words next to it", async () => {
    server.use(
      http.post(`${BASE}/inspections`, () =>
        HttpResponse.json(
          inspection({ calibration_rank: { score_higher_than: 3, of_calibration_normals: 17 } }),
        ),
      ),
    );
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    const panel = (await screen.findByRole("heading", { name: "Result" })).closest("section")!;
    const text = panel.textContent ?? "";
    // score: higher means more unusual, and not a probability
    expect(text).toMatch(/higher means more unusual/i);
    expect(text).toMatch(/not a probability or a confidence/i);
    // threshold: a cutoff; "Anomalous" only when the score is above it
    expect(text).toMatch(/Threshold \(the cutoff\)/);
    expect(text).toMatch(/says “Anomalous” only when the score is above this cutoff/);
    // rank: a comparison with the calibration images, not confidence (and the count comes from
    // the server's answer, not from the page)
    expect(text).toMatch(/set using 17 normal calibration images/);
    expect(text).toMatch(/how many of the 17 normal calibration images scored lower/);
    expect(text).toMatch(/not a measure of confidence/);
    // highlighted area: a display aid, not a defect size
    expect(text).toMatch(/A display aid/);
    expect(text).toMatch(/not a measured defect size/);
    expect(text).not.toMatch(/\b42\b/); // nothing is hard-coded to this model's 42 images
  });

  it("describes the heatmap colours as lower and higher scores, not as 'ordinary'", async () => {
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    await user.click(await screen.findByRole("tab", { name: "Heatmap" }));
    const panel = (await screen.findByRole("heading", { name: "Result" })).closest("section")!;
    expect(panel.textContent).toMatch(
      /lower \(darker colours\) or higher \(brighter orange and yellow\)/,
    );
    expect(panel.textContent).not.toMatch(/ordinary|meaningless/i);
  });

  it("opens a 'How to read these results' section that keeps performance claims out", async () => {
    const user = userEvent.setup();
    const { user: inspecting } = await chooseAndInspect(pngFile());
    await clickInspect(inspecting);
    const summary = await screen.findByText("How to read these results");
    await user.click(summary);
    const details = summary.closest("details")!;
    expect(details).toHaveAttribute("open");
    expect(details.textContent).toMatch(/measured on a separate, formal test set/i);
    expect(details.textContent).toMatch(/server makes that decision; this page only shows it/i);
  });

  it("warns, before any upload, that the model is designed for MVTec AD bottle-like images", async () => {
    render(<InspectPage />);
    const warning = await screen.findByText(/designed for images like the MVTec AD bottle images/i);
    expect(warning).toHaveTextContent(/unreliable on other objects or image types/i);
    expect(warning).toHaveTextContent(/cannot reliably tell that an image is out of scope/i);
    expect(document.body.textContent).not.toMatch(/meaningless/i);
  });

  it("refuses a non-image in the browser and sends nothing", async () => {
    let calls = 0;
    server.use(
      http.post(`${BASE}/inspections`, () => {
        calls += 1;
        return HttpResponse.json(inspection());
      }),
    );
    await chooseAndInspect(textFile());
    expect(await screen.findByText(/choose a jpeg or png image/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /inspect image/i })).toBeDisabled();
    expect(calls).toBe(0);
  });

  it.each([
    [415, "UNSUPPORTED_MEDIA_TYPE", /not a jpeg or png/i],
    [413, "FILE_TOO_LARGE", /too large/i],
    [422, "INVALID_IMAGE", /could not be read/i],
    [503, "BUSY", /busy/i],
    [500, "INFERENCE_FAILED", /could not process/i],
  ])(
    "explains a %i %s answer in plain words and shows the request ID",
    async (status, code, words) => {
      server.use(
        http.post(`${BASE}/inspections`, () =>
          HttpResponse.json(envelope(code, "server words", { max_mb: 10 }, "rid-42"), { status }),
        ),
      );
      const { user } = await chooseAndInspect(pngFile());
      await clickInspect(user);
      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent(words);
      expect(alert).toHaveTextContent("rid-42");
      expect(screen.queryByRole("heading", { name: "Result" })).not.toBeInTheDocument();
    },
  );

  it("tells the visitor how long to wait after a 429", async () => {
    server.use(
      http.post(`${BASE}/inspections`, () =>
        HttpResponse.json(envelope("RATE_LIMITED", "x"), {
          status: 429,
          headers: { "Retry-After": "45" },
        }),
      ),
    );
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    expect(await screen.findByRole("alert")).toHaveTextContent(/about 45 seconds/i);
  });

  it("names the address it tried when the service cannot be reached", async () => {
    server.use(http.post(`${BASE}/inspections`, () => HttpResponse.error()));
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("could not be reached");
    expect(alert).toHaveTextContent("http://localhost:8000");
    expect(alert).toHaveTextContent(/CORS/);
  });

  it("shows a waiting message, then a 'waking up' message, and can be cancelled", async () => {
    server.use(
      http.post(`${BASE}/inspections`, async () => {
        await new Promise((resolve) => setTimeout(resolve, 500));
        return HttpResponse.json(inspection());
      }),
    );
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    expect(await screen.findByText("Inspecting the image…")).toBeInTheDocument();
    expect(await screen.findByText(/may be waking up/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /cancel/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /inspect image/i })).toBeEnabled(),
    );
    expect(screen.queryByRole("heading", { name: "Result" })).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("waits for a model that is still starting, then shows the form by itself", async () => {
    let calls = 0;
    server.use(
      http.get(`${BASE}/categories`, () => {
        calls += 1;
        return calls < 3
          ? HttpResponse.json(envelope("MODEL_NOT_READY", "later"), { status: 503 })
          : HttpResponse.json([CATEGORY]);
      }),
    );
    render(<InspectPage startupRetryMs={20} />);
    expect(await screen.findByText(/model is starting up/i)).toBeInTheDocument();
    expect(await screen.findByLabelText(/category/i)).toBeInTheDocument();
    expect(calls).toBe(3);
  });

  it("shows a retry button when the category list cannot be loaded, and it works", async () => {
    let up = false;
    server.use(
      http.get(`${BASE}/categories`, () =>
        up ? HttpResponse.json([CATEGORY]) : HttpResponse.error(),
      ),
    );
    const user = userEvent.setup();
    render(<InspectPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be reached/i);
    up = true;
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(await screen.findByLabelText(/category/i)).toBeInTheDocument();
  });

  it("has no automatically detectable accessibility problems (result and error states)", async () => {
    const { user } = await chooseAndInspect(pngFile());
    await clickInspect(user);
    await screen.findByRole("heading", { name: "Result" });
    // colour contrast cannot be computed in jsdom, so that rule is switched off here and is
    // listed as not checked in the project notes
    const config = { rules: { "color-contrast": { enabled: false } } };
    const ok = await axe.run(document.body, config);
    expect(ok.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
  });
});
