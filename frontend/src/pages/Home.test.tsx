import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import evidence from "../generated/evidence.json";

afterEach(() => vi.restoreAllMocks());

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("the Home page makes no API call", () => {
  it("however long it stays open", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const xhrSpy = vi.spyOn(XMLHttpRequest.prototype, "open");
    renderAt("/");
    await screen.findByRole("heading", { name: /unsupervised defect detection/i });
    await new Promise((resolve) => setTimeout(resolve, 150)); // time for any delayed request
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(xhrSpy).not.toHaveBeenCalled();
  });

  it("nor when going from Home to About and back", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const user = userEvent.setup();
    renderAt("/");
    const nav = screen.getByRole("navigation", { name: "Main" });
    await user.click(within(nav).getByRole("link", { name: "About" }));
    await screen.findByRole("heading", { name: /about this demo/i });
    await user.click(within(nav).getByRole("link", { name: "Home" }));
    await screen.findByRole("heading", { name: /unsupervised defect detection/i });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("control: the Demo page does call the API, so the spy above can see a call", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    renderAt("/inspect");
    await screen.findByLabelText(/category/i);
    expect(fetchSpy).toHaveBeenCalled();
    expect(String(fetchSpy.mock.calls[0]?.[0])).toContain("/api/v1/categories");
  });
});

describe("the Home page's evidence strip", () => {
  it("shows the headline evidence from the saved snapshot: 63 of 63 and 3 of 20 (15%)", () => {
    renderAt("/");
    const strip = screen.getByRole("heading", { name: /headline evidence/i }).closest("section")!;
    const text = strip.textContent ?? "";
    expect(text).toContain(`${evidence.defects_flagged} of ${evidence.n_anomalous}`);
    expect(text).toContain("63 of 63");
    expect(text).toContain("3 of 20 (15%)");
    expect(text).toMatch(/higher than the 5% the threshold aimed for/);
  });

  it("keeps the small-test-set caveat and the 'both models evaluated' statement visible", () => {
    renderAt("/");
    const strip = screen.getByRole("heading", { name: /headline evidence/i }).closest("section")!;
    expect(strip.textContent).toMatch(/test set is small/i);
    expect(strip.textContent).toMatch(/coarse estimate/i);
    expect(strip.textContent).toMatch(/not a promise for other cases/i);
    expect(strip.textContent).toMatch(/Two models were evaluated on the same images/);
    expect(strip.textContent).toContain(evidence.backbone);
    expect(within(strip).getByRole("link", { name: /docs\/evaluation\.md/ })).toHaveAttribute(
      "href",
      expect.stringMatching(/\/docs\/evaluation\.md$/),
    );
  });

  it("keeps the evidence outside any collapsed section", () => {
    renderAt("/");
    for (const phrase of [
      /63 of 63/,
      /3 of 20 \(15%\)/,
      /demo decisions,\s+not quality certifications/i,
    ]) {
      for (const hit of screen.getAllByText(phrase)) expect(hit.closest("details")).toBeNull();
    }
  });
});

describe("the rest of the Home page", () => {
  it("states what it is, the 'demo, not certification' message and the careful scope wording", () => {
    renderAt("/");
    const text = screen.getByRole("main").textContent ?? "";
    expect(text).toMatch(/educational demo/i);
    expect(text).toMatch(/demo decisions,\s+not quality certifications/i);
    expect(text).toMatch(/not a probability or a\s+confidence/i);
    expect(text).toMatch(/designed for images like the MVTec AD\s+bottle images/i);
    expect(text).toMatch(/may be unreliable on other objects or image types/i);
    expect(text).toMatch(/cannot reliably\s+tell that an image is out of scope/i);
    expect(text).not.toMatch(/\b(pass|passed|fail|failed)\b/i);
    expect(text).not.toMatch(/meaningless/i);
  });

  it("has two clear actions and links to the code and the documents", () => {
    renderAt("/");
    const main = screen.getByRole("main");
    expect(within(main).getByRole("link", { name: "Try the demo" })).toHaveAttribute(
      "href",
      "/inspect",
    );
    expect(within(main).getByRole("link", { name: "See the results" })).toHaveAttribute(
      "href",
      "/model",
    );
    const code = within(main).getByRole("link", { name: /source code/i });
    expect(code).toHaveAttribute("href", expect.stringMatching(/^https:\/\/github\.com\//));
    expect(code).toHaveAttribute("rel", "noopener noreferrer");
    expect(within(main).getByRole("link", { name: /model card/i })).toHaveAttribute(
      "href",
      expect.stringMatching(/\/docs\/model-card\.md$/),
    );
  });

  it("shows no image (no dataset images)", () => {
    renderAt("/");
    expect(document.querySelectorAll("img, svg, canvas")).toHaveLength(0);
  });
});
