import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import App from "./App";
import { BASE } from "./test/handlers";
import { metricsResponse, modelResponse } from "./test/modelFixtures";
import { server } from "./test/server";

// Code-level keyboard checks for the four pages. jsdom does not draw anything, so these cover
// order, reachability and structure only; the real focus ring, the file chooser and the layout need
// a manual check in a real browser (README checklist).

const PAGES: [string, string][] = [
  ["/", "Home"],
  ["/inspect", "Demo"],
  ["/model", "Results"],
  ["/about", "About"],
];

/** Renders the page and waits until its own content (not just the frame) is there. */
async function renderAt(path: string) {
  server.use(
    http.get(`${BASE}/model`, () => HttpResponse.json(modelResponse() as object)),
    http.get(`${BASE}/metrics/bottle`, () => HttpResponse.json(metricsResponse() as object)),
  );
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
  const ready: Record<string, string> = {
    "/": "What is in this project",
    "/inspect": "Inspect an image",
    "/model": "What the test showed",
    "/about": "The dataset and its licence",
  };
  await screen.findByRole("heading", { name: ready[path] });
}

describe.each(PAGES)("keyboard use of %s (%s)", (path) => {
  it("starts with the skip link, then the four navigation links, in that order", async () => {
    const user = userEvent.setup();
    await renderAt(path);
    await user.tab();
    expect(screen.getByRole("link", { name: /skip to the main content/i })).toHaveFocus();
    for (const name of ["Home", "Demo", "Results", "About"]) {
      await user.tab();
      expect(
        within(screen.getByRole("navigation", { name: "Main" })).getByRole("link", { name }),
      ).toHaveFocus();
    }
  });

  it("uses no positive tab index and keeps nothing interactive out of the tab order", async () => {
    await renderAt(path);
    const all = Array.from(document.querySelectorAll<HTMLElement>("[tabindex]"));
    for (const el of all) {
      const value = Number(el.getAttribute("tabindex"));
      expect(value, `${el.tagName} tabindex`).toBeLessThanOrEqual(0);
      // the only element allowed to be programmatically focusable but outside the tab order is
      // the main region (focus moves there after a navigation)
      if (value < 0) expect(el.id).toBe("main");
    }
  });
});

describe("the expandable sections", () => {
  it.each(["/model", "/about"])(
    "on %s are native details, closed by default, with no custom tab index",
    async (path) => {
      await renderAt(path);
      const summaries = Array.from(document.querySelectorAll("details > summary"));
      expect(summaries.length).toBeGreaterThan(0);
      for (const summary of summaries) {
        expect(summary).not.toHaveAttribute("tabindex");
        expect(summary.parentElement).not.toHaveAttribute("open");
      }
    },
  );
});

describe("long text that could overflow a phone screen", () => {
  it("lets the long DOI link on About wrap", async () => {
    await renderAt("/about");
    const link = await screen.findByRole("link", { name: /doi\.org\/10\.1007/ });
    expect(link.className).toContain("break-words");
  });
});
