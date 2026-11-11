import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import App from "./App";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

const noContrast = { rules: { "color-contrast": { enabled: false } } };

describe("the page frame and navigation", () => {
  it("has the landmarks, the skip link, the licence footer and a main navigation", () => {
    renderAt("/");
    expect(screen.getByRole("heading", { level: 1, name: "DefectLens" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /skip to the main content/i })).toHaveAttribute(
      "href",
      "#main",
    );
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toHaveTextContent(/CC BY-NC-SA 4\.0/);
    expect(screen.getByRole("contentinfo")).toHaveTextContent(/not quality certifications/i);
    const nav = screen.getByRole("navigation", { name: "Main" });
    expect(
      within(nav)
        .getAllByRole("link")
        .map((l) => l.textContent),
    ).toEqual(["Home", "Demo", "Results", "About"]);
  });

  it("has a one-line footer with the licence note, the demo-not-certification message and a source-code link", () => {
    renderAt("/");
    const footer = screen.getByRole("contentinfo");
    expect(footer.querySelectorAll("p")).toHaveLength(1);
    expect(footer).toHaveTextContent(/CC BY-NC-SA 4\.0, noncommercial use/);
    expect(footer).toHaveTextContent(/demo decisions, not quality certifications/);
    const code = within(footer).getByRole("link", { name: /source code/i });
    expect(code).toHaveAttribute(
      "href",
      expect.stringMatching(/^https:\/\/github\.com\/[^/]+\/[^/]+$/),
    );
    expect(code).toHaveAttribute("target", "_blank");
    expect(code).toHaveAttribute("rel", "noopener noreferrer");
  });

  it.each([
    ["/", "Home", /unsupervised defect detection/i],
    ["/inspect", "Demo", /inspect an image/i],
    ["/model", "Results", /^Results$/],
    ["/about", "About", /about this demo/i],
  ])(
    "shows the right page at %s and marks %s as the current page",
    async (path, label, heading) => {
      renderAt(path);
      expect(await screen.findByRole("heading", { level: 2, name: heading })).toBeInTheDocument();
      const nav = screen.getByRole("navigation", { name: "Main" });
      const current = within(nav)
        .getAllByRole("link")
        .filter((l) => l.getAttribute("aria-current") === "page");
      expect(current.map((l) => l.textContent)).toEqual([label]);
    },
  );

  it("moves between the pages with the links, updates the tab title, and moves focus to the content", async () => {
    const user = userEvent.setup();
    renderAt("/");
    expect(document.title).toBe("DefectLens");
    const nav = screen.getByRole("navigation", { name: "Main" });

    await user.click(within(nav).getByRole("link", { name: "About" }));
    expect(await screen.findByRole("heading", { name: /about this demo/i })).toBeInTheDocument();
    expect(document.title).toBe("About · DefectLens");
    await waitFor(() => expect(screen.getByRole("main")).toHaveFocus());

    await user.click(within(nav).getByRole("link", { name: "Demo" }));
    expect(await screen.findByRole("heading", { name: /inspect an image/i })).toBeInTheDocument();
    expect(document.title).toBe("Demo · DefectLens");

    await user.click(within(nav).getByRole("link", { name: "Home" }));
    expect(
      await screen.findByRole("heading", { name: /unsupervised defect detection/i }),
    ).toBeInTheDocument();
    expect(document.title).toBe("DefectLens");
  });

  it("the Home page's button goes to the Inspect page", async () => {
    const user = userEvent.setup();
    renderAt("/");
    await user.click(screen.getByRole("link", { name: "Try the demo" }));
    expect(await screen.findByLabelText(/category/i)).toBeInTheDocument();
  });

  it("shows a not-found page with a way home for an unknown address", async () => {
    renderAt("/nothing-here");
    expect(await screen.findByRole("heading", { name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "home page" })).toHaveAttribute("href", "/");
  });

  it.each(["/", "/inspect", "/about", "/nothing-here"])(
    "has no automatically detectable accessibility problems at %s",
    async (path) => {
      renderAt(path);
      await screen.findByRole("main");
      if (path === "/inspect") await screen.findByLabelText(/category/i);
      const result = await axe.run(document.body, noContrast);
      expect(result.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
    },
  );

  it("contains no image on the Home, About and not-found pages (no dataset images)", () => {
    for (const path of ["/", "/about", "/nothing-here"]) {
      const { unmount } = renderAt(path);
      expect(document.querySelectorAll("img")).toHaveLength(0);
      unmount();
    }
  });
});
