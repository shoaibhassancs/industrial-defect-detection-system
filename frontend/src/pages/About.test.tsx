import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import App from "../App";

function aboutText(): { text: string; main: HTMLElement } {
  render(
    <MemoryRouter initialEntries={["/about"]}>
      <App />
    </MemoryRouter>,
  );
  const main = screen.getByRole("main");
  return { text: main.textContent ?? "", main };
}

describe("the About page", () => {
  it("gives the dataset attribution and the licence, as in DATA_LICENSE.md", () => {
    const { text, main } = aboutText();
    expect(text).toMatch(/MVTec Anomaly Detection dataset \(MVTec AD\)/);
    expect(text).toMatch(/Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger/);
    expect(text).toMatch(/MVTec Software GmbH/);
    expect(text).toMatch(/Copyright 2019 MVTec Software GmbH/);
    expect(text).toMatch(/CC BY-NC-SA 4\.0/);
    expect(text).toMatch(/prohibits commercial use/i);
    expect(text).toMatch(/non-commercial educational/i);
    const licence = main.querySelector(
      'a[href="https://creativecommons.org/licenses/by-nc-sa/4.0/"]',
    );
    expect(licence).not.toBeNull();
    expect(
      main.querySelector('a[href="https://www.mvtec.com/research-teaching/datasets/mvtec-ad"]'),
    ).not.toBeNull();
  });

  it("states what is unconfirmed instead of hiding it", () => {
    const { text } = aboutText();
    expect(text).toMatch(/According to the project owner/);
    expect(text).toMatch(/Kaggle/);
    expect(text).toMatch(/unconfirmed/);
    expect(text).toMatch(/not been compared file by file with MVTec.s official archive/);
  });

  it("describes the model files as a cautious policy, not a legal finding, with no endorsement", () => {
    const { text } = aboutText();
    expect(text).toMatch(/cautious policy, not a legal finding/);
    expect(text).toMatch(/does not imply|Nothing here implies/);
    expect(text).toMatch(/support or endorse/);
    expect(text).toMatch(/not legal advice/);
    expect(text).toMatch(/not part of this project.s repository/);
  });

  it("gives both citations, with the DOI link", () => {
    const { text, main } = aboutText();
    expect(text).toMatch(/A Comprehensive\s+Real-World Dataset for Unsupervised Anomaly Detection/);
    expect(text).toMatch(/IEEE Conference on Computer Vision and Pattern Recognition, 2019/);
    expect(text).toMatch(/International Journal of Computer Vision, 129, 1038–1059/);
    expect(
      main.querySelector('a[href="https://doi.org/10.1007/s11263-020-01400-4"]'),
    ).not.toBeNull();
  });

  it("does not claim a code licence that has not been granted", () => {
    const { text } = aboutText();
    expect(text).toMatch(/no licence file has been added yet/);
    expect(text).toMatch(/no licence is granted for the code/);
  });

  it("explains privacy without overclaiming", () => {
    const { text } = aboutText();
    expect(text).toMatch(/Nothing is stored/);
    expect(text).toMatch(/do not record the file name or the image content/);
    expect(text).toMatch(/sets no cookies/);
    expect(text).toMatch(/has not been deployed yet/);
    expect(text).toMatch(/not checked or controlled by this project/);
    expect(text).toMatch(/do not upload\s+anything private/i);
  });

  it("repeats the disclaimers and makes no performance claim", () => {
    const { text } = aboutText();
    expect(text).toMatch(/not quality\s+certifications/i);
    expect(text).toMatch(/not a probability or a confidence/);
    expect(text).toMatch(/cannot reliably tell that an\s+image is out of scope/);
    expect(text).not.toMatch(/auroc|accuracy|\d+(\.\d+)?\s?%/i);
  });

  it("opens outside links safely and names them for screen readers", () => {
    const { main } = aboutText();
    const links = [...main.querySelectorAll("a")];
    expect(links.length).toBeGreaterThanOrEqual(3);
    for (const link of links) {
      expect(link.getAttribute("target")).toBe("_blank");
      expect(link.getAttribute("rel")).toBe("noopener noreferrer");
      expect(link.textContent).toMatch(/opens in a new tab/);
    }
  });
});

describe("the About page is a short summary with the detail in expandable sections", () => {
  it("keeps the attribution, the licence links and the key messages visible", () => {
    const { main } = aboutText();
    const mvtec = main.querySelector(
      'a[href="https://www.mvtec.com/research-teaching/datasets/mvtec-ad"]',
    );
    const licence = main.querySelector(
      'a[href="https://creativecommons.org/licenses/by-nc-sa/4.0/"]',
    );
    expect(mvtec).not.toBeNull();
    expect(licence).not.toBeNull();
    expect(mvtec!.closest("details")).toBeNull();
    expect(licence!.closest("details")).toBeNull();
    for (const phrase of [
      /Copyright 2019 MVTec Software GmbH/,
      /prohibits commercial use/,
      /non-commercial educational/,
      /support or endorse this project/,
      /not\s+legal advice/,
      /not\s+quality certifications/,
      /not a probability or a\s+confidence/,
    ]) {
      const hit = [...main.querySelectorAll("p")].find((p) => phrase.test(p.textContent ?? ""));
      expect(hit, String(phrase)).toBeDefined();
      expect(hit!.closest("details"), String(phrase)).toBeNull();
    }
  });

  it("puts the supporting detail in closed sections and deletes none of it", () => {
    const { main } = aboutText();
    const titles = [...main.querySelectorAll("details > summary")].map((x) => x.textContent);
    expect(titles).toEqual([
      "Dataset details and what is unconfirmed",
      "Please cite the dataset",
      "Code licence",
      "Privacy",
      "Disclaimers",
    ]);
    for (const d of main.querySelectorAll("details")) expect(d).not.toHaveAttribute("open");
  });
});

describe("the claims on the About page about the web app itself stay true", () => {
  function sourceFiles(dir: string): string[] {
    return readdirSync(dir).flatMap((name) => {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) return sourceFiles(full);
      return /\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name) ? [full] : [];
    });
  }

  it("no source file uses cookies or browser storage", () => {
    for (const file of sourceFiles(join(process.cwd(), "src"))) {
      expect(readFileSync(file, "utf8"), file).not.toMatch(
        /document\.cookie|localStorage|sessionStorage|indexedDB/,
      );
    }
  });

  it("index.html and the styles load nothing from another site", () => {
    const html = readFileSync(join(process.cwd(), "index.html"), "utf8");
    const css = readFileSync(join(process.cwd(), "src", "index.css"), "utf8");
    expect(html).not.toMatch(/(src|href)=["']https?:\/\//);
    expect(css).not.toMatch(/https?:\/\//);
  });
});
