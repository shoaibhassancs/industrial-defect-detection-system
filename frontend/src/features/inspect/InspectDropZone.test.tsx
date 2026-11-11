import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { BASE } from "../../test/handlers";
import { inspection, pngFile, textFile } from "../../test/fixtures";
import { server } from "../../test/server";
import { InspectPage } from "./InspectPage";

// What these tests can and cannot show:
//  - They check the structure (a real, visible, labelled file input that Tab reaches), the shared
//    validation, and the drop logic with simulated events.
//  - They do NOT open the operating system's file chooser. Whether Enter or Space on the focused
//    input opens it, and how the focus ring and a real drag look, need a manual check in a real
//    browser (see the checklist in the README).

async function ready() {
  render(<InspectPage slowAfterMs={60} startupRetryMs={30} />);
  const input = (await screen.findByLabelText(/^image/i)) as HTMLInputElement;
  return input;
}

function drop(files: File[]) {
  fireEvent.drop(screen.getByTestId("drop-area"), { dataTransfer: { files } });
}

describe("the real file input", () => {
  it("is a visible, labelled file input that Tab can reach, and it is not hidden", async () => {
    const user = userEvent.setup();
    const input = await ready();
    expect(input.tagName).toBe("INPUT");
    expect(input.type).toBe("file");
    expect(input).toBeVisible();
    expect(input).not.toHaveAttribute("hidden");
    expect(input).not.toHaveAttribute("tabindex", "-1");
    expect(input.className).not.toMatch(/\b(hidden|sr-only|invisible)\b/);
    expect(input).toHaveAccessibleName(/image \(jpeg or png/i);
    expect(input).toHaveAccessibleDescription(/do not upload anything private/i);
    // Tab from the category select reaches the input (the next focusable element)
    await user.click(screen.getByLabelText(/category/i));
    await user.tab();
    expect(input).toHaveFocus();
  });

  it("has an explicit visible focus style", async () => {
    const input = await ready();
    expect(input.className).toContain("focus-visible:outline-2");
  });

  it("does not make the drop area itself a control", async () => {
    await ready();
    const area = screen.getByTestId("drop-area");
    expect(area.tagName).toBe("DIV");
    expect(area).not.toHaveAttribute("tabindex");
    expect(area).not.toHaveAttribute("role");
  });

  it("has no automatically detectable accessibility problems in the form", async () => {
    await ready();
    const config = { rules: { "color-contrast": { enabled: false } } };
    const result = await axe.run(document.body, config);
    expect(result.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
  });
});

describe("dropping a file", () => {
  it("uses a valid dropped image exactly like a picked one", async () => {
    await ready();
    drop([pngFile("dropped.png")]);
    expect(await screen.findByAltText(/dropped\.png/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /inspect image/i })).toBeEnabled();
  });

  it("sends a dropped image to the same endpoint as a picked one", async () => {
    let calls = 0;
    server.use(
      http.post(`${BASE}/inspections`, () => {
        calls += 1;
        return HttpResponse.json(inspection());
      }),
    );
    const user = userEvent.setup();
    await ready();
    drop([pngFile("dropped.png")]);
    await screen.findByAltText(/dropped\.png/);
    await user.click(screen.getByRole("button", { name: /inspect image/i }));
    expect(await screen.findByRole("heading", { name: "Result" })).toBeInTheDocument();
    expect(calls).toBe(1);
  });

  it.each([
    ["a non-image", () => textFile(), /choose a jpeg or png image/i],
    ["an empty file", () => new File([], "empty.png", { type: "image/png" }), /file is empty/i],
    [
      "a file over the size limit",
      () => pngFile("big.png", 11 * 1024 * 1024),
      /larger than 10 MB/i,
    ],
  ])("refuses %s with the same message as picking, and sends nothing", async (_n, make, words) => {
    let calls = 0;
    server.use(
      http.post(`${BASE}/inspections`, () => {
        calls += 1;
        return HttpResponse.json(inspection());
      }),
    );
    const user = userEvent.setup({ applyAccept: false });
    const input = await ready();
    // picked: the message
    await user.upload(input, make());
    const picked = (await screen.findByRole("alert")).textContent;
    expect(picked).toMatch(words);
    // dropped: the same message, and nothing is sent
    drop([pngFile("ok.png")]);
    await screen.findByAltText(/ok\.png/);
    drop([make()]);
    const dropped = (await screen.findByRole("alert")).textContent;
    expect(dropped).toBe(picked);
    expect(screen.getByRole("button", { name: /inspect image/i })).toBeDisabled();
    expect(calls).toBe(0);
  });

  it("uses only the first of several dropped files and says so", async () => {
    await ready();
    drop([pngFile("first.png"), pngFile("second.png")]);
    expect(await screen.findByAltText(/first\.png/)).toBeInTheDocument();
    expect(screen.queryByAltText(/second\.png/)).not.toBeInTheDocument();
    expect(document.body.textContent).toMatch(/dropped 2 files\. Only the first one, first\.png/);
  });

  it("ignores a drop that carries no files", async () => {
    await ready();
    drop([]);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /inspect image/i })).toBeDisabled();
  });

  it("shows a text hint, not only a colour change, while a file is dragged over", async () => {
    await ready();
    const area = screen.getByTestId("drop-area");
    expect(area).toHaveTextContent(/drag an image into this box/i);
    fireEvent.dragOver(area);
    expect(area).toHaveTextContent(/drop the image to use it/i);
    fireEvent.dragLeave(area);
    expect(area).toHaveTextContent(/drag an image into this box/i);
  });
});
