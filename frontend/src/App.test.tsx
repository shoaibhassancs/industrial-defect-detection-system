import { render, screen } from "@testing-library/react";
import axe from "axe-core";
import { expect, it } from "vitest";
import App from "./App";

it("shows the page landmarks, the disclaimer and the licence note", async () => {
  render(<App />);
  expect(screen.getByRole("heading", { level: 1, name: "DefectLens" })).toBeInTheDocument();
  expect(screen.getByRole("main")).toBeInTheDocument();
  expect(screen.getByRole("contentinfo")).toHaveTextContent(/CC BY-NC-SA 4\.0/);
  expect(screen.getByRole("contentinfo")).toHaveTextContent(/not quality certifications/i);
  await screen.findByLabelText(/category/i);
  const result = await axe.run(document.body, { rules: { "color-contrast": { enabled: false } } });
  expect(result.violations.map((v) => v.id)).toEqual([]);
});

it("contains no dataset image: the only images are the ones a test creates", () => {
  render(<App />);
  expect(document.querySelectorAll("img")).toHaveLength(0);
});
