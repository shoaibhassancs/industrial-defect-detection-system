import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, vi } from "vitest";
import { server } from "./server";

// any request that no handler describes is a test bug, so it fails loudly
beforeAll(() => server.listen({ onUnhandledFrame: "error" }));
afterEach(() => {
  server.resetHandlers();
  cleanup();
});
afterAll(() => server.close());

// jsdom has no object URLs; the page only uses them for the preview of the chosen file
let counter = 0;
URL.createObjectURL = vi.fn(() => `blob:test/${(counter += 1)}`);
URL.revokeObjectURL = vi.fn();
