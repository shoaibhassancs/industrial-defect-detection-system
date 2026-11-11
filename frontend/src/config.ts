/** Where the project's documents can be read. The Model page links to docs/evaluation.md there.
 * Set VITE_DOCS_BASE_URL at build time to change it (for example for a fork). */
const DEFAULT_DOCS_BASE_URL =
  "https://github.com/shoaibhassancs/industrial-defect-detection-system/blob/main/docs";

export function docsUrl(file: string): string {
  const configured = import.meta.env["VITE_DOCS_BASE_URL"] as string | undefined;
  const base = configured && configured.trim() !== "" ? configured : DEFAULT_DOCS_BASE_URL;
  return `${base.replace(/\/+$/, "")}/${file}`;
}

const DEFAULT_SOURCE_URL = "https://github.com/shoaibhassancs/industrial-defect-detection-system";

/** The project's source code. Set VITE_SOURCE_URL at build time to change the link. */
export function sourceUrl(): string {
  const configured = import.meta.env["VITE_SOURCE_URL"] as string | undefined;
  return (configured && configured.trim() !== "" ? configured : DEFAULT_SOURCE_URL).replace(
    /\/+$/,
    "",
  );
}
