import { useEffect, useId, useRef, useState } from "react";
import { inspectImage } from "../../api/client";
import { ApiError } from "../../api/errors";
import type { InspectionResponse } from "../../api/types";
import { ErrorNotice } from "./ErrorNotice";
import { ResultPanel } from "./ResultPanel";
import { useCategories } from "./useCategories";
import { checkFile, MAX_UPLOAD_MB } from "./validateFile";

type Run =
  | { status: "idle" }
  | { status: "running"; slow: boolean }
  | { status: "done"; result: InspectionResponse }
  | { status: "failed"; error: ApiError };

export interface InspectPageProps {
  /** After this many milliseconds of waiting, say that the service may be waking up. */
  slowAfterMs?: number;
  /** How long to wait before asking again while the model is still loading. */
  startupRetryMs?: number;
}

export function InspectPage({ slowAfterMs = 4000, startupRetryMs = 5000 }: InspectPageProps) {
  const { state: categories, reload } = useCategories(startupRetryMs);
  const [category, setCategory] = useState("");
  const [chosen, setChosen] = useState<{ file: File; previewUrl: string } | null>(null);
  const [fileProblem, setFileProblem] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [dropNote, setDropNote] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [run, setRun] = useState<Run>({ status: "idle" });
  const abortRef = useRef<AbortController | null>(null);
  const previewRef = useRef<string | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const categoryId = useId();
  const fileId = useId();
  const hintId = useId();

  // the preview is made from the file in this tab only; it is released when replaced or on leaving
  function releasePreview() {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = null;
  }
  useEffect(
    () => () => {
      releasePreview();
      abortRef.current?.abort();
    },
    [],
  );

  // the first category is chosen for the visitor once the list is known
  const available = categories.status === "ready" ? categories.categories : [];
  const effectiveCategory = category || available[0]?.id || "";

  useEffect(() => {
    if (run.status === "done") resultHeading.current?.focus();
  }, [run]);

  async function onFileChosen(picked: File | undefined) {
    setRun({ status: "idle" });
    releasePreview();
    if (!picked) {
      setChosen(null);
      setFileProblem(null);
      return;
    }
    const problem = await checkFile(picked);
    if (problem) {
      setChosen(null);
      setFileProblem(problem.message);
    } else {
      const previewUrl = URL.createObjectURL(picked);
      previewRef.current = previewUrl;
      setChosen({ file: picked, previewUrl });
      setFileProblem(null);
    }
  }

  // Dropping uses exactly the same path as picking: onFileChosen -> checkFile. Only the first
  // file is used. The real file input stays the keyboard route (Tab, then Enter or Space).
  function onDrop(event: React.DragEvent) {
    event.preventDefault();
    setDragging(false);
    if (running) return;
    const files = Array.from(event.dataTransfer.files);
    const first = files[0];
    if (!first) return;
    setDropNote(
      files.length > 1
        ? `You dropped ${files.length} files. Only the first one, ${first.name}, is used.`
        : null,
    );
    try {
      // keep the chooser's own display in step with the dropped file where the browser allows it
      if (fileInput.current) fileInput.current.files = event.dataTransfer.files;
    } catch {
      /* not supported here; the page state below is what counts */
    }
    void onFileChosen(first);
  }

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!chosen || !effectiveCategory || run.status === "running") return;
    const controller = new AbortController();
    abortRef.current = controller;
    setRun({ status: "running", slow: false });
    const timer = setTimeout(() => setRun({ status: "running", slow: true }), slowAfterMs);
    try {
      const result = await inspectImage(effectiveCategory, chosen.file, controller.signal);
      setRun({ status: "done", result });
    } catch (error) {
      if (error instanceof ApiError && error.code === "ABORTED") {
        setRun({ status: "idle" });
      } else {
        setRun({
          status: "failed",
          error:
            error instanceof ApiError
              ? error
              : new ApiError({ code: "BAD_RESPONSE", message: String(error) }),
        });
      }
    } finally {
      clearTimeout(timer);
      abortRef.current = null;
    }
  }

  const running = run.status === "running";

  return (
    <div className="space-y-6">
      <section aria-labelledby="inspect-heading" className="space-y-4">
        <h2 id="inspect-heading" className="text-xl font-semibold">
          Inspect an image
        </h2>
        <p className="text-slate-700">
          Choose a category and a JPEG or PNG image. The model compares it with images of normal
          items and shows where it looks unusual. Images are checked and discarded; nothing is
          stored.
        </p>
        <p className="rounded border border-slate-300 bg-slate-50 p-3 text-sm text-slate-800">
          This demo is designed for images like the MVTec AD bottle images. It may be unreliable on
          other objects or image types, and it cannot reliably tell that an image is out of scope.
          It gives a demo decision, not a quality certification.
        </p>

        {categories.status === "loading" && (
          <p role="status">
            {categories.waking
              ? "The model is starting up. This page will try again by itself…"
              : "Loading categories…"}
          </p>
        )}
        {categories.status === "error" && <ErrorNotice error={categories.error} onRetry={reload} />}

        {categories.status === "ready" && (
          <form onSubmit={onSubmit} className="space-y-4" noValidate>
            <div>
              <label htmlFor={categoryId} className="block text-sm font-medium">
                Category
              </label>
              <select
                id={categoryId}
                value={effectiveCategory}
                onChange={(e) => setCategory(e.target.value)}
                disabled={running}
                className="mt-1 block w-full rounded border border-slate-400 bg-white p-2 sm:w-64"
              >
                {categories.categories.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.display_name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor={fileId} className="block text-sm font-medium">
                Image (JPEG or PNG, up to {MAX_UPLOAD_MB} MB)
              </label>
              <div
                data-testid="drop-area"
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!running) setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                className={`mt-1 rounded border-2 border-dashed p-4 ${
                  dragging ? "border-slate-900 bg-slate-100" : "border-slate-400"
                }`}
              >
                <p className="mb-2 text-sm text-slate-700">
                  {dragging
                    ? "Drop the image to use it."
                    : "Drag an image into this box, or use the file chooser below."}
                </p>
                <input
                  ref={fileInput}
                  id={fileId}
                  type="file"
                  accept="image/jpeg,image/png"
                  aria-describedby={hintId}
                  aria-invalid={fileProblem ? true : undefined}
                  disabled={running}
                  onChange={(e) => {
                    setDropNote(null);
                    void onFileChosen(e.target.files?.[0]);
                  }}
                  className="block w-full text-sm file:mr-3 file:rounded file:border file:border-slate-500 file:bg-white file:px-3 file:py-1.5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
                />
              </div>
              <p id={hintId} className="mt-1 text-xs text-slate-600">
                Meant for images like the chosen MVTec AD category. Do not upload anything private.
              </p>
              {dropNote && (
                <p role="status" className="mt-2 text-sm text-slate-800">
                  {dropNote}
                </p>
              )}
              {fileProblem && (
                <p role="alert" className="mt-2 text-sm font-medium text-red-800">
                  <span aria-hidden="true">⚠ </span>
                  {fileProblem}
                </p>
              )}
            </div>

            {chosen && (
              <figure className="max-w-xs">
                <img
                  src={chosen.previewUrl}
                  alt={`Preview of the chosen file ${chosen.file.name}`}
                  className="rounded border border-slate-300"
                />
                <figcaption className="mt-1 text-xs text-slate-600">
                  {chosen.file.name} ({(chosen.file.size / 1024).toFixed(0)} KB)
                </figcaption>
              </figure>
            )}

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={!chosen || running}
                className="rounded bg-slate-900 px-4 py-2 font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
              >
                {running ? "Inspecting…" : "Inspect image"}
              </button>
              {running && (
                <button
                  type="button"
                  onClick={() => abortRef.current?.abort()}
                  className="rounded border border-slate-500 px-4 py-2 font-medium hover:bg-slate-100"
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        )}

        <div aria-live="polite">
          {run.status === "running" && (
            <p role="status">
              {run.slow
                ? "Still working. The service may be waking up after a quiet period; this can take up to a minute."
                : "Inspecting the image…"}
            </p>
          )}
        </div>
        {run.status === "failed" && <ErrorNotice error={run.error} />}
      </section>

      {run.status === "done" && <ResultPanel ref={resultHeading} result={run.result} />}
    </div>
  );
}
