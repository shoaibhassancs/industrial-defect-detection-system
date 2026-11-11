import { forwardRef } from "react";
import type { InspectionResponse } from "../../api/types";

function verdictLabel(prediction: string): string {
  if (prediction === "normal") return "Normal";
  if (prediction === "anomalous") return "Anomalous";
  return prediction; // an unknown value is shown as received, never guessed
}

function formatScore(value: number): string {
  return value.toFixed(4);
}

function Figure({ title, src, caption }: { title: string; src: string; caption: string }) {
  return (
    <figure className="min-w-0">
      <img
        src={src}
        alt={`${title}: ${caption}`}
        className="w-full rounded border border-slate-300 bg-slate-100"
      />
      <figcaption className="mt-1 text-sm text-slate-700">
        <span className="font-medium">{title}</span>{" "}
        <span className="text-slate-600">{caption}</span>
      </figcaption>
    </figure>
  );
}

/** Shows one inspection result. The verdict is the server's; nothing is recomputed here. */
export const ResultPanel = forwardRef<HTMLHeadingElement, { result: InspectionResponse }>(
  function ResultPanel({ result }, headingRef) {
    const anomalous = result.prediction === "anomalous";
    const { score_higher_than: higher, of_calibration_normals: of } = result.calibration_rank;
    const area = (result.highlighted_area_fraction * 100).toFixed(2);
    return (
      <section aria-labelledby="result-heading" className="space-y-5">
        <h2
          id="result-heading"
          ref={headingRef}
          tabIndex={-1}
          className="text-xl font-semibold outline-none focus-visible:underline"
        >
          Result
        </h2>

        <div
          className={`rounded-lg border-2 p-4 ${
            anomalous ? "border-amber-700 bg-amber-50" : "border-emerald-700 bg-emerald-50"
          }`}
        >
          <p className="text-lg font-semibold">
            <span aria-hidden="true">{anomalous ? "⚠ " : "✓ "}</span>
            Demo decision: {verdictLabel(result.prediction)}
          </p>
          <p className="mt-1 text-sm">
            A demo decision from one model on one category. It is not a quality certification.
          </p>
        </div>

        <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
          <div>
            <dt className="text-sm text-slate-600">Anomaly score</dt>
            <dd className="font-mono text-lg">{formatScore(result.anomaly_score)}</dd>
          </div>
          <div>
            <dt className="text-sm text-slate-600">Decision threshold</dt>
            <dd className="font-mono text-lg">{formatScore(result.threshold)}</dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-sm text-slate-600">Rank among calibration images</dt>
            <dd>
              Score higher than <strong>{higher}</strong> of <strong>{of}</strong> calibration
              normal images.
            </dd>
          </div>
          <div>
            <dt className="text-sm text-slate-600">Highlighted area</dt>
            <dd>{area}% of the image</dd>
          </div>
          <div>
            <dt className="text-sm text-slate-600">Largest highlighted region</dt>
            <dd>
              {result.region_bbox
                ? `x ${result.region_bbox[0]}, y ${result.region_bbox[1]}, ${result.region_bbox[2]} × ${result.region_bbox[3]} px`
                : "none"}
            </dd>
          </div>
        </dl>
        <p className="text-sm text-slate-700">
          The anomaly score is not a probability and not a confidence. It is only meaningful next to
          the threshold and the calibration images.
        </p>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <Figure
            title="Original"
            src={result.images.original.url}
            caption="the image you uploaded"
          />
          <Figure
            title="Heatmap"
            src={result.images.heatmap.url}
            caption="where the model found patches unlike normal ones"
          />
          <Figure
            title="Overlay"
            src={result.images.overlay.url}
            caption="heatmap and outline drawn over your image"
          />
        </div>

        <details className="rounded border border-slate-300 p-3 text-sm">
          <summary className="cursor-pointer font-medium">How this decision was made</summary>
          <p className="mt-2">{result.decision_rule}</p>
          <p className="mt-2">
            Model: {result.model.name}, version {result.model.version}. Scoring took{" "}
            {Math.round(result.timing_ms.inference)} ms; the whole request took{" "}
            {Math.round(result.timing_ms.total)} ms. Nothing was stored
            {result.persisted ? " (the server reports otherwise)" : ""}.
          </p>
        </details>

        {result.warnings.length > 0 && (
          <ul className="list-disc pl-5 text-sm">
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}

        <div className="rounded border border-slate-300 bg-slate-50 p-3 text-sm">
          <p>{result.limitation}</p>
          <p className="mt-1">
            The model can not reliably tell that an image is out of scope, so an image of something
            else may still get a confident-looking score. {result.disclaimer}
          </p>
        </div>
      </section>
    );
  },
);
