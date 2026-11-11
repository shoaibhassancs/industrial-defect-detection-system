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

function Item({
  term,
  value,
  explanation,
  wide = false,
}: {
  term: string;
  value: React.ReactNode;
  explanation: string;
  wide?: boolean;
}) {
  return (
    <div className={wide ? "sm:col-span-2" : undefined}>
      <dt className="text-sm font-medium text-slate-800">{term}</dt>
      <dd className="text-lg">{value}</dd>
      <dd className="text-sm text-slate-700">{explanation}</dd>
    </div>
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
            The demo says “Anomalous” only when the anomaly score is above the threshold. This is a
            demo decision from one model on one category, not a quality certification.
          </p>
        </div>

        <dl className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2">
          <Item
            term="Anomaly score (higher means more unusual)"
            value={<span className="font-mono">{formatScore(result.anomaly_score)}</span>}
            explanation="How different the most unusual part of this image is from the normal images the model learned from. It is not a probability or a confidence, and it only means something next to the threshold."
          />
          <Item
            term="Threshold (the cutoff)"
            value={<span className="font-mono">{formatScore(result.threshold)}</span>}
            explanation={`The demo says “Anomalous” only when the score is above this cutoff. It was set using ${of} normal calibration images.`}
          />
          <Item
            wide
            term={`Comparison with ${of} normal calibration images`}
            value={
              <>
                Score higher than <strong>{higher}</strong> of <strong>{of}</strong> calibration
                normal images.
              </>
            }
            explanation={`This counts how many of the ${of} normal calibration images scored lower than this image. It is a comparison with those images, not a measure of confidence.`}
          />
          <Item
            term="Highlighted area"
            value={<>{area}% of the image</>}
            explanation="A display aid: the share of the image drawn above a highlight level taken from the calibration images. It is not a measured defect size and does not change the decision."
          />
          <Item
            term="Largest highlighted region"
            value={
              result.region_bbox
                ? `x ${result.region_bbox[0]}, y ${result.region_bbox[1]}, ${result.region_bbox[2]} × ${result.region_bbox[3]} px`
                : "none"
            }
            explanation="A box around the biggest highlighted patch, in pixels of your original image."
          />
        </dl>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <Figure
            title="Original"
            src={result.images.original.url}
            caption="the image you uploaded"
          />
          <Figure
            title="Heatmap"
            src={result.images.heatmap.url}
            caption="where the score is lower (darker colours) or higher (brighter orange and yellow)"
          />
          <Figure
            title="Overlay"
            src={result.images.overlay.url}
            caption="the heatmap over your image, with an outline around the highlighted area"
          />
        </div>

        <details className="rounded border border-slate-300 p-3 text-sm">
          <summary className="cursor-pointer font-medium">How to read these results</summary>
          <div className="mt-2 space-y-2">
            <p>
              The model keeps a memory of small patches taken from normal images. For your image it
              finds the patch that matches that memory least well; the anomaly score measures how
              poorly. A higher score means more unusual, compared with the normal images. The score
              is not a probability, a confidence or a percentage of defect.
            </p>
            <p>
              The threshold is a cutoff chosen from the scores of {of} normal calibration images
              that were kept aside during training. The demo says “Anomalous” only when the score is
              above it. The server makes that decision; this page only shows it.
            </p>
            <p>
              The comparison counts how many calibration images scored lower than yours. It helps
              you see where the score sits among normal images; it says nothing about how likely a
              defect is.
            </p>
            <p>
              The heatmap colours show lower and higher scores for each part of the image. The
              highlighted area is a display aid. It is not a measured defect size, and it does not
              change the decision.
            </p>
            <p>
              How well the model works is measured on a separate, formal test set, not by trying one
              image on this page.
            </p>
          </div>
        </details>

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
            The model cannot reliably tell that an image is out of scope, so results on other
            objects or image types may be unreliable. {result.disclaimer}
          </p>
        </div>
      </section>
    );
  },
);
