import { getCategories, getMetrics, getModelInfo } from "../api/client";
import { ApiError } from "../api/errors";
import type { MetricsInfo, ModelInfo } from "../api/modelTypes";
import { Detail } from "../components/Detail";
import { EvidenceCards } from "../components/EvidenceCards";
import { docsUrl } from "../config";
import { ErrorNotice } from "../features/inspect/ErrorNotice";
import { useStartupRetry } from "../features/inspect/useStartupRetry";
import { decimals, humanize, percent } from "../format";
import { useDocumentTitle } from "../useDocumentTitle";

interface ModelPageData {
  model: ModelInfo;
  metrics: MetricsInfo;
}

/** The page is for the one category the service serves; the category comes from /categories. */
async function loadModelPage(signal: AbortSignal): Promise<ModelPageData> {
  const categories = await getCategories(signal);
  const first = categories[0];
  if (!first) {
    throw new ApiError({ code: "BAD_RESPONSE", message: "The service lists no category." });
  }
  const [model, metrics] = await Promise.all([getModelInfo(signal), getMetrics(first.id, signal)]);
  return { model, metrics };
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <tr className="border-t border-slate-200">
      <th scope="row" className="py-1.5 pr-4 text-left font-normal text-slate-800">
        {label}
      </th>
      <td className="py-1.5 font-mono">{children}</td>
    </tr>
  );
}

export default function Model() {
  useDocumentTitle("Results");
  const { state, reload } = useStartupRetry(loadModelPage);
  return (
    <div className="space-y-6">
      <section aria-labelledby="model-heading" className="space-y-2">
        <h2 id="model-heading" className="text-xl font-semibold">
          Results
        </h2>
        <p className="text-slate-800">
          Two models were evaluated on the same test images with the same settings. This page shows
          the headline model; the full comparison with the other model (resnet18) is in{" "}
          <a
            href={docsUrl("evaluation.md")}
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            docs/evaluation.md
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
          . Every number is a saved result that the service reads from the model package; the page
          only displays them and does not recalculate or adjust anything.
        </p>
      </section>

      {state.status === "loading" && (
        <p role="status">
          {state.waking
            ? "The model is starting up. This page will try again by itself…"
            : "Loading the model description and results…"}
        </p>
      )}
      {state.status === "error" && <ErrorNotice error={state.error} onRetry={reload} />}
      {state.status === "ready" && (
        <Results model={state.data.model} metrics={state.data.metrics} />
      )}
    </div>
  );
}

function Results({ model, metrics }: { model: ModelInfo; metrics: MetricsInfo }) {
  const { image, pixel } = metrics;
  const at = image.at_calibrated_threshold;
  const fa = image.false_alarm_on_normal_test_images;
  const cal = model.calibration;
  const total = image.n_normal + image.n_anomalous;
  const ci = (x: { low: number; high: number; confidence: number | null }) =>
    `${x.confidence !== null ? `${percent(x.confidence, 0)} interval: ` : "interval: "}${decimals(x.low)} to ${decimals(x.high)}`;

  return (
    <>
      <section aria-labelledby="headline-heading" className="space-y-3">
        <h2 id="headline-heading" className="text-xl font-semibold">
          What the test showed
        </h2>
        <p className="text-slate-800">
          {model.headline_model === true ? "Headline model " : "Model "}
          <span className="font-mono">{model.backbone}</span>, category{" "}
          <span className="font-mono">{metrics.category}</span>, version {model.version}. Tested
          once on the official test images of this category ({total} images: {image.n_anomalous}{" "}
          defective and {image.n_normal} normal), after the model and its threshold were fixed.
          Nothing was tuned on them. Status from the package: “{metrics.status}”.
        </p>
        <EvidenceCards
          defectsFlagged={at.tp}
          defectiveTotal={image.n_anomalous}
          falseAlarms={fa.false_alarms}
          normalTotal={fa.n_normal}
          falseAlarmRate={fa.fpr}
          targetRate={fa.alpha}
          threshold={decimals(at.threshold, 4)}
        />
        <div className="rounded border border-slate-300 bg-slate-50 p-3 text-sm text-slate-800">
          <p className="font-medium">Read these results with care</p>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            <li>
              The test set is small: only {fa.n_normal} normal images, so each single image moves
              the false-alarm rate by a large step and the rate is a coarse estimate. The full
              discussion of the false alarms is in docs/evaluation.md.
            </li>
            <li>
              Flagging every defective image is a result on this one category and these{" "}
              {image.n_anomalous} defective images. It is not a promise that the model finds every
              defect elsewhere.
            </li>
            <li>
              The false-alarm result is shown on purpose: a model that flags defects is also judged
              by how often it flags normal items.
            </li>
          </ul>
        </div>
      </section>

      <section aria-labelledby="glance-heading" className="space-y-2">
        <h2 id="glance-heading" className="text-xl font-semibold">
          At a glance
        </h2>
        <table className="w-full max-w-2xl text-left">
          <caption className="pb-2 text-left text-sm text-slate-700">
            Saved values, shown to three decimals. Intervals and definitions are in the sections
            below.
          </caption>
          <tbody>
            <Row label="Precision (of the flagged images, the share that were defective)">
              {decimals(at.precision)}
            </Row>
            <Row label="F1 (one score combining precision and detection)">{decimals(at.f1)}</Row>
            <Row label="AUROC (how well scores separate defective from normal images)">
              {decimals(image.auroc.estimate)}
            </Row>
            <Row label="Pixel AUROC (how well heatmap scores separate defect pixels from normal pixels)">
              {decimals(pixel.auroc)}
            </Row>
          </tbody>
        </table>
        <p className="text-sm text-slate-700">
          A top score here describes how the {total} test images (and their pixels) were ordered by
          score. It is a result on a small test set of one category, not proof of perfect detection
          in general.
        </p>
      </section>

      <section aria-labelledby="limits-heading" className="space-y-2">
        <h2 id="limits-heading" className="text-xl font-semibold">
          Limits in short
        </h2>
        <ul className="list-disc space-y-1 pl-5 text-slate-800">
          <li>A demo decision, not a quality certification.</li>
          <li>The anomaly score is not a probability or a confidence.</li>
          <li>
            One category, one split and a small test set; the model may be unreliable on other
            objects or image types.
          </li>
        </ul>
      </section>

      <section aria-labelledby="more-heading" className="space-y-3">
        <h2 id="more-heading" className="text-xl font-semibold">
          More detail
        </h2>

        <Detail title="Image-level results: counts">
          <table className="w-full max-w-xl text-left">
            <caption className="pb-2 text-left text-sm text-slate-700">
              What the demo decided for each test image, at the calibrated threshold.
            </caption>
            <thead>
              <tr>
                <th scope="col" className="pr-4 text-sm font-medium">
                  Test images
                </th>
                <th scope="col" className="pr-4 text-sm font-medium">
                  Flagged “Anomalous”
                </th>
                <th scope="col" className="text-sm font-medium">
                  Called “Normal”
                </th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-slate-200">
                <th scope="row" className="py-1.5 pr-4 text-left font-normal">
                  Defective ({image.n_anomalous})
                </th>
                <td className="py-1.5 pr-4 font-mono">{at.tp}</td>
                <td className="py-1.5 font-mono">{at.fn}</td>
              </tr>
              <tr className="border-t border-slate-200">
                <th scope="row" className="py-1.5 pr-4 text-left font-normal">
                  Normal ({image.n_normal})
                </th>
                <td className="py-1.5 pr-4 font-mono">{at.fp}</td>
                <td className="py-1.5 font-mono">{at.tn}</td>
              </tr>
            </tbody>
          </table>
        </Detail>

        <Detail title="Image-level results: scores and intervals">
          <table className="w-full max-w-2xl text-left">
            <caption className="pb-2 text-left text-sm text-slate-700">
              Saved values, shown to three decimals. Intervals come from resampling the test images.
            </caption>
            <tbody>
              <Row label="Precision">{decimals(at.precision)}</Row>
              <Row label="F1">{decimals(at.f1)}</Row>
              <Row label="AUROC (how well scores separate defective from normal images)">
                {decimals(image.auroc.estimate)} ({ci(image.auroc)})
              </Row>
              <Row label="AUPR (a similar measure that rewards flagging defective images first)">
                {decimals(image.aupr.estimate)} ({ci(image.aupr)})
              </Row>
            </tbody>
          </table>
        </Detail>

        <Detail title="Defective images flagged, by defect type">
          <table className="w-full max-w-xl text-left">
            <thead>
              <tr>
                <th scope="col" className="pr-4 text-sm font-medium">
                  Defect type
                </th>
                <th scope="col" className="text-sm font-medium">
                  Flagged
                </th>
              </tr>
            </thead>
            <tbody>
              {image.per_defect_recall.map((d) => (
                <tr key={d.type} className="border-t border-slate-200">
                  <th scope="row" className="py-1.5 pr-4 text-left font-normal">
                    {humanize(d.type)}
                  </th>
                  <td className="py-1.5 font-mono">
                    {d.detected} of {d.n}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Detail>

        <Detail title="Pixel-level results: does the heatmap point at the defect?">
          <table className="w-full max-w-2xl text-left">
            <caption className="pb-2 text-left text-sm text-slate-700">
              Computed over the {pixel.n_images} test images against the defect masks.
            </caption>
            <tbody>
              <Row label="Pixel AUROC">{decimals(pixel.auroc)}</Row>
              <Row label="AUPRO (overlap with each defect region, up to a limited false-positive rate)">
                {decimals(pixel.aupro_fpr_0_3)}
              </Row>
            </tbody>
          </table>
          <p className="text-slate-700">Note from the evaluation: {pixel.resolution}</p>
        </Detail>

        <Detail title="How the threshold was set">
          <p>
            Method: {cal.method}. {cal.n} normal training images that the model never learned from
            were scored, and the threshold is the score ranked {cal.k} out of {cal.n}, counting up
            from the lowest: <span className="font-mono">{decimals(cal.threshold, 4)}</span>. The
            target level was {percent(cal.alpha)}; the guarantee bound for a new normal image is{" "}
            {percent(cal.false_alarm_bound)}.
          </p>
          <p className="text-slate-700">{cal.meaning}</p>
        </Detail>

        <Detail title="Model details">
          <table className="w-full max-w-2xl text-left">
            <tbody>
              <Row label="Method">{model.algorithm}</Row>
              <Row label="Backbone (feature extractor)">{model.backbone}</Row>
              <Row label="Feature layers">{model.layers.join(", ")}</Row>
              <Row label="Version">{model.version}</Row>
            </tbody>
          </table>
        </Detail>

        <Detail title="All limitations">
          <ul className="list-disc space-y-1 pl-5">
            {model.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Detail>
      </section>
    </>
  );
}
