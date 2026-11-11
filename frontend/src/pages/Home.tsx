import { Link } from "react-router";
import { EvidenceCards } from "../components/EvidenceCards";
import { docsUrl, sourceUrl } from "../config";
import evidence from "../generated/evidence.json";
import { useDocumentTitle } from "../useDocumentTitle";

function External({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="underline">
      {children}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}

const buttonBase =
  "inline-block rounded px-4 py-2 font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900";

/** The landing page: a one-screen summary for a quick reader. It makes NO API calls. The headline
 * numbers come from src/generated/evidence.json, a snapshot of the saved evaluation made by
 * `npm run gen:evidence` and checked against metrics.json by a test, so a slow or sleeping backend
 * can never make this page look broken. */
export default function Home() {
  useDocumentTitle("DefectLens");
  const total = evidence.n_anomalous + evidence.n_normal;
  return (
    <div className="space-y-8">
      <section aria-labelledby="home-heading" className="space-y-3">
        <h2 id="home-heading" className="text-2xl font-semibold">
          Unsupervised defect detection on industrial photos
        </h2>
        <p className="text-slate-800">
          DefectLens learns what normal items look like and flags images that look unusual. It was
          built only from normal bottle photos of the public MVTec AD dataset, and it shows where in
          the image it found something unusual. It is an educational demo that gives demo decisions,
          not quality certifications.
        </p>
        <p className="flex flex-wrap gap-3">
          <Link
            to="/inspect"
            className={`${buttonBase} bg-slate-900 text-white hover:bg-slate-700`}
          >
            Try the demo
          </Link>
          <Link
            to="/model"
            className={`${buttonBase} border border-slate-500 text-slate-900 hover:bg-slate-100`}
          >
            See the results
          </Link>
        </p>
      </section>

      <section aria-labelledby="evidence-heading" className="space-y-3">
        <h2 id="evidence-heading" className="text-xl font-semibold">
          Headline evidence (saved test results)
        </h2>
        <EvidenceCards
          defectsFlagged={evidence.defects_flagged}
          defectiveTotal={evidence.n_anomalous}
          falseAlarms={evidence.false_alarms}
          normalTotal={evidence.n_normal}
          falseAlarmRate={evidence.false_alarm_rate}
          targetRate={evidence.target_rate}
        />
        <p className="text-sm text-slate-800">
          Test set: {total} images ({evidence.n_anomalous} defective, {evidence.n_normal} normal),
          one category ({evidence.category}), tested once after the model and threshold were fixed.
          The test set is small, so the false-alarm rate is a coarse estimate, and flagging every
          defective image is not a promise for other cases. Two models were evaluated on the same
          images; this is the headline model, <span className="font-mono">{evidence.backbone}</span>
          . Details and the comparison with the other model are on the{" "}
          <Link to="/model" className="underline">
            Results page
          </Link>{" "}
          and in <External href={docsUrl("evaluation.md")}>docs/evaluation.md</External>.
        </p>
      </section>

      <section aria-labelledby="built-heading" className="space-y-2">
        <h2 id="built-heading" className="text-xl font-semibold">
          What is in this project
        </h2>
        <ul className="list-disc space-y-1 pl-5 text-slate-800">
          <li>
            An anomaly-detection model (PatchCore) trained on normal images only; it never saw a
            defect while learning.
          </li>
          <li>
            A threshold calibrated on 42 normal images kept aside during training, and a test set
            that stayed sealed until the model and threshold were fixed.
          </li>
          <li>A FastAPI service that runs the model, and this React and TypeScript web app.</li>
          <li>Tests, an evaluation write-up and a model card, so the claims can be checked.</li>
        </ul>
        <p className="text-sm text-slate-700">
          Built with Python, PyTorch, anomalib, FastAPI, React, TypeScript, Tailwind CSS and Vitest.
          Read the <External href={sourceUrl()}>source code</External>, the{" "}
          <External href={docsUrl("evaluation.md")}>evaluation</External> and the{" "}
          <External href={docsUrl("model-card.md")}>model card</External>.
        </p>
      </section>

      <section
        aria-labelledby="limits-heading"
        className="space-y-1 rounded border border-slate-300 bg-slate-50 p-4"
      >
        <h2 id="limits-heading" className="text-xl font-semibold">
          Limits
        </h2>
        <p className="text-slate-800">
          Only the bottle category is available. The model is designed for images like the MVTec AD
          bottle images and may be unreliable on other objects or image types; it cannot reliably
          tell that an image is out of scope. The anomaly score is not a probability or a
          confidence. More on the{" "}
          <Link to="/model" className="underline">
            Results
          </Link>{" "}
          and{" "}
          <Link to="/about" className="underline">
            About
          </Link>{" "}
          pages.
        </p>
      </section>
    </div>
  );
}
