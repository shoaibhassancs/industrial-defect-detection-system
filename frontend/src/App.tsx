import { InspectPage } from "./features/inspect/InspectPage";

export default function App() {
  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 text-slate-900">
      <a
        href="#main"
        className="sr-only rounded bg-white p-2 focus:not-sr-only focus:absolute focus:left-2 focus:top-2"
      >
        Skip to the main content
      </a>
      <header className="border-b border-slate-300 py-4">
        <h1 className="text-2xl font-bold">DefectLens</h1>
        <p className="text-sm text-slate-700">
          An educational anomaly-detection demo on the public MVTec AD dataset. Not a
          quality-certification tool.
        </p>
      </header>
      <main id="main" className="flex-1 py-6">
        <InspectPage />
      </main>
      <footer className="border-t border-slate-300 py-4 text-xs text-slate-700">
        <p>
          Trained on the MVTec AD dataset (CC BY-NC-SA 4.0, noncommercial use). Results are demo
          decisions from one model and are not quality certifications. Scores are not probabilities.
        </p>
      </footer>
    </div>
  );
}
