import { useEffect, useRef } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router";
import { sourceUrl } from "./config";
import { InspectPage } from "./features/inspect/InspectPage";
import About from "./pages/About";
import Home from "./pages/Home";
import Model from "./pages/Model";
import NotFound from "./pages/NotFound";
import { useDocumentTitle } from "./useDocumentTitle";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/inspect", label: "Demo", end: false },
  { to: "/model", label: "Results", end: false },
  { to: "/about", label: "About", end: false },
];

function Inspect() {
  useDocumentTitle("Demo");
  return <InspectPage />;
}

/** The page frame. It must be rendered inside a router (BrowserRouter in main.tsx, MemoryRouter
 * in tests). */
export default function App() {
  const { pathname } = useLocation();
  const main = useRef<HTMLElement>(null);
  const first = useRef(true);

  // after a navigation, move keyboard focus and screen-reader attention to the new page content
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    main.current?.focus();
  }, [pathname]);

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
        <nav aria-label="Main" className="mt-3">
          <ul className="flex flex-wrap gap-2">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `inline-block rounded px-3 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 ${
                      isActive
                        ? "bg-slate-900 text-white"
                        : "border border-slate-400 text-slate-900 hover:bg-slate-100"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      <main id="main" ref={main} tabIndex={-1} className="flex-1 py-6 outline-none">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/inspect" element={<Inspect />} />
          <Route path="/model" element={<Model />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <footer className="border-t border-slate-300 py-4 text-xs text-slate-700">
        <p>
          Educational demo on the MVTec AD dataset (CC BY-NC-SA 4.0, noncommercial use). Results are
          demo decisions, not quality certifications; scores are not probabilities.{" "}
          <a
            href={sourceUrl()}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium underline"
          >
            Source code
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        </p>
      </footer>
    </div>
  );
}
