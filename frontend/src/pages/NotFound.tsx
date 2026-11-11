import { Link } from "react-router";
import { useDocumentTitle } from "../useDocumentTitle";

export default function NotFound() {
  useDocumentTitle("Page not found");
  return (
    <section aria-labelledby="nf-heading" className="space-y-2">
      <h2 id="nf-heading" className="text-xl font-semibold">
        Page not found
      </h2>
      <p>
        There is no page at this address. Go to the{" "}
        <Link to="/" className="underline">
          home page
        </Link>
        .
      </p>
    </section>
  );
}
