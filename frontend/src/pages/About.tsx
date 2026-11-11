import { Detail } from "../components/Detail";
import { useDocumentTitle } from "../useDocumentTitle";

function External({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="break-words underline">
      {children}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}

/** Attribution, licence, privacy and disclaimers. The wording follows DATA_LICENSE.md, docs/data.md
 * and docs/model-card.md; it adds no legal claim beyond them. The dataset attribution and licence
 * link stay visible; the supporting detail is in expandable sections. No dataset image is shown. */
export default function About() {
  useDocumentTitle("About");
  return (
    <div className="space-y-6">
      <section aria-labelledby="about-heading" className="space-y-2">
        <h2 id="about-heading" className="text-xl font-semibold">
          About this demo
        </h2>
        <p className="text-slate-800">
          DefectLens is a non-commercial educational project. It shows how an unsupervised anomaly
          detector can flag and localise surface defects on one category of the public MVTec AD
          dataset (bottles). It is not a certified inspection system, it gives demo decisions and
          not quality certifications, and the score it reports is not a probability or a confidence.
        </p>
      </section>

      <section aria-labelledby="data-heading" className="space-y-2">
        <h2 id="data-heading" className="text-xl font-semibold">
          The dataset and its licence
        </h2>
        <p className="text-slate-800">
          The model was fitted and calibrated on the{" "}
          <External href="https://www.mvtec.com/research-teaching/datasets/mvtec-ad">
            MVTec Anomaly Detection dataset (MVTec AD)
          </External>
          , created by Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger of MVTec
          Software GmbH. The dataset&rsquo;s own readme says “Copyright 2019 MVTec Software GmbH”.
          It is released under the{" "}
          <External href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
            Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence (CC
            BY-NC-SA 4.0)
          </External>
          , which prohibits commercial use. This project is a non-commercial educational demo and
          must not be used to make money. Nothing here implies that MVTec Software GmbH or the
          dataset authors support or endorse this project. This is a plain-language notice, not
          legal advice.
        </p>
      </section>

      <section aria-labelledby="more-heading" className="space-y-3">
        <h2 id="more-heading" className="text-xl font-semibold">
          More detail
        </h2>

        <Detail title="Dataset details and what is unconfirmed">
          <ul className="list-disc space-y-2 pl-5">
            <li>
              The dataset itself is not part of this project&rsquo;s repository and is not served by
              this site.
            </li>
            <li>
              The fitted model files hold numbers computed from MVTec AD images. As a cautious
              policy, not a legal finding, the project treats the model and anything derived from
              the dataset under the same terms: attribution, non-commercial use, and share-alike for
              the derived part.
            </li>
            <li>
              According to the project owner, the copy of the dataset used here was downloaded from
              Kaggle. That Kaggle page and its terms are unconfirmed, and the copy has not been
              compared file by file with MVTec&rsquo;s official archive.
            </li>
            <li>
              The full text is in the project files <code>DATA_LICENSE.md</code> and{" "}
              <code>docs/data.md</code>.
            </li>
          </ul>
        </Detail>

        <Detail title="Please cite the dataset">
          <ul className="list-disc space-y-2 pl-5">
            <li>
              Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger, “A Comprehensive
              Real-World Dataset for Unsupervised Anomaly Detection”, IEEE Conference on Computer
              Vision and Pattern Recognition, 2019.
            </li>
            <li>
              Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., &amp; Steger, C. (2021). “The
              MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised
              Anomaly Detection.” <em>International Journal of Computer Vision</em>, 129, 1038–1059.{" "}
              <External href="https://doi.org/10.1007/s11263-020-01400-4">
                https://doi.org/10.1007/s11263-020-01400-4
              </External>
            </li>
          </ul>
        </Detail>

        <Detail title="Code licence">
          <p>
            The plan is to release the original code of this project under the MIT licence, but no
            licence file has been added yet. Until it is, no licence is granted for the code. When
            it is added, it will cover original code only, not the dataset or third-party material.
          </p>
        </Detail>

        <Detail title="Privacy">
          <ul className="list-disc space-y-2 pl-5">
            <li>
              When you press “Inspect image”, the image is sent to the API. The API checks it,
              scores it, and sends the result back. Nothing is stored: the result says so, and the
              API keeps no copy of your image.
            </li>
            <li>
              The API&rsquo;s log lines record the kind of request, its status, how long it took and
              a request ID. They do not record the file name or the image content (checked in the
              project&rsquo;s backend tests and in a local run).
            </li>
            <li>
              To limit how many requests one visitor can make per minute, the API uses the network
              address the request comes from.
            </li>
            <li>
              This web app sets no cookies, keeps nothing in your browser&rsquo;s storage, and loads
              no third-party scripts or fonts.
            </li>
            <li>
              This demo has not been deployed yet. Wherever it runs, whoever operates that computer
              or hosting service can see the requests it receives, and a hosting service may keep
              its own logs; those are not checked or controlled by this project. Please do not
              upload anything private.
            </li>
          </ul>
        </Detail>

        <Detail title="Disclaimers">
          <ul className="list-disc space-y-1 pl-5">
            <li>
              Results are demo decisions from one model on one category. They are not quality
              certifications and must not be used for real quality-control, safety or production
              decisions.
            </li>
            <li>
              The model is designed for images like the MVTec AD bottle images. It may be unreliable
              on other objects, cameras, lighting or backgrounds, and it cannot reliably tell that
              an image is out of scope.
            </li>
            <li>
              The threshold&rsquo;s guarantee concerns normal images only. It says nothing about how
              many defects are found; that is measured separately on a formal test set.
            </li>
          </ul>
        </Detail>
      </section>
    </div>
  );
}
