# Licensing and publication review (read-only)

- **Date:** 2026-11-11. **Not legal advice, not a legal finding.** It records what sources say, what is
  still unread, and the options. It makes no decision; every decision below is the owner's.
- **Nothing was changed** by this review except this file and its links in the records: no licence or
  notice text was edited, nothing was uploaded, no Hugging Face repository or token was created, no figure
  was removed, no Git or GitHub action was taken, and no dataset image, mask or test file was opened.
- **Labels:** **Verified** = read first-hand by a command or from a page in this review.
  **Relayed** = quoted by a helper that fetched the page for Claude (the quote could not be re-read by
  Claude; helpers were limited to 125-character quotes for some sites). **Unverified** = not read, or
  only reported second-hand. **Owner decision** = a choice nobody but the owner can make.
- "Says" = a source's own words. "Reading" = Claude's interpretation, to be weighed, not relied on.

## Summary table

| # | Item | Status | One line |
|---|---|---|---|
| 1 | `LICENSE` file for the code | Verified: none exists. **Owner decision** | MIT is planned, not in force. |
| 2 | Evaluation figures | Verified: **already on the public remote.** Wording corrected (A1); owner will ask MVTec (A2) | Adapted MVTec images, with a notice; sufficiency unresolved. |
| 3 | A7: ImageNet terms | **Unverified** | Official ImageNet pages could not be opened. |
| 4 | A17: Kaggle source | **Unverified** | No Kaggle page could be read; the official MVTec page could not be opened either. |
| 5 | Uploading the weights | **Owner decision; not established as permitted** | A private repository does not settle it. |
| 6 | Public use of the weights (Render demo) | **Owner decision** | Separate from the upload; not assessed here. |

## What the primary sources say

### 1. The MVTec AD dataset's own bundled files (Verified, read first-hand)

Read from the local copy `data/raw/mvtec-ad/` (text files only; SHA-256 of `readme.txt`
`60222d7e…667c` and of `license.txt` `a355a579…a330`, identical at the top level and in `bottle/`):

- `readme.txt` **says**: "Copyright 2019 MVTec Software GmbH"; "This work is licensed under a Creative
  Commons Attribution-NonCommercial-ShareAlike 4.0 International License."; and "For using the data in a
  way that falls under the commercial use clause of the license, please contact us." It asks for the
  2019 CVPR citation when used "in scientific work".
- `license.txt` is the standard CC BY-NC-SA 4.0 text (20,845 bytes; its first line is the licence name).
- **Limit:** these are the copies bundled in a copy that came from Kaggle (owner's statement). They are the
  strongest evidence available here, but they are not MVTec's own current page (see 4).

### 2. The CC BY-NC-SA 4.0 legal code (Relayed; page loaded, read in full by the helper)

Source: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en>. Quotes are short fragments.

- Grant (2(a)(1)) **says**: reproduce and Share the licensed material "for NonCommercial purposes only", and
  "produce, reproduce, and Share Adapted Material for NonCommercial purposes only".
- NonCommercial **says**: "not primarily intended for or directed towards commercial advantage or monetary
  compensation."
- "Adapted Material" is material "derived from or based upon the Licensed Material" and modified in a way
  that needs the licensor's permission. "Share" **says**: "to provide material to the public by any means or
  process that requires permission under the Licensed Rights".
- Attribution (3(a)) applies "If You Share the Licensed Material (including in modified form)": identify the
  creator, keep copyright and licence notices, and "indicate if You modified the Licensed Material".
- ShareAlike (3(b)) applies "if You Share Adapted Material You produce": the adapter's licence must be a
  Creative Commons licence "with the same License Elements, this version or later" (or a BY-NC-SA compatible
  one), and You "may not offer or impose any additional or different terms or conditions" that restrict it.
- 2(a)(2) **says**: "where Exceptions and Limitations apply to Your use, this Public License does not apply".
- The CC FAQ (partly relayed as a summary, not verbatim-checked) says whether a use is commercial "will
  depend on the specifics of the situation and the intentions of the user", and, in a line read directly,
  "if you're not sure if your intended use is permitted by the license, you should contact the rights holder".
  The FAQ pages read do not directly address private, non-public copying.
- **Reading (not a finding):** the attribution and ShareAlike duties are triggered by *Sharing*, which the
  licence defines around providing material *to the public*. Whether storing files in a private repository
  at a hosting company is "Sharing", whether a fitted model is "Adapted Material", and whether a portfolio
  demo is "NonCommercial" are **not answered by the text above**. They are open questions, not conclusions.

### 3. Hugging Face terms (Relayed; all pages loaded)

- Terms of Service (effective 2022-09-15) **say**: "You are solely responsible for the Content you post"; and
  "You represent and warrant that you have ownership, control, and responsibility for the Content you post".
  Paraphrase: the user grants Hugging Face a worldwide, royalty-free, non-exclusive licence to use,
  reproduce, distribute and make derivative works of the content to provide the services.
- On private repositories the Terms **say**: "we will use reasonable and appropriate measures designed to
  keep your Content confidential", and that Hugging Face "may access or share your private information
  pursuant to the terms set forth in our Privacy Policy". "Your Content must not be misleading or unlawful."
- Content Policy (effective 2025-04-10) lists, among restricted content, "Content that infringes the
  intellectual property rights of a third party." It defines Private as "only you or members of your
  organization can see and make changes to the Repository". It does not say whether private repositories are
  moderated differently.
- Storage (docs, relayed): free account private storage "100GB"; no per-repo limit for models; the two
  packages total 242 MB. Fine-grained tokens exist and a read-only fine-grained token for one model is shown
  in an example. **That a fine-grained token can be limited to write access on one repository is not
  confirmed by the page read.** It remains **unverified** until the owner sees the token screen.
- **Reading:** a private repository reduces who can see the files. It does not change what rights the owner
  has in them; the Terms put that responsibility (and a warranty about it) on the uploader.
  The Terms' licence to Hugging Face is a further question the owner may wish to weigh against CC's wording.

### 4. The official MVTec page (Unverified; could not be read, again)

<https://www.mvtec.com/research-teaching/datasets/mvtec-ad> failed twice with "socket hang up" in this
review (earlier attempts also failed). Nothing in the project documents about that page is first-hand. The
statement in `DATA_LICENSE.md` that the page states the licence terms is still the owner's relay of an AI
assistant's report. The bundled `readme.txt` (1) is the substitute evidence.

### 5. ImageNet terms and the backbone weights (A7)

- The two Hugging Face model cards (`timm/resnet18.tv_in1k`, `timm/wide_resnet50_2.tv_in1k`; relayed) show
  `License: bsd-3-clause` and "Trained on ImageNet-1k, original torchvision model weight." Neither contains
  ImageNet terms.
- The torchvision models page (relayed, followed to `docs.pytorch.org`) **says**: "The pre-trained models
  provided in this library may have their own licenses or terms and conditions derived from the dataset used
  for training." and "It is your responsibility to determine whether you have permission to use the models
  for your use case."
- ImageNet's own terms pages (`image-net.org/download*`) refused the connection. **Unverified.** The existing
  wording in `docs/data.md` and the notices ("not verified, not inferred from the card") remains accurate.

### 6. The Kaggle source (A17) (Unverified)

Of two guessed Kaggle URLs, one returned 404 and one returned only a page title with no licence or owner.
The owner's actual Kaggle link is not recorded in the project, so Claude could not look at the right page.
Whether the Kaggle copy equals MVTec's official archive remains **unverified**.

## What is in the repository now (Verified, first-hand)

- **No `LICENSE` file** (`git ls-files`). `README.md`, `DATA_LICENSE.md` and the notices say MIT is planned
  and not in force. GitHub's API shows licence `NOASSERTION` ("Other") for the public repository (reason not
  investigated).
- **Tracked files that derive from MVTec images** (four PNGs, all in the commit `815d024` of 2026-11-10):
  - `evaluation/wide_resnet50_2_tv_in1k/v1.0.0/bottle/figures/true_positives.png`, `false_positives.png`
  - `evaluation/resnet18_tv_in1k/v1.0.0/bottle/figures/true_positives.png`, `false_positives.png`
  Their companion JSON/CSV/log files hold file names, numbers and hashes only (per the project's notices; the
  file names refer to test-split images; the images themselves were not opened here). The figure selection
  rule (`figures/index.json`) picks false positives, false negatives and example detections, which are
  test-split images (**reading**, from the rule and from the notice, not from opening them).
- **Are they on the public remote? Yes (Verified).** The remote `shoaibhassancs/industrial-defect-detection-system`
  is public (GitHub API: `private: false`). Its `main` head is `ef269a3`, equal to local `HEAD`; the commit
  `815d024` is an ancestor of `origin/main`; the local record of the remote-tracking branch shows it arrived
  by a push on 2026-11-10 (10:01 +0500); and an unauthenticated request for each of the four PNGs on
  `raw.githubusercontent.com` returned **HTTP 200**. Git history keeps them even if a later commit removes them.
- **Notices present:** `evaluation/NOTICE.md` (attribution, CC BY-NC-SA terms, "adaptations", modified
  statement, Kaggle caveat, no endorsement), `DATA_LICENSE.md`, `docs/model-card.md`, and a `NOTICE.md` inside
  each model package. The two package notices differ only in the backbone tag (checked with `diff`).
- **Consistency check of the existing texts (Verified, by reading):**
  - Consistent: the licence name, the copyright line, the citations, the "not legal advice" wording, the
    Kaggle and ImageNet caveats, the ShareAlike-as-precaution policy for the weights (`DATA_LICENSE.md`, both
    package notices, `docs/model-card.md`, `artifacts.lock.json`, ADR-0003).
  - **Out of date:** `docs/data.md` section "Processed and derived images" says "So far, nothing of that kind
    has been published; the only such files are local and git-ignored." That is no longer true: the four
    figures are tracked and public. `DATA_LICENSE.md` says "Whether these figures are published is the owner's
    decision; if they are, they must keep this notice"; they have been published, and the notice is present.
    Not edited (the owner asked for no edits to licence or notice text); a correction is proposed below.
  - `docs/data.md` and `DATA_LICENSE.md` correctly describe the MVTec page and Kaggle as not read first-hand.
    That is still the case after this review.

## Options and consequences (no recommendation to decide for you)

**Item 1: `LICENSE` file.** (a) Add MIT for original code only, as planned: gives others clear code terms;
it must not claim to cover the dataset, the figures or the weights, and cannot be withdrawn for copies
already taken. (b) Wait: until then nobody has an explicit code licence (as the README already says).
(c) A different licence. Whichever, third-party and dataset-derived material keeps its own terms.

**Item 2: the public figures.** (a) Leave as is, with the notice: the existing notice is the project's
attempt at CC attribution and modification marking; whether it is sufficient is not assessed here, and
the ShareAlike wording for adapted images is the project's own reading. (b) Strengthen the notices
(per-figure caption or overlay text; correct `docs/data.md`): low cost, no history change. (c) Remove the
figures from the current tree: stops new views from the branch tip but **does not remove them from Git
history or from copies already taken**; rewriting history is a separate, drastic Git action that Claude must
not take without explicit instruction, and it may not remove cached or forked copies. (d) Replace them with
figures that contain no MVTec pixels (for example score plots only): changes the evidence on the Results page.
(e) Ask MVTec (the readme gives contact addresses) for confirmation. Consequences of (a)-(e) on the Results
page and on `docs/evaluation.md`, which refer to the figures, would need checking before any removal.

**Item 3 and 4.** Options: leave "unverified" in all texts (current state); the owner reads the pages
personally in a browser and tells Claude what they say; or replace the local copy with MVTec's official
archive and compare hashes (large download; disk is 3.3 GiB free, so not now).

**Item 5: uploading the weights.** Nothing in the sources read establishes that it is permitted, and nothing
establishes that it is forbidden. Options: (a) do not upload; keep packages local and reproducible with the
scripts (consequence: a Render deployment has no way to fetch them; Phase 5/7 would need another plan).
(b) Upload privately after the owner accepts the risk under the Hugging Face warranty and the CC reading above
(consequence: the owner, not the private setting, bears the responsibility; the upload is not reversible in
the sense that Hugging Face may keep history and that its own licence grant applies). (c) Ask MVTec first, then
decide. (d) Upload only the smaller `resnet18` package as a trial (still the same question). (e) Use another
private store (same question, different terms). If any option is chosen, the plan fingerprint
`0bbc74ae…7513` and the separate approval in CLAUDE.md still apply; this review does not approve it.

**Item 6: public use of the weights.** A Render demo that serves the model to anyone is a public service
running a model derived (by the project's own policy) from CC BY-NC-SA material. Whether a portfolio demo is
"NonCommercial" in the licence's sense is a question the CC FAQ leaves to the situation and intent; it is not
settled here. Options: keep the demo local-only; deploy with notices and the non-commercial statement; or
seek MVTec's confirmation first. This also decides whether a Docker image containing the weights would be
published.

## Owner's decisions after this review (2026-11-11)

- **A1 (done):** `docs/data.md` ("Processed and derived images") and `DATA_LICENSE.md` ("Derived material") now
  say the four evaluation figures are already public (tracked in Git, on the public GitHub repository since
  2026-11-10, checked 2026-11-11) and carry `evaluation/NOTICE.md`, and that whether the notice is sufficient
  is unresolved. Nothing else in those files changed. This is a factual correction, not a legal position.
- **A2 (option 5 chosen):** the owner will ask MVTec. A draft question was prepared for the owner to send
  (kept in the private `project-notes/mvtec-question-draft.md`); Claude does not contact MVTec. The figures
  are **not** removed or replaced; the other options stay open.
- **B (deferred):** no Hugging Face repository, token or upload. The licence questions for the weights and for a
  public demo stay **unresolved** until the owner has reviewed these findings and, if they choose, received
  guidance from MVTec. The Kaggle-source (A17) and ImageNet-terms (A7) questions stay **unverified**.
- ADR-0003, `docs/model-card.md` and the package notices need no change.

## Not done / still unverified

MVTec's official page; ImageNet's terms; the Kaggle page and its licence; whether the Kaggle copy equals the
official archive; whether a fitted model is "Adapted Material" or a private upload is "Sharing" under the
licence; Hugging Face's current terms beyond the pages read; fine-grained token scoping; GitHub's own terms
for the public repository (not read); anything about law outside the documents. The quotes marked "relayed"
come from helper fetches and were not re-read by Claude.
