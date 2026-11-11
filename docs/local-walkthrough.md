# Local walkthrough: run it and show it

There is no hosted demo ([deployment-readiness.md](deployment-readiness.md), section 6a). This page is the short
guide to running DefectLens on your own machine and to recording or screenshotting it. It does not repeat the
setup steps; it points to them.

**Status: unverified end to end.** The steps below come from the README and Makefile. This page was written
without starting the server, so the sequence has not been re-run as written.

## 1. What you need first

- Setup, the model packages and the dataset: see [README, "Set up and check"](../README.md#set-up-and-check).
- The weights are **not** in Git and **not** uploaded anywhere. A model package exists only if you built it from
  your own copy of the MVTec AD `bottle` data with the `make` pipeline (`fit`, `freeze`, `calibrate`, `package`;
  see [evaluation.md, section 8](evaluation.md)). Without a package folder in `artifacts/`, the app cannot score.
- Dataset terms and attribution: [DATA_LICENSE.md](../DATA_LICENSE.md).

## 2. Start it

Follow [README, "Run the web app locally"](../README.md#run-the-web-app-locally): the API in one terminal, the web
app in another, then open http://localhost:5173.

## 3. Recording and screenshot checklist (60 to 90 seconds)

1. **Home**: show the page (it makes no API calls).
2. **Demo**: choose the calibration example `data/raw/mvtec-ad/bottle/train/good/002.png`, press "Inspect image",
   and show the result: decision, score next to the threshold, the three images.
3. **Results**: show the saved evaluation numbers and the false-alarm discussion.
4. **About**: show the limits.

**Caption to show on screen or beside every screenshot:**
"Calibration example (MVTec AD training image), used only to show the app workflow. This is not evidence of
detection accuracy. Accuracy evidence is the saved test-set evaluation (docs/evaluation.md)."

## 4. Rules for the material you make

- Use only the calibration image above for the Demo step. Do not use test images or masks, synthetic images, or
  any other calibration image as evidence of accuracy.
- Whatever verdict appears for that image proves nothing about performance: its score is one of those that set
  the threshold.
- The image is MVTec AD material (CC BY-NC-SA 4.0 per the dataset's own `readme.txt`). Keep the attribution with
  the recording. Whether publishing it needs anything more is **unresolved**: see
  [licensing-review.md](licensing-review.md).

## 5. What each kind of evidence shows

| Evidence | Shows | Does not show |
|---|---|---|
| Saved evaluation ([evaluation.md](evaluation.md)) | Detection results on the held-out `bottle` test set | That the app runs for you |
| Recording or screenshots | The app workflow and interface | Detection accuracy |
| These run steps | The project can be rebuilt by someone with the dataset | That it will work on your machine without trying |

## 6. Screenshots that exist (reviewed 2026-11-11)

Six PNG screenshots are in `screenshots/` (1412 px wide). **No screen recording exists.** The files keep the
names they were saved with; they were not renamed.

| File | Shows |
|---|---|
| `a.png` | Home page: headline evidence (63 of 63, 3 of 20 false alarms) and limits |
| `b.png` | Demo page before choosing a file (empty form, scope warning) |
| `c.png` | Demo page with `002.png` chosen and the result: "Normal", score 1.2446, threshold 1.3682, start of the overlay |
| `d.png` | Demo result: overlay image, scope note, collapsed explanation sections, footer |
| `e.png` | Results page: saved test-set numbers and the "read with care" notes |
| `f.png` | About page: dataset, licence and non-endorsement text |

**Review result (looked at by Claude, 2026-11-11):**
- No test image, mask, token, personal detail or file path is visible. The only file name shown is `002.png`
  (no folder). The metadata of all six files holds only the dimensions and the comment "Screenshot".
- `c.png` and `d.png` show an MVTec training-style image. **The calibration caption in section 3 is not in these
  images.** It must be placed beside them whenever they are shown. The page footer in `d.png` carries the
  dataset and licence line (partly cut off at the bottom).
- In `c.png` a small grey "No file chosen" tooltip overlaps the result box (a capture artefact).
- `d.png` shows highlighted outlines on an image whose decision is "Normal". The page's own text says the highlight
  is a display aid, so show it with the caption.
- **Unverified:** that `002.png` in `c.png` and `d.png` is the calibration file
  (`train/good/002.png`). A file called `002.png` also exists in the sealed test folders, and the screenshots
  show only the name. I did not open any dataset file to check. This rests on your statement.
- `e.png` and `f.png` hold only text and numbers already in `docs/evaluation.md` and the About page.

**Owner decision (2026-11-11, later):** the screenshots are meant as **public portfolio evidence**, and
`screenshots/` is the intended **tracked** location. `.gitignore` was not changed (nothing needs to change for
that). Nothing was committed or pushed.

**README use:** the README shows only `a.png` (Home), `c.png` (Demo, with the calibration caption and the MVTec
attribution written directly under it) and `e.png` (Results). `b.png`, `d.png` and `f.png` stay in the folder.

**Owner review and acceptance (2026-11-11):** the owner viewed the existing screenshots on GitHub and judged
them good enough as they are. They were **not recaptured and not changed**: `c.png` still has the small
"No file chosen" tooltip and `d.png` still has the partly cut-off footer, and these are accepted as they
are. The README caption under `c.png` carries the calibration wording and the MVTec attribution. This
acceptance is the owner's judgement (reported to Claude; Claude did not view GitHub).

**Git status (checked 2026-11-11):** `screenshots/` is **untracked and not ignored**. A broad `git add` would
include it, and it would become public if pushed. `c.png` and `d.png` contain an adapted MVTec image, and
whether publishing that needs anything more is unresolved ([licensing-review.md](licensing-review.md)).
Nothing was moved or ignored; that is your decision.


## 7. Local run-through on 2026-11-11 (API and dev server checks)

Run on the MacBook Air M1 with the existing package `bottle_wide_resnet50_2_tv_in1k_v1.0.0_release` and the
README commands. Nothing was retrained, recalibrated or re-evaluated; no dependency was reinstalled.

| Check | Result |
|---|---|
| API starts (`uvicorn --factory app.main:create_app ...`) | PASS (ready after about 8 s) |
| `GET /api/v1/health` | PASS (HTTP 200, `{"status":"ok"}`) |
| `GET /api/v1/ready` | PASS (HTTP 200, model 1.0.0, category `bottle`) |
| One inspection of the calibration image `002.png` (training file) | PASS (HTTP 200 in about 0.5 s; prediction `normal`, score 1.2446 against threshold 1.3682, matching `c.png`) |
| Web dev server serves `/`, `/inspect`, `/model` | PASS (HTTP 200 each; page title "DefectLens") |
| CORS preflight from `http://localhost:5173` to `POST /api/v1/inspections` | PASS (HTTP 200) |
| Browser pass (layout, keyboard, file chooser, drop zone) | NOT RUN by Claude; the owner reports a browser check, details not recorded |

Notes: a backend and a dev server from an earlier run were already listening on ports 8000 and 5173; they were
stopped and a fresh API was started. The already-running dev server was used for the page checks (not
restarted). Both servers were stopped afterwards. This shows the services work together on this machine. It
is not evidence of detection accuracy (the only evidence for that is the saved evaluation), and it does not
cover other browsers, phones or screen readers.
