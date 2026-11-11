# Deployment readiness review (Docker and Render)

- **Date checked:** 2026-11-11. **Review only:** nothing was built or run, no Docker image, no dependency or
  configuration change, no Hugging Face repository or token, no upload, and **no decision about a public
  demo**. The artifact-store (O3) and licensing choices stay open.
- **Labels:** **Verified** = read first-hand (a command on this machine, or an official page fetched from the
  shell and read as text on the date above). **Relayed** = quoted by a helper that fetched a page for Claude.
  **Unverified** = not read, not measured, or only inferred. **Owner decision** = yours to make.
  **Blocked** = depends on MVTec's reply or on your later artifact-store decision.
- Prices and limits on vendor pages change. Re-read the page before relying on a number.

## 1. Where the project stands (verified, local)

- **No Docker or Render files exist** (`git ls-files` finds no Dockerfile, `.dockerignore`, Compose or
  `render.yaml`). Phase 5 step 9 and the Phase 7 deployment are not started.
- **The backend is ready to be containerised in principle:** a FastAPI app started with
  `uvicorn --factory app.main:create_app`, settings from environment variables (`backend/app/core/settings.py`:
  `model_dir`, `evaluation_dir`, `allowed_origins`, `max_upload_mb` default 10 and at most 50, `rate_limit`
  default `10/minute`, `trusted_proxy_hops` default 0, `torch_num_threads` default 1). `GET /health` always
  answers `ok`; `GET /ready` answers 503 with `Retry-After: 10` until the model is loaded.
- **Torch comes from PyPI on every platform.** No `[tool.uv.index]` or torch source is configured in any
  `pyproject.toml`, and `uv.lock` has no `download.pytorch.org` entry. The lock pins torch 2.15.0 and lists
  NVIDIA/CUDA packages (`nvidia-*`) for Linux. Entry 035 of `decisions.md` recorded an estimate from the lock
  of about 2.45 GB of NVIDIA wheels plus a torch wheel of about 539 MB for Linux x86_64 (estimated, never
  measured here). So a Linux image built from the lock as it is would carry GPU libraries a CPU service
  never uses. **The size of the resulting image is unmeasured and unverified.**
- **The model is not in Git** (`artifacts/` and `*.safetensors` are ignored). Render builds from a Git
  repository or pulls an image from a registry (section 3), so the weights need a separate route into the
  container. That is exactly the artifact-store question, and it is **open** (section 5).

## 2. Measured numbers so far (native macOS proxies, not container or Render measurements)

From `decisions.md` entry 058 (serving-style benchmark of the fitted packages) and entry 071 (the real local smoke test of the backend); Mac M1, 1 thread unless said (entry 052 holds the earlier synthetic F1 run):

| Fact | `resnet18` | `wide_resnet50_2` (headline) |
|---|---|---|
| Resident memory after `import defect_core` (torch + anomalib), before any model | ~505 MB | ~521 MB |
| After loading the package (load time) | ~597 MB (0.36 s) | ~706 MB (2.06 s) |
| Peak resident memory of the whole process (benchmark) | ~988-1,003 MB | ~1,042-1,102 MB |
| End-to-end latency, median / 95th percentile | 131 / 144 ms | 272 / 297 ms |
| Real server, wide package: ready after | n/a | ~9 s (load, verify, warm-up) |
| Real server, wide package: resident memory | n/a | ~752 MB after warm-up, ~898 MB after three requests |
| Real server: one inspection over HTTP | n/a | ~0.27-0.31 s |

They are **proxies**: macOS differs from a Linux container, the Mac was busy and swapping, and `ru_maxrss`
excludes compressed or swapped memory. They are not hosting measurements and were never meant to choose a plan.

## 3. What Render's official documentation says (read 2026-11-11)

Pages read as text from the shell: `render.com/docs/free`, `/pricing`, `/docker`, `/web-services`,
`/health-checks`, `/environment-variables`, `/deploys`, plus `/inbound-ip-rules`, `/websocket`, `/faq`,
`/troubleshooting-deploys` and the docs index `/docs/llms.txt`. (The helper agent could not resolve the host;
the shell could.) Short fragments are quoted; everything else is paraphrase.

**Compute sizes and price (pricing page).** Free: "$0/month", "512 MB RAM", "Less than 1 CPU". A "$7/month"
plan also lists "512 MB RAM"; a "$25/month" plan lists "2 GB RAM" (plan id `1c-2g`); "$85/month" lists "4 GB
RAM". The extracted table does not reliably align the CPU counts, so CPU is **not** reported here; the plan
ids (for example `1c-2g`) suggest CPU counts. Re-read the live table before choosing. **Verified as page
text; plan names and CPU columns not verified.**

**Free web service limits (free page).**
- "Do not use them for production applications."
- It "spins down" after "15 minutes without receiving any inbound traffic"; spin-up "takes about one minute"
  and a loading page is shown to browsers meanwhile.
- When it spins down, local filesystem changes are lost (ephemeral filesystem).
- "750 Free instance hours" per month; "Render might restart a Free web service at any time."
- A persistent disk is not available on Free (the page lists it among features paid plans have).

**Port and health (web-services, health-checks).**
- A web service must bind to `0.0.0.0` and the port in `PORT` (default `10000`). If Render cannot detect a
  bound port, the deploy fails.
- Health check: an HTTP GET path you set (`healthCheckPath`); the TCP variant "accepts the attempted TCP
  connection within five seconds"; for HTTP the page asks for a successful status "within five seconds" (the
  sentence was cut in extraction; read the live page). A new deploy must pass "within 15 minutes" or it is
  cancelled; a running instance failing for 15 seconds stops receiving traffic, and "60 seconds" leads to an
  automatic restart. The project's `/health` answers instantly and `/ready` returns 503 while loading, so
  `/health` is the natural path (a design note, not tested on Render).

**Docker deploys (docker page).**
- Render builds from a Dockerfile with BuildKit, caches intermediate layers, stores the image in "a private,
  secure container registry", honours `.dockerignore`, and can instead pull a prebuilt image from a registry.
- "do not reference any build arguments that contain sensitive values"; sensitive values at build time
  should use a secret file, which Render copies "into the root of your build context" (keep it out of the
  image with `.dockerignore`).
- Environment variables "are available at both build time and runtime" unless noted
  (environment-variables page); `RENDER_CPU_COUNT` is among the variables Render sets.
- The deploys page lists a "Build command" timeout of "120 minutes", "Pre-deploy" 30 and "Start" 15. It is
  written for build/start commands and was **not confirmed to apply to Dockerfile builds**.

**Not found in any page read (so unverified):**
- how many proxy hops Render adds and which header carries the client address (the open item behind
  `trusted_proxy_hops` and `backend/app/core/clientip.py`); the search covered the pages above and the docs
  index and found no mention of `X-Forwarded-For` or a client-address header;
- the maximum request body size and the request/response timeout of the front proxy (the app itself allows
  uploads up to `max_upload_mb`);
- the memory limit's behaviour when exceeded (restart or kill) and build-memory limits for Dockerfile builds;
- the free tier's CPU in numbers.

## 4. What uv and PyTorch documentation say (relayed; fetched 2026-11-11 by a helper)

- **uv's PyTorch guide** (<https://docs.astral.sh/uv/guides/integration/pytorch/>, page dated 2026-10-14)
  shows a CPU index: `[[tool.uv.index]]` with `name = "pytorch-cpu"`, `url =
  "https://download.pytorch.org/whl/cpu"`, `explicit = true`, and `torch`/`torchvision` entries in
  `[tool.uv.sources]` pointing at it. It recommends `explicit = true` "to ensure that the index is only used
  for" the named packages, and shows platform markers (`sys_platform == 'linux'`) so that macOS "fall[s] back
  to PyPI". It says some features "require uv version 0.5.3 or later". **It does not explain how `uv.lock`
  records marker-gated sources, and it does not cover `anomalib[cpu]`'s own torch requirement.** How the
  change would interact with this project's lock and with CI's `UV_LOCKED=1` is **unverified** and can only
  be learned by trying it (it changes `pyproject.toml` and `uv.lock`, so it needs your approval).
- **uv's Docker guide** (<https://docs.astral.sh/uv/guides/integration/docker/>, dated 2026-11-03):
  pin the uv version ("best practice to pin to a specific uv version"); `uv sync --locked` "asserting the
  lockfile is up to date"; a first layer with `uv sync --frozen --no-install-project` so dependencies cache
  separately from code; a cache mount; `UV_COMPILE_BYTECODE=1`; a multi-stage build with `--no-editable`
  copying only `.venv` into the final stage; add `.venv` to `.dockerignore`. It uses `UV_NO_DEV=1`.
- **uv workspaces** (<https://docs.astral.sh/uv/concepts/projects/workspaces/>): `--package <member>` runs
  or syncs one member; `uv run`/`uv sync` default to the workspace root. This matters because the workspace
  root also holds `ml/` and scripts that a service image does not need; whether `uv sync --package backend`
  keeps the image small and correct is **unverified**.
- **pytorch.org "Get started"** (<https://pytorch.org/get-started/locally/>, "Last updated 2026-10-28"): the
  page's command selector is script-driven; no CPU-only command could be read, and the stated stable version
  looked inconsistent with the uv guide's. The CPU index URL is therefore confirmed only by the uv guide.
  Python 3.12 support of the current stable torch is **unverified** from this page (the project's lock and CI
  already resolve torch 2.15.0 on Python 3.12).

## 5. Options, and what each depends on

Nothing below is chosen. "Blocked" means the option cannot honestly proceed until the named decision exists.

| Option | What it means | Needs weights outside your machine? | Depends on | Disk/size now |
|---|---|---|---|---|
| **D0 Local only** | Demo runs on your Mac (as today); no public service | No | Nothing | Fine |
| **D1 Local Docker run** | Build the image on the Mac, mount `artifacts/` as a volume, never push the image | No | CPU torch decision; **disk** (3.3 GiB free; image size unmeasured) | **Not possible now** |
| **D2 Public Render service, weights from a private store** | Build or start fetches the pinned package with a read token | Yes | **Blocked** by the O3/Hugging Face decision (B, deferred) and by MVTec's answer on weights and on a public demo | Needs a paid plan for memory (below) |
| **D3 Public Render service from a pushed image containing the weights** | Pull an image from a registry | Yes (a published copy) | **Blocked** by the same decisions; the image is a distribution of the weights | Same |
| **D4 Public static site only (no live model)** | Results/About pages with saved evidence; the Demo page cannot score images | No weights | The figures question (A2) still applies to what is shown; static-host rewrite (Phase 7) | Small |
| **D5 Another host or registry** | Not reviewed | Yes | New terms to read | n/a |

**Memory, the one hard fact for Render (inference, not measured on Render):** the free instance has 512 MB;
the project's *import-only* footprint on macOS was ~505-520 MB and the loaded service ~750-900 MB. The gap is
large enough that the free instance is **very unlikely to hold the service**, with the caveat that macOS and
Linux numbers differ and no container measurement exists. A plan with 2 GB would be the first listed with room
(price as read today, "$25/month"; plan names unverified). This is open item O2 (is a paid plan acceptable).
`resnet18` does not materially change the picture (peak ~1.0 GB vs ~1.1 GB), so a smaller backbone is not a
way to reach 512 MB.

**Cold starts:** on Free, a request after 15 minutes idle takes "about one minute" per Render, on top of the
service's own load and warm-up (~9 s natively). A paid plan avoids spin-down (the free page describes it as a
Free-only limitation).

## 6. Decisions and checks for you, and what each unlocks

1. **O2, plan cost:** **decided 2026-11-11: no paid hosting; hosted demo deferred** (section 6a). Kept here for history.
2. **CPU-only torch:** whether to add the PyTorch CPU index with Linux markers. Cheap to try (`uv lock`, no
   install) but it changes `pyproject.toml` and `uv.lock`, so it needs a separate approval, and its effect on
   `anomalib[cpu]`, the lock and CI is unverified. (Owner decision, can proceed independently of MVTec.)
3. **Proxy hops:** unverified from documentation. Ways to settle it, each outside this review: ask Render
   support, or test on a deployed instance with a made-up `X-Forwarded-For` header. Until then keep
   `trusted_proxy_hops = 0`, which means everyone behind the proxy would share one rate-limit bucket (see
   `clientip.py`). Only relevant if a public service is ever built.
4. **Weights route and public demo:** **blocked** on your artifact-store decision (B) and on MVTec's reply to
   your three questions. Nothing here pre-empts them.
5. **Request size and timeouts on Render:** unverified; matters for 10 MB uploads and for slow cold starts.

## 6a. Owner's decision (2026-11-11): hosted model demo deferred

- **Decision:** for this portfolio version the owner will **not** build a hosted model demo unless an option is
  predictably free and suitable. No paid hosting. The app stays runnable locally. Hosted deployment is
  **deferred, not completed**.
- **Reason (from this review):** Render's free instance (512 MB) is very likely too small (inference, not
  measured on Render); Hugging Face Spaces that run compute need a paid plan (static Spaces are free); Cloud Run
  has a monthly free allowance but bills beyond it (card requirement unverified); Fly.io's free trial is $1 for 7
  days (pricing pages read 2026-11-11, not otherwise verified); a static host cannot run the model. Hosting the weights
  would also depend on the open store and MVTec licence questions.
- **Not blocked by MVTec:** local development continues regardless of MVTec's reply. Their reply matters only if
  weights or a public demo are ever published.
- **Proof of working instead:** (1) the saved sealed evaluation (the only accuracy evidence); (2) local run
  instructions; (3) screenshots and a short screen recording of the local app using an owned or clearly labelled
  demo image. A recording shows the **app workflow**, not detection accuracy. No synthetic or calibration image
  is used as accuracy evidence, and no MVTec test image is used in new screenshots or recordings.
- **Still open:** the CPU-only torch change, a static evidence-only site, and revisiting hosting later.

## 7. Not done / not verified

No Dockerfile was written or built; no image size, build time, container memory or Linux latency was measured;
Render's proxy hops, request limits and Dockerfile build limits are unverified; the CPU-torch change and its
effect on the lock and CI are untried; the PyTorch install page could not be read for CPU commands; vendor
pages may change; the free-tier CPU and the paid plans' names and CPU counts were not reliably read.
