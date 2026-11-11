# Shortcuts for common tasks. Targets are added only when they have something to run.
.PHONY: setup lint format test fit freeze calibrate evaluate figures package infer api-contract evidence frontend-install frontend-check

setup:  ## Install all workspace packages and dev tools into .venv
	uv sync --all-packages

lint:  ## Check code style and likely mistakes (does not change files)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix import order and apply the formatter
	uv run ruff check --fix .
	uv run ruff format .

test:  ## Run the test suite
	uv run pytest

# --- the model pipeline (Phase 2 to 4.5) -------------------------------------------------------
# Run in this order, one model at a time:  make fit freeze calibrate package
# and, once and deliberately, the sealed test steps:  make evaluate figures CONFIRM_OPEN_TEST_SET=yes
# Every step refuses to overwrite an earlier result; a repeat needs a new VERSION or moved-aside
# outputs. They need the dataset under data/raw/mvtec-ad/ (see docs/data.md) and a calm machine
# (they check swap and free disk and stop at their limits). Override the models or the version:
#   make fit CONFIGS=ml/configs/bottle_resnet18.yaml     make freeze VERSION=1.0.1
CONFIGS ?= ml/configs/bottle_wrn50.yaml ml/configs/bottle_resnet18.yaml
VERSION ?= 1.0.0

fit:  ## Fit each model on the `fit` images only (heavy: minutes per model)
	for c in $(CONFIGS); do uv run python -m defect_ml.fit $$c || exit $$?; done

freeze:  ## Freeze each fitted model (copy + lock; opens no dataset file)
	for c in $(CONFIGS); do uv run python -m defect_ml.freeze $$c --version $(VERSION) || exit $$?; done

calibrate:  ## Calibrate each frozen model on the 42 `calib` images only
	for c in $(CONFIGS); do uv run python -m defect_ml.calibrate $$c --version $(VERSION) || exit $$?; done

# The next two open the SEALED test set (logged in evaluation/test_access_log.jsonl). Opening it is a
# deliberate, one-time act per version, so they stop unless you confirm in the same command.
SEALED_GUARD = test "$(CONFIRM_OPEN_TEST_SET)" = yes || { \
	echo "REFUSING: this opens the sealed MVTec test set (once per version, logged)."; \
	echo "Add CONFIRM_OPEN_TEST_SET=yes to the command if that is what you want."; exit 1; }

evaluate:  ## Sealed evaluation of all CONFIGS in one batch (needs CONFIRM_OPEN_TEST_SET=yes)
	@$(SEALED_GUARD)
	uv run python -m defect_ml.evaluate $(CONFIGS) --version $(VERSION)

figures:  ## Example figures from a finished evaluation (re-reads test files; needs the same flag)
	@$(SEALED_GUARD)
	uv run python -m defect_ml.visualize $(CONFIGS) --version $(VERSION)

package:  ## Package each calibrated, evaluated model (reads only `calib` images; no upload)
	uv run python -m defect_ml.package $(CONFIGS) --version $(VERSION)

infer:  ## Inspect one image: make infer ARTIFACT=<package folder> IMAGE=<file> [OUT=<folder>]
	@test -n "$(ARTIFACT)" -a -n "$(IMAGE)" || { echo "usage: make infer ARTIFACT=<package folder> IMAGE=<image file> [OUT=<folder>]"; exit 1; }
	uv run python -m defect_core.infer --artifact "$(ARTIFACT)" "$(IMAGE)" $(if $(OUT),--out "$(OUT)")

# --- the web app (Phase 6) ---------------------------------------------------------------------
api-contract:  ## Re-export the API description and regenerate the frontend's types (no server, no model)
	cd backend && uv run python -m app.openapi_export ../frontend/openapi.json
	cd frontend && npm run gen:api

evidence:  ## Re-make the Home page's evidence snapshot from the saved metrics.json (no server, no model)
	cd frontend && npm run gen:evidence

frontend-install:  ## Install the frontend packages exactly as locked (needs network; uses ~0.2 GB of disk)
	cd frontend && npm ci

frontend-check:  ## Frontend format check, lint, type-check, tests and production build
	cd frontend && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build
