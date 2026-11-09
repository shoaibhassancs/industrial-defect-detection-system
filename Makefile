# Shortcuts for common tasks. Targets are added only when they have something to run.
.PHONY: setup lint format test

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
