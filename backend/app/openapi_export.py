"""Write the API's OpenAPI description to a file, without starting a server or loading a model.

    uv run --package backend python -m app.openapi_export ../frontend/openapi.json

The frontend's TypeScript types are generated from this file (``npm run gen:api``), and a backend
test fails if the committed copy differs from what the app produces now.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.core.settings import Settings
from app.main import create_app


def openapi_text() -> str:
    """The description as stable, sorted JSON (the same text every time for the same code)."""
    settings = Settings(_env_file=None, app_env="test", log_level="WARNING")  # type: ignore[call-arg]
    return json.dumps(create_app(settings).openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.openapi_export OUTPUT.json", file=sys.stderr)
        return 2
    Path(argv[1]).write_text(openapi_text(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
