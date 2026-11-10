"""Export matplotlib's `inferno` colormap as a 256-entry lookup table (8-bit RGB).

The table is committed as `core/src/defect_core/inferno_lut.json`, so `defect_core` and the backend
do not need matplotlib (PLAN.md section 3.1). Run this only to regenerate the file:

    uv run python scripts/export_colormap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "core/src/defect_core/inferno_lut.json"


def main() -> None:
    rgba = matplotlib.colormaps["inferno"](np.arange(256))  # the 256 table entries, floats 0..1
    table = np.rint(rgba[:, :3] * 255).astype(int).tolist()
    OUT.write_text(
        json.dumps(
            {"source": f"matplotlib {matplotlib.__version__} inferno", "rgb": table},
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(table)} entries)")


if __name__ == "__main__":
    main()
