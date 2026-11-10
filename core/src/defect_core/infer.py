"""Inspect one image with a verified artifact (Phase 4.5): the shared inference path and a CLI.

    uv run python -m defect_core.infer --artifact artifacts/<package folder> path/to/image.png
    uv run python -m defect_core.infer --artifact <folder> image.png --out results/ --json

``inspect_image`` is the one function that turns an image into a result: it decodes and prepares
the image with ``defect_core.preprocessing``, scores it with ``defect_core.model.infer`` and applies
the artifact's stored calibration (verdict, rank among the calibration normals, heatmap and outline
levels). The command line below is a thin wrapper around it; the API is meant to use the same
function, so the two can not drift apart.

What the command does, in order: loads and **verifies** the artifact (a damaged, edited or
mismatched artifact stops here with a plain message, and nothing is scored); needs a calibrated or
packaged artifact (a fitted or frozen-only one has no threshold and is refused); reads the image
(any format Pillow can decode; EXIF orientation applied, transparency composited on white); scores
it on CPU; prints the result. It makes no network access and writes nothing unless ``--out`` is
given.

**Output files (only with ``--out DIR``).** ``<name>_heatmap.png`` is the fixed-scale heatmap and
``<name>_overlay.png`` the heatmap over the image with the highlight outline, both at the original
image size, drawn by ``defect_core.render`` with the artifact's calibration levels (the heatmap
scale is fixed by calibration, never rescaled per image). ``<name>`` is the input file's name
without its extension. The folder is created if needed; an existing output file is **never
overwritten** (the command stops before scoring anything). With ``--json`` the result is printed as
JSON instead of text.

**What the result means.** The verdict is a *demo decision*: "anomalous" if ``score > threshold``
(strictly), otherwise "normal". The score is an anomaly score, not a probability or a confidence.
The rank sentence says how many of the *n* calibration normal images the score is higher than. The
model only makes sense for images like the category it was fitted on and **cannot reliably detect
that an image is out of scope**: for such an image the score is arbitrary. "Pass" and "fail" are
never used.

Exit codes: 0 a result was printed; 2 the artifact, the image or the output location was refused.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from defect_core.artifacts import ArtifactError, LoadedArtifact, load_artifact
from defect_core.model import infer
from defect_core.preprocessing import ImageLoadError, load_image, to_tensor
from defect_core.render import RenderResult, RenderSettings, render_result

DISCLAIMER = "Demo decision based only on this model and category. Not a quality certification."


class InferenceError(Exception):
    """The artifact can not be used for inspection, or the request is refused."""


@dataclass(frozen=True)
class InspectionResult:
    verdict: str  # "anomalous" or "normal"
    score: float
    threshold: float
    rank_m: int  # the score is strictly higher than this many calibration normals ...
    rank_n: int  # ... of this many
    decision_rule: str
    highlighted_area_fraction: float
    region_bbox: tuple[int, int, int, int] | None  # (x, y, width, height), original pixels
    image_size: tuple[int, int]  # (width, height)
    category: str
    model_name: str
    model_version: str
    limitation: str
    inference_ms: float
    anomaly_map: np.ndarray  # raw 256 x 256 map
    rendering: RenderResult  # heatmap and overlay at the original image size

    def to_dict(self) -> dict[str, object]:
        """The result as plain data (no images), in the field names of the planned API."""
        return {
            "category": self.category,
            "model": {"name": self.model_name, "version": self.model_version},
            "prediction": self.verdict,
            "decision_rule": self.decision_rule,
            "anomaly_score": self.score,
            "threshold": self.threshold,
            "calibration_rank": {
                "score_higher_than": self.rank_m,
                "of_calibration_normals": self.rank_n,
            },
            "highlighted_area_fraction": self.highlighted_area_fraction,
            "region_bbox": list(self.region_bbox) if self.region_bbox else None,
            "image": {"width": self.image_size[0], "height": self.image_size[1]},
            "timing_ms": {"inference": round(self.inference_ms, 1)},
            "limitation": self.limitation,
            "disclaimer": DISCLAIMER,
        }


def _info(loaded: LoadedArtifact) -> tuple[str, str, str]:
    manifest = loaded.manifest
    package = manifest.get("package") or {}
    provenance = manifest.get("provenance") or {}
    category = str(package.get("category") or provenance.get("category") or "unknown")
    name = str(
        package.get("model_name") or provenance.get("run_name") or loaded.model_config.backbone
    )
    version = str((manifest.get("freeze") or {}).get("version", "unversioned"))
    return category, name, version


def ordinal(number: int) -> str:
    """1 -> "1st", 2 -> "2nd", 3 -> "3rd", 11 -> "11th", 41 -> "41st"."""
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


def decision_rule(loaded: LoadedArtifact) -> str:
    c = loaded.calibration
    assert c is not None
    return (
        f"anomalous if anomaly_score > threshold; threshold = {ordinal(c.k)} smallest of {c.n} "
        f"calibration normal scores (split-conformal, target alpha={c.alpha:g}; marginal "
        "guarantee under exchangeability only)"
    )


def inspect_image(
    loaded: LoadedArtifact,
    source: Image.Image | str | Path | bytes,
    render_settings: RenderSettings | None = None,
) -> InspectionResult:
    """Score one image with a loaded, verified artifact and apply its calibration.

    ``source`` is a path, the bytes of an image file, or a PIL image. Needs a calibrated or packaged
    artifact. Raises ``InferenceError`` for an artifact without calibration and ``ImageLoadError``
    for data that is not a supported image.
    """
    calibration = loaded.calibration
    if calibration is None:
        stage = loaded.manifest.get("stage", "fitted")
        raise InferenceError(
            f"this artifact is only {stage}; it has no calibration (no threshold). "
            "Use a calibrated or packaged artifact"
        )
    image = source if isinstance(source, Image.Image) else load_image(source)
    if image.mode != "RGB":
        image = load_image_from_pil(image)
    started = time.perf_counter()
    score, anomaly_map = infer(loaded.model, to_tensor(image, loaded.preprocess_config)[None])
    inference_ms = (time.perf_counter() - started) * 1000
    value = float(score[0])
    raw_map = anomaly_map[0, 0].numpy()
    rendering = render_result(image, raw_map, calibration.pixel, render_settings)
    category, name, version = _info(loaded)
    return InspectionResult(
        verdict="anomalous" if calibration.is_anomalous(value) else "normal",
        score=value,
        threshold=calibration.threshold,
        rank_m=calibration.normal_rank(value),
        rank_n=calibration.n,
        decision_rule=decision_rule(loaded),
        highlighted_area_fraction=rendering.area_fraction,
        region_bbox=rendering.bbox,
        image_size=image.size,
        category=category,
        model_name=name,
        model_version=version,
        limitation=f"Results are only meaningful for {category} images captured like the "
        "MVTec AD dataset.",
        inference_ms=inference_ms,
        anomaly_map=raw_map,
        rendering=rendering,
    )


def load_image_from_pil(image: Image.Image) -> Image.Image:
    """Bring a PIL image of another mode to RGB the same way ``load_image`` does for files."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return load_image(buffer.getvalue())


def format_report(
    result: InspectionResult, source: str = "", saved: list[Path] | None = None
) -> str:
    """The result as plain text. Never uses the words pass or fail."""
    label = "Anomalous" if result.verdict == "anomalous" else "Normal"
    width, height = result.image_size
    lines = [
        "Inspection result (demo decision, not a quality certification)",
        f"  Image:     {source or 'image'} ({width} x {height})",
        f"  Model:     {result.model_name}, version {result.model_version}",
        f"  Verdict:   {label} - demo decision",
        f"  Score:     {result.score:.4f}   Threshold: {result.threshold:.4f}",
        f"  Rank:      score higher than {result.rank_m} of the {result.rank_n} calibration normal "
        "images",
        "             (a rank among the calibration normals, not a probability or a confidence)",
    ]
    if result.region_bbox is not None:
        x, y, w, h = result.region_bbox
        region = f"largest region at x={x}, y={y}, {w} x {h} pixels"
    else:
        region = "no region above the highlight level"
    lines.append(
        f"  Highlight: {100 * result.highlighted_area_fraction:.1f}% of the image above the "
        f"highlight level; {region}"
    )
    lines += [
        f"  Rule:      {result.decision_rule}",
        f"  Note:      {result.limitation}",
        "             The model can not reliably tell that an image is out of scope.",
        f"  Time:      {result.inference_ms:.0f} ms (scoring only, CPU)",
    ]
    for path in saved or []:
        lines.append(f"  Saved:     {path}")
    lines.append(f"  {DISCLAIMER}")
    return "\n".join(lines)


# --- command line ----------------------------------------------------------------------------


def _output_paths(out_dir: Path, image_path: Path) -> tuple[Path, Path]:
    return out_dir / f"{image_path.stem}_heatmap.png", out_dir / f"{image_path.stem}_overlay.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m defect_core.infer",
        description="Inspect one image with a verified model artifact (a demo decision).",
    )
    parser.add_argument("image", help="path to the image to inspect")
    parser.add_argument(
        "--artifact", required=True, help="a calibrated or packaged artifact folder"
    )
    parser.add_argument("--out", help="folder for <name>_heatmap.png and <name>_overlay.png")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    try:
        saved_paths: tuple[Path, Path] | None = None
        if args.out:
            saved_paths = _output_paths(Path(args.out), image_path)
            existing = [p for p in saved_paths if p.exists()]
            if existing:
                raise InferenceError(f"{existing[0]} already exists; it is never overwritten")
        loaded = load_artifact(args.artifact)  # verifies everything before any scoring
        result = inspect_image(loaded, image_path)
    except FileNotFoundError as error:
        print(f"ERROR: file not found: {error.filename or error}", file=sys.stderr)
        return 2
    except (ArtifactError, InferenceError, ImageLoadError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    saved: list[Path] = []
    if saved_paths:
        saved_paths[0].parent.mkdir(parents=True, exist_ok=True)
        arrays = (result.rendering.heatmap, result.rendering.overlay)
        for path, array in zip(saved_paths, arrays, strict=True):
            Image.fromarray(array).save(path)
            saved.append(path)
    if args.json:
        payload = result.to_dict()
        payload["saved_files"] = [str(p) for p in saved]
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(result, str(image_path), saved))
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
