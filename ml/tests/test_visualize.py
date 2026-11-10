"""Tests for defect_ml.visualize (Phase 3 step 7): the fixed selection rule, the size analysis, and
the guarantees of the re-scoring (sealed path, logged, chain kept, scores identical).

The integration tests reuse the fake dataset and the random-weight evaluation of test_evaluate.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from defect_ml import evaluate, visualize
from defect_ml.audit import OpenAudit
from defect_ml.evaluate import verify_access_log
from defect_ml.visualize import VisualizeError, mask_size_analysis, select_examples
from PIL import Image

_SRC = Path(__file__).resolve().parent / "test_evaluate.py"
_spec = importlib.util.spec_from_file_location("ml_test_evaluate_for_visualize", _SRC)
ev = importlib.util.module_from_spec(_spec)
sys.modules["ml_test_evaluate_for_visualize"] = ev
_spec.loader.exec_module(ev)
calibrated = ev.calibrated  # the fixture: a calibrated artifact and a fake dataset with test files


def row(path: str, label: int, kind: str, score: float, flagged: int) -> dict:
    return {"path": path, "label": label, "type": kind, "score": score, "flagged": flagged}


# --- the fixed selection rule ---------------------------------------------------------------


def test_the_selection_rule_picks_every_error_and_the_lowest_and_median_detection() -> None:
    rows = [
        row("b/test/good/000.png", 0, "good", 1.0, 0),
        row("b/test/good/001.png", 0, "good", 1.5, 1),  # false positive
        row("b/test/good/002.png", 0, "good", 1.6, 1),  # false positive
        row("b/test/crack/000.png", 1, "crack", 2.0, 1),
        row("b/test/crack/001.png", 1, "crack", 3.0, 1),
        row("b/test/crack/002.png", 1, "crack", 2.5, 1),
        row("b/test/crack/003.png", 1, "crack", 9.0, 1),
        row("b/test/crack/004.png", 1, "crack", 0.5, 0),  # false negative
        row("b/test/hole/000.png", 1, "hole", 4.0, 1),  # a single detection: one tile only
    ]
    chosen = select_examples(rows)
    assert [(c["group"], c["path"]) for c in chosen] == [
        ("false_positive", "b/test/good/001.png"),
        ("false_positive", "b/test/good/002.png"),
        ("false_negative", "b/test/crack/004.png"),
        ("true_positive", "b/test/crack/000.png"),  # lowest of 2.0, 2.5, 3.0, 9.0
        ("true_positive", "b/test/crack/002.png"),  # median = element (4 - 1)//2 = 1 -> 2.5
        ("true_positive", "b/test/hole/000.png"),
    ]
    assert chosen[3]["why"] == "crack: lowest-scoring detection"
    assert chosen[4]["why"] == "crack: median-scoring detection"


def test_the_rule_does_not_depend_on_row_order_and_breaks_ties_by_path() -> None:
    rows = [row(f"b/test/crack/00{i}.png", 1, "crack", 2.0, 1) for i in range(3)]
    forward = select_examples(rows)
    assert forward == select_examples(list(reversed(rows)))
    assert forward[0]["path"] == "b/test/crack/000.png"  # lowest: the first path among ties
    assert forward[1]["path"] == "b/test/crack/001.png"  # median: element (3-1)//2 = 1


def test_no_errors_and_no_detections_give_no_tiles_for_them() -> None:
    rows = [row("b/test/good/000.png", 0, "good", 1.0, 0), row("b/test/x/000.png", 1, "x", 0.1, 0)]
    chosen = select_examples(rows)
    assert [c["group"] for c in chosen] == ["false_negative"]


# --- the mask-size analysis -----------------------------------------------------------------


def test_the_size_analysis_groups_by_type_and_by_thirds_of_equal_count() -> None:
    records = [
        {"path": f"p{i}", "type": "a" if i < 4 else "b", "score": 2.0 + i, "flagged": int(i != 0),
         "area_fraction": 0.01 * (i + 1), "pixel_recall": 0.5 + 0.05 * i}
        for i in range(6)
    ]  # fmt: skip
    out = mask_size_analysis(records)
    assert out["by_defect_type"]["a"]["n"] == 4 and out["by_defect_type"]["a"]["detected"] == 3
    assert out["by_defect_type"]["b"]["area_fraction_median"] == pytest.approx(0.055)
    assert [t["n"] for t in out["by_size_third"]] == [2, 2, 2]
    assert [t["group"] for t in out["by_size_third"]] == [
        "smallest third", "middle third", "largest third"
    ]  # fmt: skip
    assert out["by_size_third"][0]["detected"] == 1  # image 0 (smallest) was missed
    assert len(out["images"]) == 6


# --- the re-scoring: sealed path, log, chain, identical scores ------------------------------


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_pictures_are_made_through_the_logged_sealed_path_and_change_nothing_else(
    calibrated,
) -> None:
    config, path, root, listing = calibrated
    (result,) = evaluate.run_batch([path], eval_root=root)
    log_path = root / evaluate.LOG_NAME
    log_before = log_path.read_bytes()
    kept = {n: _hash(result / n) for n in ("metrics.json", "predictions.csv", "curves.json")}

    (figures,) = visualize.run_batch([path], eval_root=root)
    assert figures == result / "figures"
    # the evaluation files are untouched, the log only grew, and the chain still verifies
    assert {n: _hash(result / n) for n in kept} == kept
    assert log_path.read_bytes().startswith(log_before)
    assert verify_access_log(log_path) == []
    new = [json.loads(x) for x in log_path.read_bytes()[len(log_before) :].decode().splitlines()]
    assert [x["event"] for x in new] == ["start"] + ["open"] * 13 + ["finish"]
    assert {x["purpose"] for x in new} == {"visualize"}
    assert new[-1]["scores_match"] is True
    assert not any("train" in x["path"] for x in new if x["event"] == "open")

    index = json.loads((figures / "index.json").read_text())
    access = index["data_access"]
    assert access["exactly_the_listed_files_each_once"] is True
    assert access["files_opened_inside_dataset_folder"] == 13
    assert access["directory_listings_inside_dataset_folder"] == 0
    assert index["scores_identical_to_predictions_csv"] is True
    assert index["selection_rule"] == visualize.SELECTION_RULE

    # the tiles are exactly what the rule picks from the saved predictions
    rows = visualize.read_predictions(result / "predictions.csv")
    expected = [(c["group"], c["path"]) for c in select_examples(rows)]
    assert [(t["group"], t["path"]) for t in index["tiles"]] == expected
    for group, count in index["counts"].items():
        sheet = Image.open(figures / f"{group}s.png")
        assert sheet.width == 3 * visualize.PANEL + 2 * visualize.GAP
        assert sheet.height >= count * (visualize.PANEL + visualize.CAPTION) + (count - 1) * 6
    analysis = json.loads((figures / "mask_size_analysis.json").read_text())
    assert len(analysis["images"]) == 5 and set(analysis["by_defect_type"]) == {"crack", "hole"}
    assert sum(index["file_bytes"].values()) < 2_000_000  # small, as asked


def test_a_second_run_and_a_missing_evaluation_are_refused(calibrated) -> None:
    _, path, root, _ = calibrated
    with pytest.raises(VisualizeError, match="run the evaluation first"):
        visualize.run_batch([path], eval_root=root)
    evaluate.run_batch([path], eval_root=root)
    visualize.run_batch([path], eval_root=root)
    before = (root / evaluate.LOG_NAME).read_bytes()
    with pytest.raises(VisualizeError, match="already exists"):
        visualize.run_batch([path], eval_root=root)
    assert (root / evaluate.LOG_NAME).read_bytes() == before


def test_a_score_that_differs_from_predictions_csv_stops_the_run_and_is_logged(calibrated) -> None:
    config, path, root, _ = calibrated
    (result,) = evaluate.run_batch([path], eval_root=root)
    csv_path = result / "predictions.csv"
    lines = csv_path.read_text().splitlines()
    first = lines[1].split(",")
    first[3] = repr(float(first[3]) + 1e-9)  # one part in a billion is still a mismatch
    lines[1] = ",".join(first)
    csv_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(VisualizeError, match="predictions.csv has"):
        visualize.run_batch([path], eval_root=root)
    assert not (result / "figures").exists()  # nothing was written
    last = json.loads((root / evaluate.LOG_NAME).read_text().splitlines()[-1])
    assert last["event"] == "failed" and last["purpose"] == "visualize"
    assert verify_access_log(root / evaluate.LOG_NAME) == []


def test_a_tight_disk_stops_before_any_test_file_is_opened(
    calibrated, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path, root, _ = calibrated
    evaluate.run_batch([path], eval_root=root)
    before = (root / evaluate.LOG_NAME).read_bytes()
    monkeypatch.setattr(visualize.runinfo, "free_disk_gib", lambda: 7.0)
    import yaml

    raw = dict(config.raw)
    raw["guard"] = {"max_swap_used_mb": 1e9, "min_free_disk_gib": 7}
    tight = path.with_name("tight.yaml")
    tight.write_text(yaml.safe_dump(raw))
    with OpenAudit() as audit:
        with pytest.raises(VisualizeError, match="free disk"):
            visualize.run_batch([tight], eval_root=root)
    opened, _ = audit.inside(Path(config.data_dir))
    assert not opened
    assert (root / evaluate.LOG_NAME).read_bytes() == before
