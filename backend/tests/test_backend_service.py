"""API tests for readiness, failed loads, the model/category/metrics endpoints, request IDs, CORS,
logging and the generated documentation. Synthetic fixtures only."""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from collections.abc import Callable

import pytest
from app.core.settings import Settings
from app.main import create_app
from backend_imaging import make_image, png_bytes
from conftest import N_CALIB, Built, wait_ready
from defect_core.artifacts import load_artifact
from fastapi.testclient import TestClient

API = "/api/v1"


def wait_for_status(client: TestClient, status: str, seconds: float = 60) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        body = client.get(f"{API}/ready").json()
        if body["status"] == status:
            return body
        time.sleep(0.05)
    raise AssertionError(f"never reached status {status!r}")


# --- readiness ------------------------------------------------------------------------------


def test_ready_is_503_while_loading_and_200_after_and_always_quick(
    make_settings: Callable[..., Settings],
) -> None:
    release = threading.Event()

    def slow_loader(path):
        release.wait(30)
        return load_artifact(path)

    with TestClient(create_app(make_settings(), loader=slow_loader)) as c:
        started = time.perf_counter()
        loading = c.get(f"{API}/ready")
        elapsed = time.perf_counter() - started
        assert loading.status_code == 503 and loading.json() == {"status": "loading"}
        assert loading.headers["retry-after"] and elapsed < 1
        assert c.get(f"{API}/health").json() == {"status": "ok"}  # liveness while loading
        # inspections and the catalogue refuse politely meanwhile
        refused = c.post(
            f"{API}/inspections", data={"category": "bottle"},
            files={"file": ("x.png", png_bytes(make_image(1)), "image/png")},
        )  # fmt: skip
        assert refused.status_code == 503 and refused.json()["error"]["code"] == "MODEL_NOT_READY"
        assert refused.headers["retry-after"]
        assert c.get(f"{API}/categories").status_code == 503

        release.set()
        wait_ready(c)
        started = time.perf_counter()
        ready = c.get(f"{API}/ready")
        assert time.perf_counter() - started < 1
        assert ready.status_code == 200
        expected = {"status": "ready", "model_version": "1.0.0", "categories": ["bottle"]}
        assert ready.json() == expected


@pytest.mark.parametrize(
    "case", ["missing_folder", "no_folder_set", "fitted_only", "version_mismatch", "damaged"]
)
def test_a_model_that_can_not_be_served_leaves_the_service_up_but_not_ready(
    case: str, make_settings: Callable[..., Settings], built: Built, tmp_path
) -> None:
    overrides: dict = {}
    if case == "missing_folder":
        overrides["model_dir"] = tmp_path / "nowhere"
    elif case == "no_folder_set":
        overrides["model_dir"] = None
    elif case == "fitted_only":
        overrides["model_dir"] = built.fitted  # no calibration, no threshold
    elif case == "version_mismatch":
        overrides["model_version"] = "9.9.9"
    elif case == "damaged":
        broken = tmp_path / "broken"
        shutil.copytree(built.package, broken)
        data = bytearray((broken / "model.safetensors").read_bytes())
        data[-1] ^= 1
        (broken / "model.safetensors").write_bytes(bytes(data))
        overrides["model_dir"] = broken
    with TestClient(create_app(make_settings(**overrides))) as c:
        body = wait_for_status(c, "failed")
        assert body == {"status": "failed"}  # the reason is logged, not shown
        assert c.get(f"{API}/ready").status_code == 503
        assert c.get(f"{API}/health").status_code == 200
        response = c.post(
            f"{API}/inspections", data={"category": "bottle"},
            files={"file": ("x.png", png_bytes(make_image(1)), "image/png")},
        )  # fmt: skip
        assert response.status_code == 503 and "MODEL_NOT_READY" in response.text


def test_a_load_failure_is_logged_with_its_reason(
    make_settings: Callable[..., Settings], tmp_path, capsys: pytest.CaptureFixture
) -> None:
    settings = make_settings(model_dir=tmp_path / "nowhere", log_level="INFO")
    with TestClient(create_app(settings)) as c:
        wait_for_status(c, "failed")
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    failed = [x for x in lines if x.get("event") == "model_load_failed"]
    assert failed and failed[0]["level"] == "error" and failed[0]["reason"]


# --- model, categories, metrics ---------------------------------------------------------------


def test_categories_describe_the_calibration_and_a_short_evaluation_summary(
    client: TestClient, built: Built
) -> None:
    (entry,) = client.get(f"{API}/categories").json()
    calibration = load_artifact(built.package).calibration
    assert entry["id"] == "bottle" and entry["display_name"] == "Bottle"
    assert entry["kind"] == "object" and entry["calib_n"] == N_CALIB
    assert entry["threshold"] == calibration.threshold and entry["alpha"] == calibration.alpha
    summary = entry["metrics_summary"]
    assert summary["image_auroc"] == 1.0 and summary["n_normal_test_images"] == 20
    assert summary["fpr_on_normal_test_images"] == 0.15 and "one category" in summary["note"]


def test_the_model_endpoint_states_the_method_the_calibration_and_the_limits(
    client: TestClient,
) -> None:
    body = client.get(f"{API}/model").json()
    assert body["backbone"] == "resnet18" and body["version"] == "1.0.0"
    assert body["preprocessing"]["size"] == 256
    assert body["calibration"]["method"].startswith("split-conformal")
    assert "Says nothing about defects" in body["calibration"]["meaning"]
    limits = " ".join(body["limitations"])
    for needed in ("not a certified", "out of scope", "not a probability"):
        assert needed in limits
    assert not any(w in limits.lower().split() for w in ("pass", "fail"))


def test_metrics_without_evaluation_files_gives_the_package_summary_only(
    client: TestClient,
) -> None:
    body = client.get(f"{API}/metrics/bottle").json()
    assert body["results"]["image"]["auroc"]["estimate"] == 1.0
    assert body["curves_available"] is False and "curves" not in body


def test_metrics_with_the_matching_evaluation_files_adds_curves_and_per_defect_recall(
    make_settings: Callable[..., Settings], built: Built
) -> None:
    with TestClient(create_app(make_settings(evaluation_dir=built.evaluation))) as c:
        wait_ready(c)
        body = c.get(f"{API}/metrics/bottle").json()
    assert body["curves_available"] is True and body["curves"]["image_roc"]["tpr"] == [0, 1]
    assert body["histograms"]["bin_edges"] == [0, 1]
    assert body["per_defect_recall"]["crack"]["recall"] == 1.0


def test_evaluation_files_that_are_not_the_packages_are_refused(
    make_settings: Callable[..., Settings], built: Built, tmp_path
) -> None:
    other = tmp_path / "evaluation"
    shutil.copytree(built.evaluation, other)
    metrics = json.loads((other / "metrics.json").read_text())
    metrics["results"]["image"]["auroc"]["estimate"] = 0.5  # edited after packaging
    (other / "metrics.json").write_text(json.dumps(metrics))
    with TestClient(create_app(make_settings(evaluation_dir=other))) as c:
        assert wait_for_status(c, "failed") == {"status": "failed"}
    with TestClient(create_app(make_settings(evaluation_dir=tmp_path / "missing"))) as c:
        assert wait_for_status(c, "failed") == {"status": "failed"}


def test_metrics_for_an_unknown_category_is_404(client: TestClient) -> None:
    response = client.get(f"{API}/metrics/cable")
    assert response.status_code == 404 and response.json()["error"]["code"] == "NOT_FOUND"


# --- request IDs, CORS, logs, docs -----------------------------------------------------------


def test_a_safe_request_id_is_kept_and_an_unsafe_one_is_replaced(client: TestClient) -> None:
    assert (
        client.get(f"{API}/health", headers={"X-Request-ID": "abc-123_x.y"}).headers["x-request-id"]
        == "abc-123_x.y"
    )
    for bad in ("has spaces and !!", "x" * 100, "a/b"):
        got = client.get(f"{API}/health", headers={"X-Request-ID": bad}).headers["x-request-id"]
        assert got != bad and uuid.UUID(got)
    assert uuid.UUID(client.get(f"{API}/health").headers["x-request-id"])


def test_error_responses_carry_the_same_id_in_header_and_body(client: TestClient) -> None:
    response = client.get(f"{API}/missing", headers={"X-Request-ID": "trace-me"})
    assert response.headers["x-request-id"] == "trace-me"
    assert response.json()["error"]["request_id"] == "trace-me"


def test_only_allowed_origins_get_cors_headers(
    make_settings: Callable[..., Settings],
) -> None:
    with TestClient(create_app(make_settings(allowed_origins="https://demo.example"))) as c:
        wait_ready(c)
        ok = c.get(f"{API}/health", headers={"Origin": "https://demo.example"})
        assert ok.headers["access-control-allow-origin"] == "https://demo.example"
        assert "x-request-id" in ok.headers["access-control-expose-headers"].lower()
        other = c.get(f"{API}/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers
        preflight = c.options(
            f"{API}/inspections",
            headers={"Origin": "https://demo.example", "Access-Control-Request-Method": "POST"},
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://demo.example"
        denied = c.options(
            f"{API}/inspections",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in denied.headers
        # error responses (here a 404) carry the CORS header too
        err = c.get(f"{API}/missing", headers={"Origin": "https://demo.example"})
        assert err.headers["access-control-allow-origin"] == "https://demo.example"


def test_no_origin_is_allowed_by_default(client: TestClient) -> None:
    response = client.get(f"{API}/health", headers={"Origin": "https://anything.example"})
    assert "access-control-allow-origin" not in response.headers


def test_each_request_is_one_json_log_line_with_the_id_and_no_upload_details(
    make_settings: Callable[..., Settings], capsys: pytest.CaptureFixture
) -> None:
    with TestClient(create_app(make_settings(log_level="INFO"))) as c:
        wait_ready(c)
        capsys.readouterr()
        response = c.post(
            f"{API}/inspections", data={"category": "bottle"},
            files={"file": ("my_private_photo.png", png_bytes(make_image(1)), "image/png")},
            headers={"X-Request-ID": "log-check-1"},
        )  # fmt: skip
    out = capsys.readouterr().out
    lines = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
    requests = [
        x for x in lines if x.get("event") == "request" and x["request_id"] == "log-check-1"
    ]
    assert len(requests) == 1
    entry = requests[0]
    assert entry["method"] == "POST" and entry["path"] == f"{API}/inspections"
    assert entry["status"] == response.status_code == 200 and entry["duration_ms"] > 0
    assert "my_private_photo" not in out and "base64" not in out  # no file name, no image data


def test_the_interactive_documentation_is_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/v1/health", "/api/v1/ready", "/api/v1/categories", "/api/v1/model",
                 "/api/v1/metrics/{category}", "/api/v1/inspections"):  # fmt: skip
        assert path in paths
