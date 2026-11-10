"""API tests for POST /api/v1/inspections: success, shared behaviour with the CLI, the calibration
images through HTTP (replacing the plan's test-set check), and every Release 1 error code (6.3).

Synthetic images only; the model is a small random-weight artifact. These tests check the service,
not detection quality.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import uuid
from collections.abc import Callable

import pytest
from app.core.settings import Settings
from app.main import create_app
from app.schemas import InspectionResponse
from backend_imaging import jpeg_bytes, make_image, png_bytes
from conftest import N_CALIB, Built, wait_ready
from defect_core.artifacts import load_artifact
from defect_core.infer import inspect_image, main
from fastapi.testclient import TestClient
from PIL import Image

URL = "/api/v1/inspections"


def post(client: TestClient, data: bytes, category: str = "bottle", name: str = "x.png", **extra):
    return client.post(
        URL, data={"category": category}, files={"file": (name, data, "image/png")}, **extra
    )


def decode(data_url: str) -> Image.Image:
    assert data_url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def error_of(response) -> dict:
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {
        "code", "message", "details", "request_id",
    }  # fmt: skip
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    return body["error"]


# --- success --------------------------------------------------------------------------------


def test_a_png_is_inspected_and_the_response_has_the_documented_shape(client: TestClient) -> None:
    response = post(client, png_bytes(make_image(7)))
    assert response.status_code == 200
    body = InspectionResponse.model_validate(response.json())  # the schema in the docs
    assert uuid.UUID(body.id) and body.created_at.endswith("Z")
    assert body.category == "bottle" and body.model.name == "test_model"
    assert body.model.version == "1.0.0"
    assert body.prediction in ("normal", "anomalous")
    assert (body.prediction == "anomalous") == (body.anomaly_score > body.threshold)
    assert body.calibration_rank.of_calibration_normals == N_CALIB
    assert 0 <= body.calibration_rank.score_higher_than <= N_CALIB
    assert body.persisted is False and body.warnings == []
    assert "demo decision" in body.disclaimer.lower() or "Demo decision" in body.disclaimer
    assert body.decision_rule.startswith("anomalous if anomaly_score > threshold")
    for view in (body.images.original, body.images.heatmap, body.images.overlay):
        assert decode(view.url).size == (128, 96)  # all at the original image size
    assert body.images.original.width == 128 and body.images.original.height == 96
    assert body.timing_ms.total >= body.timing_ms.inference > 0
    assert response.headers["x-request-id"]


def test_a_jpeg_works_too_and_the_file_name_is_not_used(client: TestClient) -> None:
    response = client.post(
        URL, data={"category": "bottle"},
        files={"file": ("../../etc/passwd.txt", jpeg_bytes(make_image(2)), "text/plain")},
    )  # fmt: skip
    assert response.status_code == 200


def test_the_api_gives_the_same_result_as_the_command_line_and_the_shared_function(
    client: TestClient, built: Built, tmp_path, capsys: pytest.CaptureFixture
) -> None:
    image = make_image(11)
    path = tmp_path / "scene.png"
    path.write_bytes(png_bytes(image))
    api = post(client, path.read_bytes()).json()

    direct = inspect_image(load_artifact(built.package), image)
    assert api["anomaly_score"] == direct.score and api["prediction"] == direct.verdict

    capsys.readouterr()
    assert main(["--artifact", str(built.package), str(path), "--json"]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["anomaly_score"] == api["anomaly_score"]
    assert cli["prediction"] == api["prediction"] and cli["threshold"] == api["threshold"]
    assert cli["calibration_rank"] == api["calibration_rank"]
    assert cli["region_bbox"] == api["region_bbox"]
    assert cli["highlighted_area_fraction"] == api["highlighted_area_fraction"]
    assert cli["decision_rule"] == api["decision_rule"]


def test_the_calibration_images_scored_through_http_reproduce_the_stored_calibration(
    client: TestClient, built: Built
) -> None:
    """Replaces the plan's golden test on test-set images: the images the artifact was calibrated
    on are synthetic fixtures, so no sealed file is involved."""
    responses = [post(client, png_bytes(img)).json() for img in built.calib_images]
    scores = sorted(r["anomaly_score"] for r in responses)
    assert scores == built.calibration_scores  # exactly the stored scores
    flagged = sum(r["prediction"] == "anomalous" for r in responses)
    assert flagged == 1  # n = 20, k = 19: one calibration image lies above the threshold
    ranks = sorted(r["calibration_rank"]["score_higher_than"] for r in responses)
    assert ranks == list(range(N_CALIB))  # each calibration score is higher than all below it


def test_the_same_upload_gives_the_same_answer_twice(client: TestClient) -> None:
    data = png_bytes(make_image(3))
    first, second = post(client, data).json(), post(client, data).json()
    assert first["anomaly_score"] == second["anomaly_score"]
    assert first["images"]["overlay"] == second["images"]["overlay"]
    assert first["id"] != second["id"]


# --- the error codes ------------------------------------------------------------------------


def test_missing_fields_give_validation_error_with_field_names_only(client: TestClient) -> None:
    no_file = client.post(URL, data={"category": "bottle"})
    assert no_file.status_code == 422
    error = error_of(no_file)
    assert error["code"] == "VALIDATION_ERROR" and error["details"]["fields"][0]["field"] == "file"
    no_category = client.post(URL, files={"file": ("x.png", png_bytes(make_image(1)), "image/png")})
    assert error_of(no_category)["code"] == "VALIDATION_ERROR"
    assert "bottle" not in json.dumps(error)  # submitted values are never echoed


def test_an_unsupported_category_is_refused_before_the_file_is_read(client: TestClient) -> None:
    response = post(client, b"not even an image", category="cable")
    assert response.status_code == 422
    error = error_of(response)
    assert error["code"] == "UNSUPPORTED_CATEGORY" and error["details"] == {"supported": ["bottle"]}


@pytest.mark.parametrize("data", [b"GIF89a" + b"\x00" * 100, b"plain text", b"%PDF-1.7", b""])
def test_files_that_are_not_jpeg_or_png_are_415(client: TestClient, data: bytes) -> None:
    response = post(client, data, name="photo.png")  # the name says png; the bytes decide
    assert response.status_code == 415 and error_of(response)["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_a_cut_off_image_is_422_invalid_image(client: TestClient) -> None:
    good = png_bytes(make_image(1))
    response = post(client, good[: len(good) // 2])
    assert response.status_code == 422 and error_of(response)["code"] == "INVALID_IMAGE"


@pytest.mark.parametrize("size", [(32, 96), (63, 128), (8001, 64)])
def test_sides_outside_the_range_are_422_image_dimensions(client: TestClient, size) -> None:
    response = post(client, png_bytes(Image.new("RGB", size)))
    assert response.status_code == 422
    error = error_of(response)
    assert error["code"] == "IMAGE_DIMENSIONS" and error["details"] == {
        "min_side": 64, "max_side": 8000,
    }  # fmt: skip


def test_the_pixel_cap_is_422_image_dimensions(make_settings: Callable[..., Settings]) -> None:
    with TestClient(create_app(make_settings(max_image_pixels=10_000))) as small:
        wait_ready(small)
        response = post(small, png_bytes(make_image(1)))
    assert response.status_code == 422 and error_of(response)["code"] == "IMAGE_DIMENSIONS"


def test_an_oversized_upload_is_413_from_the_declared_length(
    make_settings: Callable[..., Settings],
) -> None:
    with TestClient(create_app(make_settings(max_upload_mb=1))) as small:
        wait_ready(small)
        response = post(small, b"\x89PNG\r\n\x1a\n" + b"0" * (1_300_000))
    assert response.status_code == 413
    error = error_of(response)
    assert error["code"] == "FILE_TOO_LARGE" and error["details"] == {"max_mb": 1}


def test_a_file_just_over_the_limit_is_413_from_the_endpoint(
    make_settings: Callable[..., Settings],
) -> None:
    # small enough to pass the body guard (it allows some multipart framing), too big as a file
    with TestClient(create_app(make_settings(max_upload_mb=1))) as small:
        wait_ready(small)
        response = post(small, b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 100))
    assert response.status_code == 413 and error_of(response)["code"] == "FILE_TOO_LARGE"


def test_a_streamed_body_without_a_length_is_413_through_the_whole_app(
    make_settings: Callable[..., Settings],
) -> None:
    def chunks():
        for _ in range(5):
            yield b"0" * 1_000_000

    with TestClient(create_app(make_settings(max_upload_mb=1))) as small:
        wait_ready(small)
        response = small.post(
            URL, content=chunks(), headers={"content-type": "multipart/form-data; boundary=x"}
        )
    assert response.status_code == 413 and error_of(response)["code"] == "FILE_TOO_LARGE"


def test_the_size_guard_stops_reading_a_body_that_has_no_length() -> None:
    import asyncio

    from app.middleware import BodySizeLimitMiddleware

    pulled = 0
    sent: list[dict] = []

    async def reads_everything(scope, receive, send):  # an app that would read the whole body
        while True:
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"parse error"})

    async def receive():
        nonlocal pulled
        pulled += 1
        return {"type": "http.request", "body": b"0" * 100_000, "more_body": True}

    async def send(message):
        sent.append(message)

    guard = BodySizeLimitMiddleware(reads_everything, max_bytes=1024 * 1024)
    scope = {"type": "http", "headers": [], "method": "POST", "path": "/"}
    asyncio.run(guard(scope, receive, send))
    assert pulled < 15  # about 1 MiB plus framing, not a long upload
    assert sent[0]["status"] == 413 and b"FILE_TOO_LARGE" in sent[1]["body"]
    assert len(sent) == 2  # the application's own 400 was replaced, not added


def test_an_unexpected_model_error_is_500_inference_failed_and_nothing_leaks(
    make_settings: Callable[..., Settings],
) -> None:
    def failing(loaded, image, *args, **kwargs):
        if image.size == (256, 256):  # the warm-up image
            return inspect_image(loaded, image, *args, **kwargs)
        raise RuntimeError("secret detail at /private/path with a stack")

    with TestClient(create_app(make_settings(), inspector=failing)) as broken:
        wait_ready(broken)
        response = post(broken, png_bytes(make_image(1)))
        again = post(broken, png_bytes(make_image(1)))  # the service survives and the turn is freed
    assert response.status_code == 500 and error_of(response)["code"] == "INFERENCE_FAILED"
    assert "secret" not in response.text and "Traceback" not in response.text
    assert again.status_code == 500  # not stuck on BUSY: the semaphore was released


def test_an_unexpected_error_elsewhere_is_500_internal_error_without_a_trace(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.routers.inspections as module

    def boom(*_args, **_kwargs):
        raise ZeroDivisionError("do not leak me")

    monkeypatch.setattr(module, "validate_upload", boom)
    with TestClient(create_app(make_settings()), raise_server_exceptions=False) as broken:
        wait_ready(broken)
        response = post(broken, png_bytes(make_image(1)))
    assert response.status_code == 500 and error_of(response)["code"] == "INTERNAL_ERROR"
    assert "leak" not in response.text and "Traceback" not in response.text


def test_unknown_routes_and_wrong_methods_use_the_envelope(client: TestClient) -> None:
    missing = client.get("/api/v1/nothing-here")
    assert missing.status_code == 404 and error_of(missing)["code"] == "NOT_FOUND"
    wrong = client.get(URL)
    assert wrong.status_code == 405 and error_of(wrong)["code"] == "METHOD_NOT_ALLOWED"


# --- concurrency: one inspection at a time, a queue timeout, and no stuck state ----------------


def test_only_one_inspection_runs_at_a_time(make_settings: Callable[..., Settings]) -> None:
    lock = threading.Lock()
    running = maximum = 0

    def counting(loaded, image, *args, **kwargs):
        nonlocal running, maximum
        with lock:
            running += 1
            maximum = max(maximum, running)
        try:
            return inspect_image(loaded, image, *args, **kwargs)
        finally:
            with lock:
                running -= 1

    results: list[int] = []
    with TestClient(
        create_app(make_settings(inference_queue_timeout_s=60), inspector=counting)
    ) as busy:
        wait_ready(busy)
        data = png_bytes(make_image(1))
        threads = [
            threading.Thread(target=lambda: results.append(post(busy, data).status_code))
            for _ in range(4)
        ]
        [t.start() for t in threads]
        [t.join(60) for t in threads]
    assert results == [200] * 4 and maximum == 1


def test_a_request_that_waits_too_long_gets_503_busy_with_retry_after(
    make_settings: Callable[..., Settings],
) -> None:
    entered, release = threading.Event(), threading.Event()

    def slow(loaded, image, *args, **kwargs):
        if image.size != (256, 256):  # not the warm-up
            entered.set()
            release.wait(30)
        return inspect_image(loaded, image, *args, **kwargs)

    first: list[int] = []
    with TestClient(
        create_app(make_settings(inference_queue_timeout_s=0.3), inspector=slow)
    ) as busy:
        wait_ready(busy)
        data = png_bytes(make_image(1))
        holder = threading.Thread(target=lambda: first.append(post(busy, data).status_code))
        holder.start()
        assert entered.wait(30)
        waiting = post(busy, data)
        release.set()
        holder.join(30)
    assert waiting.status_code == 503 and error_of(waiting)["code"] == "BUSY"
    assert waiting.headers["retry-after"] == "5"
    assert first == [200]  # the first request was not harmed


# --- rate limiting (with proxy-aware client addresses) ------------------------------------------


def limited(make_settings, **values) -> TestClient:
    return TestClient(create_app(make_settings(rate_limit="2/minute", **values)))


def test_the_third_request_in_a_minute_is_429_with_the_envelope(make_settings) -> None:
    with limited(make_settings) as c:
        wait_ready(c)
        data = png_bytes(make_image(1))
        assert [post(c, data).status_code for _ in range(2)] == [200, 200]
        third = post(c, data)
        assert third.status_code == 429
        error = error_of(third)
        assert error["code"] == "RATE_LIMITED" and third.headers["retry-after"]
        assert c.get("/api/v1/health").status_code == 200  # only inspections are limited
        assert c.get("/api/v1/ready").status_code == 200


def test_behind_a_trusted_proxy_each_client_has_its_own_limit(make_settings) -> None:
    with limited(make_settings, trusted_proxy_hops=1) as c:
        wait_ready(c)
        data = png_bytes(make_image(1))

        def hit(client_ip: str, spoof: str = "") -> int:
            return post(c, data, headers={"x-forwarded-for": f"{spoof}{client_ip}"}).status_code

        assert [hit("203.0.113.1") for _ in range(3)] == [200, 200, 429]
        assert hit("203.0.113.2") == 200  # another client: its own allowance
        # leading entries added by the client do not give it a fresh allowance
        assert hit("203.0.113.1", spoof="9.9.9.9, ") == 429
        assert hit("203.0.113.1", spoof="8.8.8.8, 7.7.7.7, ") == 429


def test_without_a_trusted_proxy_the_forwarded_header_is_ignored(make_settings) -> None:
    with limited(make_settings, trusted_proxy_hops=0) as c:
        wait_ready(c)
        data = png_bytes(make_image(1))
        codes = [
            post(c, data, headers={"x-forwarded-for": f"198.51.100.{i}"}).status_code
            for i in range(3)
        ]
        assert codes == [200, 200, 429]  # all share the one connection address
