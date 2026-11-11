"""The OpenAPI description tells the truth about errors, and the frontend's copy is current.

No server, no model and no network: the description is built from the app's routes alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.openapi_export import openapi_text
from app.schemas import ErrorEnvelope
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "frontend" / "openapi.json"
ENVELOPE_REF = "#/components/schemas/ErrorEnvelope"


def spec() -> dict:
    return json.loads(openapi_text())


def response_ref(operation: dict, status: str) -> str:
    return operation["responses"][status]["content"]["application/json"]["schema"]["$ref"]


def test_every_error_status_of_the_inspection_is_documented_as_the_envelope() -> None:
    operation = spec()["paths"]["/api/v1/inspections"]["post"]
    assert set(operation["responses"]) == {"200", "413", "415", "422", "429", "500", "503"}
    for status in ("413", "415", "422", "429", "500", "503"):
        assert response_ref(operation, status) == ENVELOPE_REF


def test_no_route_refers_to_the_default_fastapi_422_shape() -> None:
    # FastAPI keeps its own schema in `components` even when unused; what matters is that no
    # route points at it, so generated clients get the envelope and not this shape.
    assert "ValidationError" not in json.dumps(spec()["paths"])


def test_the_read_only_endpoints_document_their_errors_too() -> None:
    paths = spec()["paths"]
    for path in ("/api/v1/categories", "/api/v1/model"):
        assert response_ref(paths[path]["get"], "503") == ENVELOPE_REF
    metrics = paths["/api/v1/metrics/{category}"]["get"]
    assert response_ref(metrics, "404") == ENVELOPE_REF
    assert response_ref(metrics, "503") == ENVELOPE_REF


def test_the_documented_envelope_matches_what_the_api_really_returns(client: TestClient) -> None:
    response = client.post("/api/v1/inspections", data={"category": "bottle"})
    assert response.status_code == 422
    body = response.json()
    parsed = ErrorEnvelope.model_validate(body)  # the documented shape accepts the real answer
    assert parsed.error.code == "VALIDATION_ERROR"
    assert set(spec()["components"]["schemas"]["ErrorBody"]["properties"]) == set(body["error"])


def test_the_committed_frontend_copy_is_current() -> None:
    if not COMMITTED.exists():
        pytest.skip("frontend/openapi.json has not been created yet")
    assert COMMITTED.read_text(encoding="utf-8") == openapi_text(), (
        "frontend/openapi.json is out of date: run `make api-contract` and then "
        "`npm run gen:api` in frontend/"
    )
