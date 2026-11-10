"""One error envelope with stable codes (PLAN.md section 6.3, Phase 5 step 3).

    {"error": {"code": "FILE_TOO_LARGE", "message": "...", "details": {...}, "request_id": "..."}}

Stack traces are logged and never returned. ``INTERNAL_ERROR`` (500) is used for an unexpected
failure outside the model, ``INFERENCE_FAILED`` (500) for one inside it. ``details`` never holds
anything the client sent beyond field names and limits.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

log = get_logger("app.errors")


class ApiError(Exception):
    """An error with a stable code that is returned in the envelope."""

    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.headers = headers or {}


class ValidationFailed(ApiError):
    status_code, code = 422, "VALIDATION_ERROR"


class UnsupportedCategory(ApiError):
    status_code, code = 422, "UNSUPPORTED_CATEGORY"


class UnsupportedMediaType(ApiError):
    status_code, code = 415, "UNSUPPORTED_MEDIA_TYPE"


class FileTooLarge(ApiError):
    status_code, code = 413, "FILE_TOO_LARGE"


class InvalidImage(ApiError):
    status_code, code = 422, "INVALID_IMAGE"


class ImageDimensions(ApiError):
    status_code, code = 422, "IMAGE_DIMENSIONS"


class ModelNotReady(ApiError):
    status_code, code = 503, "MODEL_NOT_READY"


class Busy(ApiError):
    status_code, code = 503, "BUSY"


class RateLimited(ApiError):
    status_code, code = 429, "RATE_LIMITED"


class InferenceFailed(ApiError):
    status_code, code = 500, "INFERENCE_FAILED"


class NotFound(ApiError):
    status_code, code = 404, "NOT_FOUND"


def request_id_of(scope_or_request: Any) -> str:
    scope = getattr(scope_or_request, "scope", scope_or_request)
    return str(scope.get("state", {}).get("request_id", "unknown"))


def envelope(
    code: str, message: str, details: dict[str, Any] | None, request_id: str
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }


def envelope_bytes(
    code: str, message: str, details: dict[str, Any] | None, request_id: str
) -> bytes:
    return json.dumps(envelope(code, message, details, request_id)).encode()


def _response(
    status: int, code: str, message: str, details: dict | None, request: Request,
    headers: dict[str, str] | None = None,
) -> JSONResponse:  # fmt: skip
    body = envelope(code, message, details, request_id_of(request))
    return JSONResponse(body, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, error: ApiError) -> JSONResponse:
        return _response(
            error.status_code, error.code, error.message, error.details, request, error.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, error: RequestValidationError) -> JSONResponse:
        fields = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "issue": e["type"]}
            for e in error.errors()
        ]  # field names and issue kinds only: never the submitted values
        return _response(
            422, "VALIDATION_ERROR", "The request is missing a field or a field is invalid.",
            {"fields": fields}, request,
        )  # fmt: skip

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(request: Request, error: RateLimitExceeded) -> JSONResponse:
        return _response(
            429, "RATE_LIMITED", "Too many requests. Please wait and try again.",
            {"limit": str(error.limit.limit)}, request, {"Retry-After": "60"},
        )  # fmt: skip

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, error: StarletteHTTPException) -> JSONResponse:
        codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
        code = codes.get(error.status_code, "HTTP_ERROR")
        message = "Not found." if error.status_code == 404 else "The request is not allowed."
        return _response(error.status_code, code, message, None, request)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, error: Exception) -> JSONResponse:
        log.error(
            "unhandled_exception", exc_info=error, request_id=request_id_of(request),
            path=request.url.path,
        )  # fmt: skip
        message = "Something went wrong on the server."
        # this response is made outside the request-ID middleware, so it carries the ID itself
        headers = {"X-Request-ID": request_id_of(request)}
        return _response(500, "INTERNAL_ERROR", message, None, request, headers)
