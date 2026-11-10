"""Request-ID, request logging and the body-size guard (PLAN.md Phase 5 step 2).

Both are plain ASGI middlewares (they do not buffer or re-read the body).

* ``RequestContextMiddleware``: gives every request an ID (the client's ``X-Request-ID`` if it is a
  short safe string, otherwise a fresh UUID), returns it in the ``X-Request-ID`` header, binds it
  into the logs, and logs one line per request: method, path, status, duration. Query strings,
  headers and bodies are not logged.
* ``BodySizeLimitMiddleware``: refuses a request body larger than the limit with the 413
  ``FILE_TOO_LARGE`` envelope. An honest ``Content-Length`` is checked before anything is read; a
  body without one (chunked) is counted as it arrives and stopped at the limit, so an oversized
  upload is never read to the end.
"""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import envelope_bytes, request_id_of
from app.core.logging import get_logger

_SAFE_ID = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
log = get_logger("app.request")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _SAFE_ID.match(supplied) else str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log.info(
                "request", method=scope["method"], path=scope["path"], status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )  # fmt: skip
            structlog.contextvars.clear_contextvars()


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        # a multipart body carries a little framing around the file itself
        self.max_bytes = max_bytes + 64 * 1024
        self.max_mb = max_bytes // (1024 * 1024)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse(scope, send)
            return

        received = 0
        exceeded = False
        refused = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}  # the rest of the body is never read
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal refused
            if exceeded:
                # the application will answer with whatever it made of a cut-off body (FastAPI
                # says 400); replace that answer with the 413 envelope
                if not refused:
                    refused = True
                    await self._refuse(scope, send)
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not refused:
            await self._refuse(scope, send)

    async def _refuse(self, scope: Scope, send: Send) -> None:
        body = envelope_bytes(
            "FILE_TOO_LARGE", f"Image must be {self.max_mb} MB or smaller.",
            {"max_mb": self.max_mb}, request_id_of(scope),
        )  # fmt: skip
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"connection", b"close")],
        })  # fmt: skip
        await send({"type": "http.response.body", "body": body})
