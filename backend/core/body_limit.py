"""Request body size limit.

Rejects oversized requests with 413 before the body is buffered:

* a declared ``Content-Length`` above the limit is refused without reading;
* otherwise, including chunked bodies, bytes are counted as they arrive and
  reading stops as soon as the limit is crossed.

The streaming case raises Starlette's ``HTTPException(413)``. FastAPI re-raises
``HTTPException`` from body parsing unchanged (other exceptions become a
generic 400), so the standard error envelope handler produces the response.
"""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.core.errors import error_response

_CONTENT_LENGTH = b"content-length"
TOO_LARGE_MESSAGE = "The request body exceeds the maximum allowed size."


class BodySizeLimitMiddleware:
    """Pure ASGI middleware enforcing a maximum request body size."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = next((v for k, v in scope["headers"] if k == _CONTENT_LENGTH), None)
        if declared is not None:
            try:
                too_large = int(declared) > self.max_body_bytes
            except ValueError:
                too_large = False  # malformed header: rely on streaming enforcement
            if too_large:
                response = error_response(413, "payload_too_large", TOO_LARGE_MESSAGE)
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(status_code=413)
            return message

        await self.app(scope, limited_receive, send)
