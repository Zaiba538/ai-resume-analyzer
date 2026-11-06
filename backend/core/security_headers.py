"""Security response headers.

Applied to every HTTP response, including errors:

* A strict Content-Security-Policy: same-origin scripts, styles and fetches only,
  no inline script, no plugins. ``frame-ancestors`` is configurable because
  Hugging Face Spaces embeds the app in an iframe on huggingface.co.
* nosniff, no-referrer, a restrictive Permissions-Policy, and COOP/CORP.
* ``Cache-Control: no-store`` and ``X-Robots-Tag: noindex`` on the API and
  health endpoints. API responses carry resume-derived personal data, so this
  covers the report routes and every other API route.
"""

from __future__ import annotations

from collections.abc import Iterable

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

NO_STORE_PREFIXES = ("/api/",)
NO_STORE_PATHS = frozenset({"/health", "/ready"})

PERMISSIONS_POLICY = (
    "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
    "accelerometer=(), gyroscope=(), magnetometer=()"
)


def build_csp(frame_ancestors: str) -> str:
    return "; ".join(
        (
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self' data:",
            "font-src 'self'",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'self'",
            f"frame-ancestors {frame_ancestors}",
        )
    )


def _is_no_store(path: str) -> bool:
    return path in NO_STORE_PATHS or path.startswith(NO_STORE_PREFIXES)


class SecurityHeadersMiddleware:
    """Pure ASGI middleware adding security headers to every HTTP response."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        frame_ancestors: str,
        hsts_enabled: bool = False,
        csp_exempt_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.csp = build_csp(frame_ancestors)
        self.hsts_enabled = hsts_enabled
        self.csp_exempt_paths = frozenset(csp_exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = PERMISSIONS_POLICY
                headers["Cross-Origin-Opener-Policy"] = "same-origin"
                headers["Cross-Origin-Resource-Policy"] = "same-origin"
                if path not in self.csp_exempt_paths:
                    headers["Content-Security-Policy"] = self.csp
                if self.hsts_enabled:
                    headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                if _is_no_store(path):
                    headers["Cache-Control"] = "no-store"
                    headers["X-Robots-Tag"] = "noindex, nofollow"
            await send(message)

        await self.app(scope, receive, send_with_headers)
