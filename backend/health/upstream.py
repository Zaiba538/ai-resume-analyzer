"""Groq upstream availability monitor.

Distinguishes *key configured* (a local fact) from *upstream available* (an
external fact), as the approved readiness contract requires:

* A background task probes ``GET {base_url}/models`` at start-up and then every
  ``interval`` seconds. That endpoint returns model metadata only, so it costs
  no inference or tokens. The probe also checks that every configured model is
  offered to this key.
* Real LLM calls (from M3) report their outcome via ``record_request_outcome``
  as a passive signal.
* ``/ready`` only reads ``snapshot()``. It never triggers an outbound call.
* Without an API key nothing is probed and the status stays ``unknown``.

Neither the API key nor any response body is ever logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr

logger = logging.getLogger("backend.health.upstream")

UpstreamStatus = Literal["available", "unavailable", "unknown"]
SignalSource = Literal["probe", "request"]


class UpstreamSnapshot(BaseModel):
    """Point-in-time view of upstream availability, as served by ``/ready``."""

    model_config = ConfigDict(frozen=True)

    status: UpstreamStatus = "unknown"
    configured_models_present: bool | None = None
    last_checked_at: datetime | None = None
    last_success_at: datetime | None = None
    source: SignalSource | None = None
    last_error_code: str | None = None


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _error_code_for_status(status_code: int) -> str:
    if status_code in (401, 403):
        return "auth_failed"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "upstream_error"
    return "unexpected_status"


class GroqUpstreamMonitor:
    """Tracks Groq availability without making calls on the request path."""

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        base_url: str,
        required_models: Iterable[str],
        interval_seconds: float,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._api_key = api_key
        self._models_url = f"{base_url.rstrip('/')}/models"
        self._required_models = frozenset(required_models)
        self._interval = interval_seconds
        self._timeout = timeout_seconds
        self._client = client
        self._owns_client = client is None
        self._clock = clock
        self._snapshot = UpstreamSnapshot()
        self._probe_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    @property
    def key_configured(self) -> bool:
        return self._api_key is not None

    def snapshot(self) -> UpstreamSnapshot:
        return self._snapshot

    def _update(self, **changes: object) -> UpstreamSnapshot:
        self._snapshot = self._snapshot.model_copy(update=changes)
        return self._snapshot

    async def probe(self) -> UpstreamSnapshot:
        """Run one metadata probe. Never raises; failures are recorded as status."""
        if self._api_key is None:
            return self._snapshot

        async with self._probe_lock:
            if self._client is None:
                self._client = httpx.AsyncClient()
            now = self._clock()
            try:
                response = await self._client.get(
                    self._models_url,
                    headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                    timeout=self._timeout,
                )
            except httpx.TimeoutException:
                snapshot = self._failed(now, "timeout")
            except httpx.HTTPError:
                snapshot = self._failed(now, "network_error")
            else:
                snapshot = self._from_response(now, response)

        logger.info(
            "groq_probe_completed",
            extra={
                "upstream_status": snapshot.status,
                "models_present": snapshot.configured_models_present,
                "error_code": snapshot.last_error_code,
            },
        )
        return snapshot

    def _failed(self, now: datetime, error_code: str) -> UpstreamSnapshot:
        return self._update(
            status="unavailable", last_checked_at=now, source="probe", last_error_code=error_code
        )

    def _from_response(self, now: datetime, response: httpx.Response) -> UpstreamSnapshot:
        if response.status_code != 200:
            return self._failed(now, _error_code_for_status(response.status_code))
        try:
            payload = response.json()
            offered = {item["id"] for item in payload["data"] if isinstance(item, dict)}
        except (ValueError, KeyError, TypeError):
            return self._failed(now, "invalid_response")
        return self._update(
            status="available",
            configured_models_present=self._required_models <= offered,
            last_checked_at=now,
            last_success_at=now,
            source="probe",
            last_error_code=None,
        )

    def record_request_outcome(self, success: bool, error_code: str | None = None) -> None:
        """Passive signal from a real Groq call made while serving a request."""
        now = self._clock()
        if success:
            self._update(
                status="available",
                last_checked_at=now,
                last_success_at=now,
                source="request",
                last_error_code=None,
            )
        else:
            self._update(
                status="unavailable",
                last_checked_at=now,
                source="request",
                last_error_code=error_code or "request_failed",
            )

    async def _run(self) -> None:
        while True:
            await self.probe()
            await asyncio.sleep(self._interval)

    async def start(self) -> None:
        """Start periodic probing. No-op without an API key or if already running."""
        if self._api_key is None or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="groq-upstream-monitor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
