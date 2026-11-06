"""Liveness and readiness endpoints.

``GET /health`` checks liveness only: it touches no dependency and makes no
outbound call.

``GET /ready`` reports local initialisation (``checks``), which alone decides
200 vs 503, plus cached upstream availability (``upstream``), which is
informational and never makes an outbound call.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.deps import ReadinessDep, UpstreamMonitorDep
from backend.health.upstream import UpstreamSnapshot
from backend.scoring.ats_composite import evaluate_ats_gate

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["alive"]


class ReadyScoring(BaseModel):
    ats_composite: str
    calibration_version: str | None


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
    upstream: dict[str, UpstreamSnapshot]
    scoring: ReadyScoring


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse, "description": "A required component is not ready"}},
)
async def ready(readiness: ReadinessDep, upstream: UpstreamMonitorDep) -> JSONResponse:
    is_ready = readiness.is_ready
    gate = evaluate_ats_gate()
    body = ReadyResponse(
        status="ready" if is_ready else "not_ready",
        checks=readiness.snapshot(),
        upstream={"groq": upstream.snapshot()},
        scoring=ReadyScoring(
            ats_composite=gate.status, calibration_version=gate.calibration_version
        ),
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=200 if is_ready else 503)
