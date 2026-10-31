"""FastAPI dependencies exposing application-scoped services.

Services are created once in ``create_app`` and stored on ``app.state``, so
tests can build isolated applications with their own settings and fakes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.core.config import Settings
from backend.health.readiness import ReadinessRegistry
from backend.health.upstream import GroqUpstreamMonitor


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_readiness(request: Request) -> ReadinessRegistry:
    return request.app.state.readiness


def get_upstream_monitor(request: Request) -> GroqUpstreamMonitor:
    return request.app.state.upstream_monitor


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ReadinessDep = Annotated[ReadinessRegistry, Depends(get_readiness)]
UpstreamMonitorDep = Annotated[GroqUpstreamMonitor, Depends(get_upstream_monitor)]
