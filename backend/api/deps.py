"""FastAPI dependencies exposing application-scoped services.

Services are created once in ``create_app`` and stored on ``app.state``, so
tests can build isolated applications with their own settings and fakes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.core.config import Settings
from backend.graph.analysis import AnalysisWorkflow
from backend.health.readiness import ReadinessRegistry
from backend.health.upstream import GroqUpstreamMonitor
from backend.llm.client import LLMClients
from backend.scoring.embeddings import EmbeddingService
from backend.storage.report_store import ReportStore


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_readiness(request: Request) -> ReadinessRegistry:
    return request.app.state.readiness


def get_upstream_monitor(request: Request) -> GroqUpstreamMonitor:
    return request.app.state.upstream_monitor


def get_llm_clients(request: Request) -> LLMClients:
    return request.app.state.llm_clients


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embeddings


def get_report_store(request: Request) -> ReportStore:
    return request.app.state.report_store


def get_analysis_workflow(request: Request) -> AnalysisWorkflow:
    return request.app.state.analysis_workflow


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
LLMClientsDep = Annotated[LLMClients, Depends(get_llm_clients)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
ReadinessDep = Annotated[ReadinessRegistry, Depends(get_readiness)]
UpstreamMonitorDep = Annotated[GroqUpstreamMonitor, Depends(get_upstream_monitor)]
ReportStoreDep = Annotated[ReportStore, Depends(get_report_store)]
AnalysisWorkflowDep = Annotated[AnalysisWorkflow, Depends(get_analysis_workflow)]
