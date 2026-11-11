"""FastAPI dependencies exposing application-scoped services.

Services are created once in ``create_app`` and stored on ``app.state``, so
tests can build isolated applications with their own settings and fakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from backend.api.document_gate import DocumentProcessingGate
from backend.core.config import Settings
from backend.core.rate_limit import RateLimiter
from backend.graph.analysis import AnalysisWorkflow
from backend.graph.evaluation import EvaluationWorkflow
from backend.health.readiness import ReadinessRegistry
from backend.health.upstream import GroqUpstreamMonitor
from backend.llm.client import LLMClients
from backend.scoring.embeddings import EmbeddingService
from backend.services.interview import InterviewQuestionService
from backend.storage.report_store import ReportStore

if TYPE_CHECKING:  # backend.api.rate_limits imports this module
    from backend.api.rate_limits import AnalyzeConcurrencyGate


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


def get_interview_service(request: Request) -> InterviewQuestionService:
    return request.app.state.interview_service


def get_evaluation_workflow(request: Request) -> EvaluationWorkflow:
    return request.app.state.evaluation_workflow


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_analyze_gate(request: Request) -> AnalyzeConcurrencyGate:
    return request.app.state.analyze_gate


def get_document_gate(request: Request) -> DocumentProcessingGate:
    return request.app.state.document_gate


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
LLMClientsDep = Annotated[LLMClients, Depends(get_llm_clients)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
ReadinessDep = Annotated[ReadinessRegistry, Depends(get_readiness)]
UpstreamMonitorDep = Annotated[GroqUpstreamMonitor, Depends(get_upstream_monitor)]
ReportStoreDep = Annotated[ReportStore, Depends(get_report_store)]
AnalysisWorkflowDep = Annotated[AnalysisWorkflow, Depends(get_analysis_workflow)]
InterviewServiceDep = Annotated[InterviewQuestionService, Depends(get_interview_service)]
EvaluationWorkflowDep = Annotated[EvaluationWorkflow, Depends(get_evaluation_workflow)]
DocumentGateDep = Annotated[DocumentProcessingGate, Depends(get_document_gate)]
