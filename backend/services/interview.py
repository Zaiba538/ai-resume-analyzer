"""Interview question generation from a stored report (M9 Phase 2; decisions Q1, Q3, Q7).

``InterviewQuestionService.generate`` looks up the report and runs the M4
interview agent on its stored ``ResumeProfile``, which the M7 workflow already
validated and the Critic reviewed. Only the requested levels are generated,
each with exactly ``count_per_level`` questions (M4 decision B). The agent's
structured-output validation, output sanitisation and link neutralisation apply
unchanged, and its security flags are returned.

* Read-only: the report store is only read. Generated questions are returned,
  never stored, and the stored report is not changed.
* Unknown, malformed and expired report IDs raise the existing 404
  ``report_not_found``.
* The whole call, lookup included, is bounded by ``INTERVIEW_DEADLINE_SECONDS``.
  On expiry the outstanding agent call is cancelled and ``interview_timeout``
  (504) is raised. Cancellation is local: the HTTP request to Groq is abandoned
  and its connection closed, but Groq may still finish a request it has
  already received.
* Nothing here logs profile, question or report content, or the report ID.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from backend.agents.base import AgentResult
from backend.agents.interview_agent import run_interview_agent
from backend.core.config import Settings
from backend.llm.client import LLMClients
from backend.llm.structured import StructuredLLM
from backend.schemas.interview_api import InterviewQuestionsRequest, InterviewQuestionsResponse
from backend.security.injection import merge_flags
from backend.services.errors import InterviewTimeoutError, ReportNotFoundError
from backend.storage.report_store import ReportStore

logger = logging.getLogger("backend.services.interview")

InterviewAgent = Callable[..., Awaitable[AgentResult[Any]]]


class InterviewQuestionService:
    """Generates interview questions for a stored report; never writes to the store."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm_clients: LLMClients,
        report_store: ReportStore,
        agent: InterviewAgent = run_interview_agent,
    ) -> None:
        self._settings = settings
        self._llm_clients = llm_clients
        self._llm = StructuredLLM(llm_clients)
        self._store = report_store
        self._agent = agent

    async def generate(self, request: InterviewQuestionsRequest) -> InterviewQuestionsResponse:
        started = time.perf_counter()
        levels = request.requested_levels
        try:
            async with asyncio.timeout(self._settings.interview_deadline_seconds) as scope:
                report = await self._store.get(request.report_id)  # malformed IDs: never looked up
                if report is None:
                    raise ReportNotFoundError()
                result = await self._agent(
                    self._llm,
                    report.resume_profile,
                    levels=levels,
                    count_per_level=request.count_per_level,
                )
        except TimeoutError:
            if not scope.expired():
                raise
            logger.warning(
                "interview_questions_timed_out", extra={"error_code": InterviewTimeoutError.code}
            )
            raise InterviewTimeoutError() from None
        response = InterviewQuestionsResponse(
            levels=list(levels),
            count_per_level=request.count_per_level,
            questions=result.output,
            security_flags=list(merge_flags(result.flags)),
        )
        logger.info(
            "interview_questions_generated",
            extra={
                "node": "interview",
                "model": self._llm_clients.model_for("interview"),
                "attempts": result.attempts,
                "timing_ms": round((time.perf_counter() - started) * 1000, 1),
                "security_flag_types": sorted({f.type for f in response.security_flags}),
            },
        )
        return response
