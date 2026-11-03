"""Answer-evaluation workflow (M9 Phase 2; decisions Q2-Q8, N1, N3).

```
START → guard → evaluate → END
```

* guard:
  1. Looks up the report (read-only). Unknown, malformed and expired IDs raise
     the existing 404 ``report_not_found``.
  2. Resolves the question: a ``question_id`` must name a question in the
     stored report (otherwise 404 ``question_not_found``) and its stored text
     is used; a client-supplied question has already been sanitised and
     validated by its schema. The level always comes from the validated ID.
  3. Sanitises the answer (M3). Past ``MAX_ANSWER_CHARS`` it raises 422
     ``answer_too_long``; empty once sanitised, 422 ``empty_answer``.
  4. Scans for injection, flag-only (detection never rejects): the answer with
     the heuristics and Prompt Guard, and a client-supplied question text with
     the heuristics only, plus ``hidden_unicode`` if the question's schema found
     hidden Unicode before sanitising it. Prompt Guard has
     ``PROMPT_GUARD_BUDGET_SECONDS``; if it has not finished by then it is
     cancelled and the answer is scanned again with the heuristics only. The
     classifier status is logged, never returned (N3).
* evaluate: the M4 answer-evaluator agent, unchanged. It rates the five fixed
  criteria (relevance, technical_accuracy, depth, clarity, structure); the
  overall rating is their mean, rounded half up, computed in Python.

The run is bounded by ``EVALUATION_DEADLINE_SECONDS``. On expiry the graph is
cancelled and ``evaluation_timeout`` (504) is raised. Every call here is async
I/O, so cancellation really stops the work locally: outstanding HTTP requests
are abandoned and their connections closed, and no thread is left running.
Groq may still finish a request it has already received. Any failure aborts
the run with its typed error; no partial evaluation is returned.

Nothing is persisted: the graph has no checkpointer, the report store is only
read, and the answer exists in the state for one run only. The answer, the
question text and the report ID are never logged or returned.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agents.answer_evaluator_agent import run_answer_evaluator_agent
from backend.agents.base import AgentResult
from backend.core.config import Settings
from backend.llm.client import LLMClients
from backend.llm.structured import StructuredLLM
from backend.schemas.interview import AnswerEvaluation, Level
from backend.schemas.interview_api import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    ClientQuestion,
    level_of,
)
from backend.schemas.report import AnalysisReport
from backend.schemas.security import SecurityFlag
from backend.security.injection import (
    InjectionScan,
    PromptGuardClassifier,
    merge_flags,
    scan_untrusted_text,
)
from backend.security.sanitize import sanitize_text
from backend.services.errors import (
    AnswerTooLongError,
    EmptyAnswerError,
    EvaluationTimeoutError,
    QuestionNotFoundError,
    ReportNotFoundError,
)
from backend.storage.report_store import ReportStore

logger = logging.getLogger("backend.graph.evaluation")

# Decision N1: Prompt Guard's share of the evaluation deadline.
PROMPT_GUARD_BUDGET_SECONDS = 30.0

EvaluatorAgent = Callable[..., Awaitable[AgentResult[Any]]]


class EvaluationState(TypedDict, total=False):
    report_id: str
    question_id: str | None
    client_question: ClientQuestion | None
    answer: str  # raw, then sanitised by guard; this run only, never persisted
    question_text: str
    target_question_id: str
    level: Level
    security_flags: Annotated[list[SecurityFlag], operator.add]
    classifier_status: str  # logged only, never returned
    evaluation: AnswerEvaluation | None
    prompt_id: str


@dataclass(frozen=True)
class EvaluationDeps:
    settings: Settings
    llm_clients: LLMClients
    report_store: ReportStore
    prompt_guard: PromptGuardClassifier | None = None
    evaluator: EvaluatorAgent = run_answer_evaluator_agent


def stored_question_text(report: AnalysisReport, question_id: str) -> str:
    """The text of a question in the stored report, or ``question_not_found``."""
    for question in getattr(report.interview_questions, level_of(question_id)):
        if question.id == question_id:
            return question.question
    raise QuestionNotFoundError()


def clean_answer(answer: str, settings: Settings) -> str:
    """The sanitised answer within ``MAX_ANSWER_CHARS``, or a typed 422.

    The limit applies to the text that is evaluated: sanitised (M3) and stripped of
    surrounding whitespace. The raw answer is already bounded by its schema.
    """
    cleaned = sanitize_text(answer).text.strip()
    if not cleaned:
        raise EmptyAnswerError()
    if len(cleaned) > settings.max_answer_chars:
        raise AnswerTooLongError()
    return cleaned


async def scan_answer(
    answer: str, classifier: PromptGuardClassifier | None, settings: Settings
) -> InjectionScan:
    """Heuristics plus Prompt Guard within its budget; heuristics only after it."""
    scan = scan_untrusted_text(
        answer,
        "answer",
        classifier=classifier,
        display_threshold=settings.prompt_guard_display_threshold,
    )
    if classifier is None:
        return await scan
    try:
        async with asyncio.timeout(PROMPT_GUARD_BUDGET_SECONDS):
            return await scan
    except TimeoutError:
        logger.warning(
            "prompt_guard_timed_out",
            extra={"node": "guard", "upstream_status": "timeout", "component": "answer"},
        )
    return await scan_untrusted_text(
        answer,
        "answer",
        classifier=None,
        display_threshold=settings.prompt_guard_display_threshold,
    )


def build_evaluation_graph(deps: EvaluationDeps) -> CompiledStateGraph:
    settings = deps.settings
    llm = StructuredLLM(deps.llm_clients)

    async def guard(state: EvaluationState) -> dict[str, Any]:
        start = time.perf_counter()
        report = await deps.report_store.get(state["report_id"])
        if report is None:
            raise ReportNotFoundError()
        client_question = state.get("client_question")
        if client_question is not None:
            target, question_text = client_question.id, client_question.question
        else:
            target = str(state["question_id"])
            question_text = stored_question_text(report, target)
        raw_answer = state["answer"]
        answer = clean_answer(raw_answer, settings)
        # The scanner sanitises internally and reports hidden Unicode, so it gets
        # the raw answer (as in the M7 guard). The client question was sanitised by
        # its schema, which recorded whether hidden Unicode was present; its text is
        # scanned with the heuristics only (no Prompt Guard call).
        scans = [scan_answer(raw_answer, deps.prompt_guard, settings)]
        if client_question is not None:
            scans.append(
                scan_untrusted_text(
                    question_text,
                    "question",
                    classifier=None,
                    display_threshold=settings.prompt_guard_display_threshold,
                )
            )
        results = await asyncio.gather(*scans)
        status = results[0].classifier.status
        flags = [flag for result in results for flag in result.flags]
        if client_question is not None and client_question.hidden_unicode:
            flags.append(SecurityFlag(type="hidden_unicode", source="question"))
        logger.info(
            "evaluation_node_completed",
            extra={
                "node": "guard",
                "upstream_status": status,
                "timing_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return {
            "answer": answer,
            "question_text": question_text,
            "target_question_id": target,
            "level": level_of(target),
            "security_flags": flags,
            "classifier_status": status,
        }

    async def evaluate(state: EvaluationState) -> dict[str, Any]:
        start = time.perf_counter()
        result = await deps.evaluator(
            llm, question=state["question_text"], answer=state["answer"], level=state["level"]
        )
        logger.info(
            "evaluation_node_completed",
            extra={
                "node": "evaluate",
                "model": deps.llm_clients.model_for("evaluator"),
                "attempts": result.attempts,
                "timing_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return {
            "evaluation": result.output,
            "security_flags": list(result.flags),
            "prompt_id": result.prompt_id,
        }

    graph = StateGraph(EvaluationState)
    graph.add_node("guard", guard)
    graph.add_node("evaluate", evaluate)
    graph.add_edge(START, "guard")
    graph.add_edge("guard", "evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()  # no checkpointer: state is never persisted


class EvaluationWorkflow:
    """Runs the compiled evaluation graph under the overall deadline."""

    def __init__(self, deps: EvaluationDeps) -> None:
        self._deps = deps
        self.graph = build_evaluation_graph(deps)

    async def run(self, request: AnswerEvaluationRequest) -> AnswerEvaluationResponse:
        started = time.perf_counter()
        state: EvaluationState = {
            "report_id": request.report_id,
            "question_id": request.question_id,
            "client_question": request.question,
            "answer": request.answer,
        }
        try:
            async with asyncio.timeout(self._deps.settings.evaluation_deadline_seconds) as scope:
                final = await self.graph.ainvoke(state)
        except TimeoutError:
            if not scope.expired():
                raise
            logger.warning(
                "evaluation_timed_out", extra={"error_code": EvaluationTimeoutError.code}
            )
            raise EvaluationTimeoutError() from None
        evaluation = final.get("evaluation")
        if evaluation is None:  # pragma: no cover - evaluate is the only path to END
            raise RuntimeError("the evaluation workflow ended without an evaluation")
        response = AnswerEvaluationResponse(
            question_id=final["target_question_id"],
            level=final["level"],
            evaluation=evaluation,
            security_flags=list(merge_flags(final.get("security_flags", []))),
        )
        logger.info(
            "evaluation_completed",
            extra={
                "timing_ms": round((time.perf_counter() - started) * 1000, 1),
                "security_flag_types": sorted({f.type for f in response.security_flags}),
            },
        )
        return response
