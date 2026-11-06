"""Resume analysis workflow (approved Rev. 3, M7 decisions D0-D9).

```
START → guard → resume ─┬─► interview ──┐
                        ├─► metrics ────┼─► critic ─┬─ approve / flagged ─────────► aggregate
                        └─► skill_gap ──┘     ▲     ├─ revise, revisions < MAX ─► revise ─┐
                                              └─────┼──────────────────────────────────────┘
                                                    └─ revise, revisions ≥ MAX ─► cap ──► aggregate
aggregate → persist → END
```

* guard: M3 sanitisation and the M2/M5 length limits (typed 422s), then
  flag-only injection detection: heuristics and Prompt Guard (D5). Detection
  never rejects. With a job description, the embedding model must be
  available before any LLM call is made.
* resume: the M4 resume agent. Strict structured-output validation, output
  sanitisation and link neutralisation happen inside every agent call (the
  approved ``validate_output`` step), and the Critic re-validates every output.
* interview, metrics, skill_gap run in parallel. Without a job description,
  metrics and skill_gap produce nothing. Interview uses the M4 defaults (D9:
  all three levels, 3 per level). Metrics are computed once (M5) and never
  reach the Critic or a revision (M6 decision Q7); ATS stays not_calibrated.
* critic: the M6 Critic. Its verdict is ``approve``, ``revise`` or ``flagged``.
* revise (D1, D2): reruns only the targeted agents, each with its own Critic
  issues as feedback. If the resume profile is revised, its dependents
  (interview, and skill_gap with a job description) rerun too. Then the
  Critic reviews again.
* cap: a ``revise`` still unresolved after ``MAX_REVISIONS`` becomes
  ``flagged`` with every issue kept (M6 decision Q2). A ``revise`` whose
  targets cannot rerun (for example skill_gap without a job description) goes
  straight to cap without using the revision allowance or a second review.
* Parallel calls inside a node use a task group: if one fails, the others are
  cancelled and awaited, and the original typed error propagates.
* aggregate builds the derived report; persist stores it (Phase 1 store).

Fail closed (D3): any agent, Critic or metrics failure aborts the run, and no
report is built or stored. Deadline (D4): the whole run is bounded by
``ANALYSIS_DEADLINE_SECONDS``; on expiry outstanding work is cancelled, nothing
is stored, and ``AnalysisTimeoutError`` (504) is raised. (A metrics thread
cannot be interrupted; its result is discarded.) Persistence is the last
step, so a report is stored only for a run that completed in time.

The graph is compiled without a checkpointer: the sanitised texts in the state
exist only for the duration of a run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agents.base import AgentResult
from backend.agents.critic_agent import CriticInput, apply_revision_cap, run_critic_agent
from backend.agents.interview_agent import run_interview_agent
from backend.agents.resume_agent import run_resume_agent
from backend.agents.skill_gap_agent import run_skill_gap_agent
from backend.core.config import Settings
from backend.core.errors import AppError
from backend.core.request_id import get_request_id
from backend.graph.state import AnalysisState, NodeError
from backend.health.upstream import GroqUpstreamMonitor
from backend.llm.client import LLMClients, LLMRole
from backend.llm.structured import StructuredLLM
from backend.schemas.critic import AgentName, CriticIssue, CriticVerdict
from backend.schemas.interview import DEFAULT_COUNT_PER_LEVEL, LEVELS
from backend.schemas.report import AnalysisReport, CriticSummary, ReportMeta
from backend.scoring.embeddings import EmbeddingService
from backend.scoring.match import clean_job_description, clean_resume_text, compute_match
from backend.security.injection import PromptGuardClassifier, merge_flags, scan_untrusted_text
from backend.storage.report_store import ReportStamp, ReportStore

logger = logging.getLogger("backend.graph.analysis")

INTERVIEW_LEVELS = LEVELS
INTERVIEW_COUNT_PER_LEVEL = DEFAULT_COUNT_PER_LEVEL
_CLASSIFIER_NOTICES = {"partial", "unavailable", "skipped"}


class AnalysisTimeoutError(AppError):
    status_code = 504
    code = "analysis_timeout"
    message = "The analysis did not finish in time. Please try again."


@dataclass(frozen=True)
class AgentFunctions:
    """The agent entry points the workflow calls (replaceable in tests)."""

    resume: Callable[..., Awaitable[AgentResult[Any]]] = run_resume_agent
    interview: Callable[..., Awaitable[AgentResult[Any]]] = run_interview_agent
    skill_gap: Callable[..., Awaitable[AgentResult[Any]]] = run_skill_gap_agent
    critic: Callable[..., Awaitable[AgentResult[Any]]] = run_critic_agent


@dataclass(frozen=True)
class AnalysisDeps:
    settings: Settings
    llm_clients: LLMClients
    embeddings: EmbeddingService
    report_store: ReportStore
    prompt_guard: PromptGuardClassifier | None = None
    agents: AgentFunctions = field(default_factory=AgentFunctions)


def build_prompt_guard(
    settings: Settings, clients: LLMClients, monitor: GroqUpstreamMonitor | None = None
) -> PromptGuardClassifier | None:
    """The flag-only Prompt Guard classifier (D5), or ``None`` without an API key."""
    if not clients.configured:
        return None
    return PromptGuardClassifier(
        clients.raw_client(),
        model=settings.groq_model_prompt_guard,
        chunk_chars=settings.prompt_guard_chunk_chars,
        max_chunks=settings.prompt_guard_max_chunks,
        concurrency=settings.prompt_guard_concurrency,
        monitor=monitor,
    )


def feedback_for(verdict: CriticVerdict, agent: AgentName) -> list[CriticIssue]:
    """The published Critic issues for one agent only (D1)."""
    return [item for item in verdict.issues if item.agent == agent]


def revision_plan(targets: set[AgentName], *, has_job: bool) -> tuple[bool, bool, bool]:
    """Which agents rerun: (resume, interview, skill_gap). D2: a revised resume
    profile also reruns its dependents; skill_gap exists only with a job."""
    resume = "resume" in targets
    interview = resume or "interview" in targets
    skill_gap = has_job and (resume or "skill_gap" in targets)
    return resume, interview, skill_gap


def can_revise(verdict: CriticVerdict, *, has_job: bool) -> bool:
    """Whether a revise verdict targets at least one agent that can rerun."""
    return any(revision_plan(set(verdict.revise_targets), has_job=has_job))


async def run_concurrently(*calls: Awaitable[Any]) -> list[Any]:
    """Run awaitables concurrently; if one fails, cancel and await the rest.

    The first failure is re-raised as itself (not as an ``ExceptionGroup``), so
    the existing typed errors and their HTTP mapping are preserved.
    """
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(call) for call in calls]
    except ExceptionGroup as failures:
        raise failures.exceptions[0] from None
    return [task.result() for task in tasks]


def recursion_limit(max_revisions: int) -> int:
    """Supersteps of the longest legal run plus a margin (a backstop only)."""
    return 10 + 2 * max_revisions


@dataclass
class _Run:
    """Per-run bookkeeping passed through the graph config (never shared)."""

    deadline: float
    report: AnalysisReport | None = None


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _log_node(node: str, timing_ms: float) -> None:
    logger.info("analysis_node_completed", extra={"node": node, "timing_ms": timing_ms})


def build_analysis_graph(deps: AnalysisDeps) -> CompiledStateGraph:
    settings = deps.settings
    llm = StructuredLLM(deps.llm_clients)
    agents = deps.agents

    async def guard(state: AnalysisState) -> dict[str, Any]:
        start = time.perf_counter()
        resume = clean_resume_text(state["resume_text"], settings)
        raw_job = state.get("job_description")
        job = clean_job_description(raw_job, settings) if raw_job is not None else None
        if job is not None:
            deps.embeddings.get()  # 503 now, before any LLM call, if the model is unavailable
        # Detection scans the original inputs: the scanner sanitises internally and
        # reports hidden Unicode, which the cleaned texts no longer contain.
        sources = [(state["resume_text"], "resume")]
        if job is not None:
            sources.append((raw_job, "job_description"))
        scans = await run_concurrently(
            *(
                scan_untrusted_text(
                    text,
                    source,
                    classifier=deps.prompt_guard,
                    display_threshold=settings.prompt_guard_display_threshold,
                )
                for text, source in sources
            )
        )
        flags = [flag for scan in scans for flag in scan.flags]
        errors = [
            NodeError("guard", f"prompt_guard_{scan.classifier.status}")
            for scan in scans
            if scan.classifier.status in _CLASSIFIER_NOTICES
        ]
        ms = _elapsed_ms(start)
        _log_node("guard", ms)
        return {
            "resume_text": resume,
            "job_description": job,
            "security_flags": flags,
            "errors": errors,
            "revision_count": 0,
            "timings_ms": {"guard": ms},
        }

    async def resume(state: AnalysisState) -> dict[str, Any]:
        start = time.perf_counter()
        result = await agents.resume(llm, state["resume_text"])
        ms = _elapsed_ms(start)
        _log_node("resume", ms)
        return {
            "resume_profile": result.output,
            "security_flags": list(result.flags),
            "prompt_ids": {"resume": result.prompt_id},
            "timings_ms": {"resume": ms},
        }

    async def _interview(profile, feedback: list[CriticIssue] | None = None):
        return await agents.interview(
            llm,
            profile,
            levels=INTERVIEW_LEVELS,
            count_per_level=INTERVIEW_COUNT_PER_LEVEL,
            feedback=feedback,
        )

    async def interview(state: AnalysisState) -> dict[str, Any]:
        start = time.perf_counter()
        result = await _interview(state["resume_profile"])
        ms = _elapsed_ms(start)
        _log_node("interview", ms)
        return {
            "interview_questions": result.output,
            "security_flags": list(result.flags),
            "prompt_ids": {"interview": result.prompt_id},
            "timings_ms": {"interview": ms},
        }

    async def metrics(state: AnalysisState) -> dict[str, Any]:
        job = state.get("job_description")
        if job is None:
            return {"match_result": None}
        start = time.perf_counter()
        embedder = deps.embeddings.get()
        result = await asyncio.to_thread(compute_match, embedder, state["resume_text"], job)
        ms = _elapsed_ms(start)
        _log_node("metrics", ms)
        return {"match_result": result, "timings_ms": {"metrics": ms}}

    async def skill_gap(state: AnalysisState) -> dict[str, Any]:
        job = state.get("job_description")
        if job is None:
            return {"skill_gap": None}
        start = time.perf_counter()
        result = await agents.skill_gap(llm, state["resume_profile"], job)
        ms = _elapsed_ms(start)
        _log_node("skill_gap", ms)
        return {
            "skill_gap": result.output,
            "security_flags": list(result.flags),
            "prompt_ids": {"skill_gap": result.prompt_id},
            "timings_ms": {"skill_gap": ms},
        }

    async def critic(state: AnalysisState) -> dict[str, Any]:
        start = time.perf_counter()
        result = await agents.critic(
            llm,
            CriticInput(
                resume_text=state["resume_text"],
                profile=state["resume_profile"],
                job_description=state.get("job_description"),
                interview=state["interview_questions"],
                interview_levels=INTERVIEW_LEVELS,
                count_per_level=INTERVIEW_COUNT_PER_LEVEL,
                skill_gap=state.get("skill_gap"),
                security_flags=tuple(state.get("security_flags", ())),
            ),
        )
        verdict: CriticVerdict = result.output
        key = f"critic_{state.get('revision_count', 0)}"
        ms = _elapsed_ms(start)
        _log_node("critic", ms)
        return {
            "critic_verdict": verdict,
            "critic_history": [verdict],
            "revise_targets": list(verdict.revise_targets),
            "security_flags": list(result.flags),
            "prompt_ids": {"critic": result.prompt_id},
            "timings_ms": {key: ms},
        }

    def after_critic(state: AnalysisState) -> str:
        verdict = state["critic_verdict"]
        if verdict.verdict != "revise":
            return "aggregate"
        has_job = state.get("job_description") is not None
        if state.get("revision_count", 0) < settings.max_revisions and can_revise(
            verdict, has_job=has_job
        ):
            return "revise"
        return "cap"

    async def revise(state: AnalysisState) -> dict[str, Any]:
        start = time.perf_counter()
        verdict = state["critic_verdict"]
        has_job = state.get("job_description") is not None
        redo_resume, redo_interview, redo_gap = revision_plan(
            set(verdict.revise_targets), has_job=has_job
        )
        update: dict[str, Any] = {"security_flags": [], "prompt_ids": {}}
        profile = state["resume_profile"]
        if redo_resume:
            result = await agents.resume(
                llm, state["resume_text"], feedback=feedback_for(verdict, "resume")
            )
            profile = result.output
            update["resume_profile"] = profile
            update["security_flags"] += list(result.flags)
            update["prompt_ids"]["resume"] = result.prompt_id

        calls: dict[str, Awaitable[AgentResult[Any]]] = {}
        if redo_interview:
            calls["interview"] = _interview(profile, feedback_for(verdict, "interview"))
        if redo_gap:
            calls["skill_gap"] = agents.skill_gap(
                llm, profile, state["job_description"], feedback=feedback_for(verdict, "skill_gap")
            )
        results = dict(zip(calls, await run_concurrently(*calls.values()), strict=True))
        interview_result, gap_result = results.get("interview"), results.get("skill_gap")
        if interview_result is not None:
            update["interview_questions"] = interview_result.output
            update["security_flags"] += list(interview_result.flags)
            update["prompt_ids"]["interview"] = interview_result.prompt_id
        if gap_result is not None:
            update["skill_gap"] = gap_result.output
            update["security_flags"] += list(gap_result.flags)
            update["prompt_ids"]["skill_gap"] = gap_result.prompt_id
        revision = state.get("revision_count", 0) + 1
        ms = _elapsed_ms(start)
        _log_node("revise", ms)
        return {**update, "revision_count": revision, "timings_ms": {f"revise_{revision}": ms}}

    async def cap(state: AnalysisState) -> dict[str, Any]:
        verdict = state["critic_verdict"]
        revisions_done = state.get("revision_count", 0)
        if not can_revise(verdict, has_job=state.get("job_description") is not None):
            # No targeted agent can rerun: the allowance is not spent, but no
            # revision is possible either, so the verdict is unresolved now.
            revisions_done = settings.max_revisions
        capped = apply_revision_cap(
            verdict, revisions_done=revisions_done, max_revisions=settings.max_revisions
        )
        return {"critic_verdict": capped, "revise_targets": []}

    async def aggregate(state: AnalysisState) -> dict[str, Any]:
        verdict = state["critic_verdict"]
        if verdict.verdict == "revise":  # pragma: no cover - routing makes this unreachable
            raise RuntimeError("an unresolved revise verdict reached aggregation")
        has_job = state.get("job_description") is not None
        roles: list[LLMRole] = ["resume", "interview", "critic"]
        if has_job:
            roles.insert(2, "skill_gap")
        models = {role: deps.llm_clients.model_for(role) for role in roles}
        if deps.prompt_guard is not None:
            models["prompt_guard"] = settings.groq_model_prompt_guard
        draft = {
            "resume_profile": state["resume_profile"],
            "interview_questions": state["interview_questions"],
            "match": state.get("match_result"),
            "skill_gap": state.get("skill_gap"),
            "critic": CriticSummary(
                verdict=verdict.verdict,
                issues=list(verdict.issues),
                revisions=state.get("revision_count", 0),
            ),
            "security_flags": list(merge_flags(state.get("security_flags", []))),
            "meta": ReportMeta(
                models=models,
                prompt_versions=dict(state.get("prompt_ids", {})),
                timings_ms=dict(state.get("timings_ms", {})),
            ),
        }
        return {"report_draft": draft}

    async def persist(state: AnalysisState, config: RunnableConfig) -> dict[str, Any]:
        run: _Run = config["configurable"]["analysis_run"]
        if asyncio.get_running_loop().time() >= run.deadline:
            raise AnalysisTimeoutError()
        draft = state["report_draft"]

        def build(stamp: ReportStamp) -> AnalysisReport:
            return AnalysisReport(
                report_id=stamp.report_id,
                created_at=stamp.created_at,
                expires_at=stamp.expires_at,
                **draft,
            )

        report = await deps.report_store.create(build)
        run.report = report
        return {"report_id": str(report.report_id), "report": report}

    graph = StateGraph(AnalysisState)
    for name, node in (
        ("guard", guard),
        ("resume", resume),
        ("interview", interview),
        ("metrics", metrics),
        ("skill_gap", skill_gap),
        ("critic", critic),
        ("revise", revise),
        ("cap", cap),
        ("aggregate", aggregate),
        ("persist", persist),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "guard")
    graph.add_edge("guard", "resume")
    for branch in ("interview", "metrics", "skill_gap"):
        graph.add_edge("resume", branch)
    graph.add_edge(["interview", "metrics", "skill_gap"], "critic")
    graph.add_conditional_edges(
        "critic", after_critic, {"aggregate": "aggregate", "revise": "revise", "cap": "cap"}
    )
    graph.add_edge("revise", "critic")
    graph.add_edge("cap", "aggregate")
    graph.add_edge("aggregate", "persist")
    graph.add_edge("persist", END)
    return graph.compile()  # no checkpointer: state is never persisted


class AnalysisWorkflow:
    """Runs the compiled analysis graph under the overall deadline."""

    def __init__(self, deps: AnalysisDeps) -> None:
        self._deps = deps
        self.graph = build_analysis_graph(deps)

    async def run(self, *, resume_text: str, job_description: str | None = None) -> AnalysisReport:
        settings = self._deps.settings
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        run = _Run(deadline=loop.time() + settings.analysis_deadline_seconds)
        config: RunnableConfig = {
            "configurable": {"analysis_run": run},
            "recursion_limit": recursion_limit(settings.max_revisions),
        }
        state: AnalysisState = {
            "request_id": get_request_id(),
            "resume_text": resume_text,
            "job_description": job_description,
        }
        try:
            async with asyncio.timeout_at(run.deadline) as scope:
                final = await self.graph.ainvoke(state, config=config)
        except TimeoutError:
            if not scope.expired():
                raise
            if run.report is not None:  # stored just as the deadline passed: the run finished
                return run.report
            logger.warning("analysis_timed_out", extra={"error_code": AnalysisTimeoutError.code})
            raise AnalysisTimeoutError() from None
        report = final.get("report")
        if report is None:  # pragma: no cover - persist is the only path to END
            raise RuntimeError("the analysis workflow ended without a report")
        logger.info(
            "analysis_completed",
            extra={
                "critic_verdict": report.critic.verdict,
                "revisions": report.critic.revisions,
                "timing_ms": _elapsed_ms(started),
                "security_flag_types": sorted({f.type for f in report.security_flags}),
            },
        )
        return report
