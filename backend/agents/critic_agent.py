"""Critic agent (M6): reviews M4 agent outputs and derives a verdict server-side.

Pipeline:

1. Inputs are sanitised (M3) and validated. A skill-gap report requires a job
   description. Requested interview levels and counts follow the M4 rules.
2. Every agent output is re-validated against its schema; a violation is a
   ``schema_violation`` issue and that output skips the remaining checks.
3. Deterministic checks (``critic_checks``) report what they can establish:
   unsupported skills, skill-gap contradictions, interview completeness and
   duplicates, and suspicious outputs (instruction-like generated text matching
   an injection signal in the inputs). Undecidable claims and suspicious fields
   are passed to the LLM as ``claims_to_verify`` and ``suspicious_outputs``.
4. The LLM review (role ``critic``) always runs. Inputs and outputs appear only
   in untrusted user-message blocks, and the model returns categorised issues
   (``CriticReviewDraft``). Failures propagate as the existing typed errors:
   fail closed, never a deterministic-only approval (decision Q5).
5. Issues are merged (one per agent, field and category: the most severe,
   with the other distinct messages appended), at most ``MAX_CRITIC_ISSUES``
   are selected for publication, and the verdict is derived from them by
   ``derive_verdict``.

Verdict rules (decisions Q1, Q2): confirmed injection influence -> flagged;
otherwise any medium or high issue -> revise, targeting the responsible agents;
otherwise approve. Confirmation (``injection_confirmed``) requires a suspicious
output field (deterministic evidence in the generated text plus a matching
input signal) *and* an LLM ``injection_influence`` issue for the same field. No
single signal confirms: an input-only flag, a heuristic match or an LLM claim
alone is reported as an issue and leads to revise, never directly to flagged.
``apply_revision_cap`` turns a revise verdict that is still unresolved after
``MAX_REVISIONS`` into flagged, keeping every issue. The revision loop
itself belongs to the M7 workflow.

The Critic never calculates, modifies or replaces scores, and M5 metrics are
not inputs (decision Q7). Content is never truncated (decision Q9): very large
inputs may exceed the model's context window, in which case the provider error
is returned as the existing typed error.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError

from backend.agents.base import (
    AgentInputError,
    AgentResult,
    clean_input,
    sanitize_data,
    sanitize_model,
)
from backend.agents.critic_checks import (
    CheckResult,
    Claim,
    check_interview,
    check_profile_grounding,
    check_schema,
    check_skill_gap,
    check_suspicious_outputs,
    input_injection_types,
    issue,
)
from backend.llm.prompts import build_messages
from backend.llm.prompts.critic import CRITIC_PROMPT
from backend.llm.structured import StructuredLLM
from backend.schemas.critic import (
    MAX_CRITIC_ISSUES,
    MAX_ISSUE_MESSAGE,
    AgentName,
    CriticReviewDraft,
    CriticVerdict,
    ReviewIssue,
)
from backend.schemas.interview import (
    DEFAULT_COUNT_PER_LEVEL,
    LEVELS,
    MAX_COUNT_PER_LEVEL,
    MIN_COUNT_PER_LEVEL,
    InterviewQuestions,
    normalise_levels,
)
from backend.schemas.resume import ResumeProfile
from backend.schemas.security import SecurityFlag
from backend.schemas.skill_gap import SkillGapReport

logger = logging.getLogger("backend.agents.critic")

AGENT_ORDER: tuple[AgentName, ...] = ("resume", "interview", "skill_gap")
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
_REVISE_SEVERITIES = frozenset({"medium", "high"})
_PAYLOAD_KEYS: dict[AgentName, str] = {
    "resume": "profile",
    "interview": "interview",
    "skill_gap": "skill_gap",
}


@dataclass(frozen=True)
class CriticInput:
    resume_text: str
    profile: ResumeProfile
    job_description: str | None = None
    interview: InterviewQuestions | None = None
    interview_levels: Sequence[str] = LEVELS
    count_per_level: int = DEFAULT_COUNT_PER_LEVEL
    skill_gap: SkillGapReport | None = None
    security_flags: Sequence[SecurityFlag] = ()


# --- verdict --------------------------------------------------------------------------


def _paths_overlap(a: str, b: str) -> bool:
    """Same field, or one path contains the other (``projects`` / ``projects[0]``)."""
    a, b = a.casefold(), b.casefold()
    short, long = sorted((a, b), key=len)
    return a == b or (long.startswith(short) and long[len(short)] in ".[")


def injection_confirmed(
    suspicious: Iterable[ReviewIssue], llm_issues: Iterable[ReviewIssue]
) -> bool:
    """Deterministic evidence and an LLM judgement agree on the same output field."""
    claims = [i for i in llm_issues if i.category == "injection_influence"]
    return any(
        claim.agent == evidence.agent and _paths_overlap(claim.field, evidence.field)
        for evidence in suspicious
        for claim in claims
    )


def _publish_order(item: ReviewIssue) -> tuple[int, bool]:
    # Severity first; within a severity, injection evidence first (stable otherwise).
    return _SEVERITY_RANK[item.severity], item.category != "injection_influence"


def select_published(issues: Iterable[ReviewIssue]) -> list[ReviewIssue]:
    """At most ``MAX_CRITIC_ISSUES`` issues, most severe first.

    When the limit applies, the most severe issue of every agent that needs
    revision is always kept, so cutting issues never removes an agent's only
    reason for revision.
    """
    ordered = sorted(issues, key=_publish_order)  # stable
    if len(ordered) <= MAX_CRITIC_ISSUES:
        return ordered
    keep: set[int] = set()
    for agent in AGENT_ORDER:
        first = next(
            (n for n, i in enumerate(ordered) if i.agent == agent),
            None,
        )
        if first is not None and ordered[first].severity in _REVISE_SEVERITIES:
            keep.add(first)
    for n in range(len(ordered)):
        if len(keep) >= MAX_CRITIC_ISSUES:
            break
        keep.add(n)
    return [ordered[n] for n in sorted(keep)]


def derive_verdict(issues: Iterable[ReviewIssue], *, injection_confirmed: bool) -> CriticVerdict:
    """Server-side verdict from validated issues (approved decisions Q1 and Q2).

    The verdict and ``revise_targets`` are derived from the published issues
    (``select_published``). Issue categories never decide ``flagged`` on their
    own; only the explicit confirmation does.
    """
    selected = select_published(issues)
    published = [i.published() for i in selected]
    if injection_confirmed:
        return CriticVerdict(verdict="flagged", issues=published, revise_targets=[])
    targets = {i.agent for i in selected if i.severity in _REVISE_SEVERITIES}
    if targets:
        return CriticVerdict(
            verdict="revise",
            issues=published,
            revise_targets=[agent for agent in AGENT_ORDER if agent in targets],
        )
    return CriticVerdict(verdict="approve", issues=published, revise_targets=[])


def apply_revision_cap(
    verdict: CriticVerdict, *, revisions_done: int, max_revisions: int
) -> CriticVerdict:
    """A revise verdict still unresolved after ``max_revisions`` becomes flagged."""
    if revisions_done < 0 or max_revisions < 0:
        raise ValueError("revision counts must be non-negative")
    if verdict.verdict == "revise" and revisions_done >= max_revisions:
        return CriticVerdict(verdict="flagged", issues=list(verdict.issues), revise_targets=[])
    return verdict


def normalize_field(item: ReviewIssue) -> ReviewIssue:
    """Make an LLM field path agent-relative, as the deterministic checks report it.

    The model sees outputs nested under their payload key and may report
    ``profile.skills[2]`` for ``skills[2]`` (observed in the live test).
    """
    prefix = f"{_PAYLOAD_KEYS[item.agent]}."
    if item.field.startswith(prefix) and len(item.field) > len(prefix):
        return item.model_copy(update={"field": item.field[len(prefix) :]})
    return item


def _message_key(message: str) -> str:
    return " ".join(message.casefold().split())


def _combine(primary: ReviewIssue, others: Sequence[ReviewIssue]) -> ReviewIssue:
    """``primary`` with the distinct messages of ``others`` appended, within the limit.

    Messages that do not fit are counted in an explicit "(+N more)" marker.
    """
    seen = {_message_key(primary.message)}
    extras: list[str] = []
    for other in others:
        key = _message_key(other.message)
        if key not in seen:
            seen.add(key)
            extras.append(other.message)
    if not extras:
        return primary
    room = MAX_ISSUE_MESSAGE - len(" (+99 more)")
    message, omitted = primary.message, 0
    for extra in extras:
        candidate = f"{message} Also: {extra}"
        if len(candidate) <= room:
            message = candidate
        else:
            omitted += 1
    if omitted:
        suffix = f" (+{omitted} more)"
        if len(message) + len(suffix) > MAX_ISSUE_MESSAGE:
            message = message[: MAX_ISSUE_MESSAGE - len(suffix) - 1].rstrip() + "…"
        message += suffix
    return ReviewIssue.model_validate({**primary.model_dump(), "message": message})


def merge_issues(*groups: Iterable[ReviewIssue]) -> list[ReviewIssue]:
    """One issue per agent, field and category, never losing the highest severity.

    Issues are grouped by key in first-seen order (deterministic checks are
    passed first). Each group publishes its most severe issue (the earliest on a
    tie) with the distinct messages of the rest appended as further evidence.
    """
    grouped: dict[tuple[str, str, str], list[ReviewIssue]] = {}
    for group in groups:
        for item in group:
            grouped.setdefault((item.agent, item.field.casefold(), item.category), []).append(item)
    merged: list[ReviewIssue] = []
    for items in grouped.values():
        primary = min(items, key=lambda i: _SEVERITY_RANK[i.severity])  # first on ties
        merged.append(_combine(primary, [i for i in items if i is not primary]))
    return merged


# --- agent ----------------------------------------------------------------------------


def _validated_output[T: BaseModel](
    agent: AgentName, model: T | None, result: CheckResult
) -> T | None:
    """Schema-check and sanitise one agent output; None if it is unusable."""
    if model is None:
        return None
    violation = check_schema(agent, model)
    if violation is not None:
        result.issues.append(violation)
        return None
    try:
        return sanitize_model(model)
    except ValidationError:
        result.issues.append(
            issue(
                agent,
                "schema",
                "schema_violation",
                "schema_violation",
                "Output becomes invalid once invisible or control characters are removed.",
            )
        )
        return None


def _dump_for_review(model: BaseModel | None) -> object:
    """Sanitised JSON-ready output for the LLM; unserialisable objects are marked.

    Raw outputs are dumped (so schema-violating ones are still reviewed), and
    every string is sanitised because generated text derives from untrusted input.
    """
    if model is None:
        return None
    try:
        return sanitize_data(model.model_dump(mode="json", warnings=False))
    except PydanticSerializationError:
        return {"unserialisable_output": type(model).__name__}


def _agent_outputs_json(
    critic_input: CriticInput,
    levels: tuple[str, ...],
    outputs: dict[AgentName, BaseModel | None],
    ambiguous: Sequence[Claim],
    suspicious: Sequence[ReviewIssue],
) -> str:
    payload = {
        "profile": _dump_for_review(critic_input.profile),
        "interview": _dump_for_review(critic_input.interview),
        "skill_gap": _dump_for_review(critic_input.skill_gap),
        "requested_interview_levels": list(levels) if critic_input.interview else [],
        "count_per_level": critic_input.count_per_level if critic_input.interview else None,
        "claims_to_verify": [{"agent": c.agent, "field": c.field} for c in ambiguous],
        "suspicious_outputs": [{"agent": i.agent, "field": i.field} for i in suspicious],
        "input_security_flags": sorted({f.type for f in critic_input.security_flags}),
        "validated_outputs": sorted(agent for agent, model in outputs.items() if model),
    }
    return json.dumps(payload, ensure_ascii=False)


async def run_critic_agent(
    llm: StructuredLLM, critic_input: CriticInput
) -> AgentResult[CriticVerdict]:
    resume = clean_input(critic_input.resume_text, field="resume")
    job = (
        clean_input(critic_input.job_description, field="job description")
        if critic_input.job_description is not None
        else None
    )
    if critic_input.skill_gap is not None and job is None:
        raise AgentInputError("A skill-gap report can only be reviewed with its job description.")
    count = critic_input.count_per_level
    if isinstance(count, bool) or not isinstance(count, int):
        raise AgentInputError("count_per_level must be an integer.")
    if not MIN_COUNT_PER_LEVEL <= count <= MAX_COUNT_PER_LEVEL:
        raise AgentInputError(
            f"count_per_level must be between {MIN_COUNT_PER_LEVEL} and {MAX_COUNT_PER_LEVEL}."
        )
    try:
        levels = normalise_levels(critic_input.interview_levels)
    except ValueError as exc:
        raise AgentInputError("levels must be beginner, intermediate and/or advanced.") from exc

    checks = CheckResult()
    profile = _validated_output("resume", critic_input.profile, checks)
    interview = _validated_output("interview", critic_input.interview, checks)
    skill_gap = _validated_output("skill_gap", critic_input.skill_gap, checks)

    input_types = input_injection_types(resume, job, critic_input.security_flags)
    suspicious: list[ReviewIssue] = []
    if profile is not None:
        checks.extend(check_profile_grounding(profile, resume))
        suspicious += check_suspicious_outputs("resume", profile, input_types)
    if interview is not None:
        checks.extend(check_interview(interview, levels, count))
        suspicious += check_suspicious_outputs("interview", interview, input_types)
    if skill_gap is not None and job is not None:
        checks.extend(check_skill_gap(skill_gap, resume, job))
        suspicious += check_suspicious_outputs("skill_gap", skill_gap, input_types)
    checks.issues.extend(suspicious)

    outputs_json = _agent_outputs_json(
        critic_input,
        levels,
        {"resume": profile, "interview": interview, "skill_gap": skill_gap},
        checks.ambiguous,
        suspicious,
    )
    review = await llm.generate(
        "critic",
        CriticReviewDraft,
        build_messages(
            CRITIC_PROMPT,
            {"resume": resume, "job_description": job or "", "agent_outputs": outputs_json},
        ),
        source_texts=[resume, job or "", outputs_json],
    )
    llm_issues = [normalize_field(item) for item in review.output.issues]
    verdict = derive_verdict(
        merge_issues(checks.issues, llm_issues),
        injection_confirmed=injection_confirmed(suspicious, llm_issues),
    )
    logger.info("critic_completed", extra={"node": "critic", "critic_verdict": verdict.verdict})
    return AgentResult.from_structured(review, output=verdict, prompt_id=CRITIC_PROMPT.id)
