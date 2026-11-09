"""Deterministic Critic checks (M6): pure functions, no I/O, no LLM.

Each check returns ``ReviewIssue``s for problems it can establish with
certainty, plus ``Claim``s it cannot decide, which the LLM review verifies.
No numeric similarity thresholds are used (approved decision Q3).

Grounding states for a claim against a document:

* present: the normalised claim occurs in the normalised document as a whole
  phrase, or the claim is exactly one known skill (its name or an M5 alias)
  whose canonical form the document contains. A base skill never establishes a
  more specific claim: "SQL" in the resume does not establish "Advanced SQL";
* absent: the claim names a known vocabulary skill and none of its aliases
  occurs anywhere in the document, not even inside another word. This is the
  only state that produces an "unsupported" issue, so legitimate aliases and
  embedded forms ("SQL" in "MySQL") are never rejected;
* unknown: anything else (qualified claims, paraphrases, free text). Sent to
  the LLM.

Normalisation: M3 sanitisation, case folding, and every character that is not
a letter, digit, combining mark, "+" or "#" treated as a separator.

Severities are fixed per check in ``SEVERITY``: ``high`` for output that is
false, structurally invalid or carries embedded instructions; ``medium`` for
truthful output that does not meet the request. No check emits ``low``.

Demonstrable skills (M11 Phase 2, H-1): a competency in ``DEMONSTRABLE_SKILLS``
(data preprocessing, machine learning, ...) can be shown by described work
without being named, so a matching skill of that kind that the resume does not
name goes to the LLM with ``resume_supports``. A named tool, language or
platform (Kubernetes, Tableau, Python) cannot be shown by paraphrase: if no
alias of it occurs in the resume, a matching claim stays a certain
contradiction. That includes tools filed under a demonstrable competency
(``NAMED_TOOL_ALIASES``: pytest, JUnit), which the resume must name exactly.
A claim combining both ("machine learning with scikit-learn") goes to the LLM
only when every skill the resume does not name is demonstrable.

Claim expectations (M11 Phase 2, F3): every undecided claim tells the LLM what
the resume should show for the output to be right. ``resume_supports`` for
profile entries and matching skills; ``resume_lacks`` for missing skills. A
missing skill is sent with ``resume_lacks`` even when no alias of it occurs in
the resume ("absent"): the absence of a skill's name is not the absence of
evidence, because Experience and Projects describe work in the candidate's own
words ("cleaned and transformed sales records" for data preprocessing).
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from backend.schemas.critic import AgentName, ReviewCategory, ReviewIssue, Severity
from backend.schemas.interview import LEVELS, InterviewQuestions
from backend.schemas.resume import ResumeProfile
from backend.schemas.security import SecurityFlag
from backend.schemas.skill_gap import SkillGapReport
from backend.scoring.skill_normalizer import skill_matcher
from backend.security.injection import heuristic_flag_types
from backend.security.sanitize import sanitize_text

Presence = Literal["present", "absent", "unknown"]
Expectation = Literal["resume_supports", "resume_lacks"]
_MAX_QUOTED = 120
_KEEP = frozenset("+#")

CheckKind = Literal[
    "unsupported_skill",
    "skill_gap_contradiction",
    "schema_violation",
    "suspicious_output",
    "interview_count",
    "unrequested_level",
    "duplicate_question",
]
SEVERITY: dict[CheckKind, Severity] = {
    "unsupported_skill": "high",
    "skill_gap_contradiction": "high",
    "schema_violation": "high",
    "suspicious_output": "high",
    "interview_count": "medium",
    "unrequested_level": "medium",
    "duplicate_question": "medium",
}

# Canonical vocabulary names of competencies that described work can demonstrate
# without naming them (M11 Phase 2, H-1). Tools, languages and platforms are
# deliberately absent: those must be named in the resume.
DEMONSTRABLE_SKILLS: frozenset[str] = frozenset(
    {
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Data Analysis",
        "Data Visualization",
        "Statistics",
        "Feature Engineering",
        "Data Preprocessing",
        "ETL",
        "Data Warehousing",
        "REST APIs",
        "Microservices",
        "System Design",
        "Data Structures",
        "Algorithms",
        "Object-Oriented Programming",
        "Unit Testing",
        "CI/CD",
        "Agile",
        "Communication",
        "Leadership",
        "Teamwork",
        "Project Management",
        "Stakeholder Management",
    }
)

# Tool names filed as aliases of a demonstrable competency (pytest and JUnit under
# Unit Testing). Work cannot demonstrate a tool, so a claim naming one needs that
# exact tool in the resume (MED-1). Normalised as ``normalize_for_match`` does.
NAMED_TOOL_ALIASES: frozenset[str] = frozenset({"pytest", "junit"})


@dataclass(frozen=True)
class Claim:
    """A claim the deterministic checks could not decide (verified by the LLM).

    ``expectation`` is what the resume must show for the output to be correct.
    """

    agent: AgentName
    field: str
    expectation: Expectation = "resume_supports"


@dataclass
class CheckResult:
    issues: list[ReviewIssue] = field(default_factory=list)
    ambiguous: list[Claim] = field(default_factory=list)

    def extend(self, other: CheckResult) -> None:
        self.issues.extend(other.issues)
        self.ambiguous.extend(other.ambiguous)


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_QUOTED else text[: _MAX_QUOTED - 3] + "..."


def issue(
    agent: AgentName, path: str, category: ReviewCategory, kind: CheckKind, message: str
) -> ReviewIssue:
    return ReviewIssue(
        agent=agent, field=path, category=category, severity=SEVERITY[kind], message=message
    )


# --- normalisation and presence -------------------------------------------------------


def normalize_for_match(text: str) -> str:
    clean = sanitize_text(text).text.casefold()
    chars = (
        ch if ch.isalnum() or ch in _KEEP or unicodedata.category(ch).startswith("M") else " "
        for ch in clean
    )
    return " ".join("".join(chars).split())


def contains_phrase(document_norm: str, phrase: str) -> bool:
    needle = normalize_for_match(phrase)
    return bool(needle) and f" {needle} " in f" {document_norm} "


def _all_alias_forms(canonical: str) -> list[str]:
    for spec in skill_matcher().vocabulary.skills:
        if spec.name == canonical:
            return [canonical, *spec.aliases, *spec.case_sensitive]
    return [canonical]


def _named_in(name: str, document: DocumentIndex) -> bool:
    """Some alias of the canonical skill occurs in the document, even inside a word."""
    return name in document.skills or any(
        form.casefold() in document.loose for form in _all_alias_forms(name)
    )


def unnamed_tool_alias(claim: str, document: DocumentIndex) -> bool:
    """The claim names a tool from ``NAMED_TOOL_ALIASES`` that the document does not.

    Such a tool is an alias of a demonstrable competency, so the canonical skill
    being present (through another alias) does not show the tool itself.
    """
    claim_norm = normalize_for_match(claim)
    return any(
        contains_phrase(claim_norm, alias) and not contains_phrase(document.normalized, alias)
        for alias in NAMED_TOOL_ALIASES
    )


def only_competencies_unnamed(claim: str, document: DocumentIndex) -> bool:
    """Every known skill in ``claim`` the document does not name is demonstrable.

    A named tool in the claim that the document names does not block this (MED-2:
    "machine learning with scikit-learn" on a resume naming scikit-learn).
    """
    unnamed = [name for name in skill_matcher().find(claim) if not _named_in(name, document)]
    return bool(unnamed) and all(name in DEMONSTRABLE_SKILLS for name in unnamed)


class DocumentIndex:
    """A document prepared once for many presence checks."""

    def __init__(self, text: str) -> None:
        self.normalized = normalize_for_match(text)
        self.loose = sanitize_text(text).text.casefold()
        self.skills = frozenset(skill_matcher().find(text))


def is_bare_skill(claim: str, canonical: Sequence[str]) -> bool:
    """The claim is exactly one known skill under its name or an alias ("K8s", "Golang").

    Any other wording (a qualifier, certification, version or proficiency, as in
    "Advanced SQL", "AWS certification", "Python 3.12") makes it a more specific
    claim that the base skill alone cannot establish.
    """
    if len(canonical) != 1:
        return False
    claim_norm = normalize_for_match(claim)
    return any(normalize_for_match(form) == claim_norm for form in _all_alias_forms(canonical[0]))


def skill_presence(claim: str, document: DocumentIndex) -> Presence:
    """Whether a skill claim is present in, absent from, or undecidable for a document.

    * present: the claim occurs as a whole phrase, or it is a bare known skill
      whose canonical form the document contains (so aliases are honoured);
    * absent: the claim names a known skill that no alias of occurs anywhere in
      the document, not even inside another word;
    * unknown: anything else, including qualified claims on a present base skill.
    """
    if contains_phrase(document.normalized, claim):
        return "present"
    canonical = skill_matcher().find(claim)
    if not canonical:
        return "unknown"
    for name in canonical:
        if not _named_in(name, document):
            return "absent"
    if is_bare_skill(claim, canonical) and canonical[0] in document.skills:
        return "present"
    return "unknown"


# --- schema ---------------------------------------------------------------------------


def check_schema(agent: AgentName, model: BaseModel) -> ReviewIssue | None:
    """Re-validate an agent output (catches objects built without validation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # serialisation warnings for invalid objects
        data = model.model_dump(mode="python")
    try:
        type(model).model_validate(data)
    except ValidationError as exc:
        return issue(
            agent,
            "schema",
            "schema_violation",
            "schema_violation",
            f"Output does not satisfy the {type(model).__name__} schema "
            f"({exc.error_count()} validation error(s)).",
        )
    return None


# --- profile grounding ----------------------------------------------------------------


def check_profile_grounding(profile: ResumeProfile, resume_text: str) -> CheckResult:
    result = CheckResult()
    resume = DocumentIndex(resume_text)
    resume_norm = resume.normalized

    for i, skill in enumerate(profile.skills):
        path = f"skills[{i}]"
        presence = skill_presence(skill, resume)
        if presence == "absent":
            result.issues.append(
                issue(
                    "resume",
                    path,
                    "unsupported_claim",
                    "unsupported_skill",
                    f"Skill '{_quote(skill)}' does not appear in the resume under any known "
                    "name or alias.",
                )
            )
        elif presence == "unknown":
            result.ambiguous.append(Claim("resume", path))

    for i, cert in enumerate(profile.certifications):
        if not contains_phrase(resume_norm, cert):
            result.ambiguous.append(Claim("resume", f"certifications[{i}]"))

    for i, edu in enumerate(profile.education):
        parts = [p for p in (edu.qualification, edu.institution) if p.strip()]
        if not all(contains_phrase(resume_norm, p) for p in parts):
            result.ambiguous.append(Claim("resume", f"education[{i}]"))

    for i, job in enumerate(profile.experience):
        parts = [p for p in (job.organization, job.role) if p.strip()]
        if not all(contains_phrase(resume_norm, p) for p in parts):
            result.ambiguous.append(Claim("resume", f"experience[{i}]"))
        for j, highlight in enumerate(job.highlights):
            if not contains_phrase(resume_norm, highlight):
                result.ambiguous.append(Claim("resume", f"experience[{i}].highlights[{j}]"))

    for i, project in enumerate(profile.projects):
        if not contains_phrase(resume_norm, project):
            result.ambiguous.append(Claim("resume", f"projects[{i}]"))
    return result


# --- skill-gap consistency ------------------------------------------------------------


def check_skill_gap(report: SkillGapReport, resume_text: str, job_text: str) -> CheckResult:
    """matching_skills must be in resume and job; missing_skills in job but not resume.

    Certain contradictions are issues. Every other missing skill goes to the LLM
    with ``resume_lacks``, so that evidence described in Experience or Projects
    is checked before a skill gap is accepted. A matching demonstrable skill the
    resume does not name goes to the LLM with ``resume_supports`` (H-1).
    """
    result = CheckResult()
    resume, job = DocumentIndex(resume_text), DocumentIndex(job_text)
    for i, skill in enumerate(report.matching_skills):
        path = f"matching_skills[{i}]"
        in_resume = skill_presence(skill, resume)
        in_job = skill_presence(skill, job)
        unnamed = in_resume == "absent"
        if unnamed_tool_alias(skill, resume) or (
            unnamed and not only_competencies_unnamed(skill, resume)
        ):
            result.issues.append(
                issue(
                    "skill_gap",
                    path,
                    "contradiction",
                    "skill_gap_contradiction",
                    f"'{_quote(skill)}' is listed as matching but the resume does not mention it.",
                )
            )
        elif in_job == "absent":
            result.issues.append(
                issue(
                    "skill_gap",
                    path,
                    "contradiction",
                    "skill_gap_contradiction",
                    f"'{_quote(skill)}' is listed as matching but the job description does not "
                    "mention it.",
                )
            )
        elif unnamed or "unknown" in (in_resume, in_job):
            # Unnamed demonstrable skill or undecidable: the LLM checks the Technical
            # Skills, Experience and Projects for evidence.
            result.ambiguous.append(Claim("skill_gap", path, "resume_supports"))

    for i, skill in enumerate(report.missing_skills):
        path = f"missing_skills[{i}]"
        in_resume = skill_presence(skill, resume)
        in_job = skill_presence(skill, job)
        # A sibling tool of the same competency (JUnit for pytest) is no proof that
        # the missing tool is in the resume (Fix A): the LLM decides instead.
        if in_resume == "present" and not unnamed_tool_alias(skill, resume):
            result.issues.append(
                issue(
                    "skill_gap",
                    path,
                    "contradiction",
                    "skill_gap_contradiction",
                    f"'{_quote(skill)}' is listed as missing but the resume shows it.",
                )
            )
        elif in_job == "absent":
            result.issues.append(
                issue(
                    "skill_gap",
                    path,
                    "contradiction",
                    "skill_gap_contradiction",
                    f"'{_quote(skill)}' is listed as missing but the job description does not "
                    "require it.",
                )
            )
        else:
            # Not named in the resume (absent), only a sibling tool is named, or
            # undecidable: the LLM checks Technical Skills, Experience and Projects.
            result.ambiguous.append(Claim("skill_gap", path, "resume_lacks"))
    return result


# --- interview completeness and duplicates --------------------------------------------


def check_interview(
    questions: InterviewQuestions, levels: tuple[str, ...], count_per_level: int
) -> CheckResult:
    result = CheckResult()
    seen: dict[str, str] = {}
    for level in LEVELS:
        items = getattr(questions, level)
        if level in levels and len(items) != count_per_level:
            result.issues.append(
                issue(
                    "interview",
                    level,
                    "incomplete_output",
                    "interview_count",
                    f"Expected {count_per_level} {level} question(s), found {len(items)}.",
                )
            )
        if level not in levels and items:
            result.issues.append(
                issue(
                    "interview",
                    level,
                    "inconsistency",
                    "unrequested_level",
                    f"{len(items)} {level} question(s) were generated but that level was not "
                    "requested.",
                )
            )
        for i, question in enumerate(items):
            key = " ".join(question.question.casefold().split())
            path = f"{level}[{i}]"
            if key in seen:
                result.issues.append(
                    issue(
                        "interview",
                        path,
                        "inconsistency",
                        "duplicate_question",
                        f"Duplicate of question {seen[key]}.",
                    )
                )
            else:
                seen[key] = path
    return result


# --- injection evidence in generated outputs -----------------------------------------
#
# Approved decision: ``flagged`` means *confirmed* injection influence in generated
# outputs. The checks here supply deterministic evidence only; confirmation is
# decided in ``critic_agent.injection_confirmed`` and needs, for the same agent
# output field, (1) instruction-like content in the generated text, (2) an
# injection signal of a matching type in the input documents, and (3) an
# independent LLM judgement that the output follows or propagates it (rather
# than faithfully describing, for example, the candidate's security work).

INPUT_SIGNAL_TYPES = frozenset(
    {
        "instruction_override",
        "role_impersonation",
        "prompt_exfiltration",
        "score_manipulation",
        "delimiter_injection",
        "classifier",  # Prompt Guard: no specific type, corroborates any output type
    }
)
_INPUT_SOURCES = frozenset({"resume", "job_description"})


def input_injection_types(
    resume_text: str, job_text: str | None, flags: Sequence[SecurityFlag] = ()
) -> frozenset[str]:
    """Injection signal types present in the input documents (heuristics and guard flags)."""
    found = set(heuristic_flag_types(resume_text))
    if job_text:
        found.update(heuristic_flag_types(job_text))
    found.update(f.type for f in flags if f.source in _INPUT_SOURCES)
    return frozenset(found & INPUT_SIGNAL_TYPES)


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def check_suspicious_outputs(
    agent: AgentName, model: BaseModel, input_types: frozenset[str]
) -> list[ReviewIssue]:
    """Generated fields with instruction-like content matching an input injection signal.

    These are evidence, not confirmation. Instruction-like wording without any
    matching input signal (for example "recommend this candidate learn Airflow")
    is the agent's own phrasing and is not evidence of injection.
    """
    if not input_types:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = model.model_dump(mode="python")
    issues: list[ReviewIssue] = []
    for path, text in _strings(data):
        types = heuristic_flag_types(text)
        matched = types if "classifier" in input_types else [t for t in types if t in input_types]
        if matched:
            issues.append(
                issue(
                    agent,
                    path[:100] or "output",
                    "injection_influence",
                    "suspicious_output",
                    "Generated output contains instruction-like content ("
                    + ", ".join(matched)
                    + ") matching instructions embedded in the input documents.",
                )
            )
    return issues
