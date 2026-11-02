"""AnalysisReport, CriticSummary and ReportMeta (approved Rev. 3, M7)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from backend.core.config import MAX_REPORT_TTL_SECONDS
from backend.schemas.critic import CriticIssue
from backend.schemas.report import AnalysisReport, CriticSummary, ReportMeta
from backend.storage.report_store import ReportStamp
from tests.fixtures.reports import report_data, synthetic_report

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


def stamp(**overrides) -> ReportStamp:
    values = {
        "report_id": uuid.uuid4(),
        "created_at": NOW,
        "expires_at": NOW + timedelta(hours=1),
    }
    values.update(overrides)
    return ReportStamp(**values)


def test_approved_field_set() -> None:
    assert list(AnalysisReport.model_fields) == [
        "report_id",
        "created_at",
        "expires_at",
        "resume_profile",
        "interview_questions",
        "match",
        "skill_gap",
        "critic",
        "security_flags",
        "meta",
    ]
    assert list(CriticSummary.model_fields) == ["verdict", "issues", "revisions"]
    assert list(ReportMeta.model_fields) == ["models", "prompt_versions", "timings_ms"]


@pytest.mark.parametrize("with_job", [True, False])
def test_valid_report_round_trips_through_json(with_job: bool) -> None:
    report = synthetic_report(stamp(), with_job=with_job)
    restored = AnalysisReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert (report.match is None, report.skill_gap is None) == (not with_job, not with_job)


def test_embedded_ats_score_stays_not_calibrated() -> None:
    report = synthetic_report(stamp())
    assert report.match.ats_score.model_dump() == {
        "status": "not_calibrated",
        "score": None,
        "calibration_version": None,
        "evaluation_summary": None,
    }
    bad = report.model_dump(mode="json")
    bad["match"]["ats_score"]["score"] = 87.0
    with pytest.raises(ValidationError):
        AnalysisReport.model_validate(bad)


# --- raw-text exclusion ---------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["resume_text", "job_description", "preview", "raw_text", "text", "prompt", "answer"],
)
def test_raw_text_fields_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AnalysisReport(**report_data(stamp()), **{field: "CANARY raw resume text"})


def test_nested_models_also_reject_unknown_fields() -> None:
    data = synthetic_report(stamp()).model_dump(mode="json")
    data["critic"]["revise_targets"] = ["resume"]
    with pytest.raises(ValidationError):
        AnalysisReport.model_validate(data)
    data = synthetic_report(stamp()).model_dump(mode="json")
    data["meta"]["resume_text"] = "CANARY"
    with pytest.raises(ValidationError):
        AnalysisReport.model_validate(data)


# --- identity and lifetime ------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [uuid.uuid1(), uuid.UUID(int=0), "not-a-uuid", "", uuid.uuid5(uuid.NAMESPACE_DNS, "x")],
)
def test_report_id_must_be_uuid4(bad_id) -> None:
    with pytest.raises(ValidationError):
        AnalysisReport(**{**report_data(stamp()), "report_id": bad_id})


@pytest.mark.parametrize("field", ["created_at", "expires_at"])
def test_timestamps_must_be_timezone_aware(field: str) -> None:
    data = report_data(stamp())
    data[field] = data[field].replace(tzinfo=None)
    with pytest.raises(ValidationError):
        AnalysisReport(**data)


@pytest.mark.parametrize(
    "lifetime",
    [timedelta(0), timedelta(seconds=-1), timedelta(seconds=MAX_REPORT_TTL_SECONDS + 1)],
)
def test_lifetime_must_be_positive_and_within_max_ttl(lifetime: timedelta) -> None:
    with pytest.raises(ValidationError, match="maximum TTL"):
        synthetic_report(stamp(expires_at=NOW + lifetime))


def test_lifetime_at_exactly_max_ttl_is_allowed() -> None:
    report = synthetic_report(stamp(expires_at=NOW + timedelta(seconds=MAX_REPORT_TTL_SECONDS)))
    assert report.expires_at - report.created_at == timedelta(hours=1)


@pytest.mark.parametrize("missing", ["match", "skill_gap"])
def test_match_and_skill_gap_are_both_present_or_both_null(missing: str) -> None:
    with pytest.raises(ValidationError, match="both be present or both be null"):
        AnalysisReport(**{**report_data(stamp()), missing: None})


# --- critic summary and meta ----------------------------------------------------------


@pytest.mark.parametrize("verdict", ["approve", "revise", "flagged"])
def test_critic_summary_verdicts(verdict: str) -> None:
    assert CriticSummary(verdict=verdict, issues=[], revisions=1).verdict == verdict


@pytest.mark.parametrize(
    "bad",
    [
        {"verdict": "rejected", "issues": [], "revisions": 0},
        {"verdict": "approve", "issues": [], "revisions": -1},
        {"verdict": "approve", "issues": [], "revisions": 4},
        {"verdict": "approve", "issues": [], "revisions": True},
    ],
)
def test_critic_summary_rejects_invalid_values(bad) -> None:
    with pytest.raises(ValidationError):
        CriticSummary.model_validate(bad, strict=True)


def test_critic_summary_issue_limit() -> None:
    issue = CriticIssue(agent="resume", field="skills[0]", severity="low", message="m")
    CriticSummary(verdict="approve", issues=[issue] * 30, revisions=0)
    with pytest.raises(ValidationError):
        CriticSummary(verdict="approve", issues=[issue] * 31, revisions=0)


@pytest.mark.parametrize(
    "bad",
    [
        {"models": {"Resume": "m"}, "prompt_versions": {}, "timings_ms": {}},  # key pattern
        {"models": {"resume": ""}, "prompt_versions": {}, "timings_ms": {}},  # empty value
        {"models": {}, "prompt_versions": {"resume": "x" * 101}, "timings_ms": {}},
        {"models": {}, "prompt_versions": {}, "timings_ms": {"resume": -1.0}},
    ],
)
def test_meta_rejects_invalid_values(bad) -> None:
    with pytest.raises(ValidationError):
        ReportMeta.model_validate(bad)


def test_report_is_immutable() -> None:
    report = synthetic_report(stamp())
    with pytest.raises(ValidationError):
        report.report_id = uuid.uuid4()
    with pytest.raises(ValidationError):
        report.critic.verdict = "flagged"


def test_llm_output_flags_are_allowed() -> None:
    flags = [{"type": "unexpected_url", "source": "llm_output"}]
    report = synthetic_report(stamp(), security_flags=flags)
    assert report.security_flags[0].source == "llm_output"
