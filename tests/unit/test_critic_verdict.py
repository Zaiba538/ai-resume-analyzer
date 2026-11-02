"""Server-side verdict derivation (Q1/Q2), revision cap, and issue merging."""

from __future__ import annotations

import pytest

from backend.agents.critic_agent import (
    apply_revision_cap,
    derive_verdict,
    injection_confirmed,
    merge_issues,
    normalize_field,
    select_published,
)
from backend.core.config import Settings
from backend.schemas.critic import MAX_ISSUE_MESSAGE, CriticVerdict, ReviewIssue


def ri(agent="resume", field="skills[0]", severity="medium", category="unsupported_claim", msg="m"):
    return ReviewIssue(agent=agent, field=field, severity=severity, category=category, message=msg)


def verdict_of(issues, confirmed: bool = False) -> CriticVerdict:
    return derive_verdict(issues, injection_confirmed=confirmed)


def test_no_issues_approves() -> None:
    verdict = verdict_of([])
    assert (verdict.verdict, verdict.issues, verdict.revise_targets) == ("approve", [], [])


def test_low_only_approves_but_keeps_issues() -> None:
    verdict = verdict_of([ri(severity="low"), ri(agent="interview", severity="low")])
    assert verdict.verdict == "approve" and verdict.revise_targets == []
    assert len(verdict.issues) == 2


@pytest.mark.parametrize("severity", ["medium", "high"])
def test_medium_or_high_revises_responsible_agent(severity: str) -> None:
    verdict = verdict_of([ri(agent="skill_gap", severity=severity), ri(severity="low")])
    assert verdict.verdict == "revise"
    assert verdict.revise_targets == ["skill_gap"]  # low-severity agent not targeted


def test_revise_targets_ordered_and_unique() -> None:
    verdict = verdict_of(
        [
            ri(agent="skill_gap", severity="high"),
            ri(agent="resume", severity="medium"),
            ri(agent="skill_gap", severity="medium", field="x"),
            ri(agent="interview", severity="high"),
        ]
    )
    assert verdict.revise_targets == ["resume", "interview", "skill_gap"]


@pytest.mark.parametrize("severity", ["low", "medium", "high"])
def test_confirmed_injection_flags_and_keeps_every_issue(severity: str) -> None:
    issues = [
        ri(severity="high"),
        ri(agent="interview", category="injection_influence", severity=severity),
    ]
    verdict = verdict_of(issues, confirmed=True)
    assert verdict.verdict == "flagged"
    assert verdict.revise_targets == []
    assert len(verdict.issues) == 2  # all issues preserved


def test_injection_category_alone_never_flags() -> None:
    """An injection_influence issue without confirmation is an ordinary issue."""
    for severity, expected in (("high", "revise"), ("medium", "revise"), ("low", "approve")):
        issue = ri(agent="interview", category="injection_influence", severity=severity)
        assert verdict_of([issue]).verdict == expected


def test_no_other_flagged_conditions() -> None:
    for category in (
        "unsupported_claim",
        "contradiction",
        "irrelevant_recommendation",
        "irrelevant_question",
        "incomplete_output",
        "schema_violation",
        "inconsistency",
        "injection_influence",
    ):
        assert verdict_of([ri(category=category, severity="high")]).verdict == "revise"


# --- injection confirmation -----------------------------------------------------------

EVIDENCE = ri(agent="resume", field="projects[0]", category="injection_influence", severity="high")


def claim(agent="resume", field="projects[0]", category="injection_influence"):
    return ri(agent=agent, field=field, category=category, severity="high")


def test_confirmation_needs_evidence_and_llm_claim_on_same_field() -> None:
    assert injection_confirmed([EVIDENCE], [claim()]) is True
    assert injection_confirmed([EVIDENCE], []) is False  # heuristic evidence alone
    assert injection_confirmed([], [claim()]) is False  # LLM claim alone
    assert injection_confirmed([], []) is False


@pytest.mark.parametrize(
    ("llm", "expected"),
    [
        (claim(field="projects"), True),  # parent path
        (claim(field="PROJECTS[0]"), True),  # case-insensitive
        (claim(field="projects[1]"), False),  # different field
        (claim(field="projects[0]x"), False),  # not a path boundary
        (claim(agent="interview"), False),  # different agent
        (claim(category="unsupported_claim"), False),  # not an injection judgement
    ],
)
def test_confirmation_field_matching(llm, expected: bool) -> None:
    assert injection_confirmed([EVIDENCE], [llm]) is expected


def test_confirmation_matches_nested_evidence_paths() -> None:
    nested = ri(agent="interview", field="beginner[0].question", category="injection_influence")
    assert injection_confirmed([nested], [claim(agent="interview", field="beginner[0]")]) is True
    assert injection_confirmed([nested], [claim(agent="interview", field="beginner[1]")]) is False


def test_published_issues_drop_category_and_sort_by_severity() -> None:
    verdict = verdict_of(
        [ri(severity="low", msg="a"), ri(severity="high", msg="b"), ri(severity="medium", msg="c")]
    )
    assert [i.message for i in verdict.issues] == ["b", "c", "a"]
    assert "category" not in verdict.issues[0].model_dump()
    CriticVerdict.model_validate(verdict.model_dump())  # strict approved schema


def test_issue_cap_but_verdict_uses_all_issues() -> None:
    issues = [ri(severity="low", field=f"f{i}") for i in range(40)]
    issues.append(ri(agent="interview", severity="high", field="late"))
    verdict = verdict_of(issues)
    assert len(verdict.issues) == 30
    assert verdict.issues[0].field == "late"  # highest severity survives the cap
    assert verdict.verdict == "revise" and verdict.revise_targets == ["interview"]


# --- H1: the 30-issue limit -----------------------------------------------------------


def test_cap_keeps_every_revise_targets_issue_visible() -> None:
    issues = [ri(severity="high", field=f"s{i}") for i in range(30)]
    issues.append(ri(agent="interview", severity="medium", field="beginner"))
    verdict = verdict_of(issues)
    assert len(verdict.issues) == 30
    assert verdict.revise_targets == ["resume", "interview"]
    assert {i.agent for i in verdict.issues} == {"resume", "interview"}


def test_cap_does_not_reserve_slots_for_low_only_agents() -> None:
    issues = [ri(severity="high", field=f"s{i}") for i in range(30)]
    issues.append(ri(agent="interview", severity="low", field="beginner"))
    verdict = verdict_of(issues)
    assert [i.agent for i in verdict.issues] == ["resume"] * 30
    assert verdict.revise_targets == ["resume"]


def test_cap_keeps_injection_evidence_when_flagged() -> None:
    issues = [ri(severity="high", field=f"s{i}") for i in range(35)]
    issues.append(ri(severity="high", field="projects[0]", category="injection_influence"))
    verdict = verdict_of(issues, confirmed=True)
    assert verdict.verdict == "flagged" and len(verdict.issues) == 30
    assert verdict.issues[0].field == "projects[0]"


def test_published_order_is_severity_then_original() -> None:
    issues = [
        ri(severity="low", field="a"),
        ri(severity="high", field="b"),
        ri(severity="medium", field="c"),
        ri(severity="high", field="d"),
    ]
    assert [i.field for i in select_published(issues)] == ["b", "d", "c", "a"]


def test_revise_targets_always_derive_from_published_issues() -> None:
    import random

    rng = random.Random(6)  # noqa: S311 - deterministic test data, not security
    agents, severities = ("resume", "interview", "skill_gap"), ("low", "medium", "high")
    for _ in range(300):
        issues = [
            ri(agent=rng.choice(agents), severity=rng.choice(severities), field=f"f{n}")
            for n in range(rng.randint(0, 45))
        ]
        verdict = verdict_of(issues)
        published = {i.agent for i in verdict.issues if i.severity in ("medium", "high")}
        all_needing = {i.agent for i in issues if i.severity in ("medium", "high")}
        assert set(verdict.revise_targets) == published == all_needing
        assert len(verdict.issues) == min(len(issues), 30)
        assert verdict.verdict == ("revise" if published else "approve")


# --- revision cap ---------------------------------------------------------------------

MAX = Settings(_env_file=None).max_revisions  # configured MAX_REVISIONS (approved: 1)
REVISE = verdict_of([ri(severity="high"), ri(agent="interview", severity="low", field="b")])


def test_configured_max_revisions_is_one() -> None:
    assert MAX == 1


def test_initial_revise_with_zero_revisions_done_is_unchanged() -> None:
    assert apply_revision_cap(REVISE, revisions_done=0, max_revisions=MAX) == REVISE


def test_revise_after_final_permitted_revision_is_flagged_with_all_issues() -> None:
    capped = apply_revision_cap(REVISE, revisions_done=MAX, max_revisions=MAX)
    assert capped.verdict == "flagged"
    assert capped.issues == REVISE.issues  # every issue, in order, including low ones
    assert capped.revise_targets == []


@pytest.mark.parametrize(("done", "maximum"), [(1, 1), (2, 1), (0, 0), (3, 3)])
def test_unresolved_revise_at_or_after_limit_is_flagged(done: int, maximum: int) -> None:
    capped = apply_revision_cap(REVISE, revisions_done=done, max_revisions=maximum)
    assert (capped.verdict, capped.issues, capped.revise_targets) == (
        "flagged",
        REVISE.issues,
        [],
    )


@pytest.mark.parametrize(("done", "maximum"), [(0, 1), (1, 2), (2, 3)])
def test_revise_within_limit_is_unchanged(done: int, maximum: int) -> None:
    assert apply_revision_cap(REVISE, revisions_done=done, max_revisions=maximum) == REVISE


@pytest.mark.parametrize("done", [0, MAX, MAX + 1])
@pytest.mark.parametrize(
    "verdict",
    [
        verdict_of([]),
        verdict_of([ri(severity="low")]),
        verdict_of([ri(category="injection_influence")], confirmed=True),
    ],
    ids=["approve-empty", "approve-low", "flagged"],
)
def test_cap_leaves_approve_and_flagged_alone(verdict, done: int) -> None:
    assert apply_revision_cap(verdict, revisions_done=done, max_revisions=MAX) == verdict


def test_cap_preserves_a_full_issue_list() -> None:
    full = verdict_of([ri(severity="medium", field=f"f{i}") for i in range(30)])
    capped = apply_revision_cap(full, revisions_done=MAX, max_revisions=MAX)
    assert len(capped.issues) == 30 and capped.issues == full.issues


def test_cap_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        apply_revision_cap(REVISE, revisions_done=-1, max_revisions=1)
    with pytest.raises(ValueError):
        apply_revision_cap(REVISE, revisions_done=0, max_revisions=-1)


# --- merging --------------------------------------------------------------------------


def test_merge_one_issue_per_agent_field_category() -> None:
    deterministic = [ri(field="skills[1]", msg="deterministic")]
    llm = [
        ri(field="Skills[1]", msg="llm duplicate"),
        ri(field="skills[1]", category="contradiction", msg="different category"),
        ri(agent="interview", field="skills[1]", msg="different agent"),
    ]
    merged = merge_issues(deterministic, llm)
    assert [i.message for i in merged] == [
        "deterministic Also: llm duplicate",  # distinct evidence kept, first issue primary
        "different category",
        "different agent",
    ]
    assert merged[0].field == "skills[1]"  # the first issue's field spelling is kept


# --- H1: merging never lowers severity ------------------------------------------------


def _gap_issue(severity: str, msg: str) -> ReviewIssue:
    return ri(
        agent="skill_gap",
        field="recommendations[0]",
        severity=severity,
        category="irrelevant_recommendation",
        msg=msg,
    )


@pytest.mark.parametrize(
    "order",
    [("low", "high"), ("high", "low"), ("low", "medium"), ("low", "low", "high")],
)
def test_lower_severity_never_replaces_higher(order) -> None:
    issues = [_gap_issue(sev, f"problem {n}") for n, sev in enumerate(order)]
    (merged,) = merge_issues([], issues)
    worst = min(order, key=["high", "medium", "low"].index)
    assert merged.severity == worst
    verdict = verdict_of(merge_issues([], issues))
    assert verdict.verdict == "revise" and verdict.revise_targets == ["skill_gap"]


def test_low_then_high_from_llm_cannot_approve() -> None:
    llm = [_gap_issue("low", "minor wording"), _gap_issue("high", "not grounded in any gap")]
    verdict = verdict_of(merge_issues([], llm))
    assert verdict.verdict == "revise"
    (published,) = verdict.issues
    assert published.severity == "high"
    assert published.message == "not grounded in any gap Also: minor wording"


def test_higher_severity_llm_issue_overrides_deterministic_severity() -> None:
    deterministic = [ri(agent="interview", field="beginner", severity="medium", msg="count")]
    llm = [ri(agent="interview", field="beginner", severity="high", msg="llm")]
    (merged,) = merge_issues(deterministic, llm)
    assert (merged.severity, merged.message) == ("high", "llm Also: count")


def test_equal_severity_keeps_deterministic_primary_and_llm_evidence() -> None:
    evidence = ri(
        field="projects[0]",
        severity="high",
        category="injection_influence",
        msg="Generated output contains instruction-like content.",
    )
    claim = ri(
        field="projects[0]",
        severity="high",
        category="injection_influence",
        msg="Reproduces the embedded instruction 'ignore previous instructions'.",
    )
    (merged,) = merge_issues([evidence], [claim])
    assert merged.message.startswith(evidence.message)
    assert claim.message in merged.message  # the LLM's explanation is preserved


def test_identical_messages_are_not_repeated() -> None:
    (merged,) = merge_issues([ri(msg="Same  problem")], [ri(msg="same problem")])
    assert merged.message == "Same  problem"


def test_combined_message_stays_within_limit_and_counts_the_rest() -> None:
    primary = ri(severity="high", msg="p" * 200)  # room for one 88-char extra, not two
    others = [ri(severity="low", msg=f"other {n} " + "x" * 80) for n in range(4)]
    (merged,) = merge_issues([primary], others)
    assert len(merged.message) <= MAX_ISSUE_MESSAGE
    assert merged.message.endswith("(+3 more)")
    assert merged.message.count("Also:") == 1


def test_full_length_primary_still_reports_omitted_evidence() -> None:
    (merged,) = merge_issues([ri(severity="high", msg="p" * 400)], [ri(msg="other")])
    assert len(merged.message) <= MAX_ISSUE_MESSAGE
    assert merged.message.endswith("… (+1 more)")


@pytest.mark.parametrize(
    ("agent", "raw", "expected"),
    [
        ("resume", "profile.skills[2]", "skills[2]"),
        ("interview", "interview.advanced[0]", "advanced[0]"),
        ("skill_gap", "skill_gap.missing_skills[1]", "missing_skills[1]"),
        ("resume", "skills[2]", "skills[2]"),  # already relative
        ("resume", "profile.", "profile."),  # nothing left after the prefix
        ("resume", "interview.advanced[0]", "interview.advanced[0]"),  # other agent's key
    ],
)
def test_normalize_field_strips_own_payload_prefix(agent: str, raw: str, expected: str) -> None:
    assert normalize_field(ri(agent=agent, field=raw)).field == expected
