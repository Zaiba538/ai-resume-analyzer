"""M11 Phase 2 (F1-F5): skill evidence across the profile, claim expectations,
evidence-based severity and the Data Preprocessing vocabulary entry.

Offline only: the LLM is the fake Groq API, so these tests check prompts,
payloads and server-side rules, not model behaviour (see tests/live for that).
"""

from __future__ import annotations

import json
import sys

import pytest

from backend.agents.critic_checks import (
    DEMONSTRABLE_SKILLS,
    NAMED_TOOL_ALIASES,
    SEVERITY,
    DocumentIndex,
    check_profile_grounding,
    check_skill_gap,
    normalize_for_match,
    skill_presence,
)
from backend.llm.prompts.critic import CRITIC_PROMPT
from backend.llm.prompts.skill_gap import SKILL_GAP_PROMPT, SKILL_GAP_REVISION_PROMPT
from backend.scoring.skill_coverage import skill_coverage
from backend.scoring.skill_normalizer import skill_matcher
from tests.fixtures.groq import FakeGroqChat, json_reply
from tests.unit.test_critic_agent import full_input, llm_issue, run, user_block
from tests.unit.test_critic_checks import gap, profile

# The manually tested case: preprocessing is evidenced only in the Experience
# section, in the candidate's own words; Tableau and Docker are only preferred.
DS_RESUME = (
    "Taylor Example - Junior Data Scientist\n"
    "TECHNICAL SKILLS\n"
    "Python, SQL, Pandas, scikit-learn\n"
    "EXPERIENCE\n"
    "Data Analyst Intern, Example Retail (fictional), 2024\n"
    "- Cleaned and transformed 15,000+ sales records using Python and Pandas, documenting "
    "missing-value and duplicate handling.\n"
    "PROJECTS\n"
    "- Churn model: trained a scikit-learn classifier on a public dataset.\n"
)
DS_JOB = (
    "Junior Data Scientist\n"
    "Required: Python, SQL, data preprocessing and machine learning.\n"
    "Preferred: Tableau and Docker."
)


def ds_profile():
    """A profile faithful to DS_RESUME, so only the skill-gap output is in question."""
    return profile(
        skills=["Python", "SQL", "Pandas", "scikit-learn"],
        education=[],
        experience=[
            {
                "organization": "Example Retail (fictional)",
                "role": "Data Analyst Intern",
                "period": "2024",
                "highlights": [
                    "Cleaned and transformed 15,000+ sales records using Python and Pandas, "
                    "documenting missing-value and duplicate handling."
                ],
            }
        ],
        projects=["Churn model: trained a scikit-learn classifier on a public dataset."],
        certifications=[],
    )


def ds_input(skill_gap, resume: str = DS_RESUME):
    return full_input(
        resume_text=resume,
        profile=ds_profile(),
        job_description=DS_JOB,
        interview=None,
        skill_gap=skill_gap,
    )


def expectations(result) -> list[tuple[str, str, str]]:
    return [(c.agent, c.field, c.expectation) for c in result.ambiguous]


# --- F5: Data Preprocessing vocabulary --------------------------------------------------


@pytest.mark.parametrize(
    "term",
    ["data preprocessing", "Data Pre-processing", "data pre processing", "data cleaning",
     "Data cleansing"],
)  # fmt: skip
def test_data_preprocessing_aliases_are_recognised(term: str) -> None:
    assert skill_matcher().find(f"Experience with {term} in Pandas.") == [
        "Data Preprocessing",
        "Pandas",
    ]
    assert skill_matcher().normalize(term) == "Data Preprocessing"


@pytest.mark.parametrize(
    "text",
    ["metadata cleaning scripts", "a preprocessing step", "big data processing",
     "data cleaningservice"],
)  # fmt: skip
def test_data_preprocessing_is_not_matched_inside_other_terms(text: str) -> None:
    assert "Data Preprocessing" not in skill_matcher().find(text)


def test_coverage_counts_data_preprocessing_from_an_alias() -> None:
    coverage = skill_coverage("Did data cleaning with Pandas and SQL.", DS_JOB)
    assert "Data Preprocessing" in coverage.matched
    assert "Data Preprocessing" in skill_coverage(DS_RESUME, DS_JOB).missing  # paraphrase only
    assert skill_coverage(DS_RESUME, DS_JOB).jd_skills_total == 6


# --- F3: explicit claim expectations ----------------------------------------------------


def test_skill_gap_claims_state_what_the_resume_should_show() -> None:
    report = gap(["Advanced SQL"], ["Data Preprocessing", "Tableau"])
    result = check_skill_gap(report, DS_RESUME, DS_JOB + " Advanced SQL is a plus.")
    assert result.issues == []
    assert expectations(result) == [
        ("skill_gap", "matching_skills[0]", "resume_supports"),
        ("skill_gap", "missing_skills[0]", "resume_lacks"),
        ("skill_gap", "missing_skills[1]", "resume_lacks"),
    ]


def test_missing_skill_without_a_named_alias_still_goes_to_the_reviewer() -> None:
    """No alias of Data Preprocessing occurs in the resume, but the Experience
    section describes the work: absence of a name is not absence of evidence."""
    assert skill_presence("Data Preprocessing", DocumentIndex(DS_RESUME)) == "absent"
    result = check_skill_gap(gap([], ["Data Preprocessing"]), DS_RESUME, DS_JOB)
    assert result.issues == []
    assert expectations(result) == [("skill_gap", "missing_skills[0]", "resume_lacks")]


def test_missing_skill_named_in_the_resume_is_still_a_certain_contradiction() -> None:
    resume = DS_RESUME + "Responsible for data cleaning of all sales data.\n"
    result = check_skill_gap(gap([], ["Data Preprocessing"]), resume, DS_JOB)
    assert [(i.field, i.category, i.severity) for i in result.issues] == [
        ("missing_skills[0]", "contradiction", "high")
    ]
    assert result.ambiguous == []


def test_profile_claims_expect_resume_support() -> None:
    result = check_profile_grounding(profile(skills=["Advanced SQL"]), "Skills: SQL")
    assert result.ambiguous
    assert {c.expectation for c in result.ambiguous} == {"resume_supports"}


@pytest.mark.asyncio
async def test_reviewer_payload_carries_claim_expectations() -> None:
    fake = FakeGroqChat().queue(json_reply({"issues": []}))
    await run(
        fake,
        ds_input(gap(["Python"], ["Data Preprocessing"])),
    )
    claims = json.loads(user_block(fake, "agent_outputs"))["claims_to_verify"]
    assert {
        "agent": "skill_gap",
        "field": "missing_skills[0]",
        "expectation": "resume_lacks",
    } in claims
    assert all(c["expectation"] in {"resume_supports", "resume_lacks"} for c in claims)


# --- F4: severity from evidence certainty -----------------------------------------------


@pytest.mark.asyncio
async def test_inferred_contradiction_on_an_undecided_claim_is_medium() -> None:
    contradiction = llm_issue(
        field="missing_skills[0]",
        severity="high",
        category="contradiction",
        message="The Experience section shows data cleaning with Pandas.",
    )
    fake = FakeGroqChat().queue(json_reply({"issues": [contradiction]}))
    result = await run(
        fake,
        ds_input(gap(["Python"], ["Data Preprocessing"])),
    )
    issue = next(i for i in result.output.issues if i.field == "missing_skills[0]")
    assert issue.severity == "medium"  # inferred from paraphrased evidence
    assert result.output.verdict == "revise"  # medium still asks for one revision
    assert result.output.revise_targets == ["skill_gap"]


@pytest.mark.asyncio
async def test_provable_contradiction_stays_high() -> None:
    resume = DS_RESUME + "Responsible for data cleaning of all sales data.\n"
    contradiction = llm_issue(
        field="missing_skills[0]", severity="high", category="contradiction", message="Shown."
    )
    fake = FakeGroqChat().queue(json_reply({"issues": [contradiction]}))
    result = await run(
        fake,
        ds_input(gap(["Python"], ["Data Preprocessing"]), resume),
    )
    issue = next(i for i in result.output.issues if i.field == "missing_skills[0]")
    assert issue.severity == "high" == SEVERITY["skill_gap_contradiction"]


@pytest.mark.asyncio
async def test_contradiction_on_a_field_the_checks_decided_keeps_its_severity() -> None:
    contradiction = llm_issue(
        field="matching_skills[0]", severity="high", category="contradiction", message="x"
    )
    fake = FakeGroqChat().queue(json_reply({"issues": [contradiction]}))
    result = await run(
        fake,
        ds_input(gap(["Python"], [])),
    )
    issue = next(i for i in result.output.issues if i.field == "matching_skills[0]")
    assert issue.severity == "high"


@pytest.mark.asyncio
async def test_other_categories_on_undecided_claims_keep_their_severity() -> None:
    unsupported = llm_issue(
        field="missing_skills[0]", severity="high", category="unsupported_claim", message="x"
    )
    fake = FakeGroqChat().queue(json_reply({"issues": [unsupported]}))
    result = await run(
        fake,
        ds_input(gap([], ["Data Preprocessing"])),
    )
    issue = next(i for i in result.output.issues if i.field == "missing_skills[0]")
    assert issue.severity == "high"


# --- F1 and F2: prompts ------------------------------------------------------------------


def test_skill_gap_prompt_looks_for_evidence_across_the_profile() -> None:
    assert SKILL_GAP_PROMPT.id == "skill_gap@1.2.0"
    assert SKILL_GAP_REVISION_PROMPT.id == "skill_gap@1.3.0"
    assert SKILL_GAP_REVISION_PROMPT.system.startswith(SKILL_GAP_PROMPT.system)
    text = SKILL_GAP_PROMPT.system
    for part in ("skills", "experience", "highlights", "projects", "certifications"):
        assert part in text, part
    lowered = text.casefold()
    assert "demonstrated" in lowered and "evidence" in lowered
    # The worked example from the manually tested case: described work is evidence.
    assert (
        "cleaning and transforming records with Pandas, including missing-value and duplicate "
        "handling, is evidence of data preprocessing"
    ) in text
    assert "experience (each entry's highlights)" in text
    assert "required" in lowered and "preferred" in lowered
    assert "never list a preferred" in lowered


def test_reviewer_prompt_checks_every_section_and_claim_direction() -> None:
    assert CRITIC_PROMPT.id == "critic@1.4.0"
    text = CRITIC_PROMPT.system
    for section in ("Technical Skills", "Experience", "Projects"):
        assert section in text, section
    assert "resume_supports" in text and "resume_lacks" in text
    lowered = text.casefold()
    assert "paraphrase" in lowered and "severity medium" in lowered
    assert "not the importance of the candidate's skill gap" in lowered


# --- H-1: matching skills demonstrated by described work ---------------------------------


def test_matching_skill_shown_only_by_described_work_goes_to_the_reviewer() -> None:
    """Data preprocessing and machine learning are demonstrated by the Experience
    and Projects bullets, not named: the LLM must judge them, not a name lookup."""
    for skill in ("Data Preprocessing", "machine learning"):
        assert skill_presence(skill, DocumentIndex(DS_RESUME)) == "absent"
    report = gap(["Python", "Data Preprocessing", "machine learning"], [])
    result = check_skill_gap(report, DS_RESUME, DS_JOB)
    assert result.issues == []
    assert expectations(result) == [
        ("skill_gap", "matching_skills[1]", "resume_supports"),
        ("skill_gap", "matching_skills[2]", "resume_supports"),
    ]


def test_demonstrable_skills_are_vocabulary_competencies_not_tools() -> None:
    names = {spec.name for spec in skill_matcher().vocabulary.skills}
    assert names >= DEMONSTRABLE_SKILLS  # a typo would silently disable the rule
    assert {"Data Preprocessing", "Machine Learning"} <= DEMONSTRABLE_SKILLS
    assert not {"Python", "SQL", "Pandas", "Tableau", "Docker", "Kubernetes"} & DEMONSTRABLE_SKILLS


# The approved list after the final M11 Phase 2 refinement (Problem Solving removed).
APPROVED_DEMONSTRABLE_SKILLS = frozenset(
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


def test_exactly_the_25_approved_demonstrable_skills_remain() -> None:
    assert len(APPROVED_DEMONSTRABLE_SKILLS) == 25
    assert len(DEMONSTRABLE_SKILLS) == 25
    assert DEMONSTRABLE_SKILLS == APPROVED_DEMONSTRABLE_SKILLS


def test_problem_solving_is_not_demonstrable_but_stays_in_the_vocabulary() -> None:
    assert "Problem Solving" not in DEMONSTRABLE_SKILLS
    names = {spec.name for spec in skill_matcher().vocabulary.skills}
    assert "Problem Solving" in names
    assert skill_matcher().normalize("problem-solving") == "Problem Solving"
    # Not demonstrable: an unnamed matching claim is a certain contradiction again.
    job = DS_JOB + " Strong problem solving is required."
    result = check_skill_gap(gap(["Problem Solving"], []), DS_RESUME, job)
    assert [(i.field, i.severity) for i in result.issues] == [("matching_skills[0]", "high")]
    assert result.ambiguous == []


@pytest.mark.asyncio
async def test_paraphrase_supported_matching_skill_is_approved() -> None:
    fake = FakeGroqChat().queue(json_reply({"issues": []}))
    result = await run(fake, ds_input(gap(["Python", "data preprocessing"], [])))
    assert result.output.verdict == "approve" and result.output.issues == []
    claims = json.loads(user_block(fake, "agent_outputs"))["claims_to_verify"]
    assert {
        "agent": "skill_gap",
        "field": "matching_skills[1]",
        "expectation": "resume_supports",
    } in claims


def test_fabricated_matching_tool_is_still_a_deterministic_contradiction() -> None:
    """A named tool cannot be demonstrated by paraphrase: absent means fabricated."""
    job = DS_JOB + " Kubernetes is required."
    result = check_skill_gap(gap(["Kubernetes", "Tableau"], []), DS_RESUME, job)
    assert [(i.field, i.category, i.severity) for i in result.issues] == [
        ("matching_skills[0]", "contradiction", "high"),
        ("matching_skills[1]", "contradiction", "high"),
    ]
    assert result.ambiguous == []


def test_demonstrable_matching_skill_absent_from_the_job_is_still_a_contradiction() -> None:
    result = check_skill_gap(gap(["Statistics"], []), DS_RESUME, DS_JOB)
    assert [(i.field, i.severity) for i in result.issues] == [("matching_skills[0]", "high")]
    assert "job description does not" in result.issues[0].message


@pytest.mark.asyncio
async def test_fabricated_demonstrable_skill_is_caught_by_the_reviewer_at_high() -> None:
    """No resume evidence of preprocessing at all: the reviewer's unsupported claim
    keeps its high severity (F4 only calibrates contradictions)."""
    resume = "Taylor Example\nSKILLS\nPython, SQL\nPROJECTS\n- A to-do list web app.\n"
    unsupported = llm_issue(
        field="matching_skills[1]",
        severity="high",
        category="unsupported_claim",
        message="No part of the resume shows data preprocessing.",
    )
    fake = FakeGroqChat().queue(json_reply({"issues": [unsupported]}))
    result = await run(fake, ds_input(gap(["Python", "Data Preprocessing"], []), resume))
    issue = next(i for i in result.output.issues if i.field == "matching_skills[1]")
    assert issue.severity == "high"  # published issues carry no category
    assert result.output.verdict == "revise" and "skill_gap" in result.output.revise_targets


# --- MED-1: named tools filed under a demonstrable competency ----------------------------

TOOL_JOB = DS_JOB + " Required: unit testing with pytest and JUnit."


@pytest.mark.parametrize("tool", ["pytest", "JUnit"])
def test_unnamed_testing_tool_is_a_deterministic_contradiction(tool: str) -> None:
    """pytest and JUnit are aliases of Unit Testing, but they are tools: work
    cannot demonstrate them, so a resume that does not name them contradicts."""
    resume = DS_RESUME + "- Wrote automated tests that check every transformation step.\n"
    result = check_skill_gap(gap([tool], []), resume, TOOL_JOB)
    assert [(i.field, i.category, i.severity) for i in result.issues] == [
        ("matching_skills[0]", "contradiction", "high")
    ]
    assert result.ambiguous == []


def test_testing_tool_is_not_vouched_for_by_a_different_testing_tool() -> None:
    resume = DS_RESUME + "- Tested the Java service with JUnit.\n"
    result = check_skill_gap(gap(["pytest", "JUnit"], []), resume, TOOL_JOB)
    assert [(i.field, i.severity) for i in result.issues] == [("matching_skills[0]", "high")]
    assert result.ambiguous == []


def test_unit_testing_shown_by_described_work_goes_to_the_reviewer() -> None:
    resume = DS_RESUME + "- Wrote automated tests that check every transformation step.\n"
    assert skill_presence("Unit Testing", DocumentIndex(resume)) == "absent"
    result = check_skill_gap(gap(["Unit Testing"], []), resume, TOOL_JOB)
    assert result.issues == []
    assert expectations(result) == [("skill_gap", "matching_skills[0]", "resume_supports")]


def test_named_testing_tools_are_not_contradicted() -> None:
    resume = DS_RESUME + "- Tested pipelines with pytest and a Java service with JUnit.\n"
    report = gap(["pytest", "JUnit", "Unit Testing", "unit testing with pytest"], [])
    result = check_skill_gap(report, resume, TOOL_JOB)
    assert result.issues == []
    assert all(c.field != "matching_skills[0]" for c in result.ambiguous)
    assert all(c.field != "matching_skills[1]" for c in result.ambiguous)


# --- Fix A: a missing tool is not shown by a sibling tool of the same competency ---------


@pytest.mark.parametrize(("named", "missing"), [("JUnit", "pytest"), ("pytest", "JUnit")])
def test_missing_tool_is_not_contradicted_by_a_sibling_tool(named: str, missing: str) -> None:
    """Both are aliases of Unit Testing, but distinct tools: the named one is no
    proof of the other, so the gap goes to the reviewer instead of a certain HIGH."""
    resume = DS_RESUME + f"- Tested the service with {named}.\n"
    result = check_skill_gap(gap([], [missing]), resume, TOOL_JOB)
    assert result.issues == []
    assert expectations(result) == [("skill_gap", "missing_skills[0]", "resume_lacks")]


def test_missing_tool_named_nowhere_stays_a_reviewer_claim() -> None:
    result = check_skill_gap(gap([], ["pytest"]), DS_RESUME, TOOL_JOB)
    assert result.issues == []
    assert expectations(result) == [("skill_gap", "missing_skills[0]", "resume_lacks")]


def test_missing_tool_the_resume_names_is_still_a_certain_contradiction() -> None:
    resume = DS_RESUME + "- Tested the service with pytest.\n"
    result = check_skill_gap(gap([], ["pytest"]), resume, TOOL_JOB)
    assert [(i.field, i.category, i.severity) for i in result.issues] == [
        ("missing_skills[0]", "contradiction", "high")
    ]
    assert "listed as missing but the resume shows it" in result.issues[0].message
    assert result.ambiguous == []


def test_missing_competency_shown_by_a_named_tool_is_still_a_contradiction() -> None:
    """pytest is evidence of unit testing itself: that contradiction stays certain."""
    resume = DS_RESUME + "- Tested the service with pytest.\n"
    result = check_skill_gap(gap([], ["unit testing"]), resume, TOOL_JOB)
    assert [(i.field, i.severity) for i in result.issues] == [("missing_skills[0]", "high")]


# --- Alias-classification safeguard -------------------------------------------------------

# Every raw alias and case-sensitive alias of a DEMONSTRABLE_SKILLS entry, exactly as
# written in skill_aliases.json, classified once as (skill, kind, raw alias), where kind
# is "alias" (matched case-insensitively) or "case_sensitive". Raw strings are compared,
# so removing "ci/cd" is caught even though "ci cd" normalises to the same matching
# form, and the kind is compared, so moving "ML" between the lists (which changes how it
# matches) is caught too. Scrum and Kanban stay competencies by decision. A new alias on
# one of the 25 skills fails these tests until it is added deliberately to one group.
NAMED_TOOL_RAW_ALIASES = frozenset(
    {("Unit Testing", "alias", "pytest"), ("Unit Testing", "alias", "junit")}
)
_COMPETENCY_CASE_SENSITIVE = frozenset(
    {
        ("Machine Learning", "case_sensitive", "ML"),
        ("Natural Language Processing", "case_sensitive", "NLP"),
        ("Object-Oriented Programming", "case_sensitive", "OOP"),
        ("Object-Oriented Programming", "case_sensitive", "OOPS"),
    }
)
COMPETENCY_RAW_ALIASES = _COMPETENCY_CASE_SENSITIVE | frozenset(
    (skill, "alias", alias)
    for skill, aliases in {
        "REST APIs": ("rest api", "rest apis", "restful", "restful api", "restful apis"),
        "Machine Learning": ("machine learning",),
        "Deep Learning": ("deep learning",),
        "Natural Language Processing": ("natural language processing",),
        "Computer Vision": ("computer vision",),
        "Data Analysis": ("data analysis", "data analytics"),
        "Data Visualization": ("data visualization", "data visualisation"),
        "Statistics": (
            "statistics",
            "statistical analysis",
            "statistical modeling",
            "statistical modelling",
        ),
        "Feature Engineering": ("feature engineering",),
        "Data Preprocessing": (
            "data preprocessing",
            "data pre-processing",
            "data cleaning",
            "data cleansing",
        ),
        "ETL": ("etl", "elt", "extract transform load"),
        "Data Warehousing": ("data warehousing", "data warehouse", "data warehouses"),
        "CI/CD": (
            "ci/cd",
            "ci cd",
            "cicd",
            "continuous integration",
            "continuous delivery",
            "continuous deployment",
        ),
        "Microservices": ("microservices", "microservice", "micro services"),
        "System Design": ("system design", "systems design"),
        "Data Structures": ("data structures", "data structure"),
        "Algorithms": ("algorithms", "algorithm design"),
        "Object-Oriented Programming": (
            "object oriented programming",
            "object-oriented programming",
        ),
        "Unit Testing": ("unit testing", "unit tests"),
        "Agile": ("agile", "scrum", "kanban"),
        "Communication": ("communication", "communication skills"),
        "Leadership": ("leadership", "team lead", "team leadership"),
        "Teamwork": ("teamwork", "team player", "collaboration"),
        "Project Management": ("project management",),
        "Stakeholder Management": ("stakeholder management",),
    }.items()
    for alias in aliases
)

# Distinct raw aliases of one skill that share a normalised form, each reviewed:
# "ci/cd" matches a slash that "ci cd" does not, so both are needed; the two OOP
# spellings compile to the same pattern (redundant; removing one is a vocabulary
# decision, not made here). Any other collision fails the test below.
KNOWN_NORMALIZATION_COLLISIONS = {
    "ci cd": {("CI/CD", "alias", "ci/cd"), ("CI/CD", "alias", "ci cd")},
    "object oriented programming": {
        ("Object-Oriented Programming", "alias", "object oriented programming"),
        ("Object-Oriented Programming", "alias", "object-oriented programming"),
    },
}

AliasEntry = tuple[str, str, str]  # (skill, kind, raw alias)

NO_PROBLEMS = {"duplicates": set(), "unclassified": set(), "in_both_groups": set(), "stale": set()}


def demonstrable_alias_inventory(skills=None) -> list[AliasEntry]:
    """(skill, kind, raw alias) for every alias and case-sensitive alias; duplicates kept."""
    specs = skill_matcher().vocabulary.skills if skills is None else skills
    return [
        (spec.name, kind, alias)
        for spec in specs
        if spec.name in DEMONSTRABLE_SKILLS
        for kind, aliases in (("alias", spec.aliases), ("case_sensitive", spec.case_sensitive))
        for alias in aliases
    ]


def alias_classification_problems(
    inventory: list[AliasEntry],
    named: frozenset[AliasEntry] = NAMED_TOOL_RAW_ALIASES,
    competency: frozenset[AliasEntry] = COMPETENCY_RAW_ALIASES,
) -> dict[str, set[AliasEntry]]:
    """Duplicate, unclassified, doubly classified and stale raw alias entries."""
    raw = set(inventory)
    return {
        "duplicates": {entry for entry in raw if inventory.count(entry) > 1},
        "unclassified": raw - named - competency,
        "in_both_groups": set(named & competency),
        "stale": set((named | competency) - raw),
    }


def normalization_collisions(inventory: list[AliasEntry]) -> dict[str, set[AliasEntry]]:
    """Normalised forms (production normalize_for_match) shared by distinct raw aliases."""
    forms: dict[str, set[AliasEntry]] = {}
    for entry in set(inventory):
        forms.setdefault(normalize_for_match(entry[2]), set()).add(entry)
    return {form: entries for form, entries in forms.items() if len(entries) > 1}


def assert_collision(collisions: dict[str, set[AliasEntry]], form: str, expected: set) -> None:
    """A readable failure when an expected collision is missing or different (LOW-C)."""
    actual = collisions.get(form)
    assert actual == expected, (
        f"expected normalisation collision {form!r} between {sorted(expected)}, "
        f"found {sorted(actual) if actual else 'no collision'}"
    )


def _with_alias(skill: str, alias: str):
    """The real vocabulary with one extra raw alias on a skill."""
    return [
        spec.model_copy(update={"aliases": (*spec.aliases, alias)}) if spec.name == skill else spec
        for spec in skill_matcher().vocabulary.skills
    ]


def _without_alias(skill: str, alias: str):
    """The real vocabulary with one raw alias (or case-sensitive alias) removed."""
    specs = []
    for spec in skill_matcher().vocabulary.skills:
        if spec.name == skill:
            assert alias in (*spec.aliases, *spec.case_sensitive), alias
            spec = spec.model_copy(
                update={
                    "aliases": tuple(a for a in spec.aliases if a != alias),
                    "case_sensitive": tuple(a for a in spec.case_sensitive if a != alias),
                }
            )
        specs.append(spec)
    return specs


def test_every_raw_demonstrable_alias_is_classified_exactly_once() -> None:
    inventory = demonstrable_alias_inventory()
    assert alias_classification_problems(inventory) == NO_PROBLEMS
    assert {skill for skill, _, _ in inventory} == DEMONSTRABLE_SKILLS  # all 25 inspected
    assert {kind for _, kind, _ in inventory} == {"alias", "case_sensitive"}
    assert {("Agile", "alias", "scrum"), ("Agile", "alias", "kanban")} <= COMPETENCY_RAW_ALIASES


def test_named_tool_aliases_match_the_production_set() -> None:
    assert frozenset({"pytest", "junit"}) == NAMED_TOOL_ALIASES  # no silent expansion
    assert {normalize_for_match(a) for _, _, a in NAMED_TOOL_RAW_ALIASES} == NAMED_TOOL_ALIASES


def test_normalization_collisions_are_exactly_the_reviewed_ones() -> None:
    inventory = demonstrable_alias_inventory()
    assert normalization_collisions(inventory) == KNOWN_NORMALIZATION_COLLISIONS
    competency_forms = {normalize_for_match(a) for _, _, a in COMPETENCY_RAW_ALIASES}
    assert not competency_forms & NAMED_TOOL_ALIASES  # no competency alias reads as a tool


def test_safeguard_catches_an_unclassified_new_alias() -> None:
    problems = alias_classification_problems(
        demonstrable_alias_inventory(_with_alias("Unit Testing", "jest"))
    )
    assert problems == {**NO_PROBLEMS, "unclassified": {("Unit Testing", "alias", "jest")}}


def test_safeguard_catches_a_stale_named_tool_entry() -> None:
    problems = alias_classification_problems(
        demonstrable_alias_inventory(),
        named=NAMED_TOOL_RAW_ALIASES | {("Unit Testing", "alias", "mocha")},
    )
    assert problems == {**NO_PROBLEMS, "stale": {("Unit Testing", "alias", "mocha")}}


def test_safeguard_catches_an_alias_classified_in_both_groups() -> None:
    problems = alias_classification_problems(
        demonstrable_alias_inventory(),
        named=NAMED_TOOL_RAW_ALIASES | {("Agile", "alias", "scrum")},
    )
    assert problems == {**NO_PROBLEMS, "in_both_groups": {("Agile", "alias", "scrum")}}


def test_safeguard_catches_a_duplicate_raw_alias() -> None:
    problems = alias_classification_problems(
        demonstrable_alias_inventory(_with_alias("Agile", "agile"))
    )
    assert problems == {**NO_PROBLEMS, "duplicates": {("Agile", "alias", "agile")}}


@pytest.mark.parametrize(
    ("skill", "kind", "alias"),
    [
        ("Data Preprocessing", "alias", "data cleaning"),
        ("Machine Learning", "case_sensitive", "ML"),
        ("Unit Testing", "alias", "junit"),  # named tool
    ],
)
def test_safeguard_catches_removal_of_an_alias(skill: str, kind: str, alias: str) -> None:
    problems = alias_classification_problems(
        demonstrable_alias_inventory(_without_alias(skill, alias))
    )
    assert problems == {**NO_PROBLEMS, "stale": {(skill, kind, alias)}}


def test_safeguard_catches_removal_of_a_collision_sensitive_alias() -> None:
    """Removing 'ci/cd' must not pass because 'ci cd' normalises to the same form."""
    inventory = demonstrable_alias_inventory(_without_alias("CI/CD", "ci/cd"))
    assert alias_classification_problems(inventory) == {
        **NO_PROBLEMS,
        "stale": {("CI/CD", "alias", "ci/cd")},
    }
    assert normalization_collisions(inventory) != KNOWN_NORMALIZATION_COLLISIONS


def test_safeguard_catches_a_new_normalization_collision() -> None:
    inventory = demonstrable_alias_inventory(_with_alias("REST APIs", "rest-api"))
    assert alias_classification_problems(inventory)["unclassified"] == {
        ("REST APIs", "alias", "rest-api")
    }
    assert_collision(
        normalization_collisions(inventory),
        "rest api",
        {("REST APIs", "alias", "rest api"), ("REST APIs", "alias", "rest-api")},
    )


def _moved_alias(skill: str, alias: str, to: str):
    """The real vocabulary with one raw alias moved to the other list (to: alias or
    case_sensitive)."""
    specs = []
    for spec in skill_matcher().vocabulary.skills:
        if spec.name == skill:
            aliases = tuple(a for a in spec.aliases if a != alias)
            case_sensitive = tuple(a for a in spec.case_sensitive if a != alias)
            assert len(aliases) + len(case_sensitive) == (
                len(spec.aliases) + len(spec.case_sensitive) - 1
            ), alias
            if to == "alias":
                aliases = (*aliases, alias)
            else:
                case_sensitive = (*case_sensitive, alias)
            spec = spec.model_copy(update={"aliases": aliases, "case_sensitive": case_sensitive})
        specs.append(spec)
    return specs


def test_safeguard_catches_moving_a_case_sensitive_alias_to_aliases() -> None:
    """'ML' matched case-insensitively would match 'ml' anywhere: the move must fail."""
    problems = alias_classification_problems(
        demonstrable_alias_inventory(_moved_alias("Machine Learning", "ML", "alias"))
    )
    assert problems == {
        **NO_PROBLEMS,
        "unclassified": {("Machine Learning", "alias", "ML")},
        "stale": {("Machine Learning", "case_sensitive", "ML")},
    }


def test_safeguard_catches_moving_an_alias_back_to_case_sensitive() -> None:
    """With 'ML' classified as an ordinary alias, the real case-sensitive entry fails."""
    reclassified = (COMPETENCY_RAW_ALIASES - {("Machine Learning", "case_sensitive", "ML")}) | {
        ("Machine Learning", "alias", "ML")
    }
    problems = alias_classification_problems(
        demonstrable_alias_inventory(), competency=reclassified
    )
    assert problems == {
        **NO_PROBLEMS,
        "unclassified": {("Machine Learning", "case_sensitive", "ML")},
        "stale": {("Machine Learning", "alias", "ML")},
    }


@pytest.mark.parametrize(
    ("skill", "alias"), [("Machine Learning", "ML"), ("Object-Oriented Programming", "OOPS")]
)
def test_safeguard_catches_removal_of_a_case_sensitive_alias(skill: str, alias: str) -> None:
    problems = alias_classification_problems(
        demonstrable_alias_inventory(_without_alias(skill, alias))
    )
    assert problems == {**NO_PROBLEMS, "stale": {(skill, "case_sensitive", alias)}}


def test_missing_expected_collision_fails_with_a_readable_assertion(monkeypatch) -> None:
    """LOW-C: a lost collision is an assertion naming the collision, not a KeyError."""
    monkeypatch.setattr(sys.modules[__name__], "normalization_collisions", lambda inventory: {})
    with pytest.raises(AssertionError, match="expected normalisation collision 'rest api'"):
        test_safeguard_catches_a_new_normalization_collision()


# --- Fix A: exact-tool evidence, case and term boundaries ------------------------------


@pytest.mark.parametrize(
    ("resume_line", "missing"),
    [
        ("- Wrote unit tests for every transformation step.", "pytest"),  # competency alias
        ("- Built a pytestify helper for the team.", "pytest"),  # not a whole term
        ("- Tested the Java service with JUnit5.", "JUnit"),  # version-suffixed name
    ],
)
def test_missing_tool_without_its_exact_name_goes_to_the_reviewer(
    resume_line: str, missing: str
) -> None:
    result = check_skill_gap(gap([], [missing]), DS_RESUME + resume_line + "\n", TOOL_JOB)
    assert result.issues == []
    assert expectations(result) == [("skill_gap", "missing_skills[0]", "resume_lacks")]


@pytest.mark.parametrize(
    ("resume_line", "missing"),
    [
        ("- Tested the service with PyTest.", "pytest"),  # case-insensitive
        ("- Tested the Java service with JUNIT.", "JUnit"),  # case-insensitive
        ("- Measured coverage with pytest-cov.", "pytest"),  # hyphen splits the term
    ],
)
def test_missing_tool_the_resume_names_in_any_case_is_a_certain_contradiction(
    resume_line: str, missing: str
) -> None:
    result = check_skill_gap(gap([], [missing]), DS_RESUME + resume_line + "\n", TOOL_JOB)
    assert [(i.field, i.category, i.severity) for i in result.issues] == [
        ("missing_skills[0]", "contradiction", "high")
    ]
    assert result.ambiguous == []


# --- MED-2: a demonstrable competency combined with a named tool -------------------------

COMBINED = "machine learning with scikit-learn"
COMBINED_JOB = DS_JOB + " Required: machine learning with scikit-learn."
RESUME_WITHOUT_SKLEARN = DS_RESUME.replace(", scikit-learn", "").replace(
    "a scikit-learn classifier", "a classifier"
)


def test_combined_claim_with_a_named_tool_goes_to_the_reviewer() -> None:
    assert "scikit-learn" in DocumentIndex(DS_RESUME).skills
    result = check_skill_gap(gap([COMBINED], []), DS_RESUME, COMBINED_JOB)
    assert result.issues == []
    assert expectations(result) == [("skill_gap", "matching_skills[0]", "resume_supports")]


def test_combined_claim_with_an_unnamed_tool_is_a_deterministic_contradiction() -> None:
    assert "scikit-learn" not in RESUME_WITHOUT_SKLEARN.casefold()
    result = check_skill_gap(gap([COMBINED], []), RESUME_WITHOUT_SKLEARN, COMBINED_JOB)
    assert [(i.field, i.severity) for i in result.issues] == [("matching_skills[0]", "high")]
    assert "resume does not mention" in result.issues[0].message
    assert result.ambiguous == []


def test_combined_claim_absent_from_the_job_is_still_a_contradiction() -> None:
    job = DS_JOB + " Required: scikit-learn."  # no statistics anywhere in the job
    result = check_skill_gap(gap(["statistics with scikit-learn"], []), DS_RESUME, job)
    assert [(i.field, i.severity) for i in result.issues] == [("matching_skills[0]", "high")]
    assert "job description does not" in result.issues[0].message
    assert result.ambiguous == []


# --- M-1: preferred skills listed as missing ---------------------------------------------


def test_reviewer_prompt_rejects_preferred_skills_in_missing_skills() -> None:
    assert CRITIC_PROMPT.id == "critic@1.4.0"
    lowered = CRITIC_PROMPT.system.casefold()
    for word in ("preferred", "optional", "nice-to-have", "bonus"):
        assert word in lowered, word
    assert "missing_skills must contain only skills the job requires" in lowered
    assert "quote the job description" in lowered
    # The old wording accepted every resume_lacks claim the resume lacks.
    assert "otherwise the claim is correct" not in lowered
    assert (
        "a required skill that no part of the resume supports is correctly listed as missing"
        in lowered
    )


def test_reviewer_prompt_requires_concrete_evidence_for_four_skills() -> None:
    assert CRITIC_PROMPT.id == "critic@1.4.0"
    text = CRITIC_PROMPT.system
    lowered = text.casefold()
    assert "for a resume_supports claim on these skills, require concrete evidence" in lowered
    for skill, required, insufficient in (
        (
            "System Design",
            "explicit architecture decisions, component interactions, scalability or design "
            "trade-offs",
            "building an application is not enough",
        ),
        (
            "Data Structures",
            "actual use or implementation of identifiable data structures",
            "general programming experience is not enough",
        ),
        (
            "Algorithms",
            "concrete algorithm design, implementation or optimization",
            "general coding is not enough",
        ),
        (
            "Communication",
            "presenting findings, writing documentation or communicating with stakeholders",
            "teamwork alone is not enough",
        ),
    ):
        assert f"{skill} needs" in text, skill
        assert required in lowered, skill
        assert insufficient in lowered, skill
    # Valid paraphrased evidence must still count.
    assert "supports the skill even when it is paraphrased" in lowered


@pytest.mark.asyncio
async def test_preferred_skill_listed_as_missing_reaches_the_reviewer_and_is_revised() -> None:
    report = gap(["Python"], ["Data Preprocessing", "Tableau"])
    checks = check_skill_gap(report, DS_RESUME, DS_JOB)
    assert checks.issues == []
    assert ("skill_gap", "missing_skills[1]", "resume_lacks") in expectations(checks)

    contradiction = llm_issue(
        field="missing_skills[1]",
        severity="high",
        category="contradiction",
        message="The job description lists Tableau under 'Preferred: Tableau and Docker.'",
    )
    fake = FakeGroqChat().queue(json_reply({"issues": [contradiction]}))
    result = await run(fake, ds_input(report))
    issue = next(i for i in result.output.issues if i.field == "missing_skills[1]")
    assert "Preferred" in issue.message  # the job description is quoted as evidence
    assert result.output.verdict == "revise"
    assert result.output.revise_targets == ["skill_gap"]


@pytest.mark.asyncio
async def test_genuinely_missing_required_skill_stays_valid() -> None:
    job = DS_JOB.replace("machine learning.", "machine learning and Apache Spark.")
    report = gap(["Python", "SQL"], ["Apache Spark"])
    checks = check_skill_gap(report, DS_RESUME, job)
    assert checks.issues == []
    assert expectations(checks) == [("skill_gap", "missing_skills[0]", "resume_lacks")]

    fake = FakeGroqChat().queue(json_reply({"issues": []}))
    result = await run(
        fake,
        full_input(
            resume_text=DS_RESUME,
            profile=ds_profile(),
            job_description=job,
            interview=None,
            skill_gap=report,
        ),
    )
    assert result.output.verdict == "approve" and result.output.issues == []
