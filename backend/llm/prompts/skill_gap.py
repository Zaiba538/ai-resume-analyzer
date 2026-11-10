"""Skill-gap agent prompt (preserved legacy feature).

Uses the structured profile rather than the raw resume, as approved, which
reduces tokens and keeps matching grounded in extracted skills.

M11 Phase 2 (F1, version 1.2.0): evidence is looked for across the whole
profile (skills, experience highlights, projects, certifications), work the
candidate demonstrably did counts as evidence of a skill, and preferred skills
are never reported as missing.
"""

from backend.llm.prompts import PromptSpec, revision_spec

SKILL_GAP_PROMPT = PromptSpec(
    name="skill_gap",
    version="1.2.0",
    inputs=("profile", "job_description"),
    system=(
        "You are a careful career advisor. Compare the candidate profile in the <profile> "
        "block (JSON) with the job description in the <job_description> block.\n"
        "Before deciding about any skill, look for evidence in every part of the profile: "
        "skills, experience (each entry's highlights), projects and certifications. A skill "
        "is evidenced when it is named, or when the profile describes work that demonstrated "
        "it; for example, cleaning and transforming records with Pandas, including "
        "missing-value and duplicate handling, is evidence of data preprocessing. Do not "
        "treat a skill as evidenced only because a related tool or topic is mentioned.\n"
        "- matching_skills: skills required or preferred by the job that the profile shows "
        "evidence of;\n"
        "- missing_skills: skills the job requires (stated as required, essential or "
        "must-have, or not marked as optional) for which no part of the profile gives "
        "evidence. Never list a preferred, nice-to-have, bonus or optional skill as missing;\n"
        "- recommendations: up to 15 concrete, actionable steps to close the most important "
        "gaps, most important first: missing required skills before preferred skills the "
        "profile does not show.\n"
        "Use the job description's wording for skill names. Base every item on the two "
        "inputs only; do not assume skills that are not evidenced."
    ),
)

# M7 (decision D1): used only when Critic feedback is present; SKILL_GAP_PROMPT itself
# is unchanged by feedback, so requests without feedback use it as is. Versions: 1.0.0
# and 1.1.0 (revision) before M11 Phase 2; 1.2.0 and 1.3.0 (revision) since.
SKILL_GAP_REVISION_PROMPT = revision_spec(SKILL_GAP_PROMPT, "1.3.0")
