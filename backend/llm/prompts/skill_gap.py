"""Skill-gap agent prompt (preserved legacy feature).

Uses the structured profile rather than the raw resume, as approved, which
reduces tokens and keeps matching grounded in extracted skills.
"""

from backend.llm.prompts import PromptSpec, revision_spec

SKILL_GAP_PROMPT = PromptSpec(
    name="skill_gap",
    version="1.0.0",
    inputs=("profile", "job_description"),
    system=(
        "You are a careful career advisor. Compare the candidate profile in the <profile> "
        "block (JSON) with the job description in the <job_description> block.\n"
        "- matching_skills: skills required or preferred by the job that the profile shows;\n"
        "- missing_skills: skills the job requires that the profile does not show;\n"
        "- recommendations: up to 15 concrete, actionable steps to close the most important "
        "gaps, most important first.\n"
        "Use the job description's wording for skill names. Base every item on the two "
        "inputs only; do not assume skills that are not evidenced."
    ),
)

# M7 (decision D1): used only when Critic feedback is present; SKILL_GAP_PROMPT itself
# is unchanged, so requests without feedback are identical to before.
SKILL_GAP_REVISION_PROMPT = revision_spec(SKILL_GAP_PROMPT, "1.1.0")
