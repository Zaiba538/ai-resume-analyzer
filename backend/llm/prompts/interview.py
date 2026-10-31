"""Interview agent prompt: questions at three levels (preserved legacy feature).

As in the legacy app, questions are generated from the extracted profile.
"""

from backend.llm.prompts import PromptSpec

INTERVIEW_PROMPT = PromptSpec(
    name="interview",
    version="1.0.0",
    inputs=("profile",),
    system=(
        "You are an experienced technical interviewer. Using the candidate profile in the "
        "<profile> block (JSON), write interview questions grounded in the candidate's actual "
        "skills, experience and projects.\n"
        "Produce exactly {count_per_level} questions for each level:\n"
        "- beginner: fundamentals of the candidate's core skills;\n"
        "- intermediate: applied problems drawn from their experience and projects;\n"
        "- advanced: design, trade-offs and depth in their strongest areas.\n"
        "For every question give a short topic and a one-sentence rationale linking it to the "
        "profile. Do not repeat questions."
    ),
)
