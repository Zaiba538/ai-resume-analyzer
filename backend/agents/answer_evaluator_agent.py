"""Answer-evaluator agent (approved M4 decision A).

Inputs are the interview question, the candidate's answer and the difficulty
level, all treated as untrusted; no profile context is used in M4. The model
rates five fixed criteria (integers 1-5, one-sentence justifications) and lists
strengths, improvements and a model-answer outline. ``overall_rating`` is
computed here, never by the model.
"""

from __future__ import annotations

from backend.agents.base import AgentInputError, AgentResult, clean_input
from backend.llm.prompts import build_messages
from backend.llm.prompts.evaluator import EVALUATOR_PROMPT, LEVEL_INDEX
from backend.llm.structured import StructuredLLM
from backend.schemas.interview import AnswerEvaluation, AnswerEvaluationDraft


async def run_answer_evaluator_agent(
    llm: StructuredLLM, *, question: str, answer: str, level: str
) -> AgentResult[AnswerEvaluation]:
    if level not in LEVEL_INDEX:
        raise AgentInputError("level must be beginner, intermediate or advanced.")
    clean_question = clean_input(question, field="question")
    clean_answer = clean_input(answer, field="answer")
    result = await llm.generate(
        "evaluator",
        AnswerEvaluationDraft,
        build_messages(
            EVALUATOR_PROMPT,
            {"question": clean_question, "answer": clean_answer},
            params={"level_index": LEVEL_INDEX[level]},
        ),
        source_texts=[clean_question, clean_answer],
    )
    evaluation = AnswerEvaluation.from_draft(result.output)
    return AgentResult.from_structured(
        result, output=evaluation, prompt_id=EVALUATOR_PROMPT.id
    )
