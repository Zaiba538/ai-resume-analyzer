"""Approved M6 decision Q7: M5 metrics are not Critic inputs, and the Critic never
calculates, modifies or replaces scores."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

import backend.agents.critic_agent as critic_agent
import backend.agents.critic_checks as critic_checks
from backend.agents.critic_agent import CriticInput, run_critic_agent
from tests.fixtures.groq import FakeGroqChat, json_reply
from tests.unit.test_agents import make_llm
from tests.unit.test_critic_checks import JOB, RESUME, gap, profile

SCORE_WORDS = ("match", "metric", "score", "semantic", "coverage", "completeness", "tfidf", "ats")


def test_critic_input_has_no_score_fields() -> None:
    names = [field.name for field in dataclasses.fields(CriticInput)]
    assert names == [
        "resume_text",
        "profile",
        "job_description",
        "interview",
        "interview_levels",
        "count_per_level",
        "skill_gap",
        "security_flags",
    ]
    assert not any(word in name for name in names for word in SCORE_WORDS)


@pytest.mark.parametrize("module", [critic_agent, critic_checks], ids=lambda m: m.__name__)
def test_critic_modules_import_no_scoring_metrics(module) -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    scoring = {name for name in imported if name.startswith("backend.scoring")}
    # Only the M5 skill vocabulary (aliases), never a metric or the ATS gate.
    assert scoring <= {"backend.scoring.skill_normalizer"}
    assert "backend.schemas.match" not in imported


@pytest.mark.asyncio
async def test_critic_request_contains_no_scores() -> None:
    fake = FakeGroqChat().queue(json_reply({"issues": []}))
    await run_critic_agent(
        make_llm(fake),
        CriticInput(
            resume_text=RESUME,
            profile=profile(),
            job_description=JOB,
            skill_gap=gap(["Python"], ["Apache Airflow"]),
        ),
    )
    user = fake.bodies[0]["messages"][1]["content"]
    payload = json.loads(user.split("<agent_outputs>\n", 1)[1].split("\n</agent_outputs>", 1)[0])
    assert set(payload) == {
        "profile",
        "interview",
        "skill_gap",
        "requested_interview_levels",
        "count_per_level",
        "claims_to_verify",
        "suspicious_outputs",
        "input_security_flags",
        "validated_outputs",
    }
    for word in ("semantic_similarity", "skill_coverage", "tfidf", "ats_score", "not_calibrated"):
        assert word not in user
