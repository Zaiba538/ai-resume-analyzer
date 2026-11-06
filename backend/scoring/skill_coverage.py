"""Skill coverage: how many of the job description's skills the resume shows.

Both texts are scanned with the same deterministic matcher
(``backend.scoring.skill_normalizer``); skills are compared by canonical name,
so aliases and duplicates never inflate either count.

* jd_skills_total: distinct canonical skills found in the job description.
* matched: those also found in the resume; missing: the rest. Both keep the
  job description's order of first appearance.
* value: matched / jd_skills_total * 100, rounded to two decimals; 0.0 when the
  job description mentions no known skill (``jd_skills_total`` = 0 makes that
  case explicit).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.scoring.skill_normalizer import skill_matcher


@dataclass(frozen=True)
class SkillCoverage:
    value: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    jd_skills_total: int


def skill_coverage(resume_text: str, job_description: str) -> SkillCoverage:
    matcher = skill_matcher()
    job_skills = matcher.find(job_description)
    resume_skills = set(matcher.find(resume_text))
    matched = tuple(skill for skill in job_skills if skill in resume_skills)
    missing = tuple(skill for skill in job_skills if skill not in resume_skills)
    total = len(job_skills)
    value = round(len(matched) / total * 100, 2) if total else 0.0
    return SkillCoverage(value=value, matched=matched, missing=missing, jd_skills_total=total)
