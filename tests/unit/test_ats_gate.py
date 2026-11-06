"""M5 exit criterion: the ATS composite gate proves ``not_calibrated``."""

from __future__ import annotations

import builtins
import json

import pytest
from pydantic import ValidationError

from backend.schemas.match import AtsScore
from backend.scoring.ats_composite import evaluate_ats_gate

EXPECTED = {
    "status": "not_calibrated",
    "score": None,
    "calibration_version": None,
    "evaluation_summary": None,
}


def test_gate_is_not_calibrated_with_null_score() -> None:
    assert evaluate_ats_gate().model_dump(mode="json") == EXPECTED


def test_gate_serialises_nulls_explicitly() -> None:
    assert json.loads(evaluate_ats_gate().model_dump_json()) == EXPECTED


def test_gate_is_stable_across_calls() -> None:
    assert all(evaluate_ats_gate() == evaluate_ats_gate() for _ in range(10))


@pytest.mark.parametrize(
    "override",
    [
        {"status": "calibrated"},
        {"score": 87.5},
        {"score": 0},
        {"calibration_version": "v1"},
        {"evaluation_summary": {"auc": 0.9}},
        {"approved": True},
        {"weights": [0.5, 0.3, 0.2]},
    ],
)
def test_model_rejects_any_calibrated_state(override) -> None:
    with pytest.raises(ValidationError):
        AtsScore.model_validate({**EXPECTED, **override})


def test_model_is_immutable() -> None:
    gate = evaluate_ats_gate()
    with pytest.raises(ValidationError):
        gate.score = 99.0  # type: ignore[misc]


def test_unexpected_calibration_files_are_ignored(tmp_path, monkeypatch) -> None:
    for directory in (tmp_path, tmp_path / "evaluation"):
        directory.mkdir(exist_ok=True)
        (directory / "calibration.json").write_text(
            json.dumps({"approved": True, "version": "v9", "weights": [1, 0, 0], "score": 99}),
            encoding="utf-8",
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATS_CALIBRATION", "approved")
    monkeypatch.setenv("ATS_SCORE_ENABLED", "true")
    assert evaluate_ats_gate().model_dump(mode="json") == EXPECTED


def test_gate_reads_no_files(monkeypatch) -> None:
    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("the ATS gate must not read files in M5")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    assert evaluate_ats_gate().model_dump(mode="json") == EXPECTED


def test_gate_module_defines_no_weights_or_thresholds() -> None:
    import backend.scoring.ats_composite as module

    public = {name for name in vars(module) if not name.startswith("__")}
    assert public <= {"annotations", "AtsScore", "_NOT_CALIBRATED", "evaluate_ats_gate"}
