"""M11 Phase 1: behaviour of the real frontend modules, run with Node.js.

Node's standard library only (fetch, Response, FormData, Blob); no npm
packages. Skipped when Node.js is not installed. The modules are imported
unchanged from frontend/js; ``fetch`` is replaced by a stub per scenario.
"""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

from backend.api.routes.report import PDF_DEADLINE_SECONDS
from backend.api.uploads import UPLOAD_READ_TIMEOUT_SECONDS
from backend.core.config import Settings
from backend.schemas.interview import (
    DEFAULT_COUNT_PER_LEVEL,
    MAX_COUNT_PER_LEVEL,
    MIN_COUNT_PER_LEVEL,
)
from backend.schemas.interview_api import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    InterviewQuestionsResponse,
)
from backend.schemas.report import AnalysisResponse
from tests.fixtures.frontend import (
    FRONTEND_DIR,
    backend_error_codes,
    requires_node,
    run_frontend_script,
    sample_report,
)

pytestmark = requires_node


@pytest.fixture(scope="module")
def request_results(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("requests")
    return run_frontend_script(tmp, "requests", {"backend_codes": sorted(backend_error_codes())})


def test_retry_after_is_parsed_and_shown(request_results) -> None:
    capacity = request_results["capacity"]
    assert capacity["isApiError"] and capacity["status"] == 429
    assert capacity["code"] == "analysis_capacity_exceeded" and capacity["retryAfter"] == 10
    assert capacity["requestId"] == "req-1"
    assert capacity["message"].endswith("You can try again in 10 seconds.")
    assert request_results["rateLimited"]["message"].endswith("You can try again in 2 minutes.")
    assert 28 <= request_results["retryAfterDate"] <= 30
    assert request_results["retryAfterBad"] is None


def test_an_expired_report_is_recognised_with_its_message(request_results) -> None:
    expired = request_results["expired"]
    assert expired["expired"] is True and expired["status"] == 404
    assert "expired" in expired["message"] and "run the analysis again" in expired["message"]
    others = {
        name: result
        for name, result in request_results.items()
        if isinstance(result, dict) and name not in {"expired", "messages", "errorMessages"}
    }
    assert others and not any(result.get("expired") for result in others.values())


def test_validation_details_name_the_field_never_the_value(request_results) -> None:
    validation = request_results["validation"]
    assert validation["code"] == "validation_error"
    assert validation["details"] == ["answer: String should have at most 10000 characters"]


def test_non_json_and_unexpected_bodies_fall_back_to_status_messages(request_results) -> None:
    proxy = request_results["proxyHtml"]
    assert proxy["code"] is None and proxy["status"] == 502 and proxy["requestId"] == "edge-7"
    assert proxy["message"] == "The service is temporarily unavailable. Please try again."
    assert (
        request_results["noEnvelope"]["message"]
        == "The server did not respond in time. Please try again."
    )
    # A code this frontend does not know yet: the backend's fixed message is shown.
    assert request_results["unknownCode"]["message"] == "server text"


def test_network_failures_timeouts_and_cancellation(request_results) -> None:
    assert request_results["network"]["code"] == "network_error"
    assert "could not be reached" in request_results["network"]["message"]
    assert request_results["timeout"]["code"] == "client_timeout"
    assert request_results["cancelled"]["code"] == "cancelled"
    assert request_results["badJson"]["code"] == "invalid_response"


def test_pdf_downloads_are_checked(request_results) -> None:
    assert request_results["pdf"] == {
        "ok": True,
        "value": {"filename": "resume-analysis-report.pdf", "size": 8},
    }
    assert request_results["pdfWrongType"]["code"] == "invalid_response"
    assert request_results["success"] == {"ok": True, "value": {"levels": ["beginner"]}}
    assert request_results["encodedUrl"] == "/api/v1/report/a%2Fb%3Fc/pdf"


def test_every_backend_code_gets_its_own_mapped_message(request_results) -> None:
    messages = request_results["messages"]
    mapped = request_results["errorMessages"]
    for code, message in messages.items():
        assert message == mapped[code], code
        assert message and "{" not in message


def test_requests_match_the_backend_contracts(tmp_path) -> None:
    calls = run_frontend_script(tmp_path, "bodies")
    blank, with_job, questions, stored, generated = calls
    assert blank["url"] == "/api/v1/analyze" and blank["method"] == "POST"
    assert blank["form"] == [["file", "file:cv.pdf"]]  # a blank job description is not sent
    assert with_job["form"] == [["file", "file:cv.pdf"], ["job_description", "Data engineer"]]
    assert "Content-Type" not in blank["headers"]  # the browser sets the multipart boundary
    assert questions["url"] == "/api/v1/interview/questions"
    assert questions["json"] == {
        "report_id": "rid",
        "levels": ["beginner", "advanced"],
        "count_per_level": 2,
    }
    assert questions["headers"]["Content-Type"] == "application/json"
    assert stored["json"] == {
        "report_id": "rid",
        "answer": "My answer",
        "question_id": "beginner-1",
    }
    assert generated["json"] == {
        "report_id": "rid",
        "answer": "A",
        "question": {"id": "advanced-2", "question": "Q?", "topic": "T", "rationale": "R"},
    }
    assert all(call["credentials"] == "same-origin" for call in calls)


def _upper_bound(field: str) -> int:
    """The highest value any configuration can give a settings field (its ``le``)."""
    bounds = [m.le for m in Settings.model_fields[field].metadata if getattr(m, "le", None)]
    assert len(bounds) == 1, field
    return int(bounds[0])


@pytest.fixture(scope="module")
def limit_results(tmp_path_factory) -> dict:
    return run_frontend_script(tmp_path_factory.mktemp("limits"), "limits")


def test_browser_limits_mirror_the_backend_defaults(limit_results) -> None:
    result = limit_results
    defaults = Settings(_env_file=None)
    limits = result["limits"]
    assert limits["maxUploadBytes"] == defaults.max_upload_bytes
    assert limits["maxJobDescriptionChars"] == defaults.max_job_description_chars
    assert limits["maxAnswerChars"] == defaults.max_answer_chars
    # The only hard limits are the bounds that no configuration can exceed.
    assert limits["absoluteMaxUploadBytes"] == _upper_bound("max_upload_bytes")
    assert limits["absoluteMaxJobDescriptionChars"] == _upper_bound("max_job_description_chars")
    assert limits["absoluteMaxAnswerChars"] == _upper_bound("max_answer_chars")
    assert limits["countPerLevel"] == {
        "min": MIN_COUNT_PER_LEVEL,
        "max": MAX_COUNT_PER_LEVEL,
        "default": DEFAULT_COUNT_PER_LEVEL,
    }
    timeouts = result["timeouts"]
    assert timeouts["analyze"] > defaults.analysis_deadline_seconds * 1000
    assert timeouts["interviewQuestions"] > defaults.interview_deadline_seconds * 1000
    assert timeouts["interviewEvaluate"] > defaults.evaluation_deadline_seconds * 1000
    assert timeouts["reportPdf"] > PDF_DEADLINE_SECONDS * 1000


def test_analysis_timeout_covers_upload_deadline_analysis_deadline_and_margin(
    limit_results,
) -> None:
    defaults = Settings(_env_file=None)
    budget = limit_results["budget"]
    assert budget["upload"] == UPLOAD_READ_TIMEOUT_SECONDS
    assert budget["analysis"] == defaults.analysis_deadline_seconds
    assert budget["margin"] >= 60
    analyze_ms = limit_results["timeouts"]["analyze"]
    assert analyze_ms == (budget["upload"] + budget["analysis"] + budget["margin"]) * 1000
    assert analyze_ms > (UPLOAD_READ_TIMEOUT_SECONDS + defaults.analysis_deadline_seconds) * 1000
    assert analyze_ms >= 720 * 1000


def test_an_analysis_timeout_says_the_server_may_still_finish(tmp_path) -> None:
    out = run_frontend_script(tmp_path, "analysis_timeout")
    assert out["delays"] == [780 * 1000]  # the deadline the real request used
    assert out["code"] == "analysis_client_timeout"
    assert "stopped waiting" in out["message"] and "may still finish" in out["message"]
    assert "may still finish" in out["genericTimeout"]


def test_file_validation_blocks_only_what_no_configuration_accepts(limit_results) -> None:
    files, warnings = limit_results["files"], limit_results["fileWarnings"]
    assert files["pdf"] is None and files["docx"] is None and files["atLimit"] is None
    assert files["none"] == "Choose a resume file to upload."
    assert files["doc"] == files["noExtension"] == "Only PDF and DOCX files are supported."
    assert files["empty"] == "The file is empty. Choose a different file."
    # Above the 5 MB default but within what a server may be configured to accept:
    # sent, with a warning; the server decides.
    for name in ("overLimit", "sixMegabytes", "atAbsolute"):
        assert files[name] is None, name
        assert "default limit of 5 MB" in warnings[name] and "may reject" in warnings[name]
    assert warnings["pdf"] is None and warnings["atLimit"] is None
    assert "20 MB" in files["overAbsolute"] and "no server configuration" in files["overAbsolute"]


def test_text_limits_above_the_defaults_warn_but_are_not_rejected(limit_results) -> None:
    jd, jd_warnings = limit_results["jd"], limit_results["jdWarnings"]
    assert jd["atDefault"] is None and jd_warnings["atDefault"] is None
    for name in ("overDefault", "atAbsolute"):
        assert jd[name] is None, name
        assert "20,000" in jd_warnings[name]
    assert "100,000" in jd["overAbsolute"]
    answers, answer_warnings = limit_results["answers"], limit_results["answerWarnings"]
    assert "Write an answer" in answers["blank"]
    assert answers["atDefault"] is None and answer_warnings["atDefault"] is None
    for name in ("overDefault", "atAbsolute"):
        assert answers[name] is None, name
        assert "5,000" in answer_warnings[name]
    assert "20,000" in answers["overAbsolute"]


def test_answer_form_never_truncates_and_announces_limits_once(tmp_path) -> None:
    out = run_frontend_script(tmp_path, "answer_form")
    defaults = Settings(_env_file=None)
    assert "maxlength" not in out["textareaAttrs"]  # no silent truncation
    assert out["textareaAttrs"]["aria-describedby"] == out["counterAttrs"]["id"]
    assert "aria-live" not in out["counterAttrs"]  # not announced on every keystroke
    assert "aria-live" not in out["resultAttrs"]  # the evaluation is focused instead
    assert out["announcements"] == [
        "Answer: Longer than the default limit of 5,000 characters; the server may reject it."
    ]
    assert out["counterText"].startswith(f"{defaults.max_answer_chars + 3:,} characters")
    # Above the default: sent to the server. Above every configuration: refused here.
    assert out["sent"] == [defaults.max_answer_chars + 1000]
    assert "no server configuration accepts" in out["formText"]


def test_report_rendering_uses_real_fields_and_never_an_ats_score(tmp_path) -> None:
    with_job = sample_report()
    without_job = sample_report(with_job=False)
    out = run_frontend_script(
        tmp_path, "render_report", {"withJob": with_job, "withoutJob": without_job}
    )
    rendered = out["withJob"]
    profile = with_job["resume_profile"]
    for skill in profile["skills"]:
        assert skill in rendered["profile"]
    for entry in profile["experience"]:
        assert entry["organization"] in rendered["profile"]
    metrics = with_job["match"]["metrics"]
    assert f"{metrics['skill_coverage']['value']:.1f}%" in rendered["match"]
    assert f"{with_job['match']['baseline']['tfidf_keyword_match']:.1f}%" in rendered["match"]
    assert "Not calibrated. No ATS score is produced." in rendered["match"]
    for recommendation in with_job["skill_gap"]["recommendations"]:
        assert recommendation in rendered["match"]
    assert with_job["critic"]["issues"][0]["message"] in rendered["review"]
    assert "Instruction override attempt" in rendered["review"]
    assert "Approved" in rendered["overview"]
    skipped = out["withoutJob"]
    assert "No job description was provided" in skipped["match"]
    assert "No job description was provided" in skipped["overview"]
    assert "%" not in skipped["match"]
    # Even a hypothetical calibrated status is shown as text, never as its score.
    assert out["atsOther"] == "calibrated"


def test_expired_reports_switch_interview_actions_to_the_expired_state(tmp_path) -> None:
    out = run_frontend_script(tmp_path, "expiry")
    for action in ("answer", "generate", "noReport"):
        assert out[action]["expired"] == ["report_not_found"], action
        assert "expired" in out[action]["text"] and "run the analysis again" in out[action]["text"]
    assert out["noReport"]["calls"] == 0  # an expired report is never sent to the server


# --- the real app.js: stale reports, expiry and announcements ------------------------------


class _Elements(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element_id = dict(attrs).get("id")
        if element_id:
            self.elements.append((element_id, tag))


OLD_QUESTION = {
    "id": "beginner-1",
    "question": "Old report question?",
    "topic": "Python",
    "rationale": "Generated for report A.",
}
NEW_QUESTION = {
    "id": "beginner-1",
    "question": "New report question?",
    "topic": "SQL",
    "rationale": "Generated for report B.",
}


def _questions_response(question: dict) -> dict:
    return InterviewQuestionsResponse.model_validate(
        {
            "levels": ["beginner"],
            "count_per_level": 1,
            "questions": {"beginner": [question], "intermediate": [], "advanced": []},
            "security_flags": [],
        }
    ).model_dump(mode="json")


@pytest.fixture(scope="module")
def journey(tmp_path_factory) -> tuple[dict, dict, dict]:
    report_a, report_b = sample_report(), sample_report()
    report_a["interview_questions"]["beginner"][0]["question"] = "Report A stored question?"
    report_b["interview_questions"]["beginner"][0]["question"] = "Report B stored question?"
    for report in (report_a, report_b):
        AnalysisResponse.model_validate(report)
    for question in (OLD_QUESTION, NEW_QUESTION):
        _questions_response(question)  # the stubbed responses are schema-valid
    evaluation = AnswerEvaluationResponse.model_validate(
        {
            "question_id": "beginner-1",
            "level": "beginner",
            "evaluation": {
                "rubric": {
                    criterion: {"rating": 4, "justification": "Clear and correct."}
                    for criterion in (
                        "relevance",
                        "technical_accuracy",
                        "depth",
                        "clarity",
                        "structure",
                    )
                },
                "strengths": ["Accurate."],
                "improvements": ["Add an example."],
                "model_answer_outline": ["Define a list."],
                "overall_rating": 4.0,
            },
            "security_flags": [],
        }
    ).model_dump(mode="json")
    parser = _Elements()
    parser.feed((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))
    data = {
        "elements": parser.elements,
        "reportA": report_a,
        "reportB": report_b,
        "oldQuestion": OLD_QUESTION,
        "newQuestion": NEW_QUESTION,
        "evaluation": evaluation,
    }
    out = run_frontend_script(tmp_path_factory.mktemp("journey"), "app_journey", data)
    return out, report_a, report_b


def test_an_old_reports_questions_never_appear_under_the_new_report(journey) -> None:
    out, report_a, _ = journey
    assert out["afterA"] == {"results": True, "overview": True}
    assert out["generateRequestReport"] == report_a["report_id"]  # captured at the start
    assert out["generateLabelWhilePending"] == "Generating..."
    assert out["aborted"] == {"generate": True, "evaluate": True}  # cancelled on switch
    assert out["staleQuestionsShown"] is False  # discarded even though it arrived
    late = out["afterLateGenerate"]
    assert late["generateLabel"] == "Generate questions" and late["generateDisabled"] is False
    assert late["generateAlert"] == ""


def test_no_evaluation_pairs_the_new_report_with_an_old_question(journey) -> None:
    out, report_a, report_b = journey
    assert out["missing"] == []  # every expected request was really sent
    evaluations = out["evaluations"]
    assert [body["report_id"] for body in evaluations] == [
        report_a["report_id"],  # started while A was active
        report_b["report_id"],
        report_b["report_id"],
    ]
    for body in evaluations:
        AnswerEvaluationRequest.model_validate(body)
    assert evaluations[1]["question_id"] == "beginner-1"
    assert evaluations[2]["question"] == NEW_QUESTION
    for body in evaluations:
        if body["report_id"] == report_b["report_id"]:
            assert OLD_QUESTION["question"] not in str(body)


def test_a_late_404_from_an_old_report_cannot_expire_the_new_report(journey) -> None:
    out, _, _ = journey
    assert out["afterLate404"] == {
        "bannerHidden": True,
        "generateDisabled": False,
        "generateLabel": "Generate questions",
        "pdfDisabled": False,
        "resultsAlert": "",
        "generateAlert": "",
    }


def test_generate_button_stays_disabled_after_the_report_expires(journey) -> None:
    out, _, _ = journey
    expired = out["afterExpiry"]
    assert expired["bannerHidden"] is False
    assert expired["generateDisabled"] is True  # not re-enabled by the finally block
    assert expired["pdfDisabled"] is True
    assert expired["generateLabel"] == "Generate questions"
    assert "expired" in expired["generateAlert"]
    assert out["requestsAfterDisabledClick"] == out["requestsAfterExpiry"]


def test_status_changes_are_announced_and_the_evaluation_heading_gets_focus(journey) -> None:
    out, _, _ = journey
    assert out["announcedStart"].startswith("Analysis started.")
    assert out["announcedGenerated"] == "1 interview question generated."
    assert out["evaluationFocus"] == {
        "tag": "H4",
        "text": "Evaluation of your answer",
        "tabindex": "-1",
    }


# --- expiry versus countdowns and open answer forms (audit N-1, N-2) ----------------------


@pytest.fixture(scope="module")
def expiry_run(tmp_path_factory) -> dict:
    report = sample_report()
    AnalysisResponse.model_validate(report)
    parser = _Elements()
    parser.feed((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))
    data = {"elements": parser.elements, "report": report}
    return run_frontend_script(tmp_path_factory.mktemp("expiry"), "app_expiry", data)


def test_a_countdown_never_re_enables_an_unavailable_button(expiry_run) -> None:
    out = expiry_run
    assert out["unavailableFromTheStart"] == {"text": "Generate questions", "disabled": True}
    assert out["availableDuring"] == {"text": "Try again in 1 s", "disabled": True}
    assert out["availableAfter"] == {"text": "Generate questions", "disabled": False}


def test_countdowns_running_when_the_report_expires_leave_buttons_disabled(expiry_run) -> None:
    out = expiry_run
    during = out["duringCountdowns"]
    assert during["pdf"] == {"text": "Try again in 1 s", "disabled": True}
    assert {"text": "Try again in 1 s", "disabled": True} in during["controls"]
    after = out["afterCountdowns"]
    assert after["pdf"] == {"text": "Download PDF report", "disabled": True}
    assert after["generate"] is True
    assert after["controls"] and all(control["disabled"] for control in after["controls"])
    assert {"text": "Evaluate answer", "disabled": True} in after["controls"]


def test_open_answer_forms_and_toggles_are_disabled_when_the_report_expires(expiry_run) -> None:
    out = expiry_run
    right_after = out["rightAfterExpiry"]
    assert right_after["banner"] is True and right_after["generate"] is True
    controls = right_after["controls"]
    assert {"text": "Hide answer form", "disabled": True} in controls  # the open form's toggle
    assert {"text": "Practise this question", "disabled": True} in controls
    assert all(control["disabled"] for control in controls)
    assert out["requestsAfterExpiry"] == 0  # nothing is sent for the expired report
