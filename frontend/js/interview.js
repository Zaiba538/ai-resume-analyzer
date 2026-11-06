/**
 * Interview practice.
 *
 * Stored questions come from the analysis report (report.interview_questions)
 * and are evaluated by question_id. Questions generated with
 * POST /api/v1/interview/questions are not stored on the server, so they are
 * evaluated by sending the question object itself, as the API requires
 * (exactly one question source per evaluation).
 *
 * Evaluations show only what POST /api/v1/interview/evaluate returns: the five
 * rubric ratings with justifications, the server-computed overall rating,
 * strengths, improvements, a model-answer outline and security flags.
 *
 * Every request belongs to the report that was active when it started (a
 * "ticket"). Loading a new report aborts the old report's pending requests, and
 * a response or error that still arrives for an old report is discarded: it is
 * never rendered under the new report and never marks the new report expired.
 */

import {
  ApiError,
  LIMITS,
  evaluateAnswer,
  generateQuestions,
  isExpiredReport,
  userMessage,
  validationDetails,
} from "./api.js";
import { announce, clear, el, renderAlert, startCountdown, textList } from "./dom.js";
import { flagTable } from "./results.js";

export const LEVEL_LABELS = Object.freeze({
  beginner: "Beginner",
  intermediate: "Intermediate",
  advanced: "Advanced",
});
const LEVELS = Object.freeze(["beginner", "intermediate", "advanced"]);

export const CRITERION_LABELS = Object.freeze({
  relevance: "Relevance",
  technical_accuracy: "Technical accuracy",
  depth: "Depth",
  clarity: "Clarity",
  structure: "Structure",
});

/** Show an API error in a container, with validation details and the request ID. */
export function showError(container, error) {
  renderAlert(container, {
    message: userMessage(error),
    details: validationDetails(error),
    requestId: error instanceof ApiError ? error.requestId : null,
  });
}

/**
 * Disable a button for Retry-After seconds, showing the remaining time on it.
 *
 * isAvailable says whether the button may be used at all (for example, false
 * once the report has expired). It is checked on every tick and at the end: if
 * the button became unavailable meanwhile, the countdown stops, the label is
 * restored and the button stays disabled.
 *
 * @param {Element} button
 * @param {unknown} error
 * @param {() => boolean} [isAvailable]
 */
export function coolDown(button, error, isAvailable = () => true) {
  if (!(error instanceof ApiError) || !Number.isInteger(error.retryAfter) || error.retryAfter <= 0) {
    return;
  }
  const original = button.textContent;
  button.disabled = true;
  let stopped = false;
  let stop = () => {};
  const finish = () => {
    stopped = true;
    stop();
    button.textContent = original;
    button.disabled = !isAvailable();
  };
  stop = startCountdown(
    error.retryAfter,
    (remaining) => {
      if (stopped) {
        return;
      }
      if (!isAvailable()) {
        finish(); // expired during the countdown: never re-enable
        return;
      }
      button.textContent = `Try again in ${remaining} s`;
    },
    () => {
      if (!stopped) {
        finish();
      }
    },
  );
  if (stopped) {
    stop(); // became unavailable on the very first tick, before stop was known
  }
}

/** Render an AnswerEvaluationResponse. */
export function renderEvaluation(container, response) {
  const evaluation = response.evaluation;
  const rows = Object.keys(CRITERION_LABELS).map((criterion) => {
    const score = evaluation.rubric[criterion];
    return el("tr", {}, [
      el("th", { text: CRITERION_LABELS[criterion], attrs: { scope: "row" } }),
      el("td", { className: "rating" }, [
        el("span", { text: `${score.rating} / 5` }),
        el("meter", { className: "meter meter--small", attrs: { min: 1, max: 5, value: score.rating } }),
      ]),
      el("td", { text: score.justification }),
    ]);
  });
  const children = [
    el("h4", { className: "evaluation__title", text: "Evaluation of your answer", attrs: { tabindex: "-1" } }),
    el("div", { className: "evaluation__overall" }, [
      el("p", { className: "stat__label", text: "Overall rating" }),
      el("p", { className: "stat__value", text: `${evaluation.overall_rating.toFixed(1)} / 5` }),
      el("p", { className: "stat__note", text: "The mean of the five criterion ratings." }),
    ]),
    el("div", { className: "table-wrap" }, [
      el("table", { className: "table" }, [
        el("thead", {}, [
          el("tr", {}, [
            el("th", { text: "Criterion", attrs: { scope: "col" } }),
            el("th", { text: "Rating", attrs: { scope: "col" } }),
            el("th", { text: "Justification", attrs: { scope: "col" } }),
          ]),
        ]),
        el("tbody", {}, rows),
      ]),
    ]),
    el("div", { className: "columns" }, [
      el("div", {}, [el("h4", { text: "Strengths" }), textList(evaluation.strengths, "None listed.")]),
      el("div", {}, [
        el("h4", { text: "Improvements" }),
        textList(evaluation.improvements, "None listed."),
      ]),
    ]),
    el("h4", { text: "Model answer outline" }),
    textList(evaluation.model_answer_outline, "No outline was given."),
  ];
  if (response.security_flags.length > 0) {
    children.push(
      el("h4", { text: "Security flags" }),
      el("p", {
        className: "muted",
        text: "Informational only: parts of the input resemble prompt-injection attempts.",
      }),
      flagTable(response.security_flags, ""),
    );
  }
  clear(container);
  container.append(el("div", { className: "evaluation" }, children));
  return children[0];
}

/** A message describing why an answer cannot be sent, or null. */
export function validateAnswer(answer) {
  if (answer.trim() === "") {
    return "Write an answer before submitting it for evaluation.";
  }
  if (answer.length > LIMITS.absoluteMaxAnswerChars) {
    return `The answer is longer than ${LIMITS.absoluteMaxAnswerChars.toLocaleString("en")} characters, which no server configuration accepts. Please shorten it.`;
  }
  return null;
}

/** A non-blocking warning when an answer exceeds the default limit, or null. */
export function answerWarning(answer) {
  if (answer.length > LIMITS.maxAnswerChars) {
    return `Longer than the default limit of ${LIMITS.maxAnswerChars.toLocaleString("en")} characters; the server may reject it.`;
  }
  return null;
}

/**
 * The answer form for one question.
 *
 * There is no maxlength: the server's answer limit is configurable, so a long
 * answer is never cut off silently. It is flagged, and the server decides.
 *
 * @param {{id: string, question: string, topic: string, rationale: string}} question
 * @param {{evaluate: (question: object, answer: string) => Promise<object>,
 *          onExpired: (error: unknown) => void, isStale: () => boolean,
 *          isAvailable: () => boolean}} context
 */
function answerForm(question, { evaluate, onExpired, isStale, isAvailable, track }) {
  const fieldId = `answer-${Math.random().toString(36).slice(2)}`;
  const counterId = `${fieldId}-count`;
  const textarea = el("textarea", {
    attrs: { id: fieldId, rows: 6, "aria-describedby": counterId },
  });
  // Not a live region: the count is read with the field (aria-describedby), and
  // crossing the default limit is announced once.
  const counter = el("span", { className: "counter", attrs: { id: counterId } });
  const submit = el("button", {
    className: "button button--primary",
    text: "Evaluate answer",
    attrs: { type: "submit" },
  });
  submit.disabled = !isAvailable();
  track(submit); // disabled together with the other controls when the report expires
  const alert = el("div", { className: "alert-slot", attrs: { "aria-live": "polite" } });
  // Not a live region either: focus moves to the evaluation heading instead, so
  // the whole evaluation is not read out on every change.
  const result = el("div", { className: "evaluation-slot" });
  const form = el("form", { className: "form answer", attrs: { novalidate: true } }, [
    el("label", { className: "form__label", text: "Your answer", attrs: { for: fieldId } }),
    textarea,
    el("div", { className: "form__meta" }, [
      el("span", { text: "Answer in English. Do not include personal or confidential information." }),
      counter,
    ]),
    alert,
    el("div", { className: "form__actions" }, [submit]),
    result,
  ]);

  let warned = false;
  const updateCounter = () => {
    const warning = answerWarning(textarea.value);
    counter.textContent = `${textarea.value.length.toLocaleString("en")} characters${warning ? `. ${warning}` : ""}`;
    counter.classList.toggle("counter--over", Boolean(warning));
    if (warning && !warned) {
      announce(`Answer: ${warning}`);
    }
    warned = Boolean(warning);
  };
  textarea.addEventListener("input", updateCounter);
  updateCounter();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submit.disabled) {
      return;
    }
    const answer = textarea.value;
    const problem = validateAnswer(answer);
    if (problem) {
      renderAlert(alert, { message: problem });
      return;
    }
    clear(alert);
    submit.disabled = true;
    submit.textContent = "Evaluating...";
    form.setAttribute("aria-busy", "true");
    let failed = null;
    try {
      const response = await evaluate(question, answer);
      if (isStale()) {
        return; // the report changed while waiting: discard the old report's result
      }
      renderEvaluation(result, response).focus(); // focus the heading, not a live region
    } catch (error) {
      if (isStale()) {
        return; // an old report's error (including its 404) must not touch the new one
      }
      failed = error;
      showError(alert, error);
      if (isExpiredReport(error)) {
        onExpired(error);
      }
    } finally {
      if (!isStale()) {
        form.removeAttribute("aria-busy");
        submit.textContent = "Evaluate answer";
        submit.disabled = !isAvailable(); // stays disabled once the report has expired
        if (failed && !submit.disabled) {
          coolDown(submit, failed, isAvailable);
        }
      }
    }
  });
  return form;
}

/**
 * Render question cards grouped by level.
 *
 * @param {Element} container
 * @param {{beginner: object[], intermediate: object[], advanced: object[]}} questions
 * @param {{evaluate: (question: object, answer: string) => Promise<object>,
 *          onExpired: (error: unknown) => void, emptyText: string,
 *          isStale?: () => boolean, isAvailable?: () => boolean}} options
 */
export function renderQuestionList(
  container,
  questions,
  {
    evaluate,
    onExpired,
    emptyText,
    isStale = () => false,
    isAvailable = () => true,
    track = () => {},
  },
) {
  const context = { evaluate, onExpired, isStale, isAvailable, track };
  clear(container);
  const groups = LEVELS.filter((level) => Array.isArray(questions[level]) && questions[level].length > 0);
  if (groups.length === 0) {
    container.append(el("p", { className: "empty", text: emptyText }));
    return;
  }
  for (const level of groups) {
    const items = questions[level].map((question) => {
      const slot = el("div");
      const toggle = el("button", {
        className: "button button--secondary",
        text: "Practise this question",
        attrs: { type: "button", "aria-expanded": "false" },
      });
      toggle.disabled = !isAvailable();
      track(toggle);
      toggle.addEventListener("click", () => {
        const open = toggle.getAttribute("aria-expanded") === "true";
        if (open) {
          clear(slot);
          toggle.setAttribute("aria-expanded", "false");
          toggle.textContent = "Practise this question";
        } else {
          slot.append(answerForm(question, context));
          toggle.setAttribute("aria-expanded", "true");
          toggle.textContent = "Hide answer form";
        }
      });
      return el("li", { className: "question" }, [
        el("p", { className: "question__text", text: question.question }),
        el("p", { className: "question__meta" }, [
          el("span", { className: "badge badge--neutral", text: question.topic }),
        ]),
        question.rationale ? el("p", { className: "question__rationale", text: question.rationale }) : null,
        toggle,
        slot,
      ]);
    });
    container.append(
      el("div", { className: "question-group" }, [
        el("h3", { text: LEVEL_LABELS[level] }),
        el("ol", { className: "questions" }, items),
      ]),
    );
  }
}

/**
 * Wire up the interview panel.
 *
 * @param {{getReportId: () => string|null,
 *          onExpired: (error: unknown, reportId: string|null) => void}} options
 *   getReportId returns the active, unexpired report's ID, or null.
 */
export function initInterview({ getReportId, onExpired }) {
  const stored = document.getElementById("stored-questions");
  const generated = document.getElementById("generated-questions");
  const form = document.getElementById("generate-form");
  const button = document.getElementById("generate-button");
  const alert = document.getElementById("generate-alert");
  const select = document.getElementById("count-per-level");

  for (let count = LIMITS.countPerLevel.min; count <= LIMITS.countPerLevel.max; count += 1) {
    select.append(
      el("option", {
        text: String(count),
        attrs: { value: count, selected: count === LIMITS.countPerLevel.default },
      }),
    );
  }

  // One epoch per active report. Its AbortController cancels that report's
  // pending interview requests when another report becomes active.
  let epoch = 0;
  let controller = new AbortController();
  // The current report's "Practise" toggles and "Evaluate answer" buttons, so
  // that expiry can disable all of them, including forms that are already open.
  const controls = new Set();

  /** The report a request belongs to, captured when the request starts. */
  const ticketFor = (reportId) => ({ reportId, epoch, signal: controller.signal });
  const isStale = (ticket) => ticket.epoch !== epoch;
  const isAvailable = () => getReportId() !== null;

  /** The ticket's report ID, while that report is still active and unexpired. */
  const reportIdFor = (ticket) => {
    if (!ticket.reportId || getReportId() !== ticket.reportId) {
      throw new ApiError({ status: 404, code: "report_not_found" });
    }
    return ticket.reportId;
  };

  /** Evaluation context for the questions of one report. */
  const contextFor = (ticket, source) => ({
    evaluate: (question, answer) =>
      evaluateAnswer(reportIdFor(ticket), source(question), answer, { signal: ticket.signal }),
    onExpired: (error) => {
      if (!isStale(ticket)) {
        onExpired(error, ticket.reportId);
      }
    },
    isStale: () => isStale(ticket),
    isAvailable,
    track: (control) => {
      if (!isStale(ticket)) {
        controls.add(control);
      }
    },
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (button.disabled) {
      return;
    }
    const levels = LEVELS.filter((level) =>
      form.querySelector(`input[name="level"][value="${level}"]`).checked,
    );
    if (levels.length === 0) {
      renderAlert(alert, { message: "Choose at least one level." });
      return;
    }
    const count = Number.parseInt(select.value, 10);
    const ticket = ticketFor(getReportId());
    clear(alert);
    button.disabled = true;
    button.textContent = "Generating...";
    form.setAttribute("aria-busy", "true");
    let failed = null;
    try {
      const response = await generateQuestions(reportIdFor(ticket), levels, count, {
        signal: ticket.signal,
      });
      if (isStale(ticket)) {
        return; // generated for a report that is no longer shown
      }
      // Generated questions stay bound to the report they were generated for.
      renderQuestionList(generated, response.questions, {
        ...contextFor(ticket, (question) => ({ question })),
        emptyText: "No questions were generated.",
      });
      if (response.security_flags.length > 0) {
        generated.prepend(
          el("div", { className: "notice" }, [
            el("p", { text: "Informational security flags were raised while generating these questions." }),
            flagTable(response.security_flags, ""),
          ]),
        );
      }
      const total = LEVELS.reduce((sum, level) => sum + (response.questions[level] || []).length, 0);
      announce(total === 1 ? "1 interview question generated." : `${total} interview questions generated.`);
    } catch (error) {
      if (isStale(ticket)) {
        return; // an old report's failure (including its 404) is not shown under the new one
      }
      failed = error;
      showError(alert, error);
      if (isExpiredReport(error)) {
        onExpired(error, ticket.reportId);
      }
    } finally {
      if (!isStale(ticket)) {
        form.removeAttribute("aria-busy");
        button.textContent = "Generate questions";
        button.disabled = !isAvailable(); // an expired report keeps the button disabled
        if (failed && !button.disabled) {
          coolDown(button, failed, isAvailable);
        }
      }
    }
  });

  return {
    /**
     * Disable every answer control of the current report once it has expired.
     * (A report never becomes unexpired: a new report replaces all controls.)
     */
    setExpired(expired) {
      if (!expired) {
        return;
      }
      for (const control of controls) {
        control.disabled = true;
      }
    },

    /**
     * Show the questions of a newly loaded report: cancel the previous report's
     * pending requests and forget its generated questions.
     */
    showReport(report) {
      controller.abort();
      controller = new AbortController();
      epoch += 1;
      controls.clear(); // the old report's controls are discarded with its questions
      form.removeAttribute("aria-busy");
      button.textContent = "Generate questions";
      clear(alert);
      clear(generated);
      renderQuestionList(stored, report.interview_questions, {
        ...contextFor(ticketFor(report.report_id), (question) => ({ questionId: question.id })),
        emptyText: "The analysis did not include interview questions.",
      });
    },
  };
}
