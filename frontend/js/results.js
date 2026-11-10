/**
 * Rendering of the AnalysisReport returned by POST /api/v1/analyze.
 *
 * Only fields of the backend schemas are shown (backend/schemas/report.py and the
 * models it references). All values are inserted as text. The ATS composite is
 * deliberately not calibrated: its status is shown and no score is displayed or
 * derived. Match metrics are labelled as uncalibrated informational metrics.
 */

import { chipList, clear, el, textList } from "./dom.js";

export const VERDICT_LABELS = Object.freeze({
  approve: {
    label: "Approved",
    tone: "success",
    text: "The reviewer found no issues that require changes.",
  },
  revise: {
    label: "Revisions suggested",
    tone: "warning",
    text: "The reviewer noted issues; check them before relying on these results.",
  },
  flagged: {
    label: "Flagged",
    tone: "danger",
    text: "The reviewer flagged these results. Review them carefully before relying on them.",
  },
});

export const FLAG_TYPE_LABELS = Object.freeze({
  hidden_unicode: "Hidden Unicode characters",
  instruction_override: "Instruction override attempt",
  role_impersonation: "Role impersonation",
  prompt_exfiltration: "Prompt extraction attempt",
  score_manipulation: "Score manipulation attempt",
  delimiter_injection: "Delimiter injection",
  classifier: "Prompt-injection classifier",
  unexpected_url: "Unexpected URL in generated output",
});

export const FLAG_SOURCE_LABELS = Object.freeze({
  resume: "Resume",
  job_description: "Job description",
  answer: "Answer",
  question: "Question",
  llm_output: "Generated output",
});

const AGENT_LABELS = Object.freeze({ resume: "Resume", interview: "Interview", skill_gap: "Skill gap" });

function label(map, key) {
  return Object.hasOwn(map, key) ? map[key] : String(key);
}

function percent(value) {
  return `${Number(value).toFixed(1)}%`;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function section(title, children, className = "card") {
  return el("section", { className }, [el("h2", { text: title }), ...children]);
}

function stat(title, value, note = null) {
  return el("div", { className: "stat" }, [
    el("p", { className: "stat__label", text: title }),
    el("p", { className: "stat__value", text: value }),
    note ? el("p", { className: "stat__note", text: note }) : null,
  ]);
}

function badge(text, tone) {
  return el("span", { className: `badge badge--${tone}`, text });
}

function meter(value, max = 100) {
  return el("meter", { className: "meter", attrs: { min: 0, max, value: Number(value) } });
}

/** The verdict presentation for a critic verdict value. */
export function verdictInfo(verdict) {
  return Object.hasOwn(VERDICT_LABELS, verdict)
    ? VERDICT_LABELS[verdict]
    : { label: String(verdict), tone: "neutral", text: "" };
}

/** Human-readable text for the ATS composite status. Never a score. */
export function atsStatusText(atsScore) {
  if (atsScore && atsScore.status === "not_calibrated") {
    return "Not calibrated. No ATS score is produced.";
  }
  return atsScore && typeof atsScore.status === "string" ? atsScore.status : "Unavailable";
}

/** A short status label for the ATS composite (never a score). */
export function atsShortText(atsScore) {
  if (atsScore && atsScore.status === "not_calibrated") {
    return "Not calibrated";
  }
  return atsScore && typeof atsScore.status === "string" ? atsScore.status : "Unavailable";
}

// --- Overview ---------------------------------------------------------------------------

export function renderOverview(container, report) {
  const profile = report.resume_profile;
  const verdict = verdictInfo(report.critic.verdict);
  const stats = [
    stat("Skills", String(profile.skills.length)),
    stat("Experience entries", String(profile.experience.length)),
    stat("Education entries", String(profile.education.length)),
    stat("Projects", String(profile.projects.length)),
    stat("Certifications", String(profile.certifications.length)),
    stat("Security flags", String(report.security_flags.length), "Informational only"),
  ];
  const matchChildren = [];
  if (report.match) {
    const metrics = report.match.metrics;
    matchChildren.push(
      el("div", { className: "stats" }, [
        stat("Skill coverage", percent(metrics.skill_coverage.value), "Uncalibrated metric"),
        stat("Semantic similarity", percent(metrics.semantic_similarity.value), "Uncalibrated metric"),
        stat("Section completeness", percent(metrics.section_completeness.value), "Uncalibrated metric"),
        stat("ATS composite", atsShortText(report.match.ats_score), "No ATS score is produced"),
      ]),
    );
  } else {
    matchChildren.push(
      el("p", {
        className: "empty",
        text: "No job description was provided, so job matching and skill-gap analysis were skipped.",
      }),
    );
  }
  clear(container);
  container.append(
    section("Resume summary", [el("div", { className: "stats" }, stats)]),
    section("Reviewer verdict", [
      el("p", {}, [badge(verdict.label, verdict.tone)]),
      el("p", { text: verdict.text }),
      el("p", {
        className: "muted",
        text: `Issues noted: ${report.critic.issues.length}. Revision rounds: ${report.critic.revisions}.`,
      }),
    ]),
    section("Job match", matchChildren),
  );
}

// --- Resume profile ---------------------------------------------------------------------

function experienceItem(item) {
  const heading = [item.role, item.organization].filter((part) => part).join(" - ");
  return el("li", { className: "entry" }, [
    el("p", { className: "entry__title", text: heading || "Untitled role" }),
    item.period ? el("p", { className: "entry__meta", text: item.period }) : null,
    item.highlights.length > 0 ? textList(item.highlights, "") : null,
  ]);
}

function educationItem(item) {
  const meta = [item.institution, item.year].filter((part) => part).join(", ");
  return el("li", { className: "entry" }, [
    el("p", { className: "entry__title", text: item.qualification }),
    meta ? el("p", { className: "entry__meta", text: meta }) : null,
  ]);
}

function entryList(items, render, emptyText) {
  if (items.length === 0) {
    return el("p", { className: "empty", text: emptyText });
  }
  return el("ul", { className: "entries" }, items.map(render));
}

export function renderProfile(container, report) {
  const profile = report.resume_profile;
  clear(container);
  container.append(
    section("Skills", [chipList(profile.skills, "No skills were found in the resume.")]),
    section("Experience", [
      entryList(profile.experience, experienceItem, "No experience was found in the resume."),
    ]),
    section("Education", [
      entryList(profile.education, educationItem, "No education was found in the resume."),
    ]),
    section("Projects", [textList(profile.projects, "No projects were found in the resume.")]),
    section("Certifications", [
      chipList(profile.certifications, "No certifications were found in the resume."),
    ]),
  );
}

// --- Job match and skill gap ------------------------------------------------------------

function metricBlock(title, value, description, extra = []) {
  return el("div", { className: "metric" }, [
    el("div", { className: "metric__head" }, [
      el("p", { className: "metric__title", text: title }),
      el("p", { className: "metric__value", text: percent(value) }),
    ]),
    meter(value),
    el("p", { className: "metric__note", text: description }),
    ...extra,
  ]);
}

export function renderMatch(container, report) {
  clear(container);
  if (!report.match || !report.skill_gap) {
    container.append(
      section("Job match", [
        el("p", {
          className: "empty",
          text: "No job description was provided, so job matching and skill-gap analysis were skipped. Run a new analysis with a job description to see them.",
        }),
      ]),
    );
    return;
  }
  const { metrics, baseline, ats_score: atsScore } = report.match;
  const coverage = metrics.skill_coverage;
  const sections = metrics.section_completeness;
  const gap = report.skill_gap;
  container.append(
    section("Match metrics", [
      el("p", {
        className: "muted",
        text: "These are separate, uncalibrated metrics. They are not an ATS score and are not combined into one.",
      }),
      el("div", { className: "metrics" }, [
        metricBlock(
          "Skill coverage",
          coverage.value,
          `Skills from the job description found in the resume: ${coverage.matched.length} of ${coverage.jd_skills_total}.`,
          [
            el("p", { className: "metric__subtitle", text: "Matched" }),
            chipList(coverage.matched, "None matched.", "success"),
            el("p", { className: "metric__subtitle", text: "Missing" }),
            chipList(coverage.missing, "None missing.", "warning"),
          ],
        ),
        metricBlock(
          "Semantic similarity",
          metrics.semantic_similarity.value,
          "Embedding similarity between the resume and the job description. Not calibrated.",
        ),
        metricBlock(
          "Section completeness",
          sections.value,
          "Standard resume sections found in the document.",
          [
            el("p", { className: "metric__subtitle", text: "Present" }),
            chipList(sections.present, "None found.", "success"),
            el("p", { className: "metric__subtitle", text: "Absent" }),
            chipList(sections.absent, "None missing.", "warning"),
          ],
        ),
        metricBlock(
          "Keyword baseline (TF-IDF)",
          baseline.tfidf_keyword_match,
          "The original keyword-overlap score, kept for comparison.",
        ),
      ]),
      el("div", { className: "ats" }, [
        el("p", { className: "ats__title", text: "ATS composite score" }),
        el("p", { className: "ats__status", text: atsStatusText(atsScore) }),
      ]),
    ]),
    section("Skill gap", [
      el("div", { className: "columns" }, [
        el("div", {}, [
          el("h3", { text: "Matching skills" }),
          chipList(gap.matching_skills, "No matching skills were identified.", "success"),
        ]),
        el("div", {}, [
          el("h3", { text: "Missing skills" }),
          chipList(gap.missing_skills, "No missing skills were identified.", "warning"),
        ]),
      ]),
      el("h3", { text: "Recommendations" }),
      textList(gap.recommendations, "No recommendations were given."),
    ]),
  );
}

// --- Review and safety ------------------------------------------------------------------

/** A table of security flags, or an empty-state paragraph. */
export function flagTable(flags, emptyText) {
  if (!Array.isArray(flags) || flags.length === 0) {
    return el("p", { className: "empty", text: emptyText });
  }
  const rows = flags.map((flag) =>
    el("tr", {}, [
      el("td", { text: label(FLAG_TYPE_LABELS, flag.type) }),
      el("td", { text: label(FLAG_SOURCE_LABELS, flag.source) }),
      el("td", {
        text:
          flag.score === null || flag.score === undefined
            ? "Not scored"
            : `${Number(flag.score).toFixed(2)}${flag.calibrated ? "" : " (uncalibrated)"}`,
      }),
    ]),
  );
  return el("div", { className: "table-wrap" }, [
    el("table", { className: "table" }, [
      el("thead", {}, [
        el("tr", {}, [
          el("th", { text: "Type", attrs: { scope: "col" } }),
          el("th", { text: "Found in", attrs: { scope: "col" } }),
          el("th", { text: "Classifier score", attrs: { scope: "col" } }),
        ]),
      ]),
      el("tbody", {}, rows),
    ]),
  ]);
}

function issueList(issues) {
  if (issues.length === 0) {
    return el("p", { className: "empty", text: "The reviewer raised no issues." });
  }
  return el(
    "ul",
    { className: "entries" },
    issues.map((issue) =>
      el("li", { className: "entry" }, [
        el("p", { className: "entry__title" }, [
          badge(issue.severity, issue.severity === "high" ? "danger" : issue.severity === "medium" ? "warning" : "neutral"),
          ` ${label(AGENT_LABELS, issue.agent)}: ${issue.field}`,
        ]),
        el("p", { text: issue.message }),
      ]),
    ),
  );
}

function definitionList(entries, emptyText) {
  if (entries.length === 0) {
    return el("p", { className: "empty", text: emptyText });
  }
  return el(
    "dl",
    { className: "definitions" },
    entries.flatMap(([term, value]) => [el("dt", { text: term }), el("dd", { text: value })]),
  );
}

export function renderReview(container, report) {
  const verdict = verdictInfo(report.critic.verdict);
  const meta = report.meta;
  clear(container);
  container.append(
    section("Reviewer", [
      el("p", {}, [badge(verdict.label, verdict.tone)]),
      el("p", { text: verdict.text }),
      el("p", { className: "muted", text: `Revision rounds: ${report.critic.revisions}` }),
      issueList(report.critic.issues),
    ]),
    section("Security flags", [
      el("p", {
        className: "muted",
        text: "Flags mark text that resembles prompt-injection attempts or unexpected content. They are informational: they never change or block the analysis.",
      }),
      flagTable(report.security_flags, "No security flags were raised."),
    ]),
    section("Run details", [
      definitionList(
        [
          ["Created", formatDate(report.created_at)],
          ["Available until", formatDate(report.expires_at)],
        ],
        "",
      ),
      el("h3", { text: "Models" }),
      definitionList(Object.entries(meta.models), "No model information."),
      el("h3", { text: "Step timings" }),
      definitionList(
        Object.entries(meta.timings_ms).map(([step, ms]) => [step, `${Math.round(ms)} ms`]),
        "No timing information.",
      ),
    ]),
  );
}

/** Render every results panel. */
export function renderReport(report, panels) {
  renderOverview(panels.overview, report);
  renderProfile(panels.profile, report);
  renderMatch(panels.match, report);
  renderReview(panels.review, report);
}

/** The "available until" sentence shown in the results header. */
export function expiryText(report) {
  const date = new Date(report.expires_at);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return `This report is kept on the server until ${date.toLocaleString()}.`;
}
