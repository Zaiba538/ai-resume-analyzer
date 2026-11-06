import "./fake_dom.mjs";
const { renderReport } = await import("./js/results.js");
const { renderQuestionList, renderEvaluation, showError } = await import("./js/interview.js");
const { ApiError } = await import("./js/api.js");
const { el } = await import("./js/dom.js");
const input = JSON.parse(process.env.FRONTEND_INPUT);
const panels = Object.fromEntries(
  ["overview", "profile", "match", "review"].map((name) => [name, document.createElement("div")]),
);
renderReport(input.report, panels);
const questions = document.createElement("div");
renderQuestionList(questions, input.report.interview_questions, {
  evaluate: async () => ({}), onExpired: () => {}, emptyText: "none",
});
const evaluation = document.createElement("div");
renderEvaluation(evaluation, input.evaluation);
const alert = document.createElement("div");
showError(alert, new ApiError({ status: 422, code: "brand_new_code", serverMessage: input.payload,
  requestId: input.payload, details: [{ loc: ["body", input.payload], type: "x", msg: input.payload }] }));
const rejected = [];
for (const attrs of [{ onclick: "x" }, { style: "color:red" }, { href: "javascript:alert(1)" }, { srcdoc: "x" }]) {
  try { el("a", { attrs }); } catch { rejected.push(Object.keys(attrs)[0]); }
}
emit({
  trees: [...Object.values(panels), questions, evaluation, alert].map(serialize),
  rejected,
});
