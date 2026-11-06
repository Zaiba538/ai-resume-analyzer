import "./fake_dom.mjs";

// Every report-bound action in interview.js must switch to the expired state on a
// 404 report_not_found, and show the expired message.
const { ApiError } = await import("./js/api.js");
const { renderQuestionList, initInterview } = await import("./js/interview.js");
const out = {};
const question = { id: "beginner-1", question: "What is a list?", topic: "Python", rationale: "" };

// 1. Answering a question whose report has expired.
{
  const expired = [];
  const container = document.createElement("div");
  renderQuestionList(container, { beginner: [question], intermediate: [], advanced: [] }, {
    evaluate: async () => {
      throw new ApiError({ status: 404, code: "report_not_found" });
    },
    onExpired: (error) => expired.push(error.code),
    emptyText: "none",
  });
  await fire(findTag(container, "BUTTON"), "click");
  const form = findTag(container, "FORM");
  findTag(form, "TEXTAREA").value = "A list is an ordered sequence.";
  await fire(form, "submit");
  out.answer = { expired, text: form.textContent };
}

// 2. Generating questions for an expired report (the server answers 404).
{
  const expired = [];
  for (const id of ["stored-questions", "generated-questions", "generate-alert"]) register(id);
  const form = register("generate-form", "form");
  form.querySelector = () => ({ checked: true });
  register("generate-button", "button").textContent = "Generate questions";
  register("count-per-level", "select").value = "3";
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: { code: "report_not_found", message: "x", request_id: "r" } }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  initInterview({ getReportId: () => "3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10", onExpired: (e) => expired.push(e.code) });
  await fire(form, "submit");
  out.generate = { expired, text: document.getElementById("generate-alert").textContent };
}

// 3. No report ID any more (already marked expired): nothing is sent.
{
  const expired = [];
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new Error("must not be called");
  };
  const form = register("generate-form", "form");
  form.querySelector = () => ({ checked: true });
  register("generate-button", "button");
  register("generate-alert");
  initInterview({ getReportId: () => null, onExpired: (e) => expired.push(e.code) });
  await fire(form, "submit");
  out.noReport = { expired, calls, text: document.getElementById("generate-alert").textContent };
}
emit(out);
