import "./fake_dom.mjs";

// Runs the real app.js (which wires upload.js, analysis.js, interview.js and
// results.js) against a fake page built from the IDs in index.html, with a fetch
// stub whose responses the script releases one by one. This is a simulated DOM,
// not a browser: no layout, CSS or real events.
const input = JSON.parse(process.env.FRONTEND_INPUT);
const { reportA, reportB, oldQuestion } = input;
for (const [id, tag] of input.elements) register(id, tag);
const byId = (id) => document.getElementById(id);
byId("generate-form").querySelector = () => ({ checked: true });
byId("count-per-level").value = "1";

const settle = () => new Promise((resolve) => setTimeout(resolve, 80));
const json = (status, body) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const notFound = () =>
  json(404, { error: { code: "report_not_found", message: "Report not found.", request_id: "r-404" } });

// Every request waits until the script releases it. A request's response can
// arrive even after it was aborted, the worst case for stale-response handling.
const calls = [];
globalThis.fetch = (url, init) =>
  new Promise((resolve) => {
    calls.push({
      url,
      body: typeof init.body === "string" ? JSON.parse(init.body) : null,
      signal: init.signal,
      release: resolve,
    });
  });
const pending = (path) => calls.filter((call) => call.url.includes(path) && !call.done);
// A request that was never sent (for example, refused locally as expired) is
// recorded as missing instead of crashing, so the test reports the real outcome.
const missing = [];
const release = async (call, response, name = "request") => {
  if (!call) {
    missing.push(name);
    await settle();
    return;
  }
  call.done = true;
  call.release(response);
  await settle();
};

const analyse = async (report) => {
  byId("resume-file").files = [new File(["%PDF-1.7"], "cv.pdf", { type: "application/pdf" })];
  await fire(byId("resume-file"), "change");
  await fire(byId("upload-form"), "submit");
  await settle();
  await release(pending("/analyze")[0], json(200, report));
};
const openAnswerForm = async (container) => {
  const toggle = findTag(container, "BUTTON"); // "Practise this question"
  if (!toggle) return document.createElement("form");
  await fire(toggle, "click");
  const form = findTag(container, "FORM");
  findTag(form, "TEXTAREA").value = "A list is an ordered, mutable sequence.";
  return form;
};
const generatedText = () => byId("generated-questions").textContent;
const stateOf = () => ({
  bannerHidden: byId("expired-banner").hidden,
  generateDisabled: byId("generate-button").disabled,
  generateLabel: byId("generate-button").textContent,
  pdfDisabled: byId("pdf-button").disabled,
  resultsAlert: byId("results-alert").textContent,
  generateAlert: byId("generate-alert").textContent,
});

await import("./js/app.js");
byId("generate-button").textContent = "Generate questions";
byId("pdf-button").textContent = "Download PDF report";
const out = {};

// --- Report A, with requests left pending ----------------------------------------
await analyse(reportA);
out.announcedStart = byId("app-status").textContent;
out.afterA = { results: !byId("view-results").hidden, overview: byId("panel-overview").textContent.length > 0 };

const generateA = fire(byId("generate-form"), "submit"); // 1. generate for report A
await settle();
const generateRequestA = pending("/interview/questions")[0];
out.generateRequestReport = generateRequestA.body.report_id;
out.generateLabelWhilePending = byId("generate-button").textContent;

const formA = await openAnswerForm(byId("stored-questions"));
const evaluateA = fire(formA, "submit"); // an evaluation for report A, left pending
await settle();
const evaluateRequestA = pending("/interview/evaluate")[0];

const pdfA = fire(byId("pdf-button"), "click"); // a PDF download for report A, left pending
await settle();
const pdfRequestA = pending("/pdf")[0];

// --- 2. Report B becomes active before A's requests complete ----------------------
await fire(byId("new-analysis-button"), "click");
await analyse(reportB);
out.aborted = {
  generate: generateRequestA.signal.aborted,
  evaluate: evaluateRequestA.signal.aborted,
};

// --- 3. A's requests complete late -------------------------------------------------
await release(generateRequestA, json(200, {
  levels: ["beginner"],
  count_per_level: 1,
  questions: { beginner: [oldQuestion], intermediate: [], advanced: [] },
  security_flags: [],
}));
await generateA;
// 4. A's questions never appear under B.
out.staleQuestionsShown = generatedText().includes(oldQuestion.question);
out.afterLateGenerate = stateOf();

await release(evaluateRequestA, notFound()); // 6. late 404s from report A
await evaluateA;
await release(pdfRequestA, notFound());
await pdfA;
out.afterLate404 = stateOf();

// --- 5. Every evaluation for B pairs B's ID with B's own questions -----------------
const formB = await openAnswerForm(byId("stored-questions"));
const evaluateB = fire(formB, "submit");
await settle();
const evaluateRequestB = pending("/interview/evaluate")[0];
await release(evaluateRequestB, json(200, input.evaluation), "evaluate B stored");
await evaluateB;
out.evaluationFocus = {
  tag: document.activeElement?.tagName ?? null,
  text: document.activeElement?.textContent ?? null,
  tabindex: document.activeElement?.getAttribute("tabindex") ?? null,
};

const generateB = fire(byId("generate-form"), "submit");
await settle();
const generateRequestB = pending("/interview/questions")[0];
await release(generateRequestB, json(200, {
  levels: ["beginner"],
  count_per_level: 1,
  questions: { beginner: [input.newQuestion], intermediate: [], advanced: [] },
  security_flags: [],
}));
await generateB;
out.announcedGenerated = byId("app-status").textContent;
const formGenerated = await openAnswerForm(byId("generated-questions"));
const evaluateGenerated = fire(formGenerated, "submit");
await settle();
const evaluateRequestGenerated = pending("/interview/evaluate")[0];
await release(evaluateRequestGenerated, json(200, input.evaluation), "evaluate B generated");
await evaluateGenerated;
out.evaluations = calls
  .filter((call) => call.url.includes("/interview/evaluate"))
  .map((call) => call.body);

// --- L-1. B expires during Generate: the button stays disabled ---------------------
const generateExpired = fire(byId("generate-form"), "submit");
await settle();
await release(pending("/interview/questions")[0], notFound(), "generate B expiring");
await generateExpired;
out.afterExpiry = stateOf();
out.requestsAfterExpiry = calls.length;
await fire(byId("generate-form"), "submit"); // a click on the disabled button does nothing
await settle();
out.requestsAfterDisabledClick = calls.length;

out.missing = missing;
emit(out);
