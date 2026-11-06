import "./fake_dom.mjs";

// Expiry versus Retry-After countdowns and already-open answer forms (audit N-1,
// N-2), with the real app.js on a fake page built from index.html's IDs. This is
// a simulated DOM, not a browser.
const input = JSON.parse(process.env.FRONTEND_INPUT);
for (const [id, tag] of input.elements) register(id, tag);
const byId = (id) => document.getElementById(id);
byId("generate-form").querySelector = () => ({ checked: true });
byId("count-per-level").value = "1";

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const json = (status, body, headers = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
const error = (status, code, headers = {}) =>
  json(status, { error: { code, message: code, request_id: `r-${code}` } }, headers);
const limited = () => error(429, "rate_limited", { "Retry-After": "1" });

// Each request is answered from the queue for its path, in order.
const replies = { "/analyze": [], "/pdf": [], "/interview/questions": [], "/interview/evaluate": [] };
const sent = [];
globalThis.fetch = async (url) => {
  sent.push(url);
  const path = Object.keys(replies).find((key) => url.includes(key));
  return replies[path].shift();
};
const controls = (container) => {
  const found = [];
  (function walk(node) {
    for (const child of node.childNodes ?? []) {
      if (child.tagName === "BUTTON") found.push({ text: child.textContent, disabled: child.disabled });
      walk(child);
    }
  })(container);
  return found;
};

const out = {};

// --- The countdown helper on its own -------------------------------------------------
{
  const { ApiError } = await import("./js/api.js");
  const { coolDown } = await import("./js/interview.js");
  const button = document.createElement("button");
  button.textContent = "Generate questions";
  coolDown(button, new ApiError({ status: 429, retryAfter: 30 }), () => false);
  out.unavailableFromTheStart = { text: button.textContent, disabled: button.disabled };
  const free = document.createElement("button");
  free.textContent = "Generate questions";
  coolDown(free, new ApiError({ status: 429, retryAfter: 1 }), () => true);
  out.availableDuring = { text: free.textContent, disabled: free.disabled };
  await wait(1300);
  out.availableAfter = { text: free.textContent, disabled: free.disabled };
}

await import("./js/app.js");
byId("generate-button").textContent = "Generate questions";
byId("pdf-button").textContent = "Download PDF report";

// --- A report with countdowns running and an answer form open ---------------------
replies["/analyze"].push(json(200, input.report));
byId("resume-file").files = [new File(["%PDF-1.7"], "cv.pdf", { type: "application/pdf" })];
await fire(byId("resume-file"), "change");
await fire(byId("upload-form"), "submit");
await wait(80);

const stored = byId("stored-questions");
await fire(findTag(stored, "BUTTON"), "click"); // "Practise this question": open a form
const form = findTag(stored, "FORM");
findTag(form, "TEXTAREA").value = "A list is an ordered, mutable sequence.";

replies["/interview/evaluate"].push(limited()); // Evaluate: countdown starts
await fire(form, "submit");
replies["/pdf"].push(limited()); // PDF: countdown starts
await fire(byId("pdf-button"), "click");
out.duringCountdowns = {
  pdf: { text: byId("pdf-button").textContent, disabled: byId("pdf-button").disabled },
  controls: controls(stored),
};

// The report expires (Generate returns 404) while both countdowns are running.
replies["/interview/questions"].push(error(404, "report_not_found"));
await fire(byId("generate-form"), "submit");
out.rightAfterExpiry = {
  banner: !byId("expired-banner").hidden,
  generate: byId("generate-button").disabled,
  controls: controls(stored),
};

await wait(1300); // both Retry-After countdowns (1 s) have now ended
out.afterCountdowns = {
  pdf: { text: byId("pdf-button").textContent, disabled: byId("pdf-button").disabled },
  generate: byId("generate-button").disabled,
  controls: controls(stored),
};

// Nothing reaches the server any more: neither the PDF button nor the open form.
const before = sent.length;
await fire(byId("pdf-button"), "click");
await fire(form, "submit");
out.requestsAfterExpiry = sent.length - before;
emit(out);
