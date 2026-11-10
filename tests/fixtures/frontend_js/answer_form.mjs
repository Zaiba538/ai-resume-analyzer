import "./fake_dom.mjs";

// The answer form: no maxlength, a counter that is not a live region, one
// announcement when the default limit is crossed, and the backend left to decide
// on answers above the default limit.
const { LIMITS } = await import("./js/api.js");
const { renderQuestionList } = await import("./js/interview.js");
const status = register("app-status", "p");
let text = "";
const announcements = [];
Object.defineProperty(status, "textContent", {
  get: () => text,
  set: (value) => {
    text = String(value);
    if (text) announcements.push(text);
  },
});
const settle = () => new Promise((resolve) => setTimeout(resolve, 80));
const sent = [];
const container = document.createElement("div");
const question = { id: "beginner-1", question: "What is a list?", topic: "Python", rationale: "" };
renderQuestionList(container, { beginner: [question], intermediate: [], advanced: [] }, {
  evaluate: async (_, answer) => {
    sent.push(answer.length);
    throw new Error("stub");
  },
  onExpired: () => {},
  emptyText: "none",
});
await fire(findTag(container, "BUTTON"), "click");
const form = findTag(container, "FORM");
const textarea = findTag(form, "TEXTAREA");
const spans = [];
(function collect(node) {
  for (const child of node.childNodes ?? []) {
    if (child.tagName === "SPAN") spans.push(child);
    collect(child);
  }
})(form);
const counter = spans.find((span) => span.className === "counter");
const result = form.childNodes.find((child) => child.className === "evaluation-slot");

const out = {
  textareaAttrs: textarea.attributes,
  counterAttrs: counter.attributes,
  resultAttrs: result.attributes,
};
for (const length of [10, LIMITS.maxAnswerChars + 1, LIMITS.maxAnswerChars + 2, LIMITS.maxAnswerChars + 3]) {
  textarea.value = "x".repeat(length);
  await fire(textarea, "input");
  await settle(); // past the announcement delay: every keystroke could be announced
}
out.counterText = counter.textContent;
out.announcements = announcements;

textarea.value = "x".repeat(LIMITS.maxAnswerChars + 1000); // above the default: still sent
await fire(form, "submit");
textarea.value = "x".repeat(LIMITS.absoluteMaxAnswerChars + 1); // above every configuration
await fire(form, "submit");
out.sent = sent;
out.formText = form.textContent;
emit(out);
