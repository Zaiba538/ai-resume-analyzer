import "./fake_dom.mjs";
const api = await import("./js/api.js");
const calls = [];
globalThis.fetch = async (url, init) => {
  const entry = { url, method: init.method, headers: init.headers, credentials: init.credentials };
  if (init.body instanceof FormData) {
    entry.form = [...init.body.entries()].map(([k, v]) => [k, typeof v === "string" ? v : `file:${v.name}`]);
  } else if (init.body !== undefined) {
    entry.json = JSON.parse(init.body);
  }
  calls.push(entry);
  return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
};
const file = new File(["%PDF"], "cv.pdf", { type: "application/pdf" });
await api.analyzeResume(file, "   ");
await api.analyzeResume(file, "Data engineer");
await api.generateQuestions("rid", ["beginner", "advanced"], 2);
await api.evaluateAnswer("rid", { questionId: "beginner-1" }, "My answer");
await api.evaluateAnswer("rid", { question: { id: "advanced-2", question: "Q?", topic: "T", rationale: "R", extra: "x" } }, "A");
emit(calls);
