import "./fake_dom.mjs";

// The browser-side analysis deadline, without waiting 13 minutes: timers of a
// minute or more fire at once, and their requested delay is recorded.
const delays = [];
const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms, ...args) => {
  if (ms >= 60000) {
    delays.push(ms);
    return realSetTimeout(fn, 0, ...args);
  }
  return realSetTimeout(fn, ms, ...args);
};
globalThis.fetch = (url, init) =>
  new Promise((_, reject) => {
    init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
  });
const api = await import("./js/api.js");
let error = null;
try {
  await api.analyzeResume(new File(["%PDF-1.7"], "cv.pdf"), "");
} catch (caught) {
  error = caught;
}
emit({
  delays,
  code: error?.code ?? null,
  message: api.userMessage(error),
  genericTimeout: api.userMessage(new api.ApiError({ code: "client_timeout" })),
});
