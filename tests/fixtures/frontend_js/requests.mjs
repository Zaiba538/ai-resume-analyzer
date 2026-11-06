import "./fake_dom.mjs";
const api = await import("./js/api.js");
const { ApiError, request, userMessage, isExpiredReport, validationDetails } = api;

const json = (status, body, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
const envelope = (code, extra = {}) => ({ error: { code, message: "server text", request_id: "req-1", ...extra } });
const hanging = (_url, init) => new Promise((_, reject) => {
  init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
});

async function attempt(fetchImpl, options = {}, name = "interviewQuestions") {
  globalThis.fetch = fetchImpl;
  try {
    const value = await request(name, { json: {}, ...options });
    return { ok: true, value: value instanceof Object && value.blob ? { filename: value.filename, size: value.blob.size } : value };
  } catch (error) {
    return {
      ok: false,
      isApiError: error instanceof ApiError,
      status: error.status,
      code: error.code,
      requestId: error.requestId,
      retryAfter: error.retryAfter,
      message: userMessage(error),
      expired: isExpiredReport(error),
      details: validationDetails(error),
    };
  }
}

const results = {};
results.capacity = await attempt(async () => json(429, envelope("analysis_capacity_exceeded"), { "Retry-After": "10" }));
results.rateLimited = await attempt(async () => json(429, envelope("rate_limited"), { "Retry-After": "75" }));
results.expired = await attempt(async () => json(404, envelope("report_not_found")));
results.validation = await attempt(async () => json(422, envelope("validation_error", {
  details: [{ loc: ["body", "answer"], type: "string_too_long", msg: "String should have at most 10000 characters" }],
})));
results.proxyHtml = await attempt(async () => new Response("<html>Bad gateway</html>", {
  status: 502, headers: { "Content-Type": "text/html", "X-Request-ID": "edge-7" },
}));
results.unknownCode = await attempt(async () => json(503, envelope("brand_new_code")));
results.noEnvelope = await attempt(async () => json(504, { detail: "x" }));
results.network = await attempt(async () => { throw new TypeError("Failed to fetch"); });
results.timeout = await attempt(hanging, { timeoutMs: 20 });
const controller = new AbortController();
setTimeout(() => controller.abort(), 20);
results.cancelled = await attempt(hanging, { signal: controller.signal });
results.badJson = await attempt(async () => new Response("not json", { status: 200, headers: { "Content-Type": "application/json" } }));
results.pdf = await attempt(async () => new Response(new Blob(["%PDF-1.7"]), {
  status: 200, headers: { "Content-Type": "application/pdf", "Content-Disposition": 'attachment; filename="resume-analysis-report.pdf"' },
}), { json: undefined, params: { report_id: "abc" }, expect: "pdf" }, "reportPdf");
results.pdfWrongType = await attempt(async () => json(200, {}), { json: undefined, params: { report_id: "abc" }, expect: "pdf" }, "reportPdf");
results.success = await attempt(async () => json(200, { levels: ["beginner"] }));
results.retryAfterDate = api.parseRetryAfter(new Date(Date.now() + 30000).toUTCString());
results.retryAfterBad = api.parseRetryAfter("soon");
results.encodedUrl = api.endpointUrl("reportPdf", { report_id: "a/b?c" });
results.messages = Object.fromEntries(
  JSON.parse(process.env.FRONTEND_INPUT).backend_codes.map((code) => [code, userMessage(new ApiError({ status: 400, code }))]),
);
results.errorMessages = api.ERROR_MESSAGES;
emit(results);
