/**
 * Client for the Resume Intelligence API.
 *
 * Only existing endpoints are used, with root-relative (same-origin) URLs. Error
 * responses normally use the backend envelope
 * {"error": {"code", "message", "request_id", "details"?}}, but a proxy or a
 * network failure can produce anything, so every failure becomes an ApiError and
 * nothing assumes a JSON body.
 */

export const ENDPOINTS = Object.freeze({
  analyze: Object.freeze({ method: "POST", path: "/api/v1/analyze" }),
  interviewQuestions: Object.freeze({ method: "POST", path: "/api/v1/interview/questions" }),
  interviewEvaluate: Object.freeze({ method: "POST", path: "/api/v1/interview/evaluate" }),
  reportPdf: Object.freeze({ method: "GET", path: "/api/v1/report/{report_id}/pdf" }),
});

/*
 * Size and length limits. The API does not publish its configured limits, and an
 * operator can raise or lower them (backend/core/config.py), so the server stays
 * authoritative and its errors are always shown:
 *
 * - maxUploadBytes, maxJobDescriptionChars and maxAnswerChars are the backend's
 *   DEFAULT limits. Exceeding one only produces a warning; it never blocks a
 *   request, because the deployed server may allow more.
 * - absoluteMax* are the highest values any configuration can set (the bounds of
 *   the settings fields). Nothing above them can ever be accepted, so the browser
 *   refuses it instead of uploading it.
 */
export const LIMITS = Object.freeze({
  maxUploadBytes: 5 * 1024 * 1024,
  maxJobDescriptionChars: 20000,
  maxAnswerChars: 5000,
  absoluteMaxUploadBytes: 20 * 1024 * 1024,
  absoluteMaxJobDescriptionChars: 100000,
  absoluteMaxAnswerChars: 20000,
  acceptedExtensions: Object.freeze([".pdf", ".docx"]),
  countPerLevel: Object.freeze({ min: 1, max: 5, default: 3 }),
});

/*
 * The analysis budget. The server's analysis deadline starts only after the
 * upload has been received (up to UPLOAD_READ_TIMEOUT_SECONDS) and the document
 * has been parsed, so the browser waits for both, plus a margin for parsing,
 * queueing and the response itself.
 */
export const ANALYSIS_BUDGET_SECONDS = Object.freeze({
  upload: 60, // backend/api/uploads.py UPLOAD_READ_TIMEOUT_SECONDS
  analysis: 600, // ANALYSIS_DEADLINE_SECONDS default
  margin: 120,
});

/*
 * Browser-side deadlines, each longer than the server's default deadline for the
 * same work, so that the server's own timeout error normally arrives first.
 */
export const TIMEOUTS_MS = Object.freeze({
  analyze:
    (ANALYSIS_BUDGET_SECONDS.upload + ANALYSIS_BUDGET_SECONDS.analysis + ANALYSIS_BUDGET_SECONDS.margin) *
    1000, // 780 s
  interviewQuestions: 150 * 1000, // INTERVIEW_DEADLINE_SECONDS default: 120
  interviewEvaluate: 150 * 1000, // EVALUATION_DEADLINE_SECONDS default: 120
  reportPdf: 60 * 1000, // PDF render deadline: 15
});

export const PDF_FILENAME = "resume-analysis-report.pdf";

/** A user-facing message for every error code the backend can return. */
export const ERROR_MESSAGES = Object.freeze({
  // Reports and questions
  report_not_found:
    "This report has expired or is no longer available. Reports are kept for a limited time and are lost when the server restarts. Please run the analysis again.",
  question_not_found: "That question is not part of this report. Please choose another question.",
  // Uploads
  upload_timeout: "The upload took too long and was stopped. Check your connection and try again.",
  file_too_large: "The file is too large. Upload a smaller PDF or DOCX file.",
  payload_too_large: "The request is too large. Upload a smaller file or shorten the job description.",
  unsupported_file_type: "Only PDF and DOCX files are supported.",
  file_content_mismatch: "The file's content does not match its extension. Upload a genuine PDF or DOCX file.",
  multipart_required: "The file could not be sent. Please choose the file again and retry.",
  malformed_multipart: "The upload was damaged in transit. Please try again.",
  missing_file: "Choose a resume file to upload.",
  multiple_files: "Upload exactly one file.",
  empty_file: "The file is empty. Choose a different file.",
  invalid_upload: "The file could not be accepted. Check that it is a valid PDF or DOCX file.",
  invalid_field: "A form field has an invalid value. Please check your input.",
  unexpected_field: "The request contained an unexpected field. Please reload the page and try again.",
  encrypted_document: "Password-protected or encrypted documents are not supported.",
  unreadable_document: "The document could not be read. It may be damaged; try saving it again.",
  unsafe_archive: "The DOCX file's structure is invalid or exceeds safety limits.",
  too_many_pages: "The document has too many pages. Upload a shorter resume.",
  document_too_long: "The document contains too much text. Upload a shorter resume.",
  no_extractable_text:
    "No text could be extracted. Scanned or image-only documents are not supported; upload a text-based PDF or DOCX.",
  unsupported_language: "Only English content is supported.",
  // Text inputs
  text_too_long: "The text is too long. Please shorten it.",
  job_description_too_long: "The job description is too long. Please shorten it.",
  empty_job_description: "The job description is empty.",
  missing_job_description: "A job description is required for this action.",
  empty_resume_text: "The resume text is empty.",
  invalid_agent_input: "The input is empty or invalid. Please check the document and try again.",
  empty_answer: "Write an answer before submitting it for evaluation.",
  answer_too_long: "The answer is too long. Please shorten it.",
  validation_error: "Some of the information sent was not valid. Please check your input.",
  unsupported_media_type: "The request format was not accepted. Please reload the page and try again.",
  // Limits
  rate_limited: "Too many requests. Please wait before trying again.",
  analysis_capacity_exceeded: "The server is busy with other analyses. Please try again shortly.",
  document_processing_busy: "The server is busy processing other documents. Please try again shortly.",
  // Upstream AI service
  upstream_error: "The AI service returned an error. Please try again.",
  upstream_auth_failed: "The AI service is not available because of a server configuration problem.",
  invalid_llm_output: "The AI service returned an invalid response. Please try again.",
  upstream_rate_limited: "The AI service is busy. Please try again shortly.",
  upstream_timeout: "The AI service did not respond in time. Please try again.",
  llm_not_configured: "The AI service is not configured on this server.",
  embedding_model_unavailable: "The similarity model is not available yet. Please try again later.",
  scoring_failed: "A similarity metric could not be computed. Please try again.",
  report_store_full: "The service is at capacity. Please try again later.",
  service_unavailable: "The service is temporarily unavailable. Please try again later.",
  // Deadlines
  analysis_timeout: "The analysis did not finish in time. Please try again.",
  document_processing_timeout:
    "The document took too long to process. Please try a simpler or smaller file, or try again later.",
  interview_timeout: "Generating interview questions took too long. Please try again.",
  evaluation_timeout: "Evaluating the answer took too long. Please try again.",
  // PDF report
  pdf_page_limit_exceeded: "The PDF report would be too long to generate.",
  pdf_render_timeout: "Generating the PDF took too long. Please try again.",
  pdf_renderer_unavailable: "PDF reports are not available on this server at the moment.",
  pdf_render_error: "The PDF report could not be generated. Please try again.",
  // Generic statuses
  bad_request: "The request could not be processed. Please try again.",
  unauthorized: "You are not allowed to do this.",
  forbidden: "You are not allowed to do this.",
  not_found: "The requested resource was not found.",
  method_not_allowed: "This action is not supported.",
  client_error: "The request could not be processed. Please check your input.",
  internal_error: "An unexpected server error occurred. Please try again.",
  server_error: "An unexpected server error occurred. Please try again.",
  // Raised in the browser, never by the server
  network_error: "The server could not be reached. Check your connection and try again.",
  client_timeout:
    "The browser stopped waiting for the server. The server may still finish processing the request. Please try again.",
  analysis_client_timeout:
    "The browser stopped waiting after 13 minutes. The server may still finish the analysis, but its result cannot be shown on this page. Please run the analysis again.",
  cancelled: "The request was cancelled.",
  invalid_response: "The server returned an unexpected response. Please try again.",
});

/** Fallbacks by HTTP status for responses without a recognised error code. */
export const STATUS_MESSAGES = Object.freeze({
  400: "The request could not be processed. Please try again.",
  404: "The requested resource was not found.",
  408: "The request took too long. Please try again.",
  413: "The request is too large. Upload a smaller file.",
  415: "The request format was not accepted.",
  422: "Some of the information sent was not valid. Please check your input.",
  429: "Too many requests. Please wait before trying again.",
  500: "An unexpected server error occurred. Please try again.",
  502: "The service is temporarily unavailable. Please try again.",
  503: "The service is temporarily unavailable. Please try again later.",
  504: "The server did not respond in time. Please try again.",
});

const GENERIC_MESSAGE = "Something went wrong. Please try again.";
const MAX_DETAILS = 5;

export class ApiError extends Error {
  constructor({
    status = 0,
    code = null,
    serverMessage = null,
    requestId = null,
    retryAfter = null,
    details = null,
  } = {}) {
    super(code || (status ? `HTTP ${status}` : "request failed"));
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.serverMessage = serverMessage;
    this.requestId = requestId;
    this.retryAfter = retryAfter;
    this.details = details;
  }
}

/** Seconds to wait from a Retry-After header (delta-seconds or HTTP date), or null. */
export function parseRetryAfter(value, now = Date.now()) {
  if (value === null || value === undefined) {
    return null;
  }
  const text = String(value).trim();
  if (/^\d+$/.test(text)) {
    const seconds = Number(text);
    return Number.isSafeInteger(seconds) ? seconds : null;
  }
  const date = Date.parse(text);
  if (Number.isNaN(date)) {
    return null;
  }
  return Math.max(0, Math.ceil((date - now) / 1000));
}

/** "45 seconds", "1 minute", "3 minutes". */
export function formatWait(seconds) {
  if (seconds < 60) {
    return seconds === 1 ? "1 second" : `${seconds} seconds`;
  }
  const minutes = Math.ceil(seconds / 60);
  return minutes === 1 ? "1 minute" : `${minutes} minutes`;
}

/** The message to show for any error thrown by this module (or anything else). */
export function userMessage(error) {
  if (!(error instanceof ApiError)) {
    return GENERIC_MESSAGE;
  }
  let message = null;
  if (error.code && Object.hasOwn(ERROR_MESSAGES, error.code)) {
    message = ERROR_MESSAGES[error.code];
  } else if (error.code && typeof error.serverMessage === "string" && error.serverMessage) {
    message = error.serverMessage; // a newer backend code: its message is fixed, safe text
  } else if (Object.hasOwn(STATUS_MESSAGES, error.status)) {
    message = STATUS_MESSAGES[error.status];
  } else {
    message = GENERIC_MESSAGE;
  }
  if (Number.isInteger(error.retryAfter) && error.retryAfter > 0) {
    message += ` You can try again in ${formatWait(error.retryAfter)}.`;
  }
  return message;
}

/** Validation problems as short "field: problem" lines (never the submitted values). */
export function validationDetails(error) {
  if (!(error instanceof ApiError) || !Array.isArray(error.details)) {
    return [];
  }
  return error.details.slice(0, MAX_DETAILS).flatMap((detail) => {
    if (!detail || typeof detail.msg !== "string") {
      return [];
    }
    const loc = Array.isArray(detail.loc)
      ? detail.loc.filter((part) => typeof part === "string" && part !== "body").join(".")
      : "";
    return [loc ? `${loc}: ${detail.msg}` : detail.msg];
  });
}

/** Whether the error means the report has expired or never existed. */
export function isExpiredReport(error) {
  return error instanceof ApiError && error.code === "report_not_found";
}

/** The URL of an endpoint, with path parameters percent-encoded. */
export function endpointUrl(name, params = {}) {
  const endpoint = ENDPOINTS[name];
  if (!endpoint) {
    throw new Error(`Unknown endpoint: ${name}`);
  }
  return endpoint.path.replace(/\{([a-z_]+)\}/g, (_, key) => {
    const value = params[key];
    if (typeof value !== "string" || value === "") {
      throw new Error(`Missing path parameter: ${key}`);
    }
    return encodeURIComponent(value);
  });
}

async function errorFromResponse(response) {
  const retryAfter = parseRetryAfter(response.headers.get("Retry-After"));
  let body = null;
  const type = response.headers.get("Content-Type") || "";
  if (type.toLowerCase().includes("application/json")) {
    try {
      body = await response.json();
    } catch {
      body = null;
    }
  }
  const envelope =
    body && typeof body === "object" && body.error && typeof body.error === "object" ? body.error : {};
  return new ApiError({
    status: response.status,
    code: typeof envelope.code === "string" ? envelope.code : null,
    serverMessage: typeof envelope.message === "string" ? envelope.message : null,
    requestId:
      typeof envelope.request_id === "string" ? envelope.request_id : response.headers.get("X-Request-ID"),
    retryAfter,
    details: Array.isArray(envelope.details) ? envelope.details : null,
  });
}

function filenameFrom(response) {
  const header = response.headers.get("Content-Disposition") || "";
  const match = /filename="([A-Za-z0-9._-]{1,100})"/.exec(header);
  return match ? match[1] : PDF_FILENAME;
}

/**
 * Call an endpoint.
 *
 * @param {keyof typeof ENDPOINTS} name
 * @param {{params?: Record<string,string>, json?: unknown, form?: FormData,
 *          signal?: AbortSignal, timeoutMs?: number, expect?: "json"|"pdf"}} [options]
 */
export async function request(name, options = {}) {
  const { params = {}, json, form, signal, timeoutMs, expect = "json" } = options;
  const endpoint = ENDPOINTS[name];
  const url = endpointUrl(name, params);
  const controller = new AbortController();
  let timedOut = false;
  const timer = timeoutMs
    ? setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs)
    : null;
  const onAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", onAbort, { once: true });
    }
  }
  const failure = () =>
    new ApiError({
      code: timedOut ? "client_timeout" : controller.signal.aborted ? "cancelled" : "network_error",
    });

  const init = {
    method: endpoint.method,
    credentials: "same-origin",
    cache: "no-store",
    signal: controller.signal,
    headers: { Accept: expect === "pdf" ? "application/pdf, application/json" : "application/json" },
  };
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (form !== undefined) {
    init.body = form; // the browser sets the multipart boundary
  }

  try {
    let response;
    try {
      response = await fetch(url, init);
    } catch {
      throw failure();
    }
    if (!response.ok) {
      throw await errorFromResponse(response);
    }
    try {
      if (expect === "pdf") {
        const type = (response.headers.get("Content-Type") || "").toLowerCase();
        if (!type.startsWith("application/pdf")) {
          throw new ApiError({ status: response.status, code: "invalid_response" });
        }
        return { blob: await response.blob(), filename: filenameFrom(response) };
      }
      return await response.json();
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      if (controller.signal.aborted) {
        throw failure();
      }
      throw new ApiError({ status: response.status, code: "invalid_response" });
    }
  } finally {
    if (timer !== null) {
      clearTimeout(timer);
    }
    if (signal) {
      signal.removeEventListener("abort", onAbort);
    }
  }
}

/** POST /api/v1/analyze: the job description is sent only when it is not blank. */
export async function analyzeResume(file, jobDescription, { signal } = {}) {
  const form = new FormData();
  form.append("file", file, file.name);
  if (typeof jobDescription === "string" && jobDescription.trim() !== "") {
    form.append("job_description", jobDescription);
  }
  try {
    return await request("analyze", { form, signal, timeoutMs: TIMEOUTS_MS.analyze });
  } catch (error) {
    if (error instanceof ApiError && error.code === "client_timeout") {
      // The server may still be working: say so, rather than "no response".
      throw new ApiError({ code: "analysis_client_timeout" });
    }
    throw error;
  }
}

/** POST /api/v1/interview/questions */
export function generateQuestions(reportId, levels, countPerLevel, { signal } = {}) {
  return request("interviewQuestions", {
    json: { report_id: reportId, levels, count_per_level: countPerLevel },
    signal,
    timeoutMs: TIMEOUTS_MS.interviewQuestions,
  });
}

/**
 * POST /api/v1/interview/evaluate with exactly one question source: a question
 * stored in the report ({questionId}) or a client-supplied question object
 * ({question: {id, question, topic, rationale}}), for example a generated one.
 */
export function evaluateAnswer(reportId, source, answer, { signal } = {}) {
  const body = { report_id: reportId, answer };
  if (source.question) {
    const { id, question, topic, rationale } = source.question;
    body.question = { id, question, topic, rationale };
  } else {
    body.question_id = source.questionId;
  }
  return request("interviewEvaluate", { json: body, signal, timeoutMs: TIMEOUTS_MS.interviewEvaluate });
}

/** GET /api/v1/report/{report_id}/pdf as a Blob. */
export function fetchReportPdf(reportId) {
  return request("reportPdf", {
    params: { report_id: reportId },
    expect: "pdf",
    timeoutMs: TIMEOUTS_MS.reportPdf,
  });
}
