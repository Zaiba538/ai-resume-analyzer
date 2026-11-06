import "./fake_dom.mjs";
const api = await import("./js/api.js");
const upload = await import("./js/upload.js");
const interview = await import("./js/interview.js");
const L = api.LIMITS;
const files = {
  none: null,
  pdf: { name: "Resume.PDF", size: 1000 },
  docx: { name: "cv.docx", size: 1000 },
  doc: { name: "cv.doc", size: 1000 },
  noExtension: { name: "resume", size: 1000 },
  empty: { name: "cv.pdf", size: 0 },
  atLimit: { name: "cv.pdf", size: L.maxUploadBytes },
  overLimit: { name: "cv.pdf", size: L.maxUploadBytes + 1 },
  sixMegabytes: { name: "cv.pdf", size: 6 * 1024 * 1024 },
  atAbsolute: { name: "cv.pdf", size: L.absoluteMaxUploadBytes },
  overAbsolute: { name: "cv.pdf", size: L.absoluteMaxUploadBytes + 1 },
};
const texts = {
  atDefault: "x".repeat(L.maxJobDescriptionChars),
  overDefault: "x".repeat(L.maxJobDescriptionChars + 1),
  atAbsolute: "x".repeat(L.absoluteMaxJobDescriptionChars),
  overAbsolute: "x".repeat(L.absoluteMaxJobDescriptionChars + 1),
};
const answers = {
  blank: "   ",
  atDefault: "x".repeat(L.maxAnswerChars),
  overDefault: "x".repeat(L.maxAnswerChars + 1000),
  atAbsolute: "x".repeat(L.absoluteMaxAnswerChars),
  overAbsolute: "x".repeat(L.absoluteMaxAnswerChars + 1),
};
const map = (object, fn) => Object.fromEntries(Object.entries(object).map(([k, v]) => [k, fn(v)]));
emit({
  limits: L,
  timeouts: api.TIMEOUTS_MS,
  budget: api.ANALYSIS_BUDGET_SECONDS,
  files: map(files, upload.validateFile),
  fileWarnings: map(files, upload.fileWarning),
  jd: map(texts, upload.validateJobDescription),
  jdWarnings: map(texts, upload.jobDescriptionWarning),
  answers: map(answers, interview.validateAnswer),
  answerWarnings: map(answers, interview.answerWarning),
  jdOk: upload.validateJobDescription("x".repeat(L.maxJobDescriptionChars)),
});
