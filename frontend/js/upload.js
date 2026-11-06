/**
 * Upload screen: file selection (picker or drag and drop), job description and
 * the browser-side checks that run before anything is sent.
 *
 * The backend's size and length limits are configurable and not published, so
 * the browser blocks only what no configuration can accept (wrong type, empty
 * file, above the absolute maxima). Exceeding a DEFAULT limit produces a
 * warning; the request is still sent and the server's own error is shown.
 */

import { LIMITS } from "./api.js";
import { announce, formatBytes } from "./dom.js";

/** The lower-case extension of a file name, including the dot, or "". */
export function fileExtension(name) {
  const index = typeof name === "string" ? name.lastIndexOf(".") : -1;
  return index > 0 ? name.slice(index).toLowerCase() : "";
}

/** A message describing why the file cannot be uploaded, or null when it can. */
export function validateFile(file) {
  if (!file) {
    return "Choose a resume file to upload.";
  }
  if (!LIMITS.acceptedExtensions.includes(fileExtension(file.name))) {
    return "Only PDF and DOCX files are supported.";
  }
  if (file.size === 0) {
    return "The file is empty. Choose a different file.";
  }
  if (file.size > LIMITS.absoluteMaxUploadBytes) {
    return `The file is larger than ${formatBytes(LIMITS.absoluteMaxUploadBytes)}, which no server configuration accepts. Upload a smaller file.`;
  }
  return null;
}

/** A non-blocking warning when the file exceeds the default size limit, or null. */
export function fileWarning(file) {
  if (file && file.size > LIMITS.maxUploadBytes && file.size <= LIMITS.absoluteMaxUploadBytes) {
    return `The file is larger than the default limit of ${formatBytes(LIMITS.maxUploadBytes)}; the server may reject it.`;
  }
  return null;
}

/** A message describing why the job description cannot be sent, or null. */
export function validateJobDescription(text) {
  if (typeof text === "string" && text.length > LIMITS.absoluteMaxJobDescriptionChars) {
    return `The job description is longer than ${LIMITS.absoluteMaxJobDescriptionChars.toLocaleString("en")} characters, which no server configuration accepts. Please shorten it.`;
  }
  return null;
}

/** A non-blocking warning when the job description exceeds the default limit, or null. */
export function jobDescriptionWarning(text) {
  if (typeof text === "string" && text.length > LIMITS.maxJobDescriptionChars) {
    return `Longer than the default limit of ${LIMITS.maxJobDescriptionChars.toLocaleString("en")} characters; the server may reject it.`;
  }
  return null;
}

/**
 * Wire up the upload form.
 *
 * @param {{onSubmit: (file: File, jobDescription: string) => void,
 *          onInvalid: (message: string) => void, onReset: () => void}} handlers
 */
export function initUpload({ onSubmit, onInvalid, onReset }) {
  const form = document.getElementById("upload-form");
  const input = document.getElementById("resume-file");
  const dropzone = document.getElementById("dropzone");
  const status = document.getElementById("file-status");
  const jobDescription = document.getElementById("job-description");
  const counter = document.getElementById("job-description-count");
  const resetButton = document.getElementById("reset-button");
  document.getElementById("file-limit").textContent = formatBytes(LIMITS.maxUploadBytes);

  let selected = null;
  let warnedLength = false;

  const describeSelection = () => {
    if (!selected) {
      status.textContent = "No file selected.";
      dropzone.classList.remove("dropzone--selected", "dropzone--invalid");
      return;
    }
    const problem = validateFile(selected) || fileWarning(selected);
    status.textContent = `Selected: ${selected.name} (${formatBytes(selected.size)})${problem ? ` - ${problem}` : ""}`;
    dropzone.classList.toggle("dropzone--selected", !problem);
    dropzone.classList.toggle("dropzone--invalid", Boolean(problem));
  };

  // The counter is not a live region: it is read with the field (aria-describedby).
  // Crossing the default limit is announced once, not on every keystroke.
  const updateCounter = () => {
    const text = jobDescription.value;
    const warning = validateJobDescription(text) || jobDescriptionWarning(text);
    counter.textContent = `${text.length.toLocaleString("en")} characters${warning ? `. ${warning}` : ""}`;
    counter.classList.toggle("counter--over", Boolean(warning));
    if (warning && !warnedLength) {
      announce(`Job description: ${warning}`);
    }
    warnedLength = Boolean(warning);
  };

  input.addEventListener("change", () => {
    selected = input.files && input.files.length > 0 ? input.files[0] : null;
    describeSelection();
  });

  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("dropzone--active");
    });
  }
  for (const type of ["dragleave", "dragend"]) {
    dropzone.addEventListener(type, () => dropzone.classList.remove("dropzone--active"));
  }
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dropzone--active");
    const files = event.dataTransfer ? event.dataTransfer.files : null;
    if (!files || files.length === 0) {
      return;
    }
    if (files.length > 1) {
      onInvalid("Upload exactly one file.");
      return;
    }
    selected = files[0];
    input.value = ""; // the dropped file is kept separately from the picker
    describeSelection();
  });

  jobDescription.addEventListener("input", updateCounter);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const problem = validateFile(selected) || validateJobDescription(jobDescription.value);
    if (problem) {
      onInvalid(problem);
      return;
    }
    onSubmit(selected, jobDescription.value);
  });

  resetButton.addEventListener("click", () => {
    form.reset();
    selected = null;
    describeSelection();
    updateCounter();
    onReset();
  });

  describeSelection();
  updateCounter();
}
