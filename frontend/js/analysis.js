/**
 * Running an analysis: one request to POST /api/v1/analyze.
 *
 * The backend has no progress-streaming contract, so the progress screen shows
 * only the elapsed time, never an estimated percentage. Only one analysis runs at
 * a time; cancelling aborts the browser request (the server may still finish the
 * work, but its result is discarded).
 */

import { analyzeResume } from "./api.js";
import { announce, formatDuration } from "./dom.js";

export function createAnalysisRunner() {
  const elapsed = document.getElementById("progress-elapsed");
  const cancelButton = document.getElementById("cancel-button");
  let controller = null;

  cancelButton.addEventListener("click", () => {
    if (controller) {
      controller.abort();
    }
  });

  return {
    get running() {
      return controller !== null;
    },

    /** Resolves with the AnalysisResponse; rejects with an ApiError. */
    async run(file, jobDescription) {
      if (controller) {
        throw new Error("An analysis is already running.");
      }
      controller = new AbortController();
      const started = Date.now();
      // The elapsed time is not a live region (it changes every second); the
      // start is announced once.
      announce("Analysis started. This can take a few minutes; the results open when it finishes.");
      elapsed.textContent = formatDuration(0);
      const timer = setInterval(() => {
        elapsed.textContent = formatDuration((Date.now() - started) / 1000);
      }, 1000);
      cancelButton.disabled = false;
      try {
        return await analyzeResume(file, jobDescription, { signal: controller.signal });
      } finally {
        clearInterval(timer);
        controller = null;
      }
    },
  };
}
