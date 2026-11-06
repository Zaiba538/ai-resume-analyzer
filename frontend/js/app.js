/**
 * Application entry point: views, navigation, tabs, PDF download and the
 * expired-report state. The current report lives in memory only; nothing is
 * written to browser storage, because reports contain personal data.
 */

import { ApiError, fetchReportPdf, isExpiredReport, userMessage, validationDetails } from "./api.js";
import { createAnalysisRunner } from "./analysis.js";
import { clear, renderAlert, setVisible } from "./dom.js";
import { coolDown, initInterview } from "./interview.js";
import { expiryText, renderReport } from "./results.js";
import { initUpload } from "./upload.js";

const VIEWS = ["upload", "progress", "results"];

const state = {
  report: null,
  expired: false,
};

/** The interview panel (set in init), told about expiry so it can disable its controls. */
let interview = null;

function byId(id) {
  return document.getElementById(id);
}

function showView(name, { focus = true } = {}) {
  for (const view of VIEWS) {
    setVisible(byId(`view-${view}`), view === name);
  }
  const links = [
    [byId("nav-upload"), name === "upload"],
    [byId("nav-results"), name === "results"],
  ];
  for (const [link, active] of links) {
    if (active) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  }
  byId("nav-upload").disabled = name === "progress";
  byId("nav-results").disabled = name === "progress" || state.report === null;
  const heading = byId(`view-${name}`).querySelector("h1");
  if (focus && heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus(); // move screen-reader and keyboard focus to the new view
  }
}

function showApiError(container, error) {
  renderAlert(container, {
    message: userMessage(error),
    details: validationDetails(error),
    requestId: error instanceof ApiError ? error.requestId : null,
  });
}

// --- Tabs -------------------------------------------------------------------------------

function initTabs() {
  const tablist = byId("result-tabs");
  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));

  const select = (tab, focus = true) => {
    for (const other of tabs) {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      other.tabIndex = selected ? 0 : -1;
      setVisible(byId(other.getAttribute("aria-controls")), selected);
    }
    if (focus) {
      tab.focus();
    }
  };

  for (const tab of tabs) {
    tab.addEventListener("click", () => select(tab));
  }
  tablist.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(document.activeElement);
    if (index === -1) {
      return;
    }
    const keys = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1 };
    if (!(event.key in keys)) {
      return;
    }
    event.preventDefault();
    select(tabs[(keys[event.key] + tabs.length) % tabs.length]);
  });
  return { reset: () => select(tabs[0], false) };
}

// --- Expired reports ----------------------------------------------------------------------

function setExpired(expired) {
  state.expired = expired;
  setVisible(byId("expired-banner"), expired);
  byId("pdf-button").disabled = expired;
  byId("generate-button").disabled = expired;
  byId("panel-interview").classList.toggle("is-expired", expired);
  if (interview) {
    interview.setExpired(expired); // open answer forms and "Practise" toggles too
  }
}

// --- PDF ----------------------------------------------------------------------------------

async function downloadPdf(button) {
  const alert = byId("results-alert");
  if (!state.report || button.disabled) {
    return;
  }
  // The report this download belongs to: if another report becomes active while
  // waiting, the old report's outcome must not be shown under it or expire it.
  const reportId = state.report.report_id;
  const isCurrent = () => state.report !== null && state.report.report_id === reportId;
  clear(alert);
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Preparing PDF...";
  let failed = null;
  try {
    const { blob, filename } = await fetchReportPdf(reportId);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    if (isCurrent()) {
      renderAlert(alert, { message: "The PDF report has been downloaded.", tone: "success" });
    }
  } catch (error) {
    if (!isCurrent()) {
      return; // an old report's failure, including its 404, is ignored
    }
    failed = error;
    showApiError(alert, error);
    if (isExpiredReport(error)) {
      setExpired(true);
    }
  } finally {
    button.textContent = label;
    button.disabled = state.expired;
    if (failed && !state.expired) {
      // The countdown must not re-enable the button if the report expires meanwhile.
      coolDown(button, failed, () => state.report !== null && !state.expired);
    }
  }
}

// --- Start-up -----------------------------------------------------------------------------

function init() {
  const runner = createAnalysisRunner();
  const tabs = initTabs();
  const uploadAlert = byId("upload-alert");
  const analyzeButton = byId("analyze-button");
  const panels = {
    overview: byId("panel-overview"),
    profile: byId("panel-profile"),
    match: byId("panel-match"),
    review: byId("panel-review"),
  };
  interview = initInterview({
    getReportId: () => (state.report && !state.expired ? state.report.report_id : null),
    // Only the active report can expire; interview.js already drops stale requests.
    onExpired: (error, reportId) => {
      if (!reportId || (state.report && state.report.report_id === reportId)) {
        setExpired(true);
      }
    },
  });

  const showReport = (report) => {
    state.report = report;
    setExpired(false);
    clear(byId("results-alert"));
    byId("results-expiry").textContent = expiryText(report);
    renderReport(report, panels);
    interview.showReport(report);
    tabs.reset();
    showView("results");
  };

  initUpload({
    onInvalid: (message) => renderAlert(uploadAlert, { message }),
    onReset: () => clear(uploadAlert),
    onSubmit: async (file, jobDescription) => {
      if (runner.running || analyzeButton.disabled) {
        return; // no duplicate submissions
      }
      clear(uploadAlert);
      analyzeButton.disabled = true;
      showView("progress");
      let failed = null;
      try {
        const report = await runner.run(file, jobDescription);
        showReport(report);
      } catch (error) {
        failed = error;
        showView("upload");
        showApiError(uploadAlert, error);
      } finally {
        analyzeButton.disabled = false;
        if (failed) {
          coolDown(analyzeButton, failed);
        }
      }
    },
  });

  byId("nav-upload").addEventListener("click", () => showView("upload"));
  byId("nav-results").addEventListener("click", () => {
    if (state.report) {
      showView("results");
    }
  });
  byId("new-analysis-button").addEventListener("click", () => showView("upload"));
  byId("expired-restart-button").addEventListener("click", () => showView("upload"));
  byId("pdf-button").addEventListener("click", (event) => downloadPdf(event.currentTarget));

  showView("upload", { focus: false });
}

init();
