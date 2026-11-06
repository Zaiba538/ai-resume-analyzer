/**
 * Safe DOM construction helpers.
 *
 * Every piece of text, including resume-derived data returned by the API, is
 * inserted as a text node (textContent or append(string)). There is deliberately
 * no helper that accepts markup, and no markup-parsing DOM sink is used anywhere.
 * Inline styles and event-handler attributes are rejected because the page's
 * Content-Security-Policy forbids them.
 */

const FORBIDDEN_ATTRIBUTE = /^(on|style$|srcdoc$|formaction$)/i;
const URL_ATTRIBUTES = new Set(["href", "src", "action", "xlink:href"]);

/**
 * Create an element.
 *
 * @param {string} tag
 * @param {{className?: string, text?: unknown, attrs?: Record<string, unknown>}} [options]
 * @param {Array<Node|string|null|undefined|false>} [children]
 */
export function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  const { className, text, attrs } = options;
  if (className) {
    node.className = className;
  }
  if (text !== undefined && text !== null) {
    node.textContent = String(text);
  }
  if (attrs) {
    for (const [name, value] of Object.entries(attrs)) {
      setAttribute(node, name, value);
    }
  }
  appendAll(node, children);
  return node;
}

/** Set one attribute, refusing event handlers, inline styles and script URLs. */
export function setAttribute(node, name, value) {
  if (FORBIDDEN_ATTRIBUTE.test(name)) {
    throw new Error(`Attribute not allowed: ${name}`);
  }
  if (value === false || value === null || value === undefined) {
    return;
  }
  const text = value === true ? "" : String(value);
  if (URL_ATTRIBUTES.has(name.toLowerCase()) && /^\s*(javascript|data|vbscript):/i.test(text)) {
    throw new Error(`URL scheme not allowed in ${name}`);
  }
  node.setAttribute(name, text);
}

/** Append children; strings become text nodes, empty values are skipped. */
export function appendAll(parent, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) {
      continue;
    }
    parent.append(child); // a string is appended as a text node, never parsed
  }
}

/** Remove every child of a node. */
export function clear(node) {
  node.replaceChildren();
}

/** Show or hide a node with the hidden attribute (no inline styles). */
export function setVisible(node, visible) {
  node.hidden = !visible;
}

/** A list of strings as a <ul> of text items, or an empty-state paragraph. */
export function textList(items, emptyText, className = "list") {
  if (!Array.isArray(items) || items.length === 0) {
    return el("p", { className: "empty", text: emptyText });
  }
  return el(
    "ul",
    { className },
    items.map((item) => el("li", { text: item })),
  );
}

/** A list of strings as chips, or an empty-state paragraph. */
export function chipList(items, emptyText, tone = "") {
  if (!Array.isArray(items) || items.length === 0) {
    return el("p", { className: "empty", text: emptyText });
  }
  const className = tone ? `chip chip--${tone}` : "chip";
  return el(
    "ul",
    { className: "chips" },
    items.map((item) => el("li", { className, text: item })),
  );
}

/**
 * Render an alert into a container, replacing its content.
 *
 * @param {Element} container
 * @param {{message: string, details?: string[], requestId?: string|null,
 *          tone?: "error"|"warning"|"info"|"success"}} alert
 */
export function renderAlert(container, { message, details = [], requestId = null, tone = "error" }) {
  const children = [el("p", { className: "alert__message", text: message })];
  if (details.length > 0) {
    children.push(textList(details, "", "alert__details"));
  }
  if (requestId) {
    children.push(el("p", { className: "alert__reference", text: `Reference: ${requestId}` }));
  }
  const role = tone === "error" || tone === "warning" ? "alert" : "status";
  container.replaceChildren(el("div", { className: `alert alert--${tone}`, attrs: { role } }, children));
}

let pendingAnnouncement = null;

/**
 * Announce a short status message to screen readers through the page's single
 * status region (#app-status). The region is emptied first and filled a moment
 * later, so that repeating the same message is announced again. Frequently
 * changing text (timers, character counters) is never announced this way.
 */
export function announce(message) {
  const region = document.getElementById("app-status");
  if (!region) {
    return;
  }
  if (pendingAnnouncement !== null) {
    clearTimeout(pendingAnnouncement);
  }
  region.textContent = "";
  pendingAnnouncement = setTimeout(() => {
    pendingAnnouncement = null;
    region.textContent = message;
  }, 50);
}

/**
 * Count down whole seconds, calling onTick(remaining) each second and onDone()
 * at zero. Returns a function that stops the countdown early.
 */
export function startCountdown(seconds, onTick, onDone) {
  let remaining = Math.max(0, Math.ceil(seconds));
  onTick(remaining);
  if (remaining === 0) {
    onDone();
    return () => {};
  }
  const timer = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(timer);
      onDone();
    } else {
      onTick(remaining);
    }
  }, 1000);
  return () => clearInterval(timer);
}

/** Format seconds as m:ss (elapsed time). */
export function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

/** Format a byte count for people (1 MB = 1024 * 1024 bytes, as the backend counts). */
export function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) {
    const value = bytes / (1024 * 1024);
    return `${Number.isInteger(value) ? value : value.toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${bytes} bytes`;
}
