// Minimal fake DOM for the frontend tests: throws on every markup-parsing sink.
class FakeNode {
  constructor() { this.childNodes = []; }
  append(...nodes) {
    for (const node of nodes) this.childNodes.push(typeof node === "string" ? new FakeText(node) : node);
  }
  prepend(...nodes) {
    this.childNodes.unshift(...nodes.map((n) => (typeof n === "string" ? new FakeText(n) : n)));
  }
  replaceChildren(...nodes) { this.childNodes = []; this.append(...nodes); }
  get textContent() { return this.childNodes.map((child) => child.textContent).join(""); }
  set textContent(value) { this.childNodes = [new FakeText(value)]; }
}
class FakeText extends FakeNode {
  constructor(value) { super(); this.data = String(value); }
  get textContent() { return this.data; }
  set textContent(value) { this.data = String(value); }
}
const forbidden = (name) => () => { throw new Error(`${name} is not allowed`); };
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = String(tag).toUpperCase();
    this.attributes = {};
    this.className = "";
    this.hidden = false;
    this.value = "";
    this.listeners = {};
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  focus() { document.activeElement = this; }
  click() { this.clicks = (this.clicks ?? 0) + 1; }
  remove() {}
}
for (const name of ["innerHTML", "outerHTML"]) {
  Object.defineProperty(FakeElement.prototype, name, { get: forbidden(name), set: forbidden(name) });
}
FakeElement.prototype.insertAdjacentHTML = forbidden("insertAdjacentHTML");
const registry = new Map();
globalThis.document = {
  activeElement: null,
  body: new FakeElement("body"),
  createElement: (tag) => new FakeElement(tag),
  createTextNode: (text) => new FakeText(text),
  getElementById: (id) => registry.get(id) ?? null,
  write: forbidden("document.write"),
};
/** Register an element under an ID, as if it were in index.html. */
globalThis.register = (id, tag = "div") => {
  const element = new FakeElement(tag);
  element.attributes.id = id;
  registry.set(id, element);
  return element;
};
/** Call the element's listeners for an event type and wait for async handlers. */
globalThis.fire = async (element, type, event = {}) => {
  const handlers = element.listeners[type] ?? [];
  await Promise.all(handlers.map((handler) => handler({ preventDefault() {}, currentTarget: element, target: element, ...event })));
};
/** The first descendant with a tag name (depth first). */
globalThis.findTag = function findTag(node, tag) {
  for (const child of node.childNodes ?? []) {
    if (child.tagName === tag) return child;
    const found = findTag(child, tag);
    if (found) return found;
  }
  return null;
};
globalThis.serialize = function serialize(node) {
  if (node instanceof FakeText) return { text: node.data };
  return {
    tag: node.tagName,
    className: node.className,
    attrs: node.attributes,
    children: node.childNodes.map(serialize),
  };
};
globalThis.emit = (value) => console.log(JSON.stringify(value));
