import "./fake_dom.mjs";
const { renderReport, atsStatusText } = await import("./js/results.js");
const input = JSON.parse(process.env.FRONTEND_INPUT);
const out = {};
for (const [name, report] of Object.entries(input)) {
  const panels = Object.fromEntries(["overview", "profile", "match", "review"].map((p) => [p, document.createElement("div")]));
  renderReport(report, panels);
  out[name] = Object.fromEntries(Object.entries(panels).map(([p, node]) => [p, node.textContent]));
}
out.atsOther = atsStatusText({ status: "calibrated", score: 88 });
emit(out);
