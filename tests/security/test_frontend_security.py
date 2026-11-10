"""M11 Phase 1 security: the frontend under the unchanged Content-Security-Policy.

Static checks (Python only): no inline script or style, no event-handler
attributes, no markup-parsing DOM sinks, no dynamic code, no browser storage,
no secrets, labelled form controls.

Behavioural checks (Node.js, skipped without it): hostile strings in an API
response are rendered as text by the real rendering code, under a fake DOM that
throws on any markup sink.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tests.fixtures.frontend import (
    FRONTEND_DIR,
    JS_DIR,
    requires_node,
    run_frontend_script,
    sample_report,
)

HTML = FRONTEND_DIR / "index.html"
PAYLOAD = '<img src=x onerror="alert(1)"><script>alert(2)</script>javascript:alert(3)'
STRUCTURAL_KEYS = {
    "report_id",
    "created_at",
    "expires_at",
    "verdict",
    "severity",
    "agent",
    "type",
    "source",
    "status",
    "id",
    "calibrated",
}
SAFE_TAGS = {
    "SECTION", "H2", "H3", "H4", "P", "DIV", "UL", "OL", "LI", "SPAN", "TABLE", "THEAD",
    "TBODY", "TR", "TH", "TD", "DL", "DT", "DD", "METER", "BUTTON", "FORM", "LABEL", "TEXTAREA",
}  # fmt: skip
SINKS = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "DOMParser",
    "createContextualFragment",
)


class _Markup(HTMLParser):
    """Collects everything that could carry inline code or style."""

    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self.style_elements = 0
        self.attributes: list[tuple[str, str, str | None]] = []
        self.controls: list[tuple[str, dict[str, str | None], bool]] = []
        self.label_for: set[str] = set()
        self._label_depth = 0
        self._in_script: dict[str, str | None] | None = None
        self._script_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for name, value in attrs:
            self.attributes.append((tag, name, value))
        if tag == "script":
            self._in_script, self._script_text = values, ""
        elif tag == "style":
            self.style_elements += 1
        elif tag == "label":
            self._label_depth += 1
            if values.get("for"):
                self.label_for.add(values["for"] or "")
        elif tag in {"input", "textarea", "select"}:
            self.controls.append((tag, values, self._label_depth > 0))

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script is not None:
            self.scripts.append((self._in_script, self._script_text))
            self._in_script = None
        elif tag == "label":
            self._label_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_script is not None:
            self._script_text += data


def _markup() -> _Markup:
    parser = _Markup()
    parser.feed(HTML.read_text(encoding="utf-8"))
    return parser


def _js_sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(JS_DIR.glob("*.js"))}


def _all_sources() -> dict[str, str]:
    return {
        path.relative_to(FRONTEND_DIR).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(FRONTEND_DIR.rglob("*"))
        if path.is_file()
    }


# --- inline code and style ----------------------------------------------------------------


def test_scripts_are_external_same_origin_modules_only() -> None:
    scripts = _markup().scripts
    assert scripts == [({"type": "module", "src": "js/app.js"}, "")]


def test_no_style_elements_or_style_attributes() -> None:
    parser = _markup()
    assert parser.style_elements == 0
    assert not [a for a in parser.attributes if a[1] == "style"]


def test_no_event_handler_attributes_or_script_urls_in_html() -> None:
    parser = _markup()
    assert not [a for a in parser.attributes if a[1].lower().startswith("on")]
    for _, name, value in parser.attributes:
        if name in {"href", "src", "action", "formaction"} and value:
            assert not re.match(r"\s*(javascript|vbscript):", value, flags=re.IGNORECASE)
    links = [a for a in parser.attributes if a[0] == "link" and a[1] == "href"]
    assert {value for _, _, value in links} == {"data:,", "css/styles.css"}


def test_javascript_uses_no_markup_sinks_or_dynamic_code() -> None:
    for name, source in _js_sources().items():
        for sink in SINKS:
            assert sink not in source, f"{name} uses {sink}"
        assert not re.search(r"\beval\s*\(|\bnew\s+Function\s*\(", source), name
        assert not re.search(r"set(?:Timeout|Interval)\(\s*[\"'`]", source), name
        assert not re.search(r"\.style\b|cssText|setAttribute\(\s*[\"']style", source), name
        assert not re.search(r"\.on[a-z]+\s*=", source), name  # listeners only via addEventListener
        assert "import(" not in source, name


def test_no_browser_storage_or_cookies_for_personal_data() -> None:
    for name, source in _js_sources().items():
        for api in ("localStorage", "sessionStorage", "indexedDB", "document.cookie", "caches."):
            assert api not in source, f"{name} uses {api}"


def test_no_secrets_or_credentials_in_frontend_files() -> None:
    for name, text in _all_sources().items():
        lowered = text.lower()
        for marker in (
            "gsk_",
            "groq_api",
            "api_key",
            "apikey",
            "authorization",
            "bearer ",
            "secret",
        ):
            assert marker not in lowered, f"{name} contains {marker!r}"


def test_every_form_control_has_an_accessible_label() -> None:
    parser = _markup()
    assert parser.controls
    for tag, values, wrapped in parser.controls:
        labelled = (
            wrapped
            or (values.get("id") or "") in parser.label_for
            or values.get("aria-label")
            or values.get("aria-labelledby")
        )
        assert labelled, f"unlabelled {tag} {values.get('id')}"
    html = HTML.read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert 'name="viewport"' in html


# --- rendering untrusted content (Node.js) ------------------------------------------------


def _poison(value, key: str | None = None):
    """Replace every free-text string of a report with the payload."""
    if isinstance(value, dict):
        poisoned_keys = key in {"models", "prompt_versions", "timings_ms"}
        return {(f"{PAYLOAD}{k}" if poisoned_keys else k): _poison(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_poison(item, key) for item in value]
    if isinstance(value, str) and key not in STRUCTURAL_KEYS:
        return PAYLOAD
    return value


def _walk(node, texts: list[str], tags: set[str], attrs: list[tuple[str, str]]) -> None:
    if "text" in node:
        texts.append(node["text"])
        return
    tags.add(node["tag"])
    attrs.extend(node["attrs"].items())
    for child in node["children"]:
        _walk(child, texts, tags, attrs)


@requires_node
def test_hostile_api_content_is_rendered_as_text_only(tmp_path) -> None:
    report = _poison(sample_report())
    evaluation = {
        "question_id": "beginner-1",
        "level": "beginner",
        "evaluation": {
            "rubric": {
                criterion: {"rating": 3, "justification": PAYLOAD}
                for criterion in (
                    "relevance",
                    "technical_accuracy",
                    "depth",
                    "clarity",
                    "structure",
                )
            },
            "strengths": [PAYLOAD],
            "improvements": [PAYLOAD],
            "model_answer_outline": [PAYLOAD],
            "overall_rating": 3.0,
        },
        "security_flags": [
            {"type": "classifier", "source": "answer", "score": 0.97, "calibrated": False}
        ],
    }
    data = {"report": report, "evaluation": evaluation, "payload": PAYLOAD}
    result = run_frontend_script(tmp_path, "render_hostile", data)
    texts: list[str] = []
    tags: set[str] = set()
    attrs: list[tuple[str, str]] = []
    for tree in result["trees"]:
        _walk(tree, texts, tags, attrs)
    assert tags <= SAFE_TAGS, tags - SAFE_TAGS  # no element was created from the payload
    assert sum(PAYLOAD in text for text in texts) >= 25  # the payload is shown, as text
    for name, value in attrs:
        assert not name.lower().startswith("on")
        assert "<" not in value and "javascript:" not in value.lower(), (name, value)
    assert sorted(result["rejected"]) == ["href", "onclick", "srcdoc", "style"]


def test_payload_poisoning_covers_free_text_fields() -> None:
    report = _poison(sample_report())
    assert report["resume_profile"]["skills"][0] == PAYLOAD
    assert report["critic"]["issues"][0]["message"] == PAYLOAD
    assert report["critic"]["verdict"] == "approve"  # structural values are kept
    assert any(key.startswith(PAYLOAD) for key in report["meta"]["models"])


def test_frontend_files_have_expected_layout() -> None:
    names = {p.relative_to(FRONTEND_DIR).as_posix() for p in FRONTEND_DIR.rglob("*") if p.is_file()}
    assert names == {
        "index.html",
        "css/styles.css",
        *(f"js/{name}" for name in (
            "app.js", "api.js", "dom.js", "upload.js", "analysis.js", "results.js", "interview.js",
        )),
    }  # fmt: skip
    assert all(Path(FRONTEND_DIR / name).stat().st_size > 0 for name in names)


@pytest.mark.parametrize("name", ["upload.js", "results.js", "interview.js", "app.js"])
def test_rendering_modules_build_dom_through_the_safe_helpers(name: str) -> None:
    source = (JS_DIR / name).read_text(encoding="utf-8")
    assert "createElement(" not in source or name == "app.js"  # app.js: one download link
    if name == "app.js":
        assert source.count("createElement(") == 1 and 'createElement("a")' in source
