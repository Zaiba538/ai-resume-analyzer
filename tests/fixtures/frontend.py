"""Helpers for the frontend tests (M11 Phase 1).

The static-file tests need only Python. The behavioural tests run the real
frontend ES modules with Node.js's standard library (no npm packages): the
modules are copied into a temporary directory marked ``"type": "module"``, a
minimal fake DOM is installed (tests/fixtures/frontend_js/fake_dom.mjs), and
the script prints one JSON line with its findings. When Node.js is not
installed those tests are skipped locally; the static security tests still run.
CI installs Node.js and sets FRONTEND_REQUIRE_NODE=1, so there the Node.js
tests can never be skipped: without Node.js they fail.

The fake DOM throws on every markup-parsing sink, so a rendering test fails if
the code ever tries to insert HTML instead of text.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from backend.core.config import PROJECT_ROOT
from backend.storage.report_store import ReportStamp
from tests.fixtures.reports import synthetic_report

FRONTEND_DIR = PROJECT_ROOT / "frontend"
JS_DIR = FRONTEND_DIR / "js"
API_JS = JS_DIR / "api.js"
SCRIPTS_DIR = Path(__file__).parent / "frontend_js"

NODE = shutil.which("node")
# Set in CI: the Node.js tests must run there, never skip silently.
NODE_REQUIRED = os.environ.get("FRONTEND_REQUIRE_NODE") == "1"
requires_node = pytest.mark.skipif(
    NODE is None and not NODE_REQUIRED, reason="Node.js is not installed"
)


def run_frontend_script(tmp_path: Path, script: str, data: Any = None) -> Any:
    """Run tests/fixtures/frontend_js/<script>.mjs next to a copy of frontend/js.

    ``data`` is passed as JSON in the ``FRONTEND_INPUT`` environment variable; the
    script's last output line is parsed as JSON and returned.
    """
    assert NODE is not None, "Node.js is required (FRONTEND_REQUIRE_NODE=1) but not installed"
    work = tmp_path / "frontend-node"
    shutil.copytree(JS_DIR, work / "js")
    for source in SCRIPTS_DIR.glob("*.mjs"):
        shutil.copy(source, work / source.name)
    (work / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    env = {**os.environ, "FRONTEND_INPUT": json.dumps(data)}
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned code
        [NODE, str(work / f"{script}.mjs")],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, "the script printed nothing"
    return json.loads(lines[-1])


def object_block(source: str, name: str) -> str:
    """The text of ``export const NAME = Object.freeze({ ... });`` in a JS source."""
    match = re.search(
        rf"export const {name} = Object\.freeze\(\{{(.*?)\n\}}\);", source, flags=re.DOTALL
    )
    assert match, f"{name} not found"
    return match.group(1)


def error_message_codes() -> set[str]:
    """The keys of ERROR_MESSAGES in frontend/js/api.js."""
    block = object_block(API_JS.read_text(encoding="utf-8"), "ERROR_MESSAGES")
    return set(re.findall(r"^\s{2}([a-z_]+):", block, flags=re.MULTILINE))


def frontend_endpoints() -> dict[str, tuple[str, str]]:
    """ENDPOINTS in frontend/js/api.js: name -> (method, path template)."""
    block = object_block(API_JS.read_text(encoding="utf-8"), "ENDPOINTS")
    found = re.findall(
        r'^\s{2}(\w+): Object\.freeze\(\{ method: "([A-Z]+)", path: "([^"]+)" \}\),$',
        block,
        flags=re.MULTILINE,
    )
    return {name: (method, path) for name, method, path in found}


def backend_error_codes() -> set[str]:
    """Every error code the backend can put in an error envelope.

    All ``AppError`` subclasses (every backend module is imported first, so none
    is missed), the PDF renderer errors, and the status-derived codes used for
    HTTP exceptions and request validation.
    """
    import importlib
    import pkgutil

    import backend
    from backend.core.errors import STATUS_CODES, AppError
    from backend.reports.errors import PdfRenderError

    for module in pkgutil.walk_packages(backend.__path__, "backend."):
        importlib.import_module(module.name)

    codes = {AppError.code, PdfRenderError.code, *STATUS_CODES.values()}
    codes |= {"client_error", "server_error"}  # code_for_status() fallbacks

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            codes.add(sub.code)
            walk(sub)

    walk(AppError)
    walk(PdfRenderError)
    return codes


def sample_report(*, with_job: bool = True) -> dict[str, Any]:
    """A schema-valid AnalysisReport as the API would return it (JSON)."""
    now = datetime.now(UTC)
    stamp = ReportStamp(uuid.uuid4(), now, now + timedelta(hours=1))
    return synthetic_report(stamp, with_job=with_job).model_dump(mode="json")
