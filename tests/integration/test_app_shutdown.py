"""Process shutdown: the final garbage collection is skipped at interpreter exit.

Without this, ``docker stop`` could exceed its grace period while the
interpreter garbage-collected the large ML heap after uvicorn had returned
(exit 137). See ``backend.main.register_exit_gc_freeze``.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.main as main

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registered(monkeypatch) -> list:
    """Record ``atexit.register`` calls made by ``backend.main``, starting unregistered."""
    calls: list = []
    monkeypatch.setattr(main, "_exit_gc_freeze_registered", False)
    monkeypatch.setattr(main, "atexit", SimpleNamespace(register=calls.append))
    return calls


def test_gc_freeze_is_registered_when_the_lifespan_shuts_down(start_app, registered) -> None:
    client = start_app()
    assert registered == []  # not at start-up
    client.__exit__(None, None, None)
    assert registered == [gc.freeze]


def test_gc_freeze_is_registered_once_per_process(start_app, registered) -> None:
    first = start_app()
    first.__exit__(None, None, None)
    second = start_app()
    second.__exit__(None, None, None)
    first.__enter__()  # the same application started again
    first.__exit__(None, None, None)
    assert registered == [gc.freeze]


def test_freeze_runs_at_interpreter_exit_before_the_final_collection() -> None:
    # atexit runs handlers last-registered first, so a handler registered before
    # the freeze runs after it and sees the frozen objects.
    script = (
        "import atexit, gc\n"
        "atexit.register(lambda: print('frozen', gc.get_freeze_count() > 0, flush=True))\n"
        "from backend.main import register_exit_gc_freeze\n"
        "register_exit_gc_freeze()\n"
        "register_exit_gc_freeze()\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned code
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "frozen True"
