"""M0: the active interpreter and installed packages match the approved baseline."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def _applicable_top_level_requirements() -> list[Requirement]:
    reqs = []
    for source in ("requirements.in", "requirements-dev.in"):
        for line in (ROOT / source).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r")):
                continue
            req = Requirement(line)
            if req.marker is None or req.marker.evaluate():
                reqs.append(req)
    return reqs


def test_python_is_3_12() -> None:
    assert sys.version_info[:2] == (3, 12), sys.version


def test_installed_versions_match_top_level_pins() -> None:
    mismatches = {}
    for req in _applicable_top_level_requirements():
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            mismatches[req.name] = "not installed"
            continue
        expected = "2.15.0+cpu" if req.url else next(iter(req.specifier)).version
        if installed != expected:
            mismatches[req.name] = f"{installed} != {expected}"
    assert not mismatches, mismatches


def test_torch_is_cpu_build() -> None:
    import torch

    assert torch.__version__ == "2.15.0+cpu"
    assert torch.version.cuda is None
    assert not torch.cuda.is_available()


def test_core_libraries_import() -> None:
    import fastapi  # noqa: F401
    import langchain_groq  # noqa: F401
    import langgraph.graph  # noqa: F401
    import pydantic_settings  # noqa: F401
    import reportlab  # noqa: F401
    import sentence_transformers  # noqa: F401
    import slowapi  # noqa: F401
