"""M0: hashed dependency locks are complete, CPU-only and mutually consistent."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_LOCK = ROOT / "requirements.txt"
DEV_LOCK = ROOT / "requirements-dev.txt"
TORCH_CPU_URL_PREFIX = "https://download.pytorch.org/whl/cpu/torch-2.15.0%2Bcpu-cp312-cp312-"
FORBIDDEN_PREFIXES = ("nvidia-", "cuda-", "triton")


def _lock_entries(path: Path) -> list[tuple[Requirement, list[str]]]:
    """Parse a uv-generated hashed lock into (requirement, hashes) pairs."""
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[tuple[str, list[str]]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--hash="):
            entries[-1][1].append(stripped.removeprefix("--hash=").rstrip(" \\"))
        elif stripped.startswith("-"):
            pytest.fail(f"{path.name}: unexpected option line {stripped!r}")
        else:
            entries.append((stripped.rstrip(" \\"), []))
    return [(Requirement(spec), hashes) for spec, hashes in entries]


def _top_level(path: Path) -> list[Requirement]:
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #")[0].strip()
        if line and not line.startswith(("#", "-r")):
            reqs.append(Requirement(line))
    return reqs


def _pinned_versions(entries) -> dict[str, str]:
    versions: dict[str, str] = {}
    for req, _ in entries:
        name = canonicalize_name(req.name)
        if req.url:
            versions.setdefault(name, req.url)
        else:
            (spec,) = req.specifier
            versions[name] = spec.version
    return versions


@pytest.fixture(scope="module", params=[RUNTIME_LOCK, DEV_LOCK], ids=lambda p: p.name)
def lock(request) -> list[tuple[Requirement, list[str]]]:
    return _lock_entries(request.param)


def test_every_entry_is_exactly_pinned_and_hashed(lock) -> None:
    for req, hashes in lock:
        assert hashes, f"{req.name} has no hashes"
        assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", h) for h in hashes), req.name
        if not req.url:
            (spec,) = req.specifier
            assert spec.operator == "==", f"{req.name} is not exactly pinned"


def test_no_cuda_or_triton_packages(lock) -> None:
    offenders = [req.name for req, _ in lock if req.name.lower().startswith(FORBIDDEN_PREFIXES)]
    assert not offenders, offenders


def test_torch_is_cpu_only_direct_url_per_platform(lock) -> None:
    torch = [req for req, _ in lock if canonicalize_name(req.name) == "torch"]
    assert {req.url.removeprefix(TORCH_CPU_URL_PREFIX) for req in torch} == {
        "manylinux_2_28_x86_64.whl",
        "win_amd64.whl",
    }
    for req in torch:
        assert req.marker is not None
        linux = req.marker.evaluate({"sys_platform": "linux", "platform_machine": "x86_64"})
        windows = req.marker.evaluate({"sys_platform": "win32", "platform_machine": "AMD64"})
        assert linux != windows, "each torch wheel must apply to exactly one platform"


def test_only_pypi_or_pytorch_cpu_sources(lock) -> None:
    for req, _ in lock:
        if req.url:
            assert canonicalize_name(req.name) == "torch", f"unexpected direct URL: {req.name}"
            assert req.url.startswith(TORCH_CPU_URL_PREFIX)


@pytest.mark.parametrize("source", ["requirements.in", "requirements-dev.in"])
def test_top_level_pins_match_lock(source: str) -> None:
    lock_path = RUNTIME_LOCK if source == "requirements.in" else DEV_LOCK
    locked = _pinned_versions(_lock_entries(lock_path))
    for req in _top_level(ROOT / source):
        name = canonicalize_name(req.name)
        assert name in locked, f"{req.name} missing from {lock_path.name}"
        if not req.url:
            (spec,) = req.specifier
            assert locked[name] == spec.version, f"{req.name}: {locked[name]} != {spec.version}"


def test_dev_lock_has_identical_runtime_pins() -> None:
    runtime = _pinned_versions(_lock_entries(RUNTIME_LOCK))
    dev = _pinned_versions(_lock_entries(DEV_LOCK))
    mismatched = {name: (v, dev.get(name)) for name, v in runtime.items() if dev.get(name) != v}
    assert not mismatched, mismatched
