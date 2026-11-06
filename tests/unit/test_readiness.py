"""Readiness registry semantics."""

from __future__ import annotations

import pytest

from backend.health.readiness import ReadinessRegistry


def test_empty_registry_is_not_ready() -> None:
    assert ReadinessRegistry().is_ready is False


def test_ready_only_when_all_components_ready() -> None:
    registry = ReadinessRegistry()
    registry.register("a", ready=True)
    registry.register("b")
    assert registry.is_ready is False
    assert registry.snapshot() == {"a": True, "b": False}
    registry.set_ready("b")
    assert registry.is_ready is True
    registry.set_ready("a", False)
    assert registry.is_ready is False


def test_duplicate_registration_rejected() -> None:
    registry = ReadinessRegistry()
    registry.register("a")
    with pytest.raises(ValueError):
        registry.register("a")


def test_unknown_component_rejected() -> None:
    with pytest.raises(KeyError):
        ReadinessRegistry().set_ready("missing")


def test_snapshot_is_a_copy() -> None:
    registry = ReadinessRegistry()
    registry.register("a", ready=True)
    snapshot = registry.snapshot()
    snapshot["a"] = False
    assert registry.is_ready is True
