"""Readiness registry.

Each required local component registers itself during application start-up and
reports whether it initialised. ``/ready`` is ready only when every registered
component is. Components join the registry in the milestone that introduces
them (embedding model, compiled graphs, report store, PDF renderer), so the
``checks`` object grows towards the full approved contract.

Upstream (Groq) availability is deliberately *not* a readiness check: it is
reported separately and never flips readiness (see ``backend.health.upstream``).
"""

from __future__ import annotations

from threading import Lock

CONFIG_VALID = "config_valid"
GROQ_API_KEY_CONFIGURED = "groq_api_key_configured"
EMBEDDING_MODEL_LOADED = "embedding_model_loaded"
REPORT_STORE_READY = "report_store_ready"
ANALYSIS_GRAPH_COMPILED = "analysis_graph_compiled"


class ReadinessRegistry:
    """Thread-safe map of required component name to initialised flag."""

    def __init__(self) -> None:
        self._checks: dict[str, bool] = {}
        self._lock = Lock()

    def register(self, name: str, *, ready: bool = False) -> None:
        with self._lock:
            if name in self._checks:
                raise ValueError(f"readiness component already registered: {name}")
            self._checks[name] = ready

    def set_ready(self, name: str, ready: bool = True) -> None:
        with self._lock:
            if name not in self._checks:
                raise KeyError(f"readiness component not registered: {name}")
            self._checks[name] = ready

    def snapshot(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._checks)

    @property
    def is_ready(self) -> bool:
        checks = self.snapshot()
        return bool(checks) and all(checks.values())
