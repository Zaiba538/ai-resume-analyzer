"""M9 test helpers: a routed fake Groq that also serves the evaluator, with
optional blocking routes, plus a report store holding one synthetic report."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx

from backend.core.config import Settings
from backend.llm.client import LLMClients
from backend.schemas.report import AnalysisReport
from backend.storage.report_store import ReportStore
from tests.conftest import FAKE_API_KEY
from tests.fixtures.groq import Reply, RoutedFakeGroq
from tests.fixtures.reports import synthetic_report

EVALUATOR_PREFIX = "You are a fair, rigorous technical interviewer."


class InterviewFakeGroq(RoutedFakeGroq):
    """``RoutedFakeGroq`` plus an ``evaluator`` route. Routes named in ``block``
    never answer: their requests wait until cancelled, and each cancellation is
    recorded in ``cancelled``. ``release()`` lets waiting requests fail instead,
    so that a test can end cleanly when the code under test never cancels them."""

    PREFIXES = {**RoutedFakeGroq.PREFIXES, "evaluator": EVALUATOR_PREFIX}

    def __init__(self, guard_score: str = "0.01", *, block: tuple[str, ...] = ()) -> None:
        super().__init__(guard_score)
        self.block = set(block)
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self._released = threading.Event()

    @staticmethod
    def route_of(body: dict[str, Any]) -> str:
        is_guard = "prompt-guard" in body.get("model", "")
        if not is_guard and body["messages"][0]["content"].startswith(EVALUATOR_PREFIX):
            return "evaluator"
        return RoutedFakeGroq.route_of(body)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        route = self.route_of(json.loads(request.content))
        if route in self.block:
            self.requests.append(request)
            self.routes.append(route)
            self.started.append(route)
            try:
                while not self._released.is_set():  # only cancellation or release ends it
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                self.cancelled.append(route)
                raise
        return self.handler(request)

    def release(self) -> None:
        """End every blocked request (thread-safe); it then fails (no scripted reply)."""
        self._released.set()

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.async_handler))

    def user_messages(self, route: str) -> list[str]:
        return [body["messages"][1]["content"] for body in self.bodies_for(route)]

    def system_messages(self, route: str) -> list[str]:
        return [body["messages"][0]["content"] for body in self.bodies_for(route)]


def settings_for(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **{"groq_api_key": FAKE_API_KEY, **overrides})


def clients_for(fake: InterviewFakeGroq, settings: Settings) -> LLMClients:
    return LLMClients(settings, http_client=fake.client())


async def store_with_report(**overrides: Any) -> tuple[ReportStore, AnalysisReport]:
    store = ReportStore(ttl_seconds=3600, max_reports=10, sweep_interval_seconds=60)
    report = await store.create(lambda stamp: synthetic_report(stamp, **overrides))
    return store, report


def evaluation_reply(*ratings: int, **changes: Any) -> Reply:
    from tests.unit.test_agents import evaluation

    return Reply(content=json.dumps({**evaluation(*ratings), **changes}, ensure_ascii=False))


def snapshot(store: ReportStore) -> dict:
    return {key: report.model_dump(mode="json") for key, report in store._reports.items()}
