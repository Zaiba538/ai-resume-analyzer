"""In-memory TTL report store: UUID4 IDs, TTL, expiry, capacity, concurrency, no listing."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from backend.schemas.report import AnalysisReport
from backend.storage.report_store import (
    ReportStamp,
    ReportStore,
    ReportStoreFullError,
    parse_report_id,
)
from tests.fixtures.reports import report_data, synthetic_report

pytestmark = pytest.mark.asyncio

START = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
TTL = 3600


class FakeClock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_store(clock: FakeClock, *, max_reports: int = 500, sweep: float = 60.0) -> ReportStore:
    return ReportStore(
        ttl_seconds=TTL, max_reports=max_reports, sweep_interval_seconds=sweep, clock=clock
    )


async def add(store: ReportStore, **kwargs) -> AnalysisReport:
    return await store.create(lambda stamp: synthetic_report(stamp, **kwargs))


# --- identity and lifetime ------------------------------------------------------------


async def test_create_assigns_uuid4_and_one_hour_lifetime(clock) -> None:
    store = make_store(clock)
    report = await add(store)
    assert report.report_id.version == 4
    assert parse_report_id(str(report.report_id)) == report.report_id
    assert report.created_at == START
    assert report.expires_at == START + timedelta(seconds=TTL)
    assert await store.get(str(report.report_id)) == report


async def test_ids_are_unique(clock) -> None:
    store = make_store(clock, max_reports=2000)
    ids = {(await add(store)).report_id for _ in range(1000)}
    assert len(ids) == 1000


@pytest.mark.parametrize("field", ["report_id", "created_at", "expires_at"])
async def test_report_must_use_the_assigned_stamp(clock, field: str) -> None:
    store = make_store(clock)
    replacement = {
        "report_id": uuid.uuid4(),
        "created_at": START + timedelta(seconds=1),  # still a valid lifetime for the schema
        "expires_at": START + timedelta(seconds=TTL - 1),
    }[field]
    with pytest.raises(ValueError, match="must come from the stamp"):
        await store.create(lambda stamp: synthetic_report(stamp, **{field: replacement}))
    assert await store.count() == 0


async def test_only_exact_analysis_reports_are_stored(clock) -> None:
    class SmugglingReport(AnalysisReport):
        resume_text: str = "CANARY raw resume text"

    store = make_store(clock)
    for build in (
        lambda stamp: report_data(stamp),  # a plain dict
        lambda stamp: SmugglingReport(**report_data(stamp)),  # subclass with a raw-text field
    ):
        with pytest.raises(TypeError, match="AnalysisReport objects only"):
            await store.create(build)
    assert await store.count() == 0


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-uuid",
        str(uuid.uuid1()),
        str(uuid.UUID(int=0)),
        "{" + str(uuid.uuid4()) + "}",
        "urn:uuid:" + str(uuid.uuid4()),
        uuid.uuid4().hex,  # no hyphens
        " " + str(uuid.uuid4()),
        str(uuid.uuid4()) + "\n",
        "../" + str(uuid.uuid4()),
        None,
        12345,
    ],
)
async def test_parse_report_id_accepts_only_canonical_uuid4(value) -> None:
    assert parse_report_id(value) is None


async def test_parse_report_id_is_case_insensitive() -> None:
    value = uuid.uuid4()
    assert parse_report_id(str(value).upper()) == value


async def test_unknown_malformed_and_expired_ids_are_indistinguishable(clock) -> None:
    store = make_store(clock)
    report = await add(store)
    clock.advance(seconds=TTL)
    results = [
        await store.get(str(uuid.uuid4())),  # unknown
        await store.get("not-a-uuid"),  # malformed
        await store.get(str(report.report_id)),  # expired
    ]
    assert results == [None, None, None]


# --- TTL and expiry -------------------------------------------------------------------


async def test_expiry_boundary_and_purge_on_access(clock) -> None:
    store = make_store(clock)
    report = await add(store)
    key = str(report.report_id)
    clock.advance(seconds=TTL, microseconds=-1)
    assert await store.get(key) == report
    clock.advance(microseconds=1)  # exactly expires_at
    assert await store.count() == 1  # still held until accessed or swept
    assert await store.get(key) is None
    assert await store.count() == 0  # purged on access
    clock.advance(seconds=-TTL)  # even if the clock went back, the report is gone
    assert await store.get(key) is None


async def test_configured_ttl_is_applied(clock) -> None:
    store = ReportStore(ttl_seconds=120, max_reports=5, sweep_interval_seconds=60, clock=clock)
    report = await add(store)
    assert report.expires_at - report.created_at == timedelta(seconds=120)
    clock.advance(seconds=119)
    assert await store.get(str(report.report_id)) == report
    clock.advance(seconds=1)
    assert await store.get(str(report.report_id)) is None


async def test_purge_expired_removes_only_expired_reports(clock) -> None:
    store = make_store(clock)
    old = await add(store)
    clock.advance(seconds=1800)
    young = await add(store)
    clock.advance(seconds=1800)
    assert await store.purge_expired() == 1
    assert await store.get(str(old.report_id)) is None
    assert await store.get(str(young.report_id)) == young


async def test_periodic_sweep_purges_without_access(clock) -> None:
    store = make_store(clock, sweep=0.01)
    await store.start()
    try:
        assert store.running
        await add(store)
        clock.advance(seconds=TTL)
        for _ in range(200):
            if await store.count() == 0:
                break
            await asyncio.sleep(0.01)
        assert await store.count() == 0
    finally:
        await store.stop()
    assert not store.running


async def test_sweep_survives_errors_and_logs_type_only(clock, captured_logs, monkeypatch) -> None:
    logging.getLogger("backend").setLevel(logging.INFO)
    store = make_store(clock, sweep=0.01)
    calls = {"n": 0}
    original = store.purge_expired

    async def flaky() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("SECRET-DETAIL")
        return await original()

    monkeypatch.setattr(store, "purge_expired", flaky)
    await store.start()
    try:
        for _ in range(200):
            if calls["n"] >= 2:
                break
            await asyncio.sleep(0.01)
        assert calls["n"] >= 2 and store.running
    finally:
        await store.stop()
    (event,) = captured_logs.events("report_store_sweep_failed")
    assert event["error_code"] == "RuntimeError"
    assert "SECRET-DETAIL" not in captured_logs.text


async def test_start_is_idempotent_and_stop_clears_memory(clock) -> None:
    store = make_store(clock)
    await store.start()
    await store.start()
    report = await add(store)
    await store.stop()
    await store.stop()
    assert not store.running
    assert await store.count() == 0
    assert await store.get(str(report.report_id)) is None


# --- capacity -------------------------------------------------------------------------


async def test_capacity_limit_returns_503_when_still_full(clock, captured_logs) -> None:
    store = make_store(clock, max_reports=3)
    for _ in range(3):
        await add(store)
    with pytest.raises(ReportStoreFullError) as info:
        await add(store)
    assert (info.value.status_code, info.value.code) == (503, "report_store_full")
    assert await store.count() == 3
    assert captured_logs.events("report_store_full")


async def test_full_store_purges_expired_reports_before_refusing(clock) -> None:
    store = make_store(clock, max_reports=3)
    first = await add(store)
    clock.advance(seconds=10)
    for _ in range(2):
        await add(store)
    clock.advance(seconds=TTL - 5)  # only the first report has expired
    replacement = await add(store)
    assert await store.count() == 3
    assert await store.get(str(first.report_id)) is None
    assert await store.get(str(replacement.report_id)) == replacement
    with pytest.raises(ReportStoreFullError):  # nothing else has expired
        await add(store)


async def test_default_capacity_is_500(make_settings) -> None:
    settings = make_settings()
    assert settings.max_stored_reports == 500
    assert settings.report_ttl_seconds == 3600


# --- isolation and concurrency --------------------------------------------------------


async def test_stored_reports_cannot_be_mutated_by_callers(clock) -> None:
    store = make_store(clock)
    built: list[AnalysisReport] = []

    def build(stamp: ReportStamp) -> AnalysisReport:
        built.append(synthetic_report(stamp))
        return built[-1]

    returned = await store.create(build)
    built[0].resume_profile.skills.append("Injected after create")
    returned.resume_profile.skills.append("Injected via returned copy")
    fetched = await store.get(str(returned.report_id))
    fetched.interview_questions.beginner.clear()
    again = await store.get(str(returned.report_id))
    assert again.resume_profile.skills == ["Python", "SQL", "Apache Spark"]
    assert len(again.interview_questions.beginner) == 1


async def test_concurrent_creates_respect_capacity_exactly(clock) -> None:
    store = make_store(clock, max_reports=500)
    results = await asyncio.gather(*(add(store) for _ in range(600)), return_exceptions=True)
    stored = [r for r in results if isinstance(r, AnalysisReport)]
    refused = [r for r in results if isinstance(r, ReportStoreFullError)]
    assert (len(stored), len(refused)) == (500, 100)
    assert len({r.report_id for r in stored}) == 500
    assert await store.count() == 500


async def test_concurrent_reads_and_writes_stay_consistent(clock) -> None:
    store = make_store(clock, max_reports=1000)
    seeded = [await add(store) for _ in range(50)]

    async def reader(report: AnalysisReport) -> bool:
        return await store.get(str(report.report_id)) == report

    tasks = [reader(r) for r in seeded] + [add(store) for _ in range(200)]
    results = await asyncio.gather(*tasks)
    assert all(results[:50])
    assert await store.count() == 250


# --- no listing -----------------------------------------------------------------------


async def test_store_offers_no_way_to_enumerate_reports(clock) -> None:
    store = make_store(clock)
    public = {name for name in dir(store) if not name.startswith("_")}
    assert public == {"create", "get", "purge_expired", "count", "start", "stop", "running"}
    for dunder in ("__iter__", "__len__", "__contains__", "__getitem__", "keys", "values"):
        assert not hasattr(store, dunder) or dunder not in type(store).__dict__


# --- configuration --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl_seconds": 0, "max_reports": 1, "sweep_interval_seconds": 1},
        {"ttl_seconds": 60, "max_reports": 0, "sweep_interval_seconds": 1},
        {"ttl_seconds": 60, "max_reports": 1, "sweep_interval_seconds": 0},
    ],
)
async def test_invalid_configuration_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        ReportStore(**kwargs)


async def test_naive_clock_is_rejected() -> None:
    store = ReportStore(
        ttl_seconds=60, max_reports=1, sweep_interval_seconds=1, clock=lambda: datetime(2026, 1, 1)
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        await add(store)
