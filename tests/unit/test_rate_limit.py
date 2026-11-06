"""M10 rate-limiting core: parsing, identity, moving windows, atomicity and capacity.

Every test uses a fake clock and a fixed secret, so results are deterministic and
nothing sleeps. Addresses are documentation ranges (RFC 5737, RFC 3849).
"""

from __future__ import annotations

import asyncio
import ipaddress
import threading
import time

import pytest

from backend.core import rate_limit
from backend.core.rate_limit import (
    ANALYZE,
    INTERVIEW_EVALUATE,
    INTERVIEW_QUESTIONS,
    INVALID_CLIENT,
    MATCH,
    PARSE,
    REPORT,
    REPORT_PDF,
    LimitRule,
    RateLimitDecision,
    RateLimiter,
    RateLimitPolicy,
    build_policies,
    canonical_client,
    parse_rate_limits,
)

SECRET = b"s" * 32
V4_A, V4_B, V4_C = "192.0.2.10", "198.51.100.7", "203.0.113.99"
V6_A = "2001:db8:1:2:aaaa:bbbb:cccc:dddd"
V6_A_SAME_64 = "2001:db8:1:2::1"
V6_OTHER_64 = "2001:db8:1:3::1"


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def rule(bucket: str, limits: str, scope: rate_limit.Scope = "client") -> LimitRule:
    return LimitRule(bucket, scope, parse_rate_limits(limits))


def policy(name: str, *rules: LimitRule) -> RateLimitPolicy:
    return RateLimitPolicy(name, rules)


def limiter(*policies: RateLimitPolicy, clock: FakeClock, max_keys: int = 1000, **kw):
    return RateLimiter(
        {p.name: p for p in policies}, max_client_keys=max_keys, clock=clock, secret=SECRET, **kw
    )


def allowed(decision: RateLimitDecision) -> bool:
    return decision.allowed


# --- limit strings -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5/minute;30/hour", [(5, 60), (30, 3600)]),
        ("  20/minute  ", [(20, 60)]),
        ("10/2 minutes", [(10, 120)]),
        ("5 per second", [(5, 1)]),
        ("1/day", [(1, 86_400)]),
        ("500/second;1/minute;2/hour;3/day", [(500, 1), (1, 60), (2, 3600), (3, 86_400)]),
    ],
)
def test_valid_limit_strings(value: str, expected: list[tuple[int, int]]) -> None:
    assert [(i.amount, i.get_expiry()) for i in parse_rate_limits(value)] == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "five per minute",
        "5/fortnight",
        "0/minute",
        "501/minute",
        "1000/second",
        "5/minute;6/minute",
        "5/minute;5/60 seconds",
        "1/second;2/minute;3/hour;4/day;5/2 hours",
        "1/2 days",
        "5/minute;" * 30,
        " " * 300 + "5/minute",
        None,
        5,
    ],
)
def test_invalid_limit_strings_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        parse_rate_limits(value)  # type: ignore[arg-type]


def test_parse_errors_do_not_echo_the_input() -> None:
    marker = "INJECTED-MARKER"
    with pytest.raises(ValueError) as excinfo:
        parse_rate_limits(f"5/{marker}")
    assert marker not in str(excinfo.value)


# --- identity ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (V4_A, f"4:{V4_A}"),
        ("::ffff:192.0.2.10", f"4:{V4_A}"),
        (V6_A, "6:2001:db8:1:2::/64"),
        (V6_A_SAME_64, "6:2001:db8:1:2::/64"),
        ("2001:0db8:0001:0002:0000:0000:0000:0001", "6:2001:db8:1:2::/64"),
        ("fe80::1234%eth0", "6:fe80::/64"),
        ("::1", "6:::/64"),
        ("127.0.0.1", "4:127.0.0.1"),
    ],
)
def test_canonical_client(host: str, expected: str) -> None:
    assert canonical_client(host) == expected


@pytest.mark.parametrize(
    "host",
    [None, "", "testclient", "localhost", "999.1.1.1", "1.2.3.4.5", "01.2.3.4", "[::1]",
     "192.0.2.10:443", " 192.0.2.10", "2001:db8::/64", "x" * 10_000],
)  # fmt: skip
def test_non_ip_hosts_share_the_invalid_bucket(host: str | None) -> None:
    assert canonical_client(host) == INVALID_CLIENT


def test_client_keys_group_ipv6_by_64_and_separate_everything_else() -> None:
    lim = limiter(clock=FakeClock())
    assert lim.client_key(V6_A) == lim.client_key(V6_A_SAME_64)
    assert lim.client_key(V6_A) != lim.client_key(V6_OTHER_64)
    assert lim.client_key(V4_A) == lim.client_key("::ffff:192.0.2.10")
    assert lim.client_key(V4_A) != lim.client_key(V4_B)
    assert lim.client_key(None) == lim.client_key("testclient") == lim.client_key("")
    assert lim.client_key(None) != lim.client_key(V4_A)


def test_client_keys_are_hmacs_that_depend_on_the_secret() -> None:
    clock = FakeClock()
    lim = limiter(clock=clock)
    key = lim.client_key(V4_A)
    assert len(key) == 32 and all(ch in "0123456789abcdef" for ch in key)
    assert V4_A not in key
    other = RateLimiter({}, max_client_keys=1, clock=clock, secret=b"t" * 32)
    assert other.client_key(V4_A) != key


def test_default_secret_is_random_per_limiter() -> None:
    first = RateLimiter({}, max_client_keys=1)
    second = RateLimiter({}, max_client_keys=1)
    assert first.client_key(V4_A) != second.client_key(V4_A)


def test_ipv6_clients_in_one_64_share_a_limit() -> None:
    clock = FakeClock()
    lim = limiter(policy("p", rule("b", "2/minute")), clock=clock)
    assert allowed(lim.hit("p", V6_A))
    assert allowed(lim.hit("p", V6_A_SAME_64))
    assert not allowed(lim.hit("p", "2001:db8:1:2:ffff::9"))
    assert allowed(lim.hit("p", V6_OTHER_64))


# --- moving window and Retry-After -----------------------------------------------------


def test_limit_admits_exactly_the_amount_and_rejections_consume_nothing() -> None:
    clock = FakeClock()
    lim = limiter(policy("p", rule("b", "3/minute")), clock=clock)
    assert [lim.hit("p", V4_A).allowed for _ in range(3)] == [True, True, True]
    for _ in range(50):
        clock.advance(1)
        assert lim.hit("p", V4_A) == RateLimitDecision(False, 60 - int(clock.now - 1000), "limited")
    clock.now = 1060.0  # all three entries leave the window together
    assert [lim.hit("p", V4_A).allowed for _ in range(4)] == [True, True, True, False]


def test_window_boundary_is_exclusive() -> None:
    clock = FakeClock()
    lim = limiter(policy("p", rule("b", "1/minute")), clock=clock)
    assert allowed(lim.hit("p", V4_A))
    clock.now = 1059.999
    assert lim.hit("p", V4_A) == RateLimitDecision(False, 1, "limited")
    clock.now = 1060.0
    assert allowed(lim.hit("p", V4_A))


def test_window_moves_with_each_entry() -> None:
    """Unlike a fixed window, capacity returns one entry at a time."""
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "2/minute")), clock=clock)
    assert allowed(lim.hit("p", V4_A))  # t=0
    clock.now = 30.0
    assert allowed(lim.hit("p", V4_A))  # t=30
    clock.now = 59.0
    assert lim.hit("p", V4_A).retry_after_seconds == 1
    clock.now = 60.0
    assert allowed(lim.hit("p", V4_A))  # the t=0 entry left; t=30 and t=60 remain
    clock.now = 61.0
    assert lim.hit("p", V4_A) == RateLimitDecision(False, 29, "limited")
    clock.now = 90.0
    assert allowed(lim.hit("p", V4_A))


@pytest.mark.parametrize(
    ("elapsed", "retry"),
    [
        (0.0, 60),
        (0.4, 60),
        (10.2, 50),
        (10.7, 50),
        (30.9, 30),
        (58.0, 2),
        (59.01, 1),
        (59.5, 1),
        (59.999, 1),
    ],  # fmt: skip
)
def test_retry_after_rounds_up_and_is_at_least_one(elapsed: float, retry: int) -> None:
    clock = FakeClock()
    lim = limiter(policy("p", rule("b", "1/minute")), clock=clock)
    lim.hit("p", V4_A)
    clock.advance(elapsed)
    assert lim.hit("p", V4_A) == RateLimitDecision(False, retry, "limited")


def test_retry_after_waits_for_every_exhausted_window() -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "2/minute;3/hour")), clock=clock)
    assert allowed(lim.hit("p", V4_A))  # t=0
    clock.now = 60.0
    assert allowed(lim.hit("p", V4_A))  # minute: [60]; hour: [0, 60]
    assert allowed(lim.hit("p", V4_A))  # minute: [60, 60]; hour: [0, 60, 60]
    clock.now = 61.0  # minute frees at 120 (59 s), hour at 3600 (3539 s)
    assert lim.hit("p", V4_A) == RateLimitDecision(False, 3539, "limited")


def test_retry_after_is_the_longest_of_the_exhausted_windows() -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "2/minute;3/hour")), clock=clock)
    assert allowed(lim.hit("p", V4_A))  # t=0
    assert allowed(lim.hit("p", V4_A))  # t=0, minute window now full
    assert lim.hit("p", V4_A).retry_after_seconds == 60
    clock.now = 60.0
    assert allowed(lim.hit("p", V4_A))  # hour window now full (t=0, 0, 60)
    assert lim.hit("p", V4_A).retry_after_seconds == 3540
    clock.now = 3600.0  # both t=0 entries leave the hour window
    assert allowed(lim.hit("p", V4_A))  # minute: [3600]; hour: [60, 3600]
    assert allowed(lim.hit("p", V4_A))  # minute: [3600, 3600]; hour: [60, 3600, 3600]
    assert lim.hit("p", V4_A) == RateLimitDecision(False, 60, "limited")  # both full


def test_allowed_decision_has_no_retry_after() -> None:
    lim = limiter(policy("p", rule("b", "1/minute")), clock=FakeClock())
    assert lim.hit("p", V4_A) == RateLimitDecision(True, 0, "allowed")


def test_unknown_policy_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        limiter(clock=FakeClock()).hit("missing", V4_A)


# --- per-client and global limits ---------------------------------------------------------


def test_client_limits_are_independent() -> None:
    lim = limiter(policy("p", rule("b", "1/minute")), clock=FakeClock())
    assert allowed(lim.hit("p", V4_A))
    assert not allowed(lim.hit("p", V4_A))
    assert allowed(lim.hit("p", V4_B))
    assert allowed(lim.hit("p", None))
    assert not allowed(lim.hit("p", "testclient"))


def test_global_limit_is_shared_by_all_clients() -> None:
    clock = FakeClock()
    lim = limiter(policy("p", rule("b", "5/minute"), rule("g", "3/minute", "global")), clock=clock)
    assert [lim.hit("p", host).allowed for host in (V4_A, V4_B, V4_C)] == [True, True, True]
    assert lim.hit("p", "192.0.2.200") == RateLimitDecision(False, 60, "limited")
    clock.advance(60)
    assert allowed(lim.hit("p", "192.0.2.200"))


def test_global_pool_is_shared_between_policies() -> None:
    pool = rule("pool", "2/minute", "global")
    lim = limiter(
        policy("q", rule("q", "10/minute"), pool),
        policy("e", rule("e", "10/minute"), pool),
        clock=FakeClock(),
    )
    assert allowed(lim.hit("q", V4_A))
    assert allowed(lim.hit("e", V4_B))
    assert not allowed(lim.hit("q", V4_C))
    assert not allowed(lim.hit("e", V4_A))


def test_client_rejection_does_not_consume_the_global_limit() -> None:
    lim = limiter(
        policy("p", rule("b", "1/minute"), rule("g", "2/minute", "global")), clock=FakeClock()
    )
    assert allowed(lim.hit("p", V4_A))
    for _ in range(10):
        assert not allowed(lim.hit("p", V4_A))  # per-client limit
    assert allowed(lim.hit("p", V4_B))  # the global pool still had one slot


def test_global_rejection_does_not_consume_client_limits() -> None:
    clock = FakeClock()
    lim = limiter(policy("p", rule("b", "2/minute"), rule("g", "1/minute", "global")), clock=clock)
    assert allowed(lim.hit("p", V4_B))
    for _ in range(10):
        assert not allowed(lim.hit("p", V4_A))  # global limit
    clock.advance(60)
    assert allowed(lim.hit("p", V4_A))
    clock.advance(1)
    assert lim.hit("p", V4_A).reason == "limited"  # global again, not the client's 2/minute
    clock.advance(60)
    assert allowed(lim.hit("p", V4_A))


def test_later_rule_failure_rolls_back_nothing_because_nothing_was_recorded() -> None:
    clock = FakeClock()
    lim = limiter(
        policy("p", rule("a", "5/minute"), rule("b", "1/minute"), rule("g", "5/minute", "global")),
        clock=clock,
    )
    assert allowed(lim.hit("p", V4_A))
    for _ in range(3):
        assert not allowed(lim.hit("p", V4_A))  # rule "b" fails
    # Rule "a" and the global rule recorded only the single admitted request.
    assert [lim.hit("p", host).allowed for host in (V4_B, V4_C, "192.0.2.1", "192.0.2.2")] == [
        True, True, True, True,
    ]  # fmt: skip
    assert not allowed(lim.hit("p", "192.0.2.3"))


# --- atomicity under concurrency --------------------------------------------------------------


def _hammer(lim: RateLimiter, policy_name: str, hosts: list[str]) -> list[bool]:
    barrier = threading.Barrier(len(hosts))
    results: list[bool] = [False] * len(hosts)

    def run(index: int) -> None:
        barrier.wait()
        results[index] = lim.hit(policy_name, hosts[index]).allowed

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(hosts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    return results


@pytest.mark.parametrize("round_", range(5))
def test_concurrent_hits_from_one_client_admit_exactly_the_limit(round_: int) -> None:
    lim = limiter(policy("p", rule("b", "10/minute")), clock=FakeClock())
    assert sum(_hammer(lim, "p", [V4_A] * 64)) == 10


@pytest.mark.parametrize("round_", range(5))
def test_concurrent_hits_from_many_clients_admit_exactly_the_global_limit(round_: int) -> None:
    lim = limiter(
        policy("p", rule("b", "1/minute"), rule("g", "7/minute", "global")), clock=FakeClock()
    )
    hosts = [f"192.0.2.{i}" for i in range(1, 65)]
    assert sum(_hammer(lim, "p", hosts)) == 7


def test_check_and_record_are_one_critical_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread switch between checking and recording must not let others in.

    Checking is slowed down so that, without the lock, every thread would pass the
    check before any of them records its request.
    """
    real_full = rate_limit._Window.full

    def slow_full(self: rate_limit._Window) -> bool:
        result = real_full(self)
        time.sleep(0.005)
        return result

    monkeypatch.setattr(rate_limit._Window, "full", slow_full)
    lim = limiter(
        policy("p", rule("b", "2/minute"), rule("g", "3/minute", "global")), clock=FakeClock()
    )
    lim.hit("p", V4_B)  # create both windows so that every thread runs the slow check
    assert sum(_hammer(lim, "p", [V4_A] * 16)) == 2
    assert sum(_hammer(lim, "p", [V4_C] * 16)) == 0  # global pool: 1 + 2 = 3


def test_concurrent_hits_never_exceed_either_limit() -> None:
    lim = limiter(
        policy("p", rule("b", "3/minute"), rule("g", "10/minute", "global")), clock=FakeClock()
    )
    hosts = [f"198.51.100.{i % 8}" for i in range(80)]  # 8 clients, 10 requests each
    results = _hammer(lim, "p", hosts)
    assert sum(results) == 10
    per_client: dict[str, int] = {}
    for host, ok in zip(hosts, results, strict=True):
        per_client[host] = per_client.get(host, 0) + ok
    assert max(per_client.values()) <= 3


@pytest.mark.asyncio
async def test_hits_from_worker_threads_and_the_event_loop_are_atomic() -> None:
    lim = limiter(policy("p", rule("b", "25/minute")), clock=FakeClock())
    threaded = [asyncio.to_thread(lim.hit, "p", V4_A) for _ in range(40)]
    results = await asyncio.gather(*threaded)
    inline = [lim.hit("p", V4_A) for _ in range(10)]
    assert sum(d.allowed for d in (*results, *inline)) == 25


# --- capacity, expiry and memory bounds ---------------------------------------------------------


def test_capacity_rejects_new_clients_but_keeps_serving_known_ones() -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "2/minute")), clock=clock, max_keys=3)
    for host in (V4_A, V4_B, V4_C):
        assert allowed(lim.hit("p", host))
        clock.advance(10)  # keys expire at 60, 70 and 80
    assert lim.hit("p", "192.0.2.1") == RateLimitDecision(False, 30, "capacity")
    assert allowed(lim.hit("p", V4_A))  # known client, own counters intact
    assert lim.hit("p", V4_A).reason == "limited"
    assert lim.tracked_client_keys() == 3


def test_capacity_frees_up_when_keys_expire_and_never_evicts_live_keys() -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "1/minute")), clock=clock, max_keys=2)
    assert allowed(lim.hit("p", V4_A))  # expires at 60
    clock.now = 20.0
    assert allowed(lim.hit("p", V4_B))  # expires at 80
    clock.now = 59.0
    assert lim.hit("p", V4_C).reason == "capacity"
    assert lim.hit("p", V4_A).reason == "limited"  # not evicted: still counted
    clock.now = 60.0
    assert allowed(lim.hit("p", V4_C))  # V4_A's key expired and was swept
    assert lim.hit("p", V4_B).reason == "limited"
    assert lim.tracked_client_keys() == 2


def test_capacity_counts_keys_per_window() -> None:
    lim = limiter(policy("p", rule("b", "5/minute;30/hour")), clock=FakeClock(), max_keys=3)
    assert allowed(lim.hit("p", V4_A))  # two keys
    assert lim.hit("p", V4_B).reason == "capacity"  # would need two more


def test_global_keys_do_not_use_capacity() -> None:
    lim = limiter(
        policy("p", rule("b", "5/minute"), rule("g", "100/minute;500/hour", "global")),
        clock=FakeClock(),
        max_keys=2,
    )
    assert allowed(lim.hit("p", V4_A))
    assert allowed(lim.hit("p", V4_B))
    assert lim.tracked_client_keys() == 2
    assert len(lim.tracked_keys()) == 4


def test_capacity_retry_after_tracks_refreshed_keys_and_is_at_least_one_second() -> None:
    clock = FakeClock(0.0)
    lim = limiter(
        policy("p", rule("b", "10/minute")), clock=clock, max_keys=1, sweep_interval_seconds=1e6
    )
    assert allowed(lim.hit("p", V4_A))  # expires at 60
    clock.now = 30.0
    assert allowed(lim.hit("p", V4_A))  # refreshed: now expires at 90; recorded 60 is stale
    clock.now = 59.0
    assert lim.hit("p", V4_B) == RateLimitDecision(False, 1, "capacity")  # stale estimate
    clock.now = 89.5  # past the stale estimate: sweep, find the real expiry (90)
    assert lim.hit("p", V4_B) == RateLimitDecision(False, 1, "capacity")
    clock.now = 90.2  # expired, but within the sweep gap: no sweep, estimate is in the past
    assert lim.hit("p", V4_B) == RateLimitDecision(False, 1, "capacity")
    clock.now = 90.5  # gap elapsed: sweep frees the key
    assert allowed(lim.hit("p", V4_B))


def test_capacity_does_not_sweep_when_nothing_can_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "10/hour")), clock=clock, max_keys=1)
    assert allowed(lim.hit("p", V4_A))
    sweeps = 0
    real_sweep = lim._sweep

    def counting(now: float) -> None:
        nonlocal sweeps
        sweeps += 1
        real_sweep(now)

    monkeypatch.setattr(lim, "_sweep", counting)
    for i in range(1000):
        clock.now = i * 0.001  # one second in total
        assert lim.hit("p", f"198.51.100.{i % 250}").reason == "capacity"
    assert sweeps == 0  # nothing has expired, so a sweep cannot help


def test_capacity_exhaustion_is_logged_once_without_identities(captured_logs) -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "1/minute")), clock=clock, max_keys=1)
    lim.hit("p", V4_A)
    for _ in range(5):
        assert lim.hit("p", V4_B).reason == "capacity"
    (record,) = captured_logs.events("rate_limit_capacity_exhausted")
    assert record["component"] == "rate_limit" and record["level"] == "WARNING"
    clock.now = 61.0
    assert allowed(lim.hit("p", V4_B))  # capacity recovered
    clock.now = 70.0
    assert lim.hit("p", V4_C).reason == "capacity"
    assert len(captured_logs.events("rate_limit_capacity_exhausted")) == 2
    for host in (V4_A, V4_B, V4_C, lim.client_key(V4_A), lim.client_key(V4_B)):
        assert host not in captured_logs.text


def test_sweep_removes_expired_global_keys() -> None:
    clock = FakeClock(0.0)
    lim = limiter(
        policy("p", rule("b", "5/minute"), rule("g", "50/minute", "global")),
        policy("q", rule("q", "5/minute")),
        clock=clock,
        sweep_interval_seconds=30,
    )
    lim.hit("p", V4_A)
    clock.now = 100.0
    lim.hit("q", V4_B)  # sweep before recording: everything from t=0 has expired
    assert lim.tracked_keys() == (f"client:q:5/60:{lim.client_key(V4_B)}",)


def test_periodic_sweep_removes_expired_keys() -> None:
    clock = FakeClock(0.0)
    lim = limiter(
        policy("p", rule("b", "5/minute"), rule("g", "50/minute", "global")),
        clock=clock,
        sweep_interval_seconds=30,
    )
    for i in range(20):
        lim.hit("p", f"192.0.2.{i}")
    assert len(lim.tracked_keys()) == 21
    clock.now = 29.0
    lim.hit("p", "203.0.113.5")
    assert len(lim.tracked_keys()) == 22  # no sweep yet
    clock.now = 61.0
    lim.hit("p", V4_B)  # sweep: only the t=29 entries and this request remain
    assert lim.tracked_client_keys() == 2
    assert len(lim.tracked_keys()) == 3


def test_store_never_exceeds_its_capacity() -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "3/minute;10/hour")), clock=clock, max_keys=100)
    for i in range(5_000):
        clock.now = i * 0.5
        lim.hit("p", str(ipaddress.IPv4Address(0xC0000200 + i)))
        assert lim.tracked_client_keys() <= 100


def test_entries_per_window_never_exceed_the_amount() -> None:
    lim = limiter(policy("p", rule("b", "3/minute")), clock=FakeClock())
    for _ in range(100):
        lim.hit("p", V4_A)
    (window,) = lim._client_windows.values()
    assert len(window.entries) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_client_keys": 0}, "at least 1"),
        ({"max_client_keys": 1}, "smaller than one request"),
        ({"max_client_keys": 10, "sweep_interval_seconds": 0}, "positive"),
    ],
)
def test_invalid_limiter_configuration(kwargs: dict, message: str) -> None:
    two_keys = policy("p", rule("b", "1/minute;2/hour"))
    with pytest.raises(ValueError, match=message):
        RateLimiter({"p": two_keys}, clock=FakeClock(), **kwargs)


def test_default_clock_is_monotonic() -> None:
    lim = RateLimiter({"p": policy("p", rule("b", "1/minute"))}, max_client_keys=10)
    assert lim._clock is rate_limit.time.monotonic
    assert allowed(lim.hit("p", V4_A))
    assert not allowed(lim.hit("p", V4_A))


# --- privacy ------------------------------------------------------------------------------------


def test_storage_never_holds_raw_or_canonical_addresses() -> None:
    hosts = [V4_A, V6_A, "::ffff:203.0.113.99", "fe80::1%eth0", "testclient", None]
    lim = limiter(
        policy("p", rule("b", "5/minute;20/hour"), rule("g", "100/minute", "global")),
        clock=FakeClock(),
    )
    for host in hosts:
        lim.hit("p", host)
    stored = " ".join(lim.tracked_keys())
    forbidden = {h for h in hosts if h} | {canonical_client(h) for h in hosts}
    forbidden |= {"2001:db8:1:2", "203.0.113.99", "fe80", "eth0", "testclient"}
    for value in forbidden:
        assert value not in stored
    for window in (*lim._client_windows.values(), *lim._global_windows.values()):
        assert all(isinstance(entry, float) for entry in window.entries)


def test_storage_keys_name_only_scope_bucket_window_and_hmac() -> None:
    lim = limiter(
        policy("p", rule("b", "5/minute"), rule("g", "9/hour", "global")), clock=FakeClock()
    )
    lim.hit("p", V4_A)
    assert sorted(lim.tracked_keys()) == sorted(
        [f"client:b:5/60:{lim.client_key(V4_A)}", "global:g:9/3600:*"]
    )


def test_limiter_logs_nothing_on_normal_traffic(captured_logs) -> None:
    lim = limiter(
        policy("p", rule("b", "1/minute"), rule("g", "2/minute", "global")), clock=FakeClock()
    )
    for host in (V4_A, V4_A, V6_A, V4_B, V4_C):
        lim.hit("p", host)
    assert captured_logs.text == ""


# --- approved policies ---------------------------------------------------------------------------


def _shape(p: RateLimitPolicy) -> list[tuple[str, str, list[tuple[int, int]]]]:
    return [(r.bucket, r.scope, [(i.amount, i.get_expiry()) for i in r.limits]) for r in p.rules]


def test_approved_default_policies(make_settings) -> None:
    policies = build_policies(make_settings())
    assert set(policies) == {
        ANALYZE, INTERVIEW_QUESTIONS, INTERVIEW_EVALUATE, MATCH, PARSE, REPORT, REPORT_PDF,
    }  # fmt: skip
    assert _shape(policies[ANALYZE]) == [
        ("analyze", "client", [(5, 60), (30, 3600)]),
        ("analyze_global", "global", [(20, 60)]),
    ]
    assert _shape(policies[INTERVIEW_QUESTIONS]) == [
        ("interview_questions", "client", [(10, 60), (60, 3600)]),
        ("interview_llm", "global", [(60, 60)]),
    ]
    assert _shape(policies[INTERVIEW_EVALUATE]) == [
        ("interview_evaluate", "client", [(20, 60), (200, 3600)]),
        ("interview_llm", "global", [(60, 60)]),
    ]
    assert _shape(policies[MATCH]) == [("match", "client", [(30, 60)])]
    assert _shape(policies[PARSE]) == [("parse", "client", [(30, 60)])]
    assert _shape(policies[REPORT]) == [("report", "client", [(60, 60)])]
    assert _shape(policies[REPORT_PDF]) == [("report_pdf", "client", [(20, 60)])]


def test_policies_follow_settings(make_settings) -> None:
    policies = build_policies(
        make_settings(rate_limit_report_pdf="2/second;7/day", rate_limit_interview_global="9/hour")
    )
    assert _shape(policies[REPORT_PDF]) == [("report_pdf", "client", [(2, 1), (7, 86_400)])]
    assert policies[INTERVIEW_QUESTIONS].rules[1] == policies[INTERVIEW_EVALUATE].rules[1]
    assert _shape(policies[INTERVIEW_EVALUATE])[1] == ("interview_llm", "global", [(9, 3600)])


def test_from_settings_uses_the_capacity_and_policies(make_settings) -> None:
    clock = FakeClock()
    lim = RateLimiter.from_settings(make_settings(rate_limit_max_tracked_keys=100), clock=clock)
    assert lim._max_client_keys == 100
    assert set(lim.policies) == set(build_policies(make_settings()))
    for _ in range(5):
        assert allowed(lim.hit(ANALYZE, V4_A))
    assert lim.hit(ANALYZE, V4_A) == RateLimitDecision(False, 60, "limited")


def test_default_interview_pool_is_shared_end_to_end(make_settings) -> None:
    lim = RateLimiter.from_settings(make_settings(), clock=FakeClock(), secret=SECRET)
    hosts = [f"192.0.2.{i}" for i in range(1, 61)]
    for i, host in enumerate(hosts):
        name = INTERVIEW_QUESTIONS if i % 2 else INTERVIEW_EVALUATE
        assert allowed(lim.hit(name, host))
    assert not allowed(lim.hit(INTERVIEW_QUESTIONS, "198.51.100.1"))
    assert not allowed(lim.hit(INTERVIEW_EVALUATE, "198.51.100.2"))


def test_default_analyze_global_limit(make_settings) -> None:
    lim = RateLimiter.from_settings(make_settings(), clock=FakeClock(), secret=SECRET)
    for i in range(20):
        assert allowed(lim.hit(ANALYZE, f"192.0.2.{i}"))
    assert lim.hit(ANALYZE, "192.0.2.250").reason == "limited"


# --- Phase 1 corrections: capacity overshoot and bounds -------------------------------------


def _two_window_and_day_policies() -> tuple[RateLimitPolicy, RateLimitPolicy]:
    return (
        policy("a", rule("a", "5/minute;30/hour")),
        policy("b", rule("b", "1/day")),
    )


def test_capacity_sweep_that_removes_the_requests_own_key_cannot_overshoot() -> None:
    """Regression: this exact sequence used to leave 4 keys with a capacity of 3."""
    clock = FakeClock(0.0)
    lim = limiter(
        *_two_window_and_day_policies(), clock=clock, max_keys=3, sweep_interval_seconds=1e9
    )
    assert allowed(lim.hit("a", V4_A))  # t=0: A minute key (expires 60), A hour key (3600)
    clock.now = 3000.0
    assert allowed(lim.hit("b", V4_B))  # store full: A minute, A hour, B
    clock.now = 3001.0
    assert allowed(lim.hit("b", V4_C))  # capacity sweep drops A minute; C admitted
    assert lim.tracked_client_keys() == 3
    b_key, c_key = (f"client:b:1/86400:{lim.client_key(h)}" for h in (V4_B, V4_C))
    clock.now = 3600.0  # A hour has just expired but is still stored
    # A needs both keys again (the sweep removes the expired hour key), but only one fits.
    assert lim.hit("a", V4_A) == RateLimitDecision(False, 85800, "capacity")  # B expires at 89400
    assert lim.tracked_client_keys() == 2
    assert set(lim.tracked_keys()) == {b_key, c_key}  # live keys kept, counters intact
    assert not allowed(lim.hit("b", V4_B))
    assert not allowed(lim.hit("b", V4_C))


@pytest.mark.parametrize("seed", range(3))
def test_store_never_exceeds_capacity_when_clients_return(seed: int) -> None:
    """Returning clients with partly expired multi-window keys never push past the cap."""
    clock = FakeClock(0.0)
    lim = limiter(
        policy("a", rule("a", "2/minute;5/hour;9/day")),
        policy("b", rule("b", "3/minute")),
        clock=clock,
        max_keys=12,
        sweep_interval_seconds=45,
    )
    state = seed * 7919 + 1
    for _ in range(6_000):
        state = (state * 1_103_515_245 + 12_345) % 2**31
        clock.advance((state >> 8) % 400 / 10)  # 0 to 40 s between requests
        host = f"192.0.2.{(state >> 4) % 9}"
        lim.hit("a" if state % 3 else "b", host)
        assert lim.tracked_client_keys() <= 12


def test_partly_expired_known_client_is_rejected_at_capacity_with_counters_kept() -> None:
    clock = FakeClock(0.0)
    lim = limiter(
        policy("a", rule("a", "5/minute;30/hour")),
        policy("b", rule("b", "1/minute")),
        clock=clock,
        max_keys=3,
        sweep_interval_seconds=30,
    )
    assert allowed(lim.hit("a", V4_A))  # t=0: A minute + A hour
    clock.now = 30.0
    assert allowed(lim.hit("b", V4_B))  # B (expires 90); store full
    clock.now = 61.0
    assert allowed(lim.hit("b", V4_C))  # periodic sweep drops A minute; C (expires 121)
    hour_key = f"client:a:30/3600:{lim.client_key(V4_A)}"
    assert hour_key in lim.tracked_keys()
    clock.now = 70.0  # A is known (hour key) but needs a new minute key: no room
    assert lim.hit("a", V4_A) == RateLimitDecision(False, 20, "capacity")
    assert len(lim._client_windows[hour_key].entries) == 1  # not reset, not consumed
    clock.now = 91.0  # B expired at 90 and is swept to make room
    assert allowed(lim.hit("a", V4_A))
    assert len(lim._client_windows[hour_key].entries) == 2  # the old count carried over


def test_capacity_rejection_leaves_the_global_pool_untouched() -> None:
    clock = FakeClock(0.0)
    lim = limiter(
        policy("p", rule("b", "5/minute"), rule("g", "4/minute", "global")),
        clock=clock,
        max_keys=2,
    )
    assert allowed(lim.hit("p", V4_A))
    assert allowed(lim.hit("p", V4_B))
    (pool,) = lim._global_windows.values()
    before = list(pool.entries)
    for i in range(20):
        clock.now = 1.0 + i
        assert lim.hit("p", f"203.0.113.{i}").reason == "capacity"
    assert list(pool.entries) == before
    clock.now = 30.0
    assert allowed(lim.hit("p", V4_A))
    assert allowed(lim.hit("p", V4_B))  # exactly the two remaining global slots
    assert lim.hit("p", V4_A).reason == "limited"


def test_global_key_count_is_fixed_as_clients_grow(make_settings) -> None:
    clock = FakeClock(0.0)
    lim = RateLimiter.from_settings(make_settings(), clock=clock, secret=SECRET)
    for i in range(2_000):
        clock.now = i * 0.01
        host = str(ipaddress.IPv4Address(0xC6336400 + i))
        lim.hit((ANALYZE, INTERVIEW_QUESTIONS, INTERVIEW_EVALUATE, MATCH)[i % 4], host)
        globals_ = [k for k in lim.tracked_keys() if k.startswith("global:")]
        assert len(globals_) <= 2
    assert sorted(globals_) == ["global:analyze_global:20/60:*", "global:interview_llm:60/60:*"]
    assert lim.tracked_client_keys() > 500  # many clients, still two global keys


def test_global_windows_never_exceed_their_limits() -> None:
    clock = FakeClock(0.0)
    lim = limiter(
        policy("p", rule("b", "500/second"), rule("g", "7/minute;11/hour", "global")),
        clock=clock,
    )
    for i in range(3_000):
        clock.now = i * 1.5
        lim.hit("p", f"192.0.2.{i % 200}")
        for window in lim._global_windows.values():
            assert len(window.entries) <= window.amount
    # The hour pool filled (11) and stopped admissions; the minute pool then expired.
    assert {w.amount: len(w.entries) for w in lim._global_windows.values()} == {11: 11}


def test_expired_keys_remain_while_there_is_no_traffic() -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "1/minute"), rule("g", "9/minute", "global")), clock=clock)
    for host in (V4_A, V4_B, V4_C):
        lim.hit("p", host)
    keys = set(lim.tracked_keys())
    for later in (61.0, 600.0, 86_400.0):
        clock.now = later  # long past expiry, but nothing calls hit
        assert set(lim.tracked_keys()) == keys
        assert lim.tracked_client_keys() == 3


def test_expired_keys_are_removed_when_traffic_resumes() -> None:
    clock = FakeClock(0.0)
    lim = limiter(policy("p", rule("b", "1/minute"), rule("g", "9/minute", "global")), clock=clock)
    for host in (V4_A, V4_B, V4_C):
        lim.hit("p", host)
    clock.now = 86_400.0
    assert allowed(lim.hit("p", "203.0.113.1"))
    assert sorted(lim.tracked_keys()) == sorted(
        [f"client:b:1/60:{lim.client_key('203.0.113.1')}", "global:g:9/60:*"]
    )
    (pool,) = lim._global_windows.values()
    assert list(pool.entries) == [86_400.0]  # the old global entries went too


def test_recreating_only_a_global_key_does_not_rearm_the_capacity_warning(captured_logs) -> None:
    """Only a new client key means capacity recovered; a re-created global key does not."""
    clock = FakeClock(0.0)
    lim = limiter(
        policy("p", rule("b", "10/hour"), rule("g", "5/minute", "global")),
        clock=clock,
        max_keys=1,
        sweep_interval_seconds=30,
    )
    assert allowed(lim.hit("p", V4_A))  # client key (expires 3600), global key (60)
    clock.now = 1.0
    assert lim.hit("p", V4_B).reason == "capacity"  # warning logged
    clock.now = 100.0
    assert allowed(lim.hit("p", V4_A))  # sweep dropped the global key; only it is re-created
    clock.now = 101.0
    assert lim.hit("p", V4_B).reason == "capacity"  # still exhausted: no second warning
    assert len(captured_logs.events("rate_limit_capacity_exhausted")) == 1


# --- duplicate storage keys ------------------------------------------------------------------

DUPLICATE_MESSAGE = "a policy must not repeat a limit on the same bucket and scope"


@pytest.mark.parametrize(
    "rules",
    [
        (rule("b", "5/minute"), rule("b", "5/minute")),
        (rule("b", "5/minute;30/hour"), rule("b", "30/hour")),
        (rule("g", "9/minute", "global"), rule("g", "9/minute", "global")),
        (rule("b", "5/minute"), rule("g", "9/hour", "global"), rule("b", "5/60 seconds")),
    ],
)
def test_rules_that_repeat_a_storage_key_are_rejected(rules: tuple[LimitRule, ...]) -> None:
    with pytest.raises(ValueError) as excinfo:
        RateLimiter({"p": RateLimitPolicy("p", rules)}, max_client_keys=100, clock=FakeClock())
    assert str(excinfo.value) == DUPLICATE_MESSAGE  # fixed: no bucket, limit or policy name


def test_one_duplicate_policy_rejects_the_whole_limiter() -> None:
    good = policy("ok", rule("b", "5/minute"))
    bad = policy("bad", rule("x", "1/hour"), rule("x", "1/hour"))
    with pytest.raises(ValueError, match=DUPLICATE_MESSAGE):
        RateLimiter({"ok": good, "bad": bad}, max_client_keys=100, clock=FakeClock())


@pytest.mark.parametrize(
    "rules",
    [
        (rule("b", "5/minute"), rule("b", "30/hour")),  # same bucket, other window
        (rule("b", "5/minute"), rule("b", "6/minute")),  # same window, other amount
        (rule("b", "5/minute"), rule("b", "5/minute", "global")),  # same bucket, other scope
        (rule("a", "5/minute"), rule("b", "5/minute")),  # other bucket
    ],
)
def test_distinct_rules_are_accepted_and_counted_once(rules: tuple[LimitRule, ...]) -> None:
    clock = FakeClock()
    lim = RateLimiter({"p": RateLimitPolicy("p", rules)}, max_client_keys=100, clock=clock)
    assert allowed(lim.hit("p", V4_A))
    assert len(lim.tracked_keys()) == len(rules)
    for window in (*lim._client_windows.values(), *lim._global_windows.values()):
        assert list(window.entries) == [clock.now]  # one entry per window, never two


def test_the_same_bucket_may_be_shared_across_policies() -> None:
    shared_rule = rule("shared", "2/minute")
    lim = RateLimiter(
        {"p": policy("p", shared_rule), "q": policy("q", shared_rule)},
        max_client_keys=100,
        clock=FakeClock(),
    )
    assert allowed(lim.hit("p", V4_A))
    assert allowed(lim.hit("q", V4_A))
    assert not allowed(lim.hit("p", V4_A))


def test_application_policies_have_no_duplicate_keys(make_settings) -> None:
    for settings in (
        make_settings(),
        make_settings(
            rate_limit_analyze="1/second;2/minute;3/hour;4/day",
            rate_limit_interview_questions="1/second;2/minute;3/hour;4/day",
            rate_limit_interview_evaluate="1/second;2/minute;3/hour;4/day",
            rate_limit_interview_global="1/second;2/minute;3/hour;4/day",
            rate_limit_analyze_global="1/second;2/minute;3/hour;4/day",
        ),
    ):
        lim = RateLimiter.from_settings(settings, clock=FakeClock())
        assert set(lim.policies) == set(build_policies(settings))
