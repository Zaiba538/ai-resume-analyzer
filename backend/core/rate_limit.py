"""Rate limiting core (M10): bounded, in-memory, moving-window limits.

Limits are written in the ``limits`` library's string notation (``"5/minute;30/hour"``)
and parsed into ``limits.RateLimitItem`` objects. The window bookkeeping is kept
here rather than in ``limits.storage.MemoryStorage`` because the approved design
needs three things that storage does not provide: a bound on the number of tracked
keys, an injectable clock, and atomic checking of several keys at once.

Policies. A policy is what one endpoint enforces: one or more rules, each a bucket
name, a scope and one or more limits. ``client`` rules are counted per client;
``global`` rules are shared by every client. Policies that name the same global
bucket share one pool (both interview endpoints use ``interview_llm``).

Atomicity. ``RateLimiter.hit`` holds one lock while it checks every limit of the
policy and, only if all of them pass, records the request against all of them. A
rejected request consumes nothing. Nothing in ``hit`` awaits, so it is also atomic
with respect to other coroutines on the event loop.

Moving window. A request counts against a limit for exactly ``window`` seconds: an
entry recorded at ``t`` still counts at ``t + window - epsilon`` and no longer at
``t + window``. ``Retry-After`` is the time until the oldest counted entry of the
limiting window leaves it, rounded up to whole seconds and at least 1.

Identity and privacy. The client host is canonicalised (IPv4 address, IPv6 /64
network, or one shared bucket for anything that is not an IP address) and then
replaced by an HMAC-SHA256 under a random per-limiter secret. Raw addresses are
never stored, logged or returned. The secret never leaves the process, so storage
keys cannot be linked to an address outside it.

Capacity. At most ``max_client_keys`` per-client keys are tracked (one key per
client, bucket and window). Expired keys are removed by a periodic sweep. When the
store is full and a request needs a new key, expired keys are swept first; if none
can be freed, the request is rejected (fail closed) with ``Retry-After`` set to the
time until the earliest tracked key expires. Existing keys are never evicted early,
because evicting a key would reset that client's counters. Global keys are few and
bounded by the policy set, so they do not count towards the capacity.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import math
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import limits

if TYPE_CHECKING:
    from backend.core.config import Settings

logger = logging.getLogger("backend.rate_limit")

Clock = Callable[[], float]
Scope = Literal["client", "global"]

# Bounds on configured limits. They keep the worst-case memory of one key small
# (at most MAX_LIMIT_AMOUNT timestamps) and reject meaningless configurations.
MAX_LIMIT_STRING_CHARS = 200
MAX_LIMITS_PER_RULE = 4
MAX_LIMIT_AMOUNT = 500
MAX_LIMIT_WINDOW_SECONDS = 86_400

IPV6_GROUP_PREFIX = 64
DEFAULT_SWEEP_INTERVAL_SECONDS = 30.0
# At capacity, a sweep is attempted at most this often, so a full store cannot be
# made to sweep on every request.
MIN_CAPACITY_SWEEP_GAP_SECONDS = 1.0

INVALID_CLIENT = "invalid"
GLOBAL_IDENTITY = "*"

# Policy names, one per rate-limited endpoint.
ANALYZE = "analyze"
INTERVIEW_QUESTIONS = "interview_questions"
INTERVIEW_EVALUATE = "interview_evaluate"
MATCH = "match"
PARSE = "parse"
REPORT = "report"
REPORT_PDF = "report_pdf"


def parse_rate_limits(value: str) -> tuple[limits.RateLimitItem, ...]:
    """Parse and validate a limit string such as ``"5/minute;30/hour"``.

    Raises ``ValueError`` when the string is malformed, has more than
    ``MAX_LIMITS_PER_RULE`` limits, repeats a window length, or has an amount or
    window outside the supported bounds.
    """
    if not isinstance(value, str) or len(value) > MAX_LIMIT_STRING_CHARS:
        raise ValueError(f"must be a string of at most {MAX_LIMIT_STRING_CHARS} characters")
    try:
        items = tuple(limits.parse_many(value.strip()))
    except ValueError:
        raise ValueError("must use the limits notation, e.g. '5/minute;30/hour'") from None
    if not items or len(items) > MAX_LIMITS_PER_RULE:
        raise ValueError(f"must contain between 1 and {MAX_LIMITS_PER_RULE} limits")
    windows = [item.get_expiry() for item in items]
    if len(set(windows)) != len(windows):
        raise ValueError("must not repeat a window length")
    for item in items:
        if not 1 <= item.amount <= MAX_LIMIT_AMOUNT:
            raise ValueError(f"each amount must be between 1 and {MAX_LIMIT_AMOUNT}")
        if not 1 <= item.get_expiry() <= MAX_LIMIT_WINDOW_SECONDS:
            raise ValueError(f"each window must be at most {MAX_LIMIT_WINDOW_SECONDS} seconds")
    return items


def canonical_client(host: str | None) -> str:
    """The identity a client is counted under, before hashing.

    IPv4 addresses (including IPv4-mapped IPv6) are used as they are; IPv6
    addresses are grouped by their /64 network, with any zone ID removed. A
    missing host and anything that is not an IP address share one bucket.
    """
    if not host:
        return INVALID_CLIENT
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return INVALID_CLIENT
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return f"4:{address.ipv4_mapped}"
        network = ipaddress.IPv6Network((address.packed, IPV6_GROUP_PREFIX), strict=False)
        return f"6:{network.network_address}/{IPV6_GROUP_PREFIX}"
    return f"4:{address}"


@dataclass(frozen=True, slots=True)
class LimitRule:
    """Limits on one bucket, counted per client or shared by all clients."""

    bucket: str
    scope: Scope
    limits: tuple[limits.RateLimitItem, ...]


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Everything one endpoint enforces; all rules must pass."""

    name: str
    rules: tuple[LimitRule, ...]


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """The outcome of ``RateLimiter.hit``. ``retry_after_seconds`` is 0 when allowed."""

    allowed: bool
    retry_after_seconds: int
    reason: Literal["allowed", "limited", "capacity"]


def build_policies(settings: Settings) -> dict[str, RateLimitPolicy]:
    """The approved M10 policies, built from validated settings."""

    def client(bucket: str, value: str) -> LimitRule:
        return LimitRule(bucket, "client", parse_rate_limits(value))

    def shared(bucket: str, value: str) -> LimitRule:
        return LimitRule(bucket, "global", parse_rate_limits(value))

    interview_pool = shared("interview_llm", settings.rate_limit_interview_global)
    policies = (
        RateLimitPolicy(
            ANALYZE,
            (
                client(ANALYZE, settings.rate_limit_analyze),
                shared("analyze_global", settings.rate_limit_analyze_global),
            ),
        ),
        RateLimitPolicy(
            INTERVIEW_QUESTIONS,
            (client(INTERVIEW_QUESTIONS, settings.rate_limit_interview_questions), interview_pool),
        ),
        RateLimitPolicy(
            INTERVIEW_EVALUATE,
            (client(INTERVIEW_EVALUATE, settings.rate_limit_interview_evaluate), interview_pool),
        ),
        RateLimitPolicy(MATCH, (client(MATCH, settings.rate_limit_match),)),
        RateLimitPolicy(PARSE, (client(PARSE, settings.rate_limit_parse),)),
        RateLimitPolicy(REPORT, (client(REPORT, settings.rate_limit_report),)),
        RateLimitPolicy(REPORT_PDF, (client(REPORT_PDF, settings.rate_limit_report_pdf),)),
    )
    return {policy.name: policy for policy in policies}


class _Window:
    """Timestamps (oldest first) of the requests counted in one moving window."""

    __slots__ = ("amount", "entries", "seconds")

    def __init__(self, amount: int, seconds: float) -> None:
        self.amount = amount
        self.seconds = seconds
        self.entries: deque[float] = deque()

    def prune(self, now: float) -> None:
        cutoff = now - self.seconds
        while self.entries and self.entries[0] <= cutoff:
            self.entries.popleft()

    def full(self) -> bool:
        return len(self.entries) >= self.amount

    def retry_after(self, now: float) -> float:
        """Seconds until the oldest counted entry leaves the window (window is full)."""
        return self.entries[0] + self.seconds - now

    def expires_at(self) -> float:
        return self.entries[-1] + self.seconds if self.entries else -math.inf


def _retry_seconds(delay: float) -> int:
    return max(1, math.ceil(delay))


class RateLimiter:
    """Bounded in-memory moving-window limiter (see the module docstring).

    ``clock`` must be monotonic and returns seconds; it defaults to
    ``time.monotonic``. Tests inject a fake clock and, optionally, a fixed
    ``secret``.
    """

    def __init__(
        self,
        policies: Mapping[str, RateLimitPolicy],
        *,
        max_client_keys: int,
        clock: Clock = time.monotonic,
        secret: bytes | None = None,
        sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
    ) -> None:
        if max_client_keys < 1:
            raise ValueError("max_client_keys must be at least 1")
        if sweep_interval_seconds <= 0:
            raise ValueError("sweep_interval_seconds must be positive")
        for policy in policies.values():
            # Every key of one request must be distinct: a repeated key would count
            # the request twice against the same window. The identity is the same
            # for all client rules of a request, so any placeholder detects repeats.
            keys = [key for key, _, _ in self._targets(policy, "identity")]
            if len(set(keys)) != len(keys):
                raise ValueError("a policy must not repeat a limit on the same bucket and scope")
            needed = sum(len(r.limits) for r in policy.rules if r.scope == "client")
            if needed > max_client_keys:
                # Such a policy could never admit a new client.
                raise ValueError("max_client_keys is smaller than one request's client keys")
        self._policies = dict(policies)
        self._max_client_keys = max_client_keys
        self._clock = clock
        self._secret = secret if secret is not None else secrets.token_bytes(32)
        self._sweep_interval = sweep_interval_seconds
        self._lock = threading.Lock()
        self._client_windows: dict[str, _Window] = {}
        self._global_windows: dict[str, _Window] = {}
        self._last_sweep = clock()
        self._next_expiry = math.inf
        self._exhausted_logged = False

    @classmethod
    def from_settings(
        cls, settings: Settings, *, clock: Clock = time.monotonic, secret: bytes | None = None
    ) -> RateLimiter:
        return cls(
            build_policies(settings),
            max_client_keys=settings.rate_limit_max_tracked_keys,
            clock=clock,
            secret=secret,
        )

    @property
    def policies(self) -> Mapping[str, RateLimitPolicy]:
        return dict(self._policies)

    def client_key(self, host: str | None) -> str:
        """HMAC of the canonical client identity; never contains the address."""
        canonical = canonical_client(host).encode("ascii")
        return hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()[:32]

    def hit(self, policy_name: str, host: str | None) -> RateLimitDecision:
        """Check every limit of the policy and, if all pass, count the request."""
        policy = self._policies[policy_name]
        identity = self.client_key(host)
        with self._lock:
            now = self._clock()
            if now - self._last_sweep >= self._sweep_interval:
                self._sweep(now)
            targets = list(self._targets(policy, identity))
            if self._missing_client_keys(targets) and not self._has_room(targets, now):
                return self._capacity_rejection(now)

            retry = 0.0
            for key, scope, _ in targets:
                window = self._windows(scope).get(key)
                if window is None:
                    continue
                window.prune(now)
                if window.full():
                    retry = max(retry, window.retry_after(now))
            if retry > 0:
                return RateLimitDecision(False, _retry_seconds(retry), "limited")

            created = False
            for key, scope, item in targets:
                store = self._windows(scope)
                window = store.get(key)
                if window is None:
                    window = store[key] = _Window(item.amount, float(item.get_expiry()))
                    created = created or scope == "client"
                window.entries.append(now)
                if scope == "client":
                    self._next_expiry = min(self._next_expiry, now + window.seconds)
            if created:
                self._exhausted_logged = False
            return RateLimitDecision(True, 0, "allowed")

    def tracked_keys(self) -> tuple[str, ...]:
        """Every storage key currently held (for tests and diagnostics)."""
        with self._lock:
            return (*self._client_windows, *self._global_windows)

    def tracked_client_keys(self) -> int:
        with self._lock:
            return len(self._client_windows)

    def _windows(self, scope: Scope) -> dict[str, _Window]:
        return self._client_windows if scope == "client" else self._global_windows

    @staticmethod
    def _targets(
        policy: RateLimitPolicy, identity: str
    ) -> Iterable[tuple[str, Scope, limits.RateLimitItem]]:
        for rule in policy.rules:
            owner = identity if rule.scope == "client" else GLOBAL_IDENTITY
            for item in rule.limits:
                key = f"{rule.scope}:{rule.bucket}:{item.amount}/{item.get_expiry()}:{owner}"
                yield key, rule.scope, item

    def _missing_client_keys(self, targets: list[tuple[str, Scope, limits.RateLimitItem]]) -> int:
        return len(
            {
                key
                for key, scope, _ in targets
                if scope == "client" and key not in self._client_windows
            }
        )

    def _has_room(self, targets: list[tuple[str, Scope, limits.RateLimitItem]], now: float) -> bool:
        """Whether the request's missing client keys fit, sweeping once if that can help.

        The missing keys are counted again after a sweep: the sweep may have removed
        one of this request's own expired keys, which must then be created anew.
        """
        if len(self._client_windows) + self._missing_client_keys(targets) <= self._max_client_keys:
            return True
        if now >= self._next_expiry and now - self._last_sweep >= MIN_CAPACITY_SWEEP_GAP_SECONDS:
            self._sweep(now)
        return len(self._client_windows) + self._missing_client_keys(targets) <= (
            self._max_client_keys
        )

    def _capacity_rejection(self, now: float) -> RateLimitDecision:
        if not self._exhausted_logged:
            self._exhausted_logged = True
            logger.warning(
                "rate_limit_capacity_exhausted",
                extra={"component": "rate_limit", "error_code": "rate_limited"},
            )
        delay = self._next_expiry - now if math.isfinite(self._next_expiry) else 1.0
        return RateLimitDecision(False, _retry_seconds(delay), "capacity")

    def _sweep(self, now: float) -> None:
        """Drop expired entries and keys; recompute the earliest client-key expiry."""
        self._last_sweep = now
        next_expiry = math.inf
        for store, is_client in ((self._client_windows, True), (self._global_windows, False)):
            for key in list(store):
                window = store[key]
                window.prune(now)
                if not window.entries:
                    del store[key]
                elif is_client:
                    next_expiry = min(next_expiry, window.expires_at())
        self._next_expiry = next_expiry
