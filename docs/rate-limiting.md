# Rate limiting and analysis concurrency (M10)

The API limits how often each client may call it, caps how many analyses run at
once, and gives every multipart upload a deadline. All of it is in memory, in the
single application process. There is no authentication; clients are told apart
by their network address only.

## What is limited

Each API endpoint has a per-client limit. The two expensive LLM areas also have
a global limit, shared by all clients. Limits use a moving window: a request
counts against a limit for exactly the length of its window.

| Endpoint | Per client (default) | Global (default) |
|---|---|---|
| `POST /api/v1/analyze` | `5/minute;30/hour` | `20/minute` |
| `POST /api/v1/interview/questions` | `10/minute;60/hour` | `60/minute`, shared with evaluate |
| `POST /api/v1/interview/evaluate` | `20/minute;200/hour` | `60/minute`, shared with questions |
| `POST /api/v1/match` | `30/minute` | none |
| `POST /api/v1/parse` | `30/minute` | none |
| `GET /api/v1/report/{report_id}` | `60/minute` | none |
| `GET /api/v1/report/{report_id}/pdf` | `20/minute` | none |

`/health`, `/ready`, the API documentation, the static frontend and paths with
no route are not limited. A request with a method the route does not support
gets 405 without being counted.

The defaults are starting values: the Groq quotas they protect have not been
verified, so they are configurable.

### Order of checks

The rate limit runs after routing and before the request body is read. A
limited request does no other work: no upload is read, no document is parsed,
no report is looked up and no model is called. A request is counted when it
passes the check, whatever the endpoint answers afterwards, including errors
such as 404 or 415.

On `/api/v1/analyze` the rate limit runs first and the concurrency check
second, so a request rejected as busy has used its quota, and a rate-limited
request never takes an analysis slot.

## Responses

Errors use the standard envelope, with the `X-Request-ID` header, `no-store`
and the usual security headers:

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Too many requests. Please try again later.",
    "request_id": "29ce56af6b584e55a6e91d65c4e07894"
  }
}
```

| Status | Code | When | `Retry-After` |
|---:|---|---|---|
| 429 | `rate_limited` | A per-client or global limit is used up | Seconds until the oldest counted request leaves the limiting window (rounded up, at least 1). With several windows, the longest wait. |
| 429 | `rate_limited` | The limiter's store is full and the client is new (see [Capacity](#capacity)) | Seconds until the earliest tracked key expires (at least 1) |
| 429 | `analysis_capacity_exceeded` | `/analyze` only: the maximum number of analyses is already running | Always `10` |
| 408 | `upload_timeout` | A multipart upload did not arrive in full within 60 seconds | None |

* Successful responses carry no rate-limit headers.
* `Retry-After` is added only to these 429 responses. The separate 503
  `upstream_rate_limited` (the language-model service is rate limiting) is
  unchanged: it passes on the service's own `Retry-After` when there is one.
* The busy-server `Retry-After: 10` is a fixed hint. An analysis can take much
  longer, so a client may be told to retry more than once.
* The OpenAPI schema documents 429 on every limited endpoint, and 408 on the
  three upload endpoints.

## Concurrency of analyses

At most `ANALYZE_MAX_CONCURRENT` analyses (default 2) run at a time in the
process. A request takes a slot only once its whole upload has been received,
and holds it through document parsing and the analysis workflow. The slot is
released however the analysis ends: success, error, deadline or cancellation.
When no slot is free the request is rejected at once, not queued.

## Upload deadline

The multipart body of `POST /api/v1/parse`, `POST /api/v1/analyze` and the
multipart form of `POST /api/v1/match` must arrive in full within 60 seconds,
so a client that stalls or sends data slowly cannot hold a request open. The
deadline is fixed in code (`UPLOAD_READ_TIMEOUT_SECONDS`); it is not a setting.
It does not apply to JSON bodies. Nothing is truncated: an upload either arrives
in time or is rejected with 408.

## Configuration

All settings are environment variables (see `.env.example`). Each limit uses
the `limits` notation: one to four limits separated by `;`, such as
`5/minute;30/hour`, each with an amount from 1 to 500 and a window of at most
one day, without repeating a window length. An invalid value stops the
application at start-up.

| Variable | Default | Notes |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | `false` turns off every quota. The analysis concurrency cap stays on. Intended for local development and tests. |
| `RATE_LIMIT_ANALYZE` | `5/minute;30/hour` | Per client |
| `RATE_LIMIT_ANALYZE_GLOBAL` | `20/minute` | All clients |
| `RATE_LIMIT_INTERVIEW_QUESTIONS` | `10/minute;60/hour` | Per client |
| `RATE_LIMIT_INTERVIEW_EVALUATE` | `20/minute;200/hour` | Per client |
| `RATE_LIMIT_INTERVIEW_GLOBAL` | `60/minute` | All clients, both interview endpoints |
| `RATE_LIMIT_MATCH` | `30/minute` | Per client |
| `RATE_LIMIT_PARSE` | `30/minute` | Per client |
| `RATE_LIMIT_REPORT` | `60/minute` | Per client |
| `RATE_LIMIT_REPORT_PDF` | `20/minute` | Per client |
| `RATE_LIMIT_MAX_TRACKED_KEYS` | `10000` | 100 to 25,000. Per-client keys held in memory |
| `ANALYZE_MAX_CONCURRENT` | `2` | 1 to 8 |

At start-up the application also checks its own route table: every route under
`/api/` must have a rate-limit policy and the expected methods, and must run the
rate limit first. Otherwise the application refuses to start.

## Client identity and privacy

* A client is identified only by the connection address the server reports
  (`request.client.host`). The application never reads forwarding headers itself.
* IPv4 addresses are used as they are; IPv6 addresses are grouped by their /64
  network; an IPv4-mapped IPv6 address counts as its IPv4 address. A missing
  address, or anything that is not an IP address, shares a single bucket.
* The address is replaced by an HMAC-SHA256 under a random secret created with
  the application at start-up. Raw addresses are never stored, logged or
  returned.
* A rejection is logged like any other error, with its error code and status
  and without the address. When the store becomes full, one
  `rate_limit_capacity_exhausted` warning is logged, also without an address.

## Capacity

Each client uses one key per endpoint bucket and window (for example two for
`/analyze`: a minute and an hour). At most `RATE_LIMIT_MAX_TRACKED_KEYS`
per-client keys are held; each takes roughly 1 KB, plus 32 bytes per counted
request. Expired keys are removed by a periodic sweep.

When the store is full and a request needs a new key, expired keys are swept
first. If none can be freed, the request is rejected with 429 `rate_limited`:
the store fails closed. Clients that already have keys are not affected, and
existing keys are never evicted early, because that would reset their counts.
A client that arrives while the store is full is turned away until keys expire.

## Single process

Every counter lives in the memory of the one application process:

* restarting the process resets all limits;
* running more than one worker process or replica would give each its own
  limits, multiplying them, and would also split the in-memory report store.

The service is therefore run as exactly one process, for example:

```
uvicorn --factory backend.main:create_app --no-access-log --port 7860
```

Shared external storage (such as Redis) would be needed only for more than one
process; it is not used.

## Proxies and Hugging Face Spaces

The client address comes from uvicorn. With its defaults, uvicorn applies
`X-Forwarded-For` only when the connection comes from `127.0.0.1` or `::1`
(`--forwarded-allow-ips`, or the `FORWARDED_ALLOW_IPS` environment variable,
changes this). It then uses the right-most address that is not itself trusted,
so a client cannot pick a new identity by prepending addresses. Other headers,
such as `X-Real-IP` or `Forwarded`, are not used.

On Hugging Face Spaces the application runs behind the platform's proxy. Its
behaviour has not been verified: the repository has no deployment
configuration, and the address the proxy connects from is unknown.

* If the proxy does not connect from a trusted address, uvicorn ignores
  `X-Forwarded-For`, and every visitor appears with the proxy's address. The
  per-client limits then act as one limit shared by everyone behind that
  address.
* Trusting addresses other than the real proxy would let clients set their own
  identity with `X-Forwarded-For`.

Check the proxy's behaviour on a running Space before changing
`FORWARDED_ALLOW_IPS`. The application's proxy configuration is unchanged.

## Testing

The test suite disables quotas by default (`RATE_LIMIT_ENABLED=false` in
`tests/conftest.py`), because every in-process test request comes from the same
address; the rate-limit tests turn them back on. A test also checks that
production defaults keep rate limiting enabled. `tests/integration/test_api_uvicorn_e2e.py`
runs the application on a real uvicorn server on a loopback port to check the
429s, recovery, uvicorn's default proxy trust and the exemption of `/health`.
