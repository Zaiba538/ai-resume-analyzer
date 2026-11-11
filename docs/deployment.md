# Deployment guide

> **Status: Docker image built and smoke-tested locally; not yet deployed to
> Hugging Face Spaces.** The repository contains a production `Dockerfile`
> (section 3) and the Hugging Face Spaces front matter at the top of
> `README.md` (section 4). There is no automated deployment pipeline. The
> open items before a Space deployment are listed in section 13.

Each section separates three kinds of statement:

- **Verified**: checked against the code in this repository or the pinned
  library versions (`uvicorn` 0.54.0, `huggingface_hub`, `groq` 0.37.1,
  `langchain-groq` 1.1.3), or, where stated, in the local Docker verification
  (section 3).
- **Recommended**: guidance for operating a deployment, or a proposal not yet
  carried out.
- **Unresolved**: risks or facts that can only be confirmed on a running
  deployment.

---

## 1. Supported deployment targets

| Target | Status |
|---|---|
| Local development (Windows AMD64, Linux x86_64) | Supported and tested; see section 2. |
| Standalone Docker container (Linux x86_64) | Supported; image built and smoke-tested locally (section 3). |
| Hugging Face Spaces, Docker SDK | Configured (`README.md` front matter, section 4); not yet deployed. |

**Verified constraints for every target:**

- Python 3.12 only (`pyproject.toml`: `requires-python = ">=3.12,<3.13"`).
- The hashed locks (`requirements.txt`, `requirements-dev.txt`) are resolved
  only for `linux/x86_64` and `win32/AMD64` (`[tool.uv] environments` in
  `pyproject.toml`). A container image must therefore be **linux/amd64**.
  ARM images (including Apple Silicon without emulation) cannot install the
  lock.
- PyTorch is CPU-only, installed by direct URL from `download.pytorch.org`
  (see `requirements.in`). Building an image needs network access to that host
  as well as to PyPI.
- One process only (section 10).
- The product scope is unchanged by deployment: English-only resumes and job
  descriptions, and the ATS composite stays `not_calibrated`.

---

## 2. Local development

Verified (see also the [README](../README.md)):

```bash
python -m pip install --require-hashes -r requirements.txt
cp .env.example .env          # then set GROQ_API_KEY in .env
uvicorn --factory backend.main:create_app --no-access-log --port 7860
```

- Run from the repository root: settings read `.env` from the **working
  directory** (`backend/core/config.py`, `env_file=".env"`).
- `--factory` is required: `backend.main:create_app` is a function, and
  importing `backend.main` has no side effects.
- Without `--host`, Uvicorn listens on `127.0.0.1` only. This is correct for
  local development and **wrong for a container** (section 8).
- The access log is disabled in code as well (`backend/core/logging.py`), so
  `--no-access-log` is a second safeguard.

`.env` is only a local convenience. In Docker and on Spaces, configuration
must come from real environment variables and secrets; `.env` must never be
copied into an image (`.dockerignore` excludes it, and the `Dockerfile` copies
only `requirements.txt`, `backend/` and `frontend/`).

---

## 3. Docker deployment architecture

### Verified

- The application serves the API and the frontend from one process: the
  `frontend/` directory is mounted at `/` after all API routes
  (`backend/main.py`). No separate web server or build step is needed for the
  frontend.
- If `FRONTEND_DIR` does not exist, the app still starts and only logs
  `frontend_dir_missing`; the UI then returns 404. The image must contain
  `frontend/`.
- PDF fonts are bundled in `backend/assets/fonts/`; no font download is needed.
- Reports, PDFs and rate-limit state are held in memory. The application
  itself writes no data files. The only expected disk writes are the
  Hugging Face model cache (section 6), which the Docker image fills during
  the build.
- `.dockerignore` keeps `.env` files, `.git`, `.github`, virtual
  environments, caches and all `*.pdf` / `*.docx` files out of the build
  context.

### The image (`Dockerfile`, verified)

1. Base image `python:3.12-slim-bookworm`, pinned by digest, always built for
   `linux/amd64` (the platform the lock is resolved for).
2. Installs only the runtime lock, with hashes required, no dependency
   resolution outside the lock and wheels only:
   `python -m pip install --no-cache-dir --require-hashes --no-deps --only-binary=:all: -r requirements.txt`.
   `requirements-dev.txt` and `requirements-legacy.txt` are not installed.
3. Copies only what the app needs: `backend/` and `frontend/`. The legacy files
   (`app.py`, `agents/`, `utils/`), `tests/` and `docs/` are not in the image.
   The application code is owned by root and read-only for the runtime user.
4. Runs as the non-root user `user` (UID/GID 1000) with home `/home/user` and
   `HF_HOME=/home/user/.cache/huggingface` (section 6).
5. Downloads the pinned embedding model during the build, as the runtime user,
   checks that the application's loader finds it offline, and sets
   `HF_HUB_OFFLINE=1` for run time (section 6).
6. Start command (single process, all interfaces, port 7860; `EXPOSE 7860`):

   ```text
   uvicorn --factory backend.main:create_app --no-access-log --host 0.0.0.0 --port 7860
   ```

   Never add `--workers`, and do not set `WEB_CONCURRENCY` (section 10).
7. `HEALTHCHECK` against `GET /health` (not `GET /ready`, section 7), using
   Python's standard library: interval 30 s, timeout 5 s, start period 60 s,
   3 retries.
8. No secret is used at build time. `GROQ_API_KEY` is supplied only at run
   time.

Build and run (from the repository root):

```bash
docker build --platform linux/amd64 -t resume-intelligence:local .
docker run --rm -p 7860:7860 -e GROQ_API_KEY resume-intelligence:local
```

`-e GROQ_API_KEY` without a value copies the key from the shell's environment
into the container, so it never appears in the command line or the image.

### Local verification (2026-11-11)

The image `resume-intelligence:local` was built on Docker Engine 29.6.2
(linux/amd64) and smoke-tested in a container on port 7860:

- The build succeeded: every locked package installed from a wheel, and the
  embedding model was downloaded and then loaded offline during the build.
- Image metadata: architecture `amd64`, user `1000:1000`, exposed port
  `7860/tcp`, about 568 MB uncompressed. Its environment variables and build
  history contain no API key.
- The container runs as the non-root user; `GET /health` and `GET /ready`
  returned 200, and the embedding model loaded offline.
- Through the web interface: the frontend rendered; resume upload and parsing,
  profile extraction, job matching and skill-gap analysis, the Critic and
  security checks, interview question generation, answer evaluation and PDF
  report download all worked.
- The live Groq test suite (`tests/live`, 21 tests) passed.

### Unresolved

- Memory use, start-up time and build time have not been measured. CPU
  PyTorch and the Sentence-Transformer dominate all three.
- Whether any library writes temporary files during uploads has not been
  checked.

---

## 4. Hugging Face Spaces (Docker SDK)

### Configuration prepared (verified in this repository)

- The default `CSP_FRAME_ANCESTORS` is `'self' https://huggingface.co`, so the
  app may be embedded in a Space page on huggingface.co
  (`backend/core/config.py`).
- `.env.example` already says to set `GROQ_API_KEY` as a Space secret.
- `README.md` starts with the Space's YAML front matter:

  ```yaml
  ---
  title: Resume Intelligence Platform
  emoji: 📄
  colorFrom: blue
  colorTo: gray
  sdk: docker
  app_port: 7860
  pinned: false
  ---
  ```

  `sdk: docker` makes the Space build the repository's `Dockerfile`;
  `app_port: 7860` matches the port Uvicorn listens on (section 8). The other
  fields only affect how the Space is shown. GitHub renders the block as a
  small table at the top of the README.
- The image already meets the Spaces runtime requirements checked locally
  (section 3): it runs as UID 1000 with a writable home and Hugging Face
  cache, the embedding model is in the image (no download at start-up or after
  a restart), and it listens on `0.0.0.0:7860`.

### Deployment status: not yet completed

- No Space has been created, nothing has been pushed to a Space repository,
  and nothing in this section has been tested on Spaces yet.
- The steps that remain are listed in section 13; the checks to run on the
  Space are in section 11.

### Space settings (to configure when the Space is created)

| Setting | Value |
|---|---|
| SDK | Docker |
| Secret `GROQ_API_KEY` | **Required.** Add it as a Space **secret**, never as a public variable, and never in the repository or image. Secrets are passed to the container as environment variables. |
| Variables | None required; every other setting has a working default (section 5). Do not set `WEB_CONCURRENCY`, `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `SSL_CERT_FILE` or `SSL_CERT_DIR`. `FORWARDED_ALLOW_IPS` only after the proxy check below. |
| Visibility | Private until the open items in section 13 are resolved. |

### Recommended (based on Hugging Face's Docker Spaces documentation; confirm against the current docs before relying on it)

- Container storage is not persistent across restarts. Reports are held in
  memory and are lost on restart (section 10); the model is in the image and
  is not affected.
- Spaces apply a start-up time limit. The port opens only after the model has
  loaded (section 7); with the model in the image this is a local load, not a
  download.

### Unresolved issues

- **Font files.** The four bundled fonts in `backend/assets/fonts/` (`*.ttf`,
  about 0.2–0.8 MB each) are binary files stored in this repository as
  ordinary Git objects. Hugging Face may reject binary files pushed to a Space
  repository unless they are stored with Xet or Git LFS; confirm against the
  current documentation. Converting them in this repository would rewrite its
  history, so how the files reach the Space (for example a separate snapshot
  of the tracked files for the Space repository) is still to be decided. The
  application needs these files: without them PDF rendering is not ready.
- **Client addresses behind the Spaces proxy.** The address the proxy connects
  from, and whether it sends `X-Forwarded-For` in a form Uvicorn will trust
  (section 9), are unknown. Until checked, all visitors may share one
  rate-limit identity, so the per-client limits (for example 5 analyses per
  minute and 30 per hour) apply to all visitors together. Check this on the
  running Space before setting `FORWARDED_ALLOW_IPS` or making the Space
  public.
- Available CPU and memory, and start-up time, on the chosen Space hardware.
- Whether Spaces sets `WEB_CONCURRENCY` or similar variables (section 10).

---

## 5. Environment variables and secrets

### Application settings (verified)

Every application setting is a field of `Settings` in
`backend/core/config.py`, read from the environment variable of the same name
in upper case (and, locally, from `.env`). There is no prefix. All of them are
listed in [`.env.example`](../.env.example): most with their default values,
and `GROQ_API_BASE_URL`, `CSP_FRAME_ANCESTORS` and `FRONTEND_DIR` as
commented-out lines, because changing them is rarely right. Invalid values stop
the application at start-up with a validation error.

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | **Yes (secret)** | Without it the app starts, but `/ready` returns 503 (`groq_api_key_configured: false`) and analysis and interview requests fail. Surrounding whitespace is stripped. |
| `LOG_LEVEL` | No | `INFO` by default. |
| `GROQ_MODEL_*` (6) | No | Defaults in `.env.example`. |
| `GROQ_API_BASE_URL` | No | Used only by the availability probe; it does not change where LLM calls go. Must be `https://`. **The probe sends `GROQ_API_KEY` to this URL** (`backend/health/upstream.py`), so pointing it at any other host exposes the key. Leave it at its default. |
| `GROQ_PROBE_INTERVAL_SECONDS`, `GROQ_PROBE_TIMEOUT_SECONDS` | No | Probe timing. |
| `LLM_*`, `PROMPT_GUARD_*`, `MAX_REVISIONS` | No | Model-call behaviour. |
| `REPORT_TTL_SECONDS`, `MAX_STORED_REPORTS`, `REPORT_SWEEP_INTERVAL_SECONDS` | No | In-memory reports; the TTL cannot exceed 3600. |
| `ANALYSIS_DEADLINE_SECONDS`, `INTERVIEW_DEADLINE_SECONDS`, `EVALUATION_DEADLINE_SECONDS` | No | Request deadlines. |
| `MAX_UPLOAD_BYTES`, `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`, `MAX_JOB_DESCRIPTION_CHARS`, `MAX_ANSWER_CHARS` | No | Input limits. |
| `RATE_LIMIT_ENABLED`, `RATE_LIMIT_*`, `RATE_LIMIT_MAX_TRACKED_KEYS`, `ANALYZE_MAX_CONCURRENT` | No | Keep rate limiting enabled in production. |
| `PARSE_MAX_CONCURRENT`, `PARSE_DEADLINE_SECONDS` | No | Documents parsed at once across `/parse`, `/match` and `/analyze` (default 2), and the soft parsing deadline (default 30 s). See section 10. |
| `EMBEDDING_MODEL_NAME`, `EMBEDDING_MODEL_REVISION` | No | Pinned model and revision. |
| `API_DOCS_ENABLED` | No | Keep `false` in production (section 9). |
| `CSP_FRAME_ANCESTORS` | No | Who may embed the app in a frame. The value contains spaces and single quotes. Set it without surrounding double quotes in tools that pass values literally, such as `docker run --env-file`: the validator accepts a double-quoted value, which then produces an invalid `frame-ancestors` directive. |
| `HSTS_ENABLED` | No | See section 9. |
| `FRONTEND_DIR` | No | Defaults to the absolute path of `frontend/` next to `backend/`. |

### Variables the application sets itself (verified)

- `LANGSMITH_TRACING` and `LANGCHAIN_TRACING_V2` are **forced** to `false` at
  start-up (`backend/core/logging.py`), overriding any value provided, because
  prompts contain resume content.
- `HF_HUB_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_PROGRESS_BARS=1` and
  `TOKENIZERS_PARALLELISM=false` are set as **defaults** when the embedding
  module is imported (`backend/scoring/embeddings.py`); a value already in the
  environment wins.

### Third-party variables that affect a deployment (verified in the pinned libraries)

| Variable | Read by | Effect | Guidance |
|---|---|---|---|
| `HF_HOME`, `HF_HUB_CACHE` | `huggingface_hub` | Location of the model cache (default `~/.cache/huggingface`). | Point at a writable directory (section 6). |
| `HF_HUB_OFFLINE` | `huggingface_hub` | No Hub requests; only cached files are used. | Set to `1` by the Docker image, which contains the model (section 6). Do not set it for a local run without a cached model. |
| `FORWARDED_ALLOW_IPS` | `uvicorn` | Which proxy addresses may set the client address through `X-Forwarded-For`; default `127.0.0.1,::1`. | Change only after checking the real proxy (section 9). |
| `WEB_CONCURRENCY` | `uvicorn` | Starts that many worker processes when `--workers` is not given. | **Must not be set** (section 10). |
| `GROQ_BASE_URL` | `groq` SDK | Changes the Groq API host. | **Must not be set.** The app does not pass a base URL to the LLM client, so this would redirect every model call, with the API key, to another host. |
| `GROQ_API_BASE` | `langchain-groq` | Changes the Groq API host for the agent calls (not Prompt Guard). | **Must not be set**, for the same reason. |
| `GROQ_PROXY` | `langchain-groq` (read, unused) | None in the pinned versions: langchain-groq 1.1.3 reads it but never passes it to the client, and the `groq` SDK does not read it. | Not needed; do not rely on it to route traffic. |
| `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY` | `httpx` | Route outbound HTTP traffic through a proxy. Both of the app's HTTP clients use httpx defaults (`backend/llm/client.py`, `backend/health/upstream.py`), so these apply to every Groq call, Prompt Guard and the probe. The model download honours proxy variables too. | Leave unset unless the network requires a proxy, and then use only a trusted one. |
| `SSL_CERT_FILE`, `SSL_CERT_DIR` | `httpx` | Replace the trusted certificate authorities for outbound HTTPS. | Leave unset. A substituted authority would let a proxy read Groq traffic, including the key. |

The application code reads no other environment variables; the table above
lists the variables read by its libraries that matter for a deployment.

---

## 6. Embedding model caching and writable directories

### Verified

- The Sentence-Transformer (`EMBEDDING_MODEL_NAME`, pinned to
  `EMBEDDING_MODEL_REVISION`) is loaded once, at start-up, on CPU, with
  `trust_remote_code=False` (`backend/scoring/embeddings.py`). It is read from
  the `huggingface_hub` cache: `HF_HUB_CACHE`, or `$HF_HOME/hub`, or
  `~/.cache/huggingface/hub` by default. Whether it can be downloaded when the
  cache does not have it depends on how the app is run (below).
- A load failure does **not** stop the application. It is logged as
  `embedding_model_load_failed` with only the exception type, `/ready` reports
  `embedding_model_loaded: false` (503), and `/api/v1/analyze` and
  `/api/v1/match` return 503 `embedding_model_unavailable`. Other endpoints
  (parsing, stored reports, interview) keep working.

### In the Docker image (verified)

- `HF_HOME` is `/home/user/.cache/huggingface`, owned by the runtime user
  (UID 1000).
- The model is **baked into the image**: the pinned revision is downloaded once
  during the build, as the runtime user, into that `HF_HOME`, using the same
  arguments as the application's loader. A second build step loads it with
  the application's own loader under `HF_HUB_OFFLINE=1` and fails the build if
  the cache and the settings disagree.
- The image sets `HF_HUB_OFFLINE=1`, so the running container never contacts
  the Hub. There is no download at start-up, and the exact revision is
  recorded in the image. In the local verification (section 3) the model
  loaded offline and `/ready` returned 200.
- Overriding `EMBEDDING_MODEL_NAME` or `EMBEDDING_MODEL_REVISION` at run time
  therefore needs `HF_HUB_OFFLINE=0` and Hub access; otherwise `/ready`
  reports `embedding_model_loaded: false`. Change the model by rebuilding the
  image instead.

### Local runs (without Docker)

- If the model is not cached, it is downloaded from the Hub on first start
  into the cache (or `HF_HOME`) and reused afterwards. The cache must be
  writable by the user running the app, or the load fails, and the machine
  needs outbound access to `huggingface.co` for the first start.

### Unresolved

- Time and memory needed to load the model on the target Space hardware.

---

## 7. Health and readiness checks

### Verified

| Endpoint | Meaning | Status codes |
|---|---|---|
| `GET /health` | Liveness: the process answers. Touches no dependency and makes no outbound call. Not rate limited. | 200 `{"status": "alive"}` |
| `GET /ready` | Readiness of local components. `checks` decides the status code; `upstream.groq` is informational and served from the cached probe result. | 200 when every check is true, otherwise 503 |

`/ready` checks: `config_valid`, `groq_api_key_configured`,
`embedding_model_loaded`, `report_store_ready`, `analysis_graph_compiled`,
`pdf_renderer_ready`. It also reports the ATS status (`not_calibrated`).

**Start-up order (verified in `backend/main.py` and `uvicorn/server.py`):**
Uvicorn runs the application's start-up (model load, font registration,
report store, Groq probe) **before** it opens the listening socket. While the
model loads (and, in a local run without a cached model, downloads), the port
is closed and even `/health` cannot answer. In the Docker image the model is
only loaded from the image, never downloaded. The socket opens once start-up
finishes, whether or not the model loaded.

### In the Docker image (verified)

- The `HEALTHCHECK` uses `GET /health` with a 60 s start period, through
  Python's standard library (`python:3.12-slim` has no `curl`).
- In the local verification (section 3), `/health` and `/ready` both returned
  200.

### Recommended

- Readiness or a post-deployment check: `GET /ready`. Do **not** use `/ready`
  for liveness: a missing key or a failed model load returns 503 permanently,
  and restarting the container will not fix either.

---

## 8. Port and host configuration

### Verified

- Port 7860 appears only in the documented commands; the application does
  not read a `PORT` variable. The port is whatever Uvicorn is started with.
- Uvicorn's default host is `127.0.0.1`.
- The Docker image starts Uvicorn with `--host 0.0.0.0 --port 7860` and
  exposes port 7860.
- The Spaces front matter in `README.md` sets `app_port: 7860`, matching the
  Uvicorn port.

### Recommended

- Keep `app_port` and the Uvicorn port in step if either is ever changed.
- On a platform that assigns the port through `PORT`, the start command must
  pass it to Uvicorn explicitly; the application will not pick it up.

---

## 9. Security and reverse-proxy considerations

### Verified

- Security headers on every response: a strict same-origin
  Content-Security-Policy (with the configured `frame-ancestors`), `nosniff`,
  `no-referrer`, a restrictive Permissions-Policy and COOP/CORP. API responses
  add `Cache-Control: no-store` and `X-Robots-Tag: noindex`
  (`backend/core/security_headers.py`).
- `API_DOCS_ENABLED=false` (default): `/docs`, `/redoc` and `/openapi.json`
  are not served. When `true`, `/docs` and `/redoc` are exempted from the CSP
  because they load assets from a CDN. Keep it `false` in production.
- `HSTS_ENABLED=false` by default.
- Rate limits identify clients only by the connection address as seen by
  Uvicorn (`request.client.host`); the app never reads forwarding headers
  itself. Uvicorn applies `X-Forwarded-For` only for connections from
  `FORWARDED_ALLOW_IPS` (default `127.0.0.1,::1`). See
  [rate-limiting.md](rate-limiting.md#proxies-and-hugging-face-spaces).
- Logs are JSON on stdout and never contain document text, the API key, the
  query string or client addresses.
- The Groq key is a `SecretStr` and is never logged or serialised.

### Recommended

- Serve over HTTPS only, through the platform or a reverse proxy that
  terminates TLS. Turn on `HSTS_ENABLED` only when every access to that host
  is over HTTPS.
- Keep `RATE_LIMIT_ENABLED=true`. Keep `CSP_FRAME_ANCESTORS` limited to the
  hosts that really embed the app (for a standalone Docker deployment that is
  not embedded, `'self'` alone is enough).
- Set `FORWARDED_ALLOW_IPS` only to the real proxy's address, after checking
  it. Trusting any other address (or `*`) lets clients choose their own
  rate-limit identity.
- Never set `GROQ_BASE_URL` or `GROQ_API_BASE`, and leave `GROQ_API_BASE_URL`
  at its default (section 5).
- Leave `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `SSL_CERT_FILE` and
  `SSL_CERT_DIR` unset unless the network requires them (section 5).
- Never put `GROQ_API_KEY` in the image, the repository, build arguments or
  logs.

### Unresolved

- Behind any proxy not on loopback, every visitor may appear with the proxy's
  address and share one set of per-client limits until `FORWARDED_ALLOW_IPS`
  is set correctly. For Hugging Face Spaces this is an open issue (section 4).
- Whether to enable HSTS on Spaces, where TLS is terminated by the platform.

---

## 10. Single-process and in-memory limitations

Verified:

- Reports are stored in process memory for at most one hour
  (`REPORT_TTL_SECONDS` ≤ 3600; at most `MAX_STORED_REPORTS`) and are lost on
  restart or redeploy. There is no database.
- Rate-limit counters and the `/analyze` concurrency gate are per process.
- Running more than one process (`--workers`, `WEB_CONCURRENCY`, several
  replicas behind a load balancer) gives each process its own reports and
  limits: a `report_id` created by one process returns 404 from another, and
  effective limits multiply. See
  [rate-limiting.md](rate-limiting.md#single-process).
- `ANALYZE_MAX_CONCURRENT` (default 2, maximum 8) caps simultaneous analyses
  in the single process; further requests get 429
  `analysis_capacity_exceeded`.
- `PARSE_MAX_CONCURRENT` (default 2, maximum 8) caps documents parsed at the
  same time by `/parse`, `/match` and `/analyze` together; further requests get
  503 `document_processing_busy`. `PARSE_DEADLINE_SECONDS` (default 30) bounds
  how long a request waits for parsing (then 504 `document_processing_timeout`).
  This deadline is soft: Python cannot stop a running thread, so a parse that
  outlives it keeps running, and keeps its slot, until it finishes. A parse that
  never finishes would hold its slot until the process restarts.

Recommended: run exactly one container with one Uvicorn process. Scale up
(more CPU and memory), not out.

---

## 11. Deployment verification

Recommended checks after any deployment (replace `BASE` with the deployment
URL; never paste the API key into commands or logs):

1. `GET BASE/health` returns 200 `{"status": "alive"}`.
2. `GET BASE/ready` returns 200 and every value in `checks` is `true`;
   `upstream.groq.status` reports the probe result.
3. `GET BASE/` serves the web interface, and the browser console shows no CSP
   violations.
4. `GET BASE/docs` and `GET BASE/openapi.json` return 404 (docs disabled).
5. Response headers include `Content-Security-Policy` and, under `/api/`,
   `Cache-Control: no-store`.
6. Run one analysis through the interface with a test resume that contains no
   real personal data; download its PDF; generate interview questions.
7. Check the logs: JSON lines, `application_started`,
   `embedding_model_loaded`, and no document text.
8. Confirm that one process is running.
9. Behind a proxy: check which client address the app sees before relying on
   per-client rate limits.

---

## 12. Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Port never opens; platform reports a start-up timeout | Model still loading, or downloading in a local run (start-up runs before the socket opens) | Logs for `embedding_model_loaded` / `embedding_model_load_failed`; the Docker image contains the model, so on Docker check memory and CPU instead. |
| Connection refused from outside the container | Uvicorn bound to `127.0.0.1` | Start command includes `--host 0.0.0.0`. |
| `/ready` 503, `groq_api_key_configured: false` | Key missing or blank | Secret name is exactly `GROQ_API_KEY` and is set for the running container. |
| `/ready` 503, `embedding_model_loaded: false`; `/api/v1/analyze` and `/api/v1/match` 503 `embedding_model_unavailable` | Local run: cache not writable or no Hub access. Docker: `EMBEDDING_MODEL_*` overridden at run time, so the model is not in the image's cache while `HF_HUB_OFFLINE=1` | Local: `HF_HOME` ownership and permissions; outbound access to huggingface.co. Docker: remove the overrides, or rebuild the image with the new model. |
| Start-up fails with a validation error | An environment value is out of range or malformed | Compare with `.env.example` ranges. |
| UI returns 404 at `/`; log shows `frontend_dir_missing` | `frontend/` not in the image, or wrong `FRONTEND_DIR` | Image contents. |
| Analyses fail with upstream errors | Groq unavailable, quota exhausted, invalid key, a redirected base URL, or a proxy | `/ready` `upstream.groq`; that no `GROQ_BASE_URL` / `GROQ_API_BASE` is set, and whether `HTTPS_PROXY` / `ALL_PROXY` is set. |
| Report links return 404 shortly after creation | Process restarted, TTL passed, or more than one process | Restarts; `WEB_CONCURRENCY` / `--workers`; replica count. |
| Everyone is rate limited together | All clients appear as the proxy's address | Section 9, `FORWARDED_ALLOW_IPS`. |
| Page will not load inside the Space frame | `CSP_FRAME_ANCESTORS` does not allow the embedding page, or was set with literal double quotes | The configured value (section 5). |

---

## 13. Open items before deployment

Done: the `Dockerfile` (section 3), the Spaces front matter (section 4), a
local build and smoke test, and the live Groq test suite (21 tests).

1. Decide how the bundled font files reach the Space repository (section 4),
   without rewriting this repository's history.
2. Create the Space (Docker SDK, private) with `GROQ_API_KEY` as a secret, push
   to it, and run the checks in section 11 on the Space.
3. Verify the Spaces proxy address and set `FORWARDED_ALLOW_IPS` if needed
   (sections 4 and 9), before making the Space public.
4. Measure memory and start-up time on the Space hardware.
5. Decide on `HSTS_ENABLED` and the final `CSP_FRAME_ANCESTORS` for each target.
6. Confirm the live Groq test suite passes with the production key, if it
   differs from the key used for the local verification.
