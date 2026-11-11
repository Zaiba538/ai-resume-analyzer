---
title: Resume Intelligence Platform
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Resume Intelligence Platform

## Overview

A multi-agent AI platform that analyses a resume against an optional job
description, reviews its own output, and helps the candidate practise for
interviews.

It is a FastAPI backend with LangGraph workflows and Groq-hosted models, plus a
Vanilla JavaScript frontend served by the same application.

The repository also contains an older prototype that is not part of this
platform; see [Legacy Streamlit application](#legacy-streamlit-application).

---

## Features

- **Resume upload**: PDF or DOCX, validated by extension and file content,
  with size, page and text-length limits.
- **Resume profile**: skills, education, experience, projects and
  certifications extracted by an LLM agent into a strict schema.
- **Job match**: semantic similarity (Sentence-Transformer embeddings), skill
  coverage and a TF-IDF baseline, computed without an LLM.
- **Skill-gap analysis**: matching and missing skills, with evidence taken
  from the resume's skills, experience, projects and certifications.
- **Critic review**: a Critic agent and deterministic checks review the other
  agents' output; flagged output is revised at most once (configurable).
- **Prompt-injection signals**: a Prompt Guard classifier flags suspicious text
  in the documents. It only flags; it never rejects a request.
- **Interview practice**: questions at beginner, intermediate and advanced
  levels, and an evaluation of the candidate's answers.
- **PDF report**: the stored analysis rendered as a downloadable PDF.
- **Web interface**: Upload, Analysing and Results screens, with Overview,
  Resume profile, Job match, Review and safety, and Interview practice tabs.
- **Protection**: rate limiting, a limit on concurrent analyses, request-size
  limits, security headers with a strict Content Security Policy, and logs
  that never contain document text.

### Scope and limitations

- **ATS score is not calibrated.** The ATS composite always has the status
  `not_calibrated` and no score is produced. The other match metrics are
  reported separately.
- **English only.** Resumes and job descriptions written mainly in a
  non-Latin script (for example Hindi or Marathi in Devanagari) are rejected
  before any model is called. Other Latin-script languages are not detected,
  and results for them are not supported.
- **Reports are kept in memory for at most one hour** (`REPORT_TTL_SECONDS`,
  default and maximum 3600) and are lost when the process restarts. There is
  no database and no listing endpoint; a report can only be fetched by its ID.
  Uploaded files and extracted text are not stored.
- **Single process.** The report store, rate limits and analysis concurrency
  limit live in the memory of one process. Run one Uvicorn process without
  `--workers`; several workers or replicas would each keep separate reports and
  separate limits. See [docs/rate-limiting.md](docs/rate-limiting.md).
- **LLM output can vary** between runs, and the Prompt Guard display threshold
  is not calibrated.

---

## Architecture

```text
Browser (frontend/, Vanilla JavaScript)
   │  same origin
   ▼
FastAPI application  (backend/main.py → create_app)
   ├── middleware: request ID, security headers, access log, body-size limit
   ├── rate limits on every /api/ route; analysis concurrency gate
   ├── /api/v1/analyze ──► LangGraph analysis workflow
   │        resume agent ─► interview agent ┐
   │                      ├► match metrics  ├► Critic ─► revise (≤ MAX_REVISIONS) ─► report
   │                      └► skill-gap agent┘
   │        + Prompt Guard (flag only)
   ├── /api/v1/interview/* ──► question service and answer-evaluation workflow
   ├── in-memory report store (TTL ≤ 1 hour)
   └── static frontend mounted at /
            │
            ▼
   Groq API (agents, Critic, evaluator, Prompt Guard)
   Hugging Face Hub (embedding model: downloaded on first start when run
                     locally; built into the Docker image)
```

---

## Tech stack

- Python 3.12, FastAPI, Uvicorn, Pydantic and pydantic-settings
- LangGraph for workflow orchestration; langchain-core, langchain-groq and the
  Groq SDK for structured model calls
- Sentence-Transformers with CPU-only PyTorch, scikit-learn (TF-IDF)
- pypdf and python-docx for parsing, ReportLab for PDF reports
- Vanilla JavaScript, HTML and CSS (no framework, no build step, no CDN)
- pytest, pytest-asyncio and Ruff for testing and linting

Default models (configurable, see `.env.example`):

| Role | Model |
|---|---|
| Resume, skill gap, Critic | `openai/gpt-oss-120b` |
| Interview questions, answer evaluator | `openai/gpt-oss-20b` |
| Prompt Guard | `meta-llama/llama-prompt-guard-2-86m` |

---

## Getting started

### Requirements

- **Python 3.12** (`>=3.12,<3.13`). Other versions are not supported.
- **Linux x86_64 or Windows AMD64.** The hashed locks are resolved only for
  these platforms (PyTorch is pinned to CPU-only wheels for each).
- A **Groq API key**.
- Network access to the Groq API and to the Hugging Face Hub (for a local
  run, the embedding model is downloaded on first start; the Docker image
  already contains it, see [Deployment](#deployment)).
- Optional: **Node.js 24** to run the frontend behaviour tests.

### Install

```bash
git clone https://github.com/Zaiba538/ai-resume-analyzer.git
cd ai-resume-analyzer
```

Create and activate a Python 3.12 virtual environment.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the hashed runtime lock:

```bash
python -m pip install --require-hashes -r requirements.txt
```

For development and tests, install the dev lock instead. It contains every
runtime dependency plus pytest, pytest-asyncio and Ruff:

```bash
python -m pip install --require-hashes -r requirements-dev.txt
```

The locks are generated from `requirements.in` and `requirements-dev.in`; the
regeneration commands are in the headers of those files.

### Configure

Copy the example file and set your Groq key in `.env`:

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env`.

`.env` is git-ignored. Never commit it or share its contents.

### Run locally (development only)

This command is for local development on your own machine. It is not a
production or deployment command: it listens on `127.0.0.1` only (see
[Deployment](#deployment)).

Run it from the repository root, because settings read `.env` from the working
directory. The application is created by a factory, so `--factory` is
required:

```bash
uvicorn --factory backend.main:create_app --no-access-log --port 7860
```

Then open <http://127.0.0.1:7860>. The frontend is served at `/` by the same
process.

On first start the embedding model is downloaded from the Hugging Face Hub,
which can take a while. `GET /ready` returns 200 once every component is ready,
and 503 until then.

---

## Configuration

Settings come from environment variables and, for local development, from
`.env` in the working directory. Every setting has a validated default, and
an invalid value stops the application at start-up.

### Required

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key. Without it the application starts, but `/ready` reports the key as not configured and analysis fails. |

### Optional

`.env.example` lists every optional setting with comments (a few as
commented-out lines, because changing them is rarely right). The main groups:

| Group | Variables |
|---|---|
| Models | `GROQ_MODEL_RESUME`, `GROQ_MODEL_SKILL_GAP`, `GROQ_MODEL_CRITIC`, `GROQ_MODEL_INTERVIEW`, `GROQ_MODEL_EVALUATOR`, `GROQ_MODEL_PROMPT_GUARD` |
| LLM calls | `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_REASONING_EFFORT`, `LLM_MAX_OUTPUT_TOKENS` |
| Prompt Guard | `PROMPT_GUARD_CHUNK_CHARS`, `PROMPT_GUARD_MAX_CHUNKS`, `PROMPT_GUARD_CONCURRENCY`, `PROMPT_GUARD_DISPLAY_THRESHOLD` |
| Reports and Critic | `REPORT_TTL_SECONDS`, `MAX_REVISIONS` |
| Upload limits | `MAX_UPLOAD_BYTES`, `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`, `MAX_JOB_DESCRIPTION_CHARS` |
| Interview | `MAX_ANSWER_CHARS` |
| Embeddings | `EMBEDDING_MODEL_NAME`, `EMBEDDING_MODEL_REVISION` |
| Rate limiting | `RATE_LIMIT_ENABLED`, `RATE_LIMIT_*`, `RATE_LIMIT_MAX_TRACKED_KEYS`, `ANALYZE_MAX_CONCURRENT` |
| Document processing | `PARSE_MAX_CONCURRENT`, `PARSE_DEADLINE_SECONDS` |

#### Defaults and allowed ranges (also in `.env.example`)

These settings are also listed in `.env.example`; `GROQ_API_BASE_URL`,
`CSP_FRAME_ANCESTORS` and `FRONTEND_DIR` appear there as commented-out lines.
Their defaults and allowed ranges are defined in `backend/core/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `API_DOCS_ENABLED` | `false` | Serves `/docs`, `/redoc` and `/openapi.json` (see below). |
| `MAX_STORED_REPORTS` | `500` | Maximum reports held in memory. |
| `ANALYSIS_DEADLINE_SECONDS` | `600` | Deadline for one analysis run. |
| `INTERVIEW_DEADLINE_SECONDS`, `EVALUATION_DEADLINE_SECONDS` | `120` | Deadlines for the interview endpoints. |
| `REPORT_SWEEP_INTERVAL_SECONDS` | `60` | How often expired reports are removed. |
| `GROQ_API_BASE_URL` | `https://api.groq.com/openai/v1` | Used by the upstream availability probe; must be `https://`. |
| `GROQ_PROBE_INTERVAL_SECONDS`, `GROQ_PROBE_TIMEOUT_SECONDS` | `300`, `5` | Upstream availability probe. |
| `FRONTEND_DIR` | `frontend/` in the repository | Directory served at `/`. |
| `CSP_FRAME_ANCESTORS` | `'self' https://huggingface.co` | CSP `frame-ancestors` sources. |
| `HSTS_ENABLED` | `false` | Sends `Strict-Transport-Security`. |

LangSmith and LangChain tracing is always forced off, because prompts contain
resume content.

### API documentation

`API_DOCS_ENABLED` is off by default. When it is off, `/docs`, `/redoc` and
`/openapi.json` are not served (they return 404). Setting
`API_DOCS_ENABLED=true` serves all three. Swagger UI and ReDoc load their
assets from a CDN, so `/docs` and `/redoc` are exempted from the Content
Security Policy while the setting is on. Keep it off in production.

---

## API

All `/api/` routes are rate limited (429 `rate_limited` with `Retry-After`)
and return errors in a common error envelope. Upload endpoints take
`multipart/form-data` with a `file` field (PDF or DOCX).

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness only; no dependency checks, no outbound calls. |
| `GET /ready` | Readiness of local components (200 or 503), plus cached Groq availability. |
| `POST /api/v1/parse` | Validate an upload and extract its text. Nothing is stored. |
| `POST /api/v1/match` | Match metrics, TF-IDF baseline and ATS status. Multipart (`file`, `job_description`) or JSON (`resume_text`, `job_description`). No LLM call; nothing is stored. |
| `POST /api/v1/analyze` | Run the full analysis workflow. `file`, optional `job_description`. Returns the report, including `report_id`. |
| `GET /api/v1/report/{report_id}` | The stored report as JSON. |
| `GET /api/v1/report/{report_id}/pdf` | The stored report rendered as a PDF. |
| `POST /api/v1/interview/questions` | Interview questions from a stored report (JSON). |
| `POST /api/v1/interview/evaluate` | Evaluate one answer (JSON). |

Unknown, malformed and expired report IDs all return the same 404
`report_not_found`.

Detailed documentation:

- [docs/interview-api.md](docs/interview-api.md): interview endpoints,
  request and response formats, configuration.
- [docs/pdf-reports.md](docs/pdf-reports.md): PDF report rendering and its
  limits.
- [docs/rate-limiting.md](docs/rate-limiting.md): rate limits, analysis
  concurrency, the single-process model and proxy considerations.

---

## Testing

Install the dev lock first (see [Install](#install)).

Offline suite (the default; live tests are deselected):

```bash
python -m pytest
```

Lint and formatting:

```bash
ruff check .
ruff format --check .
```

Live tests call the real Groq API and download the embedding model. They need
`GROQ_API_KEY` set in the environment (the live tests do not read `.env`) and
use your Groq quota:

```bash
python -m pytest -m live tests/live
```

The frontend behaviour tests run with Node.js and are skipped if it is not
installed. Set `FRONTEND_REQUIRE_NODE=1` to make a missing Node.js a failure
instead, as CI does.

CI (GitHub Actions, `.github/workflows/ci.yml`) runs lint, formatting and the
offline suite, checks that the locks match their `.in` files and that the
legacy application is unmodified. The live tests run only when started
manually.

---

## Project structure

```text
.
├── backend/
│   ├── main.py            # application factory (create_app)
│   ├── agents/            # resume, interview, skill-gap, Critic and evaluator agents
│   ├── api/               # routes, upload handling, rate limits
│   ├── core/              # settings, logging, errors, middleware
│   ├── graph/             # LangGraph analysis and evaluation workflows
│   ├── health/            # readiness and upstream availability
│   ├── llm/               # Groq clients, prompts, structured output
│   ├── parsing/           # PDF/DOCX validation, extraction, language check
│   ├── reports/           # PDF report rendering
│   ├── schemas/           # request and response models
│   ├── scoring/           # embeddings, match metrics, ATS status
│   ├── security/          # injection signals, sanitisation, output validation
│   ├── services/          # interview question service
│   ├── storage/           # in-memory report store
│   ├── assets/            # bundled fonts for PDF reports
│   └── data/              # skill aliases and expected resume sections
├── frontend/              # Vanilla JavaScript web interface
├── tests/                 # unit, integration, security and live tests
├── docs/                  # deployment, interview API, PDF reports, rate limiting
├── .github/workflows/     # CI
├── Dockerfile             # production image (also used by Hugging Face Spaces)
├── .dockerignore          # keeps secrets, environments and documents out of builds
├── requirements.in        # runtime top-level pins
├── requirements.txt       # hashed runtime lock
├── requirements-dev.in    # dev top-level pins
├── requirements-dev.txt   # hashed dev lock
├── pyproject.toml         # project metadata, Ruff and pytest configuration
├── .env.example           # configuration template (copy to .env)
├── app.py, agents/, utils/, requirements-legacy.txt   # legacy application (see below)
└── README.md
```

---

## Deployment

The repository includes a production `Dockerfile` for a single container,
used both for standalone Docker and for Hugging Face Spaces (Docker SDK). The
full guide, including configuration, security and open items, is
[docs/deployment.md](docs/deployment.md).

The image (linux/amd64) installs only the hashed runtime lock, runs as a
non-root user (UID 1000), copies only `backend/` and `frontend/`, and starts
one Uvicorn process on `0.0.0.0:7860`. Its health check uses `GET /health`.

Status:

- The Docker image has been built successfully, and the local Docker smoke
  tests passed (health, readiness, the web interface, analysis, interview
  practice and PDF reports).
- The Hugging Face Spaces configuration is prepared, but the application has
  not yet been deployed to a Space.

### Docker

```bash
docker build --platform linux/amd64 -t resume-intelligence:local .
docker run --rm -p 7860:7860 -e GROQ_API_KEY resume-intelligence:local
```

`-e GROQ_API_KEY` without a value passes the key from your shell's
environment at run time; the key is never part of the image. Then open
<http://127.0.0.1:7860>.

### Hugging Face Spaces

The YAML front matter at the top of this README configures a Docker Space
(`sdk: docker`, `app_port: 7860`). When the Space is created, set
`GROQ_API_KEY` as a **Space secret**; no other setting is required. See
[docs/deployment.md](docs/deployment.md#4-hugging-face-spaces-docker-sdk) for
the remaining open items, including client addresses for rate limiting behind
the Spaces proxy and how the bundled font files are pushed.

### Points to consider for any deployment

- The local development command listens on `127.0.0.1` only; the Docker
  image listens on `0.0.0.0`.
- Run a single process (see [Scope and limitations](#scope-and-limitations)).
- Provide `GROQ_API_KEY` as a secret, never in the image or repository.
- Client addresses for rate limiting depend on Uvicorn's proxy settings; see
  the proxy section of [docs/rate-limiting.md](docs/rate-limiting.md).
- Embedding model: in a local (non-Docker) run it is downloaded from the
  Hugging Face Hub on first start unless already cached. The Docker image
  downloads the pinned revision during the build and loads it offline at run
  time (`HF_HUB_OFFLINE=1`), so the container needs no Hub access.

---

## Legacy Streamlit application

`app.py`, `agents/` and `utils/` are the original Streamlit prototype. They
are kept unmodified for reference (CI checks this), are not used by the
FastAPI platform, and are excluded from linting.

Its dependencies are separate, unpinned and not part of the hashed locks.
Install them in a **separate** virtual environment:

```bash
python -m pip install -r requirements-legacy.txt
streamlit run app.py
```

Do not install `requirements-legacy.txt` into the platform's environment.

---

## Future improvements

- Calibrated ATS scoring
- Calibrated Prompt Guard threshold
- Hugging Face Space deployment (the Docker image and Space configuration
  are in place; see [Deployment](#deployment))

---

## Author

Zaiba Ansari

Aspiring Data Scientist | Machine Learning Enthusiast | AI Developer
