# Interview API (M9)

The interview API works on a stored analysis report, the one returned by
`POST /api/v1/analyze`. It has two endpoints:

* `POST /api/v1/interview/questions` generates interview questions for chosen
  difficulty levels from the report's resume profile.
* `POST /api/v1/interview/evaluate` evaluates one answer to one interview
  question.

Neither endpoint stores anything. Generated questions, answers and
evaluations are returned in the response and then discarded.

Contents:

* [Common rules](#common-rules)
* [POST /api/v1/interview/questions](#post-apiv1interviewquestions)
* [POST /api/v1/interview/evaluate](#post-apiv1interviewevaluate)
* [Error reference](#error-reference)
* [Security and privacy](#security-and-privacy)
* [Configuration](#configuration)
* [Running locally](#running-locally)
* [Deadlines and timeouts](#deadlines-and-timeouts)
* [Limitations](#limitations)

## Common rules

* **JSON only.** Send the body as `Content-Type: application/json` (a
  `charset` parameter is accepted). Any other content type is rejected with
  415 `unsupported_media_type`.
* **Strict request bodies.** Unknown fields are rejected, values are not
  coerced (`"3"`, `3.0` and `true` are not integers), and a JSON object must
  not contain the same key twice, at any nesting level. All of these are 422
  `validation_error`.
* **Report IDs.** `report_id` is the UUID4 returned by `/api/v1/analyze`. A
  malformed, unknown or expired ID always gives the same 404
  `report_not_found`, so the three cases cannot be told apart.
* **Reports expire.** Reports are held in server memory for
  `REPORT_TTL_SECONDS` (default and maximum 3600 seconds) after they are
  created, and are lost if the server restarts. After that, both endpoints
  return 404 `report_not_found` for the report.
* **Response headers.** Every response, including errors, carries
  `Cache-Control: no-store` and `X-Robots-Tag: noindex, nofollow`, plus the
  security headers used on every API route.
* **No authentication.** The application has no user accounts or API keys.
  Anyone who has a report ID can use it with these endpoints until the
  report expires, so treat report IDs as secrets.

## POST /api/v1/interview/questions

Generates interview questions from the resume profile in a stored report. The
profile was extracted, validated and reviewed when the report was created; the
endpoint does not re-read the resume. Only the requested levels are generated,
each with exactly `count_per_level` questions.

### Request

| Field | Type | Required | Rules |
|---|---|---|---|
| `report_id` | string | yes | The report's ID. |
| `levels` | array of strings | no | One to three distinct values from `beginner`, `intermediate`, `advanced`. Default: all three. Repeated values are rejected. |
| `count_per_level` | integer | no | 1 to 5. Default: 3. |

Levels can be given in any order; the response lists them in the order
beginner, intermediate, advanced.

```http
POST /api/v1/interview/questions
Content-Type: application/json

{
  "report_id": "3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10",
  "levels": ["advanced", "beginner"],
  "count_per_level": 2
}
```

### Response (200)

| Field | Type | Meaning |
|---|---|---|
| `levels` | array of strings | The requested levels, in canonical order. |
| `count_per_level` | integer | Questions per requested level. |
| `questions` | object | `beginner`, `intermediate` and `advanced` lists. Levels that were not requested are empty. |
| `security_flags` | array | Security flags for the generated output (see [Security flags](#prompt-injection-detection-and-security-flags)). |

Each question has an `id`, `question` (5 to 500 characters), `topic` (1 to
100 characters) and `rationale` (up to 300 characters). IDs are assigned by
the server: the level, a hyphen and a 1-based position, such as
`advanced-2`.

```json
{
  "levels": ["beginner", "advanced"],
  "count_per_level": 2,
  "questions": {
    "beginner": [
      {
        "question": "What is the difference between a list and a tuple in Python?",
        "topic": "Python",
        "rationale": "Python is listed as a core skill.",
        "id": "beginner-1"
      },
      {
        "question": "How do you read a CSV file with pandas?",
        "topic": "pandas",
        "rationale": "The profile mentions data processing.",
        "id": "beginner-2"
      }
    ],
    "intermediate": [],
    "advanced": [
      {
        "question": "How would you make a Spark ETL job idempotent?",
        "topic": "Apache Spark",
        "rationale": "The candidate built batch ETL pipelines in Spark.",
        "id": "advanced-1"
      },
      {
        "question": "How would you size partitions for a skewed Spark join?",
        "topic": "Apache Spark",
        "rationale": "Spark performance is central to the listed experience.",
        "id": "advanced-2"
      }
    ]
  },
  "security_flags": []
}
```

These questions are not stored. Their IDs number them within this response
only: `advanced-1` here is not the report's own `advanced-1`. To evaluate an
answer to one of them, send the whole question object (see below).

### Errors

404 `report_not_found`, 415 `unsupported_media_type`, 422 `validation_error`,
502, 503 and 504 `interview_timeout`. See [Error reference](#error-reference).

## POST /api/v1/interview/evaluate

Evaluates one answer to one interview question and returns a rubric with
ratings and written feedback.

### Workflow

1. The report is looked up (404 `report_not_found` if it is missing or
   expired).
2. The question is resolved from exactly one source (below).
3. The answer is sanitised and its length checked.
4. The answer and a client-supplied question are scanned for prompt
   injection. Scanning only raises flags; it never rejects a request.
5. The answer is evaluated by a language model against five fixed criteria.
   The server computes the overall rating.

### Request

| Field | Type | Required | Rules |
|---|---|---|---|
| `report_id` | string | yes | The report's ID. |
| `question_id` | string | one of these two | The ID of a question in the stored report, such as `beginner-1`. |
| `question` | object | one of these two | A client-supplied question: `id`, `question`, `topic` and `rationale`, all required. |
| `answer` | string | yes | The candidate's answer. See [Answer limits](#answer-limits). |

Provide **exactly one** question source. Sending both `question_id` and
`question`, or neither, is a 422 `validation_error`.

* **`question_id`**: the question must exist in the stored report's
  interview questions, otherwise 404 `question_not_found`. The stored text is
  used.
* **`question`**: for questions that are not in the report, for example
  questions returned by `/interview/questions`. The object has the same
  fields and limits as a generated question (see
  [Input limits](#input-limits)). Only the question text is sent to the
  model; `topic` and `rationale` are validated but not used for the
  evaluation.

**Level.** The difficulty level is taken from the validated question ID (the
part before the hyphen): `intermediate-2` is evaluated as intermediate. It
cannot be supplied separately; a `level` field is rejected as unknown. A
question ID must match `^(beginner|intermediate|advanced)-([1-9]|10)$`.

```http
POST /api/v1/interview/evaluate
Content-Type: application/json

{
  "report_id": "3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10",
  "question_id": "beginner-1",
  "answer": "A list is an ordered, mutable sequence; a tuple is immutable."
}
```

```http
POST /api/v1/interview/evaluate
Content-Type: application/json

{
  "report_id": "3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10",
  "question": {
    "id": "advanced-2",
    "question": "How would you size partitions for a skewed Spark join?",
    "topic": "Apache Spark",
    "rationale": "Spark performance is central to the listed experience."
  },
  "answer": "I would salt the skewed keys and check the partition sizes in the Spark UI."
}
```

### Answer limits

* The raw `answer` may be at most **40,000 characters**. Longer input is
  rejected with 422 `validation_error` before it is sanitised.
* The answer is then sanitised (see [Sanitisation](#sanitisation)) and
  surrounding whitespace is removed. What remains must not be empty (422
  `empty_answer`) and must be at most `MAX_ANSWER_CHARS` characters (default
  5,000; 422 `answer_too_long`).

### Evaluation criteria

The model rates the answer on five fixed criteria, each with an integer from
1 to 5 and a one-sentence justification (3 to 300 characters, a single line):

| Criterion | Question it answers |
|---|---|
| `relevance` | Does the answer address the question asked? |
| `technical_accuracy` | Are the statements technically correct? |
| `depth` | How thorough and insightful is the answer? |
| `clarity` | How clearly is it expressed? |
| `structure` | How logically is it organised? |

The model is told to judge the answer against what the question's level
requires, and to use these anchors for every criterion:

| Rating | Anchor |
|---|---|
| 1 | Incorrect or largely irrelevant. |
| 2 | Major gaps or inaccuracies. |
| 3 | Partially correct, with noticeable gaps. |
| 4 | Correct and sufficiently detailed. |
| 5 | Accurate, comprehensive and well-structured. |

It also returns up to 5 `strengths`, up to 5 `improvements` and a
`model_answer_outline` of up to 8 short points (each item 1 to 300
characters). An empty, evasive or off-topic answer is to be rated low.

**Overall rating.** The model does not produce an overall score. The server
computes `overall_rating` as the arithmetic mean of the five ratings, rounded
half up to one decimal place: ratings 4, 3, 3, 4 and 4 give 18 / 5 = 3.6. The
response is rejected if the model supplies an overall rating of its own.

The ratings are AI-generated and unverified. They are not calibrated against
human graders, and they are unrelated to the ATS score, which remains
uncalibrated and is never reported as a number.

### Response (200)

| Field | Type | Meaning |
|---|---|---|
| `question_id` | string | The ID of the evaluated question. |
| `level` | string | The level derived from that ID. |
| `evaluation` | object | `rubric` (the five criteria), `strengths`, `improvements`, `model_answer_outline` and `overall_rating`. |
| `security_flags` | array | Flags for the answer, a client-supplied question and the model output. |

The response never contains the answer or the question text.

```json
{
  "question_id": "beginner-1",
  "level": "beginner",
  "evaluation": {
    "rubric": {
      "relevance": {"rating": 4, "justification": "The answer addresses the difference asked about."},
      "technical_accuracy": {"rating": 3, "justification": "Mostly correct but omits hashability."},
      "depth": {"rating": 3, "justification": "Covers the main point without examples."},
      "clarity": {"rating": 4, "justification": "Short and easy to follow."},
      "structure": {"rating": 4, "justification": "States each type's property in turn."}
    },
    "strengths": ["Correct core distinction between mutability and immutability"],
    "improvements": ["Mention that tuples can be dictionary keys"],
    "model_answer_outline": ["Define both types", "Contrast mutability", "Give a use case for each"],
    "overall_rating": 3.6
  },
  "security_flags": []
}
```

### Errors

404 `report_not_found` and `question_not_found`, 415
`unsupported_media_type`, 422 `validation_error`, `empty_answer` and
`answer_too_long`, 502, 503 and 504 `evaluation_timeout`. See
[Error reference](#error-reference).

## Error reference

Every error uses the same envelope. `details` appears only on 422
`validation_error`.

```json
{
  "error": {
    "code": "question_not_found",
    "message": "The question was not found in the report.",
    "request_id": "29ce56af6b584e55a6e91d65c4e07894"
  }
}
```

`request_id` matches the `X-Request-ID` response header.

| Status | Code | Endpoints | Cause |
|---:|---|---|---|
| 404 | `report_not_found` | both | The report ID is malformed, unknown or expired. |
| 404 | `question_not_found` | evaluate | `question_id` is not a question in the report. |
| 405 | `method_not_allowed` | both | A method other than POST. |
| 413 | `payload_too_large` | both | The request body exceeds the server-wide body size limit. |
| 415 | `unsupported_media_type` | both | The body is not `application/json`. |
| 422 | `validation_error` | both | Malformed JSON, a duplicate key, an unknown field, a wrong type or a value outside its limits. |
| 422 | `empty_answer` | evaluate | Nothing is left of the answer after sanitisation. |
| 422 | `answer_too_long` | evaluate | The sanitised answer exceeds `MAX_ANSWER_CHARS`. |
| 502 | `upstream_error` | both | The language-model service returned an error. |
| 502 | `upstream_auth_failed` | both | The language-model service rejected the server's credentials. |
| 502 | `invalid_llm_output` | both | The model's output was invalid, even after one repair attempt. |
| 503 | `llm_not_configured` | both | No `GROQ_API_KEY` is configured. |
| 503 | `upstream_rate_limited` | both | The language-model service is rate limiting. A `Retry-After` header (seconds) is passed on when the service sends one. |
| 504 | `upstream_timeout` | both | A model call timed out after its retries. |
| 504 | `interview_timeout` | questions | The request exceeded `INTERVIEW_DEADLINE_SECONDS`. |
| 504 | `evaluation_timeout` | evaluate | The request exceeded `EVALUATION_DEADLINE_SECONDS`. |
| 500 | `internal_error` | both | An unexpected error. |

Validation errors list each problem as `loc`, `type` and `msg` only. Submitted
values are never included. The location may name an unknown field, as on
every endpoint of this API. Examples:

```json
{"loc": ["body"], "type": "duplicate_key", "msg": "JSON objects must not contain duplicate keys."}
{"loc": ["body"], "type": "json_invalid", "msg": "JSON decode error"}
{"loc": ["body"], "type": "value_error", "msg": "Value error, provide exactly one of question_id or question"}
{"loc": ["body", "question"], "type": "string_too_long", "msg": "Field topic should have at most 200 characters"}
```

## Security and privacy

### Sanitisation

All untrusted text is normalised before use: line endings and Unicode (NFKC)
are normalised, and control characters, bidirectional controls, zero-width and
other invisible characters, tag characters and excessive combining marks are
removed. This applies to the answer, to every field of a client-supplied
question, and to the stored resume profile, which is sanitised again before
it is sent to the question generator. Model output is sanitised the same way,
and links in it that do not appear in its input are replaced with
`[link removed]`.

### Input limits

Raw limits are checked on the input as received, before sanitisation.
Post-sanitisation limits are checked on the text after sanitisation and
removal of surrounding whitespace.

| Field | Raw limit (before sanitisation) | Limit after sanitisation |
|---|---|---|
| `question.question` | 1,000 characters | 5 to 500 characters |
| `question.topic` | 200 characters | 1 to 100 characters |
| `question.rationale` | 600 characters | up to 300 characters |
| `answer` | 40,000 characters | not empty, at most `MAX_ANSWER_CHARS` (default 5,000) |
| `question.id`, `question_id` | not sanitised; must match the ID pattern exactly | (same) |

A raw value over its limit is rejected without being sanitised.
The raw limits leave room for characters that sanitisation removes, so text
such as an answer containing invisible characters can still be accepted if
what remains is within the post-sanitisation limit. Question IDs are never
normalised: a full-width digit or surrounding space makes the ID invalid.

The whole request body is also limited by the server-wide body size limit
(`MAX_UPLOAD_BYTES` + 4 x `MAX_JOB_DESCRIPTION_CHARS` + 64 KiB, about 5.4 MB
with the defaults); larger bodies are rejected with 413.

### Duplicate JSON keys

A JSON object that contains the same key twice is rejected with 422
`validation_error` (`type: duplicate_key`), whether the duplicate is at the
top level, inside the `question` object or deeper. Keys are compared after
JSON decoding, so `"answer"` and `"answer"` count as the same key.
Without this rule, different parsers could read different values from the
same body.

### Hidden Unicode

If sanitisation removes suspicious invisible characters (bidirectional
controls, zero-width characters, tag characters, variation selectors from the
supplement block, or excessive combining marks), a `hidden_unicode` flag is
raised. For a client-supplied question, the check is made on the raw fields
before they are sanitised and is reported with source `question`. Only the
fact that such characters were present is kept, never the raw text. A byte
order mark or soft hyphen is removed without a flag.

### Prompt injection detection and security flags

Injection detection supplements the real defences (data kept in delimited,
escaped blocks separate from the instructions, models with no tools, strict
output schemas and output validation). It never rejects a request; it only
raises flags.

* **Heuristics** match known phrasings such as instructions to ignore
  previous instructions, role impersonation, requests to reveal the prompt,
  attempts to influence the score and fake delimiters. Coverage is mainly
  English, with a few Hindi and Marathi phrases.
* **Prompt Guard** (`GROQ_MODEL_PROMPT_GUARD`) scores the answer in chunks of
  `PROMPT_GUARD_CHUNK_CHARS` characters, at most `PROMPT_GUARD_MAX_CHUNKS`
  chunks per answer, with at most `PROMPT_GUARD_CONCURRENCY` requests at a
  time. A `classifier` flag is raised when the highest score reaches
  `PROMPT_GUARD_DISPLAY_THRESHOLD`. The score is not calibrated.
* **Fallback.** Prompt Guard has 30 seconds per evaluation. If it has not
  finished by then, it is cancelled and the answer is scanned with the
  heuristics only; the evaluation continues. Prompt Guard is skipped when no
  API key is configured. Whether it ran fully, partially or not at all is
  logged, not returned.
* **What is scanned.** The answer: heuristics and Prompt Guard. A
  client-supplied question: heuristics and the hidden-Unicode check only. A
  stored question is not scanned again.

Each flag has this shape. Flags never contain the matched text.

```json
{"type": "instruction_override", "source": "answer", "score": null, "calibrated": false}
```

| Field | Values |
|---|---|
| `type` | `hidden_unicode`, `instruction_override`, `role_impersonation`, `prompt_exfiltration`, `score_manipulation`, `delimiter_injection`, `classifier`, `unexpected_url` |
| `source` | `answer` and `question` (inputs), `llm_output` (a link removed from model output) |
| `score` | The Prompt Guard score for `classifier` flags, otherwise `null` |
| `calibrated` | Always `false` |

Each flag type appears at most once per source.

### No persistence

The report store is only read. Generated questions, answers, client-supplied
questions and evaluations are never written to it or anywhere else, and the
evaluation workflow keeps no checkpoints. Reports themselves live only in
server memory, at most `MAX_STORED_REPORTS` at a time, and are removed when
they expire.

### Logging

Logs are JSON lines built from a fixed list of allowed fields. They record
event names, timings, model names, attempt counts, flag types and the route
template (`/api/v1/interview/evaluate`), never the answer, question text,
resume content, report ID, submitted values or the API key. Error responses
use fixed messages and never contain submitted content.

## Configuration

These settings affect the interview API. All are environment variables (or
entries in a local `.env` file); see `.env.example`.

| Variable | Default | Allowed | Meaning |
|---|---|---|---|
| `GROQ_API_KEY` | none | | Required for both endpoints; without it they return 503 `llm_not_configured`. |
| `GROQ_MODEL_INTERVIEW` | `openai/gpt-oss-20b` | | Model that generates questions. |
| `GROQ_MODEL_EVALUATOR` | `openai/gpt-oss-20b` | | Model that evaluates answers. |
| `GROQ_MODEL_PROMPT_GUARD` | `meta-llama/llama-prompt-guard-2-86m` | | Prompt Guard classifier. |
| `MAX_ANSWER_CHARS` | `5000` | 200 to 20000 | Maximum sanitised answer length. |
| `INTERVIEW_DEADLINE_SECONDS` | `120` | 1 to 600 | Deadline for `/interview/questions`. |
| `EVALUATION_DEADLINE_SECONDS` | `120` | 1 to 600 | Deadline for `/interview/evaluate`. |
| `LLM_TIMEOUT_SECONDS` | `30` | above 0, up to 120 | Timeout of each model call attempt. |
| `LLM_MAX_RETRIES` | `2` | 0 to 5 | Retries for failed model calls. |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` | 256 to 32768 | Output-token cap of each model call. |
| `LLM_REASONING_EFFORT` | `low` | `low`, `medium`, `high` | Reasoning effort, sent only to `openai/gpt-oss` models (the defaults for both endpoints). |
| `PROMPT_GUARD_CHUNK_CHARS` | `800` | 200 to 2000 | Characters per Prompt Guard chunk. |
| `PROMPT_GUARD_MAX_CHUNKS` | `16` | 1 to 64 | Maximum chunks scanned per answer. |
| `PROMPT_GUARD_CONCURRENCY` | `4` | 1 to 16 | Concurrent Prompt Guard requests. |
| `PROMPT_GUARD_DISPLAY_THRESHOLD` | `0.9` | above 0, below 1 | Score at which a `classifier` flag is raised. |
| `REPORT_TTL_SECONDS` | `3600` | 60 to 3600 | How long reports remain available. |
| `REPORT_SWEEP_INTERVAL_SECONDS` | `60` | 1 to 3600 | How often expired reports are purged (they are also purged when looked up). |
| `MAX_STORED_REPORTS` | `500` | 1 to 10000 | Maximum reports held in memory. |
| `MAX_UPLOAD_BYTES` | `5242880` | 1024 to 20971520 | Upload limit; here it only sets part of the request-body limit. |
| `MAX_JOB_DESCRIPTION_CHARS` | `20000` | 500 to 100000 | Job-description limit; here it only sets part of the request-body limit. |
| `API_DOCS_ENABLED` | `false` | | Serves `/docs`, `/redoc` and `/openapi.json`. |

## Running locally

The backend requires Python 3.12.

```bash
python -m pip install --require-hashes -r requirements.txt
cp .env.example .env          # then set GROQ_API_KEY in .env
uvicorn --factory backend.main:create_app --no-access-log --port 7860
```

The sentence-embedding model used by `/api/v1/analyze` is downloaded on first
start. `GET /ready` reports whether every component is ready. Create a report
with `POST /api/v1/analyze`, then use its `report_id` with the interview
endpoints.

**API documentation.** Set `API_DOCS_ENABLED=true` to serve the OpenAPI
document at `/openapi.json`, Swagger UI at `/docs` and ReDoc at `/redoc`. They
are disabled by default and then return 404. The evaluate request body is
documented with both question sources and an example of each.

## Deadlines and timeouts

Each endpoint is bounded by its deadline (`INTERVIEW_DEADLINE_SECONDS`,
`EVALUATION_DEADLINE_SECONDS`, both 120 seconds by default), which includes
the report lookup, Prompt Guard and every model call with its retries. When
the deadline passes, outstanding model requests are cancelled and the request
fails with 504 `interview_timeout` or `evaluation_timeout`; no partial result
is returned.

Cancellation is local: the server stops waiting and closes the connection,
but the model provider may still finish a request it has already received.

Each model call is attempted up
to `LLM_MAX_RETRIES` + 1 times with a `LLM_TIMEOUT_SECONDS` timeout, and an
invalid model response is retried once, so a degraded upstream service can
use the whole deadline and end in a 504.

## Limitations

* **No authentication**, and **no rate limiting** yet (planned for M10). Each
  evaluation makes one evaluator call and up to `PROMPT_GUARD_MAX_CHUNKS`
  Prompt Guard calls; each question request makes one model call.
* **Detection is not prevention.** Injection flags are informational, and the
  Prompt Guard score is uncalibrated. An answer that tries to manipulate its
  own rating is flagged but still evaluated.
* **Evaluations are AI-generated and unverified.** Feedback may restate parts
  of the answer; it is returned only to the caller.
* **Long answers may be scanned partially.** With the default settings a
  5,000-character answer fits within the Prompt Guard chunk limit; if
  `MAX_ANSWER_CHARS` or `PROMPT_GUARD_CHUNK_CHARS` is changed, some chunks of a
  long answer may not be scanned. Heuristics always cover the whole answer.
* **Generated questions are not kept.** To evaluate an answer to a question
  from `/interview/questions`, the client must send the question object.
* **Reports are in memory only** and disappear on expiry or restart.
