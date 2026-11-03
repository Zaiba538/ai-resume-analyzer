# PDF reports (M8)

The FastAPI backend can return a stored analysis report as a PDF. The PDF is
rendered on demand from the same `AnalysisReport` that
`GET /api/v1/report/{report_id}` returns as JSON. The JSON report remains the
machine-readable format.

## Endpoint

```
GET /api/v1/report/{report_id}/pdf
```

`report_id` is the `report_id` returned by `POST /api/v1/analyze`. Only GET is
allowed (other methods return 405).

A successful response is `200` with:

| Header | Value |
|---|---|
| `Content-Type` | `application/pdf` |
| `Content-Disposition` | `attachment; filename="resume-analysis-report.pdf"` |
| `Cache-Control` | `no-store` |
| `X-Robots-Tag` | `noindex, nofollow` |

The filename is fixed: it never contains the report ID or any report content.
The security headers (CSP, `nosniff`, `no-referrer` and the others) are the
same as on every other API route.

The PDF is built in memory for each request. It is never cached, written to
disk or stored with the report; when the report expires, the PDF can no longer
be produced.

## Contents

The PDF contains the resume profile, the interview questions (beginner,
intermediate and advanced), the job match (semantic similarity, skill
coverage, section completeness and the TF-IDF baseline), the ATS score status,
the skill gaps and recommendations, the quality review (Critic verdict,
revisions and issues) and the security notices.

* The ATS score is shown only as "Not calibrated"; no number is shown until its
  calibration is approved.
* Security notices show flag types only, each once, without scores or sources.
* Every page has a footer with the generation and expiry times (UTC), the text
  "AI-generated, unverified" and "Page n of N". The full disclaimer is shown
  under the title.
* The PDF never contains the raw resume text, the job description, the preview
  or the report ID. Its metadata is generic and its dates are fixed.
* Report content is treated as untrusted: it is escaped before rendering, so
  markup in it appears as literal text. The PDF has no links, images,
  JavaScript or external resources.

Section names and other wording come from `backend/reports/labels.py`.

## Errors

Errors use the standard error envelope
(`{"error": {"code", "message", "request_id"}}`) and carry the same
`no-store` and `noindex` headers.

| Status | Code | When |
|---:|---|---|
| 404 | `report_not_found` | The ID is malformed, unknown or expired (indistinguishable) |
| 422 | `pdf_page_limit_exceeded` | The PDF would need more than 100 pages; nothing is truncated |
| 503 | `pdf_renderer_unavailable` | The bundled fonts could not be registered |
| 504 | `pdf_render_timeout` | Rendering did not finish within 15 seconds |
| 500 | `internal_error` | Unexpected failure (the exception message is neither returned nor logged) |

The renderer's availability is checked before the ID is looked up, so while
the fonts are unavailable every request, whatever its ID, receives the same
503. No partial PDF is ever returned.

## Readiness

`GET /ready` includes `checks.pdf_renderer_ready`. The fonts are registered
once at start-up, and the check is true only if every bundled font file
matched its SHA-256 in `backend/assets/fonts/fonts-manifest.json`, parsed,
supported shaping (Noto Sans Devanagari) and was registered without being
shadowed by another font of the same name. If it is false, `/ready` returns
503 and the PDF endpoint fails closed with 503; the JSON report endpoint is
unaffected.

## Fonts

The fonts in `backend/assets/fonts/` are unmodified upstream files whose
provenance and SHA-256 are recorded in `fonts-manifest.json`:

* DejaVu Sans 2.37 (regular, bold) for Latin and other scripts
  (`LICENSE-DejaVu.txt`).
* Noto Sans Devanagari 2.007, unhinted (regular, bold) for Devanagari, shaped
  with HarfBuzz (`reportlab[shaping]`, `uharfbuzz`), under the SIL Open Font
  Licence 1.1 (`OFL.txt`).

## Limits and operational notes

* **Page limit:** 100 pages. A longer report fails with 422.
* **Deadline:** 15 seconds per request, including any wait for another render.
  Rendering runs in a worker thread (`asyncio.to_thread`), so it does not block
  the event loop.
* **Timeouts do not stop the worker thread.** Python cannot stop a running
  thread. When a request times out it returns 504 and the worker is abandoned:
  the renderer checks the same deadline before every page and every block of
  content, so the worker stops at its next check, and its result is discarded.
  An abandoned render keeps its worker thread and the render lock until then.
* **Renders are serialised.** ReportLab's shaping updates shared font data, so
  one lock allows one render at a time in the process. Concurrent requests
  queue, and their wait counts against their own deadline, so heavy concurrent
  use can produce 504s. The service runs as a single process; rate limiting is
  planned for M10.
* **Typical cost:** a typical report renders in well under a second. Devanagari
  text is roughly ten times slower to render than Latin text, so a very large
  Devanagari report can reach the 15-second deadline before the page limit,
  and slower hardware lowers the size that fits within the deadline.

## Known rendering limitations

* Shaped Devanagari may not copy, paste or search correctly in the PDF. The
  JSON report is the machine-readable alternative.
* In text that contains Devanagari, Latin words may be drawn with ligatures
  (for example "ffi"), which can affect copying from the PDF.
* A word that joins Latin letters directly to Devanagari (with no space) is
  drawn without Devanagari conjunct shaping.
* Control and private-use characters are shown as U+FFFD (the replacement
  character) rather than dropped. A few Vedic characters have no glyph in
  either bundled font.
* The same report produces identical bytes within one process. With
  Devanagari, the bytes (not the visible text or layout) can differ between
  processes.
