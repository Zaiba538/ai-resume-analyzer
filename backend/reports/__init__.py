"""PDF report generation (approved Rev. 3, M8).

Reports are rendered on demand, in memory, from a stored ``AnalysisReport``:
derived data only, never raw resume text, the job description or the preview.
``labels`` holds the canonical display labels (shared with the M11 frontend)
and ``fonts`` the process-wide font registry and script-run splitting.
"""
