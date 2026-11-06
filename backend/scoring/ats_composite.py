"""ATS composite gate: disabled in M5.

Approved Rev. 3, correction 1: an ATS score may only be shown once evaluation
against acceptance criteria agreed with the project owner demonstrates
acceptable performance, and the result is explicitly approved. Calibration,
acceptance criteria and approval are reserved for M12.

Until then the gate has exactly one possible result. It deliberately reads no
files, settings, environment variables or metrics, so nothing present on disk
(for example a stray ``calibration.json``) can enable a score, and it never
combines the three metrics. The ``AtsScore`` model itself only admits the
``not_calibrated`` state with every other field null.
"""

from __future__ import annotations

from backend.schemas.match import AtsScore

_NOT_CALIBRATED = AtsScore()


def evaluate_ats_gate() -> AtsScore:
    """The ATS composite status. Always ``not_calibrated`` with a null score in M5."""
    return _NOT_CALIBRATED
