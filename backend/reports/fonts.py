"""Font registry and script-run splitting for PDF reports (M8, decisions D1, D7).

Fonts: DejaVu Sans (regular, bold) for all text except Devanagari, and Noto
Sans Devanagari (regular, bold) with HarfBuzz shaping for Devanagari.

Registration happens once per process (ReportLab's font registry is global).
Every ``FontRegistry`` registers under one module-level lock, so instances
never register concurrently, and each caches its outcome: ``register()`` is
thread-safe and idempotent, and a failure is not retried. Before anything is
registered, every font file is read, checked against the SHA-256 in
``fonts-manifest.json`` and parsed, and the Noto faces must support shaping.

ReportLab keeps the first font registered under a name and silently ignores
later ones. After registering, each name must therefore resolve to the verified
face (a ``TTFont`` with the same PostScript name, revision and glyph count, and
every original glyph-map and width entry unchanged) and each family to its
regular and bold names; a font registered earlier under one of these names
makes registration fail. The only tolerated difference is extra entries for
codepoints the verified face does not map: ReportLab's HarfBuzz shaping adds
private-use mappings to the registered font when it renders conjuncts.

Any failure (missing or altered file, invalid manifest or font, shaping
unavailable, a shadowed name, or an unexpected error) leaves the registry not
ready with a fixed ``error_code``; no exception escapes. No font is registered
unless every face passed its file checks. If registration itself fails part
way, ReportLab may keep the fonts registered before the failure; they are not
rolled back (that would need private ReportLab APIs), but they are verified
faces and the registry stays not ready, so the renderer never uses them.
The renderer is ready only when registration succeeded.

``split_script_runs`` splits text into runs for one font each. Characters in
the Devanagari, Devanagari Extended and Vedic Extensions blocks use the
Devanagari font; everything else uses the default font. Combining marks, ZWJ,
ZWNJ and spaces belong to the run of the preceding character (leading ones to
the following run), so conjuncts and joiner sequences are never split across
fonts. Shaped Devanagari may not copy or search correctly in the PDF (accepted
limitation, D1); the JSON report is the machine-readable alternative.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reportlab.lib.fonts import tt2ps
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger("backend.reports.fonts")

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
MANIFEST_NAME = "fonts-manifest.json"

FontRole = Literal["default", "devanagari"]
FontErrorCode = Literal[
    "manifest_invalid",
    "font_missing",
    "font_hash_mismatch",
    "font_invalid",
    "shaping_unavailable",
    "registration_failed",
]


@dataclass(frozen=True, slots=True)
class FontFace:
    name: str
    file: str
    role: FontRole
    bold: bool


FACES: tuple[FontFace, ...] = (
    FontFace("DejaVuSans", "DejaVuSans.ttf", "default", bold=False),
    FontFace("DejaVuSans-Bold", "DejaVuSans-Bold.ttf", "default", bold=True),
    FontFace("NotoSansDevanagari", "NotoSansDevanagari-Regular.ttf", "devanagari", bold=False),
    FontFace("NotoSansDevanagari-Bold", "NotoSansDevanagari-Bold.ttf", "devanagari", bold=True),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# ReportLab's font registry is process-global, so registration is too.
_REGISTRATION_LOCK = threading.Lock()


def font_name(role: FontRole, *, bold: bool = False) -> str:
    """The registered ReportLab font name for a role and weight."""
    return next(face.name for face in FACES if face.role == role and face.bold == bold)


class _FontAssetError(Exception):
    def __init__(self, code: FontErrorCode) -> None:
        super().__init__(code)
        self.code: FontErrorCode = code


class FontRegistry:
    """Registers the report fonts with ReportLab once and records readiness."""

    def __init__(self, font_dir: Path = FONT_DIR) -> None:
        self._font_dir = font_dir
        self._attempted = False
        self._error: FontErrorCode | None = None

    def register(self) -> bool:
        """Register all fonts on the first call; return whether they are ready."""
        with _REGISTRATION_LOCK:
            if not self._attempted:
                self._attempted = True
                self._error = self._register()
                if self._error is not None:
                    logger.error(
                        "pdf_fonts_unavailable",
                        extra={"component": "pdf_fonts", "error_code": self._error},
                    )
            return self._error is None

    @property
    def ready(self) -> bool:
        with _REGISTRATION_LOCK:
            return self._attempted and self._error is None

    @property
    def error_code(self) -> FontErrorCode | None:
        with _REGISTRATION_LOCK:
            return self._error

    def _register(self) -> FontErrorCode | None:
        try:
            fonts = self._load_fonts()
        except _FontAssetError as exc:
            return exc.code
        except Exception:
            return "registration_failed"
        try:
            for font in fonts:
                pdfmetrics.registerFont(font)
            for role in ("default", "devanagari"):
                regular, bold = font_name(role), font_name(role, bold=True)
                pdfmetrics.registerFontFamily(
                    regular, normal=regular, bold=bold, italic=regular, boldItalic=bold
                )
            if not (all(_resolves_to(font) for font in fonts) and _families_resolve()):
                return "registration_failed"  # a font was registered earlier under our name
        except Exception:
            return "registration_failed"
        return None

    def _load_fonts(self) -> list[TTFont]:
        """Read, verify and parse every face; nothing is registered here."""
        expected = self._expected_hashes()
        fonts = []
        for face in FACES:
            try:
                data = (self._font_dir / face.file).read_bytes()
            except OSError:
                raise _FontAssetError("font_missing") from None
            if hashlib.sha256(data).hexdigest() != expected.get(face.file):
                raise _FontAssetError("font_hash_mismatch")
            try:
                font = TTFont(face.name, io.BytesIO(data))
            except Exception:
                raise _FontAssetError("font_invalid") from None
            if face.role == "devanagari" and not font.shapable:
                raise _FontAssetError("shaping_unavailable")
            fonts.append(font)
        return fonts

    def _expected_hashes(self) -> dict[str, str]:
        try:
            manifest = json.loads((self._font_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
            hashes = {entry["file"]: entry["sha256"] for entry in manifest["fonts"]}
        except (OSError, ValueError, KeyError, TypeError):
            raise _FontAssetError("manifest_invalid") from None
        if not all(isinstance(h, str) and _SHA256.fullmatch(h) for h in hashes.values()):
            raise _FontAssetError("manifest_invalid")
        return hashes


_FACE_IDENTITY = ("name", "fontRevision", "numGlyphs")
# Compared entry by entry: HarfBuzz shaping in ReportLab adds private-use
# mappings for glyphs without a codepoint (for example conjuncts) to these
# tables of the registered font, so only those additions are tolerated.
_FACE_TABLES = ("charToGlyph", "charWidths")


def _resolves_to(font: TTFont) -> bool:
    """Whether ReportLab's font for ``font.fontName`` is this verified face."""
    registered = pdfmetrics.getFont(font.fontName)
    if registered is font:
        return True
    if not isinstance(registered, TTFont):
        return False
    theirs, ours = registered.face, font.face
    return all(
        getattr(theirs, attr, None) == getattr(ours, attr) for attr in _FACE_IDENTITY
    ) and all(
        _extends(getattr(theirs, table, None), getattr(ours, table)) for table in _FACE_TABLES
    )


def _extends(theirs: object, ours: dict[int, object]) -> bool:
    """Every entry of ``ours`` is in ``theirs`` unchanged; other codepoints may be added."""
    return isinstance(theirs, dict) and all(
        code in theirs and theirs[code] == value for code, value in ours.items()
    )


def _families_resolve() -> bool:
    """Whether bold and italic variants map to the registered face names."""
    for role in ("default", "devanagari"):
        regular, bold = font_name(role), font_name(role, bold=True)
        expected = {(0, 0): regular, (1, 0): bold, (0, 1): regular, (1, 1): bold}
        if any(tt2ps(regular, b, i) != name for (b, i), name in expected.items()):
            return False
    return True


_default_registry = FontRegistry()


def default_registry() -> FontRegistry:
    """The process-wide registry for the bundled fonts."""
    return _default_registry


# --- Script runs --------------------------------------------------------------

DEVANAGARI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari
    (0xA8E0, 0xA8FF),  # Devanagari Extended
    (0x1CD0, 0x1CFF),  # Vedic Extensions
)
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
_INHERITING = frozenset({ZWNJ, ZWJ, " "})
_COMBINING_CATEGORIES = frozenset({"Mn", "Mc", "Me"})


@dataclass(frozen=True, slots=True)
class ScriptRun:
    role: FontRole
    text: str

    def font_name(self, *, bold: bool = False) -> str:
        return font_name(self.role, bold=bold)


def is_devanagari(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in DEVANAGARI_RANGES)


def _role(char: str) -> FontRole | None:
    """The run a character starts or continues; None means it inherits."""
    if is_devanagari(char):
        return "devanagari"
    if char in _INHERITING or unicodedata.category(char) in _COMBINING_CATEGORIES:
        return None
    return "default"


def split_script_runs(text: str) -> tuple[ScriptRun, ...]:
    """Split ``text`` into maximal single-font runs; their texts join to ``text``."""
    runs: list[tuple[FontRole, list[str]]] = []
    leading: list[str] = []
    for char in text:
        role = _role(char)
        if role is None:
            (runs[-1][1] if runs else leading).append(char)
        elif runs and runs[-1][0] == role:
            runs[-1][1].append(char)
        else:
            runs.append((role, [*leading, char]))
            leading = []
    if leading:  # the text has no character that starts a run
        runs.append(("default", leading))
    return tuple(ScriptRun(role, "".join(chars)) for role, chars in runs)
