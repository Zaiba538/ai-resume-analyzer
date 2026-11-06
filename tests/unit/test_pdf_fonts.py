"""M8 Phase 2: font registry (once, thread-safe, fail-safe) and script-run splitting."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from reportlab.lib.fonts import tt2ps
from reportlab.pdfbase import pdfmetrics, ttfonts

from backend.reports import fonts
from backend.reports.fonts import (
    FACES,
    ZWJ,
    ZWNJ,
    FontRegistry,
    ScriptRun,
    default_registry,
    font_name,
    is_devanagari,
    split_script_runs,
)

VIRAMA = chr(0x094D)
NUKTA = chr(0x093C)
ANUSVARA = chr(0x0902)
VEDIC_TONE = chr(0x1CD0)  # Vedic Extensions block
DEVA_EXTENDED = chr(0xA8F2)  # Devanagari Extended block (spacing candrabindu)
COMBINING_ACUTE = chr(0x0301)


def _roles(text: str) -> list[tuple[str, str]]:
    return [(run.role, run.text) for run in split_script_runs(text)]


@pytest.fixture
def font_dir(tmp_path: Path) -> Path:
    """A private copy of the bundled fonts and manifest that a test may damage."""
    target = tmp_path / "fonts"
    shutil.copytree(fonts.FONT_DIR, target)
    return target


# --- Font names ---------------------------------------------------------------


def test_font_names_per_role_and_weight() -> None:
    assert font_name("default") == "DejaVuSans"
    assert font_name("default", bold=True) == "DejaVuSans-Bold"
    assert font_name("devanagari") == "NotoSansDevanagari"
    assert font_name("devanagari", bold=True) == "NotoSansDevanagari-Bold"
    assert {face.file for face in FACES} == {
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "NotoSansDevanagari-Regular.ttf",
        "NotoSansDevanagari-Bold.ttf",
    }


# --- Registration ---------------------------------------------------------------


def test_bundled_fonts_register_and_are_ready() -> None:
    registry = FontRegistry()
    assert not registry.ready  # nothing is ready before registration
    assert registry.register() is True
    assert registry.ready
    assert registry.error_code is None
    for face in FACES:
        assert pdfmetrics.getFont(face.name).face.name  # registered with ReportLab
    assert pdfmetrics.getFont("NotoSansDevanagari").shapable
    assert pdfmetrics.getFont("NotoSansDevanagari-Bold").shapable
    assert tt2ps("DejaVuSans", 1, 0) == "DejaVuSans-Bold"
    assert tt2ps("NotoSansDevanagari", 1, 0) == "NotoSansDevanagari-Bold"


def test_default_registry_is_a_process_singleton() -> None:
    assert default_registry() is default_registry()
    assert default_registry().register() is True


def test_registration_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = pdfmetrics.registerFont
    monkeypatch.setattr(pdfmetrics, "registerFont", lambda f: (calls.append(f.fontName), real(f)))
    registry = FontRegistry()
    assert [registry.register() for _ in range(5)] == [True] * 5
    assert sorted(calls) == sorted(face.name for face in FACES)


def _count_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every TTFont construction without replacing the TTFont type.

    The registry type-checks registered fonts with ``isinstance(..., TTFont)``,
    and ReportLab keeps fonts registered earlier in the session, so tests wrap
    ``TTFont.__init__`` instead of substituting a function or subclass.
    """
    loads: list[str] = []
    real_init = ttfonts.TTFont.__init__

    def counting_init(self, name, *args, **kwargs):
        loads.append(name)
        real_init(self, name, *args, **kwargs)

    monkeypatch.setattr(ttfonts.TTFont, "__init__", counting_init)
    return loads


def test_concurrent_registration_loads_and_registers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _count_loads(monkeypatch)
    registry = FontRegistry()
    barrier = threading.Barrier(16)
    results: list[bool] = []

    def worker() -> None:
        barrier.wait()
        results.append(registry.register())

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [True] * 16
    assert sorted(loads) == sorted(face.name for face in FACES)


def _fail(registry: FontRegistry, code: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="backend.reports.fonts"):
        assert registry.register() is False
    assert not registry.ready
    assert registry.error_code == code
    (record,) = [r for r in caplog.records if r.name == "backend.reports.fonts"]
    assert record.getMessage() == "pdf_fonts_unavailable"
    assert record.error_code == code
    assert record.component == "pdf_fonts"


@pytest.mark.parametrize("missing", [face.file for face in FACES])
def test_missing_font_is_not_ready(font_dir: Path, missing: str, caplog) -> None:
    (font_dir / missing).unlink()
    _fail(FontRegistry(font_dir), "font_missing", caplog)


def test_missing_font_directory_is_not_ready(tmp_path: Path, caplog) -> None:
    _fail(FontRegistry(tmp_path / "absent"), "manifest_invalid", caplog)


@pytest.mark.parametrize(
    "manifest",
    [
        None,  # file removed
        "{not json",
        "[]",
        '{"fonts": [{"file": "DejaVuSans.ttf"}]}',
        '{"fonts": [{"file": "DejaVuSans.ttf", "sha256": 7}]}',
        '{"fonts": [{"file": "DejaVuSans.ttf", "sha256": "abc"}]}',
    ],
)
def test_invalid_manifest_is_not_ready(font_dir: Path, manifest: str | None, caplog) -> None:
    path = font_dir / fonts.MANIFEST_NAME
    if manifest is None:
        path.unlink()
    else:
        path.write_text(manifest, encoding="utf-8")
    _fail(FontRegistry(font_dir), "manifest_invalid", caplog)


def test_manifest_without_an_entry_for_a_font_is_not_ready(font_dir: Path, caplog) -> None:
    path = font_dir / fonts.MANIFEST_NAME
    manifest = json.loads(path.read_text("utf-8"))
    manifest["fonts"] = [f for f in manifest["fonts"] if f["file"] != "DejaVuSans-Bold.ttf"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _fail(FontRegistry(font_dir), "font_hash_mismatch", caplog)


@pytest.mark.parametrize("damaged", [face.file for face in FACES])
def test_altered_font_is_not_ready(font_dir: Path, damaged: str, caplog) -> None:
    path = font_dir / damaged
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    _fail(FontRegistry(font_dir), "font_hash_mismatch", caplog)


@pytest.mark.parametrize("damaged", [face.file for face in FACES])
def test_corrupt_font_matching_its_manifest_is_not_ready(
    font_dir: Path, damaged: str, caplog
) -> None:
    # The manifest is altered too, so only parsing can detect the corruption.
    corrupt = b"\x00\x01\x00\x00" + b"\xff" * 512
    (font_dir / damaged).write_bytes(corrupt)
    path = font_dir / fonts.MANIFEST_NAME
    manifest = json.loads(path.read_text("utf-8"))
    for entry in manifest["fonts"]:
        if entry["file"] == damaged:
            entry["sha256"] = hashlib.sha256(corrupt).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _fail(FontRegistry(font_dir), "font_invalid", caplog)


def test_missing_shaping_support_is_not_ready(monkeypatch, caplog) -> None:
    monkeypatch.setattr(ttfonts, "uharfbuzz", None)
    _fail(FontRegistry(), "shaping_unavailable", caplog)


def test_unexpected_loader_error_is_contained(monkeypatch, caplog) -> None:
    def boom(self, *args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ttfonts.TTFont, "__init__", boom)
    _fail(FontRegistry(), "font_invalid", caplog)


def test_unexpected_asset_check_error_is_contained(monkeypatch, caplog) -> None:
    def boom(self):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(FontRegistry, "_expected_hashes", boom)
    _fail(FontRegistry(), "registration_failed", caplog)


def test_registration_error_is_contained(monkeypatch, caplog) -> None:
    def boom(font):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pdfmetrics, "registerFont", boom)
    _fail(FontRegistry(), "registration_failed", caplog)


def test_failed_asset_check_registers_nothing(font_dir: Path, monkeypatch, caplog) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pdfmetrics, "registerFont", lambda f: calls.append(f.fontName))
    (font_dir / "NotoSansDevanagari-Bold.ttf").unlink()  # the last face to load
    _fail(FontRegistry(font_dir), "font_missing", caplog)
    assert calls == []


def test_failure_is_cached_and_not_retried(font_dir: Path, caplog) -> None:
    (font_dir / "DejaVuSans.ttf").unlink()
    registry = FontRegistry(font_dir)
    _fail(registry, "font_missing", caplog)
    shutil.copy(fonts.FONT_DIR / "DejaVuSans.ttf", font_dir / "DejaVuSans.ttf")
    assert registry.register() is False
    assert registry.error_code == "font_missing"


def test_failure_log_contains_no_paths(tmp_path: Path, caplog) -> None:
    private = tmp_path / "private-name-Jane-Doe"
    with caplog.at_level(logging.DEBUG):
        FontRegistry(private).register()
    for record in caplog.records:
        assert "Jane" not in record.getMessage()
        assert all("Jane" not in str(value) for value in vars(record).values())


# --- Repeated calls, readiness and cross-instance locking ----------------------


def test_repeated_calls_after_success_do_no_work(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _count_loads(monkeypatch)
    registry = FontRegistry()
    assert registry.register() is True
    assert len(loads) == len(FACES)
    for _ in range(3):
        assert registry.register() is True
        assert (registry.ready, registry.error_code) == (True, None)
    assert len(loads) == len(FACES)


def test_repeated_calls_after_failure_do_no_work(
    font_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    loads = _count_loads(monkeypatch)
    (font_dir / "NotoSansDevanagari-Bold.ttf").unlink()  # fails after three loads
    registry = FontRegistry(font_dir)
    _fail(registry, "font_missing", caplog)
    assert len(loads) == 3
    caplog.clear()
    for _ in range(3):
        assert registry.register() is False
        assert (registry.ready, registry.error_code) == (False, "font_missing")
    assert len(loads) == 3
    assert not caplog.records  # the failure is logged once


def test_readiness_and_error_code_are_consistent(font_dir: Path) -> None:
    fresh, good, bad = FontRegistry(), FontRegistry(), FontRegistry(font_dir)
    (font_dir / "DejaVuSans.ttf").unlink()
    assert (fresh.ready, fresh.error_code) == (False, None)  # not attempted yet
    assert good.register() is True
    assert (good.ready, good.error_code) == (True, None)
    assert bad.register() is False
    assert (bad.ready, bad.error_code) == (False, "font_missing")
    for registry in (good, bad):
        assert registry.register() is registry.ready
        assert registry.ready is (registry.error_code is None)


def test_instances_never_register_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    peak = 0
    guard = threading.Lock()
    calls: list[str] = []
    real = pdfmetrics.registerFont

    def slow_register(font) -> None:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)  # widen the window for any overlap
        calls.append(font.fontName)
        real(font)
        with guard:
            active -= 1

    monkeypatch.setattr(pdfmetrics, "registerFont", slow_register)
    registries = [FontRegistry() for _ in range(4)]
    barrier = threading.Barrier(16)
    results: list[bool] = []

    def worker(index: int) -> None:
        barrier.wait()
        results.append(registries[index % 4].register())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [True] * 16
    assert peak == 1, "registrations overlapped"
    assert len(calls) == 4 * len(FACES)  # each instance registers once
    assert all(registry.ready for registry in registries)


def test_registered_names_resolve_to_the_verified_faces() -> None:
    assert FontRegistry().register()
    for face in FACES:
        expected = ttfonts.TTFontFile(str(fonts.FONT_DIR / face.file))
        registered = pdfmetrics.getFont(face.name).face
        assert bytes(registered.name) == bytes(expected.name) == Path(face.file).stem.encode()
        for table in ("charToGlyph", "charWidths"):
            ours, theirs = getattr(expected, table), getattr(registered, table)
            assert {code: theirs.get(code) for code in ours} == ours  # originals unchanged
            # Earlier shaped renders may have added private-use mappings, nothing else.
            assert all(0xE000 <= code <= 0xF8FF for code in set(theirs) - set(ours))
    for role in ("default", "devanagari"):
        assert tt2ps(font_name(role), 1, 1) == font_name(role, bold=True)
        assert tt2ps(font_name(role), 0, 1) == font_name(role)


# --- Shadowed font names (each scenario in a fresh interpreter) -------------------

ROOT = Path(__file__).resolve().parents[2]
_PRELUDE = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
from reportlab.lib import fonts as rl_fonts
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from backend.reports.fonts import FACES, FONT_DIR, FontRegistry
"""
_REPORT = """
registry = FontRegistry()
first = registry.register()
names = [face.name for face in FACES]
print(json.dumps({
    "first": first,
    "again": registry.register(),
    "ready": registry.ready,
    "error_code": registry.error_code,
    "faces": {
        name: bytes(pdfmetrics.getFont(name).face.name).decode()
        for name in names
        if name in pdfmetrics.getRegisteredFontNames()
    },
}))
"""


def _fresh_process(setup: str) -> dict:
    """Run ``setup`` and then a registration in a new interpreter (isolated mode)."""
    code = _PRELUDE + textwrap.dedent(setup) + _REPORT
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned code
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _assert_fails_closed(result: dict) -> None:
    assert result["first"] is False
    assert result["again"] is False
    assert result["ready"] is False
    assert result["error_code"] == "registration_failed"


def test_fresh_process_without_preregistration_is_ready() -> None:
    result = _fresh_process("")
    assert (result["first"], result["ready"], result["error_code"]) == (True, True, None)
    assert result["faces"] == {face.name: Path(face.file).stem for face in FACES}


def test_wrong_font_preregistered_under_the_noto_name_fails_closed() -> None:
    result = _fresh_process(
        'pdfmetrics.registerFont(TTFont("NotoSansDevanagari", str(FONT_DIR / "DejaVuSans.ttf")))'
    )
    _assert_fails_closed(result)
    assert result["faces"]["NotoSansDevanagari"] == "DejaVuSans"  # ReportLab kept the first


def test_noto_bold_preregistered_under_the_noto_regular_name_fails_closed() -> None:
    result = _fresh_process(
        'pdfmetrics.registerFont(TTFont("NotoSansDevanagari",'
        ' str(FONT_DIR / "NotoSansDevanagari-Bold.ttf")))'
    )
    _assert_fails_closed(result)


def test_wrong_font_preregistered_under_a_dejavu_name_fails_closed() -> None:
    result = _fresh_process(
        'pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(FONT_DIR / "DejaVuSans.ttf")))'
    )
    _assert_fails_closed(result)


@pytest.mark.parametrize(
    "tamper",
    [
        "face.charToGlyph = {k: v for k, v in face.charToGlyph.items() if k != 0x0915}",
        "face.charWidths = {k: v + 1 for k, v in face.charWidths.items()}",
        "face.numGlyphs = face.numGlyphs + 1",
        "face.fontRevision = (face.fontRevision[0] + 1, face.fontRevision[1])",
        'face.name = b"OtherFont-Regular"',
        "face.charToGlyph[0x0915] = face.charToGlyph[0x0916]",
        "face.charWidths[0x0915] = face.charWidths[0x0915] + 1",
    ],
    ids=[
        "glyph_map",
        "widths",
        "glyph_count",
        "revision",
        "postscript_name",
        "one_glyph_entry",
        "one_width_entry",
    ],
)
def test_same_postscript_name_with_different_face_data_fails_closed(tamper: str) -> None:
    # Simulates another build of the font family registered first under our name.
    result = _fresh_process(
        f"""
        _font = TTFont("NotoSansDevanagari", str(FONT_DIR / "NotoSansDevanagari-Regular.ttf"))
        face = _font.face
        {tamper}
        pdfmetrics.registerFont(_font)
        """
    )
    _assert_fails_closed(result)


def test_non_ttfont_carrying_the_genuine_face_fails_closed() -> None:
    # Only the type check rejects this: its face attributes match the verified face.
    result = _fresh_process(
        """
        _genuine = TTFont("NotoSansDevanagari", str(FONT_DIR / "NotoSansDevanagari-Regular.ttf"))
        class _Impostor:
            pass
        _impostor = _Impostor()
        _impostor.__dict__.update(vars(_genuine))
        _impostor._dynamicFont = _genuine._dynamicFont
        _impostor._multiByte = _genuine._multiByte
        pdfmetrics.registerFont(_impostor)
        """
    )
    _assert_fails_closed(result)


def test_extra_entries_for_unmapped_codepoints_are_accepted() -> None:
    # The kind of addition ReportLab's shaping makes: a private-use codepoint
    # the verified face does not map, pointing at an existing glyph.
    result = _fresh_process(
        """
        _font = TTFont("NotoSansDevanagari", str(FONT_DIR / "NotoSansDevanagari-Regular.ttf"))
        assert 0xE000 not in _font.face.charToGlyph
        _font.face.charToGlyph[0xE000] = _font.face.charToGlyph[0x0915]
        _font.face.charWidths[0xE000] = 600
        pdfmetrics.registerFont(_font)
        """
    )
    assert (result["first"], result["ready"], result["error_code"]) == (True, True, None)


def test_fresh_registry_verifies_after_harfbuzz_rendering() -> None:
    result = _fresh_process(
        """
        import io
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, SimpleDocTemplate
        from backend.reports.fonts import default_registry
        assert default_registry().register()
        _face = pdfmetrics.getFont("NotoSansDevanagari").face
        _before = dict(_face.charToGlyph)
        _style = ParagraphStyle("s", fontName="NotoSansDevanagari", shaping=1)
        SimpleDocTemplate(io.BytesIO(), invariant=1).build(
            [Paragraph("क्षमता ज्ञान श्री प्रबंधक", _style)]
        )
        _added = set(_face.charToGlyph) - set(_before)
        assert _added, "shaping added no glyph mappings"
        assert all(_face.charToGlyph[c] == g for c, g in _before.items())
        """
    )
    assert (result["first"], result["ready"], result["error_code"]) == (True, True, None)


def test_identical_face_preregistered_under_the_name_is_accepted() -> None:
    # As the test fixtures do: the same file registered earlier as a separate object.
    result = _fresh_process(
        """
        for face in FACES:
            pdfmetrics.registerFont(TTFont(face.name, str(FONT_DIR / face.file)))
        """
    )
    assert (result["first"], result["ready"], result["error_code"]) == (True, True, None)


def test_wrong_family_mapping_fails_closed() -> None:
    result = _fresh_process(
        """
        rl_fonts.addMapping("NotoSansDevanagari", 1, 0, "DejaVuSans-Bold")
        pdfmetrics.registerFontFamily = lambda *args, **kwargs: None
        """
    )
    _assert_fails_closed(result)


def test_partial_registration_is_retained_but_never_ready() -> None:
    # Documented limitation: fonts registered before a failure stay registered.
    result = _fresh_process(
        """
        _real = pdfmetrics.registerFont
        _calls = []
        def _flaky(font):
            _calls.append(font.fontName)
            if len(_calls) == 3:
                raise RuntimeError("simulated failure")
            _real(font)
        pdfmetrics.registerFont = _flaky
        """
    )
    _assert_fails_closed(result)
    assert result["faces"] == {"DejaVuSans": "DejaVuSans", "DejaVuSans-Bold": "DejaVuSans-Bold"}


# --- Script runs ----------------------------------------------------------------


def test_empty_text_has_no_runs() -> None:
    assert split_script_runs("") == ()


@pytest.mark.parametrize(
    "text",
    [
        "Python, SQL & Docker 2024",
        "José Müller-Łukasiewicz, naïve café",
        "Ελληνικά and Русский",
        "e" + COMBINING_ACUTE + "cole",  # decomposed accent stays Latin
    ],
)
def test_non_devanagari_text_is_one_default_run(text: str) -> None:
    assert _roles(text) == [("default", text)]


@pytest.mark.parametrize(
    "text",
    [
        "हिन्दी",  # Hindi
        "प्रबंधक",
        "हिन्दी प्रबंधक क्षमता",  # spaces stay inside the run
        "महाराष्ट्र विकास",  # Marathi
        "ज्ञान श्री",
        "क" + NUKTA,  # KA + NUKTA (decomposed)
        "क" + NUKTA + VIRAMA + "ष",
        "ॐ" + VEDIC_TONE + DEVA_EXTENDED,
        "१२३ ।",  # Devanagari digits and danda
    ],
)
def test_devanagari_text_is_one_devanagari_run(text: str) -> None:
    assert _roles(text) == [("devanagari", text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python और SQL", [("default", "Python "), ("devanagari", "और "), ("default", "SQL")]),
        ("हिन्दी (Hindi)", [("devanagari", "हिन्दी "), ("default", "(Hindi)")]),
        ("Team lead: प्रबंधक", [("default", "Team lead: "), ("devanagari", "प्रबंधक")]),
        ("मराठी-English", [("devanagari", "मराठी"), ("default", "-English")]),
        ("हिन्दी123", [("devanagari", "हिन्दी"), ("default", "123")]),
    ],
)
def test_mixed_text_splits_at_script_boundaries(text: str, expected) -> None:
    assert _roles(text) == expected


@pytest.mark.parametrize("joiner", [ZWJ, ZWNJ])
def test_joiners_stay_inside_a_devanagari_run(joiner: str) -> None:
    # KA + VIRAMA + joiner + SSA: explicit half form / no conjunct
    text = "क" + VIRAMA + joiner + "ष"
    assert _roles(text) == [("devanagari", text)]


@pytest.mark.parametrize("joiner", [ZWJ, ZWNJ])
def test_joiner_belongs_to_the_preceding_run(joiner: str) -> None:
    assert _roles("क" + VIRAMA + joiner + "A") == [
        ("devanagari", "क" + VIRAMA + joiner),
        ("default", "A"),
    ]
    assert _roles("A" + joiner + "क") == [("default", "A" + joiner), ("devanagari", "क")]


def test_leading_inheriting_characters_join_the_first_run() -> None:
    assert _roles(" " + ZWNJ + "हिन्दी") == [("devanagari", " " + ZWNJ + "हिन्दी")]
    assert _roles(COMBINING_ACUTE + "abc") == [("default", COMBINING_ACUTE + "abc")]


def test_text_of_only_inheriting_characters_is_one_default_run() -> None:
    text = " " + ZWJ + COMBINING_ACUTE
    assert _roles(text) == [("default", text)]


def test_combining_marks_follow_their_base_character() -> None:
    # A non-Devanagari combining mark after a Devanagari letter is not split off.
    assert _roles("क" + COMBINING_ACUTE + "x") == [
        ("devanagari", "क" + COMBINING_ACUTE),
        ("default", "x"),
    ]
    assert _roles("क" + ANUSVARA) == [("devanagari", "क" + ANUSVARA)]


def test_other_whitespace_is_not_inherited() -> None:
    assert _roles("हिन्दी\nप्रबंधक") == [
        ("devanagari", "हिन्दी"),
        ("default", "\n"),
        ("devanagari", "प्रबंधक"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Python और SQL, हिन्दी (Hindi) — मराठी" + ZWJ + "x" + ZWNJ,
        "  " + COMBINING_ACUTE + "क" + VIRAMA + ZWJ + "ष abc १२",
        "Senior Engineer — Bengaluru, प्रबंधक / Team Lead",
    ],
)
def test_runs_rejoin_to_the_input_and_alternate(text: str) -> None:
    runs = split_script_runs(text)
    assert "".join(run.text for run in runs) == text
    assert all(run.text for run in runs)
    assert all(a.role != b.role for a, b in zip(runs, runs[1:], strict=False))


def test_devanagari_ranges() -> None:
    for code, expected in [
        (0x08FF, False),
        (0x0900, True),
        (0x097F, True),
        (0x0980, False),  # Bengali
        (0xA8DF, False),
        (0xA8E0, True),
        (0xA8FF, True),
        (0x1CCF, False),
        (0x1CD0, True),
        (0x1CFF, True),
        (0x1D00, False),
    ]:
        assert is_devanagari(chr(code)) is expected, f"U+{code:04X}"


def test_run_font_names() -> None:
    latin, deva = split_script_runs("A क")
    assert latin == ScriptRun("default", "A ")
    assert latin.font_name() == "DejaVuSans"
    assert deva.font_name(bold=True) == "NotoSansDevanagari-Bold"


def test_devanagari_runs_are_covered_by_the_noto_font() -> None:
    assert FontRegistry().register()
    face = pdfmetrics.getFont("NotoSansDevanagari").face
    for text in ["हिन्दी प्रबंधक क्षमता", "महाराष्ट्र विकास", "ज्ञान श्री १२३ ।"]:
        for run in split_script_runs(text):
            assert run.role == "devanagari"
            assert all(ord(ch) in face.charToGlyph for ch in run.text), run.text
