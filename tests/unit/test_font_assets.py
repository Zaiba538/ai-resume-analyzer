"""M8: bundled font assets are unmodified, licensed and match their manifest.

``backend/assets/fonts/fonts-manifest.json`` records the source, version and
SHA-256 of every bundled font and licence file. These tests fail if a file is
changed, replaced, added without a manifest entry or shipped without its
licence.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
from reportlab.pdfbase.ttfonts import TTFontFile

FONT_DIR = Path(__file__).resolve().parents[2] / "backend" / "assets" / "fonts"
MANIFEST_PATH = FONT_DIR / "fonts-manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
FONTS = {entry["file"]: entry for entry in MANIFEST["fonts"]}
LICENSE_FILES = {entry["file"]: entry for entry in MANIFEST["license_files"]}

NOTO_RELEASE = "NotoSansDevanagari-v2.007"
NOTO_SOURCE = (
    f"https://github.com/notofonts/devanagari/releases/download/{NOTO_RELEASE}/{NOTO_RELEASE}.zip"
)
NOTO_FONTS = {
    "NotoSansDevanagari-Regular.ttf": b"NotoSansDevanagari-Regular",
    "NotoSansDevanagari-Bold.ttf": b"NotoSansDevanagari-Bold",
}
DEVANAGARI_BLOCK = range(0x0900, 0x0980)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_lists_exactly_the_bundled_files() -> None:
    bundled = {path.name for path in FONT_DIR.iterdir() if path.is_file()}
    assert bundled == set(FONTS) | set(LICENSE_FILES) | {MANIFEST_PATH.name}


@pytest.mark.parametrize("name", sorted(FONTS))
def test_font_matches_manifest_hash_and_size(name: str) -> None:
    entry = FONTS[name]
    path = FONT_DIR / name
    assert entry["modified"] is False
    assert path.stat().st_size == entry["size_bytes"]
    assert _sha256(path) == entry["sha256"]


@pytest.mark.parametrize("name", sorted(LICENSE_FILES))
def test_licence_file_matches_manifest_hash(name: str) -> None:
    assert _sha256(FONT_DIR / name) == LICENSE_FILES[name]["sha256"]


@pytest.mark.parametrize("name", sorted(FONTS))
def test_every_font_has_a_bundled_licence(name: str) -> None:
    licence = FONTS[name]["license_file"]
    assert licence in LICENSE_FILES
    assert (FONT_DIR / licence).is_file()


@pytest.mark.parametrize("name", sorted(NOTO_FONTS))
def test_noto_provenance_is_pinned_to_the_official_release(name: str) -> None:
    entry = FONTS[name]
    assert entry["license"] == "OFL-1.1"
    assert entry["license_file"] == "OFL.txt"
    assert entry["version"] == "2.007"
    assert entry["source"] == NOTO_SOURCE
    assert entry["source_release"] == NOTO_RELEASE
    assert entry["archive_member"] == f"NotoSansDevanagari/unhinted/ttf/{name}"


def test_ofl_is_the_sil_open_font_licence_1_1() -> None:
    text = (FONT_DIR / "OFL.txt").read_text(encoding="utf-8")
    assert text.startswith("Copyright 2022 The Noto Project Authors")
    assert "SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007" in text
    for heading in ("PREAMBLE", "DEFINITIONS", "PERMISSION & CONDITIONS", "TERMINATION"):
        assert heading in text
    assert "DISCLAIMER" in text


@pytest.mark.parametrize(("name", "postscript_name"), sorted(NOTO_FONTS.items()))
def test_noto_font_identity_and_coverage(name: str, postscript_name: bytes) -> None:
    font = TTFontFile(str(FONT_DIR / name))
    assert bytes(font.name) == postscript_name
    major, minor = font.fontRevision
    assert round(major + minor / 65536, 3) == 2.007
    missing = [cp for cp in DEVANAGARI_BLOCK if cp not in font.charToGlyph]
    assert not missing, f"no glyph for {[f'U+{cp:04X}' for cp in missing]}"
    # OpenType layout tables are required for HarfBuzz shaping.
    assert {"GSUB", "GPOS"} <= set(font.table)


@pytest.mark.parametrize("name", sorted(FONTS))
def test_font_permits_embedding(name: str) -> None:
    os2 = TTFontFile(str(FONT_DIR / name)).get_table("OS/2")
    (fs_type,) = struct.unpack(">H", os2[8:10])
    assert fs_type & 0x000F in (0, 4, 8), f"fsType {fs_type:#06x} restricts embedding"
