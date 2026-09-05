"""Tests for ``ytedit.media.captions``: ASS generation, glyph checks, SRT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ytedit.config import load_settings
from ytedit.media import captions as C
from ytedit.project import Project
from ytedit.timeline import Caption, Timeline

POLISH = "Zażółć gęślą jaźń — ŁÓDŹ"


def styles() -> dict:
    """The real ``config/caption_styles.yaml``."""
    return load_settings().caption_styles


def font() -> tuple[str, Path]:
    """The real configured caption font."""
    return load_settings().caption_font()


def cap(**kwargs) -> Caption:
    """Build a caption with sensible defaults."""
    base = {"id": "t001", "at": 0.5, "end": 3.0, "text": "LIZBONA",
            "style": "location", "position": "lower-left"}
    base.update(kwargs)
    return Caption(**base)


# ----------------------------------------------------------------------
# time formatting
# ----------------------------------------------------------------------
def test_ass_time() -> None:
    assert C.ass_time(0) == "0:00:00.00"
    assert C.ass_time(1.5) == "0:00:01.50"
    assert C.ass_time(61.234) == "0:01:01.23"
    assert C.ass_time(3723.99) == "1:02:03.99"
    assert C.ass_time(-5) == "0:00:00.00"


def test_srt_time() -> None:
    assert C.srt_time(0) == "00:00:00,000"
    assert C.srt_time(61.234) == "00:01:01,234"
    assert C.srt_time(3723.5) == "01:02:03,500"


# ----------------------------------------------------------------------
# glyph coverage
# ----------------------------------------------------------------------
def test_configured_font_covers_the_polish_set() -> None:
    name, path = font()
    assert path.exists(), f"configured caption font is missing: {path}"
    assert C.missing_glyphs(path) == []
    C.check_font(path, C.POLISH_GLYPHS, name=name)  # must not raise


def test_missing_font_file_reports_every_glyph(tmp_path: Path) -> None:
    ghost = tmp_path / "nope.ttf"
    assert C.missing_glyphs(ghost) == list(C.POLISH_GLYPHS)
    with pytest.raises(C.CaptionFontError) as excinfo:
        C.check_font(ghost, C.POLISH_GLYPHS, name="Ghost")
    assert "Ghost" in str(excinfo.value)
    assert "ł" in str(excinfo.value)
    assert len(excinfo.value.missing) == len(C.POLISH_GLYPHS)


def test_build_ass_hard_fails_on_a_font_without_polish(tmp_path: Path) -> None:
    with pytest.raises(C.CaptionFontError):
        C.build_ass([cap()], 1920, 1080, styles(), ("Ghost", tmp_path / "nope.ttf"))


# ----------------------------------------------------------------------
# ASS document
# ----------------------------------------------------------------------
def test_build_ass_header_matches_the_canvas() -> None:
    doc = C.build_ass([cap()], 1920, 1080, styles(), font(), duration=10.0)
    assert "[Script Info]" in doc
    assert "PlayResX: 1920" in doc
    assert "PlayResY: 1080" in doc
    assert "ScriptType: v4.00+" in doc
    assert "[V4+ Styles]" in doc and "[Events]" in doc
    for name in ("location", "hook", "subtitle"):
        assert f"Style: {name}," in doc


def test_build_ass_renders_polish_text_and_the_fade() -> None:
    doc = C.build_ass(
        [cap(text=POLISH)], 1920, 1080, styles(), font(), duration=10.0
    )
    dialogue = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue) == 1
    line = dialogue[0]
    assert POLISH.upper() in line          # the location style is uppercase
    assert "\\fad(400,400)" in line
    assert "0:00:00.50" in line and "0:00:03.00" in line
    assert ",location," in line


def test_build_ass_scales_sizes_and_margins_to_a_720p_canvas() -> None:
    full = C.build_ass([cap()], 1920, 1080, styles(), font())
    half = C.build_ass([cap()], 1280, 720, styles(), font())
    def fontsize(doc: str) -> int:
        line = next(ln for ln in doc.splitlines() if ln.startswith("Style: location,"))
        return int(line.split(",")[2])
    assert fontsize(full) == 54
    assert fontsize(half) == 36           # 54 * 720/1080
    assert "PlayResY: 720" in half


def test_build_ass_positions_map_to_alignments() -> None:
    doc = C.build_ass(
        [cap(id="a", at=0.0, end=1.0, position="center", style="hook", text="A"),
         cap(id="b", at=1.0, end=2.0, position="upper-left", style="hook", text="B")],
        1920, 1080, styles(), font(), duration=10.0,
    )
    lines = [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]
    assert len(lines) == 2
    # "center" equals the hook style's own alignment (5) so no \an override
    assert "\\an" not in lines[0]
    assert "\\an7" in lines[1]


def test_build_ass_keeps_captions_inside_the_safe_area() -> None:
    doc = C.build_ass([cap()], 1920, 1080, styles(), font(), duration=10.0)
    line = next(ln for ln in doc.splitlines() if ln.startswith("Dialogue:"))
    fields = line.split(",")
    margin_l, margin_r, margin_v = int(fields[5]), int(fields[6]), int(fields[7])
    assert margin_l >= int(0.05 * 1920)
    assert margin_r == 0                 # bottom-left: only the left edge matters
    assert margin_v >= int(0.12 * 1080)


def test_build_ass_clamps_a_cue_to_the_duration() -> None:
    doc = C.build_ass([cap(at=1.0, end=99.0)], 1920, 1080, styles(), font(), duration=5.0)
    line = next(ln for ln in doc.splitlines() if ln.startswith("Dialogue:"))
    assert "0:00:05.00" in line


def test_build_ass_drops_a_cue_that_ends_before_it_starts() -> None:
    doc = C.build_ass([cap(at=9.0, end=12.0)], 1920, 1080, styles(), font(), duration=5.0)
    assert not [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]


def test_escape_ass_text_neutralizes_override_braces() -> None:
    assert C.escape_ass_text("a{\\b1}b") == "a(∖b1)b"
    assert C.escape_ass_text("one\ntwo") == "one\\Ntwo"


# ----------------------------------------------------------------------
# SRT
# ----------------------------------------------------------------------
def test_write_srt_numbers_and_formats_cues(tmp_path: Path) -> None:
    path = C.write_srt(
        [C.Cue(0.0, 1.5, "pierwsza"), {"start": 2.0, "end": 3.25, "text": "druga"},
         (4.0, 4.0, "pusta")],
        tmp_path / "out.srt",
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith("1\n00:00:00,000 --> 00:00:01,500\npierwsza\n")
    assert "2\n00:00:02,000 --> 00:00:03,250\ndruga" in text
    assert "pusta" not in text            # zero-length cues are dropped


def test_chunk_words_respects_the_word_and_duration_limits() -> None:
    words = [(i * 0.4, i * 0.4 + 0.35, f"w{i}") for i in range(12)]
    cues = C.chunk_words(words, min_words=3, max_words=5, min_duration=1.0, max_duration=4.0)
    assert cues
    for cue in cues:
        assert 1 <= len(cue.text.split()) <= 5
        assert 1.0 - 1e-6 <= cue.end - cue.start <= 4.0 + 1e-6


def test_chunk_words_breaks_on_a_long_pause() -> None:
    words = [(0.0, 0.3, "a"), (0.3, 0.6, "b"), (5.0, 5.3, "c")]
    cues = C.chunk_words(words, max_gap=0.8)
    assert len(cues) == 2
    assert cues[0].text == "a b"
    assert cues[1].text == "c"


def test_srt_from_transcripts_maps_clip_time_to_timeline_time(project: Project) -> None:
    project.add_clip({"id": "c001", "duration": 10.0})
    (project.transcripts_dir / "c001.json").write_text(
        json.dumps({
            "clip": "c001",
            "words": [
                {"t": "Dzień", "s": 5.0, "e": 5.3},
                {"t": "dobry", "s": 5.35, "e": 5.7},
                {"t": "Łódź", "s": 5.75, "e": 6.2},
                {"t": "poza", "s": 9.0, "e": 9.4},   # outside the segment
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    timeline = Timeline.model_validate({
        "tracks": {"video": [
            {"id": "s001", "clip": "c001", "in": 4.0, "out": 8.0},
        ]},
    })
    cues = C.srt_from_transcripts(project, timeline)
    assert len(cues) == 1
    assert cues[0].text == "Dzień dobry Łódź"
    # clip 5.0 s sits 1.0 s into a segment that starts at 0.0 on the timeline
    assert cues[0].start == pytest.approx(1.0, abs=0.01)


def test_srt_from_transcripts_skips_muted_segments(project: Project) -> None:
    project.add_clip({"id": "c001", "duration": 10.0})
    (project.transcripts_dir / "c001.json").write_text(
        json.dumps({"clip": "c001", "words": [{"t": "cisza", "s": 1.0, "e": 1.5}]}),
        encoding="utf-8",
    )
    timeline = Timeline.model_validate({
        "tracks": {"video": [
            {"id": "s001", "clip": "c001", "in": 0.0, "out": 4.0, "mute_source": True},
        ]},
    })
    assert C.srt_from_transcripts(project, timeline) == []
