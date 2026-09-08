"""Tests for ``ytedit.inspect``: the ``ytedit at`` timecode inspector.

Builds on the same throwaway project used by ``test_render.py`` (real ingest,
no network calls) but never renders anything — the inspector only reads
``plan/timeline.json``, transcripts and ``state.json``.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from test_render import THIRD_SEGMENT_AT, build_render_project, fractional_document
from ytedit.inspect import (
    TimecodeError,
    format_timecode,
    inspect_moment,
    inspect_range,
    parse_timecode,
)
from ytedit.project import Project
from ytedit.timeline import Timeline


# ----------------------------------------------------------------------
# parse_timecode / format_timecode
# ----------------------------------------------------------------------
def test_parse_timecode_accepts_bare_seconds() -> None:
    assert parse_timecode("12.5") == pytest.approx(12.5)
    assert parse_timecode("90") == pytest.approx(90.0)


def test_parse_timecode_accepts_mm_ss() -> None:
    assert parse_timecode("4:27") == pytest.approx(267.0)
    assert parse_timecode("0:02.5") == pytest.approx(2.5)


def test_parse_timecode_accepts_hh_mm_ss() -> None:
    assert parse_timecode("1:04:27.5") == pytest.approx(3867.5)


def test_parse_timecode_rejects_garbage() -> None:
    with pytest.raises(TimecodeError):
        parse_timecode("not-a-time")
    with pytest.raises(TimecodeError):
        parse_timecode("1:2:3:4")


def test_format_timecode_matches_parse_timecode() -> None:
    assert format_timecode(267.0) == "4:27.0"
    assert format_timecode(3867.5) == "1:04:27.5"


# ----------------------------------------------------------------------
# the throwaway project (no render)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def inspected(tmp_path_factory) -> tuple[Project, Timeline]:
    """The ``test_render`` three-segment project, ingested but not rendered."""
    project = build_render_project(tmp_path_factory.mktemp("inspect"), slug="inspect-test")
    timeline = Timeline.load(project.timeline_file)
    return project, timeline


def test_inspect_moment_finds_the_segment_and_words(
    inspected: tuple[Project, Timeline],
) -> None:
    project, timeline = inspected
    # s001 (c001, in=0.5/out=3.5) occupies render time [0, 3.0); the transcript
    # words for c001 sit at clip time 2.0-3.4s, well inside that cut.
    moment = inspect_moment(project, timeline, at=2.5)
    assert moment.segment is not None
    assert moment.segment.id == "s001"
    assert moment.audio_clip == "c001"
    words = [w.text for w in moment.words]
    assert "Dzień" in words
    assert "Lizbony." in words
    assert "Lizbony" in moment.audio_description or "Dzień" in moment.audio_description


def test_inspect_moment_reports_the_active_caption_and_music_cue(
    inspected: tuple[Project, Timeline],
) -> None:
    project, timeline = inspected
    moment = inspect_moment(project, timeline, at=2.5)
    assert any(cap["text"].startswith("LIZBONA") for cap in moment.captions)
    assert moment.music is not None
    assert moment.music["id"] == "m001"


def test_inspect_moment_reports_music_only_over_the_silent_clip(
    inspected: tuple[Project, Timeline],
) -> None:
    project, timeline = inspected
    # s003 (c003, the audio-less fixture) occupies render time [5.5, 7.5) —
    # see test_render.EXPECTED_DURATION's derivation.
    moment = inspect_moment(project, timeline, at=6.5)
    assert moment.segment is not None
    assert moment.segment.id == "s003"
    assert moment.audio_description == "music only"
    assert not moment.words


def test_inspect_moment_reports_the_mute_source_flag(
    inspected: tuple[Project, Timeline],
) -> None:
    project, timeline = inspected
    moment = inspect_moment(project, timeline, at=6.5)
    assert moment.segment is not None
    # c003 has no audio stream at all; the segment itself is not explicitly muted.
    assert moment.segment.mute_source is False


def test_inspect_moment_clamps_a_time_past_the_programme_to_the_last_segment(
    inspected: tuple[Project, Timeline],
) -> None:
    project, timeline = inspected
    moment = inspect_moment(project, timeline, at=999.0)
    assert moment.segment is not None
    assert moment.segment.id == "s003"


def test_inspect_range_lists_every_segment_within_the_window(
    inspected: tuple[Project, Timeline],
) -> None:
    project, timeline = inspected
    segments = inspect_range(project, timeline, at=2.5, around=10.0)
    assert [s["id"] for s in segments] == ["s001", "s002", "s003"]


def test_inspect_range_excludes_segments_outside_the_window(
    inspected: tuple[Project, Timeline],
) -> None:
    project, timeline = inspected
    segments = inspect_range(project, timeline, at=0.5, around=0.4)
    assert [s["id"] for s in segments] == ["s001"]


# ----------------------------------------------------------------------
# voice pickup moment
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def voiced(tmp_path_factory) -> tuple[Project, Timeline]:
    """The ``test_render`` frame-exactness project with its voice pickup WAV."""
    project = build_render_project(
        tmp_path_factory.mktemp("inspect-voice"),
        slug="inspect-voice-test",
        document=fractional_document(),
    )
    project.voice_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=0.6",
         "-c:a", "pcm_s16le", "-ac", "2", str(project.voice_dir / "v001.wav")],
        check=True,
    )
    timeline = Timeline.load(project.timeline_file)
    return project, timeline


def test_inspect_moment_reports_a_playing_voice_pickup(
    voiced: tuple[Project, Timeline],
) -> None:
    project, timeline = voiced
    # The pickup starts exactly at THIRD_SEGMENT_AT and runs 0.6s (the fixture
    # WAV); every source segment is muted, so without the pickup this would
    # otherwise read as silence/music-only.
    moment = inspect_moment(project, timeline, at=THIRD_SEGMENT_AT + 0.1)
    assert moment.voice_pickup == "v001.wav"
    assert moment.audio_description == "voice pickup: v001.wav"


def test_inspect_moment_ignores_the_voice_pickup_outside_its_window(
    voiced: tuple[Project, Timeline],
) -> None:
    project, timeline = voiced
    moment = inspect_moment(project, timeline, at=0.05)
    assert moment.voice_pickup is None


# ----------------------------------------------------------------------
# empty timeline
# ----------------------------------------------------------------------
def test_inspect_moment_handles_an_empty_timeline(tmp_path_factory) -> None:
    project = build_render_project(
        tmp_path_factory.mktemp("inspect-empty"),
        slug="inspect-empty-test",
        document={
            "version": 1, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
            "tracks": {"video": [], "voice": [], "music": [], "captions": [], "sfx": []},
            "mute_ranges": [], "markers": [], "chapters": [], "meta": {},
        },
    )
    timeline = Timeline.load(project.timeline_file)
    moment = inspect_moment(project, timeline, at=1.0)
    assert moment.segment is None
    assert moment.audio_description == "empty timeline"
