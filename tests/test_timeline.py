"""Tests for ``ytedit.timeline``: model, geometry, validation, speech ranges."""

from __future__ import annotations

import json

import pytest

from ytedit.project import Project
from ytedit.timeline import (
    AudioFrom,
    Caption,
    Chapter,
    MusicCue,
    MuteRange,
    Timeline,
    Tracks,
    VideoSegment,
    merge_ranges,
    speech_ranges_from_transcripts,
)


def _segment(sid: str, clip: str, start: float, end: float, **kw) -> VideoSegment:
    return VideoSegment(id=sid, clip=clip, **{"in": start}, out=end, **kw)


def _timeline(*segments: VideoSegment, **kw) -> Timeline:
    return Timeline(tracks=Tracks(video=list(segments)), **kw)


# ----------------------------------------------------------------------
# model / io
# ----------------------------------------------------------------------
def test_in_alias_roundtrip(tmp_path) -> None:
    tl = _timeline(_segment("s001", "c001", 12.0, 16.5))
    path = tl.save(tmp_path / "timeline.json")
    raw = json.loads(path.read_text())
    assert raw["tracks"]["video"][0]["in"] == 12.0
    assert "in_" not in raw["tracks"]["video"][0]
    assert Timeline.load(path).tracks.video[0].in_ == 12.0


def test_architecture_example_parses() -> None:
    doc = {
        "version": 1, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
        "tracks": {
            "video": [{
                "id": "s001", "clip": "c003", "in": 12.0, "out": 16.5, "role": "cold-open",
                "transform": {"fit": "cover", "zoom": 1.0}, "grade": "default",
                "transition_in": {"type": "cut", "duration": 0.0}, "speed": 1.0,
                "mute_source": False, "source_audio_gain_db": 0, "notes": "drone reveal",
            }],
            "voice": [{"id": "v001", "file": "voice/intro_pickup.wav", "at": 0.5, "gain_db": 0}],
            "music": [{
                "id": "m001", "file": "music/arrival_warm.mp3", "at": 0.0, "end": 4.0,
                "gain_db": -18, "fade_in": 2.0, "fade_out": 3.0,
                "duck": {"mode": "auto", "amount_db": -12, "attack": 0.15, "release": 0.6},
            }],
            "captions": [{
                "id": "t001", "at": 0.5, "end": 3.0, "text": "LIZBONA, PORTUGALIA",
                "style": "location", "position": "lower-left",
            }],
            "sfx": [],
        },
        "mute_ranges": [{"clip": "c005", "s": 3.0, "e": 20.0, "gain_db": -60, "reason": "bar music"}],
        "markers": [{"at": 0.0, "label": "hook"}],
        "chapters": [{"at": 0, "title": "Przyjazd do Lizbony"}],
        "meta": {"title_candidates": [], "generated_by": "plan@2026-09-04", "edited_by_human": False},
    }
    tl = Timeline.model_validate(doc)
    assert tl.tracks.video[0].in_ == 12.0
    assert tl.tracks.music[0].duck.amount_db == -12
    assert tl.chapters[0].title.startswith("Przyjazd")


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def test_segment_positions_are_cumulative() -> None:
    tl = _timeline(
        _segment("s001", "c001", 0.0, 4.0),
        _segment("s002", "c002", 10.0, 13.0),
        _segment("s003", "c003", 0.0, 2.0),
    )
    positions = tl.segment_positions()
    assert [(p.start, p.end) for p in positions] == [(0.0, 4.0), (4.0, 7.0), (7.0, 9.0)]
    assert tl.duration() == 9.0


def test_xfade_overlaps_previous_segment() -> None:
    tl = _timeline(
        _segment("s001", "c001", 0.0, 4.0),
        _segment("s002", "c002", 0.0, 4.0, transition_in={"type": "xfade", "duration": 0.8}),
    )
    positions = tl.segment_positions()
    assert positions[1].start == pytest.approx(3.2)
    assert tl.duration() == pytest.approx(7.2)


def test_segment_positions_snap_to_whole_frames() -> None:
    """Fractional in/out points place the *next* segment on a frame boundary.

    1.62 s is 48.6 frames at 30 fps and renders as 49; 1.64 s is 49.2 frames
    and renders as 49 too. The raw sum (3.26 s) is not where the third segment
    starts — frame 98 (3.266667 s) is, and the positions say so.
    """
    tl = _timeline(
        _segment("s001", "c001", 0.51, 2.13),
        _segment("s002", "c002", 1.07, 2.71),
        _segment("s003", "c003", 0.33, 1.5),
        _segment("s004", "c001", 0.0, 1.0, transition_in={"type": "xfade", "duration": 0.75}),
    )
    fps = tl.fps
    assert [seg.frames(fps) for seg in tl.tracks.video] == [49, 49, 35, 30]
    assert tl.tracks.video[3].transition_in.frames(fps) == 23      # 22.5 rounds up
    positions = tl.segment_positions()
    assert [p.start for p in positions] == [0.0, 1.633333, 3.266667, 3.666667]
    assert positions[2].end == pytest.approx(133 / 30, abs=1e-6)
    assert tl.frame_count() == 49 + 49 + 35 + 30 - 23
    assert tl.duration() == pytest.approx(tl.frame_count() / fps, abs=1e-6)
    # every position is a whole number of frames
    for pos in positions:
        assert abs(pos.start * fps - round(pos.start * fps)) < 1e-4
        assert abs(pos.end * fps - round(pos.end * fps)) < 1e-4


def test_a_sub_frame_segment_still_renders_one_frame() -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 0.01))
    assert tl.tracks.video[0].frames(tl.fps) == 1
    assert tl.duration() == pytest.approx(1 / 30, abs=1e-6)


def test_speed_shortens_a_segment() -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 4.0, speed=2.0))
    assert tl.duration() == 2.0


def test_segment_at_and_clip_ids() -> None:
    tl = _timeline(
        _segment("s001", "c001", 0.0, 4.0),
        _segment("s002", "c002", 0.0, 4.0),
        _segment("s003", "c001", 5.0, 6.0),
    )
    assert tl.segment_at(5.0).segment.id == "s002"
    assert tl.segment_at(99.0) is None
    assert tl.clip_ids() == ["c001", "c002"]


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
def test_validate_clean_timeline() -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 4.0))
    tl.tracks.captions.append(Caption(id="t001", at=0.5, end=3.0, text="LIZBONA"))
    assert tl.validate() == []


def test_validate_catches_in_after_out() -> None:
    tl = _timeline(_segment("s001", "c001", 16.5, 12.0))
    issues = tl.validate()
    assert any("in (16.5) >= out (12.0)" in i for i in issues)


def test_validate_catches_caption_overlap() -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 10.0))
    tl.tracks.captions += [
        Caption(id="t001", at=0.0, end=4.0, text="A"),
        Caption(id="t002", at=3.0, end=6.0, text="B"),
    ]
    issues = tl.validate()
    assert any("caption 't002' overlaps 't001'" in i for i in issues)


def test_validate_catches_music_cue_overlap() -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 60.0))
    tl.tracks.music += [
        MusicCue(id="m001", file="music/a.mp3", at=0.0, end=30.0),
        MusicCue(id="m002", file="music/b.mp3", at=25.0, end=50.0),
    ]
    assert any("music cue 'm002' overlaps 'm001'" in i for i in tl.validate())


def test_validate_catches_caption_past_the_end() -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 4.0))
    tl.tracks.captions.append(Caption(id="t001", at=10.0, end=12.0, text="late"))
    assert any("outside programme duration" in i for i in tl.validate())


def test_validate_catches_video_overlap_without_xfade() -> None:
    tl = _timeline(
        _segment("s001", "c001", 0.0, 4.0),
        _segment("s002", "c002", 0.0, 4.0, transition_in={"type": "fade", "duration": 1.0}),
    )
    # A "fade" transition does not overlap, so positions stay sequential;
    # the timeline is valid but the transition must fit inside the segment.
    assert tl.validate() == []
    tl.tracks.video[1].transition_in.duration = 9.0
    assert any("longer than the segment" in i for i in tl.validate())


def test_validate_catches_missing_clip_and_bad_mute_range(project: Project) -> None:
    project.add_clip({"id": "c001", "order": 1})
    tl = _timeline(_segment("s001", "c001", 0.0, 4.0), _segment("s002", "c999", 0.0, 2.0))
    tl.mute_ranges.append(MuteRange(clip="c001", s=5.0, e=1.0))
    issues = tl.validate(project)
    assert any("missing clip 'c999'" in i for i in issues)
    assert any("s (5.0) >= e (1.0)" in i for i in issues)


def test_validate_catches_duplicate_segment_ids_and_chapters() -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 4.0), _segment("s001", "c002", 0.0, 4.0))
    tl.chapters = [Chapter(at=5.0, title="late start")]
    issues = tl.validate()
    assert any("duplicate segment id" in i for i in issues)
    assert any("first chapter at 0:00" in i for i in issues)


# ----------------------------------------------------------------------
# speech ranges
# ----------------------------------------------------------------------
def test_merge_ranges_pads_and_merges() -> None:
    assert merge_ranges([(1.0, 1.5), (1.6, 2.0), (5.0, 5.4)], merge_gap=0.4, pad=0.15) == [
        (0.85, 2.15),
        (4.85, 5.55),
    ]


def _write_transcript(project: Project, clip: str, words: list[tuple[str, float, float]]) -> None:
    project.transcripts_dir.mkdir(parents=True, exist_ok=True)
    project.transcript_path(clip).write_text(
        json.dumps({"clip": clip, "language": "pl",
                    "words": [{"t": t, "s": s, "e": e} for t, s, e in words]}),
        encoding="utf-8",
    )


def test_speech_ranges_map_clip_time_to_timeline_time(project: Project) -> None:
    _write_transcript(project, "c001", [("Dzien", 1.0, 1.4), ("dobry", 1.5, 1.9), ("Lizbona", 6.0, 6.5)])
    _write_transcript(project, "c002", [("Tramwaj", 0.2, 0.6)])
    tl = _timeline(
        _segment("s001", "c001", 0.0, 3.0),   # timeline 0.0-3.0
        _segment("s002", "c002", 0.0, 2.0),   # timeline 3.0-5.0
    )
    ranges = speech_ranges_from_transcripts(project, tl)
    # "Dzien dobry" merges into one padded run; the 6.0 s word is outside the cut.
    assert ranges == [(0.85, 2.05), (3.05, 3.75)]


def test_speech_ranges_respect_speed_and_mute_source(project: Project) -> None:
    _write_transcript(project, "c001", [("a", 2.0, 2.4)])
    _write_transcript(project, "c002", [("b", 0.0, 1.0)])
    tl = _timeline(
        _segment("s001", "c001", 1.0, 3.0, speed=2.0),          # 2 s source -> 1 s timeline
        _segment("s002", "c002", 0.0, 2.0, mute_source=True),
    )
    ranges = speech_ranges_from_transcripts(project, tl)
    assert ranges == [(0.35, 0.85)]


def test_speech_ranges_tolerate_elevenlabs_key_spelling(project: Project) -> None:
    project.transcripts_dir.mkdir(parents=True, exist_ok=True)
    project.transcript_path("c001").write_text(
        json.dumps({"words": [
            {"text": "Dzien", "start": 0.5, "end": 0.9, "type": "word"},
            {"text": " ", "start": 0.9, "end": 1.0, "type": "spacing"},
            {"text": "(laughter)", "start": 2.0, "end": 2.5, "type": "audio_event"},
        ]}),
        encoding="utf-8",
    )
    tl = _timeline(_segment("s001", "c001", 0.0, 4.0))
    assert speech_ranges_from_transcripts(project, tl) == [(0.35, 1.05)]


def test_speech_ranges_without_transcripts(project: Project) -> None:
    tl = _timeline(_segment("s001", "c001", 0.0, 4.0))
    assert speech_ranges_from_transcripts(project, tl) == []


# ----------------------------------------------------------------------
# audio_from (overlay cutaways)
# ----------------------------------------------------------------------
def test_audio_from_alias_roundtrips() -> None:
    seg = _segment(
        "s001", "c033", 0.0, 3.0,
        audio_from=AudioFrom(clip="c030", **{"in": 10.549}, out=13.549),
    )
    raw = seg.model_dump(by_alias=True, mode="json")
    assert raw["audio_from"] == {"clip": "c030", "in": 10.549, "out": 13.549}
    assert "in_" not in raw["audio_from"]

    again = VideoSegment.model_validate(raw)
    assert again.audio_from is not None
    assert again.audio_from.in_ == 10.549
    assert again.audio_from.out == 13.549
    assert again.audio_from.duration == pytest.approx(3.0)
    assert again.audio_source == ("c030", 10.549, 13.549)
    # the picture is untouched by the override
    assert (again.clip, again.in_, again.out) == ("c033", 0.0, 3.0)


def test_audio_source_defaults_to_the_segments_own_cut() -> None:
    assert _segment("s001", "c001", 2.0, 5.0).audio_source == ("c001", 2.0, 5.0)


def test_validate_catches_a_missing_audio_from_clip(project: Project) -> None:
    project.add_clip({"id": "c001", "order": 1})
    tl = _timeline(
        _segment("s001", "c001", 0.0, 4.0,
                 audio_from=AudioFrom(clip="c777", **{"in": 0.0}, out=4.0)),
    )
    issues = tl.validate(project)
    assert any("missing audio_from clip 'c777'" in i for i in issues)

    project.add_clip({"id": "c777", "order": 2})
    assert not any("audio_from" in i for i in tl.validate(project))


def test_speech_ranges_follow_audio_from_to_the_other_clip(project: Project) -> None:
    _write_transcript(project, "c001", [("Alfa", 1.0, 1.4), ("Beta", 2.0, 2.4)])
    _write_transcript(project, "c002", [("Cisza", 0.1, 0.5)])
    tl = _timeline(
        # picture from c002 (whose own word must be ignored), audio from c001
        _segment("s001", "c002", 0.0, 2.0,
                 audio_from=AudioFrom(clip="c001", **{"in": 1.0}, out=3.0)),
    )
    # 'Alfa' 1.0-1.4 in c001 is 0.0-0.4 into the segment; 'Beta' 2.0-2.4 -> 1.0-1.4
    assert speech_ranges_from_transcripts(project, tl) == [(0.0, 1.55)]
