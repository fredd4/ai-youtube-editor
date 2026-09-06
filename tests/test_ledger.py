"""Tests for ``ytedit.ai.ledger`` — the audio dedupe pass: no audio ever twice.

Offline like ``test_tidy.py``/``test_overlay.py``: synthetic transcripts on
disk, no media, no API calls.
"""

from __future__ import annotations

import pytest

from tests.test_tidy import add_clip, seg, timeline_of
from ytedit.ai.ledger import (
    AMBIENT_TAG,
    ambient_repeat_count,
    dedupe_audio,
    find_duplicate_audio,
)
from ytedit.project import Project
from ytedit.timeline import AudioFrom, VideoSegment, VoiceItem


#: Three well-separated words on one clip, used across most cases below.
WORDS = [(1.0, 1.4, "Raz"), (5.0, 5.4, "Dwa"), (9.0, 9.4, "Trzy")]


# ----------------------------------------------------------------------
# own-audio (no audio_from) segments
# ----------------------------------------------------------------------
def test_own_audio_overlap_is_advanced(project: Project) -> None:
    """A second a-roll piece overlapping the tail of the first has its ``in``
    pushed past what the first already played."""
    add_clip(project, "c001", 60.0, WORDS)
    tl = timeline_of(
        seg("s001", "c001", 0.5, 6.0, role="a-roll"),
        seg("s002", "c001", 4.0, 10.0, role="a-roll"),   # overlaps [4.0, 6.0), has 'Dwa'
    )
    tl, changes = dedupe_audio(tl, project)

    s002 = next(s for s in tl.tracks.video if s.id == "s002")
    assert s002.in_ == pytest.approx(6.0)
    assert any("s002 in" in c and "already used" in c for c in changes)
    assert find_duplicate_audio(tl, project) == []


def test_own_audio_fully_covered_is_muted_not_dropped(project: Project) -> None:
    """A segment entirely inside an already-played range keeps its picture,
    loses its sound."""
    add_clip(project, "c001", 60.0, WORDS)
    tl = timeline_of(
        seg("s001", "c001", 0.5, 10.0, role="a-roll"),
        seg("s002", "c001", 4.0, 8.0, role="a-roll"),   # wholly inside [0.5, 10.0)
    )
    tl, changes = dedupe_audio(tl, project)

    s002 = next(s for s in tl.tracks.video if s.id == "s002")
    assert s002.mute_source is True
    assert s002.in_ == pytest.approx(4.0)   # the picture is untouched
    assert s002.out == pytest.approx(8.0)
    assert any("s002 muted" in c for c in changes)


def test_own_audio_with_too_little_remainder_is_dropped(project: Project) -> None:
    """Advancing past what is already used can leave a sliver too short to
    stand as its own shot — the whole segment goes, not just its audio."""
    add_clip(project, "c001", 60.0, WORDS)
    tl = timeline_of(
        seg("s001", "c001", 0.5, 6.0, role="a-roll"),
        seg("s002", "c001", 4.0, 6.5, role="a-roll"),   # 6.5 - 6.0 = 0.5s left: < min_shot
    )
    tl, changes = dedupe_audio(tl, project)

    assert [s.id for s in tl.tracks.video] == ["s001"]
    assert any("s002 dropped" in c for c in changes)


# ----------------------------------------------------------------------
# audio_from (overlay cutaway) segments
# ----------------------------------------------------------------------
def test_audio_from_overlap_shortens_the_picture_when_enough_narration_is_left(
    project: Project,
) -> None:
    add_clip(project, "c001", 60.0, WORDS)
    add_clip(project, "c900", 20.0)   # the cutaway's own (silent) picture clip
    cutaway = VideoSegment(
        id="c1", clip="c900", **{"in": 0.0}, out=3.0, role="cutaway",
        audio_from=AudioFrom(clip="c001", **{"in": 4.0}, out=7.0),
    )
    tl = timeline_of(
        seg("s001", "c001", 0.5, 6.0, role="a-roll"),   # plays c001 up to 6.0
        cutaway,                                         # audio_from wants 4.0-7.0
    )
    tl, changes = dedupe_audio(tl, project)

    fixed = next(s for s in tl.tracks.video if s.id == "c1")
    assert fixed.audio_from is not None
    assert fixed.audio_from.in_ == pytest.approx(6.0)
    assert fixed.audio_from.out == pytest.approx(7.0)
    assert fixed.out == pytest.approx(1.0)   # picture shortened to match 1.0s left
    assert fixed.mute_source is False
    assert any("c1 out" in c and "audio_from.in" in c for c in changes)


def test_audio_from_fully_covered_is_silenced_under_music(project: Project) -> None:
    """When none of the borrowed range is left, the picture stays, the
    narration is dropped and the music takes over."""
    add_clip(project, "c001", 60.0, WORDS)
    add_clip(project, "c900", 20.0)
    cutaway = VideoSegment(
        id="c1", clip="c900", **{"in": 0.0}, out=2.0, role="cutaway",
        audio_from=AudioFrom(clip="c001", **{"in": 4.0}, out=6.0),
    )
    tl = timeline_of(
        seg("s001", "c001", 0.5, 6.0, role="a-roll"),
        cutaway,
    )
    tl, changes = dedupe_audio(tl, project)

    fixed = next(s for s in tl.tracks.video if s.id == "c1")
    assert fixed.audio_from is None
    assert fixed.mute_source is True
    assert fixed.in_ == pytest.approx(0.0)   # the picture is untouched
    assert fixed.out == pytest.approx(2.0)
    assert any("c1 audio_from muted" in c for c in changes)


def test_audio_from_leftover_too_short_is_silenced_rather_than_shrunk(
    project: Project,
) -> None:
    """A leftover shorter than ``min_shot_seconds`` is not worth a shortened
    picture either — mute it instead of showing a sub-second scrap."""
    add_clip(project, "c001", 60.0, WORDS)
    add_clip(project, "c900", 20.0)
    cutaway = VideoSegment(
        id="c1", clip="c900", **{"in": 0.0}, out=2.3, role="cutaway",
        audio_from=AudioFrom(clip="c001", **{"in": 4.0}, out=6.3),   # 0.3s left after 6.0
    )
    tl = timeline_of(
        seg("s001", "c001", 0.5, 6.0, role="a-roll"),
        cutaway,
    )
    tl, changes = dedupe_audio(tl, project)

    fixed = next(s for s in tl.tracks.video if s.id == "c1")
    assert fixed.audio_from is None
    assert fixed.mute_source is True
    assert fixed.out == pytest.approx(2.3)   # picture untouched, not shrunk


# ----------------------------------------------------------------------
# narration pickups (tracks.voice) count as used, too
# ----------------------------------------------------------------------
def test_a_narration_pickup_wav_counts_as_used(project: Project) -> None:
    """A ``voice/vo_<clip>_<in>_<out>.wav`` pickup extracted straight from a
    clip's own audio must not be replayed by a later segment on that clip."""
    add_clip(project, "c900", 100.0)               # long dummy to push s001's position
    add_clip(project, "c001", 60.0, [(16.0, 16.4, "Slowo")])
    tl = timeline_of(
        seg("s000", "c900", 0.0, 100.0, role="b-roll"),
        seg("s001", "c001", 15.0, 25.0, role="a-roll"),   # positioned at t=100
    )
    tl.tracks.voice = [
        VoiceItem(id="v1", file="voice/vo_c001_015.00_020.00.wav", at=50.0),
    ]
    tl, changes = dedupe_audio(tl, project)

    s001 = next(s for s in tl.tracks.video if s.id == "s001")
    assert s001.in_ == pytest.approx(20.0)
    assert any("s001 in" in c and "already used" in c for c in changes)


def test_a_non_vo_voice_file_is_not_treated_as_clip_audio(project: Project) -> None:
    """A recorded pickup (``n001.wav``) carries no clip-audio-range info and
    must not interfere with the ledger."""
    add_clip(project, "c001", 60.0, WORDS)
    tl = timeline_of(seg("s001", "c001", 0.5, 6.0, role="a-roll"))
    tl.tracks.voice = [VoiceItem(id="v1", file="voice/n001.wav", at=0.0)]
    tl, changes = dedupe_audio(tl, project)
    assert changes == []


# ----------------------------------------------------------------------
# ambient repeats: allowed by default, still counted
# ----------------------------------------------------------------------
def test_ambient_repeat_is_left_alone_by_default(project: Project) -> None:
    add_clip(project, "c001", 60.0)   # no transcript at all: everything is ambient
    tl = timeline_of(
        seg("s001", "c001", 0.0, 5.0, role="cutaway"),
        seg("s002", "c001", 3.0, 8.0, role="cutaway"),   # overlaps [3.0, 5.0)
    )
    tl, changes = dedupe_audio(tl, project)

    s002 = next(s for s in tl.tracks.video if s.id == "s002")
    assert s002.in_ == pytest.approx(3.0)   # untouched
    assert s002.mute_source is False
    assert ambient_repeat_count(changes) == 1
    assert any(AMBIENT_TAG in c for c in changes)


def test_ambient_repeat_can_be_disallowed(project: Project) -> None:
    add_clip(project, "c001", 60.0)
    tl = timeline_of(
        seg("s001", "c001", 0.0, 5.0, role="cutaway"),
        seg("s002", "c001", 3.0, 8.0, role="cutaway"),
    )
    tl, changes = dedupe_audio(tl, project, allow_ambient_repeat=False)

    s002 = next(s for s in tl.tracks.video if s.id == "s002")
    assert s002.in_ == pytest.approx(5.0)
    assert ambient_repeat_count(changes) == 0


# ----------------------------------------------------------------------
# the no-op case
# ----------------------------------------------------------------------
def test_nothing_changes_when_nothing_overlaps(project: Project) -> None:
    add_clip(project, "c001", 60.0, WORDS)
    tl = timeline_of(
        seg("s001", "c001", 0.5, 3.0, role="a-roll"),
        seg("s002", "c001", 3.0, 6.0, role="a-roll"),
    )
    tl, changes = dedupe_audio(tl, project)
    assert changes == []
    assert find_duplicate_audio(tl, project) == []
