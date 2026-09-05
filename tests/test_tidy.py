"""Tests for ``ytedit.ai.tidy`` — air around speech cuts and jump-cut merging.

Everything here is offline: synthetic transcripts and analyses on disk, no
media and no API calls.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from ytedit.ai.tidy import TidyError, pad_segments_to_speech, tidy
from ytedit.project import Project
from ytedit.timeline import Timeline, VideoSegment, new_timeline


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def add_clip(
    project: Project,
    clip_id: str,
    duration: float,
    words: Sequence[tuple[float, float, str]] = (),
    instructions: Sequence[tuple[float, float]] = (),
    takes: Sequence[dict[str, Any]] = (),
) -> None:
    """Register a clip and write its transcript / analysis."""
    project.add_clip(
        {"id": clip_id, "order": int(clip_id[1:]), "duration": duration,
         "width": 1920, "height": 1080, "orientation": "horizontal", "has_audio": True}
    )
    if words:
        project.transcript_path(clip_id).write_text(
            json.dumps(
                {
                    "clip": clip_id,
                    "language": "pl",
                    "words": [{"t": t, "s": s, "e": e} for s, e, t in words],
                }
            ),
            encoding="utf-8",
        )
    if instructions or takes:
        project.analysis_path(clip_id).write_text(
            json.dumps(
                {
                    "clip": clip_id,
                    "instructions": [
                        {"s": s, "e": e, "text": "cut this", "action": "drop"}
                        for s, e in instructions
                    ],
                    "takes": list(takes),
                }
            ),
            encoding="utf-8",
        )


def seg(
    seg_id: str, clip: str, start: float, end: float, **extra: Any
) -> VideoSegment:
    """Build a video segment."""
    return VideoSegment(id=seg_id, clip=clip, **{"in": start}, out=end, **extra)


def timeline_of(*segments: VideoSegment) -> Timeline:
    """Wrap segments in a timeline."""
    tl = new_timeline()
    tl.tracks.video = list(segments)
    return tl


#: Five well-spaced words: the padding never runs into a neighbour.
SPACED = [
    (1.0, 1.4, "Alfa"),
    (2.0, 2.4, "Beta"),
    (3.1, 3.5, "Gamma"),
    (5.0, 5.4, "Delta"),
    (10.0, 10.4, "Epsilon"),
]


# ----------------------------------------------------------------------
# the basic move
# ----------------------------------------------------------------------
def test_in_and_out_get_the_configured_air(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 3.1, 5.4)), project)
    moved = tl.tracks.video[0]
    assert moved.in_ == pytest.approx(2.8)   # 3.1 - 0.30
    assert moved.out == pytest.approx(5.85)  # 5.4 + 0.45
    assert changes == [
        "s001 in 3.10→2.80 (pad before 'Gamma')",
        "s001 out 5.40→5.85 (pad after 'Delta')",
    ]


def test_a_cut_far_from_speech_is_left_alone(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    # 5.4 -> 10.0 is 4.6 s of silence: neither end is within the 0.6 s window.
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 6.5, 9.0)), project)
    assert changes == []
    assert (tl.tracks.video[0].in_, tl.tracks.video[0].out) == (6.5, 9.0)


def test_air_that_already_exists_is_never_trimmed_back(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    # 0.5 s of air before 'Gamma' is more than the 0.30 s pad — leave it.
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 2.6, 4.0)), project)
    assert tl.tracks.video[0].in_ == 2.6
    assert not any(c.startswith("s001 in ") for c in changes)


def test_b_roll_without_a_transcript_is_untouched(project: Project) -> None:
    add_clip(project, "c002", 30.0)  # no transcript at all
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c002", 4.0, 8.0)), project)
    assert changes == []
    assert (tl.tracks.video[0].in_, tl.tracks.video[0].out) == (4.0, 8.0)


def test_muted_segments_are_untouched(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    tl, changes = pad_segments_to_speech(
        timeline_of(seg("s001", "c001", 3.1, 5.4, mute_source=True)), project
    )
    assert changes == []
    assert (tl.tracks.video[0].in_, tl.tracks.video[0].out) == (3.1, 5.4)


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_padding_stops_short_of_the_previous_word(project: Project) -> None:
    add_clip(project, "c001", 60.0, [(2.6, 3.0, "Beta"), (3.1, 3.5, "Gamma")])
    tl, _ = pad_segments_to_speech(timeline_of(seg("s001", "c001", 3.1, 3.5)), project)
    # 3.1 - 0.30 = 2.80 would swallow 'Beta'; stop 0.05 s after it instead.
    assert tl.tracks.video[0].in_ == pytest.approx(3.05)


def test_padding_stops_short_of_the_next_word(project: Project) -> None:
    add_clip(project, "c001", 60.0, [(3.1, 3.5, "Gamma"), (3.7, 4.1, "Delta")])
    tl, _ = pad_segments_to_speech(timeline_of(seg("s001", "c001", 3.1, 3.5)), project)
    assert tl.tracks.video[0].out == pytest.approx(3.65)


def test_padding_never_enters_an_excised_instruction(project: Project) -> None:
    add_clip(
        project, "c001", 60.0,
        [(4.1, 4.5, "Gamma"), (5.0, 5.4, "Delta")],
        instructions=[(0.0, 4.0)],
    )
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 4.05, 5.4)), project)
    # 4.10 - 0.30 = 3.80 sits inside the spoken instruction; clamp to its end.
    assert tl.tracks.video[0].in_ == pytest.approx(4.0)
    assert changes[0].startswith("s001 in 4.05→4.00")


def test_padding_never_enters_a_rejected_take(project: Project) -> None:
    # The kept take is the second attempt (2.5-6.0); the first (0.5-2.2) is a
    # flubbed line that must stay out of the cut on both sides.
    add_clip(
        project, "c001", 60.0,
        [(1.0, 1.4, "Fluff"), (2.6, 3.0, "Gamma"), (5.6, 5.9, "Delta")],
        takes=[{"topic": "x", "attempts": [{"s": 0.5, "e": 2.45}, {"s": 2.5, "e": 6.0}],
                "keep": 1}],
    )
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 2.6, 3.0)), project)
    # 2.60 - 0.30 = 2.30 would reach back into the rejected take: clamp to its end.
    assert tl.tracks.video[0].in_ == pytest.approx(2.45)
    assert changes[0].startswith("s001 in 2.60→2.45")

    # And a cut whose out would run into a *later* rejected range is clamped too.
    add_clip(
        project, "c002", 60.0,
        [(2.0, 2.4, "Beta")],
        takes=[{"topic": "y", "attempts": [{"s": 1.8, "e": 2.5}, {"s": 2.6, "e": 4.0}],
                "keep": 0}],
    )
    tl2, _ = pad_segments_to_speech(timeline_of(seg("s001", "c002", 1.9, 2.4)), project)
    assert tl2.tracks.video[0].out == pytest.approx(2.6)


def test_padding_never_goes_below_zero_or_past_the_clip(project: Project) -> None:
    add_clip(project, "c001", 3.0, [(0.05, 0.4, "Alfa"), (2.6, 2.95, "Omega")])
    tl, _ = pad_segments_to_speech(timeline_of(seg("s001", "c001", 0.05, 2.95)), project)
    assert tl.tracks.video[0].in_ == 0.0
    assert tl.tracks.video[0].out == pytest.approx(3.0)


def test_padding_never_overlaps_another_cut_of_the_same_clip(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    # Different grades keep the two cuts from being merged, so the clamp shows.
    tl, _ = pad_segments_to_speech(
        timeline_of(
            seg("s001", "c001", 1.0, 2.9, grade="warm"),
            seg("s002", "c001", 3.1, 5.4, grade="default"),
        ),
        project,
    )
    assert tl.tracks.video[1].in_ == pytest.approx(2.9)
    assert len(tl.tracks.video) == 2


# ----------------------------------------------------------------------
# snapping
# ----------------------------------------------------------------------
def test_a_cut_inside_a_word_snaps_to_the_word_then_pads(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    # in lands in the middle of 'Gamma' (3.1-3.5), out in the middle of 'Delta'.
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 3.3, 5.2)), project)
    assert tl.tracks.video[0].in_ == pytest.approx(2.8)
    assert tl.tracks.video[0].out == pytest.approx(5.85)
    assert "snap+pad before 'Gamma'" in changes[0]
    assert "snap+pad after 'Delta'" in changes[1]


# ----------------------------------------------------------------------
# jump cuts
# ----------------------------------------------------------------------
def test_tiny_same_clip_gaps_are_merged(project: Project) -> None:
    add_clip(project, "c002", 30.0)  # no transcript: only the merge rule applies
    tl, changes = pad_segments_to_speech(
        timeline_of(seg("s001", "c002", 1.0, 2.5), seg("s002", "c002", 2.6, 4.0)), project
    )
    assert len(tl.tracks.video) == 1
    assert (tl.tracks.video[0].id, tl.tracks.video[0].in_, tl.tracks.video[0].out) == (
        "s001", 1.0, 4.0,
    )
    assert changes == ["s002 merged into s001 (100 ms gap in c002)"]


def test_a_real_gap_is_not_merged(project: Project) -> None:
    add_clip(project, "c002", 30.0)
    tl, changes = pad_segments_to_speech(
        timeline_of(seg("s001", "c002", 1.0, 2.5), seg("s002", "c002", 3.0, 4.0)), project
    )
    assert len(tl.tracks.video) == 2
    assert changes == []


def test_different_clips_are_never_merged(project: Project) -> None:
    add_clip(project, "c002", 30.0)
    add_clip(project, "c003", 30.0)
    tl, _ = pad_segments_to_speech(
        timeline_of(seg("s001", "c002", 1.0, 2.5), seg("s002", "c003", 2.5, 4.0)), project
    )
    assert len(tl.tracks.video) == 2


def test_a_muted_cutaway_is_not_merged_into_a_live_one(project: Project) -> None:
    add_clip(project, "c002", 30.0)
    tl, _ = pad_segments_to_speech(
        timeline_of(
            seg("s001", "c002", 1.0, 2.5),
            seg("s002", "c002", 2.55, 4.0, mute_source=True),
        ),
        project,
    )
    assert len(tl.tracks.video) == 2


# ----------------------------------------------------------------------
# the ``ytedit tidy`` stage
# ----------------------------------------------------------------------
def test_tidy_without_a_timeline_raises(project: Project) -> None:
    with pytest.raises(TidyError):
        tidy(project)


def test_tidy_writes_the_timeline_and_a_backup(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    timeline_of(seg("s001", "c001", 3.1, 5.4)).save(project.timeline_file)

    result = tidy(project)
    assert result["written"] == "plan/timeline.json"
    assert result["backup"] and (project.plan_dir / "history" / result["backup"]).exists()
    assert Timeline.load(project.timeline_file).tracks.video[0].in_ == pytest.approx(2.8)
    assert project.stage_status("tidy") == "done"


def test_tidy_dry_run_changes_nothing_on_disk(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    timeline_of(seg("s001", "c001", 3.1, 5.4)).save(project.timeline_file)
    before = project.timeline_file.read_text(encoding="utf-8")

    result = tidy(project, dry_run=True)
    assert result["changes"] and result["written"] is None
    assert project.timeline_file.read_text(encoding="utf-8") == before
    assert not (project.plan_dir / "history").exists()


def test_tidy_writes_a_draft_for_a_human_edited_timeline(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    tl = timeline_of(seg("s001", "c001", 3.1, 5.4))
    tl.meta.edited_by_human = True
    tl.save(project.timeline_file)

    result = tidy(project)
    assert result["written"] == "plan/timeline.draft.json"
    assert Timeline.load(project.timeline_file).tracks.video[0].in_ == 3.1
    assert Timeline.load(project.plan_dir / "timeline.draft.json").tracks.video[0].in_ == (
        pytest.approx(2.8)
    )


def test_tidy_force_overwrites_a_human_edited_timeline(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    tl = timeline_of(seg("s001", "c001", 3.1, 5.4))
    tl.meta.edited_by_human = True
    tl.save(project.timeline_file)

    result = tidy(project, force=True)
    assert result["written"] == "plan/timeline.json"
    saved = Timeline.load(project.timeline_file)
    assert saved.tracks.video[0].in_ == pytest.approx(2.8)
    # A forced tidy does not un-mark the human's edit.
    assert saved.meta.edited_by_human is True


def test_tidy_reports_nothing_when_the_cuts_are_already_clean(project: Project) -> None:
    add_clip(project, "c002", 30.0)
    timeline_of(seg("s001", "c002", 1.0, 5.0)).save(project.timeline_file)
    result = tidy(project)
    assert result["changes"] == [] and result["written"] is None
