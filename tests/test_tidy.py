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
from ytedit.timeline import (
    AudioFrom,
    Caption,
    Timeline,
    VideoSegment,
    VoiceAnchor,
    VoiceItem,
    new_timeline,
)


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
# voice items (and other absolute-time tracks) shift with the padding
# ----------------------------------------------------------------------
def test_voice_items_shift_with_padding(project: Project) -> None:
    add_clip(project, "c001", 60.0, SPACED)
    # s001 pads from (3.1, 5.4) [dur 2.3] to (2.8, 5.85) [dur 3.05]: +0.75s.
    # s002 is a muted VO picture cut that starts right after it and must not
    # itself be padded, but its absolute position (and anything pinned to it)
    # still needs to move by the growth s001 picked up.
    tl = timeline_of(
        seg("s001", "c001", 3.1, 5.4),
        seg("s002", "c001", 10.0, 12.0, mute_source=True, role="b-roll",
            notes="VO picture for c004"),
    )
    tl.tracks.voice = [
        VoiceItem(id="v001", file="voice/vo_c004_000.00_002.00.wav", at=2.3, end=4.3)
    ]
    tl.tracks.captions = [Caption(id="t001", at=2.3, end=3.0, text="LIZBONA")]

    tl2, changes = pad_segments_to_speech(tl, project)
    assert changes  # s001 was in fact padded

    # Positions are frame-exact: 3.05 s is 91.5 frames at 30 fps -> 92 frames.
    frame = 1 / 30
    expected_start = 92 * frame
    seg2_start = next(p for p in tl2.segment_positions() if p.segment.id == "s002").start
    assert seg2_start == pytest.approx(expected_start, abs=1e-3)

    voice = tl2.tracks.voice[0]
    assert voice.at == pytest.approx(expected_start, abs=1e-3)
    assert voice.end == pytest.approx(expected_start + 2.0, abs=1e-3)  # duration preserved

    caption = tl2.tracks.captions[0]
    assert caption.at == pytest.approx(expected_start, abs=1e-3)
    assert caption.end == pytest.approx(expected_start + 0.7, abs=1e-3)  # duration preserved


def test_anchored_voice_item_follows_its_segment_through_padding_and_a_merge(
    project: Project,
) -> None:
    """An anchored pickup tracks its segment through the whole ``tidy`` stage.

    ``s001`` grows from padding (shifting everything after it), and ``s003``
    is merged away entirely (a tiny same-clip jump cut) — the item stays
    pinned to ``s004`` throughout, never at the absolute time it started at.
    """
    add_clip(project, "c001", 60.0, SPACED)
    add_clip(project, "c002", 30.0)
    add_clip(project, "c003", 30.0)
    tl = timeline_of(
        seg("s001", "c001", 3.1, 5.4),  # pads to (2.8, 5.85): grows the timeline
        seg("s002", "c002", 1.0, 2.5),  # merges with s003 (0.05s gap): s003 dropped
        seg("s003", "c002", 2.55, 4.0),
        seg("s004", "c003", 5.0, 8.0),  # the anchor target
    )
    tl.tracks.voice = [
        VoiceItem(
            id="v001", file="voice/n001.wav", at=999.0, end=1000.0,
            anchor=VoiceAnchor(segment="s004", offset=0.25),
        )
    ]
    tl.save(project.timeline_file)

    result = tidy(project)
    assert result["written"] == "plan/timeline.json"

    saved = Timeline.load(project.timeline_file)
    assert "s003" not in [s.id for s in saved.tracks.video]  # earlier segment dropped

    target_start = next(
        p.start for p in saved.segment_positions() if p.segment.id == "s004"
    )
    item = saved.tracks.voice[0]
    assert item.anchor is not None and item.anchor.segment == "s004"
    assert item.at == pytest.approx(target_start + 0.25, abs=1e-3)
    assert item.end == pytest.approx(item.at + 1.0, abs=1e-3)  # length (1000-999) kept


def test_voice_items_are_untouched_when_nothing_moves(project: Project) -> None:
    add_clip(project, "c002", 30.0)  # no transcript: padding never fires
    tl = timeline_of(seg("s001", "c002", 1.0, 5.0))
    tl.tracks.voice = [VoiceItem(id="v001", file="voice/vo_c004_000.00_002.00.wav", at=1.0, end=3.0)]

    tl2, changes = pad_segments_to_speech(tl, project)
    assert changes == []
    assert tl2.tracks.voice[0].at == 1.0
    assert tl2.tracks.voice[0].end == 3.0


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


# ----------------------------------------------------------------------
# sentence-boundary snapping
# ----------------------------------------------------------------------
#: Two sentences 1.0 s apart; the second one runs on well past a naive pad.
TWO_SENTENCES = [
    (1.0, 1.4, "Alfa"),
    (1.6, 3.0, "Beta."),
    (4.0, 4.4, "Gamma"),
    (4.6, 6.0, "Delta."),
    (7.0, 7.4, "Epsilon"),
    (7.6, 9.0, "Zeta."),
]

#: The first ~15 s of ``projects/the reference project/transcripts/c030.json`` — the real cut
#: that motivated sentence snapping (see ``test_the_real_c030_cut_*``).
C030 = [
    (2.18, 2.579, "Jestem"), (2.679, 2.779, "na"), (2.819, 3.259, "drodze"),
    (3.399, 3.659, "do"), (4.599, 4.799, "Góry"), (4.9, 5.159, "Siedmiu"),
    (5.199, 5.719, "Kolorów."),
    (6.42, 6.679, "Obecnie"), (6.719, 6.759, "na"), (6.819, 7.539, "wysokości"),
    (8.5, 8.76, "cztery"), (8.88, 9.26, "tysiące"), (9.38, 9.619, "sześćset"),
    (9.699, 10.099, "metrów."),
    (10.979, 11.34, "Wchodzimy"), (11.399, 11.46, "na"), (11.559, 11.699, "pięć"),
    (11.8, 12.339, "tysięcy."),
    (14.139, 14.679, "Dokładnie"), (14.739, 14.819, "tam"), (14.859, 15.399, "wchodzimy."),
]


def test_the_real_c030_cut_finishes_its_sentence(project: Project) -> None:
    """s036 ``c030 1.8-7.989`` stopped on 'wysokości'; it must reach 'metrów.'."""
    add_clip(project, "c030", 60.0, C030)
    tl, changes = pad_segments_to_speech(timeline_of(seg("s036", "c030", 1.8, 7.989)), project)
    moved = tl.tracks.video[0]
    assert moved.in_ == 1.8                    # already had more than 0.3 s of air
    assert moved.out == pytest.approx(10.549)  # 'metrów.' ends 10.099, + 0.45
    assert changes == ["s036 out 7.99→10.55 (sentence-snap to 'metrów.')"]


def test_a_cut_on_a_sentence_boundary_is_not_snapped(project: Project) -> None:
    """s038 ``c030 13.8-21.97`` deliberately skips a whole sentence — leave it."""
    add_clip(project, "c030", 60.0, C030)
    tl, changes = pad_segments_to_speech(timeline_of(seg("s038", "c030", 13.8, 21.97)), project)
    assert (tl.tracks.video[0].in_, tl.tracks.video[0].out) == (13.8, 21.97)
    assert changes == []


def test_out_snaps_forward_to_the_end_of_the_sentence(project: Project) -> None:
    add_clip(project, "c001", 60.0, TWO_SENTENCES)
    # 4.5 sits between 'Gamma' and 'Delta.' — mid-sentence.
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 3.9, 4.5)), project)
    assert tl.tracks.video[0].out == pytest.approx(6.45)  # 6.0 + 0.45
    assert any("sentence-snap to 'Delta.'" in c for c in changes)


def test_out_does_not_snap_across_a_gap_wider_than_sentence_gap_max(project: Project) -> None:
    # 'Beta' does not close a sentence, but the next word is 2.0 s away.
    add_clip(project, "c001", 60.0, [(1.0, 1.4, "Alfa"), (1.6, 2.0, "Beta"), (4.0, 4.4, "Gamma.")])
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 0.9, 2.45)), project)
    assert tl.tracks.video[0].out == pytest.approx(2.45)
    assert not any("sentence-snap" in c for c in changes)


def test_in_snaps_back_to_the_start_of_the_sentence(project: Project) -> None:
    add_clip(
        project, "c001", 60.0,
        [(1.0, 1.4, "Alfa."), (2.0, 2.4, "Beta"), (2.6, 3.0, "Gamma"),
         (3.2, 3.6, "Delta."), (6.0, 6.4, "Eps.")],
    )
    # The cut opens on 'Gamma', in the middle of "Beta Gamma Delta."
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 2.55, 3.9)), project)
    assert tl.tracks.video[0].in_ == pytest.approx(1.7)   # 'Beta' starts 2.0, − 0.30
    assert any("sentence-snap to 'Beta'" in c for c in changes)


def test_in_does_not_snap_when_the_cut_already_opens_a_sentence(project: Project) -> None:
    add_clip(project, "c001", 60.0, TWO_SENTENCES)
    # 'Gamma' opens its sentence ('Beta.' before it closed one), so only the
    # ordinary 0.30 s pad applies to ``in`` — no reach back into "Alfa Beta.".
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 3.9, 6.45)), project)
    assert tl.tracks.video[0].in_ == pytest.approx(3.7)
    assert not any(c.startswith("s001 in ") and "sentence-snap" in c for c in changes)


def test_a_sentence_longer_than_the_cap_is_only_extended_by_the_cap(project: Project) -> None:
    # One 20 s sentence: a word every second, the full stop only at the very end.
    words = [(float(i), i + 0.4, f"w{i}") for i in range(1, 20)] + [(20.0, 20.4, "w20.")]
    add_clip(project, "c001", 60.0, words)
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 0.8, 2.9)), project)
    # 2.9 + pacing.sentence_extend_max (8.0), not 20.4 + 0.45.
    assert tl.tracks.video[0].out == pytest.approx(10.9)
    assert any("sentence-snap" in c for c in changes)


def test_a_sentence_snap_never_enters_an_excised_instruction(project: Project) -> None:
    add_clip(project, "c030", 60.0, C030, instructions=[(9.0, 9.5)])
    tl, changes = pad_segments_to_speech(timeline_of(seg("s036", "c030", 1.8, 7.989)), project)
    # The sentence ends at 10.099 but a spoken editor instruction starts at 9.0.
    assert tl.tracks.video[0].out == pytest.approx(9.0)
    assert any("sentence-snap" in c for c in changes)


def test_a_sentence_snap_never_enters_the_next_cut_of_the_same_clip(project: Project) -> None:
    add_clip(project, "c030", 60.0, C030)
    tl, _ = pad_segments_to_speech(
        timeline_of(
            seg("s036", "c030", 1.8, 7.989, grade="warm"),
            seg("s037", "c030", 9.4, 12.4, grade="default"),
        ),
        project,
    )
    assert tl.tracks.video[0].out == pytest.approx(9.4)


def test_an_unpunctuated_transcript_is_never_sentence_snapped(project: Project) -> None:
    """Without a single full stop there is no sentence to snap to — only pad."""
    add_clip(project, "c001", 60.0, SPACED)
    tl, changes = pad_segments_to_speech(timeline_of(seg("s001", "c001", 3.1, 5.4)), project)
    assert not any("sentence-snap" in c for c in changes)
    assert (tl.tracks.video[0].in_, tl.tracks.video[0].out) == (
        pytest.approx(2.8), pytest.approx(5.85),
    )


# ----------------------------------------------------------------------
# true breaks: a boundary is only a cut if the audio actually stops there
# ----------------------------------------------------------------------
def test_a_continuous_audio_from_handoff_is_never_sentence_snapped(project: Project) -> None:
    """``s001``'s ``out`` lands mid-sentence, but ``s002`` (a cutaway on a
    different clip) picks up ``c001``'s audio with zero gap via ``audio_from``
    — that hand-off is not a cut at all, so sentence snapping must leave it
    alone even though, taken on its own, ``s001`` looks like it wants
    extending all the way out to "Delta.".
    """
    add_clip(
        project, "c001", 60.0,
        [(1.0, 1.4, "Alfa"), (1.6, 3.0, "Beta."),
         (4.0, 4.4, "Gamma"), (4.6, 6.0, "Delta.")],
    )
    add_clip(project, "c900", 20.0)
    cutaway = VideoSegment(
        id="s002", clip="c900", **{"in": 0.0}, out=2.5, role="cutaway",
        audio_from=AudioFrom(clip="c001", **{"in": 4.5}, out=7.0),
    )
    tl = timeline_of(seg("s001", "c001", 0.9, 4.5, role="a-roll"), cutaway)
    tl, changes = pad_segments_to_speech(tl, project)

    moved = next(s for s in tl.tracks.video if s.id == "s001")
    assert moved.out == pytest.approx(4.55)   # ordinary word-level pad only
    assert not any(
        c.startswith("s001 out") and ("sentence-snap" in c or "sentence-crop" in c)
        for c in changes
    )


def test_a_true_mid_sentence_break_retracts_when_extension_is_blocked(
    project: Project,
) -> None:
    """``s001``'s cut lands between two sentences but a later same-clip cut
    starting exactly where it stops blocks any forward extension — retract to
    the sentence already finished instead of leaving the cut mid-thought.
    """
    add_clip(
        project, "c001", 60.0,
        [(1.0, 1.4, "Alfa"), (1.6, 3.0, "Beta."),
         (4.0, 4.4, "Gamma"), (4.6, 6.0, "Delta"),
         (6.2, 7.0, "Epsilon.")],
    )
    add_clip(project, "c002", 20.0)
    tl = timeline_of(
        seg("s001", "c001", 0.9, 6.05, role="a-roll"),
        seg("s002", "c002", 0.0, 1.0, role="cutaway"),   # not a continuation of s001
        seg("s003", "c001", 6.05, 20.0, role="a-roll"),  # blocks extension past 6.05
    )
    tl, changes = pad_segments_to_speech(tl, project)

    moved = next(s for s in tl.tracks.video if s.id == "s001")
    assert moved.out == pytest.approx(3.45)   # retract to 'Beta.' (3.0) + pad_after (0.45)
    assert any(
        c.startswith("s001 out") and "sentence-crop" in c and "retract" in c
        for c in changes
    )


def test_a_true_mid_sentence_open_advances_to_the_next_sentence_when_blocked(
    project: Project,
) -> None:
    """``s001`` opens mid-sentence but an earlier same-clip cut ending exactly
    at its ``in`` blocks any backward reach — crop the half-spoken leading
    fragment and start clean at the next full sentence instead.
    """
    add_clip(
        project, "c001", 60.0,
        [(1.0, 1.4, "Alfa"), (1.6, 3.0, "Beta."),
         (4.0, 4.4, "Gamma"), (4.6, 6.0, "Delta."),
         (7.0, 7.4, "Epsilon"), (7.6, 9.0, "Zeta.")],
    )
    add_clip(project, "c002", 20.0)
    tl = timeline_of(
        seg("s000", "c001", 0.0, 4.2, role="a-roll"),    # blocks retreat past 4.2
        seg("s00a", "c002", 0.0, 1.0, role="cutaway"),   # not a continuation of s001
        seg("s001", "c001", 4.2, 9.5, role="a-roll"),
    )
    tl, changes = pad_segments_to_speech(tl, project)

    moved = next(s for s in tl.tracks.video if s.id == "s001")
    assert moved.in_ == pytest.approx(6.7)    # crops "Gamma Delta.", starts at 'Epsilon'
    assert any(
        c.startswith("s001 in") and "sentence-crop" in c and "advance" in c
        for c in changes
    )


# ----------------------------------------------------------------------
# a sentence snap must never reach into audio another segment already
# claims — even when the direct neighbour showing that no longer says so
# ----------------------------------------------------------------------
def test_a_sentence_snap_never_reaches_into_audio_a_cutaway_already_carries(
    project: Project,
) -> None:
    """The exact failure mode ``ytedit tidy`` used to churn on: a cutaway
    right before this segment carries ``c060`` audio nobody can see from the
    word-level ``_clip_bounds`` check (it lives on a *different* clip), and
    the ledger has since muted that particular hand-off (``s002``) for
    reasons of its own — but the stretch is still spoken for by another,
    still-intact cutaway (``s001b``) earlier in the track. Sentence-snapping
    ``s003.in`` back into "Mozna czasem tam..." would resurrect a duplicate
    even though the *immediate* neighbour no longer shows any continuity.
    """
    add_clip(
        project, "c060", 60.0,
        [
            (0.0, 0.4, "Alfa"), (0.6, 1.0, "Beta."),
            (1.5, 1.9, "Mozna"), (2.1, 2.5, "czasem"), (2.7, 3.1, "tam"),
            (3.3, 3.7, "chodzic"), (3.9, 4.3, "a"), (4.5, 4.9, "potem"),
            (5.1, 5.5, "wracac."),
        ],
    )
    add_clip(project, "c070", 20.0)   # the ledger-muted cutaway's own clip
    add_clip(project, "c071", 20.0)   # the still-intact cutaway's own clip
    tl = timeline_of(
        seg("s001", "c060", 0.0, 1.0, role="a-roll"),
        seg("s001b", "c071", 0.0, 3.95, role="cutaway",
            audio_from=AudioFrom(clip="c060", **{"in": 1.0}, out=4.95)),
        seg("s002", "c070", 0.0, 1.0, role="cutaway", mute_source=True),
        seg("s003", "c060", 4.95, 8.0, role="a-roll"),
    )
    tl, changes = pad_segments_to_speech(tl, project)

    moved = next(s for s in tl.tracks.video if s.id == "s003")
    assert moved.in_ == pytest.approx(4.95)   # never snapped back into 1.0-4.95
    assert not any(c.startswith("s003 in") for c in changes)


# ----------------------------------------------------------------------
# the ``ytedit tidy`` convergence loop
# ----------------------------------------------------------------------
#: A-roll, cutaway, a-roll resuming mid-sentence earlier than the first
#: piece's out — the exact shape reported as non-idempotent in
#: ``projects/the reference project``: the sentence snap reaches ``s042`` back across the
#: cutaway, then the overlay pass hands that same stretch to the cutaway and
#: pushes the cut back out again.
_CHURN_WORDS = [
    (0.0, 0.4, "Jeden"), (0.6, 1.0, "dwa."),
    (1.5, 1.9, "Mozna"), (2.1, 2.5, "tam"), (2.7, 3.1, "isc."),
]


def test_tidy_converges_on_the_reported_churn_pattern(project: Project) -> None:
    """``s042`` opens mid-sentence, well before ``s041``'s own out; a cutaway
    sits between them. One round used to leave this oscillating forever (see
    ``ytedit/ai/tidy.py`` module docstring history): pad's sentence snap pulls
    ``s042.in`` back to "Mozna", then overlay hands that stretch to the
    cutaway and pushes ``s042.in`` forward again — right back to a value that,
    on the *next* run, looked mid-sentence all over again. ``tidy()`` must
    settle this within its own call and a second, independent call must be a
    complete no-op.
    """
    add_clip(project, "c050", 60.0, _CHURN_WORDS)
    add_clip(project, "c090", 20.0)
    timeline_of(
        seg("s041", "c050", 0.0, 1.0, role="a-roll"),
        seg("s050", "c090", 0.0, 3.5, role="cutaway"),
        seg("s042", "c050", 2.0, 7.0, role="a-roll"),
    ).save(project.timeline_file)

    result = tidy(project)
    assert result["converged"] is True
    assert result["changes"][-1] == f"converged in {result['rounds']} round(s)"
    assert result["rounds"] >= 2   # at least one round of churn, then a quiet one
    assert result["issues"] == []
    first_bytes = project.timeline_file.read_bytes()

    # A second, independent ``ytedit tidy`` run (a fresh load from disk, the
    # same as a second CLI invocation) must find nothing left to do.
    result2 = tidy(project)
    assert result2["changes"] == []
    assert result2["written"] is None
    assert result2["rounds"] == 1
    assert project.timeline_file.read_bytes() == first_bytes

    saved = Timeline.load(project.timeline_file)
    from ytedit.ai.ledger import find_duplicate_audio

    assert find_duplicate_audio(saved, project) == []
    s042 = next(s for s in saved.tracks.video if s.id == "s042")
    s050 = next(s for s in saved.tracks.video if s.id == "s050")
    # s042 resumed exactly where the cutaway's carried narration ends — not
    # back at the sentence start the first round snapped it to.
    assert s050.audio_from is not None
    assert s042.in_ == pytest.approx(s050.audio_from.out, abs=1e-3)


def test_tidy_stops_at_max_rounds_and_says_so(project: Project) -> None:
    """``pacing.tidy_max_rounds`` caps the loop; hitting it is reported
    honestly (``converged: False``) rather than silently returning a
    half-settled timeline.
    """
    from ytedit.config import load_settings

    add_clip(project, "c050", 60.0, _CHURN_WORDS)
    add_clip(project, "c090", 20.0)
    timeline_of(
        seg("s041", "c050", 0.0, 1.0, role="a-roll"),
        seg("s050", "c090", 0.0, 3.5, role="cutaway"),
        seg("s042", "c050", 2.0, 7.0, role="a-roll"),
    ).save(project.timeline_file)
    project._settings = load_settings(project.path, overrides={"pacing": {"tidy_max_rounds": 1}})

    result = tidy(project)
    assert result["converged"] is False
    assert result["rounds"] == 1
    assert result["changes"][-1].startswith(
        "stopped: pacing.tidy_max_rounds (1) reached without convergence"
    )
