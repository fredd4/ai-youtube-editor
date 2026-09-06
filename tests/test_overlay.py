"""Tests for ``ytedit.ai.overlay`` — cutaways that keep the narration running.

Offline like ``test_tidy.py``: synthetic transcripts on disk, no media, no APIs.
The timings in :data:`C030` come from ``projects/the reference project/transcripts/c030.json``,
the real cut this pass was written for.
"""

from __future__ import annotations

import pytest

from tests.test_tidy import C030, add_clip, seg, timeline_of
from ytedit.ai.ledger import dedupe_audio, find_duplicate_audio
from ytedit.ai.overlay import overlay_cutaways
from ytedit.ai.tidy import pad_segments_to_speech
from ytedit.project import Project
from ytedit.timeline import Caption, MusicCue, Timeline, VoiceItem


def tidy_then_overlay(tl: Timeline, project: Project) -> tuple[Timeline, list[str]]:
    """Run the two deterministic passes in the order ``build_timeline`` uses."""
    tl, _padded = pad_segments_to_speech(tl, project)
    return overlay_cutaways(tl, project)


#: Three well-separated sentences on one clip, used for the chained pattern.
THREE_SENTENCES = [
    (1.0, 1.4, "Alfa"), (1.6, 3.0, "Beta."),
    (4.0, 4.4, "Gamma"), (4.6, 6.0, "Delta."),
    (7.0, 7.4, "Epsilon"), (7.6, 9.0, "Zeta."),
]


# ----------------------------------------------------------------------
# the genuine continuation
# ----------------------------------------------------------------------
def test_a_cutaway_inside_one_take_carries_the_narration(project: Project) -> None:
    """A-C-A2 on the same take: the cutaway is heard as the narration under it."""
    add_clip(project, "c030", 60.0, C030)
    add_clip(project, "c033", 20.0)                     # cutaway footage, no words
    tl = timeline_of(
        seg("s001", "c030", 1.8, 7.989, role="a-roll"),
        seg("s002", "c033", 0.0, 3.0, role="cutaway"),
        seg("s003", "c030", 10.6, 15.9, role="a-roll"),
    )
    tl, changes = tidy_then_overlay(tl, project)
    a, cutaway, a2 = tl.tracks.video

    # the sentence snap finished "…cztery tysiące sześćset metrów." first
    assert a.out == pytest.approx(10.549)
    assert cutaway.audio_from is not None
    assert cutaway.audio_from.clip == "c030"
    assert cutaway.audio_from.in_ == pytest.approx(10.549)
    assert cutaway.audio_from.out == pytest.approx(13.549)   # 10.549 + 3.0
    assert cutaway.mute_source is False
    assert a2.in_ == pytest.approx(13.549)
    assert any("audio_from c030" in c for c in changes)
    assert changes[-1].endswith("carry the narration from underneath")


def test_the_real_c030_skip_is_left_alone(project: Project) -> None:
    """The cut that motivated this feature: s036-s037-s038 is *not* an overlay.

    ``s038`` opens a brand new sentence 13.8 s in — the planner dropped
    "Wchodzimy na pięć tysięcy." on purpose. 13.8 > 10.549 + 1.2, so the pass
    must do nothing at all: the sentence snap alone already fixed the picture.
    """
    add_clip(project, "c030", 60.0, C030)
    add_clip(project, "c033", 20.0)
    tl = timeline_of(
        seg("s036", "c030", 1.8, 7.989, role="a-roll"),
        seg("s037", "c033", 0.0, 4.0, role="cutaway"),
        seg("s038", "c030", 13.8, 21.97, role="a-roll"),
    )
    tl, changes = tidy_then_overlay(tl, project)
    a, cutaway, a2 = tl.tracks.video

    assert a.out == pytest.approx(10.549)
    assert cutaway.audio_from is None
    assert cutaway.mute_source is False        # untouched, keeps its own ambience
    assert a2.in_ == pytest.approx(13.8)
    assert changes == []


def test_a_continuation_past_the_cutaways_is_left_alone(project: Project) -> None:
    """``A2`` within ``sentence_gap_max`` but past ``A.out + D`` is still a skip.

    Stretching the narration over the cutaway would replay 0.25 s that the
    planner cut on purpose, so the whole run is left as it is. The cuts here are
    already tidy, so the pass is called on its own.
    """
    add_clip(project, "c030", 60.0, C030)
    add_clip(project, "c033", 20.0)
    tl = timeline_of(
        seg("s001", "c030", 1.8, 10.549, role="a-roll"),
        seg("s002", "c033", 0.0, 0.5, role="cutaway"),      # only 0.5 s of cover
        seg("s003", "c030", 11.3, 14.5, role="a-roll"),     # 0.751 s later
    )
    tl, changes = overlay_cutaways(tl, project)
    assert tl.tracks.video[1].audio_from is None
    assert tl.tracks.video[2].in_ == pytest.approx(11.3)
    assert changes == []


# ----------------------------------------------------------------------
# a continuation too short to survive the move
# ----------------------------------------------------------------------
def test_a_continuation_shorter_than_min_shot_is_dropped(project: Project) -> None:
    add_clip(project, "c030", 60.0, C030)
    add_clip(project, "c033", 5.0)                    # short clip: the clamp bites
    tl = timeline_of(
        seg("s001", "c030", 1.8, 7.989, role="a-roll"),
        seg("s002", "c033", 0.0, 3.0, role="cutaway"),
        seg("s003", "c030", 10.6, 14.0, role="a-roll"),
    )
    tl, changes = tidy_then_overlay(tl, project)

    # 10.549 + 3.0 = 13.549 leaves 0.451 s of s003 — under min_shot_seconds.
    assert [s.id for s in tl.tracks.video] == ["s001", "s002"]
    cutaway = tl.tracks.video[1]
    # 3.0 + 3.4 = 6.4 s of picture would be wanted, but only 14.0 - 10.549 =
    # 3.451 s of narration audio actually exists past where the cutaway's
    # audio_from starts (a2.out is where the planner ended the sentence) — the
    # picture is clamped there, not to the clip's own (longer) 5.0 s duration,
    # so the render never plays audio past what was actually written.
    assert cutaway.out == pytest.approx(3.451)
    assert cutaway.audio_from.in_ == pytest.approx(10.549)
    assert cutaway.audio_from.out == pytest.approx(14.0)   # == a2.out, never past it
    assert any("dropped" in c for c in changes)


# ----------------------------------------------------------------------
# chains
# ----------------------------------------------------------------------
def test_two_overlay_patterns_back_to_back_both_fire(project: Project) -> None:
    add_clip(project, "c001", 60.0, THREE_SENTENCES)
    add_clip(project, "c002", 20.0)
    add_clip(project, "c003", 20.0)
    tl = timeline_of(
        seg("s001", "c001", 0.7, 3.45, role="a-roll"),
        seg("s002", "c002", 0.0, 1.0, role="cutaway"),
        seg("s003", "c001", 3.7, 6.45, role="a-roll"),
        seg("s004", "c003", 0.0, 1.0, role="b-roll"),
        seg("s005", "c001", 6.7, 9.45, role="a-roll"),
    )
    tl, changes = tidy_then_overlay(tl, project)
    _a1, c1, a2, c2, a3 = tl.tracks.video

    assert (c1.audio_from.clip, c1.audio_from.in_, c1.audio_from.out) == ("c001", 3.45, 4.45)
    assert a2.in_ == pytest.approx(4.45)
    assert (c2.audio_from.clip, c2.audio_from.in_, c2.audio_from.out) == ("c001", 6.45, 7.45)
    assert a3.in_ == pytest.approx(7.45)
    assert changes[-1].startswith("2 cutaway run(s)")


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------
def test_a_voice_over_picture_cut_is_never_an_overlay_cutaway(project: Project) -> None:
    """VO picture cuts already get their audio from ``tracks.voice``."""
    add_clip(project, "c001", 60.0, THREE_SENTENCES)
    add_clip(project, "c002", 20.0)
    tl = timeline_of(
        seg("s001", "c001", 0.7, 3.45, role="a-roll"),
        seg("s002", "c002", 0.0, 1.0, role="b-roll", mute_source=True,
            notes="VO picture for c009"),
        seg("s003", "c001", 3.7, 6.45, role="a-roll"),
    )
    tl, changes = tidy_then_overlay(tl, project)
    assert tl.tracks.video[1].audio_from is None
    assert tl.tracks.video[1].mute_source is True
    assert changes == []


def test_a_cutaway_on_the_same_clip_is_not_an_overlay(project: Project) -> None:
    add_clip(project, "c001", 60.0, THREE_SENTENCES)
    tl = timeline_of(
        seg("s001", "c001", 0.7, 3.45, role="a-roll"),
        seg("s002", "c001", 20.0, 21.0, role="cutaway"),
        seg("s003", "c001", 3.7, 6.45, role="a-roll"),
    )
    tl, changes = tidy_then_overlay(tl, project)
    assert tl.tracks.video[1].audio_from is None
    assert changes == []


def test_a_segment_that_is_not_a_cutaway_role_breaks_the_run(project: Project) -> None:
    add_clip(project, "c001", 60.0, THREE_SENTENCES)
    add_clip(project, "c002", 20.0)
    tl = timeline_of(
        seg("s001", "c001", 0.7, 3.45, role="a-roll"),
        seg("s002", "c002", 0.0, 1.0, role="transition"),
        seg("s003", "c001", 3.7, 6.45, role="a-roll"),
    )
    tl, changes = tidy_then_overlay(tl, project)
    assert tl.tracks.video[1].audio_from is None
    assert changes == []


# ----------------------------------------------------------------------
# the absolute-time tracks follow the rewrite
# ----------------------------------------------------------------------
def test_captions_music_and_voice_are_retimed_after_a_rewrite(project: Project) -> None:
    add_clip(project, "c030", 60.0, C030)
    add_clip(project, "c033", 20.0)
    add_clip(project, "c040", 20.0)
    tl = timeline_of(
        seg("s001", "c030", 1.8, 7.989, role="a-roll"),
        seg("s002", "c033", 0.0, 3.0, role="cutaway"),
        seg("s003", "c030", 10.6, 15.9, role="a-roll"),
        seg("s004", "c040", 0.0, 2.0, role="b-roll", mute_source=True),
    )
    tl, _padded = pad_segments_to_speech(tl, project)
    tail_before = tl.segment_positions()[3].start
    tl.tracks.captions = [Caption(id="t001", at=tail_before, end=tail_before + 1.0, text="X")]
    tl.tracks.music = [MusicCue(id="m001", file="music/bed.wav", at=tail_before,
                                end=tail_before + 1.0)]
    tl.tracks.voice = [VoiceItem(id="v001", file="voice/v001.wav", at=tail_before,
                                 end=tail_before + 1.0)]

    tl, changes = overlay_cutaways(tl, project)
    assert changes
    tail_after = tl.segment_positions()[3].start
    assert tail_after < tail_before                      # s003 lost its head

    assert tl.tracks.captions[0].at == pytest.approx(tail_after, abs=1e-3)
    assert tl.tracks.captions[0].end == pytest.approx(tail_after + 1.0, abs=1e-3)
    assert tl.tracks.music[0].at == pytest.approx(tail_after, abs=1e-3)
    assert tl.tracks.voice[0].at == pytest.approx(tail_after, abs=1e-3)
    assert tl.tracks.voice[0].end == pytest.approx(tail_after + 1.0, abs=1e-3)


def test_nothing_moves_when_there_is_no_overlay_pattern(project: Project) -> None:
    add_clip(project, "c001", 60.0, THREE_SENTENCES)
    tl = timeline_of(seg("s001", "c001", 0.7, 3.45, role="a-roll"))
    tl.tracks.captions = [Caption(id="t001", at=0.5, end=1.5, text="X")]
    tl, changes = overlay_cutaways(tl, project)
    assert changes == []
    assert tl.tracks.captions[0].at == 0.5


# ----------------------------------------------------------------------
# the per-clip "audio consumed up to" cursor (item 2 of the audio-dedupe fix)
# ----------------------------------------------------------------------
def test_a_stray_reappearance_is_advanced_past_audio_already_committed(
    project: Project,
) -> None:
    """A clip resurfaces outside any cutaway run, independently overlapping
    the stretch of its own audio an earlier, unrelated pattern in the same
    pass already handed to a cutaway.

    ``s004`` breaks the cutaway-run chain (its role is not a cutaway/b-roll
    role), so ``s005`` is a fresh pattern start, not ``s003``'s continuation —
    but its own ``[in, out)`` still reaches back into the ``c001`` stretch
    ``s002`` (pattern 1's cutaway) already carries as narration. Without a
    cursor spanning the whole pass, only immediately-adjacent continuations
    get caught and this reappearance would play that stretch a second time;
    what is left of it (0.15 s) is too short to stand as its own shot, so it
    is dropped rather than merely trimmed.
    """
    add_clip(project, "c001", 60.0, THREE_SENTENCES)
    add_clip(project, "c002", 20.0)
    tl = timeline_of(
        seg("s001", "c001", 0.7, 3.45, role="a-roll"),
        seg("s002", "c002", 0.0, 1.0, role="cutaway"),
        seg("s003", "c001", 3.7, 6.45, role="a-roll"),
        seg("s004", "c002", 5.0, 6.0, role="transition"),  # not a cutaway role
        seg("s005", "c001", 4.0, 4.6, role="a-roll"),
        seg("s006", "c002", 0.0, 1.0, role="cutaway"),   # trailer: s005 must not be last
    )
    tl, changes = overlay_cutaways(tl, project)

    by_id = {s.id: s for s in tl.tracks.video}
    assert by_id["s003"].in_ == pytest.approx(4.45)   # the ordinary pattern-1 resume
    assert "s005" not in by_id                        # 4.6 - 4.45 = 0.15s: below min_shot
    assert any("s005 dropped" in c and "already carried" in c for c in changes)
    assert [f for f in find_duplicate_audio(tl, project) if f.speech] == []


def test_the_real_c075_chain_ends_with_no_duplicate_audio(project: Project) -> None:
    """The exact shape audited in ``projects/the reference project/plan/timeline.json``: a
    cutaway (``s021``) was hand-inserted into an already-overlaid
    A -> cutaway -> A2 chain without re-running ``ytedit tidy`` first. The
    original A2 (``s022``) now sits entirely inside audio the first cutaway
    already carries, and the following cutaway's stale ``audio_from``
    (``s023``) overlaps the second cutaway's freshly (re)computed one.

    Running overlay — which repairs what it can and advances the picture cut
    resuming afterwards (``s024``) via the cross-pattern cursor — and then the
    audio ledger dedupe pass, which mops up what overlay cannot fix on its own
    (a sibling cutaway's stale, untouched ``audio_from``), must leave no
    speech range playing twice.
    """
    add_clip(project, "c075", 60.0, [
        (2.0, 2.4, "Raz"), (9.0, 9.4, "Dwa"), (12.0, 12.4, "Trzy"),
        (16.0, 16.4, "Cztery"), (19.5, 19.9, "Piec"), (24.0, 24.4, "Szesc"),
    ])
    add_clip(project, "c010", 20.0)
    add_clip(project, "c104", 20.0)
    add_clip(project, "c020", 20.0)

    tl = timeline_of(
        seg("s018", "c075", 0.9, 10.569, role="a-roll"),
        seg("s019", "c010", 1.0, 10.7, role="cutaway",
            audio_from={"clip": "c075", "in": 10.569, "out": 20.269}),
        seg("s021", "c104", 0.5, 4.0, role="cutaway"),
        seg("s022", "c075", 11.569, 18.889, role="a-roll"),
        seg("s023", "c020", 0.0, 3.5, role="cutaway",
            audio_from={"clip": "c075", "in": 18.889, "out": 22.389}),
        seg("s024", "c075", 22.389, 27.6, role="a-roll"),
    )
    tl, _overlaid = overlay_cutaways(tl, project)
    tl, _deduped = dedupe_audio(tl, project)

    assert find_duplicate_audio(tl, project) == []
