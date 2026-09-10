"""Tests for ``ytedit.cut`` — the script-first edit model and its resolver.

Everything here is offline: a synthetic project (clip registry, transcripts,
analyses and a hand-built sentence catalogue), no media and no API calls. The
one file that must exist on disk is an empty narration WAV, whose length is
supplied by monkeypatching :func:`ytedit.cut.probe_voice_duration`.

The clip layout every test shares (see :func:`build_project`):

* ``c001`` — 30 s of narration, four sentences, the last one an instruction.
* ``c002`` — a sentence followed by an instruction range that starts *before*
  the next word, so it clamps the audio window on its own.
* ``c003`` — B-roll with one spoken word inside it.
* ``c010``/``c011`` — silent cutaway footage, 10 s each.
* ``c012`` — 1 s of cutaway footage: too short to cover a voice pickup.
* ``c004``/``c005`` — two clips whose speech nearly touches their edges.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from ytedit.ai.sentences import build_clip_sentences, flag_instructions, sentences_path
from ytedit.words import Word
from ytedit.cut import (
    Beat,
    Caption,
    Chapter,
    Cut,
    CutError,
    Marker,
    MusicCue,
    Shot,
    cut_path,
    ensure_resolved,
    load_cut,
    resolve,
    save_cut,
    validate,
)
from ytedit.project import Project

WORDS: dict[str, list[tuple[float, float, str]]] = {
    "c001": [
        (1.0, 1.4, "Cześć"), (1.45, 1.55, "z"), (1.6, 2.2, "Lisboa."),        # c001#1
        (3.0, 3.3, "Jest"), (3.35, 3.9, "pięknie."),                          # c001#2
        (4.1, 4.5, "Idziemy"), (4.55, 5.0, "dalej."),                         # c001#3
        (6.0, 6.4, "Wytnij"), (6.45, 6.9, "to."),                             # c001#4 (instruction)
    ],
    "c002": [
        (1.0, 1.6, "Dobrze."),                                                # c002#1
        (3.0, 3.4, "Wytnij"), (3.45, 3.8, "to."),                             # c002#2 (instruction)
    ],
    "c003": [(2.0, 2.5, "Halo.")],
    "c004": [(1.6, 1.95, "Koniec.")],
    "c005": [(0.05, 0.6, "Start.")],
}

INSTRUCTIONS: dict[str, list[tuple[float, float]]] = {
    "c001": [(5.95, 6.95)],
    # Starts inside c002#1's tail pad, well before the next word: the only
    # thing that can pull the audio window back off 2.05 s.
    "c002": [(2.0, 4.0)],
}

DURATIONS: dict[str, float] = {
    "c001": 30.0, "c002": 30.0, "c003": 20.0,
    "c004": 2.0, "c005": 5.0,
    "c010": 10.0, "c011": 10.0, "c012": 1.0,
}


# ----------------------------------------------------------------------
# the synthetic project
# ----------------------------------------------------------------------
def build_project(tmp_path: Path) -> Project:
    """Create the project every test in this module resolves against."""
    project = Project.create("t-cut", language="pl", root=tmp_path / "projects")
    order = 0
    catalogue: list[dict[str, Any]] = []
    for clip_id, duration in DURATIONS.items():
        order += 1
        project.add_clip({
            "id": clip_id, "order": order, "duration": duration,
            "width": 1920, "height": 1080, "orientation": "horizontal", "has_audio": True,
        })
        words = WORDS.get(clip_id, [])
        project.transcript_path(clip_id).write_text(
            json.dumps({
                "clip": clip_id, "language": "pl",
                "words": [{"t": t, "s": s, "e": e} for s, e, t in words],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        analysis = {
            "clip": clip_id,
            "instructions": [
                {"s": s, "e": e, "text": "wytnij to", "action": "discard"}
                for s, e in INSTRUCTIONS.get(clip_id, [])
            ],
            "takes": [],
        }
        project.analysis_path(clip_id).write_text(
            json.dumps(analysis, ensure_ascii=False), encoding="utf-8"
        )
        sentences = build_clip_sentences(clip_id, [Word(s, e, t) for s, e, t in words], "pl")
        flag_instructions(sentences, analysis)
        catalogue.append({"id": clip_id, "sentences": sentences})

    sentences_path(project).write_text(
        json.dumps({
            "project": project.slug, "language": "pl", "clips": catalogue,
            "clips_count": len(catalogue),
            "sentences_count": sum(len(c["sentences"]) for c in catalogue),
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return project


@pytest.fixture()
def project(tmp_path: Path) -> Project:
    return build_project(tmp_path)


def speech(clip: str, sentences: Sequence[str], **kwargs: Any) -> Beat:
    """A speech beat on ``clip``."""
    return Beat(kind="speech", clip=clip, sentences=list(sentences), **kwargs)


def broll(clip: str, in_: float, out: float, **kwargs: Any) -> Beat:
    """A B-roll beat."""
    return Beat(kind="broll", clip=clip, **{"in": in_}, out=out, **kwargs)


def cut_of(*beats: Beat, **kwargs: Any) -> Cut:
    """A cut of ``beats`` with display ids already assigned."""
    cut = Cut(beats=list(beats), **kwargs)
    cut.renumber()
    return cut


def codes(issues: Sequence[Any], severity: str | None = None) -> list[str]:
    """The issue codes, optionally filtered by severity."""
    return [i.code for i in issues if severity is None or i.severity == severity]


def audio_spans(timeline: Any) -> list[tuple[str, float, float]]:
    """``(clip, in, out)`` of every segment's audio, in order."""
    return [seg.audio_source for seg in timeline.tracks.video]


def assert_contiguous(timeline: Any, beat_uid: str) -> None:
    """Every segment of one beat must abut the previous one to the millisecond."""
    spans = [
        seg.audio_source for seg in timeline.tracks.video if seg.beat == beat_uid
    ]
    assert spans, "the beat produced no segments"
    for (clip_a, _in_a, out_a), (clip_b, in_b, _out_b) in zip(spans, spans[1:]):
        assert clip_a == clip_b, (clip_a, clip_b)
        assert abs(out_a - in_b) < 1e-3, (out_a, in_b)


# ----------------------------------------------------------------------
# speech: sentences -> seconds
# ----------------------------------------------------------------------
def test_a_sentence_becomes_one_segment_with_the_full_pads(project: Project) -> None:
    beat = speech("c001", ["c001#1"], role="a-roll")
    timeline = resolve(project, cut_of(beat))

    assert timeline.version == 2
    assert timeline.meta.source.startswith("cut.json@")
    assert len(timeline.tracks.video) == 1
    seg = timeline.tracks.video[0]
    assert (seg.clip, seg.in_, seg.out) == ("c001", 0.7, 2.65)   # 1.0-0.30, 2.2+0.45
    assert seg.audio_window is None                              # nothing in the way
    assert seg.beat == beat.uid
    assert seg.role == "a-roll"
    assert seg.id == "s001"
    assert not timeline.validate(project, skip_music=True, skip_voice=True)


def test_a_close_next_word_windows_the_audio_and_extends_the_picture(
    project: Project,
) -> None:
    """c001#2 ends 3.9 s, the next word starts 4.1 s: 0.45 s of pad does not fit."""
    timeline = resolve(project, cut_of(speech("c001", ["c001#2"])))

    seg = timeline.tracks.video[0]
    assert (seg.in_, seg.out) == (2.7, 4.35)                     # the picture gets the pad
    assert seg.audio_window is not None
    assert (seg.audio_window.in_, seg.audio_window.out) == (2.7, 4.05)   # 4.1 - 0.05 guard
    assert not timeline.validate(project, skip_music=True, skip_voice=True)


def test_a_close_previous_word_windows_the_head_of_the_audio(project: Project) -> None:
    """c001#3 starts 4.1 s, the previous word ends 3.9 s."""
    timeline = resolve(project, cut_of(speech("c001", ["c001#3"])))

    seg = timeline.tracks.video[0]
    assert (seg.in_, seg.out) == (3.8, 5.45)
    assert (seg.audio_window.in_, seg.audio_window.out) == (3.95, 5.45)


def test_an_instruction_range_clamps_the_audio_window(project: Project) -> None:
    """c002's instruction starts at 2.0 s — inside c002#1's 0.45 s tail pad."""
    timeline = resolve(project, cut_of(speech("c002", ["c002#1"])))

    seg = timeline.tracks.video[0]
    assert (seg.in_, seg.out) == (0.7, 2.05)
    assert (seg.audio_window.in_, seg.audio_window.out) == (0.7, 2.0)


def test_a_words_range_is_the_escape_hatch_into_a_sentence(project: Project) -> None:
    beat = Beat(kind="speech", clip="c001", words=(0, 2))
    timeline = resolve(project, cut_of(beat))

    seg = timeline.tracks.video[0]
    assert (seg.in_, seg.out) == (0.7, 2.65)


def test_a_words_range_inside_an_instruction_is_an_error(project: Project) -> None:
    issues = validate(project, cut_of(Beat(kind="speech", clip="c001", words=(7, 8))))
    assert "words_in_instruction" in codes(issues, "error")


def test_two_beats_may_not_claim_the_same_words(project: Project) -> None:
    cut = cut_of(
        Beat(kind="speech", clip="c001", words=(0, 2)),
        Beat(kind="speech", clip="c001", words=(2, 4)),
    )
    assert "words_reused" in codes(validate(project, cut), "error")


# ----------------------------------------------------------------------
# speech: the error rules
# ----------------------------------------------------------------------
def test_non_contiguous_sentences_are_an_error(project: Project) -> None:
    issues = validate(project, cut_of(speech("c001", ["c001#1", "c001#3"])))
    assert "sentences_not_contiguous" in codes(issues, "error")


def test_a_sentence_used_twice_is_an_error(project: Project) -> None:
    cut = cut_of(speech("c001", ["c001#1"]), speech("c001", ["c001#1"]))
    assert "sentence_reused" in codes(validate(project, cut), "error")


def test_an_instruction_sentence_is_an_error(project: Project) -> None:
    issues = validate(project, cut_of(speech("c001", ["c001#4"])))
    assert "instruction_sentence" in codes(issues, "error")


def test_an_unknown_sentence_is_an_error(project: Project) -> None:
    issues = validate(project, cut_of(speech("c001", ["c001#99"])))
    assert "unknown_sentence" in codes(issues, "error")


def test_an_unknown_clip_is_an_error(project: Project) -> None:
    issues = validate(project, cut_of(speech("c999", ["c001#1"])))
    assert "unknown_clip" in codes(issues, "error")


def test_resolve_refuses_a_cut_with_errors(project: Project) -> None:
    with pytest.raises(CutError) as excinfo:
        resolve(project, cut_of(speech("c001", ["c001#4"])))
    assert "instruction_sentence" in str(excinfo.value)


# ----------------------------------------------------------------------
# shots
# ----------------------------------------------------------------------
def whole_take(**kwargs: Any) -> Beat:
    """c001#1..#3 — 1.0 s to 5.0 s of speech, padded to 0.70-5.45 s."""
    return speech("c001", ["c001#1", "c001#2", "c001#3"], **kwargs)


def test_a_shot_after_a_sentence_splits_the_beat_in_three(project: Project) -> None:
    beat = whole_take(shots=[Shot(clip="c010", **{"in": 0.0}, out=1.0, after="c001#1")])
    timeline = resolve(project, cut_of(beat))

    video = timeline.tracks.video
    assert [seg.clip for seg in video] == ["c001", "c010", "c001"]
    assert audio_spans(timeline) == [
        ("c001", 0.7, 2.2),     # up to the end of the sentence, no air: speech continues
        ("c001", 2.2, 3.2),     # the cutaway borrows the narration underneath
        ("c001", 3.2, 5.45),
    ]
    assert (video[1].in_, video[1].out) == (0.0, 1.0)
    assert video[1].audio_from is not None and video[1].audio_from.clip == "c001"
    assert not video[1].mute_source
    assert all(seg.beat == beat.uid for seg in video)
    assert_contiguous(timeline, beat.uid)


def test_two_shots_after_the_same_sentence_chain_in_list_order(project: Project) -> None:
    beat = whole_take(shots=[
        Shot(clip="c010", **{"in": 0.0}, out=1.0, after="c001#1"),
        Shot(clip="c011", **{"in": 2.0}, out=3.0, after="c001#1"),
    ])
    timeline = resolve(project, cut_of(beat))

    assert [seg.clip for seg in timeline.tracks.video] == ["c001", "c010", "c011", "c001"]
    assert audio_spans(timeline) == [
        ("c001", 0.7, 2.2), ("c001", 2.2, 3.2), ("c001", 3.2, 4.2), ("c001", 4.2, 5.45),
    ]
    assert_contiguous(timeline, beat.uid)


def test_a_shot_is_trimmed_at_the_end_of_the_beat(project: Project) -> None:
    beat = whole_take(shots=[Shot(clip="c010", **{"in": 0.0}, out=3.0, after="c001#3")])
    cut = cut_of(beat)
    assert "shot_trimmed" in codes(validate(project, cut), "warning")

    timeline = resolve(project, cut)
    assert audio_spans(timeline) == [("c001", 0.7, 5.0), ("c001", 5.0, 5.45)]
    assert timeline.tracks.video[1].out == 0.45      # 3.0 s asked for, 0.45 s left


def test_the_min_shot_rule_extends_the_last_shot_over_a_stub(project: Project) -> None:
    """0.55 s of own picture would be left after the shot — under min_shot_seconds."""
    beat = whole_take(shots=[Shot(clip="c010", **{"in": 0.0}, out=1.0, after="c001#2")])
    timeline = resolve(project, cut_of(beat))

    assert [seg.clip for seg in timeline.tracks.video] == ["c001", "c010"]
    assert timeline.tracks.video[1].out == 1.55      # 1.0 + the 0.55 s stub
    assert audio_spans(timeline) == [("c001", 0.7, 3.9), ("c001", 3.9, 5.45)]
    assert "short_piece" not in codes(validate(project, cut_of(beat)))


def test_a_stub_the_shot_clip_cannot_cover_stays_and_warns(project: Project) -> None:
    beat = whole_take(shots=[Shot(clip="c012", **{"in": 0.0}, out=1.0, after="c001#2")])
    cut = cut_of(beat)
    assert "short_piece" in codes(validate(project, cut), "warning")
    timeline = resolve(project, cut)
    assert [seg.clip for seg in timeline.tracks.video] == ["c001", "c012", "c001"]


def test_a_shot_after_a_sentence_of_another_beat_is_an_error(project: Project) -> None:
    beat = speech("c001", ["c001#1"], shots=[
        Shot(clip="c010", **{"in": 0.0}, out=1.0, after="c001#3"),
    ])
    assert "unknown_shot_after" in codes(validate(project, cut_of(beat)), "error")


def test_off_camera_shots_cover_the_whole_beat(project: Project) -> None:
    beat = whole_take(on_camera=False, shots=[
        Shot(clip="c010", **{"in": 0.0}, out=4.75),
    ])
    cut = cut_of(beat)
    assert not codes(validate(project, cut))

    timeline = resolve(project, cut)
    assert [seg.clip for seg in timeline.tracks.video] == ["c010"]
    assert audio_spans(timeline) == [("c001", 0.7, 5.45)]


def test_an_off_camera_beat_a_shot_does_not_cover_warns(project: Project) -> None:
    beat = whole_take(on_camera=False, shots=[Shot(clip="c010", **{"in": 0.0}, out=2.0)])
    cut = cut_of(beat)
    assert "narrator_visible" in codes(validate(project, cut), "warning")

    timeline = resolve(project, cut)
    assert [seg.clip for seg in timeline.tracks.video] == ["c010", "c001"]
    assert audio_spans(timeline) == [("c001", 0.7, 2.7), ("c001", 2.7, 5.45)]
    assert_contiguous(timeline, beat.uid)


# ----------------------------------------------------------------------
# broll
# ----------------------------------------------------------------------
def test_broll_keeps_its_ambient_sound(project: Project) -> None:
    timeline = resolve(project, cut_of(broll("c003", 5.0, 8.0, role="b-roll")))
    seg = timeline.tracks.video[0]
    assert (seg.clip, seg.in_, seg.out) == ("c003", 5.0, 8.0)
    assert not seg.mute_source
    assert seg.audio_window is None


def test_muted_broll_is_silent(project: Project) -> None:
    timeline = resolve(project, cut_of(broll("c003", 5.0, 8.0, audio="mute")))
    assert timeline.tracks.video[0].mute_source


def test_ambient_broll_over_speech_warns(project: Project) -> None:
    issues = validate(project, cut_of(broll("c003", 1.0, 4.0)))
    assert "speech_in_broll" in codes(issues, "warning")
    assert not codes(validate(project, cut_of(broll("c003", 1.0, 4.0, audio="mute"))))


def test_a_broll_range_past_the_end_of_the_clip_is_an_error(project: Project) -> None:
    assert "bad_range" in codes(validate(project, cut_of(broll("c003", 5.0, 99.0))), "error")


# ----------------------------------------------------------------------
# voice pickups
# ----------------------------------------------------------------------
@pytest.fixture()
def pickup(project: Project, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A 4.0 s narration pickup on disk (its length comes from the patch)."""
    project.voice_dir.mkdir(parents=True, exist_ok=True)
    path = project.voice_dir / "n001.wav"
    path.write_bytes(b"RIFF")
    monkeypatch.setattr("ytedit.cut.probe_voice_duration", lambda _p: 4.0)
    return path


def test_a_voice_beat_places_its_pickup_and_mutes_the_picture(
    project: Project, pickup: Path
) -> None:
    beat = Beat(kind="voice", file="voice/n001.wav", gain_db=-1.0, shots=[
        Shot(clip="c010", **{"in": 0.0}, out=2.0),
        Shot(clip="c011", **{"in": 0.0}, out=1.5),
    ])
    timeline = resolve(project, cut_of(broll("c003", 0.0, 3.0), beat))

    video = [seg for seg in timeline.tracks.video if seg.beat == beat.uid]
    assert [(seg.clip, seg.in_, seg.out) for seg in video] == [
        ("c010", 0.0, 2.0), ("c011", 0.0, 2.0),     # the last shot grew to the WAV end
    ]
    assert all(seg.mute_source for seg in video)
    assert all(seg.audio_from is None for seg in video)

    item = timeline.tracks.voice[0]
    start = timeline.segment_positions()[1].start
    assert item.file == "voice/n001.wav"
    assert item.at == pytest.approx(start)
    assert item.end == pytest.approx(start + 4.0)
    assert item.gain_db == -1.0


def test_a_voice_beat_without_enough_picture_is_an_error(
    project: Project, pickup: Path
) -> None:
    beat = Beat(kind="voice", file="voice/n001.wav", shots=[
        Shot(clip="c012", **{"in": 0.0}, out=1.0),   # a 1 s clip, nothing left to stretch
    ])
    assert "voice_picture_short" in codes(validate(project, cut_of(beat)), "error")


def test_a_voice_beat_with_no_shots_at_all_is_an_error(
    project: Project, pickup: Path
) -> None:
    beat = Beat(kind="voice", file="voice/n001.wav")
    assert "voice_picture_short" in codes(validate(project, cut_of(beat)), "error")


def test_a_missing_voice_file_is_an_error(project: Project) -> None:
    beat = Beat(kind="voice", file="voice/nope.wav", shots=[
        Shot(clip="c010", **{"in": 0.0}, out=2.0),
    ])
    assert "voice_file_missing" in codes(validate(project, cut_of(beat)), "error")


# ----------------------------------------------------------------------
# absolute tracks
# ----------------------------------------------------------------------
def test_music_captions_chapters_and_markers_land_on_their_beats(
    project: Project,
) -> None:
    first = broll("c003", 5.0, 8.0, role="cold-open")
    second = speech("c001", ["c001#1"])
    cut = cut_of(first, second)
    cut.music = [MusicCue(id="m001", file="music/bed.mp3", **{"from": "b001"}, to="b002")]
    cut.captions = [Caption(id="t001", beat="b002", offset=0.3, duration=3.0, text="LISBOA")]
    cut.chapters = [Chapter(beat="b001", title="Portugal")]
    cut.markers = [Marker(beat="b002", label="hook")]

    timeline = resolve(project, cut)
    positions = timeline.segment_positions()
    beat_two_start = positions[1].start

    assert timeline.tracks.music[0].at == pytest.approx(positions[0].start)
    assert timeline.tracks.music[0].end == pytest.approx(positions[-1].end)
    assert timeline.tracks.captions[0].at == pytest.approx(beat_two_start + 0.3)
    # 3.0 s of card would outlast the 1.95 s beat: clamped to its end + 0.5 s
    assert timeline.tracks.captions[0].end == pytest.approx(positions[1].end + 0.5)
    assert timeline.chapters[0].at == pytest.approx(positions[0].start)
    assert timeline.markers[0].at == pytest.approx(beat_two_start)


def test_a_caption_end_is_clamped_to_its_beat(project: Project) -> None:
    cut = cut_of(broll("c003", 5.0, 8.0))
    cut.captions = [Caption(id="t001", beat="b001", offset=0.0, duration=30.0, text="X")]
    timeline = resolve(project, cut)
    assert timeline.tracks.captions[0].end == pytest.approx(
        timeline.segment_positions()[0].end + 0.5
    )


def test_a_caption_on_a_beat_that_is_gone_is_dropped_with_a_warning(
    project: Project,
) -> None:
    cut = cut_of(broll("c003", 5.0, 8.0))
    cut.captions = [Caption(id="t001", beat="deadbeef", text="X")]
    cut.markers = [Marker(beat="deadbeef", label="hook")]
    issues = validate(project, cut)
    assert "caption_dropped" in codes(issues, "warning")
    assert "marker_dropped" in codes(issues, "warning")
    assert not resolve(project, cut).tracks.captions


def test_a_music_cue_that_lost_one_end_snaps_to_a_surviving_beat(
    project: Project,
) -> None:
    cut = cut_of(broll("c003", 5.0, 8.0), speech("c001", ["c001#1"]))
    cut.music = [MusicCue(id="m001", file="music/bed.mp3", **{"from": "gone"}, to="b002")]
    issues = validate(project, cut)
    assert "music_beat_snapped" in codes(issues, "warning")
    timeline = resolve(project, cut)
    assert timeline.tracks.music[0].at == pytest.approx(0.0)


def test_music_cues_meeting_inside_one_beat_never_overlap(project: Project) -> None:
    # Inclusive ``to`` ranges make two cues share the beat where the bed
    # changes; the later cue's start wins and the earlier one ends there.
    cut = cut_of(broll("c003", 5.0, 8.0), speech("c001", ["c001#1"]), broll("c003", 0.0, 2.0))
    cut.music = [
        MusicCue(id="m001", file="music/a.mp3", **{"from": "b001"}, to="b002"),
        MusicCue(id="m002", file="music/b.mp3", **{"from": "b002"}, to="b003"),
    ]
    timeline = resolve(project, cut)
    positions = timeline.segment_positions()
    first, second = timeline.tracks.music
    assert first.end == pytest.approx(positions[1].start)
    assert second.at == pytest.approx(positions[1].start)
    assert not [m for m in timeline.validate(project) if "overlap" in m]


def test_a_music_cue_that_lost_both_ends_is_an_error(project: Project) -> None:
    cut = cut_of(broll("c003", 5.0, 8.0))
    cut.music = [MusicCue(id="m001", file="music/bed.mp3", **{"from": "gone"}, to="also-gone")]
    assert "music_beat_missing" in codes(validate(project, cut), "error")


# ----------------------------------------------------------------------
# air
# ----------------------------------------------------------------------
def test_two_speech_beats_that_meet_without_air_warn(project: Project) -> None:
    """c004 ends 0.05 s after its last word; c005 starts 0.05 s before its first."""
    cut = cut_of(speech("c004", ["c004#1"]), speech("c005", ["c005#1"]))
    issues = validate(project, cut)
    assert "air_short" in codes(issues, "warning")


def test_two_speech_beats_with_room_do_not_warn(project: Project) -> None:
    cut = cut_of(speech("c001", ["c001#1"]), speech("c002", ["c002#1"]))
    assert "air_short" not in codes(validate(project, cut))


# ----------------------------------------------------------------------
# io
# ----------------------------------------------------------------------
def test_save_and_load_round_trip_renumbers_and_keeps_uids(
    project: Project, tmp_path: Path
) -> None:
    beat_a = broll("c003", 5.0, 8.0)
    beat_b = speech("c001", ["c001#1"])
    cut = Cut(beats=[beat_a, beat_b])
    cut.captions = [Caption(id="t001", beat=beat_b.uid, text="LISBOA")]
    path = tmp_path / "cut.json"

    save_cut(cut, path)
    assert [b.id for b in cut.beats] == ["b001", "b002"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["beats"][0]["id"] == "b001"
    assert "words" not in payload["beats"][0]        # exclude_none
    assert path.read_text(encoding="utf-8").splitlines()[1].startswith("  ")

    again = load_cut(path)
    assert [b.uid for b in again.beats] == [beat_a.uid, beat_b.uid]
    assert again.captions[0].beat == beat_b.uid
    assert save_cut(again, path).read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_a_reference_written_as_a_display_id_resolves_to_the_uid(
    project: Project, tmp_path: Path
) -> None:
    beat = speech("c001", ["c001#1"])
    cut = cut_of(broll("c003", 5.0, 8.0), beat)
    cut.captions = [Caption(id="t001", beat="b002", text="LISBOA")]
    path = tmp_path / "cut.json"
    path.write_text(
        json.dumps(cut.model_dump(by_alias=True, mode="json", exclude_none=True)),
        encoding="utf-8",
    )
    assert load_cut(path).captions[0].beat == beat.uid


def test_ensure_resolved_writes_only_when_the_cut_is_newer(project: Project) -> None:
    assert not cut_path(project).exists()
    assert ensure_resolved(project) == project.timeline_file
    assert not project.timeline_file.exists()       # a v1 project is left alone

    save_cut(cut_of(broll("c003", 5.0, 8.0)), cut_path(project))
    ensure_resolved(project)
    assert project.timeline_file.exists()
    stamp = project.timeline_file.stat().st_mtime_ns

    ensure_resolved(project)
    assert project.timeline_file.stat().st_mtime_ns == stamp   # still fresh: no rewrite

    os_touch_newer(cut_path(project), project.timeline_file)
    ensure_resolved(project)
    assert project.timeline_file.stat().st_mtime_ns != stamp


def os_touch_newer(path: Path, reference: Path) -> None:
    """Make ``path`` unambiguously newer than ``reference``."""
    import os

    stamp = reference.stat().st_mtime_ns + 2_000_000_000
    os.utime(path, ns=(stamp, stamp))


def test_ensure_resolved_raises_on_a_broken_cut(project: Project) -> None:
    save_cut(cut_of(speech("c001", ["c001#4"])), cut_path(project))
    with pytest.raises(CutError):
        ensure_resolved(project)


# ----------------------------------------------------------------------
# determinism
# ----------------------------------------------------------------------
def test_resolving_twice_produces_an_identical_timeline(project: Project) -> None:
    cut = cut_of(
        broll("c003", 5.0, 8.0),
        whole_take(shots=[Shot(clip="c010", **{"in": 0.0}, out=1.0, after="c001#1")]),
    )
    first = resolve(project, cut).to_dict()
    second = resolve(project, cut).to_dict()
    first["meta"]["source"] = second["meta"]["source"] = ""
    assert first == second
