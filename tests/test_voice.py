"""Tests for ``ytedit.ai.voice`` — narration pickup cleanup and placement.

The cleanup cases exercise the pure functions directly (plain ``Word`` lists,
no media, no network). The end-to-end tests build a synthetic project the way
``tests/test_cut.py`` does (clip registry, transcripts, analyses, sentence
catalogue), generate a short WAV with ffmpeg and monkeypatch the STT call
(same pattern as ``test_transcribe.py``) plus
:func:`ytedit.cut.probe_voice_duration`, so the whole manifest -> transcribe
-> clean -> render -> place pipeline runs offline.

The clip layout every placement test shares (see :func:`build_project`):

* ``c001`` — 10 s of narration, two sentences.
* ``c010``/``c011`` — 10 s of silent B-roll each.
* ``c012`` — 1 s of B-roll: too short to cover a pickup on its own.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from ytedit import cut as cutlib
from ytedit.ai import voice as V
from ytedit.ai.sentences import build_clip_sentences, flag_instructions, sentences_path
from ytedit.cut import Beat, Cut, Marker, Shot, load_cut, save_cut
from ytedit.project import Project
from ytedit.words import Word

WORDS: dict[str, list[tuple[float, float, str]]] = {
    "c001": [
        (1.0, 1.4, "Cześć"), (1.45, 1.55, "z"), (1.6, 2.2, "Lizbony."),        # c001#1
        (3.0, 3.3, "Jest"), (3.35, 3.9, "pięknie."),                          # c001#2
    ],
}

DURATIONS: dict[str, float] = {"c001": 10.0, "c010": 10.0, "c011": 10.0, "c012": 1.0}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def build_project(tmp_path: Path) -> Project:
    """The synthetic project the placement tests edit the cut of."""
    project = Project.create("t-voice", language="pl", root=tmp_path / "projects")
    catalogue: list[dict[str, Any]] = []
    for order, (clip_id, duration) in enumerate(DURATIONS.items(), 1):
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
        analysis = {"clip": clip_id, "instructions": [], "takes": []}
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
def cut_project(tmp_path: Path) -> Project:
    """A project with clips, transcripts and a sentence catalogue, but no cut."""
    return build_project(tmp_path)


def speech(clip: str, sentences: Sequence[str], **kwargs: Any) -> Beat:
    return Beat(kind="speech", clip=clip, sentences=list(sentences), **kwargs)


def broll(clip: str, in_: float, out: float, **kwargs: Any) -> Beat:
    return Beat(kind="broll", clip=clip, **{"in": in_}, out=out, **kwargs)


def write_cut(project: Project, *beats: Beat, **kwargs: Any) -> Cut:
    """Save a cut of ``beats`` as ``plan/cut.json`` and hand it back."""
    cut = Cut(beats=list(beats), **kwargs)
    save_cut(cut, cutlib.cut_path(project))
    return cut


def sine(path: Path, seconds: float = 4.0, freq: int = 440) -> Path:
    """Write a short mono wav with ffmpeg (content doesn't matter, only length)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={seconds}",
         "-c:a", "pcm_s16le", "-ac", "1", str(path)],
        check=True,
    )
    return path


def transcript_doc(pairs: list[tuple[float, float, str]]) -> dict[str, Any]:
    return {
        "language": "pl",
        "text": " ".join(t for _, _, t in pairs),
        "words": [{"t": t, "s": s, "e": e, "p": 0.0, "speaker": None} for s, e, t in pairs],
        "duration": pairs[-1][1] if pairs else 0.0,
        "engine": "elevenlabs",
    }


def stage_pickup(
    project: Project, monkeypatch: pytest.MonkeyPatch, entry: dict[str, Any],
    wav_seconds: float = 4.0, pickup_seconds: float = 4.0,
) -> None:
    """Write a one-row manifest + its WAV, and stub the STT / probe seams."""
    project.voice_incoming_dir.mkdir(parents=True, exist_ok=True)
    V.manifest_path(project).write_text(
        yaml.safe_dump([entry], allow_unicode=True), encoding="utf-8"
    )
    sine(project.voice_incoming_dir / str(entry["file"]), seconds=wav_seconds)
    monkeypatch.setattr(
        V, "_transcribe_voice_file",
        lambda proj, path: transcript_doc([(0.3, 0.8, "Cześć,"), (1.0, 1.6, "jedziemy.")]),
    )
    monkeypatch.setattr(cutlib, "probe_voice_duration", lambda _p: pickup_seconds)


def kinds(cut: Cut) -> list[str]:
    return [beat.kind for beat in cut.beats]


#: Two attempts at the same line, close enough (Jaccard 4/6 = 0.667 >= 0.6)
#: to count as a retake of each other.
RETAKE_WORDS = [
    Word(0.0, 0.5, "Zaczyna"), Word(0.6, 0.9, "się"), Word(1.0, 1.4, "tutaj"),
    Word(1.5, 1.9, "wielka"), Word(2.0, 2.5, "impreza."),
    Word(3.0, 3.5, "Zaczyna"), Word(3.6, 3.9, "się"), Word(4.0, 4.4, "tutaj"),
    Word(4.5, 4.9, "wielka"), Word(5.0, 5.5, "feta."),
]


# ----------------------------------------------------------------------
# retake removal
# ----------------------------------------------------------------------
def test_retake_keeps_the_later_attempt_by_default() -> None:
    entry = V.ManifestEntry(file="a.wav", request="n001")
    keep, cuts = V._clean_words(RETAKE_WORDS, entry, "pl")
    assert not any(keep[:5])
    assert all(keep[5:])
    assert any(c["reason"] == "retake_of" for c in cuts)


def test_retake_keeps_the_first_attempt_when_requested() -> None:
    entry = V.ManifestEntry(file="a.wav", request="n001", keep_takes="first")
    keep, cuts = V._clean_words(RETAKE_WORDS, entry, "pl")
    assert all(keep[:5])
    assert not any(keep[5:])
    assert any(c["reason"] == "retake_of" for c in cuts)


# ----------------------------------------------------------------------
# instruction sentence dropped
# ----------------------------------------------------------------------
def test_instruction_sentence_is_dropped() -> None:
    words = [
        Word(0.0, 0.4, "To"), Word(0.5, 0.9, "wstaw"), Word(1.0, 1.4, "na"),
        Word(1.5, 1.9, "koniec."),
        Word(2.5, 2.9, "Właściwa"), Word(3.0, 3.6, "narracja."),
    ]
    entry = V.ManifestEntry(file="a.wav", request="n001")
    keep, cuts = V._clean_words(words, entry, "pl")
    assert not any(keep[:4])
    assert all(keep[4:])
    assert cuts[0]["reason"] == "instruction"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("To wstaw na koniec jako podsumowanie.", True),
        ("Użyj tego do intra.", True),
        ("Wytnij to, powtarzam.", True),
        ("Wsiadamy do tramwaju dwadzieścia osiem.", False),
    ],
)
def test_is_instruction_sentence_patterns(text: str, expected: bool) -> None:
    assert V.is_instruction_sentence(text, "pl") is expected


# ----------------------------------------------------------------------
# stutters
# ----------------------------------------------------------------------
def test_immediate_repeated_word_is_merged_to_one() -> None:
    words = [Word(0.0, 0.4, "że,"), Word(0.5, 0.9, "że"), Word(1.0, 1.6, "mówię.")]
    entry = V.ManifestEntry(file="a.wav", request="n001")
    keep, cuts = V._clean_words(words, entry, "pl")
    assert keep == [True, False, True]  # keeps the first occurrence, drops the repeat
    assert cuts[0]["reason"] == "stutter_repeat"


def test_cutoff_word_is_dropped() -> None:
    words = [Word(0.0, 0.3, "je-"), Word(0.4, 0.9, "jedziemy.")]
    entry = V.ManifestEntry(file="a.wav", request="n001")
    keep, cuts = V._clean_words(words, entry, "pl")
    assert keep == [False, True]
    assert cuts[0]["reason"] == "stutter_cutoff"


# ----------------------------------------------------------------------
# pause shortening + speech pads
# ----------------------------------------------------------------------
def test_pause_longer_than_max_is_shortened_not_removed() -> None:
    words = [Word(1.0, 1.4, "Alfa."), Word(3.6, 4.0, "Beta.")]  # 2.2 s gap
    ranges = V._keep_ranges(words, [True, True], max_pause=0.8, pause_keep=0.5,
                             pad_before=0.0, pad_after=0.0)
    assert ranges == [(1.0, 1.9), (3.6, 4.0)]


def test_pause_under_max_is_left_alone() -> None:
    words = [Word(1.0, 1.4, "Alfa."), Word(1.9, 2.3, "Beta.")]  # 0.5 s gap
    ranges = V._keep_ranges(words, [True, True], max_pause=0.8, pause_keep=0.5,
                             pad_before=0.0, pad_after=0.0)
    assert ranges == [(1.0, 2.3)]


def test_speech_pads_applied_to_first_and_last_range_only() -> None:
    words = [Word(1.0, 1.4, "Alfa."), Word(1.9, 2.3, "Beta.")]
    ranges = V._keep_ranges(words, [True, True], max_pause=0.8, pause_keep=0.5,
                             pad_before=0.3, pad_after=0.45)
    assert ranges == [(0.7, 2.75)]


def test_a_dropped_word_hard_cuts_with_no_filler_pause() -> None:
    words = [Word(1.0, 1.4, "Alfa."), Word(2.0, 2.4, "Wytnij."), Word(5.0, 5.4, "Beta.")]
    ranges = V._keep_ranges(words, [True, False, True], max_pause=0.8, pause_keep=0.5,
                             pad_before=0.0, pad_after=0.0)
    assert ranges == [(1.0, 1.4), (5.0, 5.4)]


# ----------------------------------------------------------------------
# resolving a manifest entry to a beat
# ----------------------------------------------------------------------
def test_resolve_beat_explicit_after() -> None:
    cut = Cut(beats=[broll("c010", 0.0, 2.0), broll("c011", 0.0, 2.0)])
    cut.renumber()
    beat, reason = V.resolve_beat(cut, {}, V.ManifestEntry(file="a.wav", after="b002"))
    assert beat is cut.beats[1]
    assert "explicit after b002" in reason


def test_resolve_beat_accepts_a_uid_as_well_as_a_display_id() -> None:
    cut = Cut(beats=[broll("c010", 0.0, 2.0), broll("c011", 0.0, 2.0)])
    cut.renumber()
    entry = V.ManifestEntry(file="a.wav", after=cut.beats[1].uid)
    beat, _reason = V.resolve_beat(cut, {}, entry)
    assert beat is cut.beats[1]


def test_resolve_beat_by_request_via_clip_reference() -> None:
    cut = Cut(beats=[broll("c010", 0.0, 2.0), broll("c011", 0.0, 2.0)])
    cut.renumber()
    edit_plan = {"plan": {"narration_requests": [
        {"id": "n001", "place_after_segment": "after the second B-roll (c011)"}
    ]}}
    beat, reason = V.resolve_beat(cut, edit_plan, V.ManifestEntry(file="a.wav", request="n001"))
    assert beat is cut.beats[1]
    assert "n001" in reason


def test_resolve_beat_by_request_via_marker_label() -> None:
    cut = Cut(beats=[broll("c010", 0.0, 2.0), broll("c011", 0.0, 2.0)])
    cut.renumber()
    cut.markers = [Marker(beat=cut.beats[1].uid, label="hook")]
    edit_plan = {"plan": {"narration_requests": [
        {"id": "n001", "place_after_segment": "right after the hook"}
    ]}}
    beat, _reason = V.resolve_beat(cut, edit_plan, V.ManifestEntry(file="a.wav", request="n001"))
    assert beat is cut.beats[1]


def test_resolve_beat_unresolved_is_reported_not_raised() -> None:
    cut = Cut(beats=[broll("c010", 0.0, 2.0)])
    cut.renumber()
    beat, reason = V.resolve_beat(cut, {}, V.ManifestEntry(file="a.wav", request="n404"))
    assert beat is None
    assert "n404" in reason


# ----------------------------------------------------------------------
# inserting the voice beat and absorbing the picture under it
# ----------------------------------------------------------------------
def test_place_voice_beat_absorbs_the_following_broll(cut_project: Project) -> None:
    cut = Cut(beats=[
        speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0), speech("c001", ["c001#2"]),
    ])
    cut.renumber()
    beat, absorbed = V.place_voice_beat(cut, cut.beats[0], "voice/n001.wav", 4.0)

    assert kinds(cut) == ["speech", "voice", "speech"]        # the B-roll beat is gone
    assert cut.beats[1] is beat
    assert [(s.clip, s.in_, s.out) for s in beat.shots] == [("c010", 0.0, 6.0)]
    assert len(absorbed) == 1 and "c010" in absorbed[0]


def test_place_voice_beat_absorbs_two_broll_beats_for_a_long_pickup(
    cut_project: Project,
) -> None:
    cut = Cut(beats=[
        speech("c001", ["c001#1"]),
        broll("c010", 0.0, 3.0), broll("c011", 0.0, 5.0), broll("c012", 0.0, 1.0),
    ])
    cut.renumber()
    beat, absorbed = V.place_voice_beat(cut, cut.beats[0], "voice/n001.wav", 7.0)

    assert [s.clip for s in beat.shots] == ["c010", "c011"]   # 3 + 5 >= 7, stop there
    assert len(absorbed) == 2
    assert kinds(cut) == ["speech", "voice", "broll"]         # c012 stays a beat of its own


def test_place_voice_beat_with_no_broll_after_it_takes_no_shots(cut_project: Project) -> None:
    cut = Cut(beats=[speech("c001", ["c001#1"]), speech("c001", ["c001#2"])])
    cut.renumber()
    beat, absorbed = V.place_voice_beat(cut, cut.beats[0], "voice/n001.wav", 4.0)

    assert beat.shots == [] and absorbed == []
    assert not beat.keeps_source_audio
    assert kinds(cut) == ["speech", "voice", "speech"]


def test_place_voice_beat_keeps_ambient_picture_ambient(cut_project: Project) -> None:
    cut = Cut(beats=[speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0)])
    cut.renumber()
    beat, _absorbed = V.place_voice_beat(cut, cut.beats[0], "voice/n001.wav", 4.0)
    assert beat.keeps_source_audio       # the B-roll was heard; it still is


def test_place_voice_beat_keeps_muted_picture_muted(cut_project: Project) -> None:
    cut = Cut(beats=[speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0, audio="mute")])
    cut.renumber()
    beat, _absorbed = V.place_voice_beat(cut, cut.beats[0], "voice/n001.wav", 4.0)
    assert not beat.keeps_source_audio


def test_place_voice_beat_is_idempotent_for_the_same_file(cut_project: Project) -> None:
    cut = Cut(beats=[
        speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0), broll("c011", 0.0, 6.0),
    ])
    cut.renumber()
    V.place_voice_beat(cut, cut.beats[0], "voice/n001.wav", 4.0)
    before = kinds(cut)

    beat, absorbed = V.place_voice_beat(cut, cut.beats[0], "voice/n001.wav", 4.0)
    assert kinds(cut) == before                    # no second voice beat
    assert absorbed == []                          # and no more B-roll swallowed
    assert [s.clip for s in beat.shots] == ["c010"]


# ----------------------------------------------------------------------
# end to end: manifest -> transcribe (monkeypatched) -> clean -> render -> place
# ----------------------------------------------------------------------
def test_full_pipeline_places_a_pickup_after_the_named_beat(
    cut_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cut(cut_project, speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0))
    stage_pickup(cut_project, monkeypatch,
                 {"file": "pickup.wav", "after": "b001", "label": "intro"})

    result = V.run_voice(cut_project)
    row = result["rows"][0]
    assert row["status"] == "ok"
    assert row["beat"] == "b002"

    assert (cut_project.voice_dir / "intro.wav").exists()
    assert (cut_project.voice_dir / "intro.json").exists()
    assert result["written"] == "plan/cut.json"
    assert result["backup"]
    # 6 s of B-roll under a 4 s pickup: the resolver trims the last shot and
    # says so, but nothing errors.
    assert not [issue for issue in result["issues"] if issue.startswith("error")]
    assert any("shot_trimmed" in issue for issue in result["issues"])

    cut = load_cut(cutlib.cut_path(cut_project))
    assert kinds(cut) == ["speech", "voice"]
    voice_beat = cut.beats[1]
    assert voice_beat.file == "voice/intro.wav"
    assert [s.clip for s in voice_beat.shots] == ["c010"]

    # the derived timeline was rewritten from the edited cut
    from ytedit.timeline import Timeline

    timeline = Timeline.load(cut_project.timeline_file)
    assert [item.file for item in timeline.tracks.voice] == ["voice/intro.wav"]

    report = (cut_project.voice_incoming_dir / "report.md").read_text(encoding="utf-8")
    assert "intro" in report
    assert "c010" in report


def test_full_pipeline_resolves_a_request_and_absorbs_two_broll_beats(
    cut_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cut(
        cut_project,
        speech("c001", ["c001#1"]), broll("c010", 0.0, 3.0), broll("c011", 0.0, 5.0),
    )
    cut_project.edit_plan_file.write_text(
        json.dumps({"plan": {"narration_requests": [
            {"id": "n001", "place_after_segment": "after the opening line (c001)"}
        ]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    stage_pickup(cut_project, monkeypatch, {"file": "pickup.wav", "request": "n001"},
                 wav_seconds=7.0, pickup_seconds=7.0)

    result = V.run_voice(cut_project)
    assert result["rows"][0]["status"] == "ok"
    assert len(result["rows"][0]["absorbed"]) == 2

    cut = load_cut(cutlib.cut_path(cut_project))
    assert kinds(cut) == ["speech", "voice"]
    assert [s.clip for s in cut.beats[1].shots] == ["c010", "c011"]
    assert cut.beats[1].file == "voice/n001.wav"     # label defaults to the request id


def test_a_pickup_with_no_picture_after_it_is_flagged_in_the_report(
    cut_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cut(cut_project, broll("c010", 0.0, 6.0), speech("c001", ["c001#1"]))
    stage_pickup(cut_project, monkeypatch,
                 {"file": "pickup.wav", "after": "b002", "label": "outro"})

    result = V.run_voice(cut_project)
    row = result["rows"][0]
    assert row["status"] == "voice_picture_short"

    cut = load_cut(cutlib.cut_path(cut_project))
    assert cut.beats[-1].kind == "voice" and cut.beats[-1].shots == []
    assert any("voice_picture_short" in issue for issue in result["issues"])

    report = (cut_project.voice_incoming_dir / "report.md").read_text(encoding="utf-8")
    assert "voice_picture_short" in report
    assert "action needed" in report


def test_an_unresolvable_entry_is_unplaced_and_leaves_the_cut_alone(
    cut_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cut(cut_project, speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0))
    before = cutlib.cut_path(cut_project).read_text(encoding="utf-8")
    stage_pickup(cut_project, monkeypatch,
                 {"file": "pickup.wav", "request": "n404", "label": "orphan"})

    result = V.run_voice(cut_project)
    row = result["rows"][0]
    assert row["status"] == "unplaced"
    assert "n404" in row["placement"]
    assert result["written"] is None
    assert cutlib.cut_path(cut_project).read_text(encoding="utf-8") == before
    assert (cut_project.voice_dir / "orphan.wav").exists()   # the clean-up still ran


def test_rerun_is_idempotent(
    cut_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cut(
        cut_project,
        speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0), broll("c011", 0.0, 6.0),
    )
    stage_pickup(cut_project, monkeypatch,
                 {"file": "pickup.wav", "after": "b001", "label": "intro"})
    calls = {"n": 0}

    def fake_transcribe(proj: Project, path: Path) -> dict[str, Any]:
        calls["n"] += 1
        return transcript_doc([(0.3, 0.8, "Cześć,"), (1.0, 1.6, "jedziemy.")])

    monkeypatch.setattr(V, "_transcribe_voice_file", fake_transcribe)

    V.run_voice(cut_project)
    first = load_cut(cutlib.cut_path(cut_project))
    assert calls["n"] == 1

    result = V.run_voice(cut_project)
    assert result["rows"][0]["status"] == "skipped (cached)"
    assert calls["n"] == 1                                   # no re-transcription
    assert kinds(load_cut(cutlib.cut_path(cut_project))) == kinds(first)

    # --force re-processes the same WAV: still one voice beat, same shots.
    result = V.run_voice(cut_project, force=True)
    assert result["rows"][0]["status"] == "ok"
    again = load_cut(cutlib.cut_path(cut_project))
    assert kinds(again) == kinds(first)
    assert [s.clip for s in again.beats[1].shots] == [
        s.clip for s in first.beats[1].shots
    ]


def test_a_placed_pickup_is_never_dragged_across_the_cut(
    cut_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cut(
        cut_project,
        speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0),
        speech("c001", ["c001#2"]), broll("c011", 0.0, 6.0),
    )
    stage_pickup(cut_project, monkeypatch,
                 {"file": "pickup.wav", "after": "b001", "label": "intro"})
    V.run_voice(cut_project)          # -> [speech, voice, speech, broll]

    V.manifest_path(cut_project).write_text(
        yaml.safe_dump([{"file": "pickup.wav", "after": "b003", "label": "intro"}]),
        encoding="utf-8",
    )
    result = V.run_voice(cut_project, force=True)

    assert "left there" in result["rows"][0]["placement"]
    cut = load_cut(cutlib.cut_path(cut_project))
    assert kinds(cut) == ["speech", "voice", "speech", "broll"]


def test_a_human_edited_cut_is_written_to_a_draft(
    cut_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = Cut(beats=[speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0)])
    cut.meta.edited_by_human = True
    save_cut(cut, cutlib.cut_path(cut_project))
    before = cutlib.cut_path(cut_project).read_text(encoding="utf-8")
    stage_pickup(cut_project, monkeypatch,
                 {"file": "pickup.wav", "after": "b001", "label": "intro"})

    result = V.run_voice(cut_project)
    assert result["edited_by_human"] is True
    assert result["written"] == "plan/cut.draft.json"
    assert (cut_project.plan_dir / "cut.draft.json").exists()
    assert cutlib.cut_path(cut_project).read_text(encoding="utf-8") == before

    draft = load_cut(cut_project.plan_dir / "cut.draft.json")
    assert kinds(draft) == ["speech", "voice"]


# ----------------------------------------------------------------------
# manifest drafting and the missing-cut guard
# ----------------------------------------------------------------------
def test_missing_manifest_writes_a_draft_listing_the_beats(cut_project: Project) -> None:
    cut = write_cut(
        cut_project, speech("c001", ["c001#1"]), broll("c010", 0.0, 6.0),
        Beat(kind="voice", file="voice/old.wav", shots=[Shot(clip="c011", out=2.0)]),
    )
    cut_project.voice_incoming_dir.mkdir(parents=True, exist_ok=True)
    sine(cut_project.voice_incoming_dir / "pickup.wav", seconds=1.0)

    with pytest.raises(V.VoiceError) as excinfo:
        V.load_manifest(cut_project, cut)
    assert "offset" in str(excinfo.value)          # the format change is spelled out

    text = V.manifest_path(cut_project).read_text(encoding="utf-8")
    assert "# Beats of plan/cut.json:" in text
    assert "b001  speech c001" in text
    assert '"Cześć z Lizbony."' in text
    assert "b002  broll  c010   0.00-6.00s" in text
    assert "voice/old.wav" in text
    rows = yaml.safe_load(text)
    assert rows == [{"file": "pickup.wav", "request": None, "after": None}]


def test_missing_cut_raises(cut_project: Project) -> None:
    cut_project.voice_incoming_dir.mkdir(parents=True, exist_ok=True)
    V.manifest_path(cut_project).write_text("[]\n", encoding="utf-8")
    with pytest.raises(V.VoiceError) as excinfo:
        V.run_voice(cut_project)
    assert "ytedit plan" in str(excinfo.value)
