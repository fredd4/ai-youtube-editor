"""Tests for ``ytedit.ai.voice`` — narration pickup cleanup and placement.

Most cases exercise the pure functions directly (plain ``Word`` lists, no
media, no network). The end-to-end tests generate a short synthetic WAV with
ffmpeg and monkeypatch the STT call (same pattern as ``test_transcribe.py``)
to exercise the full manifest -> transcribe -> clean -> render -> place
pipeline offline.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from ytedit.ai import voice as V
from ytedit.ai.tidy import Word
from ytedit.project import Project
from ytedit.timeline import Timeline, VideoSegment, VoiceAnchor, VoiceItem, new_timeline


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
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


def seg(seg_id: str, clip: str, start: float, end: float, **extra: Any) -> VideoSegment:
    return VideoSegment(id=seg_id, clip=clip, **{"in": start}, out=end, **extra)


def add_clip(
    project: Project,
    clip_id: str,
    duration: float,
    words: list[tuple[float, float, str]] = (),
    order: int = 1,
) -> None:
    project.add_clip(
        {"id": clip_id, "order": order, "duration": duration,
         "width": 1920, "height": 1080, "orientation": "horizontal", "has_audio": True}
    )
    if words:
        project.transcript_path(clip_id).write_text(
            json.dumps({"clip": clip_id, "language": "pl",
                        "words": [{"t": t, "s": s, "e": e} for s, e, t in words]}),
            encoding="utf-8",
        )


def transcript_doc(pairs: list[tuple[float, float, str]]) -> dict[str, Any]:
    return {
        "language": "pl",
        "text": " ".join(t for _, _, t in pairs),
        "words": [{"t": t, "s": s, "e": e, "p": 0.0, "speaker": None} for s, e, t in pairs],
        "duration": pairs[-1][1] if pairs else 0.0,
        "engine": "elevenlabs",
    }


#: Two attempts at the same line, close enough (Jaccard 4/6 = 0.667 >= 0.6)
#: to count as a retake of each other.
RETAKE_WORDS = [
    Word(0.0, 0.5, "Zaczyna"), Word(0.6, 0.9, "się"), Word(1.0, 1.4, "tutaj"),
    Word(1.5, 1.9, "wielka"), Word(2.0, 2.5, "street party."),
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
# placement
# ----------------------------------------------------------------------
def test_resolve_anchor_explicit() -> None:
    tl = new_timeline()
    tl.tracks.video = [seg("s001", "c001", 0.0, 5.0)]
    entry = V.ManifestEntry(file="a.wav", anchor="s001", offset=1.5)
    anchor, reason = V.resolve_anchor(tl, {}, entry)
    assert anchor == VoiceAnchor(segment="s001", offset=1.5)
    assert "explicit anchor" in reason


def test_resolve_anchor_by_request_via_clip_reference() -> None:
    tl = new_timeline()
    tl.tracks.video = [
        seg("s001", "c001", 0.0, 5.0),
        seg("s002", "c002", 0.0, 5.0),
        seg("s003", "c003", 0.0, 5.0),
    ]
    edit_plan = {
        "plan": {"narration_requests": [
            {"id": "n001", "place_after_segment": "cold-open segment 1 (c001)"}
        ]}
    }
    entry = V.ManifestEntry(file="a.wav", request="n001")
    anchor, reason = V.resolve_anchor(tl, edit_plan, entry)
    assert anchor == VoiceAnchor(segment="s002", offset=0.0)
    assert "n001" in reason


def test_resolve_anchor_unresolved_is_reported_not_raised() -> None:
    tl = new_timeline()
    tl.tracks.video = [seg("s001", "c001", 0.0, 5.0)]
    entry = V.ManifestEntry(file="a.wav", request="n404")
    anchor, reason = V.resolve_anchor(tl, {}, entry)
    assert anchor is None
    assert "n404" in reason


# ----------------------------------------------------------------------
# overlap: grow the muted picture, then pull from broll_pool
# ----------------------------------------------------------------------
def test_overlong_pickup_grows_picture_then_uses_broll_pool(project: Project) -> None:
    add_clip(project, "c001", 3.0, order=1)   # muted picture, only 1s of room to grow
    add_clip(project, "c002", 5.0, words=[(0.0, 1.0, "Mówię.")], order=2)  # blocks further growth
    add_clip(project, "c003", 10.0, order=3)  # broll_pool source

    tl = new_timeline()
    tl.tracks.video = [
        seg("s001", "c001", 0.0, 2.0, mute_source=True),
        seg("s002", "c002", 0.0, 3.0),
    ]
    item = VoiceItem(id="v001", file="voice/x.wav", at=0.0, end=6.0,
                      anchor=VoiceAnchor(segment="s001", offset=0.0))
    tl.tracks.voice = [item]

    ok, changes, record = V._cover_voice_item(
        project, tl, project.settings, item, [("c003", 0.0, 5.0)]
    )

    assert ok, changes
    assert tl.tracks.video[0].out == pytest.approx(3.0)  # grown to clip c001's own duration
    assert len(tl.tracks.video) == 3
    inserted = tl.tracks.video[1]
    assert inserted.clip == "c003"
    assert inserted.mute_source is True
    assert inserted.out - inserted.in_ == pytest.approx(3.0)
    assert record["extended_segment"] == "s001"
    assert record["extended_by"] == pytest.approx(1.0)
    assert record["inserted_segments"] == [inserted.id]
    # the original s002 shifted right by the amount inserted+grown ahead of it
    positions = {p.segment.id: p.start for p in tl.segment_positions()}
    assert positions["s002"] == pytest.approx(6.0)
    # length preserved
    assert item.end - item.at == pytest.approx(6.0)


def test_overlong_pickup_reports_when_broll_pool_runs_out(project: Project) -> None:
    add_clip(project, "c001", 2.0, order=1)  # no room to grow at all
    add_clip(project, "c002", 5.0, words=[(0.0, 1.0, "Mówię.")], order=2)

    tl = new_timeline()
    tl.tracks.video = [
        seg("s001", "c001", 0.0, 2.0, mute_source=True),
        seg("s002", "c002", 0.0, 3.0),
    ]
    item = VoiceItem(id="v001", file="voice/x.wav", at=0.0, end=6.0,
                      anchor=VoiceAnchor(segment="s001", offset=0.0))
    tl.tracks.voice = [item]

    ok, changes, record = V._cover_voice_item(project, tl, project.settings, item, [])
    assert not ok
    assert any("overlap unresolved" in c for c in changes)


# ----------------------------------------------------------------------
# end to end: manifest -> transcribe (monkeypatched) -> clean -> render -> place
# ----------------------------------------------------------------------
def _write_minimal_timeline(project: Project, duration: float = 8.0) -> None:
    add_clip(project, "c001", duration, order=1)
    tl = new_timeline()
    tl.tracks.video = [seg("s001", "c001", 0.0, duration, mute_source=True)]
    tl.save(project.timeline_file)


def test_full_pipeline_places_a_clean_pickup(project: Project, tmp_path: Path, monkeypatch) -> None:
    _write_minimal_timeline(project)

    manifest = [{"file": "pickup.wav", "anchor": "s001", "offset": 0.0, "label": "intro"}]
    project.voice_incoming_dir.mkdir(parents=True, exist_ok=True)
    V.manifest_path(project).write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    sine(project.voice_incoming_dir / "pickup.wav", seconds=4.0)

    doc = transcript_doc([(0.3, 0.8, "Cześć,"), (1.0, 1.6, "jedziemy.")])
    monkeypatch.setattr(V, "_transcribe_voice_file", lambda proj, path: doc)

    result = V.run_voice(project)
    row = result["rows"][0]
    assert row["status"] == "ok"
    assert row["voice_item"] == "v001"

    out_wav = project.voice_dir / "intro.wav"
    assert out_wav.exists()
    assert (project.voice_dir / "intro.json").exists()
    assert Path(project.path / result["written"]).exists()

    timeline = Timeline.load(project.timeline_file)
    assert len(timeline.tracks.voice) == 1
    assert timeline.tracks.voice[0].file == "voice/intro.wav"
    assert timeline.tracks.voice[0].anchor == VoiceAnchor(segment="s001", offset=0.0)

    report = project.voice_incoming_dir / "report.md"
    assert report.exists()
    assert "intro" in report.read_text(encoding="utf-8")


def test_rerun_is_idempotent_when_source_is_unchanged(
    project: Project, tmp_path: Path, monkeypatch
) -> None:
    _write_minimal_timeline(project)
    manifest = [{"file": "pickup.wav", "anchor": "s001", "offset": 0.0, "label": "intro"}]
    project.voice_incoming_dir.mkdir(parents=True, exist_ok=True)
    V.manifest_path(project).write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    sine(project.voice_incoming_dir / "pickup.wav", seconds=4.0)

    doc = transcript_doc([(0.3, 0.8, "Cześć,"), (1.0, 1.6, "jedziemy.")])
    calls = {"n": 0}

    def fake_transcribe(proj, path):
        calls["n"] += 1
        return doc

    monkeypatch.setattr(V, "_transcribe_voice_file", fake_transcribe)

    V.run_voice(project)
    first_timeline = Timeline.load(project.timeline_file)
    assert len(first_timeline.tracks.voice) == 1
    assert calls["n"] == 1

    result2 = V.run_voice(project)
    assert result2["rows"][0]["status"] == "skipped (cached)"
    assert calls["n"] == 1  # no re-transcription

    second_timeline = Timeline.load(project.timeline_file)
    assert len(second_timeline.tracks.voice) == 1  # no duplicate
    assert len(second_timeline.tracks.video) == len(first_timeline.tracks.video)


def test_missing_manifest_writes_a_draft_and_raises(project: Project) -> None:
    (project.voice_incoming_dir / "a.wav").parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(V.VoiceError):
        V.load_manifest(project)
    assert V.manifest_path(project).exists()


def test_missing_timeline_raises(project: Project) -> None:
    project.voice_incoming_dir.mkdir(parents=True, exist_ok=True)
    V.manifest_path(project).write_text("[]\n", encoding="utf-8")
    with pytest.raises(V.VoiceError):
        V.run_voice(project)
