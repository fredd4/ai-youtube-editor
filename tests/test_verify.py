"""Tests for ``ytedit.verify`` — the ``ytedit check-render`` alignment pass.

Offline: synthetic timelines, synthetic source/render transcripts, no ffmpeg
and no network. The one end-to-end case replaces the ElevenLabs client and the
ffmpeg extraction the same way ``test_transcribe.py`` does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from ytedit import verify
from ytedit.words import Word
from ytedit.cut import Beat, Cut, Shot
from ytedit.project import Project
from ytedit.timeline import AudioFrom, Timeline, VideoSegment, VoiceItem, new_timeline


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def seg(seg_id: str, clip: str, start: float, end: float, **extra: Any) -> VideoSegment:
    """Build a video segment (the resolver's output shape, stated by hand)."""
    return VideoSegment(id=seg_id, clip=clip, **{"in": start}, out=end, **extra)


def timeline_of(*segments: VideoSegment) -> Timeline:
    """Wrap segments in a timeline."""
    tl = new_timeline()
    tl.tracks.video = list(segments)
    return tl


def words(*spec: tuple[str, float, float]) -> list[Word]:
    """``("tekst", start, end)`` triples -> :class:`Word` list."""
    return [Word(s, e, text) for text, s, e in spec]


def by_label(reports: Sequence[verify.BoundaryReport], label: str) -> verify.BoundaryReport:
    """The single report carrying ``label`` (fails loudly when absent)."""
    found = [r for r in reports if r.label == label]
    assert len(found) == 1, f"{label} not in {[r.label for r in reports]}"
    return found[0]


#: Two clips of well-separated speech, one word per second.
C1 = words(("Raz", 1.0, 1.4), ("Dwa", 2.0, 2.4), ("Trzy", 3.0, 3.4), ("Cztery", 8.0, 8.4))
C2 = words(("Pięć", 0.6, 1.0), ("Sześć", 2.0, 2.4), ("Siedem", 4.0, 4.4))


# ----------------------------------------------------------------------
# transcript_words
# ----------------------------------------------------------------------
def test_transcript_words_drops_artefacts_and_non_words() -> None:
    payload = {
        "words": [
            {"t": "Raz", "s": 0.0, "e": 0.4, "type": "word"},
            {"t": " ", "s": 0.4, "e": 0.5, "type": "spacing"},
            {"t": "(music)", "s": 1.0, "e": 9.0, "type": "audio_event"},
            # A Scribe artefact over the music bed: 4 s is not a word.
            {"t": "mmmm", "s": 10.0, "e": 14.0, "type": "word"},
            {"t": "Dwa", "s": 20.0, "e": 20.4},
        ]
    }
    assert [w.text for w in verify.transcript_words(payload)] == ["Raz", "Dwa"]


def test_transcript_words_sorts_by_start() -> None:
    payload = {"words": [{"t": "b", "s": 5.0, "e": 5.2}, {"t": "a", "s": 1.0, "e": 1.2}]}
    assert [w.text for w in verify.transcript_words(payload)] == ["a", "b"]


# ----------------------------------------------------------------------
# which boundaries are real
# ----------------------------------------------------------------------
def test_a_continuous_handoff_is_not_a_boundary() -> None:
    """Two same-clip pieces glued frame to frame are a jump cut, not a cut."""
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 3.6),
        seg("s002", "c001", 3.6, 8.6),
    )
    reports = verify.analyze(timeline, {"c001": C1}, [])
    assert reports == []


def test_a_shot_carrying_the_narration_is_not_a_boundary() -> None:
    """``audio_from`` continuing the same take under a cutaway is a hand-off."""
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 3.6),
        seg("s002", "c009", 0.0, 2.0, audio_from=AudioFrom(clip="c001", **{"in": 3.6}, out=5.6)),
    )
    assert verify.analyze(timeline, {"c001": C1}, []) == []


def test_two_muted_picture_segments_report_no_boundary() -> None:
    """A voice-over picture run has no source audio on either side."""
    timeline = timeline_of(
        seg("s001", "c001", 0.0, 2.0, mute_source=True),
        seg("s002", "c002", 0.0, 2.0, mute_source=True),
    )
    assert [r.label for r in verify.analyze(timeline, {}, [])] == []


def test_a_real_clip_change_is_a_boundary() -> None:
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 3.85),
        seg("s002", "c002", 0.3, 2.85),
    )
    reports = verify.analyze(timeline, {"c001": C1, "c002": C2}, [])
    report = by_label(reports, "s001 -> s002")
    assert report.kind == "cut"
    assert report.at == pytest.approx(3.35, abs=0.05)  # 3.85 - 0.5, on a frame
    # A: 'Trzy' ends 3.4, the cut is at 3.85 -> 0.45s of air.
    assert report.before is not None and report.before.word == "Trzy"
    assert report.before.margin == pytest.approx(0.45)
    # B: 'Pięć' starts 0.6, the cut opens at 0.3 -> 0.30s of air.
    assert report.after is not None and report.after.word == "Pięć"
    assert report.after.margin == pytest.approx(0.30)
    assert report.flags == []


# ----------------------------------------------------------------------
# source-side flags
# ----------------------------------------------------------------------
def test_an_out_inside_a_word_is_flagged() -> None:
    """The c212 bug seen from the render: ``out`` lands mid-word."""
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 8.2),   # 8.2 is inside 'Cztery' [8.0, 8.4]
        seg("s002", "c002", 0.3, 2.85),
    )
    report = by_label(verify.analyze(timeline, {"c001": C1, "c002": C2}, []), "s001 -> s002")
    assert report.before is not None
    assert report.before.word == "Cztery" and report.before.inside_word
    assert report.before.margin == pytest.approx(-0.2)
    assert "out inside word" in report.flags


def test_an_in_inside_a_word_is_flagged() -> None:
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 3.85),
        seg("s002", "c002", 0.8, 2.85),  # 0.8 is inside 'Pięć' [0.6, 1.0]
    )
    report = by_label(verify.analyze(timeline, {"c001": C1, "c002": C2}, []), "s001 -> s002")
    assert report.after is not None and report.after.inside_word
    assert "in inside word" in report.flags


def test_the_same_clip_replayed_across_a_boundary_is_flagged() -> None:
    """Not a hand-off (the second piece steps *back*): audio heard twice."""
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 3.85),
        seg("s002", "c001", 3.0, 8.5),
    )
    report = by_label(verify.analyze(timeline, {"c001": C1}, []), "s001 -> s002")
    assert report.replayed == pytest.approx(0.85)
    assert any(f.startswith("replayed") for f in report.flags)


def test_a_replay_under_the_threshold_is_not_flagged() -> None:
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 3.85),
        seg("s002", "c001", 3.81, 8.5),  # 40 ms — more than a frame, less than 50 ms
    )
    report = by_label(verify.analyze(timeline, {"c001": C1}, []), "s001 -> s002")
    assert report.replayed == pytest.approx(0.04)
    assert report.flags == []


# ----------------------------------------------------------------------
# render-side flags
# ----------------------------------------------------------------------
def test_the_render_gap_is_measured_between_the_two_nearest_words() -> None:
    timeline = timeline_of(seg("s001", "c001", 0.5, 3.85), seg("s002", "c002", 0.3, 2.85))
    render = words(("Trzy", 2.4, 2.9), ("Pięć", 3.7, 4.1))
    report = by_label(verify.analyze(timeline, {"c001": C1, "c002": C2}, render), "s001 -> s002")
    assert report.render_before == "Trzy" and report.render_after == "Pięć"
    assert report.render_gap == pytest.approx(0.8)
    assert report.render_phrase() == "...Trzy [0.80s] Pięć..."
    assert report.flags == []


def test_too_little_air_in_the_render_is_flagged_as_no_air() -> None:
    timeline = timeline_of(seg("s001", "c001", 0.5, 3.85), seg("s002", "c002", 0.3, 2.85))
    render = words(("Trzy", 2.9, 3.3), ("Pięć", 3.5, 4.1))
    report = by_label(verify.analyze(timeline, {"c001": C1, "c002": C2}, render), "s001 -> s002")
    assert report.render_gap == pytest.approx(0.2)
    assert "no air" in report.flags


def test_a_word_straddling_the_boundary_is_flagged_as_chopped() -> None:
    timeline = timeline_of(seg("s001", "c001", 0.5, 3.85), seg("s002", "c002", 0.3, 2.85))
    render = words(("początkowo", 3.0, 3.8))  # the cut lands at 3.35, mid-word
    report = by_label(verify.analyze(timeline, {"c001": C1, "c002": C2}, render), "s001 -> s002")
    assert report.straddle is not None and report.straddle["text"] == "początkowo"
    assert "chopped word" in report.flags


def test_a_word_barely_touching_the_boundary_is_not_chopped() -> None:
    timeline = timeline_of(seg("s001", "c001", 0.5, 3.85), seg("s002", "c002", 0.3, 2.85))
    render = words(("kończy", 3.3, 3.4))  # 50 ms either side: under the threshold
    report = by_label(verify.analyze(timeline, {"c001": C1, "c002": C2}, render), "s001 -> s002")
    assert report.straddle is None
    assert "chopped word" not in report.flags


def test_a_boundary_with_words_on_one_side_only_reports_no_gap() -> None:
    timeline = timeline_of(seg("s001", "c001", 0.5, 3.85), seg("s002", "c002", 0.3, 2.85))
    report = by_label(
        verify.analyze(timeline, {"c001": C1, "c002": C2}, words(("Trzy", 2.4, 2.9))),
        "s001 -> s002",
    )
    assert report.render_gap is None
    assert report.render_phrase() == "(no words both sides)"
    assert report.flags == []


# ----------------------------------------------------------------------
# voice pickups
# ----------------------------------------------------------------------
def test_every_voice_pickup_contributes_a_start_and_an_end() -> None:
    timeline = timeline_of(seg("s001", "c001", 0.0, 10.0, mute_source=True))
    timeline.tracks.voice = [VoiceItem(id="v001", file="voice/vo.wav", at=2.0, end=5.0)]
    render = words(
        ("koniec", 1.0, 1.5), ("start", 2.2, 2.6), ("narracja", 4.2, 4.9), ("dalej", 5.4, 5.9)
    )
    reports = verify.analyze(timeline, {"c001": C1}, render)
    assert [r.label for r in reports] == ["v001 start", "v001 end"]
    assert [r.kind for r in reports] == ["voice-in", "voice-out"]
    assert by_label(reports, "v001 start").render_gap == pytest.approx(0.7)
    assert by_label(reports, "v001 end").render_gap == pytest.approx(0.5)


def test_a_pickup_landing_mid_word_in_the_render_is_flagged() -> None:
    timeline = timeline_of(seg("s001", "c001", 0.0, 10.0, mute_source=True))
    timeline.tracks.voice = [VoiceItem(id="v001", file="voice/vo.wav", at=2.0, end=5.0)]
    render = words(("wybrzeżu", 1.5, 2.6))
    start = by_label(verify.analyze(timeline, {"c001": C1}, render), "v001 start")
    assert start.straddle is not None and "chopped word" in start.flags


def test_reports_are_sorted_by_render_time() -> None:
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 3.85),
        seg("s002", "c002", 0.3, 2.85),
        seg("s003", "c001", 7.9, 9.0),
    )
    timeline.tracks.voice = [VoiceItem(id="v001", file="voice/vo.wav", at=1.0, end=4.0)]
    reports = verify.analyze(timeline, {"c001": C1, "c002": C2}, [])
    assert [round(r.at, 3) for r in reports] == sorted(round(r.at, 3) for r in reports)
    assert {r.label for r in reports} == {
        "s001 -> s002", "s002 -> s003", "v001 start", "v001 end"
    }


# ----------------------------------------------------------------------
# beat labels — where a correction actually goes
# ----------------------------------------------------------------------
def _beat_cut() -> tuple[Cut, list]:
    """A cut of two beats and the segments a resolve of it would produce.

    ``b001`` is a speech beat with one shot in the middle (own picture, shot,
    own picture); ``b002`` is plain B-roll after it.
    """
    beat = Beat(
        kind="speech", clip="c001", sentences=["c001#1", "c001#2"],
        shots=[Shot(clip="c009", **{"in": 0.0}, out=2.0, after="c001#1")],
    )
    other = Beat(kind="broll", clip="c002", **{"in": 0.3}, out=2.85)
    cut = Cut(beats=[beat, other])
    cut.renumber()
    segments = [
        seg("s001", "c001", 0.5, 3.6, beat=beat.uid, uid="u1"),
        seg("s002", "c009", 0.0, 2.0, beat=beat.uid, uid="u2",
            audio_from=AudioFrom(clip="c001", **{"in": 3.6}, out=5.6)),
        seg("s003", "c001", 5.6, 8.6, beat=beat.uid, uid="u3"),
        seg("s004", "c002", 0.3, 2.85, beat=other.uid, uid="u4"),
    ]
    return cut, segments


def test_beat_labels_name_the_beat_and_which_shot_is_on_screen() -> None:
    cut, segments = _beat_cut()
    labels = verify.beat_labels(cut, segments)
    assert labels["u1"] == "b001 (own picture)"
    assert labels["u2"] == "b001 (shot 1)"
    assert labels["u3"] == "b001 (own picture)"
    assert labels["u4"] == "b002 (own picture)"


def test_a_boundary_carries_the_beats_on_both_sides() -> None:
    cut, segments = _beat_cut()
    reports = verify.analyze(timeline_of(*segments), {"c001": C1, "c002": C2}, [], cut=cut)
    boundary = by_label(reports, "s003 -> s004")
    assert boundary.beat_before == "b001 (own picture)"
    assert boundary.beat_after == "b002 (own picture)"
    assert boundary.beats == "b001 (own picture) -> b002 (own picture)"
    assert boundary.to_dict()["beats"] == boundary.beats


def test_without_a_cut_a_boundary_has_no_beats() -> None:
    _cut, segments = _beat_cut()
    reports = verify.analyze(timeline_of(*segments), {"c001": C1, "c002": C2}, [])
    assert by_label(reports, "s003 -> s004").beats == "-"


def test_a_voice_boundary_is_labelled_from_the_programme() -> None:
    cut, segments = _beat_cut()
    timeline = timeline_of(*segments)
    timeline.tracks.voice.append(
        VoiceItem(id="v001", file="voice/n001.wav", at=0.5, end=1.5)
    )
    reports = verify.analyze(timeline, {"c001": C1, "c002": C2}, [], cut=cut)
    assert by_label(reports, "v001 start").beat_after == "b001 (own picture)"


def test_the_markdown_report_carries_a_beats_column() -> None:
    cut, segments = _beat_cut()
    reports = verify.analyze(timeline_of(*segments), {"c001": C1, "c002": C2}, [], cut=cut)
    counts = verify.summarize(reports)

    class _Fake:
        slug = "t"

    md = verify.report_markdown(_Fake(), Path("draft.mp4"), reports, counts)
    assert "| at | boundary | beats | flags | source | render |" in md
    assert "b001 (own picture) -> b002 (own picture)" in md


# ----------------------------------------------------------------------
# summary / report
# ----------------------------------------------------------------------
def test_summarize_counts_every_flag() -> None:
    timeline = timeline_of(
        seg("s001", "c001", 0.5, 8.2),   # out inside 'Cztery'
        seg("s002", "c001", 7.0, 9.0),   # replays 1.2s of c001
    )
    counts = verify.summarize(verify.analyze(timeline, {"c001": C1}, []))
    assert counts["boundaries"] == 1
    assert counts["flagged"] == 1
    assert counts["out inside word"] == 1
    assert counts["replayed"] == 1


def test_timecode_formats_minutes_and_hours() -> None:
    assert verify.timecode(3.4) == "00:03.40"
    assert verify.timecode(125.5) == "02:05.50"
    assert verify.timecode(3725.25) == "1:02:05.25"


# ----------------------------------------------------------------------
# render resolution
# ----------------------------------------------------------------------
def test_resolve_render_finds_the_named_tiers(project: Project) -> None:
    from ytedit.timeline import new_timeline

    timeline = new_timeline()
    (project.renders_dir / "draft.mp4").write_bytes(b"x")
    (project.exports_dir / "master_1080p.mp4").write_bytes(b"x")
    assert verify.resolve_render(project, timeline, "draft").name == "draft.mp4"
    assert verify.resolve_render(project, timeline, "master").name == "master_1080p.mp4"
    with pytest.raises(verify.VerifyError, match="preview.mp4"):
        verify.resolve_render(project, timeline, "preview")


def test_resolve_render_accepts_an_explicit_path(project: Project) -> None:
    from ytedit.timeline import new_timeline

    custom = project.path / "renders" / "cut-v3.mp4"
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_bytes(b"x")
    assert verify.resolve_render(project, new_timeline(), str(custom)) == custom


# ----------------------------------------------------------------------
# end to end, with the STT client and ffmpeg replaced
# ----------------------------------------------------------------------
class FakeTranscript:
    """Minimal stand-in for :class:`ytedit.ai.elevenlabs.Transcript`."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self.payload


class FakeElevenLabs:
    """Stand-in for the STT client — records its calls, spends nothing real."""

    calls: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}

    def __init__(self, api_key: str, cost_callback: Any = None, **_: Any) -> None:
        self.cost_callback = cost_callback

    def transcribe(self, audio_path: Path, **kwargs: Any) -> FakeTranscript:
        type(self).calls.append({"audio": str(audio_path), **kwargs})
        if self.cost_callback:
            self.cost_callback(
                service="elevenlabs", op="stt", model="scribe_v2", units="60.0s", usd=0.36
            )
        return FakeTranscript(type(self).payload)

    def close(self) -> None:
        return None


@pytest.fixture()
def checkable(project: Project, monkeypatch: pytest.MonkeyPatch) -> Project:
    """A project with a timeline, a fake draft render and a fake STT client."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    for clip_id, clip_words in (("c001", C1), ("c002", C2)):
        project.add_clip({"id": clip_id, "order": int(clip_id[1:]), "duration": 20.0,
                          "has_audio": True})
        project.transcript_path(clip_id).write_text(
            json.dumps({"words": [{"t": w.text, "s": w.s, "e": w.e} for w in clip_words]}),
            encoding="utf-8",
        )
    timeline = timeline_of(seg("s001", "c001", 0.5, 3.85), seg("s002", "c002", 0.3, 2.85))
    timeline.save(project.timeline_file)
    (project.renders_dir / "draft.mp4").write_bytes(b"not really an mp4")

    FakeElevenLabs.calls = []
    FakeElevenLabs.payload = {
        "language": "pol",
        "words": [
            {"t": "Trzy", "s": 2.9, "e": 3.3, "type": "word"},
            {"t": "Pięć", "s": 3.5, "e": 4.1, "type": "word"},
        ],
    }
    monkeypatch.setattr("ytedit.ai.elevenlabs.ElevenLabs", FakeElevenLabs)
    monkeypatch.setattr(
        verify, "extract_audio", lambda proj, render: verify.check_dir(proj) / "draft.wav"
    )
    return project


def test_check_render_transcribes_aligns_and_writes_a_report(checkable: Project) -> None:
    result = verify.check_render(checkable, render="draft", write_json=True)

    assert result["cached"] is False
    call = FakeElevenLabs.calls[0]
    assert call["language_code"] == "pol"      # the project language, pinned
    assert call["diarize"] is False
    assert result["counts"] == {"boundaries": 1, "flagged": 1, "no air": 1}
    boundary = result["boundaries"][0]
    assert boundary["label"] == "s001 -> s002"
    assert boundary["before"]["word"] == "Trzy" and boundary["after"]["word"] == "Pięć"
    assert boundary["render_gap"] == pytest.approx(0.2)

    report = Path(result["report_md"])
    assert report.name == "draft.report.md"
    body = report.read_text(encoding="utf-8")
    assert "s001 -> s002" in body and "no air" in body
    assert Path(result["report_json"]).exists()

    # The call was charged to the project ledger under its own stage.
    stages = [c.get("stage") for c in checkable.state.get("costs", [])]
    assert "check-render" in stages


def test_check_render_reuses_the_cached_transcript(checkable: Project) -> None:
    verify.check_render(checkable, render="draft")
    assert len(FakeElevenLabs.calls) == 1

    second = verify.check_render(checkable, render="draft")
    assert second["cached"] is True
    assert len(FakeElevenLabs.calls) == 1  # free

    # Touching the render invalidates the cache.
    (checkable.renders_dir / "draft.mp4").write_bytes(b"a different render entirely")
    third = verify.check_render(checkable, render="draft")
    assert third["cached"] is False
    assert len(FakeElevenLabs.calls) == 2


def test_check_render_force_re_transcribes(checkable: Project) -> None:
    verify.check_render(checkable, render="draft")
    verify.check_render(checkable, render="draft", force=True)
    assert len(FakeElevenLabs.calls) == 2


def test_check_render_without_a_timeline_fails_cleanly(project: Project) -> None:
    with pytest.raises(verify.VerifyError, match="no timeline"):
        verify.check_render(project)
