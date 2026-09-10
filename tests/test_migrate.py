"""Tests for ``ytedit.migrate`` — a v1 timeline rebuilt as a v2 cut.

Everything here is offline: synthetic transcripts, a synthetic clip registry
and a hand-built v1 ``Timeline``. No media, no ffmpeg, no API calls — narration
WAVs are empty files with a pre-seeded ``.probe.json`` so the resolver never
shells out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from ytedit.ai.sentences import write_sentences
from ytedit.cut import Beat
from ytedit.migrate import MigrationError, migrate_project
from ytedit.project import Project
from ytedit.timeline import (
    AudioFrom,
    Caption,
    Chapter,
    Marker,
    MuteRange,
    MusicCue,
    Timeline,
    VideoSegment,
    VoiceItem,
)


# ----------------------------------------------------------------------
# fixture helpers
# ----------------------------------------------------------------------
def add_clip(
    project: Project,
    clip_id: str,
    duration: float,
    words: Sequence[tuple[float, float, str]] = (),
) -> None:
    """Register a clip and write its transcript / (empty) analysis."""
    project.add_clip(
        {
            "id": clip_id, "order": int(clip_id[1:]), "duration": duration,
            "width": 1920, "height": 1080, "orientation": "horizontal",
            "has_audio": True,
        }
    )
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
    project.analysis_path(clip_id).write_text(
        json.dumps({"clip": clip_id, "instructions": [], "takes": []}), encoding="utf-8"
    )


#: c001 — the narrator, four clean sentences.
C001_WORDS = [
    (1.0, 1.5, "Pierwsze"), (1.6, 3.0, "zdanie."),
    (3.5, 4.2, "Drugie"), (4.3, 5.5, "zdanie."),
    (6.0, 6.7, "Trzecie"), (6.8, 8.0, "zdanie."),
    (8.5, 9.2, "Czwarte"), (9.3, 10.5, "zdanie."),
]

#: c004 — a second speaker, used as the source of a ``vo_`` extract.
C004_WORDS = [
    (1.0, 1.8, "Alfa"), (1.9, 3.0, "beta."),
    (3.5, 4.4, "Gamma"), (4.5, 5.6, "delta."),
    (6.2, 7.0, "Epsilon"), (7.1, 8.5, "dzeta."),
    (9.5, 10.2, "Eta"), (10.3, 11.4, "theta."),
]


@pytest.fixture()
def project(tmp_path: Path) -> Project:
    """A project with four registered clips and a written sentence catalogue."""
    proj = Project.create("t-migrate", language="pl", root=tmp_path / "projects")
    add_clip(proj, "c001", 30.0, C001_WORDS)
    add_clip(proj, "c002", 20.0)          # silent cutaway picture
    add_clip(proj, "c003", 20.0)          # silent B-roll picture
    add_clip(proj, "c004", 30.0, C004_WORDS)
    write_sentences(proj)
    return proj


def seg(
    sid: str,
    clip: str,
    in_: float,
    out: float,
    *,
    role: str = "a-roll",
    mute: bool = False,
    audio_from: tuple[str, float, float] | None = None,
    notes: str = "",
) -> VideoSegment:
    """One v1 video segment."""
    kwargs: dict[str, Any] = {
        "id": sid, "clip": clip, "out": out, "role": role, "mute_source": mute,
        "notes": notes, "in": in_,
    }
    if audio_from is not None:
        kwargs["audio_from"] = AudioFrom(
            clip=audio_from[0], out=audio_from[2], **{"in": audio_from[1]}
        )
    return VideoSegment(**kwargs)


def write_timeline(project: Project, segments: Sequence[VideoSegment], **tracks: Any) -> Timeline:
    """Assemble and save a v1 ``plan/timeline.json``."""
    timeline = Timeline(version=1, fps=30, language="pl")
    timeline.tracks.video = list(segments)
    timeline.tracks.voice = list(tracks.get("voice", []))
    timeline.tracks.music = list(tracks.get("music", []))
    timeline.tracks.captions = list(tracks.get("captions", []))
    timeline.chapters = list(tracks.get("chapters", []))
    timeline.markers = list(tracks.get("markers", []))
    timeline.save(project.timeline_file)
    return timeline


def add_voice_wav(project: Project, name: str, duration: float) -> str:
    """Create an empty ``voice/<name>`` with a pre-seeded ffprobe cache."""
    path = project.voice_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    cache = path.with_name(path.name + ".probe.json")
    cache.write_text(
        json.dumps({"mtime_ns": path.stat().st_mtime_ns, "duration": duration}),
        encoding="utf-8",
    )
    return f"voice/{name}"


def kinds(beats: Sequence[Beat]) -> list[str]:
    return [b.kind for b in beats]


def issue_codes(issues: Sequence[str]) -> list[str]:
    return [line.split(" ", 1)[0] for line in issues]


def run(project: Project, tmp_path: Path) -> Any:
    """A dry run into a scratch directory (never touches the project)."""
    return migrate_project(project, dry_run=True, out_dir=tmp_path / "out")


# ----------------------------------------------------------------------
# speech
# ----------------------------------------------------------------------
def test_speech_segment_becomes_a_speech_beat(project: Project, tmp_path: Path) -> None:
    write_timeline(project, [seg("s001", "c001", 0.8, 5.7)])

    report = run(project, tmp_path)

    assert kinds(report.cut.beats) == ["speech"]
    beat = report.cut.beats[0]
    assert beat.clip == "c001"
    assert beat.sentences == ["c001#1", "c001#2"]
    assert beat.on_camera is True
    assert beat.shots == []


def test_partial_edge_sentence_is_trimmed_and_reported(
    project: Project, tmp_path: Path
) -> None:
    # c001#2 runs 3.5-5.5; the v1 cut only reached 4.0, i.e. 25 % of it.
    write_timeline(project, [seg("s001", "c001", 0.8, 4.0)])

    report = run(project, tmp_path)

    assert report.cut.beats[0].sentences == ["c001#1"]
    trimmed = [i for i in report.issues if i.startswith("edge_trimmed")]
    assert len(trimmed) == 1
    assert "c001#2" in trimmed[0]
    assert "Drugie" in trimmed[0]


def test_cutaway_between_two_takes_becomes_a_shot(project: Project, tmp_path: Path) -> None:
    write_timeline(
        project,
        [
            seg("s001", "c001", 0.8, 3.2),
            seg("s002", "c002", 0.0, 3.0, role="cutaway", audio_from=("c001", 3.2, 6.2)),
            seg("s003", "c001", 6.2, 8.5),
        ],
    )

    report = run(project, tmp_path)

    assert kinds(report.cut.beats) == ["speech"]
    beat = report.cut.beats[0]
    assert beat.sentences == ["c001#1", "c001#2", "c001#3"]
    assert len(beat.shots) == 1
    shot = beat.shots[0]
    assert (shot.clip, shot.in_, shot.out) == ("c002", 0.0, 3.0)
    # c001#1 ends at 3.0, the cutaway borrowed audio from 3.2 — within 0.3 s.
    assert shot.after == "c001#1"
    assert "shot_after_approx" not in issue_codes(report.issues)


def test_drifted_handoff_still_merges_into_one_beat(project: Project, tmp_path: Path) -> None:
    # v1 replayed ~0.09 s at each hand-off; that is drift, not a second take.
    write_timeline(
        project,
        [
            seg("s001", "c001", 0.8, 3.29),
            seg("s002", "c002", 0.0, 3.0, role="cutaway", audio_from=("c001", 3.20, 6.20)),
            seg("s003", "c001", 6.11, 8.5),
        ],
    )

    report = run(project, tmp_path)

    assert kinds(report.cut.beats) == ["speech"]
    assert report.cut.beats[0].sentences == ["c001#1", "c001#2", "c001#3"]
    assert len(report.cut.beats[0].shots) == 1


def test_a_real_gap_in_the_same_clip_splits_the_beat(project: Project, tmp_path: Path) -> None:
    write_timeline(
        project,
        [seg("s001", "c001", 0.8, 3.2), seg("s002", "c001", 6.0, 8.5)],
    )

    report = run(project, tmp_path)

    assert kinds(report.cut.beats) == ["speech", "speech"]
    assert report.cut.beats[0].sentences == ["c001#1"]
    assert report.cut.beats[1].sentences == ["c001#3"]


def test_muted_segment_becomes_muted_broll(project: Project, tmp_path: Path) -> None:
    write_timeline(
        project,
        [
            seg("s001", "c003", 0.0, 4.0, role="b-roll", mute=True),
            seg("s002", "c002", 1.0, 5.0, role="b-roll"),
        ],
    )

    report = run(project, tmp_path)

    assert kinds(report.cut.beats) == ["broll", "broll"]
    assert report.cut.beats[0].audio == "mute"
    assert (report.cut.beats[0].in_, report.cut.beats[0].out) == (0.0, 4.0)
    assert report.cut.beats[1].audio == "ambient"


# ----------------------------------------------------------------------
# voice track
# ----------------------------------------------------------------------
def test_vo_extract_becomes_an_off_camera_speech_beat(
    project: Project, tmp_path: Path
) -> None:
    file = add_voice_wav(project, "vo_c004_001.00_009.00.wav", 8.0)
    write_timeline(
        project,
        [
            seg("s001", "c003", 0.0, 4.0, role="b-roll", mute=True),
            seg("s002", "c002", 0.0, 4.0, role="b-roll", mute=True),
        ],
        voice=[VoiceItem(id="v001", file=file, at=0.0, end=8.0)],
    )

    report = run(project, tmp_path)

    # the two muted segments are consumed by the beat, not emitted as B-roll
    assert kinds(report.cut.beats) == ["speech"]
    beat = report.cut.beats[0]
    assert beat.clip == "c004"
    assert beat.on_camera is False
    assert beat.sentences == ["c004#1", "c004#2", "c004#3"]
    assert [(s.clip, s.in_, s.out) for s in beat.shots] == [
        ("c003", 0.0, 4.0), ("c002", 0.0, 4.0)
    ]
    assert "vo_extract" in issue_codes(report.issues)


def test_recorded_pickup_becomes_a_voice_beat(project: Project, tmp_path: Path) -> None:
    file = add_voice_wav(project, "n001.wav", 6.0)
    write_timeline(
        project,
        [
            seg("s001", "c003", 0.0, 4.0, role="b-roll", mute=True),
            seg("s002", "c002", 0.0, 4.0, role="b-roll", mute=True),
        ],
        voice=[VoiceItem(id="v001", file=file, at=0.0, end=6.0)],
    )

    report = run(project, tmp_path)

    assert kinds(report.cut.beats) == ["voice", "broll"]
    beat = report.cut.beats[0]
    assert beat.file == "voice/n001.wav"
    assert [(s.clip, s.in_, s.out) for s in beat.shots] == [
        ("c003", 0.0, 4.0), ("c002", 0.0, 2.0)
    ]
    # the picture the pickup did not reach stays ordinary B-roll
    assert (report.cut.beats[1].clip, report.cut.beats[1].in_) == ("c002", 2.0)
    assert "voice_pickup" in issue_codes(report.issues)


def test_pickup_starting_mid_segment_splits_the_picture(
    project: Project, tmp_path: Path
) -> None:
    file = add_voice_wav(project, "n001.wav", 2.0)
    write_timeline(
        project,
        [seg("s001", "c003", 0.0, 6.0, role="b-roll", mute=True)],
        voice=[VoiceItem(id="v001", file=file, at=2.0, end=4.0)],
    )

    report = run(project, tmp_path)

    assert kinds(report.cut.beats) == ["broll", "voice", "broll"]
    assert (report.cut.beats[0].in_, report.cut.beats[0].out) == (0.0, 2.0)
    assert [(s.clip, s.in_, s.out) for s in report.cut.beats[1].shots] == [("c003", 2.0, 4.0)]
    assert (report.cut.beats[2].in_, report.cut.beats[2].out) == (4.0, 6.0)


# ----------------------------------------------------------------------
# absolute tracks
# ----------------------------------------------------------------------
def test_absolute_tracks_move_onto_beats(project: Project, tmp_path: Path) -> None:
    write_timeline(
        project,
        [
            seg("s001", "c003", 0.0, 4.0, role="b-roll", mute=True),
            seg("s002", "c001", 0.8, 5.7),
        ],
        captions=[Caption(id="t001", at=4.3, end=6.3, text="LISBOA", style="location")],
        music=[MusicCue(id="m001", file="music/m001.mp3", at=0.0, end=8.9)],
        chapters=[Chapter(at=0.0, title="Start")],
        markers=[Marker(at=4.0, label="hook")],
    )

    report = run(project, tmp_path)
    first, second = report.cut.beats

    caption = report.cut.captions[0]
    assert caption.beat in (second.id, second.uid)
    assert caption.offset == pytest.approx(0.3, abs=1e-3)
    assert caption.duration == pytest.approx(2.0, abs=1e-3)

    cue = report.cut.music[0]
    assert cue.from_ in (first.id, first.uid)
    assert cue.to in (second.id, second.uid)

    assert report.cut.chapters[0].beat in (first.id, first.uid)
    assert report.cut.markers[0].beat in (second.id, second.uid)


def test_mute_ranges_and_meta_are_carried_over(project: Project, tmp_path: Path) -> None:
    timeline = Timeline(version=1, fps=30, language="pl")
    timeline.tracks.video = [seg("s001", "c001", 0.8, 5.7)]
    timeline.mute_ranges = [MuteRange(clip="c001", s=0.0, e=0.5, reason="radio")]
    timeline.meta.generated_by = "plan@2026-09-06"
    timeline.meta.title_candidates = ["A", "B"]
    timeline.save(project.timeline_file)

    report = run(project, tmp_path)

    assert [m.clip for m in report.cut.mute_ranges] == ["c001"]
    assert report.cut.meta.title_candidates == ["A", "B"]
    assert report.cut.meta.generated_by.startswith("plan@2026-09-06 + migrate@")


# ----------------------------------------------------------------------
# report and io
# ----------------------------------------------------------------------
def test_markdown_report_lists_every_beat(project: Project, tmp_path: Path) -> None:
    write_timeline(
        project,
        [
            seg("s001", "c003", 0.0, 4.0, role="b-roll", mute=True),
            seg("s002", "c001", 0.8, 3.2),
            seg("s003", "c002", 0.0, 3.0, role="cutaway", audio_from=("c001", 3.2, 6.2)),
            seg("s004", "c001", 6.2, 8.5),
        ],
    )

    text = run(project, tmp_path).markdown()

    assert "# Migration v1 timeline → cut.json" in text
    assert "`b001` broll c003 0.00–4.00 (muted), 4.0 s ← s001" in text
    assert "`b002` speech c001 #1–#3, 1 shot, 7.7 s ← s002 s003 s004" in text
    assert "## Issues" in text
    assert "## Validation" in text
    assert "## Duration" in text


def test_dry_run_writes_only_to_out_dir(project: Project, tmp_path: Path) -> None:
    write_timeline(project, [seg("s001", "c001", 0.8, 5.7)])
    before = sorted(p.name for p in project.plan_dir.iterdir())
    out = tmp_path / "elsewhere"

    report = migrate_project(project, dry_run=True, out_dir=out)

    assert sorted(p.name for p in project.plan_dir.iterdir()) == before
    assert project.timeline_file.exists()
    assert not (project.plan_dir / "cut.json").exists()
    assert sorted(p.name for p in out.iterdir()) == ["cut.json", "migrate_report.md"]
    assert report.written == [out / "cut.json", out / "migrate_report.md"]


def test_dry_run_without_a_catalogue_builds_one_in_memory(
    project: Project, tmp_path: Path
) -> None:
    (project.analysis_dir / "sentences.json").unlink()
    (project.analysis_dir / "sentences.md").unlink()
    write_timeline(project, [seg("s001", "c001", 0.8, 5.7)])

    report = migrate_project(project, dry_run=True, out_dir=tmp_path / "out")

    assert report.cut.beats[0].sentences == ["c001#1", "c001#2"]
    assert not (project.analysis_dir / "sentences.json").exists()


def test_real_run_writes_the_cut_and_archives_the_v1_timeline(project: Project) -> None:
    write_timeline(project, [seg("s001", "c001", 0.8, 5.7)])

    report = migrate_project(project)

    assert (project.plan_dir / "cut.json").exists()
    assert (project.plan_dir / "migrate_report.md").exists()
    assert (project.plan_dir / "history" / "timeline.v1.json").exists()
    assert report.duration_v1 > 0
    # the resolver ran, so a fresh derived timeline is back in place
    assert project.timeline_file.exists()
    assert report.duration_v2 is not None
    assert json.loads(project.timeline_file.read_text(encoding="utf-8"))["version"] == 2


def test_missing_timeline_raises(project: Project, tmp_path: Path) -> None:
    with pytest.raises(MigrationError):
        migrate_project(project, dry_run=True, out_dir=tmp_path / "out")
