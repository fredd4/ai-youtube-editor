"""Tests for ``ytedit.qc``: the playbook rule checker and the file conformance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_render import EXPECTED_DURATION, build_render_project
from ytedit import qc as Q
from ytedit.config import Settings
from ytedit.media.render import render
from ytedit.project import Project


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def mastered(tmp_path_factory) -> Project:
    """A project with a full master rendered exactly once."""
    project = build_render_project(tmp_path_factory.mktemp("qc"), slug="qc-test")
    render(project, master=True)
    return project


@pytest.fixture(scope="module")
def master_report(mastered: Project) -> dict:
    """The QC document for the mastered project."""
    return Q.qc(mastered, show_table=False)


def write_timeline(project: Project, document: dict) -> None:
    """Write a timeline document into a project."""
    project.timeline_file.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def minimal(**overrides) -> dict:
    """A tiny valid timeline (one 4 s segment, no media dependencies)."""
    document: dict = {
        "version": 1, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
        "tracks": {"video": [
            {"id": "s001", "clip": "c001", "in": 0.0, "out": 4.0, "role": "cold-open"},
        ], "voice": [], "music": [], "captions": [], "sfx": []},
        "mute_ranges": [], "markers": [{"at": 0.0, "label": "hook"}],
        "chapters": [], "meta": {},
    }
    document.update(overrides)
    return document


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------
@pytest.mark.parametrize("phrase", [
    "no to dzięki za oglądanie i cześć",
    "Dzieki za ogladanie!",
    "do zobaczenia w kolejnym odcinku",
    "na tym kończymy dzisiejszy odcinek",
])
def test_ending_guard_matches_the_polish_wrap_ups(phrase: str) -> None:
    assert Q.ENDING_GUARD_RE.search(phrase)


@pytest.mark.parametrize("phrase", [
    "zobaczcie to miejsce",
    "kończy się droga",
    "dzięki temu zdążyliśmy",
])
def test_ending_guard_does_not_fire_on_ordinary_speech(phrase: str) -> None:
    assert Q.ENDING_GUARD_RE.search(phrase) is None


def test_subtract_removes_holes() -> None:
    assert Q.subtract([(0.0, 10.0)], [(2.0, 4.0)]) == [(0.0, 2.0), (4.0, 10.0)]
    assert Q.subtract([(0.0, 10.0)], [(0.0, 10.0)]) == []
    assert Q.subtract([(0.0, 5.0)], [(6.0, 7.0)]) == [(0.0, 5.0)]


def test_total_length_and_coverage() -> None:
    assert Q.total_length([(0.0, 2.0), (3.0, 4.5)]) == pytest.approx(3.5)
    assert Q._covered((0.0, 4.0), [(0.0, 2.0)]) == pytest.approx(0.5)
    assert Q._covered((0.0, 4.0), [(0.0, 9.0)]) == pytest.approx(1.0)


def test_top_level_atoms_and_faststart(mastered: Project) -> None:
    master = next(mastered.exports_dir.glob("master_*.mp4"))
    boxes = Q.top_level_atoms(master)
    assert boxes[0] == "ftyp"
    assert "moov" in boxes and "mdat" in boxes
    assert Q.has_faststart(master) is True


def test_faststart_returns_none_for_a_non_mp4(tmp_path: Path) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("definitely not an mp4", encoding="utf-8")
    assert Q.has_faststart(junk) is None


def test_find_rendered_prefers_the_master(mastered: Project) -> None:
    found = Q.find_rendered(mastered)
    assert found is not None and found.name.startswith("master_")


# ----------------------------------------------------------------------
# timeline rules
# ----------------------------------------------------------------------
def test_qc_without_a_timeline_raises(project: Project) -> None:
    with pytest.raises(Q.QCError, match="no timeline"):
        Q.qc(project, show_table=False)


def test_qc_without_a_render_still_checks_the_timeline(project: Project) -> None:
    write_timeline(project, minimal())
    report = Q.qc(project, show_table=False)
    assert report["ok"] is True
    assert any("no rendered file" in w for w in report["warnings"])
    assert (project.exports_dir / "qc_report.json").exists()
    assert (project.exports_dir / "qc_report.md").exists()
    assert project.load_state()["stages"]["qc"]["status"] == "done"


def test_report_files_carry_the_findings(project: Project) -> None:
    write_timeline(project, minimal())
    Q.qc(project, show_table=False)
    document = json.loads((project.exports_dir / "qc_report.json").read_text())
    assert set(document) >= {"ok", "errors", "warnings", "info", "checks", "project"}
    markdown = (project.exports_dir / "qc_report.md").read_text(encoding="utf-8")
    assert markdown.startswith(f"# QC report — {project.slug}")
    assert "## Warnings" in markdown and "## Measurements" in markdown


def test_missing_structural_markers_are_reported(project: Project) -> None:
    long_timeline = minimal(tracks={
        "video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 60.0, "role": "cold-open"}],
        "voice": [], "music": [], "captions": [], "sfx": [],
    }, markers=[])
    write_timeline(project, long_timeline)
    report = Q.qc(project, show_table=False)
    warning = next(w for w in report["warnings"] if w.startswith("rule 4"))
    assert "hook" in warning and "promise" in warning and "pattern-interrupt" in warning
    # 3:00 and 6:00 are past the end of a 60 s programme, so they are not asked for
    assert "re-engagement" not in warning


def test_a_marked_beat_is_accepted(project: Project) -> None:
    write_timeline(project, minimal(
        tracks={"video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 40.0,
                           "role": "cold-open"}],
                "voice": [], "music": [], "captions": [], "sfx": []},
        markers=[{"at": 0.0, "label": "hook"}, {"at": 7.0, "label": "promise"},
                 {"at": 30.0, "label": "pattern-interrupt"},
                 {"at": 20.0, "label": "payoff-cta"}],
    ))
    report = Q.qc(project, show_table=False)
    assert not [w for w in report["warnings"] if w.startswith("rule 4")]


def test_caption_outside_the_programme_is_an_error(project: Project) -> None:
    write_timeline(project, minimal(tracks={
        "video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 4.0, "role": "cold-open"}],
        "voice": [], "music": [],
        "captions": [{"id": "t001", "at": 3.0, "end": 9.0, "text": "ŁÓDŹ",
                      "style": "location", "position": "lower-left"}],
        "sfx": [],
    }))
    report = Q.qc(project, show_table=False)
    assert report["ok"] is False
    assert any("t001" in e for e in report["errors"])


def test_subtitle_in_the_end_screen_window_warns(project: Project) -> None:
    write_timeline(project, minimal(tracks={
        "video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 60.0, "role": "cold-open"}],
        "voice": [], "music": [],
        "captions": [{"id": "t009", "at": 50.0, "end": 55.0, "text": "ostatnie słowo",
                      "style": "subtitle", "position": "lower-center"}],
        "sfx": [],
    }))
    report = Q.qc(project, show_table=False)
    assert any("end screen" in w and "t009" in w for w in report["warnings"])


def test_vertical_clip_cropped_with_cover_warns(project: Project) -> None:
    project.add_clip({"id": "c001", "width": 1080, "height": 1920,
                      "orientation": "vertical", "duration": 10.0})
    write_timeline(project, minimal())
    report = Q.qc(project, show_table=False)
    warning = next(w for w in report["warnings"] if w.startswith("rule 15"))
    assert "blur-fill" in warning


def test_vertical_clip_with_blur_fill_is_accepted(project: Project) -> None:
    project.add_clip({"id": "c001", "width": 1080, "height": 1920,
                      "orientation": "vertical", "duration": 10.0})
    write_timeline(project, minimal(tracks={
        "video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 4.0, "role": "cold-open",
                   "transform": {"fit": "blur-fill"}}],
        "voice": [], "music": [], "captions": [], "sfx": [],
    }))
    report = Q.qc(project, show_table=False)
    assert not [w for w in report["warnings"] if w.startswith("rule 15")]


def test_tall_vertical_clip_is_told_to_crop_pan(project: Project) -> None:
    project.add_clip({"id": "c001", "width": 2160, "height": 3840,
                      "orientation": "vertical", "duration": 10.0})
    write_timeline(project, minimal())
    report = Q.qc(project, show_table=False)
    assert any("crop-pan" in w for w in report["warnings"])


def test_unanswered_background_music_flag_warns(project: Project) -> None:
    project.add_clip({"id": "c001", "duration": 10.0, "orientation": "horizontal"})
    (project.analysis_dir / "footage_log.json").write_text(json.dumps({
        "music_flags": [{"clip": "c001", "s": 1.0, "e": 3.0,
                         "confidence": 0.8, "suggest": "mute"}],
    }), encoding="utf-8")
    write_timeline(project, minimal())
    report = Q.qc(project, show_table=False)
    assert any(w.startswith("rule 19") for w in report["warnings"])
    assert report["checks"]["music_flags_unanswered"] == 1


def test_a_muted_background_music_flag_is_accepted(project: Project) -> None:
    project.add_clip({"id": "c001", "duration": 10.0, "orientation": "horizontal"})
    (project.analysis_dir / "footage_log.json").write_text(json.dumps({
        "music_flags": [{"clip": "c001", "s": 1.0, "e": 3.0, "suggest": "mute"}],
    }), encoding="utf-8")
    write_timeline(project, minimal(
        mute_ranges=[{"clip": "c001", "s": 0.5, "e": 3.5, "gain_db": -60}],
    ))
    report = Q.qc(project, show_table=False)
    assert not [w for w in report["warnings"] if w.startswith("rule 19")]
    assert report["checks"]["music_flags_unanswered"] == 0


def test_a_flag_on_footage_that_was_cut_is_ignored(project: Project) -> None:
    project.add_clip({"id": "c001", "duration": 30.0, "orientation": "horizontal"})
    (project.analysis_dir / "footage_log.json").write_text(json.dumps({
        "music_flags": [{"clip": "c001", "s": 20.0, "e": 25.0, "suggest": "mute"}],
    }), encoding="utf-8")
    write_timeline(project, minimal())      # the segment only uses 0-4 s
    report = Q.qc(project, show_table=False)
    assert report["checks"]["music_flags_unanswered"] == 0


def test_a_non_opening_first_segment_warns(project: Project) -> None:
    write_timeline(project, minimal(tracks={
        "video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 4.0, "role": "b-roll"}],
        "voice": [], "music": [], "captions": [], "sfx": [],
    }))
    report = Q.qc(project, show_table=False)
    assert any(w.startswith("opening:") for w in report["warnings"])


def test_a_font_without_polish_glyphs_is_a_hard_failure(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Settings, "caption_font", lambda self: ("Ghost", Path("/nowhere/ghost.ttf"))
    )
    write_timeline(project, minimal())
    report = Q.qc(project, show_table=False)
    assert report["ok"] is False
    assert any(e.startswith("rule 21") for e in report["errors"])


# ----------------------------------------------------------------------
# the rendered master
# ----------------------------------------------------------------------
def test_master_passes_qc(master_report: dict) -> None:
    assert master_report["ok"] is True, master_report["errors"]


def test_master_conforms_to_the_container_rules(master_report: dict) -> None:
    checks = master_report["checks"]
    assert checks["faststart"] is True
    assert checks["video"]["codec"] == "h264"
    assert checks["video"]["profile"].lower() == "high"
    assert checks["video"]["pix_fmt"] == "yuv420p"
    assert (checks["video"]["width"], checks["video"]["height"]) == (1920, 1080)
    assert checks["video"]["color_primaries"] == "bt709"
    assert checks["video"]["color_transfer"] == "bt709"
    assert checks["video"]["color_space"] == "bt709"
    assert checks["fps"] == pytest.approx(30.0)
    assert not checks.get("stream_start_times")


def test_master_audio_is_aac_lc_48k_stereo(master_report: dict) -> None:
    audio = master_report["checks"]["audio"]
    assert audio["codec"] == "aac"
    assert audio["profile"].upper() == "LC"
    assert audio["sample_rate"] == 48000
    assert audio["channels"] == 2
    assert audio["bit_rate"] >= Q.AAC_BITRATE_FLOOR


def test_master_loudness_hits_the_target(master_report: dict) -> None:
    loudness = master_report["checks"]["loudness"]
    assert loudness["integrated_lufs"] == pytest.approx(-14.0, abs=Q.LOUDNESS_TOLERANCE)
    assert loudness["true_peak_dbtp"] <= -1.0


def test_master_duration_matches_the_timeline(master_report: dict) -> None:
    assert master_report["checks"]["rendered_duration"] == pytest.approx(
        EXPECTED_DURATION, abs=0.25
    )
    assert master_report["checks"]["rendered_master"] is True


def test_qc_records_the_stage_and_the_report_path(mastered: Project, master_report: dict) -> None:
    stage = mastered.load_state()["stages"]["qc"]
    assert stage["status"] == "done"
    assert stage["ok"] is True
    assert stage["report"] == "exports/qc_report.json"
