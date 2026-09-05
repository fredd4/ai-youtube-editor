"""End-to-end tests for ``ytedit.media.ingest`` against generated fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ytedit.media.ingest import discover_inputs, ingest, summarize
from ytedit.media.probe import probe
from ytedit.project import Project

#: input filename -> fixture name. The alphabetical order deliberately differs
#: from the recording order so the creation_time sort is actually exercised.
INPUTS: dict[str, str] = {
    "d_landscape.mp4": "landscape.mp4",   # creation_time 10:00
    "c_vertical.mp4": "vertical.mp4",     # 10:05
    "b_rotated.mov": "rotated.mov",       # 10:10
    "a_silent.mp4": "silent.mp4",         # 10:15
    "e_hlg.mp4": "hlg.mp4",               # 10:20
}

EXPECTED_ORDER = ["d_landscape.mp4", "c_vertical.mp4", "b_rotated.mov", "a_silent.mp4", "e_hlg.mp4"]


@pytest.fixture(scope="module")
def ingested(tmp_path_factory, request) -> Project:
    """A project with all fixtures ingested exactly once (module-scoped)."""
    from fixtures.make_fixtures import build_all

    media = build_all()
    root = tmp_path_factory.mktemp("projects")
    project = Project.create("ingest-test", language="pl", title="Ingest", root=root)
    for name, fixture in INPUTS.items():
        shutil.copy(media[fixture], project.input_dir / name)
    results = ingest(project, show_table=False)
    assert summarize(results).get("error", 0) == 0, [r.error for r in results]
    return project


def test_discover_inputs_finds_every_supported_file(ingested: Project) -> None:
    assert {p.name for p in discover_inputs(ingested)} == set(INPUTS)


def test_clips_are_numbered_in_recording_order(ingested: Project) -> None:
    clips = ingested.clips_in_order()
    assert [c["id"] for c in clips] == ["c001", "c002", "c003", "c004", "c005"]
    assert [Path(c["source_file"]).name for c in clips] == EXPECTED_ORDER
    assert [c["order"] for c in clips] == [1, 2, 3, 4, 5]
    assert all(c["source_file"].startswith("input/") for c in clips)


def test_every_derived_asset_exists(ingested: Project) -> None:
    for clip in ingested.clips_in_order():
        cid = clip["id"]
        assert ingested.source_path(cid).exists(), f"{cid} mezzanine"
        assert ingested.proxy_path(cid).exists(), f"{cid} proxy"
        assert ingested.audio_path(cid).exists(), f"{cid} wav"
        assert ingested.peaks_path(cid).exists(), f"{cid} peaks"
        assert ingested.poster_path(cid).exists(), f"{cid} poster"
        frames = sorted(ingested.clip_frames_dir(cid).glob("*.jpg"))
        assert frames, f"{cid} frames"
        assert frames[0].name == "000.jpg"
        assert clip["stages"]["ingest"] == "done"
        assert clip["frames_count"] == len(frames)


def test_state_json_records_probe_facts(ingested: Project) -> None:
    state = json.loads(ingested.state_file.read_text())
    assert state["project"] == "ingest-test"
    assert state["stages"]["ingest"]["status"] == "done"
    assert state["stages"]["ingest"]["clips"] == 5
    by_file = {Path(c["source_file"]).name: c for c in state["clips"].values()}

    landscape = by_file["d_landscape.mp4"]
    assert (landscape["width"], landscape["height"]) == (1920, 1080)
    assert landscape["orientation"] == "horizontal"
    assert landscape["has_audio"] is True
    assert landscape["recorded_at"].startswith("2026-08-12T10:00")

    rotated = by_file["b_rotated.mov"]
    assert rotated["rotation"] in (90, 270)
    assert (rotated["width"], rotated["height"]) == (1080, 1920)
    assert rotated["orientation"] == "vertical"

    assert by_file["a_silent.mp4"]["has_audio"] is False
    assert by_file["e_hlg.mp4"]["hdr"] == "hlg"


def test_normalized_sources_are_uniform(ingested: Project) -> None:
    for clip in ingested.clips_in_order():
        info = probe(ingested.source_path(clip["id"]))
        assert info.codec == "h264"
        assert info.pix_fmt == "yuv420p"
        assert info.bit_depth == 8
        # Every mezzanine has stereo 48 kHz audio, silent or not.
        assert info.has_audio is True
        assert info.audio_channels == 2
        assert info.audio_sample_rate == 48000
        assert info.rotation == 0, "rotation must be baked into the pixels"


def test_hlg_source_is_tonemapped_to_bt709(ingested: Project) -> None:
    hlg = next(c for c in ingested.clips_in_order() if c["hdr"] == "hlg")
    info = probe(ingested.source_path(hlg["id"]))
    assert info.color_transfer == "bt709"
    assert info.color_primaries == "bt709"
    assert info.hdr is None
    assert info.bit_depth == 8


def test_rotation_is_baked_into_the_mezzanine(ingested: Project) -> None:
    rotated = next(c for c in ingested.clips_in_order() if c["rotation"] in (90, 270))
    info = probe(ingested.source_path(rotated["id"]))
    assert (info.width, info.height) == (1080, 1920)
    assert info.rotation == 0


def test_proxies_are_720p_and_keep_aspect(ingested: Project) -> None:
    for clip in ingested.clips_in_order():
        info = probe(ingested.proxy_path(clip["id"]))
        assert max(info.width, info.height) <= 1280
        assert min(info.width, info.height) <= 720
        assert info.width % 2 == 0 and info.height % 2 == 0
        source = probe(ingested.source_path(clip["id"]))
        assert info.width / info.height == pytest.approx(source.width / source.height, rel=0.02)


def test_work_wav_is_mono_48k(ingested: Project) -> None:
    for clip in ingested.clips_in_order():
        info = json.loads(ingested.peaks_path(clip["id"]).read_text())
        assert info["sample_rate"] == 48000
        wav = ingested.audio_path(clip["id"])
        assert wav.stat().st_size > 1000


def test_peaks_format(ingested: Project) -> None:
    data = json.loads(ingested.peaks_path("c001").read_text())
    assert set(data) >= {"sample_rate", "peaks_per_second", "peaks", "duration"}
    assert data["peaks_per_second"] == pytest.approx(1000 / 60, rel=0.05)
    assert len(data["peaks"]) % 2 == 0
    # min/max pairs, in range, roughly peaks_per_second * duration pairs.
    pairs = len(data["peaks"]) // 2
    assert pairs == pytest.approx(data["peaks_per_second"] * data["duration"], abs=2)
    for lo, hi in zip(data["peaks"][::2], data["peaks"][1::2]):
        assert -1.01 <= lo <= hi <= 1.01
    assert any(hi > 0.05 for hi in data["peaks"][1::2]), "the sine tone should show up"


def test_silent_clip_has_flat_peaks(ingested: Project) -> None:
    silent = next(c for c in ingested.clips_in_order() if not c["has_audio"])
    data = json.loads(ingested.peaks_path(silent["id"]).read_text())
    assert max(abs(v) for v in data["peaks"]) < 1e-3


def test_reingest_is_idempotent(ingested: Project) -> None:
    results = ingest(ingested, show_table=False)
    assert summarize(results) == {"skipped": 5}
    assert [c["id"] for c in ingested.clips_in_order()] == ["c001", "c002", "c003", "c004", "c005"]


def test_force_rebuilds_one_clip(tmp_path: Path, media: dict[str, Path]) -> None:
    project = Project.create("force-test", language="pl", root=tmp_path / "projects")
    shutil.copy(media["silent.mp4"], project.input_dir / "one.mp4")
    assert summarize(ingest(project, show_table=False)) == {"done": 1}
    before = project.source_path("c001").stat().st_mtime_ns
    assert summarize(ingest(project, show_table=False)) == {"skipped": 1}
    results = ingest(project, force=True, show_table=False)
    assert summarize(results) == {"done": 1}
    assert "normalize" in results[0].steps
    assert project.source_path("c001").stat().st_mtime_ns != before


def test_stills_are_registered_only(tmp_path: Path, media: dict[str, Path]) -> None:
    project = Project.create("still-test", language="pl", root=tmp_path / "projects")
    from ytedit.media.ffmpeg import ff

    ff("-i", str(media["landscape.mp4"]), "-frames:v", "1", str(project.input_dir / "shot.jpg"))
    results = ingest(project, show_table=False)
    assert summarize(results) == {"registered": 1}
    clip = project.get_clip("c001")
    assert clip["kind"] == "still"
    assert clip["duration"] == 5.0
    assert clip["stages"]["ingest"] == "registered"


def test_empty_input_is_not_an_error(project: Project) -> None:
    assert ingest(project, show_table=False) == []
    assert project.stage_status("ingest") == "done"
