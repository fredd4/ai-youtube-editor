"""Tests for ``ytedit.media.ingest``.

The compatibility rule (:func:`ytedit.media.ingest.mezzanine_mode`) is unit
tested against hand-built ``MediaInfo`` values; everything else runs a real
ingest over the generated fixtures, one clip per branch of that rule.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ytedit.config import OutputFormat, load_settings
from ytedit.media.ingest import (
    discover_inputs,
    ensure_proxies,
    ingest,
    mezzanine_line,
    mezzanine_mode,
    mezzanine_summary,
    summarize,
)
from ytedit.media.probe import MediaInfo, probe
from ytedit.project import Project

#: input filename -> fixture name. The alphabetical order deliberately differs
#: from the recording order so the creation_time sort is actually exercised.
INPUTS: dict[str, str] = {
    "d_landscape.mp4": "landscape.mp4",   # creation_time 10:00
    "c_vertical.mp4": "vertical.mp4",     # 10:05
    "b_rotated.mov": "rotated.mov",       # 10:10
    "a_silent.mp4": "silent.mp4",         # 10:15
    "e_hlg.mp4": "hlg.mp4",               # 10:20
    "g_hlg8.mp4": "hlg8.mp4",             # 10:30
    "f_vfr.mp4": "vfr1080.mp4",           # 10:35 — recorded after g_hlg8, sorts before it
}

EXPECTED_ORDER = [
    "d_landscape.mp4", "c_vertical.mp4", "b_rotated.mov", "a_silent.mp4",
    "e_hlg.mp4", "g_hlg8.mp4", "f_vfr.mp4",
]

#: input filename -> ``(mode, first word of the reason)`` ingest must reach.
#: One clip per branch of :func:`ytedit.media.ingest.mezzanine_mode`, including
#: the two rules (``hdr``, ``vfr``) an earlier check hides on the older fixtures.
EXPECTED_MEZZANINE: dict[str, tuple[str, str]] = {
    "d_landscape.mp4": ("copy", ""),
    "c_vertical.mp4": ("copy", ""),                  # the vertical inverse of the format
    "b_rotated.mov": ("encode", "rotation"),
    "a_silent.mp4": ("encode", "size"),
    "e_hlg.mp4": ("encode", "pix_fmt"),              # 10-bit: pix_fmt fires before hdr
    "g_hlg8.mp4": ("encode", "hdr"),                 # 8-bit HLG: only the transfer differs
    "f_vfr.mp4": ("encode", "vfr"),                  # 1080p VFR: only the timing differs
}


# ----------------------------------------------------------------------
# the compatibility rule (no ffmpeg)
# ----------------------------------------------------------------------
FORMAT: OutputFormat = load_settings().format


def info_like(fmt: OutputFormat, **overrides) -> MediaInfo:
    """A ``MediaInfo`` that matches ``fmt`` exactly, minus ``overrides``.

    The baseline is a clip :func:`mezzanine_mode` must copy, so each test can
    break exactly one property and name the reason it expects back.
    """
    base: dict = {
        "path": "/nowhere/clip.mp4",
        "codec": fmt.codec,
        "pix_fmt": fmt.pix_fmt,
        "width": fmt.width,
        "height": fmt.height,
        "fps": float(fmt.fps),
        "r_fps": float(fmt.fps),
        "avg_fps": float(fmt.fps),
        "duration": 6.0,
    }
    return MediaInfo(**{**base, **overrides})


def test_a_source_that_already_is_the_format_is_copied() -> None:
    decision = mezzanine_mode(info_like(FORMAT), FORMAT)
    assert (decision.mode, decision.reason) == ("copy", "")
    assert decision.is_copy is True


def test_a_vertical_source_is_copied_at_its_own_size() -> None:
    """1080x1920 against a 1920x1080 format: the renderer's blur-fill places it."""
    decision = mezzanine_mode(
        info_like(FORMAT, width=FORMAT.height, height=FORMAT.width), FORMAT
    )
    assert (decision.mode, decision.reason) == ("copy", "")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"codec": "hevc"}, "codec hevc"),
        ({"pix_fmt": "yuv420p10le"}, "pix_fmt yuv420p10le"),
        # A remux keeps the display matrix instead of baking it into the pixels.
        ({"rotation": 90}, "rotation 90"),
        ({"width": 1280, "height": 720}, "size 1280x720"),
        ({"vfr": True}, "vfr"),
        ({"hdr": "hlg"}, "hdr hlg"),
        ({"hdr": "pq"}, "hdr pq"),
        ({"fps": 25.0}, "fps 25"),
    ],
    ids=["codec", "pix_fmt", "rotation", "size", "vfr", "hdr_hlg", "hdr_pq", "fps"],
)
def test_each_mismatch_forces_an_encode(overrides: dict, reason: str) -> None:
    decision = mezzanine_mode(info_like(FORMAT, **overrides), FORMAT)
    assert (decision.mode, decision.reason) == ("encode", reason)
    assert decision.is_copy is False


def test_the_first_failed_check_is_the_reported_reason() -> None:
    """The documented order is codec, pix_fmt, rotation, size, vfr, hdr, fps.

    The reason lands in ``state.json`` and in the ingest summary, so it has to
    be deterministic rather than "whichever check happened to run".
    """
    broken = {
        "codec": "hevc", "pix_fmt": "yuv420p10le", "rotation": 90,
        "width": 1280, "height": 720, "vfr": True, "hdr": "pq", "fps": 25.0,
    }
    expected = [
        "codec hevc", "pix_fmt yuv420p10le", "rotation 90",
        "size 1280x720", "vfr", "hdr pq", "fps 25",
    ]
    fixes = [
        {"codec": FORMAT.codec},
        {"pix_fmt": FORMAT.pix_fmt},
        {"rotation": 0},
        {"width": FORMAT.width, "height": FORMAT.height},
        {"vfr": False},
        {"hdr": None},
    ]
    reasons = [mezzanine_mode(info_like(FORMAT, **broken), FORMAT).reason]
    for fix in fixes:
        broken.update(fix)
        reasons.append(mezzanine_mode(info_like(FORMAT, **broken), FORMAT).reason)
    assert reasons == expected
    broken["fps"] = float(FORMAT.fps)
    assert mezzanine_mode(info_like(FORMAT, **broken), FORMAT).mode == "copy"


@pytest.mark.parametrize("fps", [29.951, 29.98, 30.0, 30.03, 30.049])
def test_fps_within_the_tolerance_still_copies(fps: float) -> None:
    """``abs(info.fps - fmt.fps) <= 0.05`` — a hair off 30 is still 30.

    The values stay clear of the nominal +-0.05 edge itself: 29.95 and 30.05
    are *outside* it in binary floating point, which is an accident of the
    representation rather than a promise the rule makes.
    """
    assert abs(fps - FORMAT.fps) <= 0.05, "the test value must be inside the tolerance"
    assert mezzanine_mode(info_like(FORMAT, fps=fps), FORMAT).mode == "copy"


@pytest.mark.parametrize("fps", [29.9, 30.06, 25.0, 59.94])
def test_fps_outside_the_tolerance_forces_an_encode(fps: float) -> None:
    assert abs(fps - FORMAT.fps) > 0.05, "the test value must be outside the tolerance"
    decision = mezzanine_mode(info_like(FORMAT, fps=fps), FORMAT)
    assert (decision.mode, decision.reason) == ("encode", f"fps {fps:g}")


def test_the_format_is_the_target_not_a_hardcoded_1080p() -> None:
    """A project that renders 4K at 24 fps measures its sources against *that*."""
    fmt = OutputFormat(width=3840, height=2160, fps=24, codec="hevc",
                       pix_fmt="yuv420p10le", sar="1/1")
    assert mezzanine_mode(info_like(fmt), fmt).mode == "copy"
    assert mezzanine_mode(info_like(FORMAT), fmt).reason == "codec h264"


# ----------------------------------------------------------------------
# end-to-end ingest
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def ingested(tmp_path_factory, request) -> Project:
    """A project with all fixtures ingested exactly once (module-scoped)."""
    from fixtures.make_fixtures import build_all

    media = build_all()
    root = tmp_path_factory.mktemp("projects")
    project = Project.create("ingest-test", language="pl", title="Ingest", root=root)
    for name, fixture in INPUTS.items():
        shutil.copy(media[fixture], project.input_dir / name)
    # Proxies are opt-in since ingest learned to remux; this module's
    # assertions cover them, so the fixture asks for them explicitly.
    results = ingest(project, show_table=False, proxies=True)
    assert summarize(results).get("error", 0) == 0, [r.error for r in results]
    return project


def test_discover_inputs_finds_every_supported_file(ingested: Project) -> None:
    assert {p.name for p in discover_inputs(ingested)} == set(INPUTS)


def test_clips_are_numbered_in_recording_order(ingested: Project) -> None:
    clips = ingested.clips_in_order()
    assert [c["id"] for c in clips] == [f"c{i:03d}" for i in range(1, len(INPUTS) + 1)]
    assert [Path(c["source_file"]).name for c in clips] == EXPECTED_ORDER
    assert [c["order"] for c in clips] == list(range(1, len(INPUTS) + 1))
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
    assert state["stages"]["ingest"]["clips"] == len(INPUTS)
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

    hlg8 = by_file["g_hlg8.mp4"]
    assert hlg8["hdr"] == "hlg"
    assert (hlg8["width"], hlg8["height"]) == (1920, 1080)

    vfr = by_file["f_vfr.mp4"]
    assert vfr["vfr"] is True
    assert (vfr["width"], vfr["height"]) == (1920, 1080)
    assert vfr["target_fps"] == 60, "a VFR source normalizes to its nominal r_frame_rate"


def test_normalized_sources_are_uniform(ingested: Project) -> None:
    """The mezzanine invariant, remuxed clips included."""
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
    assert {c["mezzanine"] for c in ingested.clips_in_order()} == {"copy", "encode"}, (
        "the invariant is only interesting while both branches are represented"
    )


# ----------------------------------------------------------------------
# remux vs. re-encode
# ----------------------------------------------------------------------
def video_stream_md5(path: Path) -> str:
    """MD5 of a file's video stream, stream-copied out of its container."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
         "-map", "0:v:0", "-c", "copy", "-f", "md5", "-"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def video_bitrate(path: Path) -> int:
    """Bit rate of a file's first video stream, as ffprobe reports it."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=bit_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(proc.stdout.strip() or 0)


def test_every_clip_takes_the_branch_the_rule_predicts(ingested: Project) -> None:
    """One clip per branch of ``mezzanine_mode``, reached through a real ingest."""
    by_file = {Path(c["source_file"]).name: c for c in ingested.clips_in_order()}
    reached = {
        name: (str(clip["mezzanine"]), str(clip["mezzanine_reason"]).split(" ")[0])
        for name, clip in by_file.items()
    }
    assert reached == EXPECTED_MEZZANINE


def test_the_recorded_reason_is_what_the_rule_says_about_the_raw_input(
    ingested: Project,
) -> None:
    fmt = ingested.settings.format
    for clip in ingested.clips_in_order():
        decision = mezzanine_mode(probe(ingested.path / clip["source_file"]), fmt)
        assert (clip["mezzanine"], clip["mezzanine_reason"]) == (
            decision.mode,
            decision.reason,
        ), clip["id"]


@pytest.mark.parametrize(
    "name", [n for n, (mode, _) in EXPECTED_MEZZANINE.items() if mode == "copy"]
)
def test_a_copied_mezzanine_carries_the_sources_own_video_stream(
    ingested: Project, name: str
) -> None:
    """The whole point of the remux: same bytes, not a CRF 16 blow-up."""
    clip = next(c for c in ingested.clips_in_order() if Path(c["source_file"]).name == name)
    src = ingested.path / clip["source_file"]
    mezz = ingested.source_path(clip["id"])

    before, after = probe(src), probe(mezz)
    assert after.codec == before.codec
    assert (after.width, after.height) == (before.width, before.height)
    assert video_bitrate(mezz) == pytest.approx(video_bitrate(src), rel=0.02)
    assert video_stream_md5(mezz) == video_stream_md5(src), "the picture was re-encoded"
    # A re-encode of a 1.7 Mb/s 1080p clip measured 10x bigger; a remux does not.
    assert mezz.stat().st_size == pytest.approx(src.stat().st_size, rel=0.05)
    # ...and the mezzanine invariant still holds on the copy path.
    assert after.has_audio is True
    assert (after.audio_channels, after.audio_sample_rate) == (2, 48000)


def test_an_encoded_mezzanine_is_a_different_picture(ingested: Project) -> None:
    """The contrast case: an encoded clip does *not* keep the source stream."""
    clip = next(
        c for c in ingested.clips_in_order()
        if Path(c["source_file"]).name == "g_hlg8.mp4"
    )
    src = ingested.path / clip["source_file"]
    assert video_stream_md5(ingested.source_path(clip["id"])) != video_stream_md5(src)


def test_hlg_source_is_tonemapped_to_bt709(ingested: Project) -> None:
    hlg = [c for c in ingested.clips_in_order() if c["hdr"] == "hlg"]
    # The 10-bit fixture and the 8-bit one: the second is only encoded *because*
    # it is HDR, so it is the one that proves the tonemap is not a side effect
    # of the pix_fmt rule.
    assert len(hlg) == 2, [c["id"] for c in hlg]
    for clip in hlg:
        info = probe(ingested.source_path(clip["id"]))
        assert info.color_transfer == "bt709", clip["id"]
        assert info.color_primaries == "bt709", clip["id"]
        assert info.hdr is None, clip["id"]
        assert info.bit_depth == 8, clip["id"]


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
    results = ingest(ingested, show_table=False, proxies=True)
    assert summarize(results) == {"skipped": len(INPUTS)}
    assert [c["id"] for c in ingested.clips_in_order()] == [
        f"c{i:03d}" for i in range(1, len(INPUTS) + 1)
    ]


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


# ----------------------------------------------------------------------
# proxies are opt-in
# ----------------------------------------------------------------------
def two_clip_project(tmp_path: Path, media: dict[str, Path], slug: str) -> Project:
    """A project holding one clip ingest copies and one it re-encodes."""
    project = Project.create(slug, language="pl", root=tmp_path / "projects")
    shutil.copy(media["landscape.mp4"], project.input_dir / "a.mp4")   # 10:00 -> copy
    shutil.copy(media["silent.mp4"], project.input_dir / "b.mp4")      # 10:15 -> encode
    return project


def test_a_default_ingest_builds_no_proxies(tmp_path: Path, media: dict[str, Path]) -> None:
    project = two_clip_project(tmp_path, media, "no-proxy-test")
    assert summarize(ingest(project, show_table=False)) == {"done": 2}

    assert list(project.proxies_dir.glob("*.mp4")) == []
    for clip in project.clips_in_order():
        assert "proxy" not in clip, f"{clip['id']} recorded a proxy it never built"
        # The poster and the vision frames come off the mezzanine instead.
        assert project.poster_path(clip["id"]).exists()
        assert sorted(project.clip_frames_dir(clip["id"]).glob("*.jpg"))
        assert clip["frames_count"] > 0
    # ...and a poster sampled from the mezzanine is full size; one sampled from
    # a 720p proxy could not be.
    poster = probe(project.poster_path("c001"))
    assert (poster.width, poster.height) == (1920, 1080)


def test_ingest_builds_proxies_when_asked(tmp_path: Path, media: dict[str, Path]) -> None:
    project = two_clip_project(tmp_path, media, "proxy-on-test")
    ingest(project, show_table=False, proxies=True)
    for clip in project.clips_in_order():
        proxy = project.proxy_path(clip["id"])
        assert proxy.exists()
        assert clip["proxy"] == project.rel(proxy)


def test_ensure_proxies_fills_in_the_missing_ones(tmp_path: Path, media: dict[str, Path]) -> None:
    project = two_clip_project(tmp_path, media, "ensure-proxy-test")
    ingest(project, show_table=False)
    assert list(project.proxies_dir.glob("*.mp4")) == []

    assert ensure_proxies(project) == ["c001", "c002"]
    built = [project.proxy_path("c001"), project.proxy_path("c002")]
    assert all(p.exists() for p in built)
    assert [project.get_clip(cid)["proxy"] for cid in ("c001", "c002")] == [
        project.rel(p) for p in built
    ]

    before = [p.stat().st_mtime_ns for p in built]
    assert ensure_proxies(project) == [], "a second call must be a no-op"
    assert [p.stat().st_mtime_ns for p in built] == before


# ----------------------------------------------------------------------
# the ingest summary
# ----------------------------------------------------------------------
#: The seven fixtures: two copies and one clip per encode reason.
SUMMARY_REASONS: dict[str, int] = {"hdr": 1, "pix_fmt": 1, "rotation": 1, "size": 1, "vfr": 1}


def test_mezzanine_summary_counts_the_registry(ingested: Project) -> None:
    assert mezzanine_summary(ingested) == {
        "copy": 2, "encode": 5, "reasons": SUMMARY_REASONS,
    }


def test_mezzanine_line_groups_the_reasons(ingested: Project) -> None:
    assert mezzanine_line(mezzanine_summary(ingested)) == (
        "mezzanine: 2 copy, 5 encode (hdr 1, pix_fmt 1, rotation 1, size 1, vfr 1)"
    )


def test_mezzanine_line_drops_the_detail_when_nothing_was_encoded() -> None:
    assert mezzanine_line({"copy": 4, "encode": 0, "reasons": {}}) == (
        "mezzanine: 4 copy, 0 encode"
    )


def test_the_ingest_stage_records_the_mezzanine_counts(ingested: Project) -> None:
    stage = ingested.load_state()["stages"]["ingest"]
    assert stage["mezzanine_copy"] == 2
    assert stage["mezzanine_encode"] == 5
    assert stage["encode_reasons"] == SUMMARY_REASONS
    assert stage["mezzanine_copy"] + stage["mezzanine_encode"] == stage["clips"]


def test_a_project_of_matching_clips_encodes_nothing(
    tmp_path: Path, media: dict[str, Path]
) -> None:
    project = Project.create("all-copy-test", language="pl", root=tmp_path / "projects")
    shutil.copy(media["landscape.mp4"], project.input_dir / "a.mp4")
    shutil.copy(media["vertical.mp4"], project.input_dir / "b.mp4")
    ingest(project, show_table=False)

    stage = project.load_state()["stages"]["ingest"]
    assert (stage["mezzanine_copy"], stage["mezzanine_encode"]) == (2, 0)
    assert stage["encode_reasons"] == {}
    assert mezzanine_line(mezzanine_summary(project)) == "mezzanine: 2 copy, 0 encode"
