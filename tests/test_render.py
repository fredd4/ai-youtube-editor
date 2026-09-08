"""Tests for ``ytedit.media.render``: geometry, caching and a real render.

The end-to-end test builds a throwaway project from the generated fixtures and
renders a three-segment timeline (a horizontal clip, a vertical clip fitted
with ``blur-fill`` behind an ``xfade``, and a silent clip), with one Polish
location card, one auto-ducked music cue and one mute range. Nothing here
touches a network API.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fixtures.make_fixtures import build_all
from ytedit.config import load_settings
from ytedit.media import audio as A
from ytedit.media import render as R
from ytedit.media.ingest import ingest
from ytedit.project import Project
from ytedit.timeline import AudioFrom, Timeline, VoiceItem

#: Programme length of :func:`timeline_document` (3.0 + 3.0 − 0.5 xfade + 2.0).
EXPECTED_DURATION: float = 7.5

LOCATION_TEXT = "LIZBONA, PORTUGALIA — ŁÓDŹ"


# ----------------------------------------------------------------------
# the test project
# ----------------------------------------------------------------------
def timeline_document() -> dict:
    """The three-segment timeline used by the render and QC tests."""
    return {
        "version": 1, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
        "tracks": {
            "video": [
                {"id": "s001", "clip": "c001", "in": 0.5, "out": 3.5, "role": "cold-open",
                 "transform": {"fit": "cover", "zoom": 1.0}, "grade": "default",
                 "transition_in": {"type": "cut", "duration": 0.0}},
                {"id": "s002", "clip": "c002", "in": 1.0, "out": 4.0, "role": "b-roll",
                 "transform": {"fit": "blur-fill", "zoom": 1.0}, "grade": "default",
                 "transition_in": {"type": "xfade", "duration": 0.5, "name": "fade"}},
                {"id": "s003", "clip": "c003", "in": 0.0, "out": 2.0, "role": "b-roll",
                 "transform": {"fit": "cover", "zoom": 1.0}, "grade": "default",
                 "transition_in": {"type": "cut", "duration": 0.0}},
            ],
            "voice": [],
            "music": [{"id": "m001", "file": "music/bed.wav", "at": 0.0, "end": 7.5,
                       "gain_db": -18, "fade_in": 1.0, "fade_out": 1.0,
                       "duck": {"mode": "auto", "amount_db": -12,
                                "attack": 0.15, "release": 0.6}}],
            "captions": [{"id": "t001", "at": 0.5, "end": 3.0, "text": LOCATION_TEXT,
                          "style": "location", "position": "lower-left"}],
            "sfx": [],
        },
        "mute_ranges": [{"clip": "c001", "s": 1.0, "e": 2.0, "gain_db": -60,
                         "reason": "copyrighted bar music"}],
        "markers": [{"at": 0.0, "label": "hook"}],
        "chapters": [],
        "meta": {"generated_by": "test", "edited_by_human": False},
    }


def build_render_project(
    root: Path, slug: str = "render-test", document: dict | None = None
) -> Project:
    """Ingest three fixtures and write a timeline into a project.

    Args:
        root: Parent directory for the project (a pytest ``tmp_path``).
        slug: Project slug.
        document: Timeline JSON to write (default :func:`timeline_document`).

    Returns:
        A project ready for :func:`ytedit.media.render.render`.
    """
    media = build_all()
    project = Project.create(slug, language="pl", title="Render", root=root)
    for name, fixture in {
        "a_landscape.mp4": "landscape.mp4",   # 1920x1080, 6 s, 440 Hz
        "b_vertical.mp4": "vertical.mp4",     # 1080x1920, 6 s, 660 Hz
        "c_silent.mp4": "silent.mp4",         # 1280x720, 4 s, no audio stream
    }.items():
        shutil.copy(media[fixture], project.input_dir / name)
    results = ingest(project, show_table=False)
    assert not [r for r in results if r.error], [r.error for r in results]

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=20",
         "-c:a", "pcm_s16le", "-ac", "2", str(project.music_dir / "bed.wav")],
        check=True,
    )
    (project.transcripts_dir / "c001.json").write_text(
        json.dumps({
            "clip": "c001", "language": "pl", "project_language": "pl",
            "text": "Dzień dobry z Lizbony.",
            "words": [
                {"t": "Dzień", "s": 2.0, "e": 2.3},
                {"t": "dobry", "s": 2.35, "e": 2.7},
                {"t": "z", "s": 2.75, "e": 2.85},
                {"t": "Lizbony.", "s": 2.9, "e": 3.4},
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    project.timeline_file.write_text(
        json.dumps(document or timeline_document(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return project


# ----------------------------------------------------------------------
# the frame-exactness project
# ----------------------------------------------------------------------
#: Whole frames each :func:`fractional_document` segment renders to at 30 fps:
#: 1.62 s -> 48.6 -> 49, 1.64 s -> 49.2 -> 49, 1.17 s -> 35.1 -> 35, 0.995 s -> 29.85 -> 30.
FRACTIONAL_FRAMES: list[int] = [49, 49, 35, 30]

#: Timeline time of the third segment's first frame: (49 + 49) / 30. The raw
#: in/out arithmetic says 3.26 s, which is *not* a frame boundary.
THIRD_SEGMENT_AT: float = 98 / 30


def fractional_document() -> dict:
    """Four hard-cut segments with in/out points that are not on frame boundaries.

    Every source is muted so the only audio in the programme is the voice pickup
    placed at the third segment's first frame; a location card starts on the
    same frame.
    """
    at = round(THIRD_SEGMENT_AT, 6)
    return {
        "version": 1, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
        "tracks": {
            "video": [
                {"id": "s001", "clip": "c001", "in": 0.51, "out": 2.13, "role": "cold-open",
                 "mute_source": True, "transition_in": {"type": "cut", "duration": 0.0}},
                {"id": "s002", "clip": "c002", "in": 1.07, "out": 2.71, "role": "b-roll",
                 "mute_source": True, "transform": {"fit": "blur-fill", "zoom": 1.0},
                 "transition_in": {"type": "cut", "duration": 0.0}},
                {"id": "s003", "clip": "c003", "in": 0.33, "out": 1.5, "role": "b-roll",
                 "mute_source": True, "transition_in": {"type": "cut", "duration": 0.0}},
                {"id": "s004", "clip": "c001", "in": 2.005, "out": 3.0, "role": "outro",
                 "mute_source": True, "transition_in": {"type": "cut", "duration": 0.0}},
            ],
            "voice": [{"id": "v001", "file": "voice/v001.wav", "at": at, "gain_db": 0.0}],
            "music": [],
            "captions": [{"id": "t001", "at": at, "end": round(at + 0.5, 6),
                          "text": "KLATKA 98", "style": "location",
                          "position": "lower-left"}],
            "sfx": [],
        },
        "mute_ranges": [],
        "markers": [{"at": 0.0, "label": "hook"}],
        "chapters": [],
        "meta": {"generated_by": "test", "edited_by_human": False},
    }


@pytest.fixture(scope="module")
def fractional(tmp_path_factory) -> tuple[Project, Path]:
    """A project built from :func:`fractional_document`, preview rendered once."""
    project = build_render_project(
        tmp_path_factory.mktemp("frames"), slug="frame-test", document=fractional_document()
    )
    project.voice_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=0.6",
         "-c:a", "pcm_s16le", "-ac", "2", str(project.voice_dir / "v001.wav")],
        check=True,
    )
    return project, R.render(project, preview=True)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> tuple[Project, Path]:
    """A project with the preview rendered exactly once."""
    project = build_render_project(tmp_path_factory.mktemp("render"))
    return project, R.render(project, preview=True)


def measure_volume(path: Path, start: float, length: float) -> float:
    """Return the mean volume in dBFS of a window of an audio file."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-ss", f"{start}", "-t", f"{length}",
         "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", proc.stderr)
    assert match, proc.stderr[-1500:]
    return float(match.group(1))


def video_stream(path: Path) -> dict:
    """ffprobe the first video stream of a file."""
    return next(s for s in probe(path)["streams"] if s["codec_type"] == "video")


def keyframe_times(path: Path) -> list[float]:
    """Presentation times of every keyframe packet in a video file."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    times: list[float] = []
    for line in proc.stdout.splitlines():
        pts, _, flags = line.partition(",")
        if "K" in flags:
            times.append(float(pts))
    return times


def probe(path: Path) -> dict:
    """ffprobe a file as JSON."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


# ----------------------------------------------------------------------
# geometry and time mapping (no ffmpeg)
# ----------------------------------------------------------------------
def test_canvas_for_preview_letterboxes_to_720p() -> None:
    timeline = Timeline.model_validate({"width": 1920, "height": 1080, "fps": 30})
    assert R.canvas_for(timeline, preview=True) == R.Canvas(1280, 720, 30)
    assert R.canvas_for(timeline, preview=False) == R.Canvas(1920, 1080, 30)


def test_canvas_for_never_upscales_a_small_timeline() -> None:
    timeline = Timeline.model_validate({"width": 640, "height": 360, "fps": 24})
    assert R.canvas_for(timeline, preview=True) == R.Canvas(640, 360, 24)


def test_render_positions_overlap_both_fade_and_xfade() -> None:
    timeline = Timeline.model_validate({"tracks": {"video": [
        {"id": "a", "clip": "c1", "in": 0, "out": 3},
        {"id": "b", "clip": "c2", "in": 0, "out": 3,
         "transition_in": {"type": "fade", "duration": 0.5}},
    ]}})
    # the timeline model treats "fade" as non-overlapping ...
    assert timeline.duration() == pytest.approx(6.0)
    # ... but ffmpeg renders it with xfade, which does overlap
    assert R.render_duration(timeline) == pytest.approx(5.5)
    assert [p.start for p in R.render_positions(timeline)] == [0.0, 2.5]


def test_build_time_map_is_the_identity_for_cuts_and_xfades() -> None:
    timeline = Timeline.model_validate({"tracks": {"video": [
        {"id": "a", "clip": "c1", "in": 0, "out": 3},
        {"id": "b", "clip": "c2", "in": 0, "out": 3,
         "transition_in": {"type": "xfade", "duration": 0.5}},
    ]}})
    to_render = R.build_time_map(timeline)
    for t in (0.0, 1.0, 2.5, 5.5):
        assert to_render(t) == pytest.approx(t)


def test_build_time_map_shifts_everything_after_a_fade() -> None:
    timeline = Timeline.model_validate({"tracks": {"video": [
        {"id": "a", "clip": "c1", "in": 0, "out": 3},
        {"id": "b", "clip": "c2", "in": 0, "out": 3,
         "transition_in": {"type": "fade", "duration": 0.5}},
    ]}})
    to_render = R.build_time_map(timeline)
    assert to_render(1.0) == pytest.approx(1.0)      # inside the first segment
    assert to_render(3.0) == pytest.approx(2.5)      # start of the second
    assert to_render(6.0) == pytest.approx(5.5)      # the tail


def test_segment_mute_ranges_are_clip_time_shifted_into_segment_time() -> None:
    timeline = Timeline.model_validate({
        "tracks": {"video": [{"id": "s1", "clip": "c001", "in": 2.0, "out": 8.0}]},
        "mute_ranges": [
            {"clip": "c001", "s": 3.0, "e": 5.0, "gain_db": -60},
            {"clip": "c001", "s": 0.0, "e": 1.0, "gain_db": -60},   # before the cut
            {"clip": "c002", "s": 3.0, "e": 5.0, "gain_db": -60},   # another clip
        ],
    })
    seg = timeline.tracks.video[0]
    assert R.segment_mute_ranges(timeline, seg) == [(1.0, 3.0, -60.0)]


def test_segment_mute_ranges_are_divided_by_the_speed() -> None:
    timeline = Timeline.model_validate({
        "tracks": {"video": [{"id": "s1", "clip": "c1", "in": 0.0, "out": 8.0, "speed": 2.0}]},
        "mute_ranges": [{"clip": "c1", "s": 2.0, "e": 4.0, "gain_db": -12}],
    })
    assert R.segment_mute_ranges(timeline, timeline.tracks.video[0]) == [(1.0, 2.0, -12.0)]


def test_escape_filter_path_escapes_the_ffmpeg_specials() -> None:
    assert R.escape_filter_path("/a b/c:d,e.ass") == "/a b/c\\:d\\,e.ass"
    assert R.escape_filter_path("/x[1]/y.ass") == "/x\\[1\\]/y.ass"


def test_atempo_chains_for_extreme_speeds() -> None:
    assert R._atempo(1.0) == ""
    assert R._atempo(2.0) == "atempo=2.000000"
    assert R._atempo(0.25) == "atempo=0.5,atempo=0.500000"


def test_parse_bitrate_understands_ffmpeg_suffixes() -> None:
    assert R.parse_bitrate("384k") == 384_000
    assert R.parse_bitrate("1.5M") == 1_500_000
    assert R.parse_bitrate(256000) == 256_000
    assert R.parse_bitrate("nonsense", default=7) == 7


def test_audio_encoder_args_target_48k_stereo(project: Project) -> None:
    args = R.audio_encoder_args(project.settings)
    assert args[args.index("-ar") + 1] == "48000"
    assert args[args.index("-ac") + 1] == "2"
    assert args[args.index("-b:a") + 1] == "384k"
    # ffmpeg's native aac cannot reach 384k, so aac_at is preferred when present
    assert args[args.index("-c:a") + 1] in ("aac", "aac_at")


def test_color_tag_filter_stamps_bt709(project: Project) -> None:
    chain = R.color_tag_filter(project.settings)
    assert chain.startswith("setparams=")
    for key in ("color_primaries=bt709", "color_trc=bt709", "colorspace=bt709"):
        assert key in chain


def test_render_refuses_a_project_without_a_timeline(project: Project) -> None:
    with pytest.raises(R.RenderError, match="no timeline"):
        R.render(project, preview=True)


def test_render_refuses_an_invalid_timeline(project: Project) -> None:
    project.timeline_file.write_text(json.dumps({
        "tracks": {"video": [{"id": "s1", "clip": "c1", "in": 5.0, "out": 1.0}]},
    }), encoding="utf-8")
    with pytest.raises(R.RenderError, match="issue"):
        R.render(project, preview=True)


# ----------------------------------------------------------------------
# the real render
# ----------------------------------------------------------------------
def test_preview_lands_where_it_should(rendered: tuple[Project, Path]) -> None:
    project, out = rendered
    assert out == project.renders_dir / "preview.mp4"
    assert out.exists() and out.stat().st_size > 10_000


def test_preview_has_the_expected_duration_and_canvas(rendered: tuple[Project, Path]) -> None:
    _project, out = rendered
    data = probe(out)
    assert float(data["format"]["duration"]) == pytest.approx(EXPECTED_DURATION, abs=0.1)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1280, 720)
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == 48000
    assert audio["channels"] == 2


def test_preview_is_colour_tagged_bt709(rendered: tuple[Project, Path]) -> None:
    _project, out = rendered
    video = next(s for s in probe(out)["streams"] if s["codec_type"] == "video")
    assert video.get("color_primaries") == "bt709"
    assert video.get("color_transfer") == "bt709"
    assert video.get("color_space") == "bt709"


def test_ass_document_is_written_with_the_caption_text(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    ass = project.renders_dir / "captions.ass"
    assert ass.exists()
    text = ass.read_text(encoding="utf-8")
    assert LOCATION_TEXT in text
    assert "PlayResY: 720" in text          # scaled to the preview canvas
    assert "\\fad(" in text


def test_srt_is_written_for_upload_and_not_burned(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    srt = project.exports_dir / "captions.srt"
    assert srt.exists()
    assert "Dzień dobry" in srt.read_text(encoding="utf-8")


def test_duck_automation_file_has_ramps(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    cmd = project.renders_dir / "duck.cmd"
    assert cmd.exists()
    commands = re.findall(r"^(\d+\.\d{3}) volume volume (\d+\.\d+);$",
                          cmd.read_text(encoding="utf-8"), re.M)
    assert len(commands) >= 6, "expected a ramped envelope, not a step"
    gains = [float(g) for _, g in commands]
    assert min(gains) < max(gains) * 0.5, "the music must actually duck"


def test_mute_range_silences_the_source_bus(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    bus = project.renders_dir / "program_audio.wav"
    assert bus.exists()
    # clip time 1.0-2.0 with the segment starting at in=0.5 -> timeline 0.5-1.5
    inside = measure_volume(bus, 0.6, 0.8)
    outside = measure_volume(bus, 2.0, 0.9)
    assert inside < -60.0, f"the muted window is not silent ({inside} dB)"
    assert outside > inside + 30.0, "audio outside the mute range went missing"


def test_preview_loudness_is_close_to_the_target(rendered: tuple[Project, Path]) -> None:
    from ytedit.media.audio import measure_loudness

    _project, out = rendered
    measured = measure_loudness(out)
    assert measured["input_i"] == pytest.approx(-14.0, abs=1.5)


def test_state_and_job_record_the_render(rendered: tuple[Project, Path]) -> None:
    project, out = rendered
    stage = project.load_state()["stages"]["render"]
    assert stage["status"] == "done"
    assert stage["preset"] == "preview"
    assert stage["output"] == project.rel(out)
    job = json.loads((project.jobs_dir / "render_preview.json").read_text())
    assert job["status"] == "done"
    assert job["percent"] == 100.0
    assert job["output"] == str(out)


def test_segments_are_cached_between_renders(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    timeline = Timeline.load(project.timeline_file)
    canvas = R.canvas_for(timeline, preview=True)
    cached = [
        project.renders_dir / "segments"
        / f"{R.segment_key(project, timeline, seg, canvas, 'preview')}.mp4"
        for seg in timeline.tracks.video
    ]
    assert all(p.exists() for p in cached), cached
    before = [p.stat().st_mtime_ns for p in cached]
    for seg in timeline.tracks.video:
        R.render_segment(project, timeline, seg, canvas, "preview")
    assert [p.stat().st_mtime_ns for p in cached] == before


def test_changing_a_segment_changes_its_cache_key(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    timeline = Timeline.load(project.timeline_file)
    canvas = R.canvas_for(timeline, preview=True)
    seg = timeline.tracks.video[0]
    first = R.segment_key(project, timeline, seg, canvas, "preview")
    moved = seg.model_copy(update={"out": seg.out + 0.5})
    assert R.segment_key(project, timeline, moved, canvas, "preview") != first
    assert R.segment_key(project, timeline, seg, canvas, "master") != first


def test_silent_clip_still_produces_an_audio_stream(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    timeline = Timeline.load(project.timeline_file)
    canvas = R.canvas_for(timeline, preview=True)
    seg = timeline.tracks.video[2]           # c003, the fixture with no audio stream
    path = R.render_segment(project, timeline, seg, canvas, "preview")
    streams = probe(path)["streams"]
    assert any(s["codec_type"] == "audio" for s in streams), "anullsrc fallback missing"
    assert float(probe(path)["format"]["duration"]) == pytest.approx(2.0, abs=0.1)


# ----------------------------------------------------------------------
# denoised source audio
# ----------------------------------------------------------------------
def _denoised_project(project: Project) -> tuple[Timeline, Path]:
    """A one-segment timeline whose clip has an active denoised WAV."""
    project.source_path("c001").parent.mkdir(parents=True, exist_ok=True)
    project.source_path("c001").write_bytes(b"not really an mp4")
    cleaned = project.audio_dir / "c001.denoised.wav"
    cleaned.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_bytes(b"not really a wav")
    project.add_clip({
        "id": "c001", "order": 1, "duration": 10.0, "width": 1920, "height": 1080,
        "has_audio": True, "denoised": "media/audio/c001.denoised.wav",
        "denoise_engine": "local", "use_denoised": True,
    })
    timeline = Timeline.model_validate(
        {"tracks": {"video": [{"id": "s001", "clip": "c001", "in": 2.0, "out": 5.0}]}}
    )
    return timeline, cleaned


def test_denoised_audio_is_only_used_when_enabled_and_present(project: Project) -> None:
    _timeline, cleaned = _denoised_project(project)
    assert R.denoised_audio(project, "c001") == cleaned

    project.set_clip_stage("c001", "denoise", "done", use_denoised=False)
    assert R.denoised_audio(project, "c001") is None

    project.set_clip_stage("c001", "denoise", "done", use_denoised=True)
    cleaned.unlink()
    assert R.denoised_audio(project, "c001") is None


def test_render_segment_feeds_ffmpeg_the_denoised_wav(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline, cleaned = _denoised_project(project)
    seg = timeline.tracks.video[0]
    canvas = R.Canvas(1920, 1080, 30)
    captured: list[list[str]] = []

    def fake_ff(*args, **kwargs) -> str:
        captured.append([str(a) for a in args])
        Path(str(args[-1])).write_bytes(b"segment")
        return ""

    monkeypatch.setattr(R, "ff", fake_ff)
    monkeypatch.setattr(R, "_segment_is_valid", lambda *a, **k: True)

    out = R.render_segment(project, timeline, seg, canvas, "preview")
    assert out.exists()

    args = captured[0]
    assert str(cleaned) in args, args
    # The denoised file is the second input, cut with the same -ss as the video.
    assert args.index(str(cleaned)) == args.index(str(project.source_path("c001"))) + 4
    assert args.count("-ss") == 2 and args.count("2.000000") == 2
    # ...and it is the audio the filter graph maps.
    graph = args[args.index("-filter_complex") + 1]
    assert "[1:a]" in graph


def test_re_denoising_a_clip_invalidates_the_segment_cache(project: Project) -> None:
    timeline, cleaned = _denoised_project(project)
    seg = timeline.tracks.video[0]
    canvas = R.Canvas(1920, 1080, 30)
    first = R.segment_key(project, timeline, seg, canvas, "preview")

    cleaned.write_bytes(b"a different denoised wav")
    assert R.segment_key(project, timeline, seg, canvas, "preview") != first

    project.set_clip_stage("c001", "denoise", "done", use_denoised=False)
    assert R.segment_key(project, timeline, seg, canvas, "preview") != first



# ----------------------------------------------------------------------
# frame-exact cutting and joining
# ----------------------------------------------------------------------
def test_fractional_segments_are_snapped_to_whole_frames() -> None:
    timeline = Timeline.model_validate(fractional_document())
    assert [seg.frames(30) for seg in timeline.tracks.video] == FRACTIONAL_FRAMES
    assert timeline.duration() == pytest.approx(sum(FRACTIONAL_FRAMES) / 30, abs=1e-6)
    assert R.render_duration(timeline) == timeline.duration()
    # the raw arithmetic (5.425 s) is not what gets rendered
    assert sum(seg.duration for seg in timeline.tracks.video) == pytest.approx(5.425)


def test_joined_programme_is_the_sum_of_frame_rounded_segments(
    fractional: tuple[Project, Path],
) -> None:
    """Four fractional cuts join to exactly 49 + 49 + 35 + 30 frames."""
    project, _out = fractional
    expected = sum(FRACTIONAL_FRAMES) / 30
    joined = project.renders_dir / "program_video.mp4"
    video = video_stream(joined)
    assert int(video["nb_frames"]) == sum(FRACTIONAL_FRAMES)
    assert float(video["duration"]) == pytest.approx(expected, abs=0.001)
    assert float(probe(joined)["format"]["duration"]) == pytest.approx(expected, abs=0.001)
    # every cached segment is exactly as long as the timeline says
    for seg, frames in zip(Timeline.load(project.timeline_file).tracks.video, FRACTIONAL_FRAMES):
        assert seg.rendered_duration(30) == pytest.approx(frames / 30, abs=1e-6)
    # ... and each one starts on the frame the positions predict (a hard-cut
    # concat keeps every segment's first frame as a keyframe)
    starts = [round(p.start, 6) for p in Timeline.load(project.timeline_file).segment_positions()]
    keyframes = keyframe_times(joined)
    for start in starts:
        assert any(abs(k - start) < 0.5 / 30 for k in keyframes), (start, keyframes)


def test_caption_and_voice_land_on_the_expected_frame(
    fractional: tuple[Project, Path],
) -> None:
    """Items placed at the third segment's start hit frame 98, not the raw 3.26 s."""
    from ytedit.media.captions import ass_time

    project, out = fractional
    frame_time = THIRD_SEGMENT_AT
    frame = 1 / 30

    # the location card is written at the frame time (centisecond ASS clock)
    ass = (project.renders_dir / "captions.ass").read_text(encoding="utf-8")
    dialogue = next(line for line in ass.splitlines() if line.startswith("Dialogue:"))
    assert ass_time(frame_time) == "0:00:03.27"
    assert dialogue.split(",")[1] == ass_time(frame_time)
    assert dialogue.split(",")[1] != ass_time(3.26)

    # the joined video really does switch to the third segment on that frame
    assert any(abs(k - frame_time) < 0.5 * frame for k in keyframe_times(
        project.renders_dir / "program_video.mp4"
    ))

    # every source is muted, so the voice pickup is the only sound: the frame
    # before its position is digital silence, the frame at its position is not
    final_audio = project.renders_dir / "final_audio.wav"
    assert measure_volume(final_audio, frame_time - frame, frame) < -80
    assert measure_volume(final_audio, frame_time, frame) > -40

    # and the finished preview carries exactly the timeline's frames
    assert int(video_stream(out)["nb_frames"]) == sum(FRACTIONAL_FRAMES)


def test_xfade_join_is_frame_exact(
    fractional: tuple[Project, Path], tmp_path: Path, monkeypatch
) -> None:
    """The filter-graph join (xfade + concat) also lands on the frame count."""
    project, _out = fractional
    doc = fractional_document()
    doc["tracks"]["video"] = doc["tracks"]["video"][:3]
    # 0.37 s is 11.1 frames -> an 11-frame overlap
    doc["tracks"]["video"][1]["transition_in"] = {"type": "xfade", "duration": 0.37, "name": "fade"}
    doc["tracks"]["voice"] = []
    doc["tracks"]["captions"] = []
    timeline = Timeline.model_validate(doc)
    expected_frames = 49 + 49 - 11 + 35
    assert timeline.frame_count() == expected_frames
    assert R.render_duration(timeline) == pytest.approx(expected_frames / 30, abs=1e-6)

    # keep the module fixture's renders untouched
    monkeypatch.setattr(Project, "renders_dir", property(lambda self: tmp_path / "renders"))
    canvas = R.canvas_for(timeline, preview=True)
    segments = [
        R.render_segment(project, timeline, seg, canvas, "preview")
        for seg in timeline.tracks.video
    ]
    video_out, audio_out = R.join_segments(project, timeline, segments, canvas, "preview")
    video = video_stream(video_out)
    assert int(video["nb_frames"]) == expected_frames
    assert float(video["duration"]) == pytest.approx(expected_frames / 30, abs=0.001)
    assert float(probe(audio_out)["format"]["duration"]) == pytest.approx(
        expected_frames / 30, abs=0.001
    )


# ----------------------------------------------------------------------
# overlay cutaways: picture from one clip, audio from another
# ----------------------------------------------------------------------
def measure_band_volume(path: Path, freq: int, start: float = 0.2,
                        length: float = 1.0, width: int = 40, passes: int = 3) -> float:
    """Mean volume in dBFS of a narrow band around ``freq``.

    The fixtures carry pure sines (440 Hz landscape, 660 Hz vertical), so a
    bandpass plus ``volumedetect`` says which clip's audio ended up in a file.
    One biquad leaks a good 15 dB of a neighbouring tone, so the filter is
    chained ``passes`` times to make the two fixtures unmistakable.
    """
    band = ",".join([f"bandpass=f={freq}:width_type=h:w={width}"] * passes)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-ss", f"{start}", "-t", f"{length}",
         "-i", str(path),
         "-af", f"{band},volumedetect",
         "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", proc.stderr)
    assert match, proc.stderr[-1500:]
    return float(match.group(1))


def test_audio_from_renders_the_other_clips_tone_at_the_pictures_length(
    rendered: tuple[Project, Path],
) -> None:
    """Picture c001 (440 Hz), audio borrowed from c002 (660 Hz)."""
    project, _out = rendered
    timeline = Timeline.model_validate({
        "fps": 30, "width": 1920, "height": 1080,
        "tracks": {"video": [
            {"id": "s001", "clip": "c001", "in": 0.5, "out": 2.0, "role": "cutaway",
             "audio_from": {"clip": "c002", "in": 1.0, "out": 2.5}},
        ]},
    })
    seg = timeline.tracks.video[0]
    canvas = R.canvas_for(timeline, preview=True)
    path = R.render_segment(project, timeline, seg, canvas, "preview")

    # exactly as many frames as the picture asks for
    frames = seg.frames(canvas.fps)
    assert frames == 45
    assert int(video_stream(path)["nb_frames"]) == frames
    assert float(probe(path)["format"]["duration"]) == pytest.approx(
        frames / canvas.fps, abs=1.0 / canvas.fps
    )

    # ... and the sound is the *audio* clip's tone, not the picture clip's
    at_660 = measure_band_volume(path, 660)
    at_440 = measure_band_volume(path, 440)
    assert at_660 > at_440 + 30, (at_660, at_440)


def test_audio_from_takes_part_in_the_segment_cache_key(rendered: tuple[Project, Path]) -> None:
    project, _out = rendered
    timeline = Timeline.model_validate({
        "tracks": {"video": [{"id": "s001", "clip": "c001", "in": 0.5, "out": 2.0}]},
    })
    seg = timeline.tracks.video[0]
    canvas = R.Canvas(1280, 720, 30)
    plain = R.segment_key(project, timeline, seg, canvas, "preview")

    borrowed = seg.model_copy(update={
        "audio_from": AudioFrom(clip="c002", **{"in": 1.0}, out=2.5)
    })
    key = R.segment_key(project, timeline, borrowed, canvas, "preview")
    assert key != plain

    moved = borrowed.model_copy(update={
        "audio_from": AudioFrom(clip="c002", **{"in": 1.5}, out=3.0)
    })
    assert R.segment_key(project, timeline, moved, canvas, "preview") != key


def test_audio_from_mute_ranges_are_read_in_the_audio_clips_time_base() -> None:
    timeline = Timeline.model_validate({
        "tracks": {"video": [
            {"id": "s1", "clip": "c033", "in": 0.0, "out": 3.0,
             "audio_from": {"clip": "c030", "in": 10.0, "out": 13.0}},
        ]},
        "mute_ranges": [
            {"clip": "c030", "s": 11.0, "e": 12.0, "gain_db": -60},   # the audio clip
            {"clip": "c033", "s": 0.0, "e": 1.0, "gain_db": -60},     # the picture clip
        ],
    })
    assert R.segment_mute_ranges(timeline, timeline.tracks.video[0]) == [(1.0, 2.0, -60.0)]


def test_a_missing_audio_from_source_is_a_render_error(project: Project) -> None:
    project.source_path("c001").parent.mkdir(parents=True, exist_ok=True)
    project.source_path("c001").write_bytes(b"not really an mp4")
    project.add_clip({"id": "c001", "order": 1, "duration": 10.0,
                      "width": 1920, "height": 1080, "has_audio": True})
    timeline = Timeline.model_validate({"tracks": {"video": [
        {"id": "s001", "clip": "c001", "in": 0.0, "out": 2.0,
         "audio_from": {"clip": "c999", "in": 0.0, "out": 2.0}},
    ]}})
    with pytest.raises(R.RenderError, match="audio_from clip 'c999'"):
        R.render_segment(project, timeline, timeline.tracks.video[0], R.Canvas(1920, 1080, 30),
                         "preview")


# ----------------------------------------------------------------------
# speech leveling (segments and voice pickups) before the mix
# ----------------------------------------------------------------------
def test_speech_leveling_evens_out_differing_segment_gains(tmp_path: Path) -> None:
    """Three cuts of the same clip at simulated mic levels 12 dB apart

    (source_audio_gain_db 0/12/6) must render to within 1.5 LU of each other
    once leveled, and each must leave behind a gain sidecar.
    """
    project = build_render_project(tmp_path, slug="level-test")
    # Custom transcript spanning the whole clip so all three segments below
    # (0-1.5, 1.5-3.0, 3.0-4.5) carry enough words to trigger leveling.
    (project.transcripts_dir / "c001.json").write_text(
        json.dumps({
            "clip": "c001", "language": "pl", "project_language": "pl",
            "text": "raz dwa trzy",
            "words": [
                {"t": "raz", "s": 0.1, "e": 1.3},
                {"t": "dwa", "s": 1.6, "e": 2.9},
                {"t": "trzy", "s": 3.1, "e": 4.4},
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    timeline = Timeline.model_validate({
        "fps": 30, "width": 1920, "height": 1080,
        "tracks": {"video": [
            {"id": "s001", "clip": "c001", "in": 0.0, "out": 1.5,
             "source_audio_gain_db": 0.0},
            {"id": "s002", "clip": "c001", "in": 1.5, "out": 3.0,
             "source_audio_gain_db": 12.0},
            {"id": "s003", "clip": "c001", "in": 3.0, "out": 4.5,
             "source_audio_gain_db": 6.0},
        ]},
    })
    canvas = R.canvas_for(timeline, preview=True)
    measured: list[float] = []
    for seg in timeline.tracks.video:
        path = R.render_segment(project, timeline, seg, canvas, "preview")
        measured.append(A.measure_loudness(path)["input_i"])
        key = R.segment_key(project, timeline, seg, canvas, "preview")
        sidecar = project.renders_dir / "segments" / f"{key}.gain.json"
        assert sidecar.exists(), f"{seg.id}: no speech-gain sidecar was written"

    assert max(measured) - min(measured) <= 1.5, measured


def test_speech_leveling_skips_muted_and_untranscribed_segments(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="level-skip-test")
    timeline = Timeline.model_validate({
        "fps": 30, "width": 1920, "height": 1080,
        "tracks": {"video": [
            {"id": "s001", "clip": "c001", "in": 0.0, "out": 1.5, "mute_source": True},
            {"id": "s002", "clip": "c002", "in": 0.0, "out": 1.5},  # c002 has no transcript
        ]},
    })
    canvas = R.canvas_for(timeline, preview=True)
    for seg in timeline.tracks.video:
        R.render_segment(project, timeline, seg, canvas, "preview")
        key = R.segment_key(project, timeline, seg, canvas, "preview")
        sidecar = project.renders_dir / "segments" / f"{key}.gain.json"
        assert not sidecar.exists(), f"{seg.id} should not have been leveled"


def test_speech_target_lufs_change_invalidates_the_segment_cache(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="target-key-test")
    timeline = Timeline.load(project.timeline_file)
    canvas = R.canvas_for(timeline, preview=True)
    seg = timeline.tracks.video[0]
    before = R.segment_key(project, timeline, seg, canvas, "preview")

    project._settings = load_settings(
        project.path, overrides={"audio": {"speech_target_lufs": -20.0}}
    )
    assert R.segment_key(project, timeline, seg, canvas, "preview") != before

    project._settings = load_settings(
        project.path, overrides={"audio": {"speech_gain_max_db": 3.0}}
    )
    assert R.segment_key(project, timeline, seg, canvas, "preview") != before


def test_voice_pickup_gain_includes_speech_leveling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_audio_bus`` must gain a quiet voice pickup toward the speech

    target, on top of the item's own ``gain_db``, before handing it to the mix.
    """
    project = build_render_project(tmp_path, slug="voice-level-test")
    project.voice_dir.mkdir(parents=True, exist_ok=True)
    quiet_voice = project.voice_dir / "v001.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=3",
         "-af", "volume=0.05", "-c:a", "pcm_s16le", "-ac", "2", str(quiet_voice)],
        check=True,
    )
    expected_gain, _measured = A.level_voice_file_gain(
        quiet_voice, target_lufs=-16.0, max_gain_db=10.0
    )
    assert expected_gain > 0, "the fixture pickup should measure quiet"

    timeline = Timeline.model_validate({
        "fps": 30, "width": 1920, "height": 1080,
        "tracks": {
            "video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 3.0,
                       "mute_source": True}],
            "voice": [{"id": "v001", "file": "voice/v001.wav", "at": 0.0, "gain_db": 1.0}],
        },
    })
    canvas = R.canvas_for(timeline, preview=True)
    seg = timeline.tracks.video[0]
    segment_path = R.render_segment(project, timeline, seg, canvas, "preview")
    _video_out, program_audio = R.join_segments(
        project, timeline, [segment_path], canvas, "preview"
    )
    to_render = R.build_time_map(timeline)

    captured: dict[str, list[dict]] = {}
    original_mix_program = R.audio_mod.mix_program

    def spy(voice_wav, music_items, sfx_items, speech_ranges, out_wav, *args, **kwargs):
        captured["items"] = list(sfx_items)
        return original_mix_program(voice_wav, music_items, sfx_items, speech_ranges, out_wav,
                                     *args, **kwargs)

    monkeypatch.setattr(R.audio_mod, "mix_program", spy)
    R.build_audio_bus(
        project, timeline, program_audio, R.render_duration(timeline), to_render, two_pass=False,
    )
    [item] = [i for i in captured["items"] if i["file"] == str(quiet_voice)]
    # the timeline's own gain_db (1.0) plus the automatic leveling gain.
    assert item["gain_db"] == pytest.approx(1.0 + expected_gain, abs=0.05)


def test_duck_ranges_for_an_audio_from_segment_come_from_the_borrowed_clips_words(
    tmp_path: Path,
) -> None:
    """``duck.mode=auto`` must ride the words of the ``audio_from`` clip, not

    the picture clip — the ducking counterpart of the mute-ranges check above.
    """
    project = build_render_project(tmp_path, slug="audio-from-duck-test")
    # c002 (the borrowed *audio* clip) gets a transcript; c001 (the picture)
    # keeps none here, so any ducking can only come from c002's words.
    (project.transcripts_dir / "c001.json").unlink()
    (project.transcripts_dir / "c002.json").write_text(
        json.dumps({
            "clip": "c002", "language": "pl", "project_language": "pl",
            "text": "slowa w tle",
            "words": [{"t": "slowa", "s": 0.5, "e": 1.0},
                      {"t": "w", "s": 1.05, "e": 1.15},
                      {"t": "tle", "s": 1.2, "e": 2.5}],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    timeline = Timeline.model_validate({
        "fps": 30, "width": 1920, "height": 1080,
        "tracks": {
            "video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 3.0,
                       "audio_from": {"clip": "c002", "in": 0.0, "out": 3.0}}],
            "music": [{"id": "m001", "file": "music/bed.wav", "at": 0.0, "end": 3.0,
                       "gain_db": -18, "fade_in": 0.0, "fade_out": 0.0,
                       "duck": {"mode": "auto", "amount_db": -15}}],
        },
    })
    canvas = R.canvas_for(timeline, preview=True)
    seg = timeline.tracks.video[0]
    segment_path = R.render_segment(project, timeline, seg, canvas, "preview")
    _video_out, program_audio = R.join_segments(
        project, timeline, [segment_path], canvas, "preview"
    )
    to_render = R.build_time_map(timeline)
    R.build_audio_bus(
        project, timeline, program_audio, R.render_duration(timeline), to_render, two_pass=False,
    )

    cmd = project.renders_dir / "duck.cmd"
    assert cmd.exists()
    commands = re.findall(r"^(\d+\.\d{3}) volume volume (\d+\.\d+);$",
                          cmd.read_text(encoding="utf-8"), re.M)
    assert len(commands) >= 3, "expected a ramped envelope driven by the borrowed clip's words"
    gains = [float(g) for _, g in commands]
    assert min(gains) < max(gains) * 0.5, "the music must actually duck under the borrowed words"


# ----------------------------------------------------------------------
# pre-flight: catch a bad range / missing file before any ffmpeg work
# ----------------------------------------------------------------------
def test_preflight_catches_an_out_of_range_segment_and_a_missing_music_file(
    tmp_path: Path,
) -> None:
    project = build_render_project(tmp_path, slug="preflight-test")
    timeline = Timeline.load(project.timeline_file)

    # c003 (silent.mp4) is a 4 s fixture; push its cut well past that.
    timeline.tracks.video[2] = timeline.tracks.video[2].model_copy(update={"out": 12.0})
    timeline.tracks.music[0] = timeline.tracks.music[0].model_copy(
        update={"file": "music/missing.wav"}
    )

    issues = R.preflight(project, timeline)
    assert any("exceeds clip duration" in i for i in issues), issues
    assert any("missing.wav" in i for i in issues), issues

    timeline.save(project.timeline_file)
    with pytest.raises(R.RenderError) as exc_info:
        R.render(project, preview=True)
    message = str(exc_info.value)
    assert "exceeds clip duration" in message
    assert "missing.wav" in message


def test_preflight_catches_a_missing_normalized_source(project: Project) -> None:
    # c001 is known to the registry but was never actually ingested.
    project.add_clip({"id": "c001", "order": 1, "duration": 10.0})
    timeline = Timeline.model_validate(
        {"tracks": {"video": [{"id": "s001", "clip": "c001", "in": 0.0, "out": 2.0}]}}
    )
    issues = R.preflight(project, timeline)
    assert any("no normalized source" in i for i in issues), issues


def test_no_music_flag_renders_despite_a_missing_music_file(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="no-music-test")
    timeline = Timeline.load(project.timeline_file)
    timeline.tracks.music[0] = timeline.tracks.music[0].model_copy(
        update={"file": "music/missing.wav"}
    )
    timeline.save(project.timeline_file)

    with pytest.raises(R.RenderError, match="missing.wav"):
        R.render(project, preview=True)

    out = R.render(project, preview=True, no_music=True)
    assert out.exists() and out.stat().st_size > 10_000
    assert project.load_state()["stages"]["render"]["status"] == "done"


def test_no_voice_flag_renders_despite_a_missing_voice_file(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="no-voice-test")
    timeline = Timeline.load(project.timeline_file)
    timeline.tracks.voice.append(
        VoiceItem(id="v001", file="voice/missing.wav", at=0.0, gain_db=0.0)
    )
    timeline.save(project.timeline_file)

    with pytest.raises(R.RenderError, match="missing.wav"):
        R.render(project, preview=True)

    out = R.render(project, preview=True, no_voice=True)
    assert out.exists()


# ----------------------------------------------------------------------
# parallel segment pass
# ----------------------------------------------------------------------
def test_parallel_segment_pass_matches_sequential_output(tmp_path: Path) -> None:
    """render.workers > 1 must produce the same segment files as workers=1."""
    project = build_render_project(tmp_path, slug="workers-test")
    timeline = Timeline.load(project.timeline_file)
    canvas = R.canvas_for(timeline, preview=True)

    sequential = R.render_segments(project, timeline, canvas, "preview", workers=1)
    assert len(sequential) == 3
    baseline = [(p, p.stat().st_size, float(probe(p)["format"]["duration"])) for p in sequential]

    # Cache keys don't depend on worker count, so a second pass would just
    # hit the cache; move the files aside so the parallel pass re-renders them.
    for path, _size, _duration in baseline:
        path.rename(path.with_name(path.stem + ".baseline" + path.suffix))

    parallel = R.render_segments(project, timeline, canvas, "preview", workers=3)
    assert [p.name for p, _s, _d in baseline] == [p.name for p in parallel]
    for (path, size, duration), rendered_path in zip(baseline, parallel):
        assert rendered_path.exists()
        assert rendered_path.stat().st_size == size, path.name
        assert float(probe(rendered_path)["format"]["duration"]) == pytest.approx(
            duration, abs=1e-6
        )


def test_render_segments_reports_progress_in_completion_order(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="progress-test")
    timeline = Timeline.load(project.timeline_file)
    canvas = R.canvas_for(timeline, preview=True)

    calls: list[tuple[int, int, str]] = []
    R.render_segments(
        project, timeline, canvas, "preview", workers=3,
        on_done=lambda completed, total, seg: calls.append((completed, total, seg.id)),
    )
    assert len(calls) == 3
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(c[1] == 3 for c in calls)
    assert {c[2] for c in calls} == {"s001", "s002", "s003"}


def test_a_bad_segment_fails_precisely_with_multiple_workers(project: Project) -> None:
    """Every segment here has no normalized source; the failure must still be precise."""
    timeline = Timeline.model_validate({"tracks": {"video": [
        {"id": "s001", "clip": "c001", "in": 0.0, "out": 2.0},
        {"id": "s002", "clip": "c002", "in": 0.0, "out": 2.0},
        {"id": "s003", "clip": "c003", "in": 0.0, "out": 2.0},
    ]}})
    with pytest.raises(R.RenderError, match=r"missing normalized source"):
        R.render_segments(project, timeline, R.Canvas(640, 360, 30), "preview", workers=3)


# ----------------------------------------------------------------------
# cache hygiene: `ytedit clean`
# ----------------------------------------------------------------------
def test_clean_removes_only_unreferenced_segments_and_all_intermediates(
    tmp_path: Path,
) -> None:
    project = build_render_project(tmp_path, slug="clean-test")
    R.render(project, preview=True)

    seg_dir = project.renders_dir / "segments"
    referenced_before = sorted(seg_dir.glob("*.mp4"))
    assert referenced_before, "expected the preview render to have cached segments"

    stale = seg_dir / "deadbeefdeadbeef1234.mp4"
    stale.write_bytes(b"stale segment from a since-changed timeline")

    assert (project.renders_dir / "program_video.mp4").exists()
    assert (project.renders_dir / "duck.cmd").exists()

    result = R.clean(project)

    assert not stale.exists()
    assert all(p.exists() for p in referenced_before), "a still-referenced segment was removed"
    assert not (project.renders_dir / "program_video.mp4").exists()
    assert not (project.renders_dir / "duck.cmd").exists()
    assert result["freed_bytes"] > 0
    assert stale.name in result["removed_segments"]
    assert "program_video.mp4" in result["removed_intermediates"]
    assert "duck.cmd" in result["removed_intermediates"]


def test_clean_segments_only_leaves_intermediates_untouched(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="clean-segments-test")
    R.render(project, preview=True)
    stale = project.renders_dir / "segments" / "deadbeefdeadbeef5678.mp4"
    stale.write_bytes(b"stale")

    result = R.clean(project, segments=True, intermediates=False)

    assert not stale.exists()
    assert (project.renders_dir / "program_video.mp4").exists()
    assert result["removed_intermediates"] == []


def test_clean_intermediates_only_leaves_segments_untouched(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="clean-intermediates-test")
    R.render(project, preview=True)
    stale = project.renders_dir / "segments" / "deadbeefdeadbeef9999.mp4"
    stale.write_bytes(b"stale")

    result = R.clean(project, segments=False, intermediates=True)

    assert stale.exists()
    assert not (project.renders_dir / "program_video.mp4").exists()
    assert result["removed_segments"] == []


def test_clean_never_touches_media_input_or_exports(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="clean-safety-test")
    R.render(project, master=True)
    master = next(project.exports_dir.glob("master_*.mp4"))
    source = project.source_path("c001")

    R.clean(project)

    assert master.exists()
    assert source.exists()


# ----------------------------------------------------------------------
# --draft tier
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def drafted(tmp_path_factory) -> tuple[Project, Path]:
    """A project with the draft tier rendered exactly once (real ingest, real proxies)."""
    project = build_render_project(tmp_path_factory.mktemp("draft"), slug="draft-test")
    return project, R.render(project, draft=True)


def test_draft_lands_where_it_should(drafted: tuple[Project, Path]) -> None:
    project, out = drafted
    assert out == project.renders_dir / "draft.mp4"
    assert out.exists() and out.stat().st_size > 1_000


def test_draft_has_the_expected_duration_and_720p_canvas(drafted: tuple[Project, Path]) -> None:
    _project, out = drafted
    data = probe(out)
    assert float(data["format"]["duration"]) == pytest.approx(EXPECTED_DURATION, abs=0.15)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1280, 720)
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    assert int(audio["sample_rate"]) == 48000
    assert audio["channels"] == 2


def test_draft_is_far_smaller_than_a_preview_of_the_same_cut(
    drafted: tuple[Project, Path], rendered: tuple[Project, Path]
) -> None:
    _draft_project, draft_out = drafted
    _preview_project, preview_out = rendered
    # ~1 MB/10s target vs. the preview's much higher bitrate — same programme
    # length (both built from timeline_document()), very different size.
    assert draft_out.stat().st_size < preview_out.stat().st_size


def test_draft_uses_its_own_segment_cache_namespace(drafted: tuple[Project, Path]) -> None:
    project, _out = drafted
    draft_segments = project.renders_dir / "segments_draft"
    assert draft_segments.is_dir()
    assert any(draft_segments.glob("*.mp4"))
    # The preview/master namespace is untouched by a draft-only render.
    assert not (project.renders_dir / "segments").exists()


def test_draft_and_preview_segment_caches_never_collide(tmp_path: Path) -> None:
    project = build_render_project(tmp_path, slug="draft-preview-cache-test")
    R.render(project, draft=True)
    R.render(project, preview=True)

    draft_keys = {p.stem for p in (project.renders_dir / "segments_draft").glob("*.mp4")}
    preview_keys = {p.stem for p in (project.renders_dir / "segments").glob("*.mp4")}
    assert draft_keys, "draft segments were not cached"
    assert preview_keys, "preview segments were not cached"
    assert draft_keys.isdisjoint(preview_keys)


def test_draft_audio_matches_the_preview_loudness_target(
    drafted: tuple[Project, Path],
) -> None:
    from ytedit.media.audio import measure_loudness

    _project, out = drafted
    loudness = measure_loudness(out)
    # Same single-pass loudnorm target as preview (-14 LUFS); draft only
    # differs in picture quality, not audio processing.
    assert loudness["input_i"] == pytest.approx(-14.0, abs=1.5)


def test_draft_falls_back_to_the_mezzanine_when_a_proxy_is_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    project = build_render_project(tmp_path, slug="draft-no-proxy-test")
    project.proxy_path("c001").unlink()

    with caplog.at_level(logging.WARNING, logger="ytedit.media.render"):
        out = R.render(project, draft=True)

    assert out.exists()
    assert any("no proxy" in rec.message for rec in caplog.records)


def test_video_dimensions_reads_the_proxys_own_coded_size(project: Project) -> None:
    # The vertical fixture's proxy is scaled to fit within 1280x720 keeping
    # its aspect ratio, so it is a very different shape from the mezzanine.
    from fixtures.make_fixtures import build_all

    media = build_all()
    shutil.copy(media["vertical.mp4"], project.input_dir / "v.mp4")
    from ytedit.media.ingest import ingest

    ingest(project, show_table=False)
    proxy = project.proxy_path("c001")
    dims = R._video_dimensions(proxy)
    assert dims is not None
    width, height = dims
    assert height == 720
    assert width < height  # still vertical, just smaller


# ----------------------------------------------------------------------
# master tier: hardware default vs --x264
# ----------------------------------------------------------------------
def test_master_default_uses_the_hardware_tier(project: Project) -> None:
    settings = load_settings(project.path)
    args = R.video_encoder_args(settings, "master", fps=30, fast=True)
    assert "h264_videotoolbox" in args


def test_master_x264_flag_selects_the_libx264_tier(project: Project) -> None:
    settings = load_settings(project.path)
    args = R.video_encoder_args(settings, "master", fps=30, fast=False)
    assert "libx264" in args
    assert "h264_videotoolbox" not in args


def test_render_passes_fast_true_by_default_and_false_with_x264(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``render(..., master=True)`` must reach ``video_encoder_args`` with
    ``fast=True`` unless ``x264=True`` is passed — this is the actual wiring
    ``ytedit render --master`` / ``--master --x264`` depends on."""
    project = build_render_project(tmp_path, slug="master-tier-wiring-test")
    seen: list[bool] = []
    real = R.video_encoder_args

    def spy(settings, mode, fps, fast=False):
        if mode == "master":
            seen.append(fast)
        return real(settings, mode, fps, fast=fast)

    monkeypatch.setattr(R, "video_encoder_args", spy)
    R.render(project, master=True)
    assert seen == [True]

    seen.clear()
    R.render(project, master=True, x264=True)
    assert seen == [False]
