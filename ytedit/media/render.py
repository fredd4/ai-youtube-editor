"""Timeline -> finished file. The render pipeline.

Five passes, each writing an artifact the next one reads (so a failed render
can be resumed and a preview can be diffed against a master):

1. **Segment pass** — every ``VideoSegment`` is cut from its normalized
   mezzanine, graded, fitted to the canvas and written to
   ``renders/segments/<hash>.mp4`` with PCM audio. The hash covers the segment
   spec, the canvas, the encoder preset and the source mtime, so unchanged
   segments are reused across renders. A clip with ``use_denoised`` set in
   ``state.json`` takes its audio from ``media/audio/<clip>.denoised.wav``
   instead of the mezzanine's own stream (that file's mtime is in the key too).
   A segment carrying ``audio_from`` keeps its own picture but reads its audio
   from another clip's range (an overlay cutaway with the narration running on
   underneath); that range is trimmed or padded to the segment's own frame
   count, and the audio clip's mute ranges, denoised WAV and mtime all take part
   in the cache key.
2. **Join pass** — hard cuts go through the concat demuxer with ``-c copy``;
   ``fade``/``xfade`` transitions build a chained ``xfade`` + ``acrossfade``
   ``filter_complex`` with accumulating offsets. Produces
   ``renders/program_video.mp4`` and ``renders/program_audio.wav``.
3. **Audio bus** — the source bus, the voice pickups, the music cues (ducked
   under speech) and the sfx meet in one ``amix``; the result is loudness
   normalized to −14 LUFS / −1 dBTP into ``renders/final_audio.wav``.
4. **Captions** — ``renders/captions.ass`` is burned in the final pass;
   ``exports/captions.srt`` is written from the transcripts for upload.
5. **Final encode** — programme video + final audio, muxed with the master or
   preview preset.

Timeline time vs render time
---------------------------
An overlapping transition makes the programme *shorter* than the sum of its
segments. :meth:`Timeline.segment_positions` already accounts for that when
``transition_in.type == "xfade"``, but the timeline model treats ``fade`` as a
non-overlapping transition while ffmpeg's ``xfade`` (which is what we render a
``fade`` with) always overlaps. :func:`render_positions` therefore recomputes
the placement with *both* transition types overlapping, and
:func:`build_time_map` maps any timeline instant to its render instant. Every
caption, music cue, voice pickup and speech range is passed through that map
before it reaches ffmpeg. On a timeline whose only transitions are cuts and
``xfade``s the map is the identity.

Frame-exact segments
--------------------
A segment's ``in``/``out`` are arbitrary floats but video only exists in whole
frames, so every segment is rendered to exactly
:meth:`~ytedit.timeline.VideoSegment.frames` frames (``-frames:v`` on a
``setpts=PTS-STARTPTS`` video graph, ``atrim``/``apad`` to the same number of
samples) and the concat list pins each file's ``duration`` to that length.
:meth:`Timeline.segment_positions` does the same arithmetic in frames, so
:func:`render_duration` equals the joined programme to the frame and an item
placed at a segment's start lands on its first frame, however long the cut.
"""

from __future__ import annotations

import functools

import hashlib
import json
import os
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import Settings
from ..log import console, get_logger
from ..project import Project
from ..timeline import (
    Caption,
    Timeline,
    VideoSegment,
    frames_to_seconds,
    speech_ranges_from_transcripts,
)
from . import audio as audio_mod
from . import captions as captions_mod
from .color import even, source_chain
from .ffmpeg import FFmpegError, Progress, ff, ffprobe_json, has_encoder

log = get_logger(__name__)

#: ``xfade`` transition used when a ``fade`` transition does not name one.
DEFAULT_XFADE: str = "fade"

#: Preview renders are letterboxed into this height.
PREVIEW_HEIGHT: int = 720


class RenderError(RuntimeError):
    """The timeline cannot be rendered (validation or media problem)."""


@dataclass(frozen=True)
class Canvas:
    """Output geometry for one render."""

    width: int
    height: int
    fps: int

    @property
    def size(self) -> str:
        """``WxH`` for logs."""
        return f"{self.width}x{self.height}"


@dataclass
class SegmentPlacement:
    """A segment with its absolute placement in *render* time."""

    segment: VideoSegment
    start: float
    end: float


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def escape_filter_path(path: Path | str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter argument.

    ``ass=`` (and every other filter that takes a filename) parses ``\\``,
    ``:``, ``'``, ``,``, ``[`` and ``]`` specially.

    Args:
        path: Path to embed.

    Returns:
        The escaped string.
    """
    text = str(path)
    for char in ("\\", ":", "'", ",", "[", "]", ";"):
        text = text.replace(char, "\\" + char)
    return text


def _atempo(speed: float) -> str:
    """Return an ``atempo`` chain for any positive speed factor."""
    factor = float(speed)
    if factor <= 0 or abs(factor - 1.0) < 1e-6:
        return ""
    parts: list[str] = []
    # atempo accepts 0.5..100; chain stages for anything outside that.
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    while factor > 100.0:
        parts.append("atempo=100")
        factor /= 100.0
    parts.append(f"atempo={factor:.6f}")
    return ",".join(parts)


def _join(parts: Sequence[str]) -> str:
    """Join non-empty filter fragments with commas."""
    return ",".join(p for p in parts if p)


def canvas_for(timeline: Timeline, preview: bool) -> Canvas:
    """Return the output canvas for a render.

    Previews are scaled to :data:`PREVIEW_HEIGHT` keeping the timeline's aspect
    ratio; masters use the timeline canvas unchanged.
    """
    if not preview:
        return Canvas(even(timeline.width), even(timeline.height), int(timeline.fps))
    height = min(PREVIEW_HEIGHT, even(timeline.height))
    width = even(round(timeline.width * height / max(1, timeline.height)))
    return Canvas(width, height, int(timeline.fps))


# ----------------------------------------------------------------------
# timeline time -> render time
# ----------------------------------------------------------------------
def render_positions(timeline: Timeline) -> list[SegmentPlacement]:
    """Compute segment placement as ffmpeg will actually lay it out.

    Unlike :meth:`Timeline.segment_positions`, a ``fade`` transition overlaps
    the previous segment too — that is what ``xfade`` does. Everything is in
    whole frames at the timeline fps, exactly like the segment pass.
    """
    return [
        SegmentPlacement(pos.segment, pos.start, pos.end)
        for pos in timeline.segment_positions(fade_overlaps=True)
    ]


def render_duration(timeline: Timeline) -> float:
    """Programme length in render time (seconds)."""
    positions = render_positions(timeline)
    return round(positions[-1].end if positions else 0.0, 6)


def build_time_map(timeline: Timeline) -> Callable[[float], float]:
    """Return a function mapping timeline seconds to render seconds.

    Inside a segment the mapping is a pure offset; past the last segment the
    tail keeps its distance from the end. The identity function is returned for
    a timeline whose transitions are all cuts or ``xfade``s.
    """
    timeline_pos = timeline.segment_positions()
    render_pos = render_positions(timeline)
    if not timeline_pos or all(
        abs(a.start - b.start) < 1e-9 for a, b in zip(timeline_pos, render_pos)
    ):
        return lambda t: max(0.0, float(t))

    def convert(t: float) -> float:
        value = float(t)
        for tp, rp in zip(timeline_pos, render_pos):
            if value < tp.end or tp is timeline_pos[-1]:
                return max(0.0, rp.start + (value - tp.start))
        return max(0.0, value)  # pragma: no cover - unreachable

    return convert


# ----------------------------------------------------------------------
# encoder presets
# ----------------------------------------------------------------------
def video_encoder_args(settings: Settings, mode: str, fps: int) -> list[str]:
    """Return the ``-c:v ...`` arguments for the final encode.

    Args:
        settings: Project settings (``encoding.master`` / ``encoding.preview``).
        mode: ``"master"`` or ``"preview"``.
        fps: Output frame rate (drives the GOP length).

    Returns:
        A flat argument list.
    """
    if mode == "preview":
        enc = settings.encoding("preview")
        if has_encoder(str(enc.get("vcodec", "h264_videotoolbox"))):
            return [
                "-c:v", str(enc.get("vcodec", "h264_videotoolbox")),
                "-q:v", str(enc.get("q", 60)),
                "-profile:v", str(enc.get("profile", "high")),
                "-pix_fmt", str(enc.get("pix_fmt", "yuv420p")),
            ]
        return [
            "-c:v", str(enc.get("fallback_vcodec", "libx264")),
            "-crf", str(enc.get("fallback_crf", 23)),
            "-preset", str(enc.get("fallback_preset", "veryfast")),
            "-pix_fmt", str(enc.get("pix_fmt", "yuv420p")),
        ]

    enc = settings.encoding("master")
    gop = int(enc.get("gop", max(1, fps // 2)))
    return [
        "-c:v", str(enc.get("vcodec", "libx264")),
        "-crf", str(enc.get("crf", 18)),
        "-preset", str(enc.get("preset", "slow")),
        "-profile:v", str(enc.get("profile", "high")),
        "-level", str(enc.get("level", "4.2")),
        "-pix_fmt", str(enc.get("pix_fmt", "yuv420p")),
        "-g", str(gop),
        "-keyint_min", str(enc.get("keyint_min", gop)),
        "-sc_threshold", str(enc.get("sc_threshold", 0)),
        "-bf", str(enc.get("bf", 2)),
        "-coder", "1",
        "-color_primaries", str(enc.get("color_primaries", "bt709")),
        "-color_trc", str(enc.get("color_trc", "bt709")),
        "-colorspace", str(enc.get("colorspace", "bt709")),
    ]


def segment_encoder_args(settings: Settings, mode: str) -> list[str]:
    """Return the ``-c:v ...`` arguments for a cached segment intermediate."""
    if mode == "preview":
        enc = settings.encoding("preview")
        if has_encoder(str(enc.get("vcodec", "h264_videotoolbox"))):
            return [
                "-c:v", str(enc.get("vcodec", "h264_videotoolbox")),
                "-q:v", str(enc.get("q", 60)),
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",
            ]
    enc = settings.encoding("segment")
    return [
        "-c:v", str(enc.get("vcodec", "libx264")),
        "-crf", str(enc.get("crf", 16)),
        "-preset", str(enc.get("preset", "fast")),
        "-pix_fmt", str(enc.get("pix_fmt", "yuv420p")),
    ]


def parse_bitrate(value: Any, default: int = 384_000) -> int:
    """Parse an ffmpeg bitrate string (``"384k"``, ``256000``) into bits/s."""
    text = str(value).strip().lower()
    multiplier = 1
    if text.endswith("k"):
        multiplier, text = 1000, text[:-1]
    elif text.endswith("m"):
        multiplier, text = 1_000_000, text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return default


def aac_encoder(bitrate: int) -> str:
    """Pick the AAC encoder that can actually reach ``bitrate``.

    ffmpeg's native ``aac`` encoder silently caps stereo output at roughly
    256 kbps, so the playbook's 384 kbps AAC-LC master is unreachable with it.
    macOS builds expose ``aac_at`` (AudioToolbox), which delivers ~320 kbps for
    the same request; it is preferred whenever more than 256 kbps is asked for.
    """
    if bitrate > 256_000 and has_encoder("aac_at"):
        return "aac_at"
    return "aac"


def audio_encoder_args(settings: Settings) -> list[str]:
    """Return the ``-c:a ...`` arguments for the final mux (AAC-LC 384k/48k)."""
    cfg = settings.section("audio")
    bitrate = str(cfg.get("abitrate", "384k"))
    codec = str(cfg.get("acodec", "aac"))
    if codec == "aac":
        codec = aac_encoder(parse_bitrate(bitrate))
    return [
        "-c:a", codec,
        "-b:a", bitrate,
        "-ar", str(cfg.get("sample_rate", 48000)),
        "-ac", str(cfg.get("channels", 2)),
    ]


def color_tag_filter(settings: Settings) -> str:
    """Return the ``setparams`` filter that stamps bt709 onto every frame.

    The ``-color_primaries``/``-color_trc`` *output* options alone do not reach
    the muxed stream on ffmpeg 7.x (only ``-colorspace`` survives), so the
    frames themselves are tagged in the filter graph.
    """
    enc = settings.encoding("master")
    return (
        f"setparams=color_primaries={enc.get('color_primaries', 'bt709')}"
        f":color_trc={enc.get('color_trc', 'bt709')}"
        f":colorspace={enc.get('colorspace', 'bt709')}:range=tv"
    )


# ----------------------------------------------------------------------
# pass 1: segments
# ----------------------------------------------------------------------
def segment_mute_ranges(
    timeline: Timeline, seg: VideoSegment
) -> list[tuple[float, float, float]]:
    """Convert the timeline's clip-time mute ranges to segment-local time.

    ``mute_ranges`` are expressed in the *clip*'s own time base and apply
    wherever that clip range is used. A segment only sees the intersection with
    ``[in, out)``, shifted by ``in`` and divided by ``speed``. For an overlay
    cutaway (``audio_from`` set) the ranges are read in the **audio** clip's
    time base — that is the audio the filter graph actually attenuates.

    Args:
        timeline: Timeline holding ``mute_ranges``.
        seg: The segment being rendered.

    Returns:
        ``(start, end, gain_db)`` triples in seconds from the segment start.
    """
    speed = seg.speed if seg.speed > 0 else 1.0
    clip_id, src_in, src_out = seg.audio_source
    out: list[tuple[float, float, float]] = []
    for mute in timeline.mute_ranges:
        if mute.clip != clip_id:
            continue
        start = max(float(mute.s), src_in)
        end = min(float(mute.e), src_out)
        if end <= start:
            continue
        out.append(
            (
                round((start - src_in) / speed, 4),
                round((end - src_in) / speed, 4),
                float(mute.gain_db),
            )
        )
    return sorted(out)


def denoised_audio(
    project: Project, clip_id: str, state: dict[str, Any] | None = None
) -> Path | None:
    """Return the denoised WAV to use for a clip's audio, when one is active.

    A clip is denoised by ``ytedit denoise`` (:func:`ytedit.media.audio.denoise_clips`),
    which writes ``media/audio/<clip>.denoised.wav`` and sets
    ``clips[<id>].use_denoised``. The file covers the *whole* clip and is
    sample-aligned with the mezzanine, so the segment cut applies the same
    ``-ss``/``-t`` to it.

    Args:
        project: Owning project.
        clip_id: Clip id.
        state: Pre-loaded ``state.json`` (avoids a re-read).

    Returns:
        The path, or ``None`` when the clip has no active denoised audio.
    """
    clip = (state if state is not None else project.load_state()).get("clips", {}).get(clip_id, {})
    if not clip.get("use_denoised"):
        return None
    rel = clip.get("denoised")
    if not rel:
        return None
    path = project.path / str(rel)
    if not path.exists():
        log.warning("%s: use_denoised is set but %s is missing; using the source audio",
                    clip_id, rel)
        return None
    return path


def _denoise_stamp(project: Project, clip_id: str) -> list[Any] | None:
    """``[name, mtime]`` of a clip's active denoised WAV, for the segment cache key."""
    denoised = denoised_audio(project, clip_id)
    if denoised is None:
        return None
    try:
        return [denoised.name, denoised.stat().st_mtime_ns]
    except OSError:  # pragma: no cover - raced deletion
        return None


def segment_key(
    project: Project,
    timeline: Timeline,
    seg: VideoSegment,
    canvas: Canvas,
    mode: str,
) -> str:
    """Return the cache key (short hash) for a rendered segment."""
    source = project.source_path(seg.clip)
    try:
        mtime = source.stat().st_mtime_ns
        size = source.stat().st_size
    except OSError:
        mtime, size = 0, 0
    # Re-denoising a clip must invalidate every segment cut from it.
    denoise_stamp = _denoise_stamp(project, seg.clip)
    # An overlay cutaway is only as fresh as the clip it borrows its audio from.
    audio_stamp: list[Any] | None = None
    audio_denoise_stamp: list[Any] | None = None
    if seg.audio_from is not None:
        audio_source = project.source_path(seg.audio_from.clip)
        try:
            audio_stamp = [
                audio_source.name,
                audio_source.stat().st_mtime_ns,
                audio_source.stat().st_size,
            ]
        except OSError:
            audio_stamp = [audio_source.name, 0, 0]
        audio_denoise_stamp = _denoise_stamp(project, seg.audio_from.clip)
    payload = {
        "segment": seg.model_dump(by_alias=True, mode="json"),
        "canvas": [canvas.width, canvas.height, canvas.fps],
        "mode": mode,
        "mute": segment_mute_ranges(timeline, seg),
        "source": [source.name, mtime, size],
        "denoised": denoise_stamp,
        "audio_from": (
            seg.audio_from.model_dump(by_alias=True, mode="json")
            if seg.audio_from is not None
            else None
        ),
        "audio_source": audio_stamp,
        "audio_denoised": audio_denoise_stamp,
        "grade": project.settings.get(f"grade.presets.{seg.grade}", []),
        "frames": seg.frames(canvas.fps),
        "version": 4,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _segment_is_valid(path: Path, expected: float, tolerance: float = 0.034) -> bool:
    """True when ``path`` probes cleanly and is within ``tolerance`` s of ``expected``.

    A crashed or interrupted ffmpeg leaves a headerless ``.mp4`` behind; the
    concat demuxer then skips it *silently* (exit code 0) and the programme
    comes out short. Never trust a cached segment without this check.
    Segments are cut to whole frames, so the default tolerance is one frame.
    """
    try:
        if path.stat().st_size == 0:
            return False
        info = ffprobe_json(path)
    except Exception:
        return False
    try:
        actual = float((info.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return False
    return actual > 0 and abs(actual - expected) <= tolerance


@functools.lru_cache(maxsize=256)
def _source_has_audio(path: Path) -> bool:
    """True when the file carries at least one audio stream (cached per path)."""
    try:
        info = ffprobe_json(path, "-show_streams", "-select_streams", "a")
    except Exception:  # pragma: no cover - probe failure -> assume audio
        return True
    return bool(info.get("streams"))


def render_segment(
    project: Project,
    timeline: Timeline,
    seg: VideoSegment,
    canvas: Canvas,
    mode: str,
    force: bool = False,
) -> Path:
    """Cut, grade and fit one segment; return the cached intermediate.

    Args:
        project: Owning project.
        timeline: The timeline (for ``mute_ranges``).
        seg: Segment to render.
        canvas: Output geometry.
        mode: ``"master"`` or ``"preview"`` (chooses the intermediate encoder).
        force: Re-render even when the cached file exists.

    Returns:
        Path to ``renders/segments/<hash>.mp4``.

    Raises:
        RenderError: When the clip's mezzanine (or that of an ``audio_from``
            clip) is missing.
    """
    settings = project.settings
    source = project.source_path(seg.clip)
    if not source.exists():
        raise RenderError(f"{seg.id}: missing normalized source {source}")

    out_dir = project.renders_dir / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{segment_key(project, timeline, seg, canvas, mode)}.mp4"
    tmp = out.with_name(out.stem + ".partial.mp4")
    frames = seg.frames(canvas.fps)
    length = frames_to_seconds(frames, canvas.fps)
    frame_tolerance = 1.0 / canvas.fps
    if out.exists() and not force:
        if _segment_is_valid(out, length, tolerance=frame_tolerance):
            log.debug("segment %s cached at %s", seg.id, out.name)
            return out
        log.warning("segment %s: cached file %s is corrupt/short, re-rendering", seg.id, out.name)
        out.unlink(missing_ok=True)

    state = project.load_state()
    clip = state.get("clips", {}).get(seg.clip, {})
    src_w = int(clip.get("width") or canvas.width)
    src_h = int(clip.get("height") or canvas.height)
    is_iphone = bool(clip.get("is_iphone", False))
    has_audio = bool(clip.get("has_audio", True))
    if has_audio and not _source_has_audio(source):
        # state.json can be stale (or hand-made); trust the file over the registry.
        log.warning("%s: source %s has no audio stream, using silence", seg.id, source.name)
        has_audio = False
    speed = seg.speed if seg.speed > 0 else 1.0

    # --- video -------------------------------------------------------
    # Input seeking leaves the first decoded frame up to one frame *after*
    # ``in``; rebasing its pts to 0 keeps the CFR grid aligned so the segment is
    # exactly ``frames`` frames long (see ``-frames:v`` below).
    in_label = "src"
    chain, needs_split = source_chain(
        hdr=None,  # ingest already tonemapped the mezzanine to bt709 SDR
        grade=seg.grade,
        is_iphone=is_iphone,
        src_w=src_w,
        src_h=src_h,
        rotation=0,  # ingest baked the display matrix into the pixels
        canvas_w=canvas.width,
        canvas_h=canvas.height,
        fps=canvas.fps,
        fit=seg.transform.fit,
        settings=settings,
        in_label=in_label,
        out_label="v",
    )
    graph: list[str] = []
    if abs(speed - 1.0) > 1e-6:
        graph.append(f"[0:v]setpts=(PTS-STARTPTS)*{1.0 / speed:.6f}[{in_label}]")
    else:
        graph.append(f"[0:v]setpts=PTS-STARTPTS[{in_label}]")
    graph.append(chain if needs_split else f"[{in_label}]{chain}[v]")

    # --- audio -------------------------------------------------------
    audio_cfg = settings.section("audio")
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    channels = int(audio_cfg.get("channels", 2))
    aformat = (
        f"aformat=sample_fmts=fltp:sample_rates={sample_rate}"
        f":channel_layouts={'stereo' if channels == 2 else 'mono'}"
    )

    # An overlay cutaway keeps this picture but borrows its sound (and its
    # denoise state, and the time base of its mute ranges) from another clip.
    audio_clip, audio_in, _audio_out = seg.audio_source
    if seg.audio_from is not None:
        audio_source = project.source_path(audio_clip)
        if not audio_source.exists():
            raise RenderError(
                f"{seg.id}: missing normalized source {audio_source} "
                f"for audio_from clip {audio_clip!r}"
            )
        audio_clip_state = state.get("clips", {}).get(audio_clip, {})
        audio_has = bool(audio_clip_state.get("has_audio", True))
        if audio_has and not _source_has_audio(audio_source):
            log.warning(
                "%s: audio_from source %s has no audio stream, using silence",
                seg.id, audio_source.name,
            )
            audio_has = False
    else:
        audio_source = source
        audio_has = has_audio

    inputs: list[Any] = ["-ss", f"{seg.in_:.6f}", "-i", str(source)]
    silent = seg.mute_source or not audio_has
    cleaned = None if silent else denoised_audio(project, audio_clip, state)
    if silent:
        inputs += [
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout={'stereo' if channels == 2 else 'mono'}"
                  f":sample_rate={sample_rate}",
        ]
        audio_index = 1
    elif cleaned is not None:
        # The denoised WAV is the full clip, aligned with the mezzanine, so the
        # same -ss applies. Gains and mute ranges are unchanged below.
        log.debug("segment %s: using denoised audio %s", seg.id, cleaned.name)
        inputs += ["-ss", f"{audio_in:.6f}", "-i", str(cleaned)]
        audio_index = 1
    elif seg.audio_from is not None:
        log.debug(
            "segment %s: audio from %s %.3f-%.3f", seg.id, audio_clip, audio_in, _audio_out
        )
        inputs += ["-ss", f"{audio_in:.6f}", "-i", str(audio_source)]
        audio_index = 1
    else:
        audio_index = 0

    achain = [aformat]
    if not silent:
        achain.append(_atempo(speed))
        if abs(seg.source_audio_gain_db) > 1e-6:
            achain.append(f"volume={audio_mod.db_to_linear(seg.source_audio_gain_db):.6f}")
        mutes = segment_mute_ranges(timeline, seg)
        if mutes:
            achain.append(audio_mod.mute_ranges_expr(mutes))
    achain.append("asetpts=N/SR/TB")
    # Exactly as many samples as the video has frames: trim a long take, pad a
    # short one (a source that ends early, or the infinite anullsrc).
    samples = int(round(frames * sample_rate / canvas.fps))
    achain.append(f"atrim=end_sample={samples}")
    achain.append(f"apad=whole_len={samples}")
    graph.append(f"[{audio_index}:a]{_join(achain)}[a]")

    args: list[Any] = [
        *inputs,
        "-filter_complex", ";".join(graph),
        "-map", "[v]",
        "-map", "[a]",
        "-frames:v", str(frames),
        "-r", str(canvas.fps),
        "-fps_mode", "cfr",
        *segment_encoder_args(settings, mode),
        "-c:a", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        str(tmp),
    ]
    log.debug("segment %s -> %s", seg.id, out.name)
    try:
        ff(*args)
    except FFmpegError:
        tmp.unlink(missing_ok=True)
        raise
    if not _segment_is_valid(tmp, length, tolerance=frame_tolerance):
        tmp.unlink(missing_ok=True)
        raise RenderError(
            f"segment {seg.id}: ffmpeg produced an unreadable or short file for {seg.clip} "
            f"(expected {frames} frames = {length:.6f}s)"
        )
    tmp.replace(out)
    return out


# ----------------------------------------------------------------------
# pass 2: join
# ----------------------------------------------------------------------
def _concat_list(
    paths: Sequence[Path],
    destination: Path,
    durations: Sequence[float] | None = None,
) -> Path:
    """Write a concat-demuxer list file and return it.

    With ``durations`` every entry gets a ``duration`` directive, so the next
    file starts exactly there instead of wherever the container's
    millisecond-precision header says the previous one ended.
    """
    lines: list[str] = []
    for i, p in enumerate(paths):
        lines.append(f"file {shlex.quote(str(p.resolve()))}")
        if durations is not None:
            lines.append(f"duration {durations[i]:.6f}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def join_segments(
    project: Project,
    timeline: Timeline,
    segments: Sequence[Path],
    canvas: Canvas,
    mode: str,
    progress_cb: Callable[[Progress], None] | None = None,
) -> tuple[Path, Path]:
    """Join the rendered segments into the programme video and audio bus.

    Hard cuts use the concat demuxer with ``-c copy`` (the segments already
    share codec, size, pixel format and frame rate). Any ``fade``/``xfade``
    transition switches to a chained ``xfade`` + ``acrossfade``
    ``filter_complex`` with accumulating offsets; runs of hard cuts inside such
    a timeline are joined with the ``concat`` filter.

    Args:
        project: Owning project.
        timeline: Timeline supplying the transitions.
        segments: Rendered segment files, in order.
        canvas: Output geometry.
        mode: ``"master"`` or ``"preview"``.
        progress_cb: Optional progress sink.

    Returns:
        ``(program_video.mp4, program_audio.wav)``.
    """
    settings = project.settings
    renders = project.renders_dir
    renders.mkdir(parents=True, exist_ok=True)
    video_out = renders / "program_video.mp4"
    audio_out = renders / "program_audio.wav"
    audio_cfg = settings.section("audio")
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    channels = int(audio_cfg.get("channels", 2))

    tracks = list(timeline.tracks.video)
    transitions = [
        (i, seg.transition_in)
        for i, seg in enumerate(tracks)
        if i > 0 and seg.transition_in.type in ("fade", "xfade")
        and seg.transition_in.duration > 0
    ]

    fps = canvas.fps
    if not transitions:
        listing = _concat_list(
            segments, renders / "concat.txt",
            durations=[seg.rendered_duration(fps) for seg in tracks],
        )
        ff(
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-map", "0:v", "-c:v", "copy", "-movflags", "+faststart", str(video_out),
            progress_cb=progress_cb, total_duration=render_duration(timeline),
        )
        ff(
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-map", "0:a", "-c:a", "pcm_s24le",
            "-ar", str(sample_rate), "-ac", str(channels), str(audio_out),
        )
        _check_joined_duration(
            video_out, render_duration(timeline), tolerance=_frame_drift_tolerance(timeline)
        )
        return video_out, audio_out

    inputs: list[Any] = []
    for path in segments:
        inputs += ["-i", str(path)]

    vgraph: list[str] = []
    agraph: list[str] = []
    vlabel, alabel = "0:v", "0:a"
    # Offsets and overlaps in whole frames, mirroring Timeline.segment_positions.
    total = tracks[0].frames(fps) if tracks else 0
    previous = total

    for i, seg in enumerate(tracks[1:], start=1):
        trans = seg.transition_in
        nv, na = f"vx{i}", f"ax{i}"
        length = seg.frames(fps)
        if trans.type in ("fade", "xfade") and trans.frames(fps) > 0:
            overlap = min(trans.frames(fps), previous, length)
            name = trans.name or DEFAULT_XFADE
            offset = max(0, total - overlap)
            vgraph.append(
                f"[{vlabel}][{i}:v]xfade=transition={name}"
                f":duration={frames_to_seconds(overlap, fps):.6f}"
                f":offset={frames_to_seconds(offset, fps):.6f}[{nv}]"
            )
            agraph.append(
                f"[{alabel}][{i}:a]acrossfade=d={frames_to_seconds(overlap, fps):.6f}"
                f":c1=tri:c2=tri[{na}]"
            )
            total += length - overlap
        else:
            vgraph.append(f"[{vlabel}][{i}:v]concat=n=2:v=1:a=0[{nv}]")
            agraph.append(f"[{alabel}][{i}:a]concat=n=2:v=0:a=1[{na}]")
            total += length
        previous = length
        vlabel, alabel = nv, na

    ff(
        *inputs,
        "-filter_complex", ";".join(vgraph),
        "-map", f"[{vlabel}]",
        "-r", str(canvas.fps),
        "-fps_mode", "cfr",
        *segment_encoder_args(settings, mode),
        "-movflags", "+faststart",
        str(video_out),
        progress_cb=progress_cb,
        total_duration=render_duration(timeline),
    )
    ff(
        *inputs,
        "-filter_complex", ";".join(agraph),
        "-map", f"[{alabel}]",
        "-c:a", "pcm_s24le",
        "-ar", str(sample_rate), "-ac", str(channels),
        str(audio_out),
    )
    _check_joined_duration(
        video_out, render_duration(timeline), tolerance=_frame_drift_tolerance(timeline)
    )
    return video_out, audio_out


def _frame_drift_tolerance(timeline: Timeline) -> float:
    """Tolerance for the joined-duration check.

    Segments are cut to whole frames and the concat list pins every offset, so
    a hard-cut programme joins to the frame. The ``xfade``/``acrossfade`` chain
    is allowed one frame per overlapping transition on top of a two-frame
    floor (never under 100 ms); anything beyond that is a missing or corrupt
    segment, not rounding.
    """
    fps = float(timeline.fps or 30)
    overlapping = sum(
        1 for i, seg in enumerate(timeline.tracks.video)
        if i > 0 and seg.transition_in.type in ("fade", "xfade")
    )
    return max(0.1, (2 + overlapping) / fps)


def _check_joined_duration(path: Path, expected: float, tolerance: float = 0.5) -> None:
    """Abort when the joined programme is shorter than the timeline says.

    ffmpeg's concat demuxer exits 0 even when it could not open one of the
    listed files, so a corrupt segment would otherwise surface only as a
    mysteriously short master.
    """
    info = ffprobe_json(path)
    actual = float((info.get("format") or {}).get("duration") or 0.0)
    if abs(actual - expected) > tolerance:
        raise RenderError(
            f"joined programme is {actual:.2f}s but the timeline renders to "
            f"{expected:.2f}s; a segment is missing or corrupt (see {path.parent / 'segments'})"
        )


# ----------------------------------------------------------------------
# pass 3: audio bus
# ----------------------------------------------------------------------
def build_audio_bus(
    project: Project,
    timeline: Timeline,
    program_audio: Path,
    duration: float,
    to_render: Callable[[float], float],
    two_pass: bool,
) -> Path:
    """Mix voice, music and sfx over the source bus and normalize the result.

    Args:
        project: Owning project.
        timeline: Timeline supplying the tracks.
        program_audio: The joined source-audio bus.
        duration: Programme length in render seconds.
        to_render: Timeline-to-render time map.
        two_pass: Use two-pass loudnorm (masters) instead of one (previews).

    Returns:
        ``renders/final_audio.wav``.
    """
    settings = project.settings
    renders = project.renders_dir
    loud = settings.section("audio").get("loudnorm", {})
    duck_cfg = settings.section("ducking")

    cleanup = str(settings.get("audio.voice_cleanup", "none"))
    bus = program_audio
    chain = audio_mod.voice_cleanup_chain(cleanup)
    if chain:
        bus = renders / "program_audio_clean.wav"
        log.info("voice cleanup (%s)", cleanup)
        ff("-i", str(program_audio), "-af", chain, "-c:a", "pcm_s24le", str(bus))

    # Voice pickups and sfx ride the mix un-ducked; music is ducked under
    # speech. Narration pickups also *create* speech, so they extend the
    # speech ranges that drive automatic ducking.
    voice_items: list[dict[str, Any]] = []
    voice_spans: list[tuple[float, float]] = []
    for item in timeline.tracks.voice:
        path = project.path / item.file
        length = audio_mod.audio_duration(path)
        at = to_render(item.at)
        end = to_render(item.end) if item.end is not None else at + length
        voice_items.append(
            {"file": str(path), "at": at, "end": end, "gain_db": float(item.gain_db)}
        )
        voice_spans.append((at, end))

    sfx_items: list[dict[str, Any]] = []
    for item in timeline.tracks.sfx:
        path = project.path / item.file
        at = to_render(item.at)
        end = to_render(item.end) if item.end is not None else at + audio_mod.audio_duration(path)
        sfx_items.append(
            {"file": str(path), "at": at, "end": end, "gain_db": float(item.gain_db)}
        )

    needs_speech = any(
        cue.duck.mode == "auto" for cue in timeline.tracks.music
    )
    speech: list[tuple[float, float]] = []
    if needs_speech:
        raw = speech_ranges_from_transcripts(
            project,
            timeline,
            merge_gap=float(duck_cfg.get("merge_gap", 0.4)),
            pad=float(duck_cfg.get("pad", 0.15)),
        )
        speech = [(to_render(s), to_render(e)) for s, e in raw] + voice_spans
        speech.sort()
        log.info("speech ranges for ducking: %d", len(speech))

    music_items: list[dict[str, Any]] = []
    for cue in timeline.tracks.music:
        duck = cue.duck
        ranges: list[tuple[float, float]] = []
        if duck.mode == "auto":
            ranges = speech
        elif duck.mode == "manual":
            ranges = [(to_render(s), to_render(e)) for s, e in duck.ranges]
        music_items.append(
            {
                "file": str(project.path / cue.file),
                "at": to_render(cue.at),
                "end": to_render(cue.end),
                "gain_db": float(cue.gain_db),
                "fade_in": float(cue.fade_in),
                "fade_out": float(cue.fade_out),
                "duck": duck.mode != "off" and bool(ranges),
                "ranges": ranges,
                "amount_db": float(duck.amount_db),
                "attack": float(duck.attack),
                "release": float(duck.release),
                "pre_roll": float(duck_cfg.get("pre_roll", 0.15)),
            }
        )

    mixed = renders / "mix.wav"
    if music_items or voice_items or sfx_items:
        log.info(
            "mixing %d music cue(s), %d voice pickup(s), %d sfx",
            len(music_items), len(voice_items), len(sfx_items),
        )
        audio_mod.mix_program(
            bus,
            music_items,
            voice_items + sfx_items,
            speech,
            mixed,
            settings=settings,
            duration=duration,
            cmd_file=renders / "duck.cmd",
        )
    else:
        mixed = bus

    final = renders / "final_audio.wav"
    audio_mod.normalize(
        mixed,
        final,
        I=float(loud.get("I", -14)),
        TP=float(loud.get("TP", -1)),
        LRA=float(loud.get("LRA", 11)),
        two_pass=two_pass,
        sample_rate=int(settings.get("audio.sample_rate", 48000)),
        channels=int(settings.get("audio.channels", 2)),
    )
    return final


# ----------------------------------------------------------------------
# pass 4: captions
# ----------------------------------------------------------------------
def build_captions(
    project: Project,
    timeline: Timeline,
    canvas: Canvas,
    duration: float,
    to_render: Callable[[float], float],
) -> tuple[Path | None, Path | None]:
    """Write the burned-in ASS and the uploadable SRT.

    Args:
        project: Owning project.
        timeline: Timeline supplying the caption track.
        canvas: Output geometry (the ASS ``PlayRes``).
        duration: Programme length in render seconds.
        to_render: Timeline-to-render time map.

    Returns:
        ``(captions.ass or None, captions.srt or None)``.
    """
    settings = project.settings
    shifted: list[Caption] = []
    for cue in timeline.tracks.captions:
        moved = cue.model_copy(update={"at": to_render(cue.at), "end": to_render(cue.end)})
        shifted.append(moved)

    ass_path: Path | None = None
    if shifted:
        document = captions_mod.build_ass(
            shifted,
            canvas.width,
            canvas.height,
            settings.caption_styles,
            settings.caption_font(),
            duration=duration,
        )
        ass_path = project.renders_dir / "captions.ass"
        ass_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path.write_text(document, encoding="utf-8")
        log.info("captions: %d cue(s) -> %s", len(shifted), ass_path.name)

    srt_path: Path | None = None
    cues = captions_mod.srt_from_transcripts(project, timeline, settings)
    if cues:
        mapped = [
            captions_mod.Cue(to_render(c.start), to_render(c.end), c.text) for c in cues
        ]
        srt_path = captions_mod.write_srt(mapped, project.exports_dir / "captions.srt")
        log.info("subtitles: %d cue(s) -> %s", len(mapped), srt_path.name)
    return ass_path, srt_path


# ----------------------------------------------------------------------
# job progress
# ----------------------------------------------------------------------
class _Job:
    """Writes ``jobs/render_<mode>.json`` as the render advances."""

    def __init__(self, project: Project, mode: str, total: float) -> None:
        self.path = project.jobs_dir / f"render_{mode}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = max(0.001, float(total))
        self.data: dict[str, Any] = {
            "job": f"render_{mode}",
            "status": "running",
            "percent": 0.0,
            "step": "starting",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished": None,
            "output": None,
        }
        self.write()

    def write(self) -> None:
        """Atomically persist the job document."""
        payload = json.dumps(self.data, indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".job-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, self.path)

    def step(self, name: str, percent: float | None = None) -> None:
        """Record the current step (and optionally the completion fraction)."""
        self.data["step"] = name
        if percent is not None:
            self.data["percent"] = round(max(0.0, min(100.0, percent)), 2)
        self.write()

    def finish(self, output: Path | None, status: str = "done", error: str = "") -> None:
        """Close the job document."""
        self.data["status"] = status
        self.data["percent"] = 100.0 if status == "done" else self.data["percent"]
        self.data["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.data["output"] = str(output) if output else None
        if error:
            self.data["error"] = error[:2000]
        self.write()


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------
def render(
    project: Project,
    preview: bool = False,
    master: bool = False,
    timeline_path: Path | str | None = None,
    out: Path | str | None = None,
    progress_cb: Callable[[Progress], None] | None = None,
    force_segments: bool = False,
) -> Path:
    """Render ``plan/timeline.json`` to a preview or a YouTube master.

    Args:
        project: Project to render.
        preview: Fast 720p render with the hardware encoder and single-pass
            loudness normalization.
        master: Full-quality ``libx264 -crf 18 -preset slow`` export with
            two-pass loudness normalization. Ignored when ``preview`` is set;
            when neither flag is given a preview is rendered.
        timeline_path: Timeline to render (default ``plan/timeline.json``).
        out: Explicit output path.
        progress_cb: Receives :class:`~ytedit.media.ffmpeg.Progress` blocks from
            the final encode. When omitted a rich progress bar is drawn.
        force_segments: Ignore the segment cache.

    Returns:
        The rendered file.

    Raises:
        RenderError: When the timeline is missing, invalid or empty.
    """
    mode = "preview" if preview or not master else "master"
    source_timeline = Path(timeline_path) if timeline_path else project.timeline_file
    if not source_timeline.exists():
        raise RenderError(f"no timeline at {source_timeline} — run the plan stage first")

    timeline = Timeline.load(source_timeline)
    issues = timeline.validate(project)
    if issues:
        for issue in issues:
            log.error("timeline: %s", issue)
        raise RenderError(f"timeline has {len(issues)} issue(s); refusing to render")
    if not timeline.tracks.video:
        raise RenderError("timeline has no video segments")

    canvas = canvas_for(timeline, mode == "preview")
    duration = render_duration(timeline)
    to_render = build_time_map(timeline)
    timeline_duration = timeline.duration()
    if abs(duration - timeline_duration) > 1e-3:
        log.info(
            "render duration %.3fs differs from timeline duration %.3fs "
            "(fade transitions overlap); positions remapped",
            duration, timeline_duration,
        )

    project.ensure_dirs()
    project.set_stage("render", "running")
    job = _Job(project, mode, duration)
    log.info(
        "render [stage]%s[/] · %s @ %d fps · %.2fs · %d segment(s)",
        mode, canvas.size, canvas.fps, duration, len(timeline.tracks.video),
    )

    try:
        # -- pass 1: segments -----------------------------------------
        job.step("segments", 0.0)
        segments: list[Path] = []
        for index, seg in enumerate(timeline.tracks.video):
            log.info("segment %s (%s %.2f-%.2f)", seg.id, seg.clip, seg.in_, seg.out)
            segments.append(
                render_segment(project, timeline, seg, canvas, mode, force=force_segments)
            )
            job.step("segments", 40.0 * (index + 1) / len(timeline.tracks.video))

        # -- pass 2: join ---------------------------------------------
        job.step("join", 40.0)
        program_video, program_audio = join_segments(
            project, timeline, segments, canvas, mode
        )

        # -- pass 3: audio bus ----------------------------------------
        job.step("audio", 55.0)
        final_audio = build_audio_bus(
            project, timeline, program_audio, duration, to_render, two_pass=(mode == "master")
        )

        # -- pass 4: captions -----------------------------------------
        job.step("captions", 70.0)
        ass_path, _srt = build_captions(project, timeline, canvas, duration, to_render)

        # -- pass 5: final encode -------------------------------------
        job.step("encode", 75.0)
        if out is not None:
            target = Path(out)
        elif mode == "preview":
            target = project.renders_dir / "preview.mp4"
        else:
            target = project.exports_dir / f"master_{canvas.height}p.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)

        args: list[Any] = ["-i", str(program_video), "-i", str(final_audio)]
        vfilters: list[str] = []
        if ass_path is not None:
            fonts_dir = project.settings.caption_styles.get("font", {}).get("fonts_dir")
            ass_filter = f"ass=filename={escape_filter_path(ass_path)}"
            if fonts_dir:
                ass_filter += f":fontsdir={escape_filter_path(fonts_dir)}"
            vfilters.append(ass_filter)
        vfilters.append(color_tag_filter(project.settings))
        args += ["-vf", _join(vfilters)]
        args += [
            "-map", "0:v:0",
            "-map", "1:a:0",
            *video_encoder_args(project.settings, mode, canvas.fps),
            *audio_encoder_args(project.settings),
            "-movflags", "+faststart",
            "-t", f"{duration:.6f}",
            str(target),
        ]

        sink, close = _progress_sink(progress_cb, job, duration)
        try:
            ff(*args, progress_cb=sink, total_duration=duration)
        finally:
            close()

        job.finish(target)
        project.set_stage(
            "render", "done",
            output=project.rel(target),
            preset=mode,
            duration=duration,
            canvas=canvas.size,
            fps=canvas.fps,
        )
        log.info("render done: [stage]%s[/] (%.2fs)", target, duration)
        return target
    except (FFmpegError, RenderError, OSError, ValueError) as exc:
        job.finish(None, status="error", error=str(exc))
        project.set_stage("render", "error", error=str(exc)[:2000], preset=mode)
        raise


def _progress_sink(
    progress_cb: Callable[[Progress], None] | None, job: _Job, duration: float
) -> tuple[Callable[[Progress], None], Callable[[], None]]:
    """Return ``(callback, close)`` bridging ffmpeg progress to the caller.

    When the caller supplied no callback a rich progress bar is drawn on the
    shared console; either way the job file keeps advancing.
    """
    if progress_cb is not None:
        def forward(p: Progress) -> None:
            job.step("encode", 75.0 + 25.0 * (p.percent or 0.0))
            progress_cb(p)

        return forward, lambda: None

    from rich.progress import BarColumn, Progress as RichProgress, TaskProgressColumn, TextColumn

    bar = RichProgress(
        TextColumn("[stage]encode[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[speed]}"),
        console=console,
        transient=True,
    )
    bar.start()
    task = bar.add_task("encode", total=100.0, speed="")

    def sink(p: Progress) -> None:
        percent = 100.0 * (p.percent or 0.0)
        bar.update(task, completed=percent, speed=f"{p.speed:.1f}x" if p.speed else "")
        job.step("encode", 75.0 + 0.25 * percent)

    return sink, bar.stop
