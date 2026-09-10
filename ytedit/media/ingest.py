"""Ingest: register raw drops and build every derived asset a clip needs.

For each file in ``input/`` the stage produces

* ``media/sources/<id>.mp4`` — the mezzanine: either a plain remux of a source
  that already matches the project ``format`` (see :func:`mezzanine_mode`) or a
  normalized re-encode (CFR, rotation baked, HDR tonemapped to bt709), always
  with a stereo 48 kHz audio track,
* ``media/proxies/<id>.mp4`` — 720p H.264 for the browser editor and
  ``render --draft``; only built when ``ingest.proxies`` is on or
  :func:`ensure_proxies` asks for it,
* ``media/audio/<id>.wav`` — mono 48 kHz PCM for STT and analysis,
* ``media/peaks/<id>.json`` — pre-computed waveform peaks for wavesurfer,
* ``media/thumbs/<id>.jpg`` — poster frame,
* ``media/thumbs/frames/<id>/NNN.jpg`` — sampled frames for the vision model.

Everything is idempotent: existing outputs are skipped unless ``force=True``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rich.table import Table

from ..config import OutputFormat
from ..log import console, get_logger
from ..project import Project
from .ffmpeg import FFMPEG, FFmpegError, ff, has_encoder
from .color import tonemap_chain
from .frames import sample_frames, save_frames
from .probe import MediaInfo, probe

log = get_logger(__name__)


@dataclass
class IngestResult:
    """Outcome of ingesting one input file.

    Attributes:
        clip_id: Assigned clip id (``c001``).
        source_file: Project-relative path of the raw input.
        status: ``done`` | ``skipped`` | ``error`` | ``registered``.
        steps: Names of the steps that actually ran.
        error: Error text when ``status == "error"``.
        info: The probe result, when probing succeeded.
        mezzanine: How the mezzanine was built, when it was built in this run.
    """

    clip_id: str
    source_file: str
    status: str = "done"
    steps: list[str] = None  # type: ignore[assignment]
    error: str = ""
    info: MediaInfo | None = None
    mezzanine: MezzanineMode | None = None

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = []


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------
def discover_inputs(project: Project) -> list[Path]:
    """List the raw files in ``input/`` that ingest understands.

    Args:
        project: Project to scan.

    Returns:
        Paths in the project's ``input/`` directory, unsorted.
    """
    cfg = project.settings
    exts = {e.lower() for e in cfg.get("ingest.extensions", [])}
    exts |= {e.lower() for e in cfg.get("ingest.still_extensions", [])}
    if not project.input_dir.is_dir():
        return []
    return [
        p
        for p in project.input_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts
    ]


def _creation_key(path: Path, info: MediaInfo | None) -> tuple[float, str]:
    """Sort key: recording time (metadata, then mtime), then filename."""
    if info and info.creation_time:
        text = info.creation_time.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (dt.timestamp(), path.name)
        except ValueError:
            pass
    return (path.stat().st_mtime, path.name)


def _recorded_at(path: Path, info: MediaInfo | None) -> str:
    """Return an ISO-8601 recording timestamp for a clip."""
    if info and info.creation_time:
        return info.creation_time
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


# ----------------------------------------------------------------------
# compatibility rule
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MezzanineMode:
    """How ingest must build one clip's mezzanine.

    Attributes:
        mode: ``"copy"`` (remux, stream copy) or ``"encode"``.
        reason: Why the clip must be encoded; ``""`` when copying.
    """

    mode: str
    reason: str = ""

    @property
    def is_copy(self) -> bool:
        """True when the mezzanine is a plain remux of the source."""
        return self.mode == "copy"


def mezzanine_mode(info: MediaInfo, fmt: OutputFormat) -> MezzanineMode:
    """Decide whether a source can become the mezzanine by remux alone.

    A clip that already *is* what the project renders costs nothing to remux
    and gains nothing from a CRF 16 re-encode — a 1080p30 H.264 source at
    1.7 Mb/s measured 10x bigger for the same pixels. Everything else still
    goes through :func:`normalize`'s encode path.

    The checks run in a fixed order so the recorded reason is deterministic
    (it ends up in ``state.json`` and in the ingest summary).

    Args:
        info: Probe result for the raw source.
        fmt: The project's output format (also its compatibility target).

    Returns:
        A :class:`MezzanineMode`; ``reason`` names the first failed check.
    """
    if info.codec != fmt.codec:
        return MezzanineMode("encode", f"codec {info.codec}")
    if info.pix_fmt != fmt.pix_fmt:
        return MezzanineMode("encode", f"pix_fmt {info.pix_fmt}")
    if info.rotation != 0:
        # A remux keeps the display matrix in metadata instead of baking it
        # into the pixels, and the whole render path assumes the mezzanine has
        # rotation already baked (see the `rotation=0` argument render.py
        # passes to `source_chain` in `render_segment`). A rotated source must
        # therefore be re-encoded, autorotated on decode.
        return MezzanineMode("encode", f"rotation {info.rotation}")
    if (info.display_width, info.display_height) not in (
        (fmt.width, fmt.height),
        # A vertical clip is kept at its own size on purpose: the renderer's
        # `blur-fill` fit is what puts it on the canvas, so re-encoding it to
        # 16:9 here would bake in a decision the cut is allowed to change.
        (fmt.height, fmt.width),
    ):
        return MezzanineMode("encode", f"size {info.display_width}x{info.display_height}")
    if info.vfr:
        return MezzanineMode("encode", "vfr")
    if info.hdr is not None:
        return MezzanineMode("encode", f"hdr {info.hdr}")
    if abs(info.fps - fmt.fps) > 0.05:
        return MezzanineMode("encode", f"fps {info.fps:g}")
    return MezzanineMode("copy", "")


# ----------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------
def normalize(project: Project, src: Path, info: MediaInfo, out: Path) -> MezzanineMode:
    """Write the mezzanine for a clip, remuxing it when that is enough.

    A source that already matches the project ``format``
    (:func:`mezzanine_mode`) is stream-copied: seconds instead of minutes, and
    the same bytes instead of a CRF 16 blow-up. Anything else is re-encoded —
    display rotation baked in (ffmpeg autorotates; ``-noautorotate`` is never
    passed), CFR at the rounded frame rate, HDR tonemapped to bt709 SDR, the
    original resolution kept.

    Either way the mezzanine comes out with a stereo 48 kHz audio track
    (silent when the source has none), because every downstream stage — the
    work WAV, the segment pass, the mix — assumes one is there.

    Args:
        project: Owning project (supplies encoding settings).
        src: Raw input file.
        info: Probe result for ``src``.
        out: Destination ``media/sources/<id>.mp4``.

    Returns:
        The :class:`MezzanineMode` that was applied.
    """
    cfg = project.settings
    enc = cfg.encoding("mezzanine")
    audio_cfg = cfg.section("audio")
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    channels = int(audio_cfg.get("channels", 2))
    decision = mezzanine_mode(info, cfg.format)

    args: list[Any] = ["-i", str(src)]
    if not info.has_audio:
        args += [
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
        ]
    args += ["-map", "0:v:0", "-map", "1:a:0" if not info.has_audio else "0:a:0?"]

    if decision.is_copy:
        # Video is bit-identical, so the only question left is the audio. An
        # MP4 will not take an arbitrary source codec, and the stereo/48 kHz
        # mezzanine invariant has to hold whatever the phone recorded, so the
        # track is copied only when it is already exactly what we would write.
        # Re-encoding it is cheap next to a video pass — this is still a remux.
        copy_audio = (
            info.has_audio
            and info.audio_codec == "aac"
            and info.audio_channels == channels
            and info.audio_sample_rate == sample_rate
        )
        args += ["-c:v", "copy"]
        if copy_audio:
            args += ["-c:a", "copy"]
        else:
            args += [
                "-c:a", enc.get("acodec", "aac"),
                "-b:a", str(enc.get("abitrate", "256k")),
                "-ar", str(sample_rate),
                "-ac", str(channels),
            ]
        args += ["-movflags", "+faststart"]
    else:
        fps = info.target_fps
        chain = ",".join(c for c in (tonemap_chain(info.hdr, cfg), f"fps={fps}") if c)
        args += [
            "-vf", chain,
            "-fps_mode", "cfr",
            "-r", str(fps),
            "-c:v", enc.get("vcodec", "libx264"),
            "-crf", str(enc.get("crf", 16)),
            "-preset", enc.get("preset", "fast"),
            "-pix_fmt", enc.get("pix_fmt", "yuv420p"),
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-c:a", enc.get("acodec", "aac"),
            "-b:a", str(enc.get("abitrate", "256k")),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-movflags", "+faststart",
        ]

    if not info.has_audio:
        args += ["-shortest"]
    args += [str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    ff(*args)
    return decision


def make_proxy(project: Project, src: Path, out: Path) -> None:
    """Write the 720p browser proxy, preferring the hardware encoder.

    Args:
        project: Owning project.
        src: Normalized mezzanine.
        out: Destination ``media/proxies/<id>.mp4``.
    """
    cfg = project.settings
    proxy = cfg.section("proxy")
    width = int(proxy.get("width", 1280))
    height = int(proxy.get("height", 720))
    scale = (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease"
        ":force_divisible_by=2:flags=bilinear"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    common = [
        "-i", str(src), "-vf", scale,
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
    ]
    if has_encoder("h264_videotoolbox"):
        try:
            ff(
                *common,
                "-c:v", "h264_videotoolbox",
                "-q:v", str(proxy.get("videotoolbox_q", 55)),
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",
                str(out),
            )
            return
        except FFmpegError as exc:
            log.warning("videotoolbox proxy failed, falling back to libx264: %s", exc.returncode)
    ff(
        *common,
        "-c:v", "libx264",
        "-crf", str(proxy.get("x264_crf", 23)),
        "-preset", proxy.get("x264_preset", "veryfast"),
        "-pix_fmt", "yuv420p",
        str(out),
    )


def _want_proxies(project: Project, proxies: bool | None) -> bool:
    """Resolve the proxy flag: ``None`` means "use ``ingest.proxies``"."""
    if proxies is None:
        return bool(project.settings.get("ingest.proxies", False))
    return bool(proxies)


def ensure_proxies(project: Project, clip_ids: Iterable[str] | None = None) -> list[str]:
    """Build any missing 720p proxy from the mezzanine.

    Ingest does not build proxies by default (see ``ingest.proxies``), but the
    web editor streams them, so ``ytedit serve`` calls this first.

    Args:
        project: Project to fill in.
        clip_ids: Restrict to these clips; ``None`` means every video clip in
            the registry.

    Returns:
        The clip ids whose proxy was built by this call, in registry order.
    """
    wanted = set(clip_ids) if clip_ids is not None else None
    built: list[str] = []
    for clip in project.clips_in_order():
        clip_id = str(clip.get("id") or "")
        if not clip_id or (wanted is not None and clip_id not in wanted):
            continue
        if clip.get("kind") == "still":
            continue
        source = project.source_path(clip_id)
        proxy = project.proxy_path(clip_id)
        if proxy.exists() or not source.exists():
            continue
        try:
            make_proxy(project, source, proxy)
        except (FFmpegError, OSError) as exc:
            log.error("proxy failed for %s: %s", clip_id, exc)
            continue
        project.add_clip({"id": clip_id, "proxy": project.rel(proxy)})
        built.append(clip_id)
    return built


def extract_audio(project: Project, src: Path, out: Path) -> None:
    """Extract the mono 48 kHz work WAV used by STT, analysis and peaks."""
    work = project.settings.section("audio").get("work_wav", {})
    out.parent.mkdir(parents=True, exist_ok=True)
    ff(
        "-i", str(src),
        "-vn",
        "-ac", str(work.get("channels", 1)),
        "-ar", str(work.get("sample_rate", 48000)),
        "-c:a", work.get("codec", "pcm_s16le"),
        str(out),
    )


def compute_peaks(wav: Path, peaks_per_second: float = 16.6667) -> dict[str, Any]:
    """Compute min/max waveform peaks from a WAV file.

    Decodes the file to float32 through ffmpeg and reduces it to
    ``peaks_per_second`` min/max pairs, the shape wavesurfer expects.

    Args:
        wav: Mono WAV file.
        peaks_per_second: Resolution of the peak data.

    Returns:
        ``{"sample_rate": int, "peaks_per_second": float, "duration": float,
        "peaks": [min0, max0, min1, max1, ...]}``.
    """
    info = probe_audio_rate(wav)
    sample_rate = info or 48000
    block = max(1, int(round(sample_rate / max(0.1, peaks_per_second))))
    argv = [
        FFMPEG, "-hide_banner", "-nostdin", "-v", "error",
        "-i", str(wav), "-map", "0:a:0", "-ac", "1",
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    peaks: list[float] = []
    total = 0
    carry = np.empty(0, dtype=np.float32)
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(block * 4 * 64)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.float32)
            total += chunk.size
            data = np.concatenate([carry, chunk]) if carry.size else chunk
            usable = (data.size // block) * block
            if usable:
                view = data[:usable].reshape(-1, block)
                mins = view.min(axis=1)
                maxs = view.max(axis=1)
                for lo, hi in zip(mins, maxs):
                    peaks.append(round(float(lo), 4))
                    peaks.append(round(float(hi), 4))
            carry = data[usable:].copy()
        proc.wait()
    if carry.size:
        peaks.append(round(float(carry.min()), 4))
        peaks.append(round(float(carry.max()), 4))
    return {
        "sample_rate": sample_rate,
        "peaks_per_second": round(sample_rate / block, 4),
        "samples_per_peak": block,
        "duration": round(total / sample_rate, 3) if sample_rate else 0.0,
        "peaks": peaks,
    }


def probe_audio_rate(path: Path) -> int:
    """Return the sample rate of the first audio stream, 0 when absent."""
    try:
        info = probe(path, detect_vfr=False)
        return info.audio_sample_rate
    except (ValueError, FileNotFoundError, FFmpegError):
        from .ffmpeg import ffprobe_json  # local import: stills have no video stream

        data = ffprobe_json(path)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return int(stream.get("sample_rate") or 0)
        return 0


def make_poster(src: Path, out: Path, at: float = 1.0) -> None:
    """Grab a poster frame at ``at`` seconds (falls back to the first frame)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ff("-ss", f"{at:.3f}", "-i", str(src), "-frames:v", "1", "-q:v", "3", str(out))
    except FFmpegError:
        ff("-i", str(src), "-frames:v", "1", "-q:v", "3", str(out))
    if not out.exists():
        ff("-i", str(src), "-frames:v", "1", "-q:v", "3", str(out))


def make_frames(project: Project, src: Path, out_dir: Path) -> int:
    """Sample frames for the vision model and write them plus an index.

    Args:
        project: Owning project (supplies ``ingest.frames.*``).
        src: Proxy video to sample.
        out_dir: ``media/thumbs/frames/<id>``.

    Returns:
        Number of frames written.
    """
    cfg = project.settings.get("ingest.frames", {})
    frames = sample_frames(
        src,
        interval=float(cfg.get("interval", 3.0)),
        max_frames=int(cfg.get("max_frames", 40)),
        width=int(cfg.get("width", 640)),
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    save_frames(frames, out_dir)
    (out_dir / "index.json").write_text(
        json.dumps({"times": [t for t, _ in frames]}, indent=2), encoding="utf-8"
    )
    return len(frames)


# ----------------------------------------------------------------------
# per-clip driver
# ----------------------------------------------------------------------
def ingest_clip(
    project: Project,
    clip_id: str,
    src: Path,
    info: MediaInfo,
    force: bool = False,
    proxies: bool | None = None,
) -> IngestResult:
    """Build every derived asset for one clip.

    Args:
        project: Owning project.
        clip_id: Clip id such as ``c001``.
        src: Raw input file.
        info: Probe result for ``src``.
        force: Rebuild outputs that already exist.
        proxies: Build the 720p browser proxy; ``None`` uses ``ingest.proxies``.
            With proxies off the poster and vision frames are sampled from the
            mezzanine instead.

    Returns:
        An :class:`IngestResult` describing what ran.
    """
    result = IngestResult(clip_id=clip_id, source_file=project.rel(src), info=info)
    want_proxy = _want_proxies(project, proxies)
    project.set_clip_stage(clip_id, "ingest", "running")
    try:
        source = project.source_path(clip_id)
        proxy = project.proxy_path(clip_id)
        wav = project.audio_path(clip_id)
        peaks = project.peaks_path(clip_id)
        poster = project.poster_path(clip_id)
        frames_dir = project.clip_frames_dir(clip_id)

        mezz: MezzanineMode | None = None
        if force or not source.exists():
            mezz = normalize(project, src, info, source)
            result.steps.append("normalize")
            result.mezzanine = mezz
        if want_proxy and (force or not proxy.exists()):
            make_proxy(project, source, proxy)
            result.steps.append("proxy")
        # The poster and the vision frames are read off the proxy when there is
        # one (cheap to decode) and off the mezzanine otherwise.
        stills_from = proxy if want_proxy or proxy.exists() else source
        if force or not wav.exists():
            extract_audio(project, source, wav)
            result.steps.append("audio")
        if force or not peaks.exists():
            per_minute = float(project.settings.get("ingest.peaks_per_minute", 1000))
            data = compute_peaks(wav, peaks_per_second=per_minute / 60.0)
            peaks.parent.mkdir(parents=True, exist_ok=True)
            peaks.write_text(json.dumps(data), encoding="utf-8")
            result.steps.append("peaks")
        if force or not poster.exists():
            make_poster(
                stills_from, poster, at=float(project.settings.get("ingest.poster_at", 1.0))
            )
            result.steps.append("poster")
        frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
        if force or frame_count == 0:
            frame_count = make_frames(project, stills_from, frames_dir)
            result.steps.append("frames")

        normalized_info = probe(source, detect_vfr=False)
        record: dict[str, Any] = {
            "id": clip_id,
            "source_file": project.rel(src),
            "recorded_at": _recorded_at(src, info),
            "kind": info.kind,
            "duration": normalized_info.duration or info.duration,
            "width": info.display_width,
            "height": info.display_height,
            "fps": info.fps,
            "target_fps": info.target_fps,
            "vfr": info.vfr,
            "rotation": info.rotation,
            "orientation": info.orientation,
            "hdr": info.hdr,
            "codec": info.codec,
            "is_iphone": info.is_iphone,
            "has_audio": info.has_audio,
            "audio_channels": info.audio_channels,
            "audio_sample_rate": info.audio_sample_rate,
            "size_bytes": info.size_bytes,
            "normalized": project.rel(source),
            "audio": project.rel(wav),
            "peaks": project.rel(peaks),
            "poster": project.rel(poster),
            "frames": project.rel(frames_dir),
            "frames_count": frame_count,
        }
        if proxy.exists():
            record["proxy"] = project.rel(proxy)
        if mezz is not None:
            # Only set when the mezzanine was actually (re)built — on a cached
            # clip whatever the registry already says still describes the file
            # on disk, and add_clip merges, so leaving the keys out keeps it.
            record["mezzanine"] = mezz.mode
            record["mezzanine_reason"] = mezz.reason
        project.add_clip(record)
        project.set_clip_stage(clip_id, "ingest", "done")
        result.status = "done" if result.steps else "skipped"
    except (FFmpegError, OSError, ValueError) as exc:
        log.error("ingest failed for %s (%s): %s", clip_id, src.name, exc)
        result.status = "error"
        result.error = str(exc)
        # Keep the probe facts in the registry even when a derived step failed,
        # so status/QC can still show what the clip is.
        project.add_clip(
            {
                "id": clip_id,
                "source_file": project.rel(src),
                "recorded_at": _recorded_at(src, info),
                "kind": info.kind,
                "duration": info.duration,
                "width": info.display_width,
                "height": info.display_height,
                "fps": info.fps,
                "rotation": info.rotation,
                "orientation": info.orientation,
                "hdr": info.hdr,
                "codec": info.codec,
                "is_iphone": info.is_iphone,
                "has_audio": info.has_audio,
                "size_bytes": info.size_bytes,
            }
        )
        project.set_clip_stage(clip_id, "ingest", "error", error=str(exc)[:2000])
    return result


def register_still(project: Project, clip_id: str, src: Path, info: MediaInfo | None) -> IngestResult:
    """Register a still image so a later stage can turn it into a clip.

    Stills are only recorded in the registry (``kind: "still"``) with the
    configured default duration; no media is derived yet.
    """
    duration = float(project.settings.get("ingest.still_duration", 5.0))
    width = info.display_width if info else 0
    height = info.display_height if info else 0
    project.add_clip(
        {
            "id": clip_id,
            "source_file": project.rel(src),
            "recorded_at": _recorded_at(src, info),
            "kind": "still",
            "duration": duration,
            "width": width,
            "height": height,
            "fps": 0.0,
            "vfr": False,
            "rotation": info.rotation if info else 0,
            "orientation": info.orientation if info else "horizontal",
            "hdr": None,
            "has_audio": False,
            "size_bytes": src.stat().st_size,
        }
    )
    project.set_clip_stage(clip_id, "ingest", "registered")
    return IngestResult(clip_id=clip_id, source_file=project.rel(src), status="registered", info=info)


# ----------------------------------------------------------------------
# stage entry point
# ----------------------------------------------------------------------
def ingest(
    project: Project,
    force: bool = False,
    show_table: bool = True,
    proxies: bool | None = None,
) -> list[IngestResult]:
    """Ingest every file in ``input/`` into the clip registry.

    Files are probed serially (cheap) and ordered by recording time, then the
    heavy per-clip work runs in a small thread pool while the registry order is
    preserved. Re-running is safe: existing outputs are skipped unless
    ``force`` is set.

    Args:
        project: Project to ingest.
        force: Rebuild assets that already exist.
        show_table: Print the summary table to the console when done.
        proxies: Build 720p browser proxies; ``None`` uses ``ingest.proxies``
            (off by default — see :func:`ensure_proxies`).

    Returns:
        One :class:`IngestResult` per input file, in clip order.
    """
    project.ensure_dirs()
    inputs = discover_inputs(project)
    if not inputs:
        log.warning("no input files in %s", project.input_dir)
        project.set_stage("ingest", "done", clips=0)
        return []

    project.set_stage("ingest", "running")
    still_exts = {e.lower() for e in project.settings.get("ingest.still_extensions", [])}

    probed: list[tuple[Path, MediaInfo | None]] = []
    for path in inputs:
        try:
            probed.append((path, probe(path)))
        except (ValueError, FFmpegError, FileNotFoundError) as exc:
            log.error("cannot probe %s: %s", path.name, exc)
            probed.append((path, None))
    probed.sort(key=lambda pair: _creation_key(pair[0], pair[1]))

    # Keep ids stable across re-runs by matching on the source file path.
    state = project.load_state()
    existing = {c.get("source_file"): cid for cid, c in state.get("clips", {}).items()}
    assignments: list[tuple[str, Path, MediaInfo | None]] = []
    used: set[str] = set(state.get("clips", {}))
    counter = 1
    for order, (path, info) in enumerate(probed, start=1):
        rel = project.rel(path)
        clip_id = existing.get(rel)
        if clip_id is None:
            while f"c{counter:03d}" in used:
                counter += 1
            clip_id = f"c{counter:03d}"
            used.add(clip_id)
        assignments.append((clip_id, path, info))
        project.add_clip({"id": clip_id, "source_file": rel, "order": order})

    results: list[IngestResult] = []
    workers = int(project.settings.get("ingest.workers", 3))

    def _run(item: tuple[str, Path, MediaInfo | None]) -> IngestResult:
        clip_id, path, info = item
        if info is None:
            project.set_clip_stage(clip_id, "ingest", "error", error="probe failed")
            return IngestResult(clip_id, project.rel(path), "error", [], "probe failed", None)
        if path.suffix.lower() in still_exts or info.kind == "still":
            return register_still(project, clip_id, path, info)
        return ingest_clip(project, clip_id, path, info, force=force, proxies=proxies)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_run, assignments))

    errors = [r for r in results if r.status == "error"]
    modes = mezzanine_summary(project)
    project.set_stage(
        "ingest",
        "error" if errors else "done",
        clips=len(results),
        cost_usd=0.0,
        mezzanine_copy=modes["copy"],
        mezzanine_encode=modes["encode"],
        encode_reasons=modes["reasons"],
        error="; ".join(f"{r.clip_id}: {r.error}" for r in errors)[:2000] if errors else None,
    )
    log.info("ingest finished: %d clip(s), %d error(s)", len(results), len(errors))
    if show_table:
        console.print(clips_table(project))
        console.print(mezzanine_line(modes))
    return results


def mezzanine_summary(project: Project) -> dict[str, Any]:
    """Count how the registry's mezzanines were built.

    Returns:
        ``{"copy": int, "encode": int, "reasons": {first word: count}}`` —
        encode reasons are grouped by their first word (``vfr``, ``hdr``,
        ``size``, ...) so a summary stays one line however many clips there are.
    """
    copies = encodes = 0
    reasons: dict[str, int] = {}
    for clip in project.clips_in_order():
        mode = str(clip.get("mezzanine") or "")
        if mode == "copy":
            copies += 1
        elif mode == "encode":
            encodes += 1
            head = str(clip.get("mezzanine_reason") or "other").split(" ")[0] or "other"
            reasons[head] = reasons.get(head, 0) + 1
    return {"copy": copies, "encode": encodes, "reasons": reasons}


def mezzanine_line(summary: dict[str, Any]) -> str:
    """Render :func:`mezzanine_summary` as the one-line ingest footer."""
    reasons = summary.get("reasons") or {}
    detail = ", ".join(f"{name} {count}" for name, count in sorted(reasons.items()))
    line = f"mezzanine: {summary['copy']} copy, {summary['encode']} encode"
    return f"{line} ({detail})" if detail else line


def clips_table(project: Project, title: str | None = None) -> Table:
    """Build a rich table of the clip registry for the CLI and ingest summary.

    Args:
        project: Project to summarize.
        title: Optional table title.

    Returns:
        A ready-to-print ``rich.table.Table``.
    """
    table = Table(title=title or f"{project.slug} — clips", header_style="bold cyan")
    for column, justify in (
        ("id", "left"), ("source", "left"), ("dur", "right"), ("size", "right"),
        ("fps", "right"), ("rot", "right"), ("orient", "left"), ("hdr", "left"),
        ("audio", "left"), ("mezz", "left"), ("frames", "right"), ("ingest", "left"),
    ):
        table.add_column(column, justify=justify)

    for clip in project.clips_in_order():
        duration = float(clip.get("duration") or 0.0)
        size_mb = float(clip.get("size_bytes") or 0) / 1e6
        status = str(clip.get("stages", {}).get("ingest", "pending"))
        style = {"done": "green", "error": "red", "registered": "yellow"}.get(status, "dim")
        table.add_row(
            str(clip.get("id", "")),
            Path(str(clip.get("source_file", ""))).name,
            f"{duration:.1f}s",
            f"{size_mb:.1f}MB",
            f"{float(clip.get('fps') or 0):.2f}",
            str(clip.get("rotation", 0)),
            str(clip.get("orientation", "")),
            str(clip.get("hdr") or "-"),
            "yes" if clip.get("has_audio") else "no",
            str(clip.get("mezzanine") or "") or "[dim]-[/]",
            str(clip.get("frames_count", 0)),
            f"[{style}]{status}[/]",
        )
    return table


def summarize(results: Iterable[IngestResult]) -> dict[str, int]:
    """Count results by status (for logs and tests)."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts
