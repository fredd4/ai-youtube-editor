"""``MediaInfo``: everything ingest and render need to know about a source file."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..log import get_logger
from .ffmpeg import ffprobe_json, ffprobe_packets

log = get_logger(__name__)

#: ``color_transfer`` values that mean HDR.
HDR_TRANSFERS: dict[str, str] = {
    "arib-std-b67": "hlg",
    "smpte2084": "pq",
}

#: Frame rates we snap to when normalizing (NTSC rates -> integers).
STANDARD_FPS: tuple[float, ...] = (23.976, 24.0, 25.0, 29.97, 30.0, 47.952, 48.0, 50.0, 59.94, 60.0)

_BITDEPTH_RE = re.compile(r"p0?(\d{1,2})(le|be)$")

#: Semi-planar formats whose name does not encode the depth the usual way.
_PACKED_DEPTHS: dict[str, int] = {
    "p010": 10, "p012": 12, "p016": 16,
    "p210": 10, "p212": 12, "p216": 16,
    "p410": 10, "p412": 12, "p416": 16,
}


def _fraction(value: Any) -> float:
    """Parse an ffprobe rational like ``30000/1001`` into a float."""
    if not value:
        return 0.0
    try:
        if isinstance(value, str) and "/" in value:
            num, _, den = value.partition("/")
            if float(den) == 0:
                return 0.0
            return float(Fraction(int(num), int(den)))
        return float(value)
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def round_fps(fps: float) -> int:
    """Round a frame rate to the nearest sane CFR integer.

    ``29.97 -> 30``, ``23.976 -> 24``, ``59.94 -> 60``. Non-standard rates are
    rounded to the nearest integer (minimum 1).

    Args:
        fps: Measured frame rate.

    Returns:
        The integer frame rate to normalize to.
    """
    if fps <= 0:
        return 30
    nearest = min(STANDARD_FPS, key=lambda s: abs(s - fps))
    if abs(nearest - fps) <= 0.15:
        return int(round(nearest))
    return max(1, int(round(fps)))


def bit_depth_from_pix_fmt(pix_fmt: str | None) -> int:
    """Infer the bit depth from a pixel format name (``yuv420p10le`` -> 10)."""
    if not pix_fmt:
        return 8
    for prefix, depth in _PACKED_DEPTHS.items():
        if pix_fmt.startswith(prefix):
            return depth
    match = _BITDEPTH_RE.search(pix_fmt)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover
            return 8
    return 8


def _rotation_from_stream(stream: dict[str, Any]) -> int:
    """Extract display rotation in degrees, normalized to 0/90/180/270."""
    rotation = 0.0
    for side in stream.get("side_data_list", []) or []:
        if side.get("side_data_type") == "Display Matrix" or "rotation" in side:
            try:
                rotation = float(side.get("rotation", 0))
                break
            except (TypeError, ValueError):  # pragma: no cover
                rotation = 0.0
    else:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag is not None:
            try:
                rotation = float(tag)
            except (TypeError, ValueError):  # pragma: no cover
                rotation = 0.0
    # ffprobe reports -90 for "rotate 90 clockwise for display".
    return int(round(-rotation)) % 360 if rotation else 0


@dataclass
class MediaInfo:
    """Probed properties of one media file.

    Attributes:
        path: Absolute path of the probed file.
        kind: ``video`` or ``still``.
        container: Container format name from ffprobe.
        duration: Duration in seconds (0 for stills).
        width: Coded width, before rotation.
        height: Coded height, before rotation.
        fps: Effective frame rate (``avg_frame_rate``, falling back to ``r_frame_rate``).
        r_fps: ``r_frame_rate`` (the base/tick rate).
        avg_fps: ``avg_frame_rate`` (frames / duration).
        vfr: True when the file looks variable-frame-rate.
        rotation: Display rotation in degrees (0/90/180/270).
        orientation: ``vertical`` / ``horizontal`` / ``square``, after rotation.
        hdr: ``hlg`` / ``pq`` / ``None``.
        color_transfer: Raw ``color_transfer`` value.
        color_primaries: Raw ``color_primaries`` value.
        color_space: Raw ``color_space`` value.
        pix_fmt: Pixel format name.
        bit_depth: Bits per component derived from ``pix_fmt``.
        codec: Video codec name.
        has_audio: Whether an audio stream exists.
        audio_codec: Audio codec name.
        audio_channels: Audio channel count.
        audio_sample_rate: Audio sample rate in Hz.
        creation_time: ``format.tags.creation_time`` (ISO-8601) if present.
        make: Camera make tag (``com.apple.quicktime.make``).
        model: Camera model tag.
        size_bytes: File size on disk.
    """

    path: str
    kind: str = "video"
    container: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    r_fps: float = 0.0
    avg_fps: float = 0.0
    vfr: bool = False
    rotation: int = 0
    orientation: str = "horizontal"
    hdr: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_space: str | None = None
    pix_fmt: str | None = None
    bit_depth: int = 8
    codec: str = ""
    has_audio: bool = False
    audio_codec: str | None = None
    audio_channels: int = 0
    audio_sample_rate: int = 0
    creation_time: str | None = None
    make: str | None = None
    model: str | None = None
    size_bytes: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @property
    def display_width(self) -> int:
        """Width after applying rotation."""
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        """Height after applying rotation."""
        return self.width if self.rotation in (90, 270) else self.height

    @property
    def is_hdr(self) -> bool:
        """True when the source carries an HDR transfer function."""
        return self.hdr is not None

    @property
    def is_iphone(self) -> bool:
        """Heuristic: footage shot on an Apple device (skip sharpening)."""
        make = (self.make or "").lower()
        model = (self.model or "").lower()
        return "apple" in make or "iphone" in model or "ipad" in model

    @property
    def target_fps(self) -> int:
        """The CFR rate ingest should normalize this file to.

        For a VFR source ``avg_frame_rate`` is meaningless (it is just
        frames/duration), so the nominal ``r_frame_rate`` is used instead.
        """
        return round_fps(self.r_fps if self.vfr and self.r_fps > 0 else self.fps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize without the raw ffprobe payload."""
        data = asdict(self)
        data.pop("raw", None)
        data["display_width"] = self.display_width
        data["display_height"] = self.display_height
        return data


def _detect_vfr(path: Path | str, r_fps: float, avg_fps: float, duration: float) -> bool:
    """Decide whether a file is variable frame rate.

    Compares ``r_frame_rate`` with ``avg_frame_rate`` first (a cheap, usually
    sufficient signal), then falls back to the spread of packet PTS deltas.
    """
    if r_fps > 0 and avg_fps > 0:
        ratio = abs(r_fps - avg_fps) / max(r_fps, avg_fps)
        if ratio > 0.02 and duration > 0.5:
            packets = ffprobe_packets(path, count=200)
            times = sorted(
                float(p["pts_time"]) for p in packets if p.get("pts_time") not in (None, "N/A")
            )
            if len(times) > 12:
                deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
                if deltas:
                    med = statistics.median(deltas)
                    jitter = sum(1 for d in deltas if abs(d - med) > 0.25 * med)
                    return jitter > len(deltas) * 0.1
            return True
    return False


def probe(path: Path | str, detect_vfr: bool = True) -> MediaInfo:
    """Probe a media file (or a still image) into a :class:`MediaInfo`.

    Args:
        path: File to probe.
        detect_vfr: Run the extra packet-level VFR check when the cheap check is
            inconclusive.

    Returns:
        A populated :class:`MediaInfo`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no video stream.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    data = ffprobe_json(p)
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"{p}: no video stream")

    fmt_tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    stream_tags = {k.lower(): v for k, v in (video.get("tags") or {}).items()}
    tags = {**fmt_tags, **stream_tags}

    format_name = str(fmt.get("format_name", ""))
    is_still = video.get("codec_name") in ("mjpeg", "png", "bmp", "webp", "gif") and (
        "image2" in format_name or _fraction(fmt.get("duration")) == 0.0
    )

    r_fps = _fraction(video.get("r_frame_rate"))
    avg_fps = _fraction(video.get("avg_frame_rate"))
    fps = avg_fps or r_fps
    duration = _fraction(fmt.get("duration")) or _fraction(video.get("duration"))

    rotation = _rotation_from_stream(video)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    disp_w, disp_h = (height, width) if rotation in (90, 270) else (width, height)
    if disp_w == disp_h:
        orientation = "square"
    elif disp_h > disp_w:
        orientation = "vertical"
    else:
        orientation = "horizontal"

    transfer = video.get("color_transfer")
    primaries = video.get("color_primaries")
    hdr = HDR_TRANSFERS.get(str(transfer or "").lower())
    if hdr is None and str(primaries or "").lower() == "bt2020" and bit_depth_from_pix_fmt(
        video.get("pix_fmt")
    ) > 8:
        # bt2020 primaries with a missing/unknown transfer: treat as HLG.
        hdr = "hlg"

    info = MediaInfo(
        path=str(p.resolve()),
        kind="still" if is_still else "video",
        container=format_name,
        duration=round(duration, 3),
        width=width,
        height=height,
        fps=round(fps, 5),
        r_fps=round(r_fps, 5),
        avg_fps=round(avg_fps, 5),
        vfr=False,
        rotation=rotation,
        orientation=orientation,
        hdr=hdr,
        color_transfer=transfer,
        color_primaries=primaries,
        color_space=video.get("color_space"),
        pix_fmt=video.get("pix_fmt"),
        bit_depth=bit_depth_from_pix_fmt(video.get("pix_fmt")),
        codec=str(video.get("codec_name") or ""),
        has_audio=audio is not None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        audio_channels=int(audio.get("channels") or 0) if audio else 0,
        audio_sample_rate=int(audio.get("sample_rate") or 0) if audio else 0,
        creation_time=tags.get("creation_time"),
        make=tags.get("com.apple.quicktime.make") or tags.get("make"),
        model=tags.get("com.apple.quicktime.model") or tags.get("model"),
        size_bytes=int(fmt.get("size") or (p.stat().st_size if p.exists() else 0)),
        raw=data,
    )
    if not is_still and detect_vfr:
        info.vfr = _detect_vfr(p, r_fps, avg_fps, duration)
    return info
