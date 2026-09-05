"""Frame sampling and annotation for vision models.

Frames are extracted with a single ffmpeg pass into an MJPEG pipe and split on
JPEG start-of-image markers. :func:`annotate_frame` draws the percentage grid
that lets an LLM answer positional questions (``subject_x_pct``) reliably — the
trick that worked in amazonia-studio.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from ..log import get_logger
from .ffmpeg import FFMPEG, FFmpegError

log = get_logger(__name__)

_SOI = b"\xff\xd8\xff"

#: Fonts tried, in order, for frame annotations.
_ANNOTATION_FONTS: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def _split_mjpeg(blob: bytes) -> list[bytes]:
    """Split a concatenated MJPEG byte stream into individual JPEG images."""
    offsets: list[int] = []
    pos = blob.find(_SOI)
    while pos != -1:
        offsets.append(pos)
        pos = blob.find(_SOI, pos + 3)
    return [blob[a:b] for a, b in zip(offsets, offsets[1:] + [len(blob)]) if b > a]


def sample_frames(
    video: Path | str,
    interval: float = 3.0,
    max_frames: int = 40,
    width: int = 640,
    start: float = 0.0,
) -> list[tuple[float, bytes]]:
    """Sample evenly spaced frames from a video.

    Args:
        video: Video file to sample.
        interval: Seconds between samples.
        max_frames: Hard cap on the number of frames returned.
        width: Output width in pixels (height keeps the aspect ratio).
        start: Seek offset in seconds before sampling.

    Returns:
        ``[(timestamp_seconds, jpeg_bytes), ...]`` in chronological order.

    Raises:
        FFmpegError: If ffmpeg exits non-zero and produced no frames.
    """
    interval = max(0.05, float(interval))
    argv = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error"]
    if start > 0:
        argv += ["-ss", f"{start:.3f}"]
    argv += [
        "-i",
        str(video),
        "-vf",
        f"fps=1/{interval},scale={int(width)}:-2:flags=bilinear",
        "-frames:v",
        str(int(max_frames)),
        "-fps_mode",
        "vfr",
        "-q:v",
        "4",
        # iPhone HEVC/10-bit sources decode as limited-range YUV; the mjpeg
        # encoder refuses that unless told to be lenient.
        "-strict",
        "unofficial",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    proc = subprocess.run(argv, capture_output=True)
    images = _split_mjpeg(proc.stdout or b"")
    if not images and proc.returncode == 0:
        # Clips shorter than the sampling interval yield nothing from the fps
        # filter; fall back to a single frame near the start so every clip has
        # at least one picture for the vision model.
        return sample_frames_at(video, [start + 0.2], width=width)
    if not images:
        raise FFmpegError(argv, proc.returncode, (proc.stderr or b"").decode(errors="replace"))
    return [(round(start + i * interval, 3), img) for i, img in enumerate(images[:max_frames])]


def sample_frames_at(
    video: Path | str, times: Sequence[float], width: int = 640
) -> list[tuple[float, bytes]]:
    """Extract frames at explicit timestamps (one fast seek per frame).

    Args:
        video: Video file.
        times: Timestamps in seconds.
        width: Output width in pixels.

    Returns:
        ``[(timestamp, jpeg_bytes), ...]``; timestamps that yield no frame are
        skipped.
    """
    out: list[tuple[float, bytes]] = []
    for t in times:
        argv = [
            FFMPEG, "-hide_banner", "-nostdin", "-v", "error",
            "-ss", f"{max(0.0, float(t)):.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={int(width)}:-2:flags=bilinear",
            "-q:v", "3", "-strict", "unofficial",
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        proc = subprocess.run(argv, capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            out.append((round(float(t), 3), proc.stdout))
        else:
            log.debug("no frame at %.2fs in %s", t, video)
    return out


def save_frames(frames: Iterable[tuple[float, bytes]], out_dir: Path | str) -> list[Path]:
    """Write sampled frames as ``NNN.jpg`` and return the written paths.

    Args:
        frames: ``(timestamp, jpeg_bytes)`` pairs.
        out_dir: Destination directory (created if missing).
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, (_t, jpeg) in enumerate(frames):
        path = d / f"{i:03d}.jpg"
        path.write_bytes(jpeg)
        written.append(path)
    return written


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font, falling back to PIL's bitmap default."""
    for candidate in _ANNOTATION_FONTS:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:  # pragma: no cover - unusable font file
                continue
    return ImageFont.load_default()


def annotate_frame(
    jpeg: bytes,
    index: int,
    t: float,
    grid: bool = True,
    label: str | None = None,
    quality: int = 85,
) -> bytes:
    """Draw a percentage grid and an index/time label onto a frame.

    Args:
        jpeg: Source JPEG bytes.
        index: Frame index shown in the label.
        t: Frame timestamp in seconds.
        grid: Draw vertical grid lines at 10%..90% with percentage labels.
        label: Extra text appended to the index/time label.
        quality: JPEG quality of the returned image.

    Returns:
        Annotated JPEG bytes.
    """
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(11, width // 40)
    font = _load_font(font_size)

    if grid:
        for pct in range(10, 100, 10):
            x = int(width * pct / 100)
            emphasis = pct == 50
            draw.line(
                [(x, 0), (x, height)],
                fill=(255, 255, 255, 190 if emphasis else 90),
                width=2 if emphasis else 1,
            )
            draw.text(
                (x + 3, height - font_size - 6),
                f"{pct}",
                font=font,
                fill=(255, 255, 255, 220),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 200),
            )
        for pct in (25, 50, 75):
            y = int(height * pct / 100)
            draw.line([(0, y), (width, y)], fill=(255, 255, 255, 70), width=1)

    text = f"#{index}  t={t:.1f}s"
    if label:
        text = f"{text}  {label}"
    pad = 6
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle(
        [(0, 0), (box[2] + 2 * pad, box[3] + 2 * pad)],
        fill=(0, 0, 0, 170),
    )
    draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def pct_to_crop_x(subject_x_pct: float, src_width: int, crop_width: int) -> int:
    """Convert a subject position in percent into a safe crop origin.

    Args:
        subject_x_pct: Horizontal subject position, 0..100 (clamped to 5..95).
        src_width: Source width in pixels.
        crop_width: Width of the crop window.

    Returns:
        An even x offset inside ``[0, src_width - crop_width]``.
    """
    pct = min(95.0, max(5.0, float(subject_x_pct)))
    centre = src_width * pct / 100.0
    x = int(centre - crop_width / 2)
    x = max(0, min(x, max(0, src_width - crop_width)))
    return x & ~1
