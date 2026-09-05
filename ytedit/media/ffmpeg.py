"""Thin, explicit wrappers around the ``ffmpeg`` and ``ffprobe`` binaries.

No ``ffmpeg-python``/``pydub``: every call is an explicit argument list. ffmpeg
writes ``loudnorm``/``silencedetect``/``showinfo`` output to stderr, so
:func:`ff` returns stderr; machine-readable progress is read from stdout via
``-progress pipe:1``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..log import get_logger

log = get_logger(__name__)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

#: Flags prepended to every ffmpeg invocation.
BASE_FLAGS: tuple[str, ...] = ("-hide_banner", "-nostdin", "-y")

ProgressCallback = Callable[["Progress"], None]


class FFmpegError(RuntimeError):
    """An ffmpeg/ffprobe call failed; carries the command and stderr tail."""

    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-25:])
        super().__init__(
            f"ffmpeg failed (exit {returncode}):\n"
            f"  {' '.join(str(c) for c in cmd)}\n--- stderr tail ---\n{tail}"
        )


@dataclass(frozen=True)
class Progress:
    """One ``-progress`` report block.

    Attributes:
        out_time: Seconds of output written so far.
        frame: Frames encoded so far.
        fps: Current encoding speed in frames per second.
        speed: Encoding speed relative to realtime (``1.0`` = realtime).
        percent: Completion fraction 0..1 when a total duration is known.
        raw: The parsed key/value block as produced by ffmpeg.
    """

    out_time: float = 0.0
    frame: int = 0
    fps: float = 0.0
    speed: float = 0.0
    percent: float | None = None
    raw: dict[str, str] | None = None


def _to_float(value: str, default: float = 0.0) -> float:
    """Parse a float, tolerating ffmpeg's ``N/A`` and trailing ``x``."""
    try:
        return float(value.rstrip("x"))
    except (TypeError, ValueError):
        return default


def _parse_progress_block(block: dict[str, str], total_duration: float | None) -> Progress:
    """Turn a raw ``-progress`` block into a :class:`Progress`."""
    us = block.get("out_time_us") or block.get("out_time_ms") or "0"
    out_time = _to_float(us) / 1_000_000.0
    if out_time == 0.0 and "out_time" in block:
        parts = block["out_time"].split(":")
        try:
            hh, mm, ss = (float(p) for p in parts)
            out_time = hh * 3600 + mm * 60 + ss
        except ValueError:
            out_time = 0.0
    percent = None
    if total_duration and total_duration > 0:
        percent = max(0.0, min(1.0, out_time / total_duration))
    return Progress(
        out_time=out_time,
        frame=int(_to_float(block.get("frame", "0"))),
        fps=_to_float(block.get("fps", "0")),
        speed=_to_float(block.get("speed", "0")),
        percent=percent,
        raw=block,
    )


def ff(
    *args: Any,
    progress_cb: ProgressCallback | None = None,
    total_duration: float | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> str:
    """Run ffmpeg with the standard flags and return its stderr.

    Args:
        *args: ffmpeg arguments after ``-hide_banner -nostdin -y``. Values are
            stringified, so ``Path`` and numbers may be passed directly.
        progress_cb: When given, ``-progress pipe:1`` is injected and the
            callback receives a :class:`Progress` per report block.
        total_duration: Expected output duration in seconds; enables
            ``Progress.percent``.
        check: Raise :class:`FFmpegError` on a non-zero exit (default).
        timeout: Kill the process after this many seconds.

    Returns:
        The complete stderr text (loudnorm JSON, silencedetect lines, ...).

    Raises:
        FFmpegError: If ffmpeg exits non-zero and ``check`` is true.
    """
    argv: list[str] = [FFMPEG, *BASE_FLAGS]
    extra = [str(a) for a in args]
    if progress_cb is not None and "-progress" not in extra:
        argv += ["-progress", "pipe:1", "-nostats"]
    argv += extra

    log.debug("ffmpeg %s", " ".join(argv[1:]))

    if progress_cb is None:
        proc = subprocess.run(
            argv, capture_output=True, text=True, errors="replace", timeout=timeout
        )
        if check and proc.returncode != 0:
            raise FFmpegError(argv, proc.returncode, proc.stderr or "")
        return proc.stderr or ""

    popen = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1,
    )
    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        assert popen.stderr is not None
        for line in popen.stderr:
            stderr_chunks.append(line)

    reader = threading.Thread(target=_drain_stderr, daemon=True)
    reader.start()

    block: dict[str, str] = {}
    assert popen.stdout is not None
    for line in popen.stdout:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        block[key] = value
        if key == "progress":
            try:
                progress_cb(_parse_progress_block(block, total_duration))
            except Exception:  # pragma: no cover - never let a callback kill a render
                log.exception("progress callback raised")
            block = {}

    popen.wait(timeout=timeout)
    reader.join(timeout=5)
    stderr = "".join(stderr_chunks)
    if check and popen.returncode != 0:
        raise FFmpegError(argv, popen.returncode or -1, stderr)
    return stderr


def ffprobe_json(path: Path | str, *extra: Any) -> dict[str, Any]:
    """Run ffprobe and return the parsed JSON for format + streams.

    Args:
        path: Media file to probe.
        *extra: Extra ffprobe arguments appended before the input.

    Returns:
        The decoded ffprobe JSON document.

    Raises:
        FFmpegError: If ffprobe fails or emits invalid JSON.
    """
    argv = [
        FFPROBE,
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        *[str(a) for a in extra],
        str(path),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(argv, proc.returncode, proc.stderr or "")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(argv, 0, f"invalid ffprobe JSON: {exc}") from exc


def ffprobe_packets(path: Path | str, count: int = 120, stream: str = "v:0") -> list[dict[str, Any]]:
    """Return the first ``count`` packets of a stream (used for VFR detection).

    Args:
        path: Media file.
        count: Maximum number of packets to read.
        stream: Stream specifier, e.g. ``v:0``.
    """
    argv = [
        FFPROBE,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        stream,
        "-show_entries",
        "packet=pts_time,dts_time,duration_time",
        "-read_intervals",
        f"%+#{int(count)}",
        "-print_format",
        "json",
        str(path),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout or "{}").get("packets", [])
    except json.JSONDecodeError:  # pragma: no cover
        return []


@lru_cache(maxsize=32)
def has_encoder(name: str) -> bool:
    """Return True when the local ffmpeg build exposes an encoder (cached)."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-v", "error", "-h", f"encoder={name}"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return proc.returncode == 0 and "Encoder" in (proc.stdout or "")


def filter_chain(parts: Iterable[str]) -> str:
    """Join non-empty filter fragments with commas."""
    return ",".join(p for p in parts if p)


def ffmpeg_version() -> str:
    """Return the first line of ``ffmpeg -version``."""
    proc = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True)
    return (proc.stdout or "").splitlines()[0] if proc.stdout else "unknown"
