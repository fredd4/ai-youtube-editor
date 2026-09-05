#!/usr/bin/env python3
"""Generate synthetic test clips with ffmpeg lavfi.

Run directly (``python tests/fixtures/make_fixtures.py``) or via ``make test``.
Everything lands in ``tests/fixtures/generated/`` which is git-ignored. Files
that already exist are left alone unless ``--force`` is passed.

Fixtures
--------
``landscape.mp4``   1920x1080, 30 fps, 6 s, testsrc2 + 440 Hz sine
``vertical.mp4``    1080x1920, 30 fps, 6 s, testsrc2 + 660 Hz sine
``rotated.mov``     landscape pixels with a 90-degree display matrix (shows vertical)
``silent.mp4``      1280x720, 30 fps, 4 s, no audio stream at all
``hlg.mp4``         1920x1080 10-bit, tagged HLG / bt2020
``vfr.mp4``         variable frame rate (r_frame_rate 60 vs avg_frame_rate 37.5)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "generated"

#: name -> ffmpeg arguments (after ``-hide_banner -nostdin -y``, before the output).
RECIPES: dict[str, list[str]] = {
    "landscape.mp4": [
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-metadata", "creation_time=2026-08-12T10:00:00.000000Z",
        "-shortest",
    ],
    "vertical.mp4": [
        "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=48000:duration=6",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-metadata", "creation_time=2026-08-12T10:05:00.000000Z",
        "-shortest",
    ],
    # Landscape pixels carrying a 90-degree display matrix: ffmpeg autorotates it
    # to portrait on decode, exactly like an iPhone clip shot vertically.
    "rotated.mov": [
        "-display_rotation", "90",
        "-i", "@landscape_src@",
        "-c", "copy",
        "-metadata", "creation_time=2026-08-12T10:10:00.000000Z",
    ],
    "silent.mp4": [
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=4",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-an",
        "-metadata", "creation_time=2026-08-12T10:15:00.000000Z",
    ],
    "hlg.mp4": [
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=4",
        "-vf", "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p10le",
        "-color_trc", "arib-std-b67", "-color_primaries", "bt2020", "-colorspace", "bt2020nc",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-metadata", "creation_time=2026-08-12T10:20:00.000000Z",
        "-shortest",
    ],
    # Drop 3 of every 4 frames in the first two seconds and keep the original
    # timestamps, so r_frame_rate (60) and avg_frame_rate (37.5) disagree.
    "vfr.mp4": [
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=60:duration=4",
        "-vf", "select='if(lt(t,2),not(mod(n,4)),1)'",
        "-fps_mode", "passthrough",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-an",
        "-metadata", "creation_time=2026-08-12T10:25:00.000000Z",
    ],
}


def build(name: str, force: bool = False, out_dir: Path | None = None) -> Path:
    """Generate one fixture and return its path.

    Args:
        name: Key of :data:`RECIPES`.
        force: Regenerate even when the file exists.
        out_dir: Destination directory (defaults to ``generated/``).
    """
    directory = out_dir or FIXTURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists() and not force:
        return path
    recipe = list(RECIPES[name])
    if "@landscape_src@" in recipe:
        source = build("landscape.mp4", force=False, out_dir=directory)
        recipe = [str(source) if a == "@landscape_src@" else a for a in recipe]
    argv = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", *recipe, str(path)]
    subprocess.run(argv, check=True)
    return path


def build_all(force: bool = False, out_dir: Path | None = None) -> dict[str, Path]:
    """Generate every fixture and return ``{name: path}``."""
    return {name: build(name, force=force, out_dir=out_dir) for name in RECIPES}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate existing fixtures")
    args = parser.parse_args()
    for name, path in build_all(force=args.force).items():
        print(f"{name:16s} {path.stat().st_size / 1e6:6.2f} MB  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
