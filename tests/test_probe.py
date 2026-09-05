"""Tests for ``ytedit.media.probe``."""

from __future__ import annotations

from pathlib import Path

import pytest

from ytedit.media.probe import (
    MediaInfo,
    bit_depth_from_pix_fmt,
    probe,
    round_fps,
)


@pytest.mark.parametrize(
    ("measured", "expected"),
    [(29.97, 30), (23.976, 24), (59.94, 60), (30.0, 30), (25.0, 25), (0.0, 30), (36.6, 37)],
)
def test_round_fps(measured: float, expected: int) -> None:
    assert round_fps(measured) == expected


@pytest.mark.parametrize(
    ("pix_fmt", "depth"),
    [("yuv420p", 8), ("yuv420p10le", 10), ("yuv422p10le", 10), ("p010le", 10), (None, 8)],
)
def test_bit_depth(pix_fmt: str | None, depth: int) -> None:
    assert bit_depth_from_pix_fmt(pix_fmt) == depth


def test_landscape(media: dict[str, Path]) -> None:
    info = probe(media["landscape.mp4"])
    assert info.kind == "video"
    assert (info.width, info.height) == (1920, 1080)
    assert info.orientation == "horizontal"
    assert info.rotation == 0
    assert info.fps == pytest.approx(30.0, abs=0.01)
    assert info.target_fps == 30
    assert info.has_audio and info.audio_channels == 2
    assert info.hdr is None
    assert info.duration == pytest.approx(6.0, abs=0.2)
    assert info.creation_time is not None


def test_vertical_orientation(media: dict[str, Path]) -> None:
    info = probe(media["vertical.mp4"])
    assert info.orientation == "vertical"
    assert (info.display_width, info.display_height) == (1080, 1920)


def test_rotation_is_detected_and_swaps_display_size(media: dict[str, Path]) -> None:
    info = probe(media["rotated.mov"])
    assert info.rotation in (90, 270)
    assert (info.width, info.height) == (1920, 1080)
    assert (info.display_width, info.display_height) == (1080, 1920)
    assert info.orientation == "vertical"


def test_missing_audio_stream(media: dict[str, Path]) -> None:
    info = probe(media["silent.mp4"])
    assert info.has_audio is False
    assert info.audio_channels == 0
    assert info.audio_codec is None


def test_hlg_is_flagged_as_hdr(media: dict[str, Path]) -> None:
    info = probe(media["hlg.mp4"])
    assert info.hdr == "hlg"
    assert info.is_hdr is True
    assert info.color_transfer == "arib-std-b67"
    assert info.bit_depth == 10


def test_vfr_detection(media: dict[str, Path]) -> None:
    info = probe(media["vfr.mp4"])
    assert info.vfr is True
    # A VFR source normalizes to its nominal rate, not frames/duration.
    assert info.target_fps == 60
    assert probe(media["landscape.mp4"]).vfr is False


def test_to_dict_drops_raw(media: dict[str, Path]) -> None:
    data = probe(media["landscape.mp4"]).to_dict()
    assert "raw" not in data
    assert data["display_width"] == 1920


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        probe("/nope/does-not-exist.mp4")


def test_is_iphone_heuristic() -> None:
    assert MediaInfo(path="x", make="Apple").is_iphone is True
    assert MediaInfo(path="x", model="iPhone 17 Pro").is_iphone is True
    assert MediaInfo(path="x", make="DJI").is_iphone is False
