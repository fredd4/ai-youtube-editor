"""Tests for the pure filter builders in ``ytedit.media.color``."""

from __future__ import annotations

import pytest

from ytedit.media.color import (
    crop_pan,
    describe,
    display_size,
    fit_chain,
    grade_chain,
    source_chain,
    tonemap_chain,
)


def test_tonemap_is_empty_for_sdr() -> None:
    assert tonemap_chain(None) == ""


@pytest.mark.parametrize(("hdr", "npl"), [("hlg", "npl=100"), ("pq", "npl=1000")])
def test_tonemap_chain(hdr: str, npl: str) -> None:
    chain = tonemap_chain(hdr)
    assert npl in chain
    assert "tonemap=tonemap=hable:desat=0" in chain
    assert chain.endswith("format=yuv420p")
    assert "zscale=t=bt709:m=bt709:r=tv" in chain


def test_grade_chain_skips_unsharp_for_iphone() -> None:
    assert "unsharp" in grade_chain("default_sharp", is_iphone=False)
    assert "unsharp" not in grade_chain("default_sharp", is_iphone=True)


def test_default_grade_has_no_unsharp_and_uses_vibrance() -> None:
    chain = grade_chain("default")
    assert "unsharp" not in chain
    assert "vibrance=intensity=0.25" in chain
    assert describe(chain)[0].startswith("eq=brightness=0.02")


def test_grade_none_is_empty() -> None:
    assert grade_chain("none") == ""


def test_display_size_swaps_on_rotation() -> None:
    assert display_size(1920, 1080, 90) == (1080, 1920)
    assert display_size(1920, 1080, 180) == (1920, 1080)


def test_fit_cover_crops() -> None:
    chain, needs_split = fit_chain(1080, 1920, 0, 1920, 1080, "cover")
    assert needs_split is False
    assert "force_original_aspect_ratio=increase" in chain
    assert "crop=1920:1080" in chain


def test_fit_contain_pads() -> None:
    chain, needs_split = fit_chain(1080, 1920, 0, 1920, 1080, "contain")
    assert needs_split is False
    assert "force_original_aspect_ratio=decrease" in chain
    assert "pad=1920:1080" in chain


def test_fit_blur_fill_is_a_labelled_fragment() -> None:
    chain, needs_split = fit_chain(
        1080, 1920, 0, 1920, 1080, "blur-fill", in_label="1:v", out_label="v2"
    )
    assert needs_split is True
    assert chain.startswith("[1:v]split=2[")
    assert chain.endswith("[v2]")
    assert "boxblur=luma_radius=45" in chain
    assert "overlay=(W-w)/2:(H-h)/2:shortest=1" in chain


def test_fit_blur_fill_labels_are_unique_per_output() -> None:
    a, _ = fit_chain(1080, 1920, 0, 1920, 1080, "blur-fill", out_label="s001")
    b, _ = fit_chain(1080, 1920, 0, 1920, 1080, "blur-fill", out_label="s002")
    assert "s001bg" in a and "s002bg" in b
    assert "s001bg" not in b


def test_matching_aspect_uses_the_simple_cover_path() -> None:
    chain, needs_split = fit_chain(3840, 2160, 0, 1920, 1080, "blur-fill")
    assert needs_split is False
    assert "boxblur" not in chain


def test_rotated_source_is_measured_after_rotation() -> None:
    # 1920x1080 pixels with a 90-degree display matrix is a vertical source.
    chain, needs_split = fit_chain(1920, 1080, 90, 1920, 1080, "blur-fill")
    assert needs_split is True
    assert "boxblur" in chain


def test_crop_pan_builds_a_zoompan() -> None:
    chain = crop_pan(1920, 1080, 1920, 1080, duration=4.0, fps=30, zoom_from=1.0, zoom_to=1.1)
    assert "zoompan=" in chain
    assert "d=120" in chain
    assert "s=1920x1080" in chain
    assert "fps=30" in chain


def test_source_chain_simple_path_order() -> None:
    chain, needs_split = source_chain(
        "hlg", "default", False, 3840, 2160, 0, 1920, 1080, 30, fit="cover"
    )
    assert needs_split is False
    assert chain.index("zscale=t=linear") < chain.index("crop=1920:1080")
    assert chain.index("crop=1920:1080") < chain.index("eq=brightness")
    assert chain.endswith("fps=30,format=yuv420p")


def test_source_chain_blur_fill_path_wraps_tonemap_and_grade() -> None:
    chain, needs_split = source_chain(
        "hlg", "default", True, 1080, 1920, 0, 1920, 1080, 30,
        fit="blur-fill", in_label="0:v", out_label="v",
    )
    assert needs_split is True
    assert chain.startswith("[0:v]zscale=t=linear")
    assert chain.endswith("[v]")
    assert "vibrance" in chain
    assert "unsharp" not in chain
    # The tonemap output feeds the split, and the graded tail produces [v].
    assert "split=2[" in chain
