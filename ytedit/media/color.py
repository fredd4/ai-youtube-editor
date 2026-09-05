"""Pure ffmpeg filter builders: tonemapping, grading, canvas fitting, Ken Burns.

Every function here is side-effect free and returns filter *strings*, so the
render layer can compose them and the tests can assert on them without touching
ffmpeg.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal, Mapping

from ..config import Settings, global_settings

FitMode = Literal["cover", "contain", "blur-fill", "crop-pan"]

_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _cfg(settings: Settings | None) -> Settings:
    """Return the given settings or the cached global ones."""
    return settings or global_settings()


def _label(name: str) -> str:
    """Sanitize a string into a usable filter_complex label."""
    return _LABEL_SAFE.sub("_", name) or "v"


def join(parts: Iterable[str]) -> str:
    """Join non-empty filter fragments with commas."""
    return ",".join(p for p in parts if p)


def even(value: int) -> int:
    """Clamp a pixel dimension to an even number (H.264 requires it)."""
    return int(value) & ~1


# ----------------------------------------------------------------------
# HDR -> SDR
# ----------------------------------------------------------------------
def tonemap_chain(hdr: str | None, settings: Settings | None = None) -> str:
    """Build the HDR to SDR (bt709) tonemap chain.

    Args:
        hdr: ``"hlg"``, ``"pq"`` or ``None``. ``None`` yields an empty string.
        settings: Source of ``tonemap.*`` defaults.

    Returns:
        A comma-separated zscale/tonemap chain, or ``""`` for SDR sources.
    """
    if not hdr:
        return ""
    cfg = _cfg(settings)
    key = str(hdr).lower()
    npl = cfg.get("tonemap.pq_npl", 1000) if key == "pq" else cfg.get("tonemap.hlg_npl", 100)
    operator = cfg.get("tonemap.tonemap", "hable")
    desat = cfg.get("tonemap.desat", 0)
    return join(
        [
            f"zscale=t=linear:npl={npl}",
            "format=gbrpf32le",
            "zscale=p=bt709",
            f"tonemap=tonemap={operator}:desat={desat}",
            "zscale=t=bt709:m=bt709:r=tv",
            "format=yuv420p",
        ]
    )


# ----------------------------------------------------------------------
# grading
# ----------------------------------------------------------------------
def grade_chain(
    preset: str = "default", is_iphone: bool = False, settings: Settings | None = None
) -> str:
    """Build the colour grade chain for a preset.

    ``unsharp`` is dropped for iPhone footage (already sharpened in camera)
    unless ``grade.unsharp_for_iphone`` is enabled.

    Args:
        preset: Key under ``grade.presets`` (``none`` yields ``""``).
        is_iphone: Whether the source came from an Apple device.
        settings: Source of the preset table.

    Returns:
        A comma-separated filter chain (possibly empty).
    """
    cfg = _cfg(settings)
    if not preset or preset == "none":
        return ""
    filters = list(cfg.grade_preset(preset))
    allow_unsharp = bool(cfg.get("grade.unsharp_for_iphone", False))
    if is_iphone and not allow_unsharp:
        filters = [f for f in filters if not f.startswith("unsharp")]
    return join(filters)


# ----------------------------------------------------------------------
# fitting a source into the canvas
# ----------------------------------------------------------------------
def display_size(src_w: int, src_h: int, rotation: int = 0) -> tuple[int, int]:
    """Return ``(w, h)`` after applying a 0/90/180/270 display rotation."""
    return (src_h, src_w) if int(rotation) % 360 in (90, 270) else (src_w, src_h)


def fit_chain(
    src_w: int,
    src_h: int,
    rotation: int = 0,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
    mode: FitMode = "cover",
    in_label: str | None = None,
    out_label: str | None = None,
    settings: Settings | None = None,
) -> tuple[str, bool]:
    """Build the chain that fits a source frame into the canvas.

    Args:
        src_w: Coded source width.
        src_h: Coded source height.
        rotation: Display rotation in degrees; the source is measured after it.
        canvas_w: Canvas width.
        canvas_h: Canvas height.
        mode: ``cover`` (fill and crop), ``contain`` (letterbox), ``blur-fill``
            (blurred background, source centred) or ``crop-pan`` (same framing
            as ``cover``; the motion comes from :func:`crop_pan`).
        in_label: Input pad label for the ``blur-fill`` fragment (default ``0:v``).
        out_label: Output pad label for the ``blur-fill`` fragment (default ``v``).
        settings: Source of ``fit.*`` defaults.

    Returns:
        ``(filter_string, needs_split)``. When ``needs_split`` is False the
        string is a plain comma chain usable with ``-vf`` or as one node of a
        ``filter_complex``. When True it is a labelled, semicolon-separated
        ``filter_complex`` fragment reading ``[in_label]`` and writing
        ``[out_label]``.
    """
    cfg = _cfg(settings)
    flags = cfg.get("fit.scale_flags", "lanczos")
    cw, ch = even(canvas_w), even(canvas_h)
    dw, dh = display_size(src_w, src_h, rotation)

    same_aspect = dw > 0 and dh > 0 and abs(dw / dh - cw / ch) < 0.01

    if mode in ("cover", "crop-pan") or (same_aspect and mode != "contain"):
        chain = join(
            [
                f"scale={cw}:{ch}:force_original_aspect_ratio=increase:flags={flags}",
                f"crop={cw}:{ch}",
                "setsar=1",
            ]
        )
        return chain, False

    if mode == "contain":
        chain = join(
            [
                f"scale={cw}:{ch}:force_original_aspect_ratio=decrease:flags={flags}",
                f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=black",
                "setsar=1",
            ]
        )
        return chain, False

    if mode == "blur-fill":
        blur = cfg.section("fit").get("blur", {})
        src = in_label or "0:v"
        dst = out_label or "v"
        base = _label(dst)
        bg, fg, bgb, fgs = f"{base}bg", f"{base}fg", f"{base}bgb", f"{base}fgs"
        fragment = (
            f"[{src}]split=2[{bg}][{fg}];"
            f"[{bg}]scale={cw}:{ch}:force_original_aspect_ratio=increase:flags={flags},"
            f"crop={cw}:{ch},"
            f"boxblur=luma_radius={blur.get('luma_radius', 45)}:"
            f"luma_power={blur.get('luma_power', 2)}:"
            f"chroma_radius={blur.get('chroma_radius', 25)}:"
            f"chroma_power={blur.get('chroma_power', 1)},"
            f"eq=brightness={blur.get('brightness', -0.06)}[{bgb}];"
            f"[{fg}]scale={cw}:{ch}:force_original_aspect_ratio=decrease:flags={flags}[{fgs}];"
            f"[{bgb}][{fgs}]overlay=(W-w)/2:(H-h)/2:shortest=1,setsar=1,format=yuv420p[{dst}]"
        )
        return fragment, True

    raise ValueError(f"unknown fit mode {mode!r}")


# ----------------------------------------------------------------------
# Ken Burns
# ----------------------------------------------------------------------
def crop_pan(
    src_w: int,
    src_h: int,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
    duration: float = 4.0,
    fps: int = 30,
    zoom_from: float = 1.0,
    zoom_to: float = 1.12,
    pan_from: tuple[float, float] = (0.5, 0.5),
    pan_to: tuple[float, float] = (0.5, 0.5),
    settings: Settings | None = None,
) -> str:
    """Build a Ken Burns (slow zoom + pan) chain using ``zoompan``.

    The source is first scaled to cover the canvas at the maximum zoom so the
    crop never runs out of pixels, and ``zoompan`` then interpolates zoom and
    centre linearly over ``duration``.

    Args:
        src_w: Source width (already display-oriented).
        src_h: Source height (already display-oriented).
        canvas_w: Canvas width.
        canvas_h: Canvas height.
        duration: Move length in seconds.
        fps: Output frame rate.
        zoom_from: Zoom factor at the first frame (1.0 = no zoom).
        zoom_to: Zoom factor at the last frame.
        pan_from: Centre of the visible area at the start, as ``(x, y)``
            fractions of the frame.
        pan_to: Centre at the end.
        settings: Source of ``fit.scale_flags``.

    Returns:
        A comma-separated filter chain producing ``canvas_w x canvas_h`` frames.
    """
    cfg = _cfg(settings)
    flags = cfg.get("fit.scale_flags", "lanczos")
    cw, ch = even(canvas_w), even(canvas_h)
    frames = max(2, int(round(max(0.04, duration) * max(1, fps))))
    zmax = max(zoom_from, zoom_to, 1.0)
    # Oversample so zoompan's integer crop does not shimmer.
    base_w, base_h = even(int(cw * zmax * 2)), even(int(ch * zmax * 2))

    prescale = (
        f"scale={base_w}:{base_h}:force_original_aspect_ratio=increase:flags={flags},"
        f"crop={base_w}:{base_h}"
    )
    n = frames - 1
    zoom_expr = f"'{zoom_from}+({zoom_to - zoom_from})*on/{n}'"
    fx, fy = pan_from
    tx, ty = pan_to
    x_expr = f"'(iw*({fx}+({tx - fx})*on/{n}))-(iw/zoom/2)'"
    y_expr = f"'(ih*({fy}+({ty - fy})*on/{n}))-(ih/zoom/2)'"
    zoompan = (
        f"zoompan=z={zoom_expr}:x={x_expr}:y={y_expr}:d={frames}:s={cw}x{ch}:fps={fps}"
    )
    return join([prescale, zoompan, "setsar=1"])


# ----------------------------------------------------------------------
# small composable helpers
# ----------------------------------------------------------------------
def fps_chain(fps: int) -> str:
    """Return the CFR conversion filter for a frame rate."""
    return f"fps={int(fps)}"


def source_chain(
    hdr: str | None,
    grade: str,
    is_iphone: bool,
    src_w: int,
    src_h: int,
    rotation: int,
    canvas_w: int,
    canvas_h: int,
    fps: int,
    fit: FitMode = "cover",
    settings: Settings | None = None,
    in_label: str | None = None,
    out_label: str | None = None,
) -> tuple[str, bool]:
    """Compose tonemap + fit + grade + fps for one segment.

    Args:
        hdr: ``hlg``/``pq``/``None``.
        grade: Grade preset name.
        is_iphone: Whether to skip sharpening.
        src_w: Coded source width.
        src_h: Coded source height.
        rotation: Display rotation in degrees.
        canvas_w: Canvas width.
        canvas_h: Canvas height.
        fps: Canvas frame rate.
        fit: Fit mode.
        settings: Settings source.
        in_label: Input label when the fit mode needs a ``filter_complex``.
        out_label: Output label when the fit mode needs a ``filter_complex``.

    Returns:
        ``(filter_string, needs_split)`` with the same contract as
        :func:`fit_chain`.
    """
    tonemap = tonemap_chain(hdr, settings)
    grading = grade_chain(grade, is_iphone, settings)
    fit_str, needs_split = fit_chain(
        src_w, src_h, rotation, canvas_w, canvas_h, fit,
        in_label=in_label, out_label=out_label, settings=settings,
    )
    tail = join([grading, fps_chain(fps), "format=yuv420p"])
    if not needs_split:
        return join([tonemap, fit_str, tail]), False

    # Labelled fragment: prepend tonemap to the split input and append the tail
    # to the overlay output.
    src = in_label or "0:v"
    dst = out_label or "v"
    pre_label = f"{_label(dst)}pre"
    head = f"[{src}]{tonemap}[{pre_label}];" if tonemap else ""
    fragment = fit_str
    if head:
        fragment = fragment.replace(f"[{src}]split=2", f"[{pre_label}]split=2", 1)
    inner = f"{_label(dst)}fit"
    fragment = fragment[: fragment.rfind(f"[{dst}]")] + f"[{inner}]"
    fragment = f"{head}{fragment};[{inner}]{tail}[{dst}]"
    return fragment, True


def describe(chain: str) -> list[str]:
    """Split a comma chain into individual filters (for logs and tests)."""
    depth = 0
    current: list[str] = []
    out: list[str] = []
    for ch in chain:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        out.append("".join(current))
    return [c for c in out if c]


def preset_names(settings: Settings | None = None) -> list[str]:
    """Return the available grade preset names."""
    cfg = _cfg(settings)
    presets: Mapping[str, Any] = cfg.get("grade.presets", {})
    return sorted(presets)
