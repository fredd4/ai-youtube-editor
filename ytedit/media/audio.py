"""Audio helpers for the render pipeline: loudness, silence, ducking, mixing.

Everything in here is deterministic and expressed as ffmpeg filter strings or
small ffmpeg invocations through :func:`ytedit.media.ffmpeg.ff`. The render
layer composes them; the tests assert on the strings without touching media.

Two ducking implementations are provided and they agree numerically:

* :func:`duck_volume_expr` — an inline ``volume='<expr>':eval=frame`` filter.
  This is what :func:`mix_program` actually uses: it needs no side file, so it
  survives path escaping, caching and parallel renders.
* :func:`duck_automation` — the equivalent ``asendcmd`` command-file *content*,
  sampled every 50 ms across the ramps. It is written to
  ``renders/duck.cmd`` for inspection and for hand-tweaking a mix.

The ``denoise`` stage also lives here (:func:`denoise_clips`): it cleans wind
and background noise off a clip's work audio, either with the ElevenLabs
speech isolator or with a free local ffmpeg chain, and marks the result in
``state.json`` so :mod:`ytedit.media.render` uses it instead of the source's
own audio stream.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ..config import Settings
from ..log import get_logger
from .ffmpeg import FFmpegError, ff, ffprobe_json

if TYPE_CHECKING:  # pragma: no cover
    from ..project import Project

log = get_logger(__name__)

#: Sample interval of the generated ``asendcmd`` ramps, in seconds.
RAMP_STEP: float = 0.05

#: Anything at or below this gain is treated as digital silence.
SILENCE_DB: float = -90.0

_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[\s\S]*?\}")
_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")

#: Keys of the pass-1 loudnorm report that pass 2 consumes.
MEASURE_KEYS: tuple[str, ...] = (
    "input_i",
    "input_tp",
    "input_lra",
    "input_thresh",
    "target_offset",
)


# ----------------------------------------------------------------------
# gain helpers
# ----------------------------------------------------------------------
def db_to_linear(db: float) -> float:
    """Convert decibels to a linear amplitude factor (``-inf`` clamps to 0)."""
    value = float(db)
    if value <= SILENCE_DB:
        return 0.0
    return float(10.0 ** (value / 20.0))


def linear_to_db(linear: float) -> float:
    """Convert a linear amplitude factor to decibels (0 maps to ``-inf``)."""
    if linear <= 0:
        return float("-inf")
    return 20.0 * math.log10(linear)


def _num(value: float) -> str:
    """Format a number for an ffmpeg expression without scientific notation."""
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def audio_duration(path: Path | str) -> float:
    """Return the duration of a media file in seconds (0.0 when unknown)."""
    try:
        data = ffprobe_json(path)
    except FFmpegError:  # pragma: no cover - unreadable file
        return 0.0
    try:
        return float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):  # pragma: no cover
        return 0.0


# ----------------------------------------------------------------------
# loudness
# ----------------------------------------------------------------------
def parse_loudnorm(stderr: str) -> dict[str, float]:
    """Extract the pass-1 ``loudnorm`` JSON report from ffmpeg's stderr.

    Args:
        stderr: Complete stderr text of a ``loudnorm=...:print_format=json`` run.

    Returns:
        ``{input_i, input_tp, input_lra, input_thresh, target_offset}`` as
        floats; ``inf``/``-inf`` strings ffmpeg emits for silent input become
        the matching float.

    Raises:
        ValueError: If no report is present in the text.
    """
    matches = _LOUDNORM_JSON.findall(stderr or "")
    if not matches:
        raise ValueError("no loudnorm JSON report in ffmpeg output")
    raw: dict[str, Any] = json.loads(matches[-1])
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = float("-inf") if str(value).lstrip("-").startswith("inf") else 0.0
    return out


def measure_loudness(
    path: Path | str,
    I: float = -14.0,
    TP: float = -1.0,
    LRA: float = 11.0,
    pre_chain: str = "",
    start: float | None = None,
    duration: float | None = None,
) -> dict[str, float]:
    """Run the loudnorm analysis pass over a file (or a range of it).

    Args:
        path: Audio or video file to measure.
        I: Target integrated loudness in LUFS.
        TP: Target true peak in dBTP.
        LRA: Target loudness range in LU.
        pre_chain: Optional filters applied before the measurement (the same
            chain :func:`normalize` will apply before ``loudnorm``).
        start: Optional ``-ss`` seconds into ``path`` — measure a range
            instead of the whole file (e.g. one cut's audio).
        duration: Optional ``-t`` seconds to read after ``start``.

    Returns:
        The parsed pass-1 report (see :func:`parse_loudnorm`).
    """
    chain = f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json"
    if pre_chain:
        chain = f"{pre_chain},{chain}"
    args: list[Any] = []
    if start is not None:
        args += ["-ss", f"{float(start):.6f}"]
    if duration is not None:
        args += ["-t", f"{float(duration):.6f}"]
    args += ["-i", str(path), "-map", "0:a:0", "-af", chain, "-f", "null", "-"]
    stderr = ff(*args)
    return parse_loudnorm(stderr)


def loudnorm_filter(
    measured: Mapping[str, float] | None = None,
    I: float = -14.0,
    TP: float = -1.0,
    LRA: float = 11.0,
) -> str:
    """Build the loudnorm filter string.

    Args:
        measured: A pass-1 report; when given the linear (pass-2) form is
            produced. ``None`` yields the single-pass form.
        I: Target integrated loudness in LUFS.
        TP: Target true peak in dBTP.
        LRA: Target loudness range in LU.

    Returns:
        e.g. ``loudnorm=I=-14:TP=-1:LRA=11:measured_I=...:linear=true``.
    """
    base = f"loudnorm=I={_num(I)}:TP={_num(TP)}:LRA={_num(LRA)}"
    if not measured:
        return base
    values = {k: float(measured.get(k, 0.0)) for k in MEASURE_KEYS}
    if any(math.isinf(v) or math.isnan(v) for v in values.values()):
        # Digital silence (or a report we cannot trust): a linear pass would
        # apply an absurd gain, so fall back to the single-pass form.
        log.warning("loudnorm measurement is not finite (%s); using single pass", values)
        return base
    return (
        f"{base}:measured_I={values['input_i']:.6f}"
        f":measured_LRA={values['input_lra']:.6f}"
        f":measured_TP={values['input_tp']:.6f}"
        f":measured_thresh={values['input_thresh']:.6f}"
        f":offset={values['target_offset']:.6f}"
        ":linear=true"
    )


def peak_limiter_filter(TP: float = -1.0, headroom_db: float = 0.5) -> str:
    """``alimiter`` string that keeps sample peaks ``headroom_db`` under ``TP``.

    ``level=false`` so the limiter only catches peaks and never re-normalizes
    the programme; loudnorm right after it sets the actual level.
    """
    limit = 10 ** ((float(TP) - headroom_db) / 20.0)
    return f"alimiter=limit={limit:.4f}:attack=5:release=50:level=false"


def normalize(
    path: Path | str,
    out: Path | str,
    I: float = -14.0,
    TP: float = -1.0,
    LRA: float = 11.0,
    two_pass: bool = True,
    sample_rate: int = 48000,
    channels: int = 2,
) -> Path:
    """Loudness-normalize an audio file to ``I`` LUFS / ``TP`` dBTP.

    Args:
        path: Input audio (or a file with an audio stream).
        out: Destination WAV; written as ``pcm_s24le`` unless the suffix says
            otherwise.
        I: Target integrated loudness.
        TP: Target true peak.
        LRA: Target loudness range.
        two_pass: Measure first and apply the linear correction (recommended
            for masters); ``False`` uses the single dynamic pass (previews).
        sample_rate: Output sample rate.
        channels: Output channel count.

    Returns:
        The output path.
    """
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A true-peak limiter ahead of loudnorm: without it a mix whose peaks sit
    # near the TP ceiling cannot be raised to the integrated target at all, and
    # a two-pass master lands a whole LU quiet (YouTube never boosts it back).
    if two_pass:
        # Pass 1a: how much gain the programme needs. Raise it by that much
        # BEFORE the limiter, so the limiter (just under the true-peak ceiling)
        # catches the peaks that would otherwise make loudnorm's linear mode
        # cap the gain and leave the master quiet. Pass 1b re-measures through
        # the same chain so the linear correction matches pass 2.
        raw = measure_loudness(path, I=I, TP=TP, LRA=LRA)
        gain_needed = max(0.0, float(I) - float(raw.get("input_i", I)))
        pre = peak_limiter_filter(TP)
        if gain_needed > 0.05:
            pre = f"volume={gain_needed:.2f}dB,{pre}"
        measured = measure_loudness(path, I=I, TP=TP, LRA=LRA, pre_chain=pre)
    else:
        pre = peak_limiter_filter(TP)
        measured = None
    chain = f"{pre},{loudnorm_filter(measured, I=I, TP=TP, LRA=LRA)}"
    codec = "pcm_s24le" if target.suffix.lower() == ".wav" else "aac"
    ff(
        "-i", str(path),
        "-map", "0:a:0",
        "-af", chain,
        "-c:a", codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(target),
    )
    log.debug("normalized %s -> %s (%s)", path, target, "two-pass" if two_pass else "single-pass")
    return target


# ----------------------------------------------------------------------
# silence
# ----------------------------------------------------------------------
def detect_silence(
    path: Path | str, noise_db: float = -35.0, min_d: float = 0.45
) -> list[tuple[float, float]]:
    """Find silent ranges with ``silencedetect``.

    Args:
        path: Audio or video file.
        noise_db: Threshold below which audio counts as silence.
        min_d: Minimum silence length in seconds.

    Returns:
        Sorted ``(start, end)`` pairs in seconds. A silence that runs to the
        end of the file is closed at the file duration.
    """
    stderr = ff(
        "-i", str(path),
        "-map", "0:a:0",
        "-af", f"silencedetect=noise={_num(noise_db)}dB:d={_num(min_d)}",
        "-f", "null", "-",
    )
    return parse_silence(stderr, duration=audio_duration(path))


def parse_silence(stderr: str, duration: float = 0.0) -> list[tuple[float, float]]:
    """Parse ``silencedetect`` output into ranges.

    Args:
        stderr: ffmpeg stderr containing ``silence_start``/``silence_end`` lines.
        duration: File duration, used to close a trailing open silence.

    Returns:
        Sorted, non-overlapping ``(start, end)`` pairs.
    """
    ranges: list[tuple[float, float]] = []
    pending: float | None = None
    for line in (stderr or "").splitlines():
        start = _SILENCE_START.search(line)
        if start:
            pending = float(start.group(1))
            continue
        end = _SILENCE_END.search(line)
        if end is not None and pending is not None:
            ranges.append((max(0.0, pending), float(end.group(1))))
            pending = None
    if pending is not None and duration > pending:
        ranges.append((max(0.0, pending), float(duration)))
    return [(round(s, 3), round(e, 3)) for s, e in sorted(ranges) if e > s]


# ----------------------------------------------------------------------
# ducking
# ----------------------------------------------------------------------
def duck_regions(
    speech_ranges: Sequence[tuple[float, float]],
    duration: float,
    attack: float = 0.15,
    release: float = 0.6,
    pre_roll: float = 0.15,
) -> list[tuple[float, float, float, float]]:
    """Turn speech ranges into merged ducking envelopes.

    Each region is ``(a0, a1, b0, b1)``: the gain ramps from unity at ``a0``
    down to the ducked level at ``a1``, holds until ``b0`` and ramps back to
    unity at ``b1``. Ducking begins ``pre_roll`` seconds *before* the first
    word so the attack never eats the start of a sentence.

    Args:
        speech_ranges: Sorted ``(start, end)`` speech spans in seconds.
        duration: Length of the programme; regions are clamped to it.
        attack: Ramp-down length in seconds.
        release: Ramp-up length in seconds.
        pre_roll: How far ahead of speech the ducked level is reached.

    Returns:
        Sorted, non-overlapping regions.
    """
    total = max(0.0, float(duration))
    attack = max(0.0, float(attack))
    release = max(0.0, float(release))
    pre_roll = max(0.0, float(pre_roll))

    regions: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in speech_ranges if e > s):
        if total and start >= total:
            continue
        a1 = max(0.0, start - pre_roll)
        a0 = max(0.0, a1 - attack)
        b0 = min(end, total) if total else end
        b0 = max(b0, a1)
        b1 = min(b0 + release, total) if total else b0 + release
        if regions and a0 <= regions[-1][3] + 1e-9:
            regions[-1][2] = max(regions[-1][2], b0)
            regions[-1][3] = max(regions[-1][3], b1)
        else:
            regions.append([a0, a1, b0, b1])
    return [(round(a0, 4), round(a1, 4), round(b0, 4), round(b1, 4)) for a0, a1, b0, b1 in regions]


def duck_envelope(
    regions: Sequence[tuple[float, float, float, float]], t: float
) -> float:
    """Evaluate the ducking envelope (0 = open, 1 = fully ducked) at ``t``."""
    for a0, a1, b0, b1 in regions:
        if t < a0 or t >= b1:
            continue
        if t < a1:
            return (t - a0) / (a1 - a0) if a1 > a0 else 1.0
        if t < b0:
            return 1.0
        return (b1 - t) / (b1 - b0) if b1 > b0 else 0.0
    return 0.0


def duck_volume_expr(
    speech_ranges: Sequence[tuple[float, float]],
    duration: float,
    amount_db: float = -12.0,
    attack: float = 0.15,
    release: float = 0.6,
    pre_roll: float = 0.15,
    base_gain_db: float = 0.0,
) -> str:
    """Build an inline ``volume`` filter that ducks under speech.

    The expression is evaluated per frame in **timeline** time, so it must be
    applied *after* the cue has been delayed into position.

    Args:
        speech_ranges: Speech spans in the same time base as the audio.
        duration: Programme length (regions are clamped to it).
        amount_db: How far to duck, relative to ``base_gain_db``.
        attack: Ramp-down length in seconds.
        release: Ramp-up length in seconds.
        pre_roll: Lead-in before the first word.
        base_gain_db: The cue's own gain when nobody is speaking.

    Returns:
        ``volume='...':eval=frame``, or a plain ``volume=<lin>`` when there is
        no speech to duck under.
    """
    base = db_to_linear(base_gain_db)
    regions = duck_regions(speech_ranges, duration, attack, release, pre_roll)
    if not regions:
        return f"volume={_num(base)}"
    ducked = db_to_linear(base_gain_db + amount_db)

    terms: list[str] = []
    for a0, a1, b0, b1 in regions:
        if a1 > a0:
            terms.append(
                f"(gte(t,{_num(a0)})*lt(t,{_num(a1)}))*(t-{_num(a0)})/{_num(a1 - a0)}"
            )
        if b0 > a1:
            terms.append(f"(gte(t,{_num(a1)})*lt(t,{_num(b0)}))")
        if b1 > b0:
            terms.append(
                f"(gte(t,{_num(b0)})*lt(t,{_num(b1)}))*({_num(b1)}-t)/{_num(b1 - b0)}"
            )
    expr = f"{_num(base)}+({_num(ducked)}-{_num(base)})*({'+'.join(terms)})"
    return f"volume='{expr}':eval=frame"


def duck_automation(
    speech_ranges: Sequence[tuple[float, float]],
    duration: float,
    amount_db: float = -12.0,
    attack: float = 0.15,
    release: float = 0.6,
    pre_roll: float = 0.15,
    base_gain_db: float = 0.0,
    step: float = RAMP_STEP,
) -> str:
    """Build the content of an ``asendcmd`` file that ducks music under speech.

    Ramps are sampled every ``step`` seconds; holds emit a single command. Use
    with ``-af "asendcmd=f=duck.cmd,volume=1:eval=frame"``.

    Args:
        speech_ranges: Speech spans in timeline seconds.
        duration: Programme length.
        amount_db: Ducking depth relative to ``base_gain_db``.
        attack: Ramp-down length.
        release: Ramp-up length.
        pre_roll: Lead-in before the first word.
        base_gain_db: Gain when nobody is speaking.
        step: Ramp sampling interval in seconds.

    Returns:
        The file content (a header comment plus ``<t> volume volume <g>;`` lines).
    """
    base = db_to_linear(base_gain_db)
    ducked = db_to_linear(base_gain_db + amount_db)
    regions = duck_regions(speech_ranges, duration, attack, release, pre_roll)
    step = max(0.005, float(step))

    points: list[tuple[float, float]] = [(0.0, base)]
    for a0, a1, b0, b1 in regions:
        times: list[float] = [a0]
        t = a0 + step
        while t < a1:
            times.append(t)
            t += step
        times += [a1, b0]
        t = b0 + step
        while t < b1:
            times.append(t)
            t += step
        times.append(b1)
        for moment in times:
            gain = base + (ducked - base) * duck_envelope(regions, moment)
            points.append((moment, gain))
        points.append((b1, base))

    lines = [
        "# ytedit ducking automation",
        f"# base {base_gain_db:+.1f} dB, duck {amount_db:+.1f} dB, "
        f"attack {attack:.2f}s, release {release:.2f}s, pre-roll {pre_roll:.2f}s",
    ]
    previous: float | None = None
    for moment, gain in points:
        if previous is not None and abs(gain - previous) < 1e-6:
            continue
        lines.append(f"{moment:.3f} volume volume {gain:.6f};")
        previous = gain
    return "\n".join(lines) + "\n"


def speech_gate_expr(ranges: Sequence[tuple[float, float]], duration: float) -> str:
    """Return a ``volume`` chain that silences everything OUTSIDE ``ranges``.

    Used to gate a loudness measurement to just the speech portion of a cut
    (see :func:`measure_speech_gain`): the complement of ``ranges`` within
    ``[0, duration]`` is dropped to :data:`SILENCE_DB`, so ``loudnorm``'s own
    below-threshold gating excludes it from the integrated measurement instead
    of needing a real ``aselect``/``atrim`` splice.

    Args:
        ranges: ``(start, end)`` speech spans, any order.
        duration: Total length of the audio being measured.

    Returns:
        A filter chain (via :func:`mute_ranges_expr`), or ``""`` when the
        ranges already cover the whole duration.
    """
    total = max(0.0, float(duration))
    spans = sorted((max(0.0, float(s)), min(total, float(e))) for s, e in ranges if e > s)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in spans:
        if start > cursor + 1e-6:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if total - cursor > 1e-6:
        gaps.append((cursor, total))
    if not gaps:
        return ""
    return mute_ranges_expr((s, e, SILENCE_DB) for s, e in gaps)


def measure_speech_gain(
    path: Path | str,
    start: float,
    raw_duration: float,
    speech_ranges: Sequence[tuple[float, float]],
    span: float,
    target_lufs: float = -16.0,
    max_gain_db: float = 10.0,
    coverage_threshold: float = 0.6,
    pre_chain: str = "",
) -> tuple[float, float]:
    """Measure a cut's speech loudness and the gain that levels it to ``target_lufs``.

    Runs the loudnorm analysis pass (:func:`measure_loudness`) over
    ``[start, start + raw_duration)`` of ``path``. When ``speech_ranges``
    (already expressed in the same post-``pre_chain`` time base as ``span`` —
    e.g. divided by speed when ``pre_chain`` includes an ``atempo``) cover less
    than ``coverage_threshold`` of ``span``, everything outside them is muted
    first (:func:`speech_gate_expr`) so the measurement reflects the
    narration, not the ambience around it; a cut where speech already
    dominates is measured whole (cheaper, and avoids a jittery gate right at a
    word boundary).

    Args:
        path: Audio file to measure — the cut's own source, its active
            denoised WAV, or an ``audio_from`` clip's source.
        start: ``-ss`` seconds into ``path`` (the cut's own ``in``, or the
            ``audio_from`` range's ``in``).
        raw_duration: ``-t`` seconds to read, in ``path``'s own (pre-speed)
            time base.
        speech_ranges: Transcript word ranges, already converted into the time
            base ``pre_chain`` leaves the audio in.
        span: Total length of the audio in that same post-``pre_chain`` time
            base — used for coverage and to bound the gate.
        target_lufs: Target integrated loudness for the cut's speech.
        max_gain_db: Symmetric clamp on the computed gain.
        coverage_threshold: Speech-to-total ratio above which gating is
            skipped and the whole cut is measured instead.
        pre_chain: Filters already destined for this cut (``aformat``,
            ``atempo``, manual gain, mute ranges) — applied before the gate and
            the loudnorm analysis, so the measurement matches what the cut
            will actually sound like once mixed.

    Returns:
        ``(gain_db, measured_lufs)``. ``gain_db`` is ``0.0`` when there is
        nothing to measure or the measurement is not finite (e.g. digital
        silence), in which case ``measured_lufs`` is ``nan``.
    """
    if not speech_ranges or span <= 0 or raw_duration <= 0:
        return 0.0, float("nan")
    covered = sum(max(0.0, min(e, span) - max(s, 0.0)) for s, e in speech_ranges)
    coverage = covered / span if span else 0.0
    chain = pre_chain
    if coverage < coverage_threshold:
        gate = speech_gate_expr(speech_ranges, span)
        if gate:
            chain = f"{chain},{gate}" if chain else gate
    try:
        report = measure_loudness(path, pre_chain=chain, start=start, duration=raw_duration)
    except (FFmpegError, ValueError) as exc:
        log.warning("speech-gain measurement failed for %s: %s", path, exc)
        return 0.0, float("nan")
    measured = float(report.get("input_i", float("nan")))
    if not math.isfinite(measured):
        return 0.0, measured
    clamp = abs(float(max_gain_db))
    gain = max(-clamp, min(clamp, float(target_lufs) - measured))
    return round(gain, 3), round(measured, 3)


def level_voice_file_gain(
    path: Path | str,
    target_lufs: float = -16.0,
    max_gain_db: float = 10.0,
) -> tuple[float, float]:
    """Return ``(gain_db, measured_lufs)`` leveling a voice pickup file.

    The measurement is cached next to the file as ``<file>.loudness.json``,
    keyed by the file's mtime and the target/clamp: a re-recorded pickup (new
    mtime) or a changed ``speech_target_lufs``/``speech_gain_max_db`` forces a
    fresh measurement; otherwise repeated mixes read the cache instead of
    re-running ffmpeg every time.

    Args:
        path: The voice pickup WAV.
        target_lufs: Target integrated loudness.
        max_gain_db: Symmetric clamp on the computed gain.

    Returns:
        ``(gain_db, measured_lufs)``; ``(0.0, nan)`` when the file cannot be
        read or the measurement is not finite.
    """
    source = Path(path)
    cache = source.with_name(source.name + ".loudness.json")
    try:
        mtime = source.stat().st_mtime_ns
    except OSError:
        return 0.0, float("nan")

    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if (
                data.get("mtime_ns") == mtime
                and data.get("target_lufs") == target_lufs
                and data.get("max_gain_db") == max_gain_db
                and data.get("measured_lufs") is not None
            ):
                return float(data["gain_db"]), float(data["measured_lufs"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass  # fall through and remeasure

    try:
        report = measure_loudness(source)
    except (FFmpegError, ValueError) as exc:
        log.warning("voice pickup %s: loudness measurement failed (%s)", source.name, exc)
        return 0.0, float("nan")
    measured = float(report.get("input_i", float("nan")))
    if math.isfinite(measured):
        clamp = abs(float(max_gain_db))
        gain = max(-clamp, min(clamp, float(target_lufs) - measured))
    else:
        gain = 0.0
    try:
        cache.write_text(
            json.dumps({
                "mtime_ns": mtime,
                "target_lufs": target_lufs,
                "max_gain_db": max_gain_db,
                "measured_lufs": round(measured, 3) if math.isfinite(measured) else None,
                "gain_db": round(gain, 3),
            }),
            encoding="utf-8",
        )
    except OSError:  # pragma: no cover - best-effort cache
        pass
    return round(gain, 3), (round(measured, 3) if math.isfinite(measured) else measured)


def mute_ranges_expr(ranges: Iterable[tuple[float, float, float]]) -> str:
    """Build a ``volume`` chain that attenuates the given ranges.

    One ``volume`` filter is emitted per distinct gain, each enabled over all
    of its ranges. Times are in the time base of the stream the chain is
    applied to — the render layer converts clip time to segment-local time
    before calling this.

    Args:
        ranges: ``(start, end, gain_db)`` triples.

    Returns:
        A comma-separated filter chain, or ``""`` when there is nothing to do.
    """
    groups: dict[float, list[tuple[float, float]]] = {}
    for start, end, gain_db in ranges:
        if float(end) <= float(start):
            continue
        groups.setdefault(round(float(gain_db), 3), []).append((float(start), float(end)))

    parts: list[str] = []
    for gain_db in sorted(groups):
        spans = sorted(groups[gain_db])
        enable = "+".join(f"between(t,{_num(s)},{_num(e)})" for s, e in spans)
        parts.append(f"volume=enable='{enable}':volume={_num(db_to_linear(gain_db))}")
    return ",".join(parts)


# ----------------------------------------------------------------------
# voice
# ----------------------------------------------------------------------
def voice_cleanup_chain(level: str = "light") -> str:
    """Return the narration cleanup chain for a level.

    Args:
        level: ``none`` (empty chain), ``light`` (rumble filter + gentle
            compression) or ``full`` (adds FFT denoise and de-essing, per the
            playbook: highpass 80 -> afftdn -> deesser -> acompressor).

    Returns:
        A comma-separated filter chain.
    """
    key = (level or "none").lower()
    if key in ("none", "off", ""):
        return ""
    # Runs on the program voice bus AFTER per-segment/per-pickup speech
    # leveling (render.py's segment pass and build_audio_bus's voice-pickup
    # gain), which already brings narration close to audio.speech_target_lufs
    # cut by cut. This stage's job is consistency glue, not gain-riding, so it
    # is a gentle ratio/attack/release with unity makeup gain (ffmpeg's
    # acompressor takes `makeup` as a linear factor, 1..64 — 1 means "add
    # none") — a hungrier setting would audibly pump an already-levelled bus
    # instead of smoothing it.
    compressor = "acompressor=threshold=-18dB:ratio=2.5:attack=15:release=250:makeup=1"
    if key == "light":
        return f"highpass=f=80,{compressor}"
    if key == "full":
        return f"highpass=f=80,afftdn=nf=-25,deesser=i=0.4,{compressor}"
    raise ValueError(f"unknown voice cleanup level {level!r} (none|light|full)")


def demucs_separate(
    wav: Path | str, out_dir: Path | str, model: str = "htdemucs_ft"
) -> dict[str, Path] | None:
    """Split an audio file into ``vocals`` / ``no_vocals`` stems with Demucs.

    Used to rescue narration that sits on top of copyrighted background music.
    Demucs is an optional dependency; when the binary is absent this logs and
    returns ``None`` so callers can fall back to muting or ducking.

    Args:
        wav: Input WAV.
        out_dir: Directory Demucs writes its tree into.
        model: Demucs model name.

    Returns:
        ``{"vocals": Path, "no_vocals": Path}`` or ``None``.
    """
    binary = shutil.which("demucs")
    if not binary:
        log.warning("demucs is not installed; skipping source separation for %s", wav)
        return None
    source = Path(wav)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    argv = [
        binary, "--two-stems=vocals", "-n", model, "-d", "mps",
        "-o", str(destination), str(source),
    ]
    log.info("demucs %s", " ".join(argv[1:]))
    proc = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        log.error("demucs failed (%s): %s", proc.returncode, (proc.stderr or "")[-2000:])
        return None
    stem_dir = destination / model / source.stem
    stems = {
        name: stem_dir / f"{name}.wav"
        for name in ("vocals", "no_vocals")
        if (stem_dir / f"{name}.wav").exists()
    }
    if not stems:  # pragma: no cover - demucs layout changed
        log.error("demucs produced no stems under %s", stem_dir)
        return None
    return stems


# ----------------------------------------------------------------------
# the programme mix
# ----------------------------------------------------------------------
def _aformat(sample_rate: int, channels: int) -> str:
    """Return the aformat filter that makes every bus mixable."""
    layout = {1: "mono", 2: "stereo"}.get(int(channels), "stereo")
    return f"aformat=sample_fmts=fltp:sample_rates={int(sample_rate)}:channel_layouts={layout}"


def _delay(seconds: float, channels: int) -> str:
    """Return an ``adelay`` filter placing a stream at ``seconds``."""
    ms = max(0, int(round(float(seconds) * 1000)))
    if ms <= 0:
        return ""
    return "adelay=" + "|".join([str(ms)] * max(1, int(channels)))


def mix_program(
    voice_wav: Path | str,
    music_items: Sequence[Mapping[str, Any]],
    sfx_items: Sequence[Mapping[str, Any]],
    speech_ranges: Sequence[tuple[float, float]],
    out_wav: Path | str,
    settings: Settings | None = None,
    duration: float | None = None,
    cmd_file: Path | str | None = None,
) -> Path:
    """Mix the voice/source bus with music cues and sound effects.

    Everything happens in ONE ``filter_complex``: each cue is trimmed (looping
    a short file to fill its span), faded, gained, delayed into position and —
    for music with ``duck`` set — put under the speech-driven volume envelope.
    The buses then meet in a single ``amix=normalize=0`` whose length is that
    of the voice bus.

    Args:
        voice_wav: The programme/voice bus; defines the output length.
        music_items: ``{file, at, end, gain_db, fade_in, fade_out, duck,
            amount_db, attack, release, ranges}`` mappings. ``ranges``
            overrides ``speech_ranges`` for that cue (manual ducking).
        sfx_items: ``{file, at, gain_db, end}`` mappings; never ducked.
        speech_ranges: Speech spans in timeline seconds for automatic ducking.
        out_wav: Destination WAV (``pcm_s24le``).
        settings: Source of ``audio.*`` and ``ducking.*`` defaults.
        duration: Programme length; probed from ``voice_wav`` when omitted.
        cmd_file: When given, the equivalent ``asendcmd`` automation for the
            first ducked cue is written here (documentation / hand tweaking).

    Returns:
        The output path.
    """
    target = Path(out_wav)
    target.parent.mkdir(parents=True, exist_ok=True)

    audio_cfg = settings.section("audio") if settings else {}
    duck_cfg = settings.section("ducking") if settings else {}
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    channels = int(audio_cfg.get("channels", 2))
    total = float(duration) if duration is not None else audio_duration(voice_wav)

    inputs: list[str] = ["-i", str(voice_wav)]
    graph: list[str] = [f"[0:a]{_aformat(sample_rate, channels)}[bus0]"]
    labels: list[str] = ["bus0"]

    def _cue(item: Mapping[str, Any], index: int, is_music: bool) -> None:
        """Append the input args and graph node for one cue."""
        path = Path(str(item["file"]))
        at = max(0.0, float(item.get("at", 0.0)))
        end = item.get("end")
        span = float(end) - at if end not in (None, 0) else audio_duration(path)
        if total:
            span = min(span, max(0.0, total - at))
        span = max(0.05, span)

        file_len = audio_duration(path)
        if file_len and file_len < span - 0.05:
            inputs.extend(["-stream_loop", "-1"])
        inputs.extend(["-i", str(path)])

        gain_db = float(item.get("gain_db", -18.0 if is_music else -6.0))
        fade_in = max(0.0, float(item.get("fade_in", 0.0)))
        fade_out = max(0.0, float(item.get("fade_out", 0.0)))
        fade_in = min(fade_in, span / 2)
        fade_out = min(fade_out, span / 2)

        parts = [
            _aformat(sample_rate, channels),
            f"atrim=0:{_num(span)}",
            "asetpts=N/SR/TB",
        ]
        if fade_in > 0:
            parts.append(f"afade=t=in:st=0:d={_num(fade_in)}")
        if fade_out > 0:
            parts.append(f"afade=t=out:st={_num(span - fade_out)}:d={_num(fade_out)}")

        delay = _delay(at, channels)
        if delay:
            parts.append(delay)

        # Ducking is evaluated in timeline time, so it comes after adelay.
        ducking = bool(item.get("duck")) and is_music
        if ducking:
            ranges = item.get("ranges") or speech_ranges
            parts.append(
                duck_volume_expr(
                    list(ranges),
                    total or (at + span),
                    amount_db=float(item.get("amount_db", duck_cfg.get("amount_db", -12.0))),
                    attack=float(item.get("attack", duck_cfg.get("attack", 0.15))),
                    release=float(item.get("release", duck_cfg.get("release", 0.6))),
                    pre_roll=float(item.get("pre_roll", duck_cfg.get("pre_roll", 0.15))),
                    base_gain_db=gain_db,
                )
            )
        else:
            parts.append(f"volume={_num(db_to_linear(gain_db))}")

        label = f"{'m' if is_music else 'x'}{index}"
        graph.append(f"[{len(labels)}:a]{','.join(p for p in parts if p)}[{label}]")
        labels.append(label)

    for i, item in enumerate(music_items):
        _cue(item, i, True)
    for i, item in enumerate(sfx_items):
        _cue(item, i, False)

    if cmd_file is not None:
        first = next((m for m in music_items if m.get("duck")), None)
        if first is not None:
            content = duck_automation(
                list(first.get("ranges") or speech_ranges),
                total,
                amount_db=float(first.get("amount_db", duck_cfg.get("amount_db", -12.0))),
                attack=float(first.get("attack", duck_cfg.get("attack", 0.15))),
                release=float(first.get("release", duck_cfg.get("release", 0.6))),
                pre_roll=float(first.get("pre_roll", duck_cfg.get("pre_roll", 0.15))),
                base_gain_db=float(first.get("gain_db", -18.0)),
            )
            Path(cmd_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd_file).write_text(content, encoding="utf-8")

    if len(labels) == 1:
        graph.append(f"[bus0]{_aformat(sample_rate, channels)}[mix]")
    else:
        joined = "".join(f"[{label}]" for label in labels)
        graph.append(f"{joined}amix=inputs={len(labels)}:normalize=0:duration=first[mix]")

    args: list[Any] = [
        *inputs,
        "-filter_complex", ";".join(graph),
        "-map", "[mix]",
        "-c:a", "pcm_s24le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
    ]
    if total:
        args += ["-t", f"{total:.3f}"]
    args.append(str(target))
    ff(*args)
    return target


def peaks_for_wav(wav: Path | str, peaks_per_second: float = 16.6667) -> dict[str, Any]:
    """Compute waveform peaks for a WAV (delegates to the ingest implementation).

    Args:
        wav: Audio file.
        peaks_per_second: Peak resolution.

    Returns:
        The wavesurfer-shaped peaks document.
    """
    from .ingest import compute_peaks

    return compute_peaks(Path(wav), peaks_per_second=peaks_per_second)


# ----------------------------------------------------------------------
# denoising (wind, hiss, room tone)
# ----------------------------------------------------------------------
#: Rumble filter frequency for the local chain: wind noise on a phone mic is
#: mostly below this, speech fundamentals are above it.
DENOISE_HIGHPASS_HZ: float = 110.0

#: ``afftdn`` reduction depth in dB (the filter's ``nr``).
DENOISE_NR_DB: float = 24.0

#: ``afftdn``'s ``nf`` is set this far *above* the clip's measured noise floor.
#: Higher removes more noise and dulls the voice sooner.
DENOISE_NF_MARGIN_DB: float = 8.0

#: ``nf`` used when the noise floor could not be measured.
DENOISE_NF_FALLBACK_DB: float = -30.0

#: ElevenLabs audio isolation, $/minute (mirrors ``prices.elevenlabs``).
ISOLATION_USD_PER_MINUTE: float = 0.12

#: Engines accepted by :func:`denoise_clips`.
DENOISE_ENGINES: tuple[str, ...] = ("elevenlabs", "local")

#: Length of each half of the A/B preview file, in seconds.
AB_PREVIEW_SECONDS: float = 6.0


class DenoiseError(RuntimeError):
    """Raised when a clip cannot be denoised (missing audio, unknown engine)."""


def denoised_path(project: "Project", clip_id: str, engine: str | None = None) -> Path:
    """Path of a denoised WAV.

    Args:
        project: The owning project.
        clip_id: Clip id.
        engine: ``elevenlabs``/``local`` for the per-engine file
            (``<clip>.denoised-<engine>.wav``, kept so both can be compared);
            ``None`` for the active one (``<clip>.denoised.wav``), which is what
            ``state.json`` points at and the renderer uses.
    """
    suffix = f".denoised-{engine}.wav" if engine else ".denoised.wav"
    return project.audio_dir / f"{clip_id}{suffix}"


def local_denoise_chain(
    noise_floor_db: float | None = None,
    highpass_hz: float = DENOISE_HIGHPASS_HZ,
    nr_db: float = DENOISE_NR_DB,
    margin_db: float = DENOISE_NF_MARGIN_DB,
) -> str:
    """Build the free wind/noise chain: ``highpass`` -> ``afftdn`` -> ``adeclick``.

    ``highpass`` kills the low-frequency rumble a phone microphone makes in
    wind, ``afftdn`` subtracts the broadband hiss and ``adeclick`` takes out the
    impulsive gusts. ``afftdn``'s own noise tracking (``tn=1``) measurably does
    nothing in ffmpeg 7.1.1, so ``nf`` is set from the clip's *measured* noise
    floor instead — a fixed ``nf`` either eats the voice on a clean clip or
    misses the noise on a loud one.

    Args:
        noise_floor_db: Measured mean level of the quietest window
            (:func:`measure_noise_floor`); ``None`` falls back to
            :data:`DENOISE_NF_FALLBACK_DB`.
        highpass_hz: Rumble filter corner frequency.
        nr_db: ``afftdn`` reduction depth.
        margin_db: How far above the measured floor ``nf`` is placed.

    Returns:
        A comma-separated ffmpeg filter chain.
    """
    if noise_floor_db is None or not math.isfinite(noise_floor_db):
        nf = DENOISE_NF_FALLBACK_DB
    else:
        nf = float(noise_floor_db) + float(margin_db)
    nf = max(-80.0, min(-20.0, nf))
    return (
        f"highpass=f={_num(highpass_hz)},"
        f"afftdn=nr={_num(nr_db)}:nf={nf:.0f},"
        "adeclick"
    )


def denoise_local(
    src: Path | str,
    out: Path | str,
    chain: str | None = None,
    sample_rate: int = 48000,
    channels: int = 1,
) -> Path:
    """Run the free ffmpeg denoise chain over a WAV.

    Cheaper but blunter than the ElevenLabs isolator: it attenuates broadband
    noise without a speech model, so heavy wind leaves audible pumping and the
    voice loses some top end (measured on real footage: about −5 dB of noise for
    −1.5 dB of speech). Good enough for light hiss; use ``elevenlabs`` for a clip
    that is mostly wind.

    Args:
        src: Input WAV.
        out: Destination WAV.
        chain: Filter chain override; by default the clip's noise floor is
            measured and :func:`local_denoise_chain` is built from it.
        sample_rate: Output sample rate.
        channels: Output channel count.

    Returns:
        The output path.
    """
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if chain is None:
        floor = measure_noise_floor(src)
        chain = local_denoise_chain(floor)
        log.info("denoise (local): noise floor %.1f dB -> %s", floor, chain)
    ff(
        "-i", str(src),
        "-map", "0:a:0",
        "-af", chain,
        "-c:a", "pcm_s16le",
        "-ar", str(int(sample_rate)),
        "-ac", str(int(channels)),
        str(target),
    )
    return target


def _to_aligned_wav(
    raw: bytes,
    out: Path | str,
    expected: float,
    sample_rate: int = 48000,
    channels: int = 1,
    suffix: str = ".mp3",
) -> Path:
    """Decode API audio bytes to a WAV that is time-aligned with the source.

    The renderer feeds the denoised file the *same* ``-ss``/``-t`` as the
    mezzanine, so a drift of even a few frames would desync the voice. The
    result is padded and hard-trimmed to ``expected`` seconds.
    """
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".raw" + suffix)
    tmp.write_bytes(raw)
    try:
        args: list[Any] = ["-i", str(tmp), "-map", "0:a:0"]
        if expected > 0:
            args += ["-af", f"apad=whole_dur={expected:.3f}"]
        args += [
            "-c:a", "pcm_s16le",
            "-ar", str(int(sample_rate)),
            "-ac", str(int(channels)),
        ]
        if expected > 0:
            args += ["-t", f"{expected:.3f}"]
        args.append(str(target))
        ff(*args)
    finally:
        tmp.unlink(missing_ok=True)
    actual = audio_duration(target)
    if expected > 0 and abs(actual - expected) > 0.05:  # pragma: no cover - defensive
        log.warning(
            "denoised audio is %.3fs but the source is %.3fs — check the alignment",
            actual, expected,
        )
    return target


def _cost_callback(project: "Project", clip_id: str) -> Any:
    """Build an ElevenLabs ``cost_callback`` that charges the project ledger."""
    from ..costs import charge

    def _callback(
        service: str,
        op: str,
        model: str | None = None,
        units: Any = None,
        usd: float | None = None,
        **extra: Any,
    ) -> None:
        seconds = 0.0
        if isinstance(units, str) and units.endswith("s"):
            try:
                seconds = float(units[:-1])
            except ValueError:
                seconds = 0.0
        charge(
            project, service=service, op=op, units=seconds or units, usd=usd,
            model=model, stage="denoise", clip=clip_id, **extra,
        )

    return _callback


def denoise_elevenlabs(project: "Project", clip_id: str, out: Path | str) -> Path:
    """Isolate speech from noise with the ElevenLabs audio-isolation endpoint.

    Charges ``$0.12`` per minute of audio to the project ledger (the budget is
    checked first) and writes a 48 kHz WAV aligned with the source clip.
    """
    from ..ai.elevenlabs import ElevenLabs
    from ..costs import check_budget

    settings = project.settings
    src = project.audio_path(clip_id)
    if not src.exists():
        raise DenoiseError(f"{clip_id}: no work audio at {src} — run ingest first")
    seconds = audio_duration(src)
    estimated = seconds / 60.0 * float(
        settings.get("prices.elevenlabs.isolation_per_minute", ISOLATION_USD_PER_MINUTE)
    )
    log.info(
        "denoise %s with elevenlabs: %.1fs of audio, estimated [cost]$%.4f[/]",
        clip_id, seconds, estimated,
    )
    check_budget(project, estimated)

    work = settings.get("audio.work_wav", {}) or {}
    client = ElevenLabs(
        api_key=settings.require_key("elevenlabs"),
        cost_callback=_cost_callback(project, clip_id),
    )
    try:
        raw = client.isolate_audio(src, duration_s=seconds)
    finally:
        client.close()
    return _to_aligned_wav(
        raw, out, seconds,
        sample_rate=int(work.get("sample_rate", 48000)),
        channels=int(work.get("channels", 1)),
    )


def denoise_clips(
    project: "Project",
    clip_ids: Sequence[str] | None = None,
    engine: str = "elevenlabs",
    force: bool = False,
) -> list[dict[str, Any]]:
    """Denoise the work audio of one or more clips.

    Both engines write ``media/audio/<clip>.denoised-<engine>.wav`` and copy the
    result to ``media/audio/<clip>.denoised.wav``, which is the *active* file:
    ``state.json`` records ``clips[<id>].denoised`` (that path),
    ``clips[<id>].denoise_engine`` and ``clips[<id>].use_denoised = true``, and
    :func:`ytedit.media.render.render_segment` substitutes it for the source's
    own audio stream. Keeping the per-engine files side by side lets both be
    A/B'd without re-spending money.

    Args:
        project: Project whose clips are processed.
        clip_ids: Clips to denoise; ``None`` means every clip that has audio.
        engine: ``elevenlabs`` (paid speech isolation, $0.12/min) or ``local``
            (free ffmpeg chain, see :func:`local_denoise_chain`).
        force: Redo a clip whose per-engine file already exists.

    Returns:
        One ``{"clip", "engine", "status", "file", "seconds", ...}`` record per
        clip; ``status`` is ``done``, ``cached`` or ``error``.

    Raises:
        DenoiseError: For an unknown engine or an empty clip selection.
    """
    if engine not in DENOISE_ENGINES:
        raise DenoiseError(f"unknown denoise engine {engine!r} (use {'/'.join(DENOISE_ENGINES)})")

    state = project.load_state()
    clips: dict[str, Any] = state.get("clips", {})
    if clip_ids:
        unknown = [c for c in clip_ids if c not in clips]
        if unknown:
            raise DenoiseError(f"unknown clips: {', '.join(unknown)}")
        targets = list(clip_ids)
    else:
        targets = [
            cid for cid, clip in sorted(clips.items())
            if clip.get("has_audio", True) and project.audio_path(cid).exists()
        ]
    if not targets:
        raise DenoiseError(f"{project.slug}: no clips with work audio to denoise")

    results: list[dict[str, Any]] = []
    for clip_id in targets:
        src = project.audio_path(clip_id)
        engine_file = denoised_path(project, clip_id, engine)
        active = denoised_path(project, clip_id)
        record: dict[str, Any] = {
            "clip": clip_id,
            "engine": engine,
            "file": project.rel(active),
            "engine_file": project.rel(engine_file),
            "seconds": round(audio_duration(src), 3) if src.exists() else 0.0,
        }
        try:
            if not src.exists():
                raise DenoiseError(f"{clip_id}: no work audio at {src} — run ingest first")
            if engine_file.exists() and not force:
                record["status"] = "cached"
                log.info("denoise %s: reusing %s (--force to redo)", clip_id, engine_file.name)
            else:
                if engine == "elevenlabs":
                    denoise_elevenlabs(project, clip_id, engine_file)
                else:
                    audio_cfg = project.settings.section("audio")
                    work = audio_cfg.get("work_wav") or {}
                    denoise_local(
                        src, engine_file,
                        sample_rate=int(work.get("sample_rate", 48000)),
                        channels=int(work.get("channels", 1)),
                    )
                record["status"] = "done"
            shutil.copyfile(engine_file, active)
            project.set_clip_stage(
                clip_id, "denoise", "done",
                denoised=project.rel(active),
                denoise_engine=engine,
                use_denoised=True,
            )
        except Exception as exc:  # a bad clip must not abort the batch
            record["status"] = "error"
            record["error"] = str(exc)
            log.error("denoise %s failed: %s", clip_id, exc)
        results.append(record)

    errors = [r for r in results if r["status"] == "error"]
    project.set_stage(
        "denoise", "error" if errors and len(errors) == len(results) else "done",
        engine=engine,
        clips=[r["clip"] for r in results if r["status"] != "error"],
        error="; ".join(str(r.get("error")) for r in errors) if errors else "",
    )
    return results


def set_use_denoised(project: "Project", clip_ids: Sequence[str], enabled: bool) -> list[str]:
    """Flip ``clips[<id>].use_denoised`` without deleting anything on disk."""
    touched: list[str] = []
    with project.edit_state() as state:
        for clip_id in clip_ids:
            clip = state.get("clips", {}).get(clip_id)
            if clip is None:
                raise DenoiseError(f"unknown clip {clip_id!r}")
            clip["use_denoised"] = bool(enabled)
            touched.append(clip_id)
    return touched


def scan_windows(
    wav: Path | str, length: float, candidates: Sequence[float] | None = None,
    limit: int = 12,
) -> list[tuple[float, float]]:
    """Measure the mean level of several windows of a file.

    Args:
        wav: Audio file to scan.
        length: Window length in seconds.
        candidates: Window start times; when omitted the file is scanned on a
            coarse eight-point grid.
        limit: Maximum number of windows actually measured.

    Returns:
        ``(start, mean_db)`` pairs in start order; empty for a file shorter than
        the window.
    """
    total = audio_duration(wav)
    if total <= length:
        return []
    starts = list(candidates) if candidates else [
        t * (total - length) / 7.0 for t in range(8)
    ]
    starts = sorted({round(max(0.0, min(float(s), total - length)), 3) for s in starts})[:limit]
    out: list[tuple[float, float]] = []
    for start in starts:
        stderr = ff(
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(wav),
            "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-",
        )
        match = re.search(r"mean_volume:\s*(-?[\d.]+) dB", stderr)
        out.append((start, float(match.group(1)) if match else float("-inf")))
    return out


def loudest_window(
    wav: Path | str, length: float = AB_PREVIEW_SECONDS,
    candidates: Sequence[float] | None = None,
) -> float:
    """Return the start time of the loudest ``length``-second window.

    Args:
        wav: Audio file to scan.
        length: Window length in seconds.
        candidates: Start times to consider; when omitted a coarse grid is used.

    Returns:
        The best start time (0.0 for a file shorter than the window).
    """
    windows = scan_windows(wav, length, candidates)
    if not windows:
        return 0.0
    best, level = max(windows, key=lambda w: w[1])
    log.debug("loudest %.1fs window of %s starts at %.2fs (%.1f dB)", length, wav, best, level)
    return best


def measure_noise_floor(wav: Path | str, length: float = 1.0) -> float:
    """Estimate a file's noise floor: the mean level of its quietest window.

    Args:
        wav: Audio file to scan.
        length: Window length in seconds.

    Returns:
        Mean level in dBFS (``-inf`` for digital silence, and the whole-file
        mean for a file shorter than one window).
    """
    windows = scan_windows(wav, length)
    if not windows:
        return noise_floor(wav, 0.0, max(0.1, audio_duration(wav)))["mean_db"]
    floor = min(level for _start, level in windows)
    log.debug("noise floor of %s: %.1f dB", wav, floor)
    return floor


def denoise_ab_preview(
    project: "Project",
    clip_id: str,
    engine: str | None = None,
    length: float = AB_PREVIEW_SECONDS,
    speech_starts: Sequence[float] | None = None,
) -> Path:
    """Write ``media/audio/<clip>.denoise_ab.wav`` — original then denoised.

    The same ``length``-second window (the loudest one, so it lands on speech)
    is taken from the source and from the denoised file and concatenated, so
    the difference can be judged by ear in one play.

    Args:
        project: Owning project.
        clip_id: Clip to preview.
        engine: Compare this engine's file; ``None`` uses the active one.
        length: Seconds taken from each side.
        speech_starts: Candidate window starts (e.g. from transcript words).

    Returns:
        The A/B file path.
    """
    src = project.audio_path(clip_id)
    cleaned = denoised_path(project, clip_id, engine)
    if not src.exists():
        raise DenoiseError(f"{clip_id}: no work audio at {src}")
    if not cleaned.exists():
        raise DenoiseError(f"{clip_id}: no denoised audio at {cleaned} — run denoise first")
    start = loudest_window(src, length=length, candidates=speech_starts)
    target = project.audio_dir / f"{clip_id}.denoise_ab.wav"
    ff(
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(cleaned),
        "-filter_complex",
        f"[0:a]{_aformat(48000, 1)}[a0];[1:a]{_aformat(48000, 1)}[a1];"
        f"[a0][a1]concat=n=2:v=0:a=1[ab]",
        "-map", "[ab]",
        "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1",
        str(target),
    )
    log.info(
        "A/B preview for %s: %.1fs original + %.1fs denoised from %.2fs -> %s",
        clip_id, length, length, start, target,
    )
    return target


def noise_floor(wav: Path | str, start: float, end: float) -> dict[str, float]:
    """Measure mean/peak level of ``[start, end)`` with ``volumedetect``.

    Used to compare the noise floor of a non-speech window before and after
    denoising.

    Returns:
        ``{"mean_db", "max_db", "start", "end"}`` (``-inf`` when silent).
    """
    length = max(0.01, float(end) - float(start))
    stderr = ff(
        "-ss", f"{float(start):.3f}", "-t", f"{length:.3f}", "-i", str(wav),
        "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-",
    )
    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", stderr)
    peak = re.search(r"max_volume:\s*(-?[\d.]+) dB", stderr)
    return {
        "mean_db": float(mean.group(1)) if mean else float("-inf"),
        "max_db": float(peak.group(1)) if peak else float("-inf"),
        "start": round(float(start), 3),
        "end": round(float(end), 3),
    }
