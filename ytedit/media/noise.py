"""Deterministic wind/background-noise scan over clip work audio.

``ytedit noise <slug>`` reads each clip's transcript (``transcripts/<clip>.json``,
via :func:`ytedit.words.load_words`) and its work audio
(``media/audio/<clip>.wav``, 48 kHz mono ``pcm_s16le`` — the same file
:mod:`ytedit.media.audio` denoises) and measures, purely from signal levels,
whether the clip needs :func:`ytedit.media.audio.denoise_clips` before it goes
into a render:

* ``speech_rms_db`` / ``gap_rms_db`` — RMS level inside transcript word spans
  vs. inside the gaps between them, both restricted to
  ``[first_word.start - 1s, last_word.end + 1s]`` so silence far outside the
  spoken part of the clip (leader/trailer) never pollutes the noise estimate.
* ``snr_db`` — ``speech_rms_db - gap_rms_db``. A clean clip's gaps are much
  quieter than its speech; a windy one is barely quieter at all.
* ``low_band_ratio`` — share of the gap signal's energy below
  :data:`LOW_BAND_HZ` (150 Hz), from a Hann-windowed FFT over up to
  :data:`MAX_LOWBAND_SECONDS` of gap audio. Wind rumble on a phone mic sits
  almost entirely below 150 Hz; room hiss and traffic do not, so this
  separates "put it through the isolator" from "it's just a quiet room".

A clip needs at least :data:`MIN_SPAN_SECONDS` of both gap and speech audio to
produce a measurement; shorter clips (or clips with under four transcript
words) are skipped rather than guessed at.

Config keys read via ``Settings.get`` (not present in ``config/defaults.yaml``
— pass ``--threshold-snr``/``--threshold-low`` on the command line, or add
them under a ``noise:`` section in ``project.yaml``/``defaults.yaml`` to
change the project-wide default):

* ``noise.snr_flag_db`` (default 2.0) — below this **and** past
  ``noise.low_band_flag`` a clip is flagged ``windy``.
* ``noise.low_band_flag`` (default 0.80) — see above.
* ``noise.snr_warn_db`` (default 4.0) — below this (and not already ``windy``)
  a clip is flagged ``noisy`` (probably fine, worth a listen).

This module only *measures* and *reports*; it never renders or spends money on
its own. ``--denoise`` on the CLI command drives the existing
:func:`ytedit.media.audio.denoise_clips` for whatever it flags ``windy``.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..log import get_logger
from ..project import utcnow
from ..timeline import merge_ranges

if TYPE_CHECKING:  # pragma: no cover
    from ..words import Word
    from ..project import Project

log = get_logger(__name__)

STAGE = "noise"

#: A clip needs at least this many transcript words before it is scanned.
MIN_WORDS = 4

#: A measurement needs at least this much gap *and* speech audio to be trusted.
MIN_SPAN_SECONDS: float = 0.5

#: Silence kept before the first word / after the last word of the analysis
#: window (seconds) — the clip's leader/trailer is excluded past this.
EDGE_PAD_SECONDS: float = 1.0

#: Longest stretch of gap audio actually run through the FFT (seconds).
MAX_LOWBAND_SECONDS: float = 20.0

#: Rumble/hiss corner frequency for ``low_band_ratio`` (Hz).
LOW_BAND_HZ: float = 150.0

#: Floor used for a silent (or empty) buffer's RMS level, in dBFS.
DIGITAL_SILENCE_DB: float = -90.0

#: Defaults for the ``windy``/``noisy`` thresholds (see module docstring).
DEFAULT_SNR_FLAG_DB: float = 2.0
DEFAULT_LOW_BAND_FLAG: float = 0.80
DEFAULT_SNR_WARN_DB: float = 4.0


class NoiseError(RuntimeError):
    """Raised when the noise scan has no candidate clips to look at."""


# ----------------------------------------------------------------------
# signal helpers
# ----------------------------------------------------------------------
def read_wav_mono(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as float32 samples in ``[-1, 1]``.

    Args:
        path: WAV file (mono or interleaved multi-channel).

    Returns:
        ``(samples, sample_rate)``; multi-channel input is averaged to mono.
    """
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if width != 2:  # pragma: no cover - work audio is always pcm_s16le
        raise NoiseError(f"{path}: expected 16-bit PCM audio, got {width * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data / 32768.0, sample_rate


def rms_db(samples: np.ndarray) -> float:
    """RMS level of ``samples`` (assumed in ``[-1, 1]``) in dBFS.

    Args:
        samples: Float32 samples; an empty array is treated as digital silence.

    Returns:
        ``20*log10(rms)``, floored at :data:`DIGITAL_SILENCE_DB`.
    """
    if samples.size == 0:
        return DIGITAL_SILENCE_DB
    mean_sq = float(np.mean(np.square(samples, dtype=np.float64)))
    if mean_sq <= 0:
        return DIGITAL_SILENCE_DB
    return max(DIGITAL_SILENCE_DB, 10.0 * math.log10(mean_sq))


def low_band_ratio(samples: np.ndarray, sample_rate: int, cutoff_hz: float = LOW_BAND_HZ) -> float:
    """Share of ``samples``' spectral energy below ``cutoff_hz``.

    Args:
        samples: Float32 samples (a Hann window is applied here).
        sample_rate: Sample rate of ``samples``.
        cutoff_hz: The rumble/hiss corner frequency.

    Returns:
        ``0.0`` for too little or fully silent audio; otherwise a ratio in
        ``[0, 1]``.
    """
    n = samples.shape[0]
    if n < 2:
        return 0.0
    windowed = samples * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    total = float(power.sum())
    if total <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    low = float(power[freqs < cutoff_hz].sum())
    return low / total


def _slice_seconds(
    samples: np.ndarray, sample_rate: int, intervals: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Concatenate the samples covered by ``intervals`` (seconds, clip time)."""
    total = samples.shape[0]
    chunks = []
    for start, end in intervals:
        i0 = max(0, int(round(start * sample_rate)))
        i1 = min(total, int(round(end * sample_rate)))
        if i1 > i0:
            chunks.append(samples[i0:i1])
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def _interval_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)


def _complement(
    intervals: Sequence[tuple[float, float]], window_start: float, window_end: float
) -> list[tuple[float, float]]:
    """Gaps between sorted, non-overlapping ``intervals`` within a window."""
    out: list[tuple[float, float]] = []
    cursor = window_start
    for start, end in intervals:
        start = max(start, window_start)
        end = min(end, window_end)
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < window_end:
        out.append((cursor, window_end))
    return out


# ----------------------------------------------------------------------
# per-clip scan
# ----------------------------------------------------------------------
def scan_clip(project: "Project", clip_id: str, words: "list[Word] | None" = None) -> dict[str, Any] | None:
    """Measure gap/speech levels and the low-frequency ratio for one clip.

    Args:
        project: Owning project.
        clip_id: Clip to scan.
        words: Pre-loaded transcript words (:func:`ytedit.words.load_words`
            output); loaded from disk when omitted.

    Returns:
        ``{"clip", "gap_rms_db", "speech_rms_db", "snr_db", "low_band_ratio",
        "gap_seconds", "speech_seconds"}``, or ``None`` when the clip has fewer
        than :data:`MIN_WORDS` words, no audio, or under
        :data:`MIN_SPAN_SECONDS` of gap or speech signal.
    """
    from ..words import load_words

    wav = project.audio_path(clip_id)
    if not wav.exists():
        return None
    if words is None:
        words = load_words(project, clip_id)
    if len(words) < MIN_WORDS:
        return None

    samples, sample_rate = read_wav_mono(wav)
    if sample_rate <= 0 or samples.size == 0:
        return None
    duration = samples.shape[0] / sample_rate

    first, last = words[0].s, words[-1].e
    window_start = max(0.0, first - EDGE_PAD_SECONDS)
    window_end = min(duration, last + EDGE_PAD_SECONDS)
    if window_end <= window_start:
        return None

    word_spans = merge_ranges([(w.s, w.e) for w in words], merge_gap=0.0, pad=0.0)
    speech_intervals = [
        (max(s, window_start), min(e, window_end)) for s, e in word_spans
    ]
    speech_intervals = [(s, e) for s, e in speech_intervals if e > s]
    gap_intervals = _complement(speech_intervals, window_start, window_end)

    speech_seconds = _interval_seconds(speech_intervals)
    gap_seconds = _interval_seconds(gap_intervals)
    if speech_seconds < MIN_SPAN_SECONDS or gap_seconds < MIN_SPAN_SECONDS:
        return None

    speech_db = rms_db(_slice_seconds(samples, sample_rate, speech_intervals))
    gap_db = rms_db(_slice_seconds(samples, sample_rate, gap_intervals))

    # Low-band ratio over up to MAX_LOWBAND_SECONDS of gap audio, taken in
    # time order (an FFT over the whole clip's gaps would be needlessly slow
    # and the spectral character of wind doesn't drift within a clip).
    budget = MAX_LOWBAND_SECONDS
    lowband_intervals: list[tuple[float, float]] = []
    for start, end in gap_intervals:
        if budget <= 0:
            break
        take = min(end - start, budget)
        lowband_intervals.append((start, start + take))
        budget -= take
    lowband_samples = _slice_seconds(samples, sample_rate, lowband_intervals)

    return {
        "clip": clip_id,
        "gap_rms_db": round(gap_db, 2),
        "speech_rms_db": round(speech_db, 2),
        "snr_db": round(speech_db - gap_db, 2),
        "low_band_ratio": round(low_band_ratio(lowband_samples, sample_rate), 3),
        "gap_seconds": round(gap_seconds, 2),
        "speech_seconds": round(speech_seconds, 2),
    }


def flag_clip(
    metrics: dict[str, Any],
    snr_flag_db: float = DEFAULT_SNR_FLAG_DB,
    low_band_flag: float = DEFAULT_LOW_BAND_FLAG,
    snr_warn_db: float = DEFAULT_SNR_WARN_DB,
) -> list[str]:
    """Classify a clip's :func:`scan_clip` metrics as ``windy``/``noisy``/clean.

    ``windy`` (speech is barely above the noise floor *and* that floor is
    dominated by sub-150 Hz rumble) takes priority over ``noisy`` (speech is
    somewhat close to the floor but it isn't wind-shaped) — a clip is never
    tagged both, since ``windy`` already implies a low SNR.

    Returns:
        ``["windy"]``, ``["noisy"]`` or ``[]``.
    """
    if metrics["snr_db"] < snr_flag_db and metrics["low_band_ratio"] > low_band_flag:
        return ["windy"]
    if metrics["snr_db"] < snr_warn_db:
        return ["noisy"]
    return []


# ----------------------------------------------------------------------
# project-wide scan
# ----------------------------------------------------------------------
def used_clip_ids(project: "Project") -> set[str] | None:
    """Clip ids referenced with unmuted audio by ``plan/timeline.json``.

    Returns:
        ``None`` when there is no timeline yet (nothing to restrict to);
        otherwise the set of clip ids used by at least one video segment whose
        ``mute_source`` is false.
    """
    if not project.timeline_file.exists():
        return None
    from ..timeline import Timeline

    timeline = Timeline.load(project.timeline_file)
    return {seg.clip for seg in timeline.tracks.video if not seg.mute_source}


def scan_project(
    project: "Project",
    used_only: bool = True,
    snr_flag_db: float | None = None,
    low_band_flag: float | None = None,
    snr_warn_db: float | None = None,
) -> dict[str, Any]:
    """Scan every eligible clip of ``project`` and classify it.

    Args:
        project: Project to scan.
        used_only: Restrict to clips used unmuted in ``plan/timeline.json``
            (see :func:`used_clip_ids`); has no effect when there is no
            timeline yet.
        snr_flag_db: Override for ``noise.snr_flag_db``.
        low_band_flag: Override for ``noise.low_band_flag``.
        snr_warn_db: Override for ``noise.snr_warn_db``.

    Returns:
        ``{"generated", "used_only", "thresholds", "clips", "skipped"}`` —
        ``clips`` is worst-first (ascending ``snr_db``), each carrying
        ``flags``, ``use_denoised`` and ``denoise_engine`` from ``state.json``;
        ``skipped`` lists clips that had a transcript but not enough gap/speech
        signal to measure.

    Raises:
        NoiseError: No clip in the project has a usable transcript (or, with
            ``used_only``, no clip in the timeline does).
    """
    settings = project.settings
    resolved_snr_flag = (
        settings.get("noise.snr_flag_db", DEFAULT_SNR_FLAG_DB) if snr_flag_db is None else snr_flag_db
    )
    resolved_low_band = (
        settings.get("noise.low_band_flag", DEFAULT_LOW_BAND_FLAG) if low_band_flag is None else low_band_flag
    )
    resolved_snr_warn = (
        settings.get("noise.snr_warn_db", DEFAULT_SNR_WARN_DB) if snr_warn_db is None else snr_warn_db
    )

    from ..words import load_words

    state = project.load_state()
    clips: dict[str, Any] = state.get("clips", {})
    restrict_to = used_clip_ids(project) if used_only else None

    candidates = 0
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for clip_id, clip in sorted(clips.items()):
        if restrict_to is not None and clip_id not in restrict_to:
            continue
        if not clip.get("has_audio", True):
            continue
        words = load_words(project, clip_id)
        if len(words) < MIN_WORDS:
            continue
        candidates += 1
        metrics = scan_clip(project, clip_id, words=words)
        if metrics is None:
            skipped.append(clip_id)
            continue
        metrics["flags"] = flag_clip(
            metrics,
            snr_flag_db=resolved_snr_flag,
            low_band_flag=resolved_low_band,
            snr_warn_db=resolved_snr_warn,
        )
        metrics["use_denoised"] = bool(clip.get("use_denoised"))
        metrics["denoise_engine"] = clip.get("denoise_engine")
        rows.append(metrics)

    if candidates == 0:
        scope = "the clips used in plan/timeline.json" if restrict_to is not None else "any clip"
        raise NoiseError(
            f"{project.slug}: no transcript with >= {MIN_WORDS} words in {scope} — "
            "run `ytedit transcribe` first"
        )

    rows.sort(key=lambda r: r["snr_db"])
    return {
        "generated": utcnow(),
        "used_only": restrict_to is not None,
        "thresholds": {
            "snr_flag_db": resolved_snr_flag,
            "low_band_flag": resolved_low_band,
            "snr_warn_db": resolved_snr_warn,
        },
        "clips": rows,
        "skipped": skipped,
    }


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def recommended_command(project: "Project", row: dict[str, Any]) -> str:
    """The denoise invocation to suggest for a flagged, not-yet-denoised clip."""
    if "windy" not in row["flags"] or row["use_denoised"]:
        return ""
    return f"ytedit denoise {project.slug} --clip {row['clip']} --engine elevenlabs"


def render_markdown(project: "Project", report: dict[str, Any]) -> str:
    """Render :func:`scan_project`'s report as a worst-first Markdown table."""
    thresholds = report["thresholds"]
    lines = [
        f"# Noise scan — {project.slug}",
        "",
        f"Generated: {report['generated']}  ",
        f"Scope: {'clips used unmuted in plan/timeline.json' if report['used_only'] else 'all clips with a transcript'}  ",
        (
            f"Thresholds: windy if `snr_db < {thresholds['snr_flag_db']}` and "
            f"`low_band_ratio > {thresholds['low_band_flag']}`; "
            f"noisy if `snr_db < {thresholds['snr_warn_db']}`."
        ),
        "",
        "| clip | snr (dB) | gap (dB) | speech (dB) | low band | gap s | speech s | flag | denoised | recommended |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["clips"]:
        flag = row["flags"][0] if row["flags"] else "-"
        denoised = f"yes ({row['denoise_engine']})" if row["use_denoised"] else "-"
        lines.append(
            f"| {row['clip']} | {row['snr_db']:.1f} | {row['gap_rms_db']:.1f} | "
            f"{row['speech_rms_db']:.1f} | {row['low_band_ratio']:.2f} | "
            f"{row['gap_seconds']:.1f} | {row['speech_seconds']:.1f} | {flag} | "
            f"{denoised} | {recommended_command(project, row)} |"
        )
    if report["skipped"]:
        lines += [
            "",
            f"Skipped (transcript present but under {MIN_SPAN_SECONDS}s of gap or speech signal): "
            + ", ".join(report["skipped"]),
        ]
    return "\n".join(lines) + "\n"


def write_report(project: "Project", report: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``analysis/noise_report.json`` and ``analysis/noise_report.md``.

    Returns:
        ``(json_path, md_path)``.
    """
    import json

    project.analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = project.analysis_dir / "noise_report.json"
    md_path = project.analysis_dir / "noise_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(project, report), encoding="utf-8")
    return json_path, md_path


# ----------------------------------------------------------------------
# denoise integration
# ----------------------------------------------------------------------
def windy_undenoised_clips(report: dict[str, Any]) -> list[str]:
    """Clip ids flagged ``windy`` that don't have ``use_denoised`` set yet."""
    return [r["clip"] for r in report["clips"] if "windy" in r["flags"] and not r["use_denoised"]]


def estimate_denoise_cost(project: "Project", clip_ids: Sequence[str], engine: str) -> float:
    """Estimate the USD cost of denoising ``clip_ids`` with ``engine``.

    ``local`` is free; ``elevenlabs`` is ``$/minute`` (``prices.elevenlabs.
    isolation_per_minute``, mirroring :func:`ytedit.media.audio.denoise_elevenlabs`)
    times the clips' full source-audio duration (not just their gap/speech
    signal — that's what the isolator actually bills for).
    """
    if engine != "elevenlabs":
        return 0.0
    from .audio import ISOLATION_USD_PER_MINUTE, audio_duration

    per_minute = project.settings.get("prices.elevenlabs.isolation_per_minute", ISOLATION_USD_PER_MINUTE)
    total_seconds = sum(audio_duration(project.audio_path(c)) for c in clip_ids)
    return total_seconds / 60.0 * float(per_minute)
