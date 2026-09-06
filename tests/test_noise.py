"""Tests for ``ytedit.media.noise``: synthetic speech + gap signals -> flags.

Everything here is offline: numpy-synthesized WAVs and hand-written
transcripts on disk, no ffmpeg and no API calls.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from ytedit.media import noise as N
from ytedit.project import Project

SAMPLE_RATE = 48000
DURATION = 9.0

#: Six 0.4s "words" a second apart, spanning 1.0s..5.8s.
WORDS = [(1.0 + i * 1.0, 1.0 + i * 1.0 + 0.4, f"word{i}") for i in range(6)]


def _synthesize(gap: str) -> np.ndarray:
    """Build a mono float64 buffer: speech-like tone in the words, ``gap`` elsewhere."""
    n = int(DURATION * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    samples = np.zeros(n, dtype=np.float64)
    if gap == "quiet":
        samples[:] = np.random.default_rng(0).normal(0.0, 0.001, size=n)
    elif gap == "rumble":
        # A loud 60 Hz tone: wind rumble on a phone mic is exactly this shape —
        # almost all energy well under LOW_BAND_HZ (150 Hz).
        samples[:] = 0.45 * np.sin(2 * np.pi * 60.0 * t)
    else:  # pragma: no cover - test misuse
        raise ValueError(gap)
    for start, end, _ in WORDS:
        i0, i1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        samples[i0:i1] = 0.5 * np.sin(2 * np.pi * 300.0 * t[i0:i1])
    return np.clip(samples, -1.0, 1.0)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _add_clip(project: Project, clip_id: str, gap: str, words=WORDS) -> None:
    """Register a clip, write its transcript and a synthesized work WAV."""
    project.add_clip(
        {
            "id": clip_id, "order": int(clip_id[1:]), "duration": DURATION,
            "width": 1920, "height": 1080, "orientation": "horizontal", "has_audio": True,
        }
    )
    project.transcript_path(clip_id).write_text(
        json.dumps(
            {"clip": clip_id, "language": "pl", "words": [{"t": t, "s": s, "e": e} for s, e, t in words]}
        ),
        encoding="utf-8",
    )
    _write_wav(project.audio_path(clip_id), _synthesize(gap))


# ----------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------
def test_clean_speech_with_quiet_gaps_is_not_flagged(project: Project) -> None:
    _add_clip(project, "c001", gap="quiet")
    report = N.scan_project(project, used_only=False)
    assert len(report["clips"]) == 1
    row = report["clips"][0]
    assert row["clip"] == "c001"
    assert row["flags"] == []
    assert row["snr_db"] > N.DEFAULT_SNR_WARN_DB
    assert row["low_band_ratio"] < N.DEFAULT_LOW_BAND_FLAG
    assert row["gap_seconds"] > N.MIN_SPAN_SECONDS
    assert row["speech_seconds"] > N.MIN_SPAN_SECONDS


def test_low_rumble_in_gaps_is_flagged_windy(project: Project) -> None:
    _add_clip(project, "c002", gap="rumble")
    report = N.scan_project(project, used_only=False)
    row = report["clips"][0]
    assert row["clip"] == "c002"
    assert row["flags"] == ["windy"]
    assert row["low_band_ratio"] > N.DEFAULT_LOW_BAND_FLAG
    assert row["snr_db"] < N.DEFAULT_SNR_FLAG_DB


def test_worst_clip_sorts_first(project: Project) -> None:
    _add_clip(project, "c001", gap="quiet")
    _add_clip(project, "c002", gap="rumble")
    report = N.scan_project(project, used_only=False)
    assert [row["clip"] for row in report["clips"]] == ["c002", "c001"]


def test_flag_clip_windy_takes_priority_over_noisy() -> None:
    metrics = {"snr_db": 0.5, "low_band_ratio": 0.95}
    assert N.flag_clip(metrics) == ["windy"]


def test_flag_clip_noisy_when_snr_low_but_not_wind_shaped() -> None:
    metrics = {"snr_db": 1.3, "low_band_ratio": 0.38}
    assert N.flag_clip(metrics) == ["noisy"]


def test_flag_clip_clean_when_snr_is_healthy() -> None:
    metrics = {"snr_db": 8.0, "low_band_ratio": 0.24}
    assert N.flag_clip(metrics) == []


# ----------------------------------------------------------------------
# eligibility
# ----------------------------------------------------------------------
def test_clip_with_too_few_words_is_not_a_candidate(project: Project) -> None:
    project.add_clip(
        {"id": "c003", "order": 3, "duration": 5.0, "width": 1920, "height": 1080,
         "orientation": "horizontal", "has_audio": True}
    )
    project.transcript_path("c003").write_text(
        json.dumps({"clip": "c003", "language": "pl", "words": [{"t": "hi", "s": 0.5, "e": 0.8}]}),
        encoding="utf-8",
    )
    with pytest.raises(N.NoiseError):
        N.scan_project(project, used_only=False)


def test_scan_clip_returns_none_for_too_short_a_clip(project: Project) -> None:
    # Only a sliver of gap around four back-to-back words: not enough gap signal.
    words = [(0.0, 0.2, "a"), (0.2, 0.4, "b"), (0.4, 0.6, "c"), (0.6, 0.8, "d")]
    project.add_clip(
        {"id": "c004", "order": 4, "duration": 1.0, "width": 1920, "height": 1080,
         "orientation": "horizontal", "has_audio": True}
    )
    project.transcript_path("c004").write_text(
        json.dumps({"clip": "c004", "language": "pl", "words": [{"t": t, "s": s, "e": e} for s, e, t in words]}),
        encoding="utf-8",
    )
    samples = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float64)
    _write_wav(project.audio_path("c004"), samples)
    assert N.scan_clip(project, "c004") is None


# ----------------------------------------------------------------------
# used-only filtering
# ----------------------------------------------------------------------
def test_used_only_filters_to_unmuted_timeline_clips(project: Project) -> None:
    from ytedit.timeline import VideoSegment, new_timeline

    _add_clip(project, "c005", gap="quiet")
    _add_clip(project, "c006", gap="rumble")
    tl = new_timeline()
    tl.tracks.video = [VideoSegment(id="s1", clip="c005", **{"in": 0.0}, out=1.0)]
    tl.save(project.timeline_file)

    report = N.scan_project(project, used_only=True)
    assert {row["clip"] for row in report["clips"]} == {"c005"}


def test_used_only_has_no_effect_without_a_timeline(project: Project) -> None:
    _add_clip(project, "c007", gap="quiet")
    assert N.used_clip_ids(project) is None
    report = N.scan_project(project, used_only=True)
    assert {row["clip"] for row in report["clips"]} == {"c007"}


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def test_write_report_creates_json_and_markdown(project: Project) -> None:
    _add_clip(project, "c008", gap="rumble")
    report = N.scan_project(project, used_only=False)
    json_path, md_path = N.write_report(project, report)
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.parent == project.analysis_dir

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["clips"][0]["clip"] == "c008"

    markdown = md_path.read_text(encoding="utf-8")
    assert "windy" in markdown
    assert f"ytedit denoise {project.slug} --clip c008 --engine elevenlabs" in markdown


def test_recommended_command_empty_when_already_denoised(project: Project) -> None:
    _add_clip(project, "c009", gap="rumble")
    project.set_clip_stage("c009", "denoise", "done", use_denoised=True, denoise_engine="elevenlabs")
    report = N.scan_project(project, used_only=False)
    row = report["clips"][0]
    assert row["use_denoised"] is True
    assert N.recommended_command(project, row) == ""


# ----------------------------------------------------------------------
# denoise-cost estimate
# ----------------------------------------------------------------------
def test_estimate_denoise_cost_is_zero_for_local_engine(project: Project) -> None:
    _add_clip(project, "c010", gap="rumble")
    assert N.estimate_denoise_cost(project, ["c010"], "local") == 0.0


def test_estimate_denoise_cost_scales_with_clip_duration(project: Project) -> None:
    _add_clip(project, "c011", gap="rumble")
    cost = N.estimate_denoise_cost(project, ["c011"], "elevenlabs")
    expected = DURATION / 60.0 * 0.12
    assert cost == pytest.approx(expected, rel=0.05)


def test_windy_undenoised_clips_excludes_already_denoised(project: Project) -> None:
    report = {
        "clips": [
            {"clip": "c012", "flags": ["windy"], "use_denoised": False},
            {"clip": "c013", "flags": ["windy"], "use_denoised": True},
            {"clip": "c014", "flags": ["noisy"], "use_denoised": False},
        ]
    }
    assert N.windy_undenoised_clips(report) == ["c012"]
