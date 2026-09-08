"""Tests for ``ytedit.media.audio``: loudness, silence, ducking, mixing."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import pytest

from ytedit.config import load_settings
from ytedit.media import audio as A

LOUDNORM_STDERR = """\
[Parsed_loudnorm_0 @ 0x7f8] \n\
{
\t"input_i" : "-23.45",
\t"input_tp" : "-3.20",
\t"input_lra" : "7.10",
\t"input_thresh" : "-33.80",
\t"output_i" : "-14.02",
\t"output_tp" : "-1.10",
\t"output_lra" : "6.90",
\t"output_thresh" : "-24.30",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.02"
}
"""

SILENCE_STDERR = """\
[silencedetect @ 0x1] silence_start: 1.5
[silencedetect @ 0x1] silence_end: 2.75 | silence_duration: 1.25
[silencedetect @ 0x1] silence_start: 6.0
"""


def sine(path: Path, seconds: float = 4.0, freq: int = 440, volume: float = 0.5) -> Path:
    """Write a mono-ish stereo sine WAV with ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i",
         f"sine=frequency={freq}:sample_rate=48000:duration={seconds}",
         "-af", f"volume={volume}", "-c:a", "pcm_s16le", "-ac", "2", str(path)],
        check=True,
    )
    return path


# ----------------------------------------------------------------------
# gain helpers
# ----------------------------------------------------------------------
def test_db_to_linear_round_trips() -> None:
    assert A.db_to_linear(0) == pytest.approx(1.0)
    assert A.db_to_linear(-6) == pytest.approx(0.5012, abs=1e-3)
    assert A.db_to_linear(-90) == 0.0
    assert A.linear_to_db(0.5) == pytest.approx(-6.02, abs=0.01)
    assert A.linear_to_db(0.0) == float("-inf")


# ----------------------------------------------------------------------
# loudnorm
# ----------------------------------------------------------------------
def test_parse_loudnorm_reads_the_last_report() -> None:
    measured = A.parse_loudnorm(LOUDNORM_STDERR)
    assert measured["input_i"] == pytest.approx(-23.45)
    assert measured["input_tp"] == pytest.approx(-3.20)
    assert measured["input_lra"] == pytest.approx(7.10)
    assert measured["input_thresh"] == pytest.approx(-33.80)
    assert measured["target_offset"] == pytest.approx(0.02)


def test_parse_loudnorm_handles_infinite_silence() -> None:
    measured = A.parse_loudnorm(LOUDNORM_STDERR.replace('"-23.45"', '"-inf"'))
    assert math.isinf(measured["input_i"])


def test_parse_loudnorm_without_a_report_raises() -> None:
    with pytest.raises(ValueError):
        A.parse_loudnorm("nothing to see here")


def test_loudnorm_filter_single_pass() -> None:
    assert A.loudnorm_filter() == "loudnorm=I=-14:TP=-1:LRA=11"


def test_loudnorm_filter_second_pass_is_linear_and_carries_the_measurement() -> None:
    chain = A.loudnorm_filter(A.parse_loudnorm(LOUDNORM_STDERR))
    assert chain.startswith("loudnorm=I=-14:TP=-1:LRA=11:")
    assert "linear=true" in chain
    assert "measured_I=-23.450000" in chain
    assert "measured_TP=-3.200000" in chain
    assert "measured_LRA=7.100000" in chain
    assert "measured_thresh=-33.800000" in chain
    assert "offset=0.020000" in chain


def test_loudnorm_filter_falls_back_when_the_measurement_is_silent() -> None:
    measured = A.parse_loudnorm(LOUDNORM_STDERR.replace('"-23.45"', '"-inf"'))
    assert A.loudnorm_filter(measured) == "loudnorm=I=-14:TP=-1:LRA=11"


def test_measure_and_normalize_hit_the_target(tmp_path: Path) -> None:
    source = sine(tmp_path / "in.wav", seconds=5.0, volume=0.05)
    before = A.measure_loudness(source)
    assert before["input_i"] < -20  # deliberately quiet
    out = A.normalize(source, tmp_path / "out.wav", two_pass=True)
    assert out.exists()
    after = A.measure_loudness(out)
    assert after["input_i"] == pytest.approx(-14.0, abs=1.0)
    assert after["input_tp"] <= -0.9


# ----------------------------------------------------------------------
# silence
# ----------------------------------------------------------------------
def test_parse_silence_pairs_starts_and_ends() -> None:
    assert A.parse_silence(SILENCE_STDERR) == [(1.5, 2.75)]


def test_parse_silence_closes_a_trailing_silence_at_the_duration() -> None:
    assert A.parse_silence(SILENCE_STDERR, duration=8.0) == [(1.5, 2.75), (6.0, 8.0)]


def test_detect_silence_finds_a_real_gap(tmp_path: Path) -> None:
    path = tmp_path / "gap.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=48000:duration=4",
         "-af", "volume=enable='between(t,1,3)':volume=0",
         "-c:a", "pcm_s16le", "-ac", "2", str(path)],
        check=True,
    )
    ranges = A.detect_silence(path, noise_db=-45, min_d=0.5)
    assert ranges, "expected a detected silence"
    start, end = ranges[0]
    assert start == pytest.approx(1.0, abs=0.15)
    assert end == pytest.approx(3.0, abs=0.15)


# ----------------------------------------------------------------------
# ducking
# ----------------------------------------------------------------------
def test_duck_regions_merge_and_apply_the_pre_roll() -> None:
    regions = A.duck_regions([(2.0, 3.0), (3.2, 4.0)], duration=10.0)
    assert len(regions) == 1
    a0, a1, b0, b1 = regions[0]
    assert a1 == pytest.approx(1.85)          # 2.0 - pre_roll 0.15
    assert a0 == pytest.approx(1.70)          # ... - attack 0.15
    assert b0 == pytest.approx(4.0)
    assert b1 == pytest.approx(4.6)           # ... + release 0.6


def test_duck_envelope_ramps_from_zero_to_one_and_back() -> None:
    regions = A.duck_regions([(2.0, 3.0)], duration=10.0)
    assert A.duck_envelope(regions, 0.0) == 0.0
    assert A.duck_envelope(regions, 1.775) == pytest.approx(0.5, abs=0.02)
    assert A.duck_envelope(regions, 2.5) == 1.0
    assert A.duck_envelope(regions, 3.3) == pytest.approx(0.5, abs=0.02)
    assert A.duck_envelope(regions, 9.0) == 0.0


def test_duck_automation_writes_ramped_sendcmd_lines() -> None:
    content = A.duck_automation([(2.0, 3.0)], duration=10.0, amount_db=-12, base_gain_db=-18)
    assert content.startswith("# ytedit ducking automation")
    commands = re.findall(r"^(\d+\.\d{3}) volume volume (\d+\.\d+);$", content, re.M)
    assert len(commands) >= 6, content
    times = [float(t) for t, _ in commands]
    gains = [float(g) for _, g in commands]
    assert times == sorted(times), "commands must be in ascending time order"
    base = A.db_to_linear(-18)
    ducked = A.db_to_linear(-30)
    assert gains[0] == pytest.approx(base, rel=1e-3)
    assert min(gains) == pytest.approx(ducked, rel=1e-3)
    assert gains[-1] == pytest.approx(base, rel=1e-3)
    # the attack is genuinely ramped, not a single step
    ramping = [g for t, g in zip(times, gains) if 1.70 < t < 1.85]
    assert ramping, "expected intermediate points inside the attack ramp"


def test_duck_automation_without_speech_is_a_single_point() -> None:
    content = A.duck_automation([], duration=10.0, base_gain_db=-18)
    assert len(re.findall(r"volume volume", content)) == 1


def test_duck_volume_expr_matches_the_automation_numerically() -> None:
    expr = A.duck_volume_expr([(2.0, 3.0)], 10.0, amount_db=-12, base_gain_db=-18)
    assert expr.startswith("volume='")
    assert expr.endswith("':eval=frame")
    assert f"{A.db_to_linear(-18):.6f}".rstrip("0") in expr


def test_duck_volume_expr_without_speech_is_a_constant() -> None:
    assert A.duck_volume_expr([], 10.0, base_gain_db=-18) == f"volume={A.db_to_linear(-18):.6f}".rstrip("0")


# ----------------------------------------------------------------------
# mute ranges
# ----------------------------------------------------------------------
def test_mute_ranges_expr_groups_by_gain() -> None:
    chain = A.mute_ranges_expr([(1.0, 2.0, -60.0), (5.0, 6.0, -60.0), (8.0, 9.0, -12.0)])
    parts = chain.split(",volume=")
    assert len(parts) == 2
    assert "between(t,1,2)+between(t,5,6)" in chain
    assert "between(t,8,9)" in chain
    assert "volume=0" in chain                       # -60 dB rounds to ~0.001
    assert f"{A.db_to_linear(-12):.6f}".rstrip("0") in chain


def test_mute_ranges_expr_ignores_empty_ranges() -> None:
    assert A.mute_ranges_expr([(2.0, 2.0, -60.0)]) == ""


# ----------------------------------------------------------------------
# speech leveling
# ----------------------------------------------------------------------
def test_speech_gate_expr_mutes_the_complement() -> None:
    expr = A.speech_gate_expr([(1.0, 2.0), (3.0, 3.5)], duration=4.0)
    # gaps are [0, 1), [2, 3), [3.5, 4)
    assert "between(t,0,1)" in expr
    assert "between(t,2,3)" in expr
    assert "between(t,3.5,4)" in expr
    assert "volume=0" in expr  # SILENCE_DB rounds to ~0


def test_speech_gate_expr_is_empty_when_speech_covers_everything() -> None:
    assert A.speech_gate_expr([(0.0, 4.0)], duration=4.0) == ""


def test_measure_speech_gain_levels_a_quiet_cut_toward_the_target(tmp_path: Path) -> None:
    # A quiet 4 s tone; only the middle half (coverage 50% < the default 60%
    # threshold) is "speech", so the measurement gates to it.
    quiet = sine(tmp_path / "quiet.wav", seconds=4.0, volume=0.05)
    gain, measured = A.measure_speech_gain(
        quiet, start=0.0, raw_duration=4.0,
        speech_ranges=[(1.0, 3.0)], span=4.0,
        target_lufs=-16.0, max_gain_db=10.0,
    )
    assert math.isfinite(measured) and measured < -16.0  # confirms it really is quiet
    assert gain > 0  # needs boosting toward the target
    assert gain <= 10.0 + 1e-6


def test_measure_speech_gain_clamps_to_max_gain_db(tmp_path: Path) -> None:
    very_quiet = sine(tmp_path / "very_quiet.wav", seconds=2.0, volume=0.01)
    gain, _measured = A.measure_speech_gain(
        very_quiet, start=0.0, raw_duration=2.0,
        speech_ranges=[(0.0, 2.0)], span=2.0,
        target_lufs=-14.0, max_gain_db=3.0,
    )
    assert gain == pytest.approx(3.0)


def test_measure_speech_gain_without_speech_is_a_no_op() -> None:
    gain, measured = A.measure_speech_gain(
        "/does/not/matter.wav", start=0.0, raw_duration=2.0,
        speech_ranges=[], span=2.0,
    )
    assert gain == 0.0
    assert math.isnan(measured)


def test_level_voice_file_gain_moves_toward_the_target_and_is_cached(tmp_path: Path) -> None:
    pickup = sine(tmp_path / "pickup.wav", seconds=3.0, volume=0.05)  # quiet
    gain, measured = A.level_voice_file_gain(pickup, target_lufs=-16.0, max_gain_db=10.0)
    assert math.isfinite(measured)
    assert gain > 0

    cache = tmp_path / "pickup.wav.loudness.json"
    assert cache.exists()
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["gain_db"] == pytest.approx(gain)
    assert data["measured_lufs"] == pytest.approx(measured)

    # Re-measuring the same (unchanged) file must not touch ffmpeg again.
    def _boom(*_args, **_kwargs):
        raise AssertionError("measure_loudness should not run again for a cached file")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(A, "measure_loudness", _boom)
    try:
        gain2, measured2 = A.level_voice_file_gain(pickup, target_lufs=-16.0, max_gain_db=10.0)
    finally:
        monkeypatch.undo()
    assert gain2 == pytest.approx(gain)
    assert measured2 == pytest.approx(measured)


def test_level_voice_file_gain_remeasures_when_the_target_changes(tmp_path: Path) -> None:
    # A generous clamp so neither target below is actually clamped — this
    # test is about the cache keying on the target, not about clamping.
    pickup = sine(tmp_path / "pickup2.wav", seconds=2.0, volume=0.05)
    gain_a, _ = A.level_voice_file_gain(pickup, target_lufs=-16.0, max_gain_db=40.0)
    gain_b, _ = A.level_voice_file_gain(pickup, target_lufs=-20.0, max_gain_db=40.0)
    assert gain_b == pytest.approx(gain_a - 4.0, abs=0.05)


# ----------------------------------------------------------------------
# voice cleanup
# ----------------------------------------------------------------------
def test_voice_cleanup_levels() -> None:
    assert A.voice_cleanup_chain("none") == ""
    assert A.voice_cleanup_chain("light").startswith("highpass=f=80,acompressor")
    full = A.voice_cleanup_chain("full")
    assert "afftdn" in full and "deesser" in full and full.startswith("highpass=f=80")
    with pytest.raises(ValueError):
        A.voice_cleanup_chain("extreme")


# ----------------------------------------------------------------------
# the programme mix
# ----------------------------------------------------------------------
def test_mix_program_loops_short_music_and_ducks_under_speech(tmp_path: Path) -> None:
    voice = sine(tmp_path / "voice.wav", seconds=8.0, freq=440, volume=0.4)
    music = sine(tmp_path / "music.wav", seconds=3.0, freq=180, volume=0.5)
    out = tmp_path / "mix.wav"
    cmd = tmp_path / "duck.cmd"

    A.mix_program(
        voice,
        [{"file": str(music), "at": 0.0, "end": 8.0, "gain_db": -18,
          "fade_in": 0.5, "fade_out": 0.5, "duck": True}],
        [],
        [(2.0, 4.0)],
        out,
        settings=load_settings(),
        duration=8.0,
        cmd_file=cmd,
    )
    assert out.exists()
    # the mix keeps the length of the voice bus even though the music is 3 s
    assert A.audio_duration(out) == pytest.approx(8.0, abs=0.05)
    assert cmd.exists() and "volume volume" in cmd.read_text()


def test_mix_program_without_cues_passes_the_bus_through(tmp_path: Path) -> None:
    voice = sine(tmp_path / "voice.wav", seconds=3.0)
    out = A.mix_program(voice, [], [], [], tmp_path / "mix.wav", duration=3.0)
    assert A.audio_duration(out) == pytest.approx(3.0, abs=0.05)


# ----------------------------------------------------------------------
# denoise
# ----------------------------------------------------------------------
def noisy_speech(path: Path, seconds: float = 4.0, quiet_from: float = 2.0) -> Path:
    """Write a WAV that is tone + noise, then noise only after ``quiet_from``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-f", "lavfi", "-i",
         f"anoisesrc=color=white:sample_rate=48000:duration={seconds}:amplitude=0.06",
         "-filter_complex",
         f"[0:a]volume=enable='between(t,{quiet_from},{seconds})':volume=0[s];"
         "[s][1:a]amix=inputs=2:normalize=0[a]",
         "-map", "[a]", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1", str(path)],
        check=True,
    )
    return path


def _register(project, clip_id: str, wav: Path) -> None:
    """Put a clip that owns ``wav`` into the project registry."""
    project.add_clip(
        {"id": clip_id, "order": 1, "duration": round(A.audio_duration(wav), 3),
         "width": 1920, "height": 1080, "has_audio": True,
         "audio": project.rel(wav)}
    )


def test_local_denoise_chain_tracks_the_measured_noise_floor() -> None:
    assert A.local_denoise_chain(-38.0) == "highpass=f=110,afftdn=nr=24:nf=-30,adeclick"
    # Unknown / silent input falls back, and nf is clamped into afftdn's range.
    assert "nf=-30" in A.local_denoise_chain(None)
    assert "nf=-20" in A.local_denoise_chain(-5.0)
    assert "nf=-80" in A.local_denoise_chain(-100.0)


def test_measure_noise_floor_finds_the_quiet_part(tmp_path: Path) -> None:
    wav = noisy_speech(tmp_path / "noisy.wav")
    floor = A.measure_noise_floor(wav)
    loud = A.noise_floor(wav, 0.5, 1.5)["mean_db"]
    assert floor < loud - 3.0


def test_local_denoise_lowers_the_noise_floor(project, tmp_path: Path) -> None:
    wav = noisy_speech(project.audio_path("c001"))
    _register(project, "c001", wav)
    before = A.noise_floor(wav, 2.5, 3.5)["mean_db"]

    results = A.denoise_clips(project, ["c001"], engine="local")
    assert [r["status"] for r in results] == ["done"]

    out = A.denoised_path(project, "c001", "local")
    active = A.denoised_path(project, "c001")
    assert out.exists() and active.exists()
    after = A.noise_floor(active, 2.5, 3.5)["mean_db"]
    assert after < before - 5.0, f"noise floor {before:.1f} -> {after:.1f} dB"
    # The tone survives roughly intact.
    assert A.noise_floor(active, 0.5, 1.5)["mean_db"] > before - 3.0

    clip = project.load_state()["clips"]["c001"]
    assert clip["use_denoised"] is True
    assert clip["denoised"] == "media/audio/c001.denoised.wav"
    assert clip["denoise_engine"] == "local"


def test_denoise_is_cached_until_forced(project) -> None:
    noisy_speech(project.audio_path("c001"), seconds=2.0, quiet_from=1.0)
    _register(project, "c001", project.audio_path("c001"))
    A.denoise_clips(project, ["c001"], engine="local")
    again = A.denoise_clips(project, ["c001"], engine="local")
    assert again[0]["status"] == "cached"
    assert A.denoise_clips(project, ["c001"], engine="local", force=True)[0]["status"] == "done"


def test_set_use_denoised_keeps_the_file(project) -> None:
    noisy_speech(project.audio_path("c001"), seconds=2.0, quiet_from=1.0)
    _register(project, "c001", project.audio_path("c001"))
    A.denoise_clips(project, ["c001"], engine="local")

    A.set_use_denoised(project, ["c001"], enabled=False)
    assert project.load_state()["clips"]["c001"]["use_denoised"] is False
    assert A.denoised_path(project, "c001").exists()

    A.set_use_denoised(project, ["c001"], enabled=True)
    assert project.load_state()["clips"]["c001"]["use_denoised"] is True


def test_unknown_engine_and_unknown_clip_are_refused(project) -> None:
    with pytest.raises(A.DenoiseError):
        A.denoise_clips(project, ["c001"], engine="magic")
    with pytest.raises(A.DenoiseError):
        A.denoise_clips(project, ["c999"], engine="local")


def test_elevenlabs_engine_writes_an_aligned_wav_and_charges(
    project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API is faked: no key, no network, no money."""
    src = noisy_speech(project.audio_path("c001"), seconds=4.0, quiet_from=2.0)
    _register(project, "c001", src)

    # What the endpoint would return: a shorter mp3, to prove we re-align it.
    mp3 = tmp_path / "isolated.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-t", "3.4",
         "-i", str(src), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)],
        check=True,
    )
    calls: list[dict] = []

    class FakeElevenLabs:
        def __init__(self, api_key: str, cost_callback=None, **kwargs) -> None:
            self.cost_callback = cost_callback

        def isolate_audio(self, path, duration_s=None, **kwargs) -> bytes:
            calls.append({"path": Path(path), "duration_s": duration_s})
            if self.cost_callback:
                self.cost_callback(
                    service="elevenlabs", op="isolate", model=None,
                    units=f"{duration_s:.1f}s", usd=duration_s / 60.0 * 0.12,
                )
            return mp3.read_bytes()

        def close(self) -> None:
            pass

    monkeypatch.setattr("ytedit.ai.elevenlabs.ElevenLabs", FakeElevenLabs)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    results = A.denoise_clips(project, ["c001"], engine="elevenlabs")
    assert [r["status"] for r in results] == ["done"]
    assert calls[0]["path"] == src
    assert calls[0]["duration_s"] == pytest.approx(4.0, abs=0.05)

    out = A.denoised_path(project, "c001", "elevenlabs")
    assert out.exists()
    # Padded back to the source length: the renderer applies the same -ss/-t.
    assert A.audio_duration(out) == pytest.approx(A.audio_duration(src), abs=0.05)

    state = project.load_state()
    assert state["clips"]["c001"]["denoise_engine"] == "elevenlabs"
    charged = [c for c in state["costs"] if c["stage"] == "denoise"]
    assert charged and charged[0]["usd"] == pytest.approx(0.008, abs=0.001)


def test_ab_preview_is_twice_the_window(project) -> None:
    src = noisy_speech(project.audio_path("c001"), seconds=20.0, quiet_from=10.0)
    _register(project, "c001", src)
    A.denoise_clips(project, ["c001"], engine="local")

    ab = A.denoise_ab_preview(project, "c001", length=3.0)
    assert ab.name == "c001.denoise_ab.wav"
    assert A.audio_duration(ab) == pytest.approx(6.0, abs=0.1)


def test_ab_preview_needs_a_denoised_file(project) -> None:
    noisy_speech(project.audio_path("c001"), seconds=2.0, quiet_from=1.0)
    _register(project, "c001", project.audio_path("c001"))
    with pytest.raises(A.DenoiseError):
        A.denoise_ab_preview(project, "c001")
