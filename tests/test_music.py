"""Offline tests for the music stage.

``ytedit.ai.music.ElevenLabs`` is monkeypatched with a fake composer, so cue
resolution, prompt building, sidecars, the manifest and the cost bookkeeping run
without generating (or paying for) a single second of audio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ytedit.ai.elevenlabs import MusicResult
from ytedit.ai.music import (
    DEFAULT_LENGTH_S,
    MusicCue,
    build_prompt,
    clamp_length_ms,
    cues_from_plan,
    estimate_cues,
    generate_music,
    get_preset,
    list_music,
    normalize_cue,
    pick_style_for,
    resolve_cues,
    styles_table,
)
from ytedit.cut import Beat, Cut, cut_path, load_cut, save_cut
from ytedit.cut import MusicCue as CutMusicCue
from ytedit.project import Project

FAKE_MP3 = b"ID3\x03\x00" + b"\x00" * 512


class FakeElevenLabs:
    """Stand-in for :class:`ytedit.ai.elevenlabs.ElevenLabs`."""

    calls: list[dict[str, Any]] = []
    raises: Exception | None = None

    def __init__(self, api_key: str, cost_callback=None, **kwargs: Any) -> None:
        self.cost_callback = cost_callback

    def compose_music(self, prompt: str, length_ms: int, **kwargs: Any) -> MusicResult:
        type(self).calls.append({"prompt": prompt, "length_ms": length_ms, **kwargs})
        if type(self).raises is not None:
            raise type(self).raises
        cost = length_ms / 60_000.0 * 0.15
        if self.cost_callback:
            self.cost_callback(
                service="elevenlabs", op="music", model=kwargs.get("model_id"),
                units=f"{length_ms / 1000:.1f}s", usd=cost,
            )
        return MusicResult(
            audio=FAKE_MP3,
            song_id=f"song-{len(type(self).calls)}",
            meta={
                "prompt": prompt,
                "music_length_ms": length_ms,
                "model_id": kwargs.get("model_id"),
                "generation_mode": kwargs.get("generation_mode"),
                "cost_usd": cost,
            },
        )

    def close(self) -> None:
        pass


@pytest.fixture()
def music_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    project = Project.create("t-mus", language="pl", root=tmp_path / "projects")
    FakeElevenLabs.calls = []
    FakeElevenLabs.raises = None
    monkeypatch.setattr("ytedit.ai.music.ElevenLabs", FakeElevenLabs)
    return project


def write_plan(project: Project, cues: list[dict[str, Any]]) -> None:
    """Write ``plan/edit_plan.json`` — only the *wording* fallback lives here."""
    project.edit_plan_file.write_text(
        json.dumps({"music_cues": cues}, ensure_ascii=False), encoding="utf-8"
    )


def write_cut(project: Project, cues: list[dict[str, Any]]) -> None:
    """Write ``plan/cut.json`` with one broll beat per cue, sized to its length.

    Since cut v2 a cue's length is a fact about the cut — the start of its
    ``from`` beat to the end of its ``to`` beat — so a cue can only be tested
    against real beats. Each cue dict takes ``length_s`` plus whatever
    style/mood/section wording it wants to carry.
    """
    project.save_state(
        {
            **project.load_state(),
            "clips": {"c001": {"id": "c001", "duration": 600.0, "has_audio": False}},
        }
    )
    beats: list[Beat] = []
    music: list[CutMusicCue] = []
    at = 0.0
    for i, cue in enumerate(cues, 1):
        length = float(cue.get("length_s", 30.0))
        beat = Beat(kind="broll", clip="c001", **{"in": at}, out=at + length, audio="mute")
        beats.append(beat)
        at += length
        extra = {k: v for k, v in cue.items() if k in ("style", "mood", "section", "prompt")}
        music.append(
            CutMusicCue(
                id=str(cue.get("id") or f"m{i:03d}"),
                file=str(cue.get("file") or f"music/m{i:03d}.mp3"),
                **{"from": beat.uid},
                to=beat.uid,
                gain_db=float(cue.get("gain_db", -18.0)),
                **extra,
            )
        )
    save_cut(Cut(beats=beats, music=music), cut_path(project))


# --------------------------------------------------------------------------- #
# style presets
# --------------------------------------------------------------------------- #


def test_styles_table_reads_the_config(music_project: Project) -> None:
    table = styles_table(music_project.settings)
    assert "arrival-warm" in table and "night-mysterious" in table
    assert "prompt" in table["arrival-warm"]


def test_pick_style_for_matches_mood_tags(music_project: Project) -> None:
    settings = music_project.settings
    assert pick_style_for(["nocturnal", "mysterious"], settings) == "night-mysterious"
    assert pick_style_for(["calm", "spacious"], settings) == "nature-calm"
    assert pick_style_for(["reflective", "bittersweet"], settings) == "sunset-reflective"
    assert pick_style_for(["driving", "optimistic"], settings) == "adventure-drive"


def test_pick_style_for_falls_back(music_project: Project) -> None:
    settings = music_project.settings
    assert pick_style_for([], settings) == "arrival-warm"
    assert pick_style_for(["completely unrelated gibberish"], settings) == "arrival-warm"


def test_get_preset_rejects_unknown_style(music_project: Project) -> None:
    with pytest.raises(KeyError):
        get_preset(music_project.settings, "does-not-exist")


def test_build_prompt_uses_preset_formula_and_cue_hint(music_project: Project) -> None:
    preset = get_preset(music_project.settings, "arrival-warm")
    cue = MusicCue(id="01", style="arrival-warm", mood="warm", section="cold open")
    prompt = build_prompt(preset, cue)
    assert "Instrumental." in prompt
    assert "92 BPM" in prompt
    assert "Director's note for this cue" in prompt and "cold open; warm" in prompt
    assert "No vocals" in prompt
    assert "\n" not in prompt  # YAML folding collapsed
    assert build_prompt(preset, MusicCue(id="x", style="arrival-warm", prompt="custom")) == "custom"


# --------------------------------------------------------------------------- #
# cue resolution
# --------------------------------------------------------------------------- #


def test_clamp_length_stays_inside_the_api_window() -> None:
    assert clamp_length_ms(0.5) == 3_000
    assert clamp_length_ms(90) == 90_000
    assert clamp_length_ms(10_000) == 600_000


def test_normalize_cue_accepts_the_documented_shape(music_project: Project) -> None:
    cue = normalize_cue(
        {"id": "intro", "style": "city-energy", "length_s": 45, "mood": "energetic", "section": "hook"},
        0,
        music_project.settings,
    )
    assert cue == MusicCue(id="intro", style="city-energy", length_s=45.0, mood="energetic", section="hook")


def test_normalize_cue_accepts_the_planner_cue_sheet_shape(music_project: Project) -> None:
    cue = normalize_cue({"section": "Peak window", "s": 180.0, "e": 300.0, "mood": "energetic modern"}, 2, music_project.settings)
    assert cue.id == "03-peak-window"
    assert cue.length_s == 120.0
    assert cue.style == "city-energy"  # picked from the mood tags


def test_cues_come_from_the_cut(music_project: Project) -> None:
    write_cut(music_project, [{"id": "m001", "section": "outro", "length_s": 30.0,
                               "mood": "hopeful warm"}])
    assert len(cues_from_plan(music_project)) == 1
    cues = resolve_cues(music_project)
    assert [c.id for c in cues] == ["m001"]
    assert cues[0].style == "outro-hopeful" and cues[0].length_s == 30.0


def test_a_cue_length_is_its_beat_range_not_the_edit_plans_guess(music_project: Project) -> None:
    """The edit plan's seconds are a planning guess; the cut is the fact."""
    write_cut(music_project, [{"id": "m001", "section": "outro", "length_s": 42.0,
                               "mood": "hopeful warm"}])
    write_plan(music_project, [{"id": "m001", "section": "outro", "s": 0.0, "e": 999.0}])
    assert cues_from_plan(music_project)[0]["length_s"] == pytest.approx(42.0)


def test_a_cue_spanning_several_beats_is_as_long_as_the_whole_range(
    music_project: Project,
) -> None:
    write_cut(music_project, [{"id": "m001", "length_s": 20.0}, {"id": "m002", "length_s": 30.0}])
    cut = load_cut(cut_path(music_project))
    cut.music = [cut.music[0]]
    cut.music[0].to = cut.beats[1].uid
    save_cut(cut, cut_path(music_project))
    assert cues_from_plan(music_project)[0]["length_s"] == pytest.approx(50.0)


def test_a_cue_whose_beats_are_gone_is_skipped(music_project: Project) -> None:
    """A bed of the wrong length is worse than no bed — and is paid for."""
    write_cut(music_project, [{"id": "m001", "length_s": 20.0}])
    cut = load_cut(cut_path(music_project))
    cut.music[0].from_ = cut.music[0].to = "deadbeef"
    save_cut(cut, cut_path(music_project))
    assert cues_from_plan(music_project) == []


def test_cue_wording_falls_back_to_the_edit_plan(music_project: Project) -> None:
    """A migrated cut keeps only file/gain/fades; the direction comes from the plan."""
    write_cut(music_project, [{"id": "m001", "length_s": 30.0}])
    write_plan(music_project, [{"id": "m001", "section": "outro", "mood": "hopeful warm"}])
    cue = cues_from_plan(music_project)[0]
    assert cue["section"] == "outro" and cue["mood"] == "hopeful warm"
    assert resolve_cues(music_project)[0].style == "outro-hopeful"


def test_no_cut_means_no_cues(music_project: Project) -> None:
    write_plan(music_project, [{"section": "outro", "s": 0.0, "e": 30.0}])
    assert cues_from_plan(music_project) == []


def test_styles_argument_used_without_a_plan(music_project: Project) -> None:
    cues = resolve_cues(music_project, styles=["night-mysterious", "nature-calm"], length_s=40)
    assert [c.style for c in cues] == ["night-mysterious", "nature-calm"]
    assert [c.length_s for c in cues] == [40.0, 40.0]
    assert [c.id for c in cues] == ["01-night-mysterious", "02-nature-calm"]


def test_default_single_cue_from_project_mood(music_project: Project) -> None:
    cues = resolve_cues(music_project)
    assert len(cues) == 1
    assert cues[0].style == "arrival-warm"  # project.yaml music_mood
    assert cues[0].length_s == DEFAULT_LENGTH_S


def test_explicit_cues_win_over_the_plan(music_project: Project) -> None:
    write_cut(music_project, [{"id": "m001", "section": "outro", "length_s": 30.0,
                               "mood": "hopeful"}])
    cues = resolve_cues(music_project, cues=[{"id": "manual", "style": "market-bustle", "length_s": 12}])
    assert [(c.id, c.style, c.length_s) for c in cues] == [("manual", "market-bustle", 30.0)]  # clamped to min_length_s


def test_estimate_cues_matches_the_published_rate() -> None:
    cues = [MusicCue(id="a", style="x", length_s=60), MusicCue(id="b", style="x", length_s=120)]
    assert estimate_cues(cues) == pytest.approx(0.45)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def test_generate_music_writes_track_sidecar_and_manifest(music_project: Project) -> None:
    write_cut(music_project, [{"id": "arrival", "style": "arrival-warm", "length_s": 60,
                               "section": "intro", "mood": "warm"}])
    results = generate_music(music_project)
    assert [r.status for r in results] == ["done"]

    mp3 = music_project.music_dir / "arrival.mp3"
    assert mp3.read_bytes() == FAKE_MP3

    sidecar = json.loads((music_project.music_dir / "arrival.json").read_text(encoding="utf-8"))
    assert sidecar["id"] == "arrival"
    assert sidecar["file"] == "music/arrival.mp3"
    assert sidecar["style"] == "arrival-warm"
    assert sidecar["song_id"] == "song-1"
    assert sidecar["length_s"] == 60.0
    assert sidecar["cost_usd"] == pytest.approx(0.15)
    assert sidecar["gain_db"] == -18
    assert "Instrumental." in sidecar["prompt"]
    assert sidecar["generated_at"]

    call = FakeElevenLabs.calls[0]
    assert call["length_ms"] == 60_000
    assert call["force_instrumental"] is True
    assert call["generation_mode"] == "loop"
    assert call["model_id"] == "music_v2"
    assert call["output_format"] == "mp3_44100_192"

    manifest = json.loads((music_project.music_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 1
    assert manifest["total_cost_usd"] == pytest.approx(0.15)
    assert manifest["tracks"][0]["id"] == "arrival"


def test_generation_is_charged_to_the_ledger(music_project: Project) -> None:
    generate_music(music_project, cues=[{"id": "bed", "style": "nature-calm", "length_s": 30}])
    state = music_project.load_state()
    entry = [c for c in state["costs"] if c["op"] == "music"][0]
    assert entry["units"] == "30.0s"
    assert entry["usd"] == pytest.approx(0.075)
    assert state["stages"]["music"]["status"] == "done"
    assert state["stages"]["music"]["cost_usd"] == pytest.approx(0.075)


def test_existing_track_is_skipped_unless_forced(music_project: Project) -> None:
    cues = [{"id": "bed", "style": "nature-calm", "length_s": 30}]
    generate_music(music_project, cues=cues)
    assert len(FakeElevenLabs.calls) == 1

    results = generate_music(music_project, cues=cues)
    assert [r.status for r in results] == ["skipped"]
    assert len(FakeElevenLabs.calls) == 1

    generate_music(music_project, cues=cues, force=True)
    assert len(FakeElevenLabs.calls) == 2


def test_length_override_applies_to_every_cue(music_project: Project) -> None:
    generate_music(
        music_project,
        cues=[{"id": "a", "style": "nature-calm", "length_s": 30}, {"id": "b", "style": "city-energy", "length_s": 90}],
        length_s=20,
    )
    assert sorted(c["length_ms"] for c in FakeElevenLabs.calls) == [20_000, 20_000]


def test_failed_cue_is_reported_without_killing_the_stage(music_project: Project) -> None:
    FakeElevenLabs.raises = RuntimeError("ElevenLabs HTTP 500")
    results = generate_music(music_project, cues=[{"id": "bed", "style": "nature-calm", "length_s": 30}])
    assert results[0].status == "error" and "500" in results[0].error
    state = music_project.load_state()
    assert state["stages"]["music"]["status"] == "error"


def test_budget_cap_blocks_generation(music_project: Project) -> None:
    from ytedit.costs import BudgetExceeded

    with music_project.edit_state() as state:
        state["budget_usd"] = 0.01
    with pytest.raises(BudgetExceeded):
        generate_music(music_project, cues=[{"id": "bed", "style": "nature-calm", "length_s": 120}])
    assert not FakeElevenLabs.calls


def test_list_music_reads_sidecars_and_ignores_the_manifest(music_project: Project) -> None:
    assert list_music(music_project) == []
    generate_music(
        music_project,
        cues=[{"id": "a", "style": "nature-calm", "length_s": 30}, {"id": "b", "style": "city-energy", "length_s": 30}],
    )
    tracks = list_music(music_project)
    assert [t["id"] for t in tracks] == ["a", "b"]
    assert all(t["file"].startswith("music/") for t in tracks)
