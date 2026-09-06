"""Tests for ``ytedit.ai.plan`` — post-processing, pacing rules, draft safety.

Every test here is offline: the planner model is replaced with a fake
``OpenRouter`` that returns a canned answer, so nothing costs money.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ytedit.ai.plan import (
    DEFAULT_PLAN_MAX_TOKENS,
    MAX_PLAN_MAX_TOKENS,
    SCHEMA_HINT,
    EditPlan,
    PlanError,
    build_timeline,
    load_footage_log,
    pacing_report,
    plan,
    subtract_ranges,
)
from ytedit.ai.openrouter import ChatResult, OpenRouterError
from ytedit.ai.prompts import load as load_prompt
from ytedit.project import Project
from ytedit.timeline import Timeline, VideoSegment, new_timeline


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
CLIPS: dict[str, dict[str, Any]] = {
    "c001": {"id": "c001", "order": 1, "duration": 60.0, "width": 1920, "height": 1080,
             "orientation": "horizontal"},
    "c002": {"id": "c002", "order": 2, "duration": 40.0, "width": 1920, "height": 1080,
             "orientation": "horizontal"},
    "c003": {"id": "c003", "order": 3, "duration": 30.0, "width": 1080, "height": 1920,
             "orientation": "vertical"},
}

FOOTAGE_LOG: dict[str, Any] = {
    "project": "t-proj",
    "clips": [
        {
            "clip": "c001",
            "summary": "Instrukcja na początku, potem narracja o tramwaju.",
            "kind": "a-roll",
            "instructions": [
                {"s": 0.0, "e": 4.0, "text": "to na koniec jako podsumowanie",
                 "action": "move_to_end"}
            ],
            "takes": [],
            "background_music": [],
            "visual": {"quality": 0.8, "best_frames": [12.0], "thumbnail_candidate": True},
            "hooks": ["tramwaj był tak pełny, że poszliśmy pieszo"],
            "numbers": ["bilet za 3 EUR"],
            "topics": ["tramwaj 28"],
            "location": {"name": "Alfama", "city": "Lizbona", "country": "Portugalia"},
        },
        {
            "clip": "c002",
            "summary": "Dwa podejścia do tego samego zdania.",
            "kind": "a-roll",
            "instructions": [],
            "takes": [
                {
                    "topic": "cena pastéis de nata",
                    "attempts": [{"s": 5.0, "e": 12.0}, {"s": 14.0, "e": 22.0}],
                    "keep": 1,
                    "reason": "ostatnie podejście, czystszy dźwięk",
                }
            ],
            "background_music": [
                {"s": 25.0, "e": 35.0, "confidence": 0.8, "suggest": "mute"}
            ],
            "visual": {"quality": 0.7, "best_frames": [8.0], "thumbnail_candidate": False},
            "hooks": [],
            "numbers": ["1.20 EUR za pastel"],
            "topics": ["pastéis de nata"],
            "location": {"name": "Belém", "city": "Lizbona", "country": "Portugalia"},
        },
        {
            "clip": "c003",
            "summary": "Pionowe ujęcie z punktu widokowego.",
            "kind": "silent-broll",
            "instructions": [],
            "takes": [],
            "background_music": [],
            "visual": {"quality": 0.9, "best_frames": [10.0], "thumbnail_candidate": True},
            "hooks": [],
            "numbers": [],
            "topics": ["miradouro"],
            "location": {"name": "Miradouro", "city": "Lizbona", "country": "Portugalia"},
        },
    ],
}

LLM_PLAN: dict[str, Any] = {
    "story": {
        "title_working": "Lizbona w jeden dzień",
        "premise": "Jeden dzień w Lizbonie za mniej niż 30 euro.",
        "hook_idea": "Najbardziej zatłoczony tramwaj w Europie.",
        "beats": [
            {"label": "cold open", "at_s_target": 0.0, "clips": ["c003"], "description": "widok"},
            {"label": "promise", "at_s_target": 7.0, "clips": ["c001"], "description": "stawka"},
        ],
    },
    "cold_open": [{"clip": "c003", "in": 8.0, "out": 12.0, "why": "najlepszy widok"}],
    # c001 0-20 straddles the 0-4 instruction; c002 4-25 straddles the rejected take 5-12.
    "segments": [
        {"clip": "c003", "in": 8.0, "out": 12.0, "role": "cold-open", "notes": "widok"},
        {"clip": "c001", "in": 0.0, "out": 20.0, "role": "a-roll", "notes": "tramwaj"},
        {"clip": "c002", "in": 4.0, "out": 25.0, "role": "a-roll", "notes": "nata"},
        {"clip": "c999", "in": 0.0, "out": 5.0, "role": "b-roll", "notes": "nie istnieje"},
        {"clip": "c001", "in": 30.0, "out": 30.0, "role": "b-roll", "notes": "pusty"},
    ],
    "captions": [
        {"at": 0.5, "end": 3.0, "text": "LIZBONA, PORTUGALIA", "style": "location"}
    ],
    "music_cues": [
        {"id": "m001", "style": "warm", "mood": "arrival", "section": "intro",
         "at": 0.0, "end": 40.0, "gain_db": -18, "duck_amount_db": -12}
    ],
    "mute_ranges": [],
    "markers": [],
    "chapters": [{"at": 0, "title": "Przyjazd do Lizbony"}],
    "narration_requests": [
        {"id": "n001", "purpose": "intro", "place_after_segment": "s001",
         "target_seconds": 4.0, "script": "Wsiadłem do najbardziej zatłoczonego tramwaju.",
         "why": "brak zdania otwierającego", "tone": "szybko"}
    ],
    "risks": ["brak beatu na 3:00"],
    "title_candidates": ["Lizbona w jeden dzień za 30 euro"],
    "thumbnail_concepts": [
        {"concept": "twarz w tramwaju", "frame_clip": "c001", "frame_t": 12.0,
         "text": "3 EURO ZA DZIEŃ", "colors": ["żółty", "granatowy"]}
    ],
    "cta": {"at_s": 12.0, "script": "Subskrybuj, jeśli lubisz takie wyjazdy."},
}


class FakeOpenRouter:
    """Stand-in for :class:`ytedit.ai.openrouter.OpenRouter` that never spends."""

    answer: Any = LLM_PLAN
    calls: list[dict[str, Any]] = []

    def __init__(self, api_key: str, cost_callback: Any = None, **_: Any) -> None:
        self.api_key = api_key
        self.cost_callback = cost_callback

    def ask_json(self, model: str, system_prompt: str, user: str, schema_hint: str, **kw: Any):
        FakeOpenRouter.calls.append(
            {"model": model, "system": system_prompt, "user": user, **kw}
        )
        if self.cost_callback:
            self.cost_callback(
                service="openrouter", op="chat", model=model, units="100+200 tok", usd=0.01
            )
        return FakeOpenRouter.answer

    def close(self) -> None:
        return None


@pytest.fixture()
def planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    """A project with a clip registry and a footage log, ready for ``plan()``."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    project = Project.create("t-proj", language="pl", root=tmp_path / "projects")
    with project.edit_state() as state:
        state["clips"] = json.loads(json.dumps(CLIPS))
        state["budget_usd"] = 20.0
    (project.analysis_dir / "footage_log.json").write_text(
        json.dumps(FOOTAGE_LOG, ensure_ascii=False), encoding="utf-8"
    )
    FakeOpenRouter.calls = []
    FakeOpenRouter.answer = LLM_PLAN
    monkeypatch.setattr("ytedit.ai.plan.OpenRouter", FakeOpenRouter)
    return project


def sine(path: Path, seconds: float = 20.0) -> Path:
    """Write a mono 48 kHz sine WAV standing in for a clip's work audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def test_subtract_ranges_splits_and_trims() -> None:
    assert subtract_ranges((0.0, 10.0), [(2.0, 4.0)]) == [(0.0, 2.0), (4.0, 10.0)]
    assert subtract_ranges((0.0, 10.0), [(0.0, 4.0)]) == [(4.0, 10.0)]
    assert subtract_ranges((0.0, 10.0), [(0.0, 20.0)]) == []


def test_missing_footage_log_is_a_clear_error(tmp_path: Path) -> None:
    project = Project.create("empty", language="pl", root=tmp_path / "projects")
    with pytest.raises(PlanError, match="footage_log.json"):
        load_footage_log(project)


# ----------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------
def test_instruction_range_is_excised(planned: Project) -> None:
    timeline, stats = build_timeline(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    c001 = [s for s in timeline.tracks.video if s.clip == "c001"]
    assert c001, "the c001 segment should survive, trimmed"
    # The spoken instruction lives at 0-4 s and must not be inside any segment.
    for segment in c001:
        assert not (segment.in_ < 4.0 and segment.out > 0.0 and segment.in_ < 4.0 <= segment.out) \
            or segment.in_ >= 4.0
        assert segment.in_ >= 4.0 - 1e-6, f"{segment.id} still contains the instruction"
    assert stats["instruction_cuts"] == 1


def test_rejected_take_is_removed(planned: Project) -> None:
    timeline, stats = build_timeline(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    c002 = [s for s in timeline.tracks.video if s.clip == "c002"]
    assert c002
    # keep=1 -> attempt 0 (5-12 s) is rejected and must not appear.
    for segment in c002:
        assert segment.out <= 5.0 + 1e-6 or segment.in_ >= 12.0 - 1e-6, (
            f"{segment.id} {segment.in_}-{segment.out} overlaps the rejected take"
        )
    assert stats["take_cuts"] == 1
    assert stats["split_segments"] >= 1


def test_rejected_take_survives_as_silent_broll(planned: Project) -> None:
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c002", "in": 5.0, "out": 12.0, "role": "b-roll", "mute_source": True,
         "notes": "pierwsze podejście jako obraz"}
    ]
    timeline, _ = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [(s.clip, s.in_, s.out, s.mute_source) for s in timeline.tracks.video] == [
        ("c002", 5.0, 12.0, True)
    ]


def test_vertical_clip_gets_blur_fill(planned: Project) -> None:
    timeline, stats = build_timeline(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    vertical = [s for s in timeline.tracks.video if s.clip == "c003"]
    assert vertical and all(s.transform.fit == "blur-fill" for s in vertical)
    horizontal = [s for s in timeline.tracks.video if s.clip == "c001"]
    assert all(s.transform.fit == "cover" for s in horizontal)
    assert "c003" in stats["vertical_fixed"]


def test_explicit_fit_is_respected(planned: Project) -> None:
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c003", "in": 2.0, "out": 8.0, "role": "b-roll",
         "transform": {"fit": "crop-pan"}}
    ]
    timeline, _ = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert timeline.tracks.video[0].transform.fit == "crop-pan"


def test_unknown_clips_and_empty_ranges_are_dropped(planned: Project) -> None:
    _, stats = build_timeline(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    assert stats["dropped_unknown_clip"] == ["c999"]
    assert stats["dropped_empty"] == ["c001 30.00-30.00"]


def test_in_out_clamped_to_clip_duration(planned: Project) -> None:
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [{"clip": "c003", "in": 20.0, "out": 999.0, "role": "b-roll"}]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert timeline.tracks.video[0].out == 30.0
    assert stats["clamped"]


def test_background_music_becomes_a_mute_range(planned: Project) -> None:
    timeline, _ = build_timeline(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    ranges = [(m.clip, m.s, m.e, m.gain_db) for m in timeline.mute_ranges]
    assert ("c002", 25.0, 35.0, -60.0) in ranges


def test_markers_come_from_settings(planned: Project) -> None:
    timeline, _ = build_timeline(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    labels = {m.label for m in timeline.markers}
    assert "hook" in labels and "promise" in labels
    assert all(m.at <= timeline.duration() + 1e-6 for m in timeline.markers)
    assert "subscribe-cta" in labels


# ----------------------------------------------------------------------
# voice-over segments (post-trip narration clips, playbook §3)
# ----------------------------------------------------------------------
def _add_narration_clip(
    planned: Project, instructions: list[dict[str, Any]] | None = None, seconds: float = 20.0
) -> dict[str, Any]:
    """Register a narration clip (``c004``) with real work audio on disk."""
    sine(planned.audio_path("c004"), seconds=seconds)
    with planned.edit_state() as state:
        state["clips"]["c004"] = {
            "id": "c004", "order": 4, "duration": seconds, "width": 1920, "height": 1080,
            "orientation": "horizontal",
        }
    log = json.loads(json.dumps(FOOTAGE_LOG))
    log["clips"].append(
        {
            "clip": "c004",
            "summary": "Nagranie z domu po podróży.",
            "kind": "a-roll",
            "instructions": instructions or [],
            "takes": [],
            "background_music": [],
        }
    )
    return log


def test_voice_over_extracts_audio_and_places_a_voice_item(planned: Project) -> None:
    footage_log = _add_narration_clip(
        planned, instructions=[{"s": 0.0, "e": 2.0, "text": "to o plaży", "action": "drop"}]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c004", "in": 2.2, "out": 8.0, "role": "a-roll",
            "voice_over": {
                "picture": [
                    {"clip": "c001", "in": 5.0, "out": 8.0},
                    {"clip": "c002", "in": 2.0, "out": 4.0},
                ]
            },
            "notes": "narracja z domu",
        }
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, footage_log)

    # No normal (heard *and* seen) segment survives for the narration clip.
    assert not any(s.clip == "c004" and not s.mute_source for s in timeline.tracks.video)

    assert stats["voice_over_segments"] == 1
    assert len(timeline.tracks.voice) == 1
    voice = timeline.tracks.voice[0]

    # The padded range (2.2-0.30, 8.0+0.45) = (1.9, 8.45) would reach into the
    # excised instruction (0.0-2.0); it must clamp to the instruction's end.
    assert voice.file == "voice/vo_c004_002.00_008.45.wav"
    assert (planned.path / voice.file).exists()
    assert voice.at == pytest.approx(0.0)
    assert voice.end == pytest.approx(6.45, abs=1e-2)

    # Picture cuts stand in for the narration clip: muted B-roll, in screen order.
    picture = [s for s in timeline.tracks.video if s.notes == "VO picture for c004"]
    assert all(p.mute_source and p.role == "b-roll" for p in picture)
    assert (picture[0].in_, picture[0].out) == (5.0, 8.0)

    # Picture cuts (3s + 2s) fall 1.45s short of the 6.45s narration. That's a
    # sub-second-shot-sized shortfall — it is absorbed by extending the LAST
    # real picture cut (c002 has 36s of headroom on its own clip), never by
    # flashing back to the narrator's own face for a fraction of a second.
    assert [p.clip for p in picture] == ["c001", "c002"]
    assert picture[1].in_ == pytest.approx(2.0)
    assert picture[1].out == pytest.approx(5.45, abs=1e-2)
    total_picture = sum(p.duration for p in picture)
    assert total_picture == pytest.approx(voice.end - voice.at, abs=1e-2)


def test_voice_over_picture_cuts_alone_are_trimmed_when_they_overshoot(planned: Project) -> None:
    footage_log = _add_narration_clip(planned)
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c004", "in": 5.0, "out": 8.0, "role": "a-roll",
            "voice_over": {
                "picture": [
                    {"clip": "c002", "in": 0.0, "out": 3.0},
                    {"clip": "c003", "in": 0.0, "out": 3.0},
                ]
            },
        }
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, footage_log)
    assert stats["voice_over_segments"] == 1

    voice = timeline.tracks.voice[0]
    assert voice.end - voice.at == pytest.approx(3.75, abs=1e-2)  # (5-0.3) to (8+0.45)

    picture = [s for s in timeline.tracks.video if s.mute_source]
    # No fallback picture from c004: the two picture cuts alone (6s) already
    # cover the 3.75s narration, so the second one is trimmed, not replaced.
    assert [p.clip for p in picture] == ["c002", "c003"]
    assert not any(p.clip == "c004" for p in picture)
    assert picture[0].duration == pytest.approx(3.0)
    assert picture[1].in_ == pytest.approx(0.0)
    assert picture[1].duration == pytest.approx(0.75, abs=1e-2)
    # c003 is vertical: the picture cut gets the vertical-clip fit rule too.
    assert picture[1].transform.fit == "blur-fill"
    total = sum(p.duration for p in picture)
    assert total == pytest.approx(voice.end - voice.at, abs=1e-2)


def test_voice_over_small_shortfall_extends_last_cut_with_no_fallback(planned: Project) -> None:
    """A ~0.75s shortfall (the speech-padding amount) must never flash the
    narrator's own face — it belongs on the real picture cut that has room."""
    footage_log = _add_narration_clip(planned)
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c004", "in": 5.0, "out": 8.0, "role": "a-roll",
            "voice_over": {"picture": [{"clip": "c002", "in": 0.0, "out": 3.0}]},
        }
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, footage_log)
    assert stats["voice_over_segments"] == 1

    voice = timeline.tracks.voice[0]
    assert voice.end - voice.at == pytest.approx(3.75, abs=1e-2)  # (5-0.3) to (8+0.45)

    picture = [s for s in timeline.tracks.video if s.mute_source]
    # The single picture cut (3s) is 0.75s short of the 3.75s narration; c002
    # has 37s of headroom on its own clip, so the cut is simply extended.
    assert [p.clip for p in picture] == ["c002"]
    assert picture[0].in_ == pytest.approx(0.0)
    assert picture[0].out == pytest.approx(3.75, abs=1e-2)
    assert not any(p.clip == "c004" for p in picture)
    total = sum(p.duration for p in picture)
    assert total == pytest.approx(voice.end - voice.at, abs=1e-2)


def test_voice_over_large_shortfall_with_no_room_falls_back_to_own_picture(
    planned: Project,
) -> None:
    """A shortfall the real cuts cannot absorb, and that is a full shot on its
    own (>= ``pacing.min_shot_seconds``), still falls back to the narration
    clip's own picture — never left uncovered."""
    footage_log = _add_narration_clip(planned)
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c004", "in": 5.0, "out": 10.0, "role": "a-roll",
            # c002's cut already runs to its clip's own duration (40s): no
            # room at all to extend it.
            "voice_over": {"picture": [{"clip": "c002", "in": 38.0, "out": 40.0}]},
        }
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, footage_log)
    assert stats["voice_over_segments"] == 1

    voice = timeline.tracks.voice[0]
    assert voice.end - voice.at == pytest.approx(5.75, abs=1e-2)  # (5-0.3) to (10+0.45)

    picture = [s for s in timeline.tracks.video if s.mute_source]
    assert [p.clip for p in picture] == ["c002", "c004"]
    assert picture[0].out == pytest.approx(40.0)  # unextended: no room left
    assert picture[1].in_ == pytest.approx(5.0)
    total = sum(p.duration for p in picture)
    assert total == pytest.approx(voice.end - voice.at, abs=1e-2)


def test_voice_over_tiny_shortfall_with_no_room_is_absorbed_by_first_cut(
    planned: Project,
) -> None:
    """A shortfall too small to justify its own shot, with no room to extend
    the real cut, is absorbed as a last resort by pulling the first cut's
    in-point back rather than appending a fallback shot."""
    footage_log = _add_narration_clip(planned)
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c004", "in": 10.0, "out": 10.25, "role": "a-roll",
            "voice_over": {"picture": [{"clip": "c002", "in": 39.5, "out": 40.0}]},
        }
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, footage_log)
    assert stats["voice_over_segments"] == 1

    voice = timeline.tracks.voice[0]
    assert voice.end - voice.at == pytest.approx(1.0, abs=1e-2)  # (10-0.3) to (10.25+0.45)

    picture = [s for s in timeline.tracks.video if s.mute_source]
    assert [p.clip for p in picture] == ["c002"]
    assert picture[0].in_ == pytest.approx(39.0)
    assert picture[0].out == pytest.approx(40.0)
    assert not any(p.clip == "c004" for p in picture)


def test_consecutive_voice_over_segments_from_same_clip_are_merged(planned: Project) -> None:
    """Three planner proposals splitting one narration take must become one
    audio extraction, not three overlapping ones (each padded independently)."""
    footage_log = _add_narration_clip(planned)
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c004", "in": 1.0, "out": 4.0, "role": "a-roll",
            "voice_over": {"picture": [{"clip": "c001", "in": 0.0, "out": 3.0}]},
            "notes": "część 1",
        },
        {
            "clip": "c004", "in": 4.2, "out": 7.0, "role": "a-roll",
            "voice_over": {"picture": [{"clip": "c002", "in": 0.0, "out": 2.5}]},
            "notes": "część 2",
        },
        {
            "clip": "c004", "in": 7.1, "out": 9.0, "role": "a-roll",
            "voice_over": {"picture": [{"clip": "c003", "in": 0.0, "out": 1.5}]},
            "notes": "część 3",
        },
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, footage_log)

    # A chain of 3 merges away 2 segments into the first.
    assert stats["voice_over_merged"] == 2
    assert stats["voice_over_segments"] == 1
    assert len(timeline.tracks.voice) == 1

    voice = timeline.tracks.voice[0]
    # merged in/out 1.0-9.0, padded by 0.3/0.45 -> 0.7-9.45 (8.75s) — one WAV,
    # not three overlapping ones.
    assert voice.end - voice.at == pytest.approx(8.75, abs=1e-2)

    picture = [s for s in timeline.tracks.video if s.mute_source]
    assert [p.clip for p in picture] == ["c001", "c002", "c003"]
    assert not any(p.clip == "c004" for p in picture)
    total = sum(p.duration for p in picture)
    assert total == pytest.approx(voice.end - voice.at, abs=1e-2)


# ----------------------------------------------------------------------
# script-first planning: sentence-id segments (ytedit/ai/sentences.py)
# ----------------------------------------------------------------------
def _write_sentence_catalogue(project: Project, sentences: list[dict[str, Any]]) -> None:
    """Write a minimal ``analysis/sentences.json`` fixture, grouped by clip."""
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for sent in sentences:
        by_clip.setdefault(sent["clip"], []).append(sent)
    doc = {"clips": [{"id": clip, "sentences": items} for clip, items in by_clip.items()]}
    (project.analysis_dir / "sentences.json").write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8"
    )


def _sentence(sid: str, clip: str, s: float, e: float, text: str, **flags: Any) -> dict[str, Any]:
    base = {
        "id": sid, "clip": clip, "s": s, "e": e, "text": text,
        "instruction": False, "retake_of": None, "duplicate_of": None,
    }
    base.update(flags)
    return base


def test_sentence_segment_derives_in_out_with_pads(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Cześć wszystkim."),
            _sentence("c001#2", "c001", 12.5, 15.0, "Jedziemy dalej."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1", "c001#2"], "role": "a-roll", "notes": "intro"}
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)

    assert len(timeline.tracks.video) == 1
    segment = timeline.tracks.video[0]
    assert segment.in_ == pytest.approx(10.0 - 0.30)
    assert segment.out == pytest.approx(15.0 + 0.45)
    assert segment.sentence_ids == ["c001#1", "c001#2"]
    assert stats["sentence_segments"] == 1
    assert not stats["unknown_sentence_refs"]
    assert not stats["excluded_sentence_refs"]


def test_unknown_sentence_ref_is_dropped_with_a_stat(planned: Project) -> None:
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 12.0, "Cześć wszystkim.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1", "c001#99"], "role": "a-roll"}
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert stats["unknown_sentence_refs"] == ["c001#99"]
    assert len(timeline.tracks.video) == 1
    assert timeline.tracks.video[0].sentence_ids == ["c001#1"]


def test_instruction_and_retake_sentence_refs_are_excluded(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 0.0, 2.0, "to na koniec", instruction=True),
            _sentence("c001#2", "c001", 10.0, 12.0, "Cześć wszystkim.", retake_of="c001#3"),
            _sentence("c001#3", "c001", 20.0, 22.0, "Cześć wszystkim, ostatnie podejście."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1"], "role": "a-roll", "notes": "instr"},
        {"clip": "c001", "sentences": ["c001#2"], "role": "a-roll", "notes": "retake"},
        {"clip": "c001", "sentences": ["c001#3"], "role": "a-roll", "notes": "kept"},
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    kept_clips = [s.sentence_ids for s in timeline.tracks.video]
    assert kept_clips == [["c001#3"]]
    assert any("instruction" in r for r in stats["excluded_sentence_refs"])
    assert any("retake_of" in r for r in stats["excluded_sentence_refs"])


def test_a_sentence_id_referenced_twice_is_only_kept_the_first_time(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Cześć wszystkim."),
            _sentence("c001#2", "c001", 12.5, 15.0, "Jedziemy dalej."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1"], "role": "a-roll", "notes": "first use"},
        {"clip": "c001", "sentences": ["c001#1", "c001#2"], "role": "a-roll", "notes": "reuse"},
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert stats["duplicate_sentence_refs"] == ["c001#1"]
    ids = [s.sentence_ids for s in timeline.tracks.video]
    assert ids == [["c001#1"], ["c001#2"]]


def test_non_contiguous_sentence_ids_split_into_separate_segments(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Jeden."),
            _sentence("c001#2", "c001", 12.5, 15.0, "Dwa.", instruction=True),
            _sentence("c001#3", "c001", 15.5, 18.0, "Trzy."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    # c001#2 is an instruction and gets dropped, leaving a gap between #1 and #3.
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1", "c001#2", "c001#3"], "role": "a-roll"}
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    ids = [s.sentence_ids for s in timeline.tracks.video]
    assert ids == [["c001#1"], ["c001#3"]]
    assert stats["non_contiguous_splits"] == 1


def test_cutaway_after_sentence_borrows_audio_and_resumes_the_a_roll(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Jedziemy tramwajem."),
            # Long enough that the resumed piece clears pacing.min_shot_seconds
            # (0.8 s) once it starts at the cutaway's audio_from.out (15.0).
            _sentence("c001#2", "c001", 12.5, 17.0, "Bardzo zatłoczonym, prawie nie weszliśmy."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c001",
            "sentences": ["c001#1", "c001#2"],
            "role": "a-roll",
            "cutaways": [{"clip": "c003", "in": 8.0, "out": 11.0, "after_sentence": "c001#1"}],
        }
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    video = timeline.tracks.video
    assert [s.clip for s in video] == ["c001", "c003", "c001"]

    piece1, cutaway, piece2 = video
    assert piece1.in_ == pytest.approx(10.0 - 0.30)
    assert piece1.out == pytest.approx(12.0)  # no trailing pad: audio continues
    assert piece1.sentence_ids == ["c001#1"]

    assert cutaway.role == "cutaway"
    assert cutaway.in_ == pytest.approx(8.0) and cutaway.out == pytest.approx(11.0)
    assert cutaway.audio_from is not None
    assert cutaway.audio_from.clip == "c001"
    assert cutaway.audio_from.in_ == pytest.approx(12.0)
    assert cutaway.audio_from.out == pytest.approx(15.0)  # 3 s cutaway duration

    assert piece2.in_ == pytest.approx(15.0)  # resumes where the borrowed audio ends
    assert piece2.out == pytest.approx(17.0 + 0.45)
    assert piece2.sentence_ids == ["c001#2"]
    assert not stats["dropped_cutaways"]


def test_a_cutaway_with_an_unmatched_after_sentence_is_dropped(planned: Project) -> None:
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 12.0, "Jedziemy tramwajem.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c001",
            "sentences": ["c001#1"],
            "role": "a-roll",
            "cutaways": [{"clip": "c003", "in": 8.0, "out": 11.0, "after_sentence": "c001#99"}],
        }
    ]
    timeline, stats = build_timeline(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [s.clip for s in timeline.tracks.video] == ["c001"]
    assert stats["dropped_cutaways"]


def test_legacy_raw_in_out_segments_still_work_without_a_sentence_catalogue(
    planned: Project,
) -> None:
    """No ``analysis/sentences.json`` at all — ``plan --from-response`` on an
    old planner answer must behave exactly as it did before this feature."""
    assert not (planned.analysis_dir / "sentences.json").exists()
    timeline, stats = build_timeline(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    assert timeline.tracks.video
    assert stats["sentence_segments"] == 0


# ----------------------------------------------------------------------
# stage behaviour
# ----------------------------------------------------------------------
def test_plan_writes_all_artifacts(planned: Project) -> None:
    result = plan(planned, notes="skróć logistykę")
    assert Path(result["timeline"]).name == "timeline.json"
    for key in ("edit_plan", "markdown", "narration"):
        assert Path(result[key]).exists(), key
    assert result["draft"] is False
    assert planned.stage_status("plan") == "done"

    md = (planned.plan_dir / "edit_plan.md").read_text(encoding="utf-8")
    assert "## Story outline" in md and "## Narration requests" in md
    assert "Lizbona w jeden dzień" in md
    narration = (planned.plan_dir / "narration_requests.md").read_text(encoding="utf-8")
    assert "Wsiadłem do najbardziej zatłoczonego tramwaju." in narration

    # The editor notes reach the planner prompt.
    assert "Editor notes: skróć logistykę" in FakeOpenRouter.calls[-1]["user"]
    assert FakeOpenRouter.calls[-1]["max_tokens"] == 32000

    # And the call was charged to the project ledger.
    costs = planned.load_state()["costs"]
    assert costs and costs[-1]["service"] == "openrouter"


def test_human_edited_timeline_is_never_overwritten(planned: Project) -> None:
    plan(planned)
    timeline = Timeline.load(planned.timeline_file)
    timeline.meta.edited_by_human = True
    timeline.meta.notes = "hand-edited cut"
    timeline.save(planned.timeline_file)

    result = plan(planned)
    assert result["draft"] is True
    assert Path(result["timeline"]).name == "timeline.draft.json"
    assert Timeline.load(planned.timeline_file).meta.notes == "hand-edited cut"
    assert (planned.plan_dir / "timeline.draft.json").exists()

    forced = plan(planned, force=True)
    assert Path(forced["timeline"]).name == "timeline.json"
    assert Timeline.load(planned.timeline_file).meta.edited_by_human is False


def test_plan_from_response_rebuilds_without_calling_the_llm(
    planned: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = plan(planned)
    raw_path = planned.plan_dir / "planner_response.json"
    assert raw_path.exists()
    assert FakeOpenRouter.calls  # the first run really did call the LLM
    costs_before = len(planned.load_state().get("costs", []))

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--from-response must never construct the LLM client")

    monkeypatch.setattr("ytedit.ai.plan.OpenRouter", boom)

    result = plan(planned, from_response=True)
    assert result["cost_usd"] == 0.0
    assert result["segments"] == first["segments"]
    timeline = Timeline.load(Path(result["timeline"]))
    assert len(timeline.tracks.video) == len(Timeline.load(Path(first["timeline"])).tracks.video)
    # No new charge was recorded against the project's cost ledger.
    assert len(planned.load_state().get("costs", [])) == costs_before


def test_plan_from_response_honours_the_human_edited_guard(planned: Project) -> None:
    plan(planned)
    timeline = Timeline.load(planned.timeline_file)
    timeline.meta.edited_by_human = True
    timeline.meta.notes = "hand-edited cut"
    timeline.save(planned.timeline_file)

    result = plan(planned, from_response=True)
    assert result["draft"] is True
    assert Path(result["timeline"]).name == "timeline.draft.json"
    assert Timeline.load(planned.timeline_file).meta.notes == "hand-edited cut"


def test_plan_from_response_without_a_prior_run_fails_clearly(planned: Project) -> None:
    with pytest.raises(PlanError, match="planner_response.json"):
        plan(planned, from_response=True)


def test_plan_records_an_error_status(planned: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOpenRouter.answer = {"segments": [{"clip": "c999", "in": 0.0, "out": 5.0}]}
    with pytest.raises(PlanError, match="no usable video segments"):
        plan(planned)
    assert planned.stage_status("plan") == "error"


def test_prompts_shape_from_promptsmd_is_accepted() -> None:
    """The planner may answer in the ``prompts.md`` timeline + edit_plan shape."""
    raw = {
        "version": 1,
        "tracks": {
            "video": [
                {"id": "s001", "clip": "c001", "in": 5.0, "out": 9.0, "role": "a-roll",
                 "transform": {"fit": "cover"}, "transition_in": {"type": "cut", "duration": 0.0}}
            ],
            "captions": [{"id": "t001", "at": 0.0, "end": 2.0, "text": "LIZBONA",
                          "style": "location"}],
            "music": [{"id": "m001", "at": 0.0, "end": 30.0, "gain_db": -18}],
        },
        "chapters": [{"at": 0, "title": "Start"}],
        "meta": {"title_candidates": ["A title"]},
        "edit_plan": {
            "story_outline": [{"beat": "hook", "target_time": 0.0, "clips": ["c001"],
                               "notes": "otwarcie"}],
            "cold_open_picks": [{"clip": "c001", "s": 5.0, "e": 9.0, "why": "best"}],
            "narration_requests": [{"where": "cold open", "why": "brak intro",
                                    "target_seconds": 4.0, "script": "Cześć", "tone": "szybko"}],
            "title_candidates": [{"text": "Inny tytuł", "rationale": "ciekawość"}],
            "thumbnail_concepts": [{"description": "twarz", "on_image_text": "3 EURO",
                                    "colors": ["żółty"], "source_frame_hint": "c001@12.4"}],
            "risk_flags": ["brak beatu na 6:00"],
        },
    }
    parsed = EditPlan.from_llm(raw)
    assert parsed.story.beats[0].label == "hook"
    assert parsed.segments[0].clip == "c001" and parsed.segments[0].in_ == 5.0
    assert parsed.cold_open[0].out == 9.0
    assert parsed.narration_requests[0].purpose == "intro"
    assert parsed.narration_requests[0].id == "n001"
    assert parsed.title_candidates == ["Inny tytuł"]
    assert parsed.thumbnail_concepts[0].frame_clip == "c001"
    assert parsed.thumbnail_concepts[0].frame_t == 12.4
    assert parsed.risks == ["brak beatu na 6:00"]
    assert parsed.chapters[0].title == "Start"


# ----------------------------------------------------------------------
# pacing report
# ----------------------------------------------------------------------
def _timeline_with_shots(lengths: list[float], role: str = "b-roll") -> Timeline:
    timeline = new_timeline()
    cursor = 0.0
    for i, length in enumerate(lengths):
        timeline.tracks.video.append(
            VideoSegment(
                id=f"s{i + 1:03d}",
                clip="c001",
                **{"in": cursor},
                out=cursor + length,
                role=role,
            )
        )
        cursor += length
    return timeline


def test_pacing_report_catches_a_long_shot_before_six_minutes() -> None:
    settings = Project.create  # placeholder to keep the import obvious
    from ytedit.config import load_settings

    timeline = _timeline_with_shots([3.0, 9.0, 3.0, 3.0])
    warnings = pacing_report(timeline, load_settings())
    assert any("9.0s" in w and "rule 5" in w for w in warnings), warnings
    assert settings is not None


def test_pacing_report_allows_a_seven_second_shot_after_six_minutes() -> None:
    from ytedit.config import load_settings

    # 100 x 4 s = 400 s of runway, then a 6.5 s shot at 6:40.
    timeline = _timeline_with_shots([4.0] * 100 + [6.5])
    warnings = pacing_report(timeline, load_settings())
    assert not any("6.5s" in w for w in warnings), warnings


def test_pacing_report_flags_dead_time_and_ending_phrase() -> None:
    from ytedit.config import load_settings

    timeline = _timeline_with_shots([20.0, 3.0])
    timeline.tracks.video[-1].notes = "kończy słowami: dzięki za oglądanie, do zobaczenia"
    warnings = pacing_report(timeline, load_settings())
    assert any("rule 6" in w for w in warnings), warnings
    assert any("rule 8" in w for w in warnings), warnings


def test_pacing_report_flags_long_aroll_run() -> None:
    from ytedit.config import load_settings

    timeline = _timeline_with_shots([4.0, 4.0, 4.0, 4.0], role="a-roll")
    warnings = pacing_report(timeline, load_settings())
    assert any("rule 9" in w and "A-roll run" in w for w in warnings), warnings


# ----------------------------------------------------------------------
# prompt / schema economy (the planner used to be asked for three schemas)
# ----------------------------------------------------------------------
def test_schema_hint_points_at_the_system_prompt_instead_of_repeating_it() -> None:
    """``ask_json`` must not append a second copy of the schema."""
    assert len(SCHEMA_HINT) < 200
    assert '"segments"' not in SCHEMA_HINT


def test_plan_system_prompt_carries_exactly_one_schema() -> None:
    text = load_prompt("plan.system")
    # The timeline.json document (built deterministically by build_timeline) is
    # never asked of the model any more.
    assert '"tracks"' not in text and '"voice"' not in text
    assert text.count('"segments"') == 1
    assert text.count('"narration_requests"') == 1
    assert "{project_language}" in text


def test_plan_user_prompt_keeps_its_placeholders() -> None:
    text = load_prompt("plan.user")
    for key in (
        "{project_slug}",
        "{project_language}",
        "{clip_count}",
        "{total_footage_minutes}",
        "{footage_log_json}",
        "{existing_timeline_json_or_null}",
    ):
        assert key in text, key


# ----------------------------------------------------------------------
# truncation handling
# ----------------------------------------------------------------------
class TruncatingOpenRouter:
    """Fake client whose first (smaller) budget stops at ``finish_reason=length``."""

    #: Budgets at or below this truncate; above it the model finishes.
    truncate_at_or_below: int = DEFAULT_PLAN_MAX_TOKENS
    calls: list[int] = []

    def __init__(self, api_key: str, cost_callback: Any = None, **_: Any) -> None:
        self.cost_callback = cost_callback

    def chat(self, model: str, messages: list[Any], **kw: Any) -> ChatResult:
        max_tokens = int(kw.get("max_tokens", 0))
        truncated = max_tokens <= type(self).truncate_at_or_below
        if self.cost_callback:
            self.cost_callback(
                service="openrouter", op="chat", model=model,
                units=f"100+{max_tokens} tok", usd=0.01,
            )
        return ChatResult(
            text='{"segments": [{"clip": "c001", "in": 0.0, "out"' if truncated
            else json.dumps(LLM_PLAN, ensure_ascii=False),
            json=None if truncated else json.loads(json.dumps(LLM_PLAN)),
            usage={"prompt_tokens": 100, "completion_tokens": max_tokens},
            cost_usd=0.01,
            model=model,
            raw={"choices": [{"finish_reason": "length" if truncated else "stop"}]},
        )

    def ask_json(self, model: str, system_prompt: str, user: str, schema_hint: str,
                 **kw: Any) -> Any:
        type(self).calls.append(int(kw.get("max_tokens", 0)))
        result = self.chat(model, [], max_tokens=kw.get("max_tokens", 0))
        if result.json is None:
            raise OpenRouterError("model did not return parseable JSON")
        return result.json

    def close(self) -> None:
        return None


def test_a_truncated_planner_answer_is_retried_once_with_a_bigger_budget(
    planned: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    TruncatingOpenRouter.calls = []
    TruncatingOpenRouter.truncate_at_or_below = DEFAULT_PLAN_MAX_TOKENS
    monkeypatch.setattr("ytedit.ai.plan.OpenRouter", TruncatingOpenRouter)

    result = plan(planned)
    assert TruncatingOpenRouter.calls == [DEFAULT_PLAN_MAX_TOKENS, DEFAULT_PLAN_MAX_TOKENS * 2]
    assert result["finish_reason"] == "stop"
    # The partial answer survives on disk for debugging.
    assert (planned.plan_dir / "planner_response.truncated.1.txt").exists()


def test_a_planner_that_truncates_twice_fails_loudly(
    planned: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    TruncatingOpenRouter.calls = []
    TruncatingOpenRouter.truncate_at_or_below = MAX_PLAN_MAX_TOKENS
    monkeypatch.setattr("ytedit.ai.plan.OpenRouter", TruncatingOpenRouter)

    with pytest.raises(PlanError, match="finish_reason=length"):
        plan(planned)
    assert TruncatingOpenRouter.calls == [DEFAULT_PLAN_MAX_TOKENS, MAX_PLAN_MAX_TOKENS]
    assert planned.stage_status("plan") == "error"
