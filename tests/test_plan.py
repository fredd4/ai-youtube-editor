"""Tests for ``ytedit.ai.plan`` — post-processing, pacing rules, draft safety.

Every test here is offline: the planner model is replaced with a fake
``OpenRouter`` that returns a canned answer, so nothing costs money.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ytedit.ai.plan import (
    DEFAULT_PLAN_MAX_TOKENS,
    MAX_PLAN_MAX_TOKENS,
    SCHEMA_HINT,
    EditPlan,
    PlanError,
    SpeechSecondsError,
    build_cut,
    load_footage_log,
    pacing_report,
    plan,
    subtract_ranges,
)
from ytedit.ai.openrouter import ChatResult, OpenRouterError
from ytedit.ai.prompts import load as load_prompt
from ytedit.cut import cut_path, load_cut, resolve, save_cut
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
# planner segments -> beats
# ----------------------------------------------------------------------
def _broll(cut, clip: str) -> list:
    """Every ``broll`` beat of one clip, in cut order."""
    return [b for b in cut.beats if b.kind == "broll" and b.clip == clip]


def test_instruction_range_is_excised(planned: Project) -> None:
    cut, stats = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    beats = _broll(cut, "c001")
    assert beats, "the c001 beat should survive, trimmed"
    # The spoken instruction lives at 0-4 s and must not be inside any beat.
    for beat in beats:
        assert beat.in_ >= 4.0 - 1e-6, f"{beat.id} still contains the instruction"
    assert stats["instruction_cuts"] == 1


def test_rejected_take_is_removed(planned: Project) -> None:
    cut, stats = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    beats = _broll(cut, "c002")
    assert beats
    # keep=1 -> attempt 0 (5-12 s) is rejected and must not appear.
    for beat in beats:
        assert beat.out <= 5.0 + 1e-6 or beat.in_ >= 12.0 - 1e-6, (
            f"{beat.id} {beat.in_}-{beat.out} overlaps the rejected take"
        )
    assert stats["take_cuts"] == 1
    assert stats["split_segments"] >= 1


def test_rejected_take_survives_as_silent_broll(planned: Project) -> None:
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c002", "in": 5.0, "out": 12.0, "role": "b-roll", "mute_source": True,
         "notes": "pierwsze podejście jako obraz"}
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [(b.clip, b.in_, b.out, b.audio) for b in cut.beats] == [
        ("c002", 5.0, 12.0, "mute")
    ]


def test_an_ambient_broll_beat_says_so_explicitly(planned: Project) -> None:
    """``audio`` is only ever a deliberate choice, so it is always written."""
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [{"clip": "c003", "in": 2.0, "out": 8.0, "role": "b-roll"}]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert cut.beats[0].audio == "ambient"
    assert cut.beats[0].keeps_source_audio is True


def test_vertical_clip_gets_blur_fill(planned: Project) -> None:
    cut, stats = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    vertical = _broll(cut, "c003")
    assert vertical and all(b.transform.fit == "blur-fill" for b in vertical)
    assert all(b.transform.fit == "cover" for b in _broll(cut, "c001"))
    assert "c003" in stats["vertical_fixed"]


def test_explicit_fit_is_respected(planned: Project) -> None:
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c003", "in": 2.0, "out": 8.0, "role": "b-roll",
         "transform": {"fit": "crop-pan"}}
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert cut.beats[0].transform.fit == "crop-pan"


def test_unknown_clips_and_empty_ranges_are_dropped(planned: Project) -> None:
    _, stats = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    assert stats["dropped_unknown_clip"] == ["c999"]
    assert stats["dropped_empty"] == ["c001 30.00-30.00"]


def test_in_out_clamped_to_clip_duration(planned: Project) -> None:
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [{"clip": "c003", "in": 20.0, "out": 999.0, "role": "b-roll"}]
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert cut.beats[0].out == 30.0
    assert stats["clamped"]


def test_background_music_becomes_a_mute_range(planned: Project) -> None:
    cut, _ = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    ranges = [(m.clip, m.s, m.e, m.gain_db) for m in cut.mute_ranges]
    assert ("c002", 25.0, 35.0, -60.0) in ranges


def test_the_first_beat_never_fades_in(planned: Project) -> None:
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c003", "in": 2.0, "out": 8.0, "role": "cold-open",
         "transition_in": {"type": "fade", "duration": 1.0}},
        {"clip": "c003", "in": 10.0, "out": 16.0, "role": "b-roll",
         "transition_in": {"type": "fade", "duration": 1.0}},
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert cut.beats[0].transition_in.type == "cut"
    assert cut.beats[1].transition_in.type == "fade"


# ----------------------------------------------------------------------
# beat-positioned tracks (captions, music, chapters, markers)
# ----------------------------------------------------------------------
def test_markers_come_from_settings_and_name_beats(planned: Project) -> None:
    cut, _ = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    labels = {m.label for m in cut.markers}
    assert "hook" in labels and "promise" in labels
    assert "subscribe-cta" in labels
    uids = {b.uid for b in cut.beats}
    assert all(m.beat in uids for m in cut.markers)


def test_a_planner_caption_lands_on_the_beat_playing_under_it(planned: Project) -> None:
    """The planner answers in seconds; only the beat survives a re-cut."""
    cut, _ = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    assert cut.captions
    caption = cut.captions[0]
    assert caption.text == "LIZBONA, PORTUGALIA"
    # 0.5 s into the programme is the first beat, 0.5 s past its start.
    assert caption.beat == cut.beats[0].uid
    assert caption.offset == pytest.approx(0.5, abs=0.05)


def test_a_music_cue_becomes_an_inclusive_beat_range(planned: Project) -> None:
    cut, _ = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    assert cut.music
    uids = [b.uid for b in cut.beats]
    assert cut.music[0].from_ == uids[0]
    assert cut.music[0].to in uids


def test_a_chapter_lands_on_a_beat(planned: Project) -> None:
    cut, _ = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    assert [c.title for c in cut.chapters] == ["Przyjazd do Lizbony"]
    assert cut.chapters[0].beat == cut.beats[0].uid


# ----------------------------------------------------------------------
# voice-over segments: the narrator heard, not seen (playbook §3)
# ----------------------------------------------------------------------
def test_voice_over_becomes_an_off_camera_speech_beat(planned: Project) -> None:
    """The post-trip clip's own audio plays; its picture never does."""
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Wróciłem do domu."),
            _sentence("c001#2", "c001", 12.5, 15.0, "I wtedy zrozumiałem."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c001",
            "sentences": ["c001#1", "c001#2"],
            "role": "a-roll",
            "voice_over": {"picture": [
                {"clip": "c003", "in": 0.0, "out": 4.0},
                {"clip": "c003", "in": 10.0, "out": 16.0},
            ]},
        }
    ]
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)

    assert len(cut.beats) == 1
    beat = cut.beats[0]
    assert beat.kind == "speech" and beat.clip == "c001"
    assert beat.on_camera is False
    assert [(s.clip, s.in_, s.out) for s in beat.shots] == [
        ("c003", 0.0, 4.0), ("c003", 10.0, 16.0)
    ]
    # Picture cuts have no sentence to hang off: they chain from the beat start.
    assert all(shot.after is None for shot in beat.shots)
    assert stats["voice_over_segments"] == 1


def test_voice_over_picture_is_silent_and_covers_the_narration(planned: Project) -> None:
    """Resolved: only the narration is heard, and no frame shows the narrator."""
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 14.0, "Wróciłem do domu.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1"], "role": "a-roll",
         "voice_over": {"picture": [{"clip": "c003", "in": 0.0, "out": 20.0}]}}
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    timeline = resolve(planned, cut)

    assert [s.clip for s in timeline.tracks.video] == ["c003"]
    segment = timeline.tracks.video[0]
    assert segment.audio_from is not None and segment.audio_from.clip == "c001"
    assert segment.mute_source is False  # the borrowed narration is what is heard


def test_voice_over_cutaways_are_dropped_with_a_reason(planned: Project) -> None:
    """A voice-over segment's picture comes from ``voice_over.picture`` only."""
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 14.0, "Wróciłem do domu.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1"], "role": "a-roll",
         "voice_over": {"picture": [{"clip": "c003", "in": 0.0, "out": 20.0}]},
         "cutaways": [{"clip": "c003", "in": 2.0, "out": 4.0, "after_sentence": "c001#1"}]}
    ]
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert len(cut.beats[0].shots) == 1
    assert any("voice-over segment" in d for d in stats["dropped_cutaways"])


def test_a_voice_over_picture_cut_on_an_unknown_clip_is_dropped(planned: Project) -> None:
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 14.0, "Wróciłem do domu.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1"], "role": "a-roll",
         "voice_over": {"picture": [
             {"clip": "c999", "in": 0.0, "out": 4.0},
             {"clip": "c003", "in": 0.0, "out": 20.0},
         ]}}
    ]
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [s.clip for s in cut.beats[0].shots] == ["c003"]
    assert any("c999" in d for d in stats["dropped_cutaways"])


# ----------------------------------------------------------------------
# speech in seconds: a planner error, not a fallback
# ----------------------------------------------------------------------
def _add_transcript(project: Project, clip: str, words: list[tuple[str, float, float]]) -> None:
    project.transcript_path(clip).write_text(
        json.dumps(
            {"clip": clip, "language": "pl",
             "words": [{"t": t, "s": s, "e": e} for t, s, e in words]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_a_speech_range_given_in_seconds_is_refused(planned: Project) -> None:
    """There is no raw-seconds path for narration since cut v2."""
    _add_transcript(planned, "c001", [("Cześć", 6.0, 6.4), ("wszystkim.", 6.5, 7.0)])
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [{"clip": "c001", "in": 5.0, "out": 9.0, "role": "a-roll"}]
    with pytest.raises(SpeechSecondsError) as excinfo:
        build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert "c001 5.00-9.00" in excinfo.value.segments[0]
    assert "`sentences`" in excinfo.value.prompt_note()


def test_a_deliberately_muted_range_over_speech_is_allowed(planned: Project) -> None:
    """`mute_source` is how a take is legitimately reused as silent B-roll."""
    _add_transcript(planned, "c001", [("Cześć", 6.0, 6.4), ("wszystkim.", 6.5, 7.0)])
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "in": 5.0, "out": 9.0, "role": "b-roll", "mute_source": True}
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [(b.kind, b.audio) for b in cut.beats] == [("broll", "mute")]


def test_a_wordless_range_is_still_plain_broll(planned: Project) -> None:
    _add_transcript(planned, "c001", [("Cześć", 30.0, 30.4)])
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [{"clip": "c001", "in": 5.0, "out": 9.0, "role": "b-roll"}]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [(b.kind, b.audio) for b in cut.beats] == [("broll", "ambient")]


def test_the_planner_is_asked_once_more_for_speech_given_in_seconds(
    planned: Project,
) -> None:
    """One retry with the offending segments named, then the plan succeeds."""
    _add_transcript(planned, "c001", [("Cześć", 6.0, 6.4), ("wszystkim.", 6.5, 7.0)])
    bad = json.loads(json.dumps(LLM_PLAN))
    bad["segments"] = [{"clip": "c001", "in": 5.0, "out": 9.0, "role": "a-roll"}]
    good = json.loads(json.dumps(LLM_PLAN))
    good["segments"] = [{"clip": "c003", "in": 2.0, "out": 8.0, "role": "b-roll"}]
    answers = [bad, good]

    class Retrying(FakeOpenRouter):
        def ask_json(self, model, system_prompt, user, schema_hint, **kw):
            FakeOpenRouter.calls.append({"model": model, "user": user, **kw})
            return answers[min(len(FakeOpenRouter.calls), len(answers)) - 1]

    FakeOpenRouter.calls = []
    import ytedit.ai.plan as plan_mod

    plan_mod.OpenRouter = Retrying
    try:
        result = plan(planned)
    finally:
        plan_mod.OpenRouter = FakeOpenRouter
    assert len(FakeOpenRouter.calls) == 2
    assert "raw in/out seconds" in FakeOpenRouter.calls[1]["user"]
    assert result["beats"] == 1


def test_a_planner_that_gives_speech_in_seconds_twice_fails_loudly(
    planned: Project,
) -> None:
    _add_transcript(planned, "c001", [("Cześć", 6.0, 6.4), ("wszystkim.", 6.5, 7.0)])
    bad = json.loads(json.dumps(LLM_PLAN))
    bad["segments"] = [{"clip": "c001", "in": 5.0, "out": 9.0, "role": "a-roll"}]
    FakeOpenRouter.answer = bad
    with pytest.raises(SpeechSecondsError):
        plan(planned)
    assert planned.stage_status("plan") == "error"



# ----------------------------------------------------------------------
# script-first planning: sentence-id segments -> speech beats
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
        "id": sid, "clip": clip, "n": int(sid.rsplit("#", 1)[1]),
        "s": s, "e": e, "text": text,
        "instruction": False, "retake_of": None, "duplicate_of": None,
    }
    base.update(flags)
    return base


def _speech(cut) -> list:
    return [b for b in cut.beats if b.kind == "speech"]


def test_a_sentence_run_becomes_one_speech_beat(planned: Project) -> None:
    """The ids travel through untouched — the resolver decides the seconds."""
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
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)

    assert len(cut.beats) == 1
    beat = cut.beats[0]
    assert beat.kind == "speech" and beat.clip == "c001"
    assert beat.sentences == ["c001#1", "c001#2"]
    assert beat.on_camera is True
    assert beat.in_ is None and beat.out is None, "a speech beat never carries seconds"
    assert stats["sentence_segments"] == 1
    assert not stats["unknown_sentence_refs"]
    assert not stats["excluded_sentence_refs"]


def test_the_resolver_is_what_turns_those_ids_into_seconds(planned: Project) -> None:
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 12.0, "Cześć wszystkim.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [{"clip": "c001", "sentences": ["c001#1"], "role": "a-roll"}]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    segment = resolve(planned, cut).tracks.video[0]
    assert segment.in_ == pytest.approx(10.0 - 0.30)
    assert segment.out == pytest.approx(12.0 + 0.45)


def test_unknown_sentence_ref_is_dropped_with_a_stat(planned: Project) -> None:
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 12.0, "Cześć wszystkim.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1", "c001#99"], "role": "a-roll"}
    ]
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert stats["unknown_sentence_refs"] == ["c001#99"]
    assert [b.sentences for b in _speech(cut)] == [["c001#1"]]


def test_an_instruction_ref_is_dropped_and_a_retake_ref_is_kept(planned: Project) -> None:
    """A spoken "cut this" must never air; a retake flag is the editor's call."""
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
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [b.sentences for b in _speech(cut)] == [["c001#2"], ["c001#3"]]
    assert any("instruction" in r for r in stats["excluded_sentence_refs"])
    assert any("retake_of" in r for r in stats["flagged_sentence_refs"])


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
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert stats["duplicate_sentence_refs"] == ["c001#1 (already in segment #1)"]
    assert [b.sentences for b in _speech(cut)] == [["c001#1"], ["c001#2"]]
    # No sentence is claimed twice, so nothing can be heard twice.
    assert resolve(planned, cut)


def test_non_contiguous_sentence_ids_split_into_separate_beats(planned: Project) -> None:
    """The resolver refuses a run that skips an ``n``, so the split happens here."""
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
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [b.sentences for b in _speech(cut)] == [["c001#1"], ["c001#3"]]
    assert stats["non_contiguous_splits"] == 1


def test_a_split_run_fades_in_only_on_its_first_beat(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Jeden."),
            _sentence("c001#2", "c001", 12.5, 15.0, "Dwa.", instruction=True),
            _sentence("c001#3", "c001", 15.5, 18.0, "Trzy."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c003", "in": 2.0, "out": 8.0, "role": "cold-open"},
        {"clip": "c001", "sentences": ["c001#1", "c001#2", "c001#3"], "role": "a-roll",
         "transition_in": {"type": "fade", "duration": 0.5}},
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    speech = _speech(cut)
    assert speech[0].transition_in.type == "fade"
    assert speech[1].transition_in.type == "cut"


def test_a_cutaway_becomes_a_shot_after_its_sentence(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Jedziemy tramwajem."),
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
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    beat = cut.beats[0]
    assert [(s.clip, s.in_, s.out, s.after) for s in beat.shots] == [
        ("c003", 8.0, 11.0, "c001#1")
    ]
    assert not stats["dropped_cutaways"]

    # Resolved: the narration keeps playing under the insert's picture.
    video = resolve(planned, cut).tracks.video
    assert [s.clip for s in video] == ["c001", "c003", "c001"]
    piece1, shot, piece2 = video
    assert piece1.out == pytest.approx(12.0)  # no trailing pad: audio continues
    assert shot.audio_from is not None and shot.audio_from.clip == "c001"
    assert shot.audio_from.in_ == pytest.approx(12.0)
    assert shot.audio_from.out == pytest.approx(15.0)
    assert piece2.in_ == pytest.approx(15.0)


def test_chained_cutaways_after_one_sentence_keep_their_order(planned: Project) -> None:
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 12.0, "Jedziemy tramwajem."),
            _sentence("c001#2", "c001", 12.5, 20.0, "Bardzo zatłoczonym, prawie nie weszliśmy."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c001",
            "sentences": ["c001#1", "c001#2"],
            "role": "a-roll",
            "cutaways": [
                {"clip": "c003", "in": 8.0, "out": 10.0, "after_sentence": "c001#1"},
                {"clip": "c002", "in": 26.0, "out": 28.0, "after_sentence": "c001#1"},
            ],
        }
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert [s.clip for s in cut.beats[0].shots] == ["c003", "c002"]


def test_a_cutaway_lands_on_the_beat_that_owns_its_sentence(planned: Project) -> None:
    """When a segment splits, each shot follows the run holding its ``after``."""
    _write_sentence_catalogue(
        planned,
        [
            _sentence("c001#1", "c001", 10.0, 14.0, "Jeden."),
            _sentence("c001#2", "c001", 14.5, 16.0, "Dwa.", instruction=True),
            _sentence("c001#3", "c001", 16.5, 22.0, "Trzy."),
        ],
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {
            "clip": "c001",
            "sentences": ["c001#1", "c001#2", "c001#3"],
            "role": "a-roll",
            "cutaways": [{"clip": "c003", "in": 8.0, "out": 11.0, "after_sentence": "c001#3"}],
        }
    ]
    cut, _ = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    first, second = _speech(cut)
    assert first.shots == []
    assert [s.after for s in second.shots] == ["c001#3"]


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
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert cut.beats[0].shots == []
    assert stats["dropped_cutaways"]


def test_a_cutaway_on_an_unknown_clip_is_dropped(planned: Project) -> None:
    """A shot that cannot show a frame is worse than no shot at all."""
    _write_sentence_catalogue(
        planned, [_sentence("c001#1", "c001", 10.0, 12.0, "Jedziemy tramwajem.")]
    )
    raw = json.loads(json.dumps(LLM_PLAN))
    raw["segments"] = [
        {"clip": "c001", "sentences": ["c001#1"], "role": "a-roll",
         "cutaways": [{"clip": "c999", "in": 0.0, "out": 3.0, "after_sentence": "c001#1"}]}
    ]
    cut, stats = build_cut(EditPlan.from_llm(raw), planned, FOOTAGE_LOG)
    assert cut.beats[0].shots == []
    assert any("c999" in d for d in stats["dropped_cutaways"])


def test_a_plan_without_a_sentence_catalogue_is_all_broll(planned: Project) -> None:
    """Nothing in the fixture log has a transcript, so nothing is speech."""
    assert not (planned.analysis_dir / "sentences.json").exists()
    cut, stats = build_cut(EditPlan.from_llm(LLM_PLAN), planned, FOOTAGE_LOG)
    assert cut.beats and all(b.kind == "broll" for b in cut.beats)
    assert stats["sentence_segments"] == 0



# ----------------------------------------------------------------------
# stage behaviour
# ----------------------------------------------------------------------
def test_plan_writes_all_artifacts(planned: Project) -> None:
    result = plan(planned, notes="skróć logistykę")
    assert Path(result["cut"]).name == "cut.json"
    assert Path(result["timeline"]).name == "timeline.json"
    assert load_cut(Path(result["cut"])).beats
    assert json.loads(Path(result["timeline"]).read_text())["version"] == 2
    for key in ("cut", "edit_plan", "markdown", "narration"):
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


def _mark_human_edited(planned: Project) -> None:
    """Flag ``plan/cut.json`` the way the web editor's save does."""
    cut = load_cut(cut_path(planned))
    cut.meta.edited_by_human = True
    cut.meta.notes = "hand-edited cut"
    save_cut(cut, cut_path(planned))


def test_a_human_edited_cut_is_never_overwritten(planned: Project) -> None:
    plan(planned)
    _mark_human_edited(planned)
    before = planned.timeline_file.read_text(encoding="utf-8")

    result = plan(planned)
    assert result["draft"] is True
    assert Path(result["cut"]).name == "cut.draft.json"
    assert (planned.plan_dir / "cut.draft.json").exists()
    assert load_cut(cut_path(planned)).meta.notes == "hand-edited cut"
    # The derived timeline still matches the cut on disk, not the draft.
    assert planned.timeline_file.read_text(encoding="utf-8") == before

    forced = plan(planned, force=True)
    assert Path(forced["cut"]).name == "cut.json"
    assert load_cut(cut_path(planned)).meta.edited_by_human is False


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
    assert result["beats"] == first["beats"]
    assert result["segments"] == first["segments"]
    assert len(load_cut(Path(result["cut"])).beats) == result["beats"]
    # No new charge was recorded against the project's cost ledger.
    assert len(planned.load_state().get("costs", [])) == costs_before


def test_plan_from_response_honours_the_human_edited_guard(planned: Project) -> None:
    plan(planned)
    _mark_human_edited(planned)

    result = plan(planned, from_response=True)
    assert result["draft"] is True
    assert Path(result["cut"]).name == "cut.draft.json"
    assert load_cut(cut_path(planned)).meta.notes == "hand-edited cut"


def test_plan_from_response_without_a_prior_run_fails_clearly(planned: Project) -> None:
    with pytest.raises(PlanError, match="planner_response.json"):
        plan(planned, from_response=True)


def test_plan_records_an_error_status(planned: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOpenRouter.answer = {"segments": [{"clip": "c999", "in": 0.0, "out": 5.0}]}
    with pytest.raises(PlanError, match="no usable beats"):
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
    # The cut (built deterministically by build_cut) is never asked of the model.
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
