"""Offline tests for the analyze stage.

``ytedit.ai.analyze.OpenRouter`` is monkeypatched with a fake that returns
canned analyst JSON, so prompt assembly, schema validation, time clamping and
the footage-log merge are all exercised without a network call.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ytedit.ai.analyze import (
    ClipAnalysis,
    analyze,
    analyze_clip,
    build_footage_log,
    build_user_prompt,
    clamp_times,
    compact_transcript,
    load_footage_log,
    render_footage_log_md,
    select_frames,
    write_footage_log,
)
from ytedit.project import Project

INSTRUCTION = "To nagranie daj na sam koniec filmu jako podsumowanie."


def canned_analysis(clip: str = "c001", **overrides: Any) -> dict[str, Any]:
    """A well-formed analyst answer for the narrated fixture clip."""
    document = {
        "clip": clip,
        "summary": "Narrator nagrywa podsumowanie i dwa razy podaje cenę biletu.",
        "location": {"name": "Alfama", "city": "Lizbona", "country": "Portugalia", "confidence": 0.7},
        "kind": "a-roll",
        "instructions": [
            {"s": 0.2, "e": 3.4, "text": INSTRUCTION, "action": "move_to_end"}
        ],
        "takes": [
            {
                "topic": "cena biletu na tramwaj 28",
                "attempts": [{"s": 5.0, "e": 8.6}, {"s": 14.0, "e": 18.4}],
                "keep": 1,
                "reason": "last take, adds the exact price",
            }
        ],
        "segments": [
            {"s": 0.2, "e": 3.4, "role": "instruction", "keep": False, "text": INSTRUCTION, "quality": 0.9},
            {"s": 14.0, "e": 18.4, "role": "narration", "keep": True, "text": "Bilet...", "quality": 0.8},
        ],
        "background_music": [{"s": 20.0, "e": 24.0, "confidence": 0.8, "suggest": "mute"}],
        "visual": {
            "quality": 0.6,
            "issues": ["slightly underexposed"],
            "best_frames": [12.0],
            "thumbnail_candidate": False,
        },
        "hooks": ["najbardziej zatłoczony tramwaj w Europie"],
        "numbers": ["3 EUR bilet"],
        "topics": ["tramwaj 28", "Alfama"],
    }
    document.update(overrides)
    return document


class FakeOpenRouter:
    """Stand-in for :class:`ytedit.ai.openrouter.OpenRouter`."""

    calls: list[dict[str, Any]] = []
    answers: list[Any] = []

    def __init__(self, api_key: str, cost_callback=None, **kwargs: Any) -> None:
        self.cost_callback = cost_callback

    def ask_json(self, model: str, system_prompt: str, user: str, schema_hint: str, images=(), **kw: Any):
        type(self).calls.append(
            {"model": model, "system": system_prompt, "user": user, "images": len(images), **kw}
        )
        if self.cost_callback:
            self.cost_callback(
                service="openrouter", op="chat", model=model, units="1000+400 tok", usd=0.006
            )
        answers = type(self).answers
        if not answers:
            return canned_analysis()
        return answers.pop(0) if len(answers) > 1 else answers[0]

    def close(self) -> None:
        pass


def jpeg(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def transcript_document(clip: str = "c001") -> dict[str, Any]:
    words = []
    t = 0.2
    for word in (INSTRUCTION + " Bilet kosztuje trzy euro.").split():
        words.append({"t": word, "s": round(t, 2), "e": round(t + 0.35, 2), "p": -0.05, "speaker": "speaker_0"})
        t += 0.4
    return {
        "clip": clip,
        "language": "pol",
        "language_probability": 0.98,
        "project_language": "pl",
        "language_mismatch": False,
        "has_audio": True,
        "duration": 25.0,
        "text": INSTRUCTION,
        "words": words,
        "events": [{"type": "music", "s": 20.0, "e": 24.0}],
        "speakers": ["speaker_0"],
        "engine": "elevenlabs/scribe_v2",
        "take_hints": [
            {
                "topic_hint": "bilet kosztuje trzy euro",
                "attempts": [
                    {"s": 5.0, "e": 8.6, "text": "Bilet kosztuje trzy euro"},
                    {"s": 14.0, "e": 18.4, "text": "Bilet kosztuje 3 euro"},
                ],
            }
        ],
    }


@pytest.fixture()
def analyzed_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    """Transcribed project with frames on disk, ready to analyze."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    project = Project.create("t-an", language="pl", root=tmp_path / "projects")
    project.add_clip(
        {
            "id": "c001", "order": 1, "source_file": "input/narration.mp4",
            "duration": 25.0, "has_audio": True, "orientation": "horizontal",
            "recorded_at": "2026-08-12T10:00:00+00:00", "stages": {"ingest": "done", "transcribe": "done"},
        }
    )
    project.add_clip(
        {
            "id": "c002", "order": 2, "source_file": "input/broll.mp4",
            "duration": 4.0, "has_audio": False, "orientation": "vertical",
            "recorded_at": "2026-08-12T10:05:00+00:00", "stages": {"ingest": "done", "transcribe": "done"},
        }
    )
    project.transcript_path("c001").write_text(
        json.dumps(transcript_document(), ensure_ascii=False), encoding="utf-8"
    )
    project.transcript_path("c002").write_text(
        json.dumps(
            {
                "clip": "c002", "language": None, "language_probability": None,
                "project_language": "pl", "language_mismatch": False, "has_audio": False,
                "duration": 4.0, "text": "", "words": [], "events": [], "speakers": [],
                "engine": "none/no-audio", "take_hints": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    frames = project.clip_frames_dir("c001")
    frames.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        (frames / f"{i:03d}.jpg").write_bytes(jpeg((10 * i, 40, 80)))
    (frames / "index.json").write_text(
        json.dumps({"times": [round(i * 3.0, 3) for i in range(12)]}), encoding="utf-8"
    )
    FakeOpenRouter.calls = []
    FakeOpenRouter.answers = []
    monkeypatch.setattr("ytedit.ai.analyze.OpenRouter", FakeOpenRouter)
    return project


# --------------------------------------------------------------------------- #
# prompt inputs
# --------------------------------------------------------------------------- #


def test_compact_transcript_drops_confidence_and_rounds() -> None:
    payload = json.loads(compact_transcript(transcript_document()))
    assert payload[0] == ["To", 0.2, 0.55]
    assert all(len(item) == 3 for item in payload)


def test_compact_transcript_degrades_when_too_long() -> None:
    document = transcript_document()
    document["words"] = [
        {"t": f"slowo{i}", "s": i * 0.4, "e": i * 0.4 + 0.3} for i in range(20_000)
    ]
    compacted = compact_transcript(document, token_budget=500)
    assert len(compacted) <= 500 * 4
    assert json.loads(compacted)  # still valid JSON


def test_compact_transcript_handles_empty_transcript() -> None:
    assert compact_transcript(None) == "[]"
    assert json.loads(compact_transcript({"words": [], "text": "abc"}))["text"] == "abc"


def test_select_frames_picks_evenly_spaced_frames(analyzed_project: Project) -> None:
    frames = select_frames(analyzed_project, "c001")
    assert len(frames) == 8
    times = [t for t, _ in frames]
    assert times[0] == 0.0 and times[-1] == 33.0
    assert times == sorted(times) and len(set(times)) == 8
    assert select_frames(analyzed_project, "c002") == []


def test_user_prompt_carries_hints_and_placeholders(analyzed_project: Project) -> None:
    clip = analyzed_project.get_clip("c001")
    prompt = build_user_prompt(analyzed_project, clip, transcript_document(), [0.0, 3.0])
    assert "Clip id: c001" in prompt
    assert "Project language: pl" in prompt
    assert "Clip duration: 25.00 s" in prompt
    assert "(see attached images)" in prompt
    assert "Deterministic take hints (verify)" in prompt
    assert "bilet kosztuje trzy euro" in prompt
    assert '"type": "music"' in prompt or '"type":"music"' in prompt


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_model_is_lenient_about_missing_and_extra_fields() -> None:
    analysis = ClipAnalysis.model_validate(
        {"clip": "c009", "kind": "aroll", "instructions": None, "extra_key": 1}
    )
    assert analysis.kind == "a-roll"
    assert analysis.instructions == [] and analysis.takes == []
    assert analysis.visual.thumbnail_candidate is False
    assert analysis.model_dump()["extra_key"] == 1


def test_unknown_kind_falls_back_to_broll() -> None:
    assert ClipAnalysis.model_validate({"kind": "montage"}).kind == "b-roll"


def test_clamp_times_keeps_everything_inside_the_clip() -> None:
    analysis = ClipAnalysis.model_validate(
        {
            "instructions": [{"s": -3.0, "e": 99.0}],
            "takes": [{"attempts": [{"s": 1.0, "e": 2.0}, {"s": 3.0, "e": 400.0}], "keep": 7}],
            "segments": [{"s": 30.0, "e": 5.0}],
            "visual": {"best_frames": [-1.0, 500.0]},
        }
    )
    clamped = clamp_times(analysis, 25.0)
    assert (clamped.instructions[0].s, clamped.instructions[0].e) == (0.0, 25.0)
    assert clamped.takes[0].attempts[1].e == 25.0
    assert clamped.takes[0].keep == 1  # clamped to the last existing attempt
    assert clamped.segments[0].e >= clamped.segments[0].s
    assert clamped.visual.best_frames == [0.0, 25.0]


# --------------------------------------------------------------------------- #
# per-clip analysis
# --------------------------------------------------------------------------- #


def test_analyze_clip_uses_the_analyst_model_and_frames(analyzed_project: Project) -> None:
    document = analyze_clip(analyzed_project, analyzed_project.get_clip("c001"))
    call = FakeOpenRouter.calls[0]
    assert call["model"] == "anthropic/claude-sonnet-5"
    assert call["images"] == 8
    assert call["detail"] == "low"
    assert len(call["labels"]) == 8 and call["labels"][0].startswith("Frame 0 (t=0.0s)")
    assert document["kind"] == "a-roll"
    assert document["instructions"][0]["action"] == "move_to_end"
    assert document["frames_used"][0] == 0.0
    assert document["model"] == "anthropic/claude-sonnet-5"
    assert document["duration"] == 25.0


def test_silent_clip_uses_the_cheaper_vision_model(analyzed_project: Project) -> None:
    FakeOpenRouter.answers = [canned_analysis("c002", kind="silent-broll", instructions=[], takes=[])]
    document = analyze_clip(analyzed_project, analyzed_project.get_clip("c002"))
    assert FakeOpenRouter.calls[0]["model"] == "google/gemini-3.8-flash"
    assert document["kind"] == "silent-broll"
    assert document["has_audio"] is False
    assert "NO audio track" in FakeOpenRouter.calls[0]["user"]


def test_invalid_answer_is_retried_once_with_the_error(analyzed_project: Project) -> None:
    FakeOpenRouter.answers = [canned_analysis(summary=123), canned_analysis()]
    document = analyze_clip(analyzed_project, analyzed_project.get_clip("c001"))
    assert len(FakeOpenRouter.calls) == 2
    assert "did not validate against" in FakeOpenRouter.calls[1]["user"]
    assert document["summary"].startswith("Narrator")


def test_answer_times_are_clamped_to_the_clip_duration(analyzed_project: Project) -> None:
    FakeOpenRouter.answers = [
        canned_analysis(instructions=[{"s": -1.0, "e": 900.0, "text": "x", "action": "discard"}])
    ]
    document = analyze_clip(analyzed_project, analyzed_project.get_clip("c001"))
    assert document["instructions"][0] == {"s": 0.0, "e": 25.0, "text": "x", "action": "discard"}


# --------------------------------------------------------------------------- #
# stage + footage log
# --------------------------------------------------------------------------- #


def test_analyze_stage_writes_everything(analyzed_project: Project) -> None:
    results = analyze(analyzed_project)
    assert {r.clip_id: r.status for r in results} == {"c001": "done", "c002": "done"}
    assert analyzed_project.analysis_path("c001").exists()
    assert (analyzed_project.analysis_dir / "footage_log.json").exists()
    assert (analyzed_project.analysis_dir / "footage_log.md").exists()

    state = analyzed_project.load_state()
    assert state["clips"]["c001"]["stages"]["analyze"] == "done"
    assert state["stages"]["analyze"]["status"] == "done"
    assert state["stages"]["analyze"]["cost_usd"] > 0
    assert [c["op"] for c in state["costs"]] == ["chat", "chat"]


def test_footage_log_header_and_entries(analyzed_project: Project) -> None:
    analyze(analyzed_project)
    document = load_footage_log(analyzed_project)
    assert document["project"] == "t-an"
    assert document["language"] == "pl"
    assert document["clips_count"] == 2
    assert document["total_duration"] == 29.0
    assert document["locations"] == ["Alfama"]
    assert document["language_mismatches"] == []
    assert len(document["instructions_found"]) == 2  # canned answer reused per clip
    assert document["instructions_found"][0]["clip"] == "c001"
    assert document["music_flags"][0]["suggest"] == "mute"

    first = document["clips"][0]
    assert set(first) >= {
        "id", "order", "source_file", "duration", "orientation", "kind", "has_audio",
        "language", "summary", "location", "instructions", "takes", "segments",
        "background_music", "visual", "hooks", "numbers", "topics", "recorded_at",
    }
    assert first["id"] == "c001" and first["language"] == "pol"
    assert first["recorded_at"] == "2026-08-12T10:00:00+00:00"


def test_footage_log_markdown_lists_instructions_takes_and_music() -> None:
    document = {
        "project": "demo", "language": "pl", "total_duration": 25.0, "clips_count": 1,
        "instructions_found": [
            {"clip": "c001", "s": 0.2, "e": 3.4, "text": INSTRUCTION, "action": "move_to_end"}
        ],
        "music_flags": [{"clip": "c001", "s": 20.0, "e": 24.0, "confidence": 0.8, "suggest": "mute"}],
        "language_mismatches": ["c001"],
        "locations": ["Alfama"],
        "clips": [
            {
                "id": "c001", "order": 1, "duration": 25.0, "kind": "a-roll", "has_audio": True,
                "location": {"name": "Alfama", "city": "Lizbona"}, "summary": "streszczenie",
                "takes": [
                    {
                        "topic": "cena biletu",
                        "attempts": [{"s": 5.0, "e": 8.6}, {"s": 14.0, "e": 18.4}],
                        "keep": 1, "reason": "last take",
                    }
                ],
            }
        ],
    }
    markdown = render_footage_log_md(document)
    assert "# Footage log — demo" in markdown
    assert "| c001 | 1 | 25.0s | a-roll | yes | Alfama / Lizbona | streszczenie |" in markdown
    assert "`move_to_end`" in markdown and INSTRUCTION in markdown
    assert "suggest `mute`" in markdown
    assert "keep #1" in markdown and "last take" in markdown
    assert "## Language mismatches" in markdown


def test_footage_log_is_rebuilt_without_reanalyzing(analyzed_project: Project) -> None:
    analyze(analyzed_project)
    calls = len(FakeOpenRouter.calls)
    (analyzed_project.analysis_dir / "footage_log.json").unlink()
    assert analyze(analyzed_project) == []
    assert len(FakeOpenRouter.calls) == calls
    assert load_footage_log(analyzed_project)["clips_count"] == 2
    analyze(analyzed_project, force=True)
    assert len(FakeOpenRouter.calls) == calls + 2


def test_failed_clip_marks_state_and_stage(
    analyzed_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("ytedit.ai.analyze.analyze_clip", boom)
    results = analyze(analyzed_project)
    assert all(r.status == "error" for r in results)
    state = analyzed_project.load_state()
    assert state["clips"]["c001"]["stages"]["analyze"] == "error"
    assert state["stages"]["analyze"]["status"] == "error"
    # The log is still written so downstream stages see an (empty) inventory.
    assert build_footage_log(analyzed_project)["clips_count"] == 2


def test_load_footage_log_missing_returns_empty(analyzed_project: Project) -> None:
    assert load_footage_log(analyzed_project) == {}
    write_footage_log(analyzed_project)
    assert load_footage_log(analyzed_project)["clips_count"] == 2
