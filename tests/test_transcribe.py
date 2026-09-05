"""Offline tests for the transcribe stage (no network, no keys, no cost).

The ElevenLabs / OpenRouter clients are monkeypatched inside
``ytedit.ai.transcribe`` with fakes that answer from a canned Scribe-shaped
payload, so the whole stage - documents, SRT, TXT, take hints, ledger, stage
bookkeeping - is exercised without spending anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ytedit.ai.elevenlabs import ElevenLabsError, normalize_transcript
from ytedit.ai.transcribe import (
    build_cues,
    build_sentences,
    detect_takes,
    language_mismatch,
    load_transcript,
    all_transcripts,
    normalize_token,
    srt_timestamp,
    transcribe,
    write_srt,
    write_txt,
)
from ytedit.project import Project


# --------------------------------------------------------------------------- #
# canned data
# --------------------------------------------------------------------------- #

INSTRUCTION = "To nagranie daj na sam koniec filmu jako podsumowanie."
TAKE_A = "Bilet na tramwaj numer dwadzieścia osiem kosztuje trzy euro."
TAKE_B = "Bilet na tramwaj numer dwadzieścia osiem kosztuje 3 euro i pięćdziesiąt."


def words_for(text: str, start: float, step: float = 0.4) -> list[dict[str, Any]]:
    """Build Scribe-shaped word items for a sentence, one word every ``step``."""
    items: list[dict[str, Any]] = []
    t = start
    for word in text.split():
        items.append(
            {
                "text": word,
                "start": round(t, 3),
                "end": round(t + step - 0.05, 3),
                "type": "word",
                "speaker_id": "speaker_0",
                "logprob": -0.05,
            }
        )
        items.append({"text": " ", "start": round(t + step - 0.05, 3), "end": round(t + step, 3), "type": "spacing"})
        t += step
    return items


def scribe_payload(language: str = "pol", probability: float = 0.98) -> dict[str, Any]:
    """A realistic ``POST /v1/speech-to-text`` response."""
    words = (
        words_for(INSTRUCTION, 0.2)
        + words_for(TAKE_A, 5.0)
        + words_for("Jeszcze raz.", 12.0)
        + words_for(TAKE_B, 14.0)
    )
    words.append({"text": "(music)", "start": 20.0, "end": 24.0, "type": "audio_event"})
    return {
        "language_code": language,
        "language_probability": probability,
        "text": f"{INSTRUCTION} {TAKE_A} Jeszcze raz. {TAKE_B}",
        "words": words,
        "audio_duration_secs": 25.0,
    }


class FakeElevenLabs:
    """Stand-in for :class:`ytedit.ai.elevenlabs.ElevenLabs`."""

    calls: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    raises: Exception | None = None

    def __init__(self, api_key: str, cost_callback=None, **kwargs: Any) -> None:
        self.cost_callback = cost_callback

    def transcribe(self, audio_path, **kwargs: Any):
        type(self).calls.append({"audio": str(audio_path), **kwargs})
        if type(self).raises is not None:
            raise type(self).raises
        transcript = normalize_transcript(type(self).payload or scribe_payload())
        if self.cost_callback:
            self.cost_callback(
                service="elevenlabs",
                op="stt",
                model="scribe_v2",
                units=f"{transcript.duration_s:.1f}s",
                usd=transcript.cost_usd,
            )
        return transcript

    def close(self) -> None:
        pass


class FakeOpenRouter:
    """Stand-in for the OpenRouter STT fallback."""

    calls: list[dict[str, Any]] = []

    def __init__(self, api_key: str, cost_callback=None, **kwargs: Any) -> None:
        self.cost_callback = cost_callback

    def transcribe_audio(self, path, model: str = "openai/whisper-large-v3", language=None, **kw: Any):
        type(self).calls.append({"path": str(path), "model": model, "language": language})
        if self.cost_callback:
            self.cost_callback(
                service="openrouter", op="transcribe", model=model, units="25.0s", usd=0.0002
            )
        return {
            "text": TAKE_A,
            "words": [{"t": w, "s": 0.4 * i, "e": 0.4 * i + 0.3} for i, w in enumerate(TAKE_A.split())],
            "language": "pl",
            "duration": 25.0,
            "cost_usd": 0.0002,
            "engine": f"openrouter/{model}",
            "raw": {},
        }

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def stt_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    """Project with one narrated clip and one silent clip, ready to transcribe."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    project = Project.create("t-stt", language="pl", root=tmp_path / "projects")
    project.add_clip(
        {
            "id": "c001", "order": 1, "source_file": "input/narration.mp4",
            "duration": 25.0, "has_audio": True, "orientation": "horizontal",
            "recorded_at": "2026-08-12T10:00:00+00:00",
            "stages": {"ingest": "done"},
        }
    )
    project.add_clip(
        {
            "id": "c002", "order": 2, "source_file": "input/broll.mp4",
            "duration": 4.0, "has_audio": False, "orientation": "vertical",
            "recorded_at": "2026-08-12T10:05:00+00:00",
            "stages": {"ingest": "done"},
        }
    )
    project.audio_path("c001").write_bytes(b"RIFF0000WAVEfake")
    FakeElevenLabs.calls = []
    FakeElevenLabs.payload = scribe_payload()
    FakeElevenLabs.raises = None
    FakeOpenRouter.calls = []
    monkeypatch.setattr("ytedit.ai.transcribe.ElevenLabs", FakeElevenLabs)
    monkeypatch.setattr("ytedit.ai.transcribe.OpenRouter", FakeOpenRouter)
    return project


# --------------------------------------------------------------------------- #
# take detection
# --------------------------------------------------------------------------- #


def test_normalize_token_folds_diacritics_and_numbers() -> None:
    assert normalize_token("Pięćdziesiąt,") == "piecdziesiat"
    assert normalize_token("Trzy") == normalize_token("3") == "3"
    assert normalize_token("...") == ""


def test_detect_takes_finds_the_repeated_polish_sentence() -> None:
    words = [
        {"t": w["text"], "s": w["start"], "e": w["end"]}
        for w in scribe_payload()["words"]
        if w["type"] == "word"
    ]
    groups = detect_takes(words)
    assert len(groups) == 1
    attempts = groups[0]["attempts"]
    assert len(attempts) == 2
    # First attempt around 5 s, second one around 14 s.
    assert attempts[0]["s"] == pytest.approx(5.0, abs=0.3)
    assert attempts[1]["s"] == pytest.approx(14.0, abs=0.3)
    assert "tramwaj" in groups[0]["topic_hint"]
    # "trzy" and "3" normalise to the same token, so both attempts cover the price.
    assert "trzy" in attempts[0]["text"] and "3" in attempts[1]["text"]


def test_detect_takes_ignores_repeats_outside_the_window() -> None:
    words = [
        {"t": w["text"], "s": w["start"], "e": w["end"]}
        for w in (words_for(TAKE_A, 0.0) + words_for(TAKE_A, 400.0))
        if w["type"] == "word"
    ]
    assert detect_takes(words, window_s=60.0) == []
    assert len(detect_takes(words, window_s=600.0)) == 1


def test_detect_takes_on_unique_speech_returns_nothing() -> None:
    words = [
        {"t": w["text"], "s": w["start"], "e": w["end"]}
        for w in words_for(INSTRUCTION, 0.0)
        if w["type"] == "word"
    ]
    assert detect_takes(words) == []


# --------------------------------------------------------------------------- #
# SRT / TXT
# --------------------------------------------------------------------------- #


def test_srt_timestamp_format() -> None:
    assert srt_timestamp(0) == "00:00:00,000"
    assert srt_timestamp(3661.5) == "01:01:01,500"


def test_cues_respect_word_and_duration_limits() -> None:
    words = [{"t": f"slowo{i}", "s": i * 0.5, "e": i * 0.5 + 0.4} for i in range(40)]
    cues = build_cues(words)
    assert cues
    for cue in cues:
        assert 1 <= len(cue["text"].split()) <= 5
        assert 0 < cue["e"] - cue["s"] <= 4.0 + 1e-6
    # Cues never overlap and are ordered.
    for left, right in zip(cues, cues[1:]):
        assert left["e"] <= right["s"] + 1e-6


def test_short_cue_is_stretched_to_one_second() -> None:
    words = [{"t": "raz", "s": 0.0, "e": 0.2}]
    cue = build_cues(words)[0]
    assert cue["e"] - cue["s"] >= 1.0


def test_write_srt_and_txt(tmp_path: Path) -> None:
    words = [
        {"t": w["text"], "s": w["start"], "e": w["end"]}
        for w in scribe_payload()["words"]
        if w["type"] == "word"
    ]
    srt = write_srt(tmp_path / "c001.srt", words)
    body = srt.read_text(encoding="utf-8")
    assert body.startswith("1\n00:00:00,200 --> ")
    assert "-->" in body and "podsumowanie" in body

    txt = write_txt(tmp_path / "c001.txt", words)
    lines = txt.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(build_sentences(words))
    assert lines[0].startswith("[00:00.2] ")
    assert lines[0].endswith(INSTRUCTION)


# --------------------------------------------------------------------------- #
# language flags
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("detected", "probability", "expected"),
    [
        ("pol", 0.98, False),
        ("pl", 0.98, False),
        ("eng", 0.98, True),
        ("eng", 0.4, False),   # low confidence -> not flagged
        (None, 0.99, False),
        ("eng", None, False),
    ],
)
def test_language_mismatch(detected, probability, expected) -> None:
    assert language_mismatch(detected, "pl", probability) is expected


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #


def test_transcribe_writes_json_srt_and_txt(stt_project: Project) -> None:
    results = transcribe(stt_project)
    assert {r.clip_id: r.status for r in results} == {"c001": "done", "c002": "stub"}

    document = json.loads(stt_project.transcript_path("c001").read_text(encoding="utf-8"))
    assert document["clip"] == "c001"
    assert document["language"] == "pol"
    assert document["project_language"] == "pl"
    assert document["language_mismatch"] is False
    assert document["has_audio"] is True
    assert document["duration"] == 25.0
    assert document["engine"] == "elevenlabs/scribe_v2"
    assert document["speakers"] == ["speaker_0"]
    assert document["events"] == [{"type": "music", "s": 20.0, "e": 24.0}]
    assert len(document["words"]) == len(document["text"].split())
    assert document["words"][0] == {
        "t": "To", "s": 0.2, "e": 0.55, "p": -0.05, "speaker": "speaker_0",
    }
    assert len(document["take_hints"]) == 1

    assert (stt_project.transcripts_dir / "c001.srt").exists()
    assert (stt_project.transcripts_dir / "c001.txt").exists()

    # Scribe was asked for the settings the playbook requires.
    call = FakeElevenLabs.calls[0]
    # Default is auto-detection: travel footage contains other languages and a
    # forced code would hide the mismatch flag (see ``_language_code``).
    assert call["language_code"] is None
    assert call["diarize"] is True and call["tag_audio_events"] is True


def test_stub_transcript_for_clip_without_audio(stt_project: Project) -> None:
    transcribe(stt_project)
    document = json.loads(stt_project.transcript_path("c002").read_text(encoding="utf-8"))
    assert document["has_audio"] is False
    assert document["words"] == [] and document["text"] == ""
    assert document["engine"] == "none/no-audio"
    assert (stt_project.transcripts_dir / "c002.srt").read_text(encoding="utf-8") == ""


def test_transcribe_updates_state_and_ledger(stt_project: Project) -> None:
    transcribe(stt_project)
    state = stt_project.load_state()
    assert state["clips"]["c001"]["stages"]["transcribe"] == "done"
    assert state["clips"]["c002"]["stages"]["transcribe"] == "done"
    assert state["stages"]["transcribe"]["status"] == "done"
    costs = [c for c in state["costs"] if c["service"] == "elevenlabs"]
    assert len(costs) == 1
    assert costs[0]["op"] == "stt" and costs[0]["units"] == "25.0s"
    assert costs[0]["usd"] == pytest.approx(25.0 / 3600 * 0.22, rel=1e-3)
    assert state["stages"]["transcribe"]["cost_usd"] > 0


def test_transcribe_is_idempotent_and_force_reruns(stt_project: Project) -> None:
    transcribe(stt_project)
    assert len(FakeElevenLabs.calls) == 1
    assert transcribe(stt_project) == []
    assert len(FakeElevenLabs.calls) == 1
    transcribe(stt_project, force=True)
    assert len(FakeElevenLabs.calls) == 2


def test_language_mismatch_flagged_on_disk(stt_project: Project) -> None:
    FakeElevenLabs.payload = scribe_payload(language="eng", probability=0.95)
    transcribe(stt_project)
    document = load_transcript(stt_project, "c001")
    assert document["language_mismatch"] is True
    assert stt_project.load_state()["clips"]["c001"]["language_mismatch"] is True


def test_falls_back_to_openrouter_on_server_error(stt_project: Project) -> None:
    FakeElevenLabs.raises = ElevenLabsError(503, "upstream unavailable")
    results = transcribe(stt_project)
    assert results[0].status == "done"
    assert FakeOpenRouter.calls and FakeOpenRouter.calls[0]["model"] == "openai/whisper-large-v3"
    document = load_transcript(stt_project, "c001")
    assert document["engine"] == "openrouter/openai/whisper-large-v3"


def test_client_error_is_not_retried_and_marks_the_clip(stt_project: Project) -> None:
    FakeElevenLabs.raises = ElevenLabsError(401, "bad key")
    results = transcribe(stt_project)
    assert results[0].status == "error"
    assert not FakeOpenRouter.calls
    state = stt_project.load_state()
    assert state["clips"]["c001"]["stages"]["transcribe"] == "error"
    assert state["stages"]["transcribe"]["status"] == "error"


def test_openrouter_engine_selected_from_settings(
    stt_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    (stt_project.path / "project.yaml").write_text(
        "project: t-stt\nlanguage: pl\nmodels:\n  stt: openrouter/openai/whisper-large-v3-turbo\n",
        encoding="utf-8",
    )
    stt_project._settings = None
    transcribe(stt_project)
    assert not FakeElevenLabs.calls
    assert FakeOpenRouter.calls[0]["model"] == "openai/whisper-large-v3-turbo"


def test_auto_language_detection_sends_no_language_code(stt_project: Project) -> None:
    (stt_project.path / "project.yaml").write_text(
        "project: t-stt\nlanguage: pl\nlanguage_detect: auto\nkeyterms:\n  - Lizbona\n",
        encoding="utf-8",
    )
    stt_project._settings = None
    transcribe(stt_project)
    call = FakeElevenLabs.calls[0]
    assert call["language_code"] is None
    assert call["keyterms"] == ["Lizbona"]


def test_all_transcripts_returns_every_document(stt_project: Project) -> None:
    transcribe(stt_project)
    documents = all_transcripts(stt_project)
    assert list(documents) == ["c001", "c002"]
    assert load_transcript(stt_project, "nope") is None


def test_language_detect_force_pins_project_language(stt_project: Project) -> None:
    cfg_path = stt_project.path / "project.yaml"
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + "\nlanguage_detect: force\n", encoding="utf-8")
    project = Project.load(stt_project.slug, root=stt_project.path.parent)
    transcribe(project, force=True)
    assert FakeElevenLabs.calls[-1]["language_code"] == "pol"
