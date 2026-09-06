"""Tests for ``ytedit.ai.sentences`` — the script-first sentence catalogue.

Everything here is offline: synthetic transcripts and analyses on disk, no
media and no API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from ytedit.ai.sentences import (
    build_sentence_catalogue,
    compact_footage_log_for_planner,
    flag_duplicates,
    flag_retakes,
    load_sentence_index,
    sentence_index as index_of,
    split_words_into_sentences,
    write_sentences,
)
from ytedit.ai.tidy import Word
from ytedit.project import Project


# ----------------------------------------------------------------------
# helpers (mirrors tests/test_tidy.py's add_clip)
# ----------------------------------------------------------------------
def add_clip(
    project: Project,
    clip_id: str,
    duration: float,
    words: Sequence[tuple[float, float, str]] = (),
    instructions: Sequence[tuple[float, float]] = (),
    takes: Sequence[dict[str, Any]] = (),
    language: str = "pl",
) -> None:
    """Register a clip and write its transcript / analysis."""
    project.add_clip(
        {"id": clip_id, "order": int(clip_id[1:]), "duration": duration,
         "width": 1920, "height": 1080, "orientation": "horizontal", "has_audio": True}
    )
    project.transcript_path(clip_id).write_text(
        json.dumps(
            {
                "clip": clip_id,
                "language": language,
                "words": [{"t": t, "s": s, "e": e} for s, e, t in words],
            }
        ),
        encoding="utf-8",
    )
    project.analysis_path(clip_id).write_text(
        json.dumps(
            {
                "clip": clip_id,
                "instructions": [
                    {"s": s, "e": e, "text": "cut this", "action": "discard"}
                    for s, e in instructions
                ],
                "takes": list(takes),
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def project(tmp_path: Path) -> Project:
    return Project.create("t-sent", language="pl", root=tmp_path / "projects")


def word(s: float, e: float, t: str) -> Word:
    return Word(s, e, t)


# ----------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------
def test_splits_on_terminal_punctuation() -> None:
    words = [
        word(0.0, 0.4, "Jedziemy"),
        word(0.5, 0.9, "dalej."),
        word(1.0, 1.4, "To"),
        word(1.5, 1.9, "jest"),
        word(2.0, 2.5, "tramwaj?"),
    ]
    chunks = split_words_into_sentences(words)
    assert [[w.text for w in c] for c in chunks] == [
        ["Jedziemy", "dalej."],
        ["To", "jest", "tramwaj?"],
    ]


def test_splits_on_a_long_pause_without_punctuation() -> None:
    words = [
        word(0.0, 0.4, "Jedziemy"),
        word(0.5, 0.9, "dalej"),
        # 1.5 s silence — over the 1.2 s threshold, no terminal punctuation.
        word(2.4, 2.8, "Teraz"),
        word(2.9, 3.3, "tramwaj"),
    ]
    chunks = split_words_into_sentences(words)
    assert [[w.text for w in c] for c in chunks] == [
        ["Jedziemy", "dalej"],
        ["Teraz", "tramwaj"],
    ]


def test_a_short_pause_does_not_split() -> None:
    words = [word(0.0, 0.4, "Jedziemy"), word(0.9, 1.3, "dalej.")]
    chunks = split_words_into_sentences(words)
    assert len(chunks) == 1


def test_trailing_words_without_terminal_punctuation_still_close_at_clip_end() -> None:
    words = [word(0.0, 0.4, "Jedziemy"), word(0.5, 0.9, "dalej")]
    chunks = split_words_into_sentences(words)
    assert len(chunks) == 1 and chunks[0][-1].text == "dalej"


# ----------------------------------------------------------------------
# instruction flag
# ----------------------------------------------------------------------
def test_instruction_flag_marks_overlapping_sentences(project: Project) -> None:
    add_clip(
        project,
        "c001",
        10.0,
        words=[
            (0.0, 0.4, "To"), (0.5, 0.9, "na"), (1.0, 1.6, "koniec."),
            (3.0, 3.4, "Jedziemy"), (3.5, 3.9, "dalej."),
        ],
        instructions=[(0.0, 1.6)],
    )
    doc = build_sentence_catalogue(project)
    sentences = doc["clips"][0]["sentences"]
    assert sentences[0]["instruction"] is True
    assert sentences[1]["instruction"] is False
    assert sentences[0]["keep_default"] is False
    assert sentences[1]["keep_default"] is True


# ----------------------------------------------------------------------
# retake mapping
# ----------------------------------------------------------------------
def test_retake_of_points_at_the_kept_attempts_sentence(project: Project) -> None:
    add_clip(
        project,
        "c002",
        30.0,
        words=[
            (5.0, 5.4, "To"), (5.5, 5.9, "jest"), (6.0, 6.6, "tramwaj."),  # rejected attempt
            (14.0, 14.4, "To"), (14.5, 14.9, "jest"), (15.0, 15.6, "tramwaj."),  # kept attempt
        ],
        takes=[
            {"topic": "tramwaj", "attempts": [{"s": 5.0, "e": 7.0}, {"s": 14.0, "e": 16.0}],
             "keep": 1, "reason": "ostatnie podejście"},
        ],
    )
    doc = build_sentence_catalogue(project)
    sentences = doc["clips"][0]["sentences"]
    assert sentences[0]["retake_of"] == sentences[1]["id"]
    assert sentences[0]["keep_default"] is False
    assert sentences[1]["retake_of"] is None
    assert sentences[1]["keep_default"] is True


def test_flag_retakes_ignores_a_single_attempt_take() -> None:
    sentences = [
        {"id": "c001#1", "clip": "c001", "s": 0.0, "e": 1.0, "text": "a",
         "instruction": False, "retake_of": None, "duplicate_of": None},
    ]
    flag_retakes(sentences, {"takes": [{"topic": "x", "attempts": [{"s": 0.0, "e": 1.0}], "keep": 0}]})
    assert sentences[0]["retake_of"] is None


# ----------------------------------------------------------------------
# duplicate detection (global, last-take rule generalized)
# ----------------------------------------------------------------------
def test_duplicate_of_points_at_the_later_sentence_across_clips(project: Project) -> None:
    add_clip(
        project, "c003", 10.0,
        words=[(0.0, 0.4, "Bilet"), (0.5, 0.9, "kosztuje"), (1.0, 1.6, "trzy"), (1.7, 2.1, "euro.")],
    )
    add_clip(
        project, "c004", 10.0,
        words=[(0.0, 0.4, "Bilet"), (0.5, 0.9, "kosztuje"), (1.0, 1.6, "trzy"), (1.7, 2.1, "euro.")],
    )
    doc = build_sentence_catalogue(project)
    by_id = index_of(doc)
    assert by_id["c003#1"]["duplicate_of"] == "c004#1"
    assert by_id["c004#1"]["duplicate_of"] is None
    assert by_id["c003#1"]["keep_default"] is False
    assert by_id["c004#1"]["keep_default"] is True


def test_duplicate_of_is_not_set_when_the_later_match_is_an_instruction() -> None:
    ordered = [
        {"id": "c001#1", "clip": "c001", "s": 0.0, "e": 1.0,
         "text": "wsiadamy do tramwaju", "instruction": False, "retake_of": None,
         "duplicate_of": None},
        {"id": "c001#2", "clip": "c001", "s": 5.0, "e": 6.0,
         "text": "wsiadamy do tramwaju", "instruction": True, "retake_of": None,
         "duplicate_of": None},
    ]
    flag_duplicates(ordered)
    assert ordered[0]["duplicate_of"] is None


def test_dissimilar_sentences_are_not_flagged_as_duplicates(project: Project) -> None:
    add_clip(project, "c005", 10.0, words=[(0.0, 0.4, "Bilet"), (0.5, 0.9, "kosztuje"), (1.0, 1.6, "trzy.")])
    add_clip(project, "c006", 10.0, words=[(0.0, 0.4, "Idziemy"), (0.5, 0.9, "na"), (1.0, 1.6, "plażę.")])
    doc = build_sentence_catalogue(project)
    by_id = index_of(doc)
    assert by_id["c005#1"]["duplicate_of"] is None
    assert by_id["c006#1"]["duplicate_of"] is None


# ----------------------------------------------------------------------
# catalogue / write / load
# ----------------------------------------------------------------------
def test_write_sentences_produces_json_and_markdown(project: Project) -> None:
    add_clip(project, "c001", 10.0, words=[(0.0, 0.4, "Cześć."), (1.0, 1.4, "Jedziemy.")])
    document = write_sentences(project)
    assert (project.analysis_dir / "sentences.json").exists()
    md = (project.analysis_dir / "sentences.md").read_text(encoding="utf-8")
    assert "c001#1" in md and "c001#2" in md
    assert document["sentences_count"] == 2
    assert project.stage_status("sentences") == "done"


def test_a_clip_with_no_transcript_gets_an_empty_sentence_list(project: Project) -> None:
    project.add_clip(
        {"id": "c001", "order": 1, "duration": 5.0, "width": 1920, "height": 1080,
         "orientation": "horizontal", "has_audio": False}
    )
    document = build_sentence_catalogue(project)
    assert document["clips"][0] == {"id": "c001", "sentences": []}


def test_load_sentence_index_round_trips(project: Project) -> None:
    add_clip(project, "c001", 10.0, words=[(0.0, 0.4, "Cześć.")])
    write_sentences(project)
    idx = load_sentence_index(project)
    assert idx["c001#1"]["text"] == "Cześć."


# ----------------------------------------------------------------------
# planner-prompt compaction
# ----------------------------------------------------------------------
def test_compact_footage_log_replaces_segments_and_takes_with_sentences() -> None:
    footage_log = {
        "clips": [
            {
                "clip": "c001",
                "segments": [{"s": 0.0, "e": 5.0, "text": "old free-text segment"}],
                "takes": [{"topic": "x", "attempts": [{"s": 0.0, "e": 5.0}], "keep": 0}],
                "summary": "keep me",
            }
        ]
    }
    catalogue = {
        "clips": [
            {
                "id": "c001",
                "sentences": [
                    {"id": "c001#1", "s": 0.0, "e": 2.0, "text": "Cześć.",
                     "instruction": True, "retake_of": None, "duplicate_of": None},
                    {"id": "c001#2", "s": 2.0, "e": 4.0, "text": "Jedziemy dalej.",
                     "instruction": False, "retake_of": None, "duplicate_of": None},
                    {"id": "c001#3", "s": 4.0, "e": 6.0, "text": "Jedziemy dalej znowu.",
                     "instruction": False, "retake_of": "c001#2", "duplicate_of": None},
                ],
            }
        ]
    }
    compacted = compact_footage_log_for_planner(footage_log, catalogue)
    clip = compacted["clips"][0]
    assert "segments" not in clip and "takes" not in clip
    assert clip["summary"] == "keep me"
    ids = [s["id"] for s in clip["sentences"]]
    # The instruction sentence (c001#1) must never appear at all.
    assert ids == ["c001#2", "c001#3"]
    skip_map = {s["id"]: s.get("skip") for s in clip["sentences"]}
    assert skip_map["c001#2"] is None
    assert "c001#2" in skip_map["c001#3"]


def test_compact_footage_log_leaves_a_clip_with_no_speech_untouched() -> None:
    footage_log = {"clips": [{"clip": "c009", "segments": [{"s": 0.0, "e": 5.0, "text": "b-roll"}]}]}
    compacted = compact_footage_log_for_planner(footage_log, {"clips": []})
    assert compacted["clips"][0]["segments"] == [{"s": 0.0, "e": 5.0, "text": "b-roll"}]
