"""Transcript words in clip time — the shared vocabulary of the cut model.

Speech is addressed by sentence ids (``analysis/sentences.json``) or word
indices, never by seconds, and both are built on top of the word spans this
module reads out of ``transcripts/<clip>.json``. Every module that needs to
know where a word starts or ends — the sentence catalogue, the resolver
(:mod:`ytedit.cut`), the narration cleanup (:mod:`ytedit.ai.voice`), the
timecode inspector, the render verifier, QC and the noise scanner — imports
from here.

It deliberately has no dependency beyond :mod:`ytedit.project`: it is the
bottom of the stack.
"""

from __future__ import annotations

import json
from typing import NamedTuple, Sequence, TYPE_CHECKING

from ytedit.log import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ytedit.project import Project

log = get_logger(__name__)

#: Minimum clearance kept from a neighbouring word when an audio window is
#: pulled back off it (seconds). The resolver's ``pacing.word_guard`` setting
#: defaults to this.
WORD_GUARD: float = 0.05

#: A transcript word ending in one of these closes a sentence.
SENTENCE_END: str = ".?!…"


class Word(NamedTuple):
    """One transcript word in clip time."""

    s: float
    e: float
    text: str


def load_words(project: "Project", clip_id: str) -> list[Word]:
    """Read ``transcripts/<clip>.json`` as sorted :class:`Word` spans.

    Args:
        project: Project holding the transcripts.
        clip_id: Clip id such as ``c004``.

    Returns:
        Words in clip time, sorted by start; empty when there is no transcript.
    """
    path = project.transcript_path(clip_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - bad transcript
        log.warning("unreadable transcript %s: %s", path, exc)
        return []
    words: list[Word] = []
    for raw in data.get("words", []):
        if not isinstance(raw, dict):
            continue
        if raw.get("type") not in (None, "word"):
            continue
        start = raw.get("s", raw.get("start"))
        end = raw.get("e", raw.get("end"))
        if start is None or end is None:
            continue
        try:
            s, e = float(start), float(end)
        except (TypeError, ValueError):  # pragma: no cover - malformed transcript
            continue
        if e <= s:
            continue
        words.append(Word(s, e, str(raw.get("t", raw.get("text", ""))).strip()))
    words.sort(key=lambda w: (w.s, w.e))
    return words


def ends_sentence(word: Word) -> bool:
    """True when the word's text closes a sentence (``.``, ``?``, ``!``, ``…``)."""
    text = word.text.strip()
    return bool(text) and text[-1] in SENTENCE_END


def has_sentence_marks(words: Sequence[Word]) -> bool:
    """True when a transcript punctuates at all, so sentences can be located.

    A transcript that never writes a full stop (a hand-made one, or an STT run
    with punctuation disabled) carries no sentence information: the catalogue
    then falls back to gap-based splitting instead of pretending to know where
    a thought ends.
    """
    return any(ends_sentence(w) for w in words)
