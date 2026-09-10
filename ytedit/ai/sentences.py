"""Stage: deterministic sentence catalogue — ``analysis/sentences.json``.

The planner used to be handed raw seconds and asked to pick ``in``/``out`` for
every speech segment; in practice that produced cuts landing mid-sentence,
narration played twice, and retakes that survived into the cut. This module
builds, from the already-transcribed and already-analyzed clips, a numbered
catalogue of every sentence in the project (``<clip>#<n>``) so the planner
(``ytedit.ai.plan``) can reference *sentences* instead of seconds, and
:func:`ytedit.cut.resolve` — the one place that turns speech into seconds —
can validate that every reference is legal.

A sentence ends at a word whose text ends in ``.``, ``?``, ``!`` or ``…``, at a
pause longer than :data:`SENTENCE_PAUSE_MAX`, or at the clip's last word —
the boundary rule of :func:`ytedit.words.ends_sentence`, which the resolver
uses too, so the catalogue and the cut always agree on where a sentence ends.

Each sentence carries three independent flags, all advisory to the planner and
enforced deterministically by :func:`ytedit.cut.validate`:

* ``instruction`` — the sentence overlaps a spoken editor instruction
  (``analysis/<clip>.json.instructions[]``) and must never be used.
* ``retake_of`` — the sentence lies inside a take attempt the analysis did
  *not* keep (``takes[].attempts``); it points at a sentence of the attempt
  that *was* kept.
* ``duplicate_of`` — the sentence's text is a near-duplicate (Jaccard word
  overlap >= :data:`DUPLICATE_JACCARD`) of a *later* sentence anywhere in the
  project (the last-take rule generalized across clips); it points at that
  later sentence, unless the later one is itself an instruction. Only
  sentences of at least :data:`DUPLICATE_MIN_WORDS` distinct words are
  compared: short remarks repeat all the time in real speech ("Zobaczcie.",
  "Jest bardzo dobre.") without being a second take of anything.

``keep_default`` is ``true`` only when none of the three flags apply — a
convenience hint for a human skimming ``analysis/sentences.md``, not something
the planner or the resolver reads.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ytedit.words import Word, ends_sentence, load_words
from ytedit.log import get_logger
from ytedit.project import Project, utcnow

log = get_logger(__name__)

STAGE = "sentences"

#: A gap longer than this between two words' timestamps ends a sentence, even
#: without terminal punctuation (matches the "or at a pause > 1.2 s" rule).
SENTENCE_PAUSE_MAX: float = 1.2

#: Minimum normalized word-overlap (Jaccard) for two sentences to count as
#: near-duplicates of each other.
DUPLICATE_JACCARD: float = 0.7

#: Distinct normalized words a sentence must have before it may be flagged a
#: duplicate (on *both* sides of the pair). Below this the overlap measure is
#: meaningless: "Zobaczcie." matches every other "Zobaczcie." in the trip, and
#: two three-word remarks sharing two words already clear the Jaccard bar. The
#: the reference project run produced four such false positives, each of them a real sentence
#: the planner was then told to skip.
DUPLICATE_MIN_WORDS: int = 4

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_EPS = 1e-6


class SentencesError(RuntimeError):
    """Raised when the sentence catalogue cannot be built."""


# ----------------------------------------------------------------------
# analysis-derived ranges (duplicated, minimal, from ytedit.ai.plan to avoid
# a plan.py <-> sentences.py import cycle: plan.py imports this module)
# ----------------------------------------------------------------------
def _span(item: Mapping[str, Any]) -> tuple[float, float]:
    """Read a ``(start, end)`` pair from an ``s``/``e`` or ``in``/``out`` dict."""
    start = item.get("s", item.get("in", item.get("start", item.get("at", 0.0))))
    end = item.get("e", item.get("out", item.get("end", 0.0)))
    try:
        return float(start or 0.0), float(end or 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _instruction_ranges(analysis: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Time ranges of spoken editor instructions in one clip's analysis."""
    ranges: list[tuple[float, float]] = []
    for item in analysis.get("instructions") or []:
        if not isinstance(item, dict):
            continue
        s, e = _span(item)
        if e > s:
            ranges.append((s, e))
    return ranges


def _take_attempts(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return ``takes[]`` with a resolved, in-range ``keep`` index."""
    out: list[dict[str, Any]] = []
    for take in analysis.get("takes") or []:
        if not isinstance(take, dict):
            continue
        attempts = [a for a in (take.get("attempts") or []) if isinstance(a, dict)]
        if len(attempts) < 2:
            continue
        try:
            keep = int(take.get("keep", len(attempts) - 1))
        except (TypeError, ValueError):
            keep = len(attempts) - 1
        if keep < 0:
            keep += len(attempts)
        if not (0 <= keep < len(attempts)):
            keep = len(attempts) - 1
        out.append({"attempts": attempts, "keep": keep})
    return out


# ----------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------
def split_words_into_sentences(words: Sequence[Word]) -> list[list[Word]]:
    """Group words into sentences: punctuation, a long pause, or clip end.

    Args:
        words: Clip-time words, sorted by start (as returned by
            :func:`ytedit.words.load_words`).

    Returns:
        A list of non-empty word groups, in order.
    """
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and word.s - current[-1].e > SENTENCE_PAUSE_MAX + _EPS:
            chunks.append(current)
            current = []
        current.append(word)
        if ends_sentence(word):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _sentence_dict(clip_id: str, n: int, chunk: Sequence[Word], lang: str | None) -> dict[str, Any]:
    text = " ".join(w.text for w in chunk if w.text).strip()
    return {
        "id": f"{clip_id}#{n}",
        "clip": clip_id,
        "n": n,
        "s": round(chunk[0].s, 3),
        "e": round(chunk[-1].e, 3),
        "text": text,
        "words": len(chunk),
        "lang": lang,
        "instruction": False,
        "retake_of": None,
        "duplicate_of": None,
        "keep_default": True,
    }


def build_clip_sentences(
    clip_id: str, words: Sequence[Word], lang: str | None = None
) -> list[dict[str, Any]]:
    """Split one clip's words into unflagged sentence dicts."""
    return [
        _sentence_dict(clip_id, i + 1, chunk, lang)
        for i, chunk in enumerate(split_words_into_sentences(words))
        if chunk
    ]


# ----------------------------------------------------------------------
# flags: instruction / retake_of (per clip, from analysis/<clip>.json)
# ----------------------------------------------------------------------
def _overlaps(sentence: Mapping[str, Any], s: float, e: float) -> bool:
    return sentence["s"] < e - _EPS and sentence["e"] > s + _EPS


def _first_overlapping(sentences: Sequence[dict[str, Any]], s: float, e: float) -> str | None:
    for sent in sentences:
        if _overlaps(sent, s, e):
            return str(sent["id"])
    return None


def flag_instructions(sentences: list[dict[str, Any]], analysis: Mapping[str, Any]) -> None:
    """Set ``instruction`` on every sentence overlapping a spoken instruction."""
    for s, e in _instruction_ranges(analysis):
        for sent in sentences:
            if _overlaps(sent, s, e):
                sent["instruction"] = True


def flag_retakes(sentences: list[dict[str, Any]], analysis: Mapping[str, Any]) -> None:
    """Set ``retake_of`` on every sentence inside a take attempt that lost.

    Points at the first sentence overlapping the *kept* attempt's range —
    the take's own transcript, so this is always in the same clip.
    """
    for take in _take_attempts(analysis):
        attempts = take["attempts"]
        keep = take["keep"]
        kept_s, kept_e = _span(attempts[keep])
        kept_id = _first_overlapping(sentences, kept_s, kept_e)
        if kept_id is None:
            continue
        for i, attempt in enumerate(attempts):
            if i == keep:
                continue
            a_s, a_e = _span(attempt)
            if a_e <= a_s:
                continue
            for sent in sentences:
                if sent["id"] == kept_id:
                    continue
                if _overlaps(sent, a_s, a_e):
                    sent["retake_of"] = kept_id


# ----------------------------------------------------------------------
# flags: duplicate_of (global, across the whole project)
# ----------------------------------------------------------------------
def _normalized_words(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def flag_duplicates(ordered_sentences: list[dict[str, Any]]) -> None:
    """Set ``duplicate_of`` on the earlier of two near-duplicate sentences.

    Only sentences with at least :data:`DUPLICATE_MIN_WORDS` distinct
    normalized words take part, on both sides of the pair. A short phrase
    ("Zobaczcie.", "Jest bardzo dobre.") is a thing people say twice in a trip
    without meaning it as a retake, and at three words or fewer the Jaccard
    measure cannot tell the two cases apart — flagging them cost the reference project four
    perfectly good sentences.

    Args:
        ordered_sentences: Every sentence in the project, in chronological
            order (project clip order, then sentence order within a clip) —
            the same order the footage log and the last-take rule use.
    """
    normalized = [_normalized_words(s["text"]) for s in ordered_sentences]
    n = len(ordered_sentences)
    for i in range(n):
        if len(normalized[i]) < DUPLICATE_MIN_WORDS:
            continue
        for j in range(i + 1, n):
            if len(normalized[j]) < DUPLICATE_MIN_WORDS:
                continue
            if _jaccard(normalized[i], normalized[j]) >= DUPLICATE_JACCARD:
                later = ordered_sentences[j]
                if later.get("instruction"):
                    continue
                if ordered_sentences[i].get("duplicate_of") is None:
                    ordered_sentences[i]["duplicate_of"] = later["id"]
                break


def _finalize_keep_default(sentences: list[dict[str, Any]]) -> None:
    for sent in sentences:
        sent["keep_default"] = not (
            sent["instruction"] or sent["retake_of"] or sent["duplicate_of"]
        )


# ----------------------------------------------------------------------
# the catalogue
# ----------------------------------------------------------------------
def build_sentence_catalogue(project: Project) -> dict[str, Any]:
    """Build the whole-project sentence catalogue from transcripts + analysis.

    Returns:
        The document written to ``analysis/sentences.json``.
    """
    from ytedit.ai.analyze import load_analysis  # local import: avoids a cycle

    clips_out: list[dict[str, Any]] = []
    all_sentences: list[dict[str, Any]] = []

    for clip in project.clips_in_order():
        clip_id = str(clip.get("id", ""))
        if not clip_id:
            continue
        words = load_words(project, clip_id)
        if not words:
            clips_out.append({"id": clip_id, "sentences": []})
            continue
        transcript_lang = None
        transcript_path = project.transcript_path(clip_id)
        if transcript_path.exists():
            try:
                transcript_lang = json.loads(
                    transcript_path.read_text(encoding="utf-8")
                ).get("language")
            except (json.JSONDecodeError, OSError):  # pragma: no cover - bad transcript
                transcript_lang = None

        sentences = build_clip_sentences(clip_id, words, transcript_lang)
        analysis = load_analysis(project, clip_id) or {}
        flag_instructions(sentences, analysis)
        flag_retakes(sentences, analysis)

        clips_out.append({"id": clip_id, "sentences": sentences})
        all_sentences.extend(sentences)

    flag_duplicates(all_sentences)
    _finalize_keep_default(all_sentences)

    return {
        "project": project.slug,
        "language": project.language,
        "generated": utcnow(),
        "clips_count": len(clips_out),
        "sentences_count": len(all_sentences),
        "clips": clips_out,
    }


def _hms(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, float(seconds)), 60)
    return f"{int(minutes)}:{secs:04.1f}"


def render_sentences_md(doc: Mapping[str, Any]) -> str:
    """Render the human-readable ``analysis/sentences.md``."""
    lines: list[str] = [
        f"# Sentence catalogue — {doc.get('project', '')}",
        "",
        f"Language: `{doc.get('language', '')}` · clips: {doc.get('clips_count', 0)} · "
        f"sentences: {doc.get('sentences_count', 0)}",
        "",
    ]
    for clip in doc.get("clips", []):
        lines.append(f"## {clip.get('id', '')}")
        lines.append("")
        sentences = clip.get("sentences") or []
        if not sentences:
            lines.append("_no transcribed speech_")
            lines.append("")
            continue
        for sent in sentences:
            flags: list[str] = []
            if sent.get("instruction"):
                flags.append("INSTRUCTION")
            if sent.get("retake_of"):
                flags.append(f"retake_of {sent['retake_of']}")
            if sent.get("duplicate_of"):
                flags.append(f"duplicate_of {sent['duplicate_of']}")
            tag = f" [{', '.join(flags)}]" if flags else ""
            text = str(sent.get("text", "")).replace("\n", " ")
            lines.append(
                f"- `{sent.get('id', '')}` {_hms(float(sent.get('s') or 0.0))}–"
                f"{_hms(float(sent.get('e') or 0.0))} — \"{text}\"{tag}"
            )
        lines.append("")
    return "\n".join(lines)


def sentences_path(project: Project) -> Path:
    """``analysis/sentences.json``."""
    return project.analysis_dir / "sentences.json"


def write_sentences(project: Project) -> dict[str, Any]:
    """Build and write ``sentences.json`` and ``sentences.md``."""
    document = build_sentence_catalogue(project)
    path = sentences_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    (project.analysis_dir / "sentences.md").write_text(
        render_sentences_md(document), encoding="utf-8"
    )
    project.set_stage(
        STAGE, "done", clips=document["clips_count"], sentences=document["sentences_count"]
    )
    return document


def load_sentences(project: Project) -> dict[str, Any]:
    """Read ``analysis/sentences.json`` (``{}`` when absent/unreadable)."""
    path = sentences_path(project)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupted artifact
        log.error("unreadable sentence catalogue %s: %s", path, exc)
        return {}


def sentences_by_clip(doc: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group a catalogue document's sentences by clip id, in sentence order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for clip in doc.get("clips", []):
        clip_id = str(clip.get("id", ""))
        out[clip_id] = list(clip.get("sentences") or [])
    return out


def sentence_index(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten a catalogue document into ``{sentence_id: sentence_dict}``."""
    out: dict[str, dict[str, Any]] = {}
    for clip in doc.get("clips", []):
        for sent in clip.get("sentences") or []:
            sid = sent.get("id")
            if sid:
                out[str(sid)] = sent
    return out


def load_sentence_index(project: Project) -> dict[str, dict[str, Any]]:
    """Convenience: :func:`load_sentences` + :func:`sentence_index` in one call."""
    return sentence_index(load_sentences(project))


# ----------------------------------------------------------------------
# planner-prompt compaction
# ----------------------------------------------------------------------
def compact_clip_sentences_for_prompt(clip_sentences: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Render one clip's sentence list for the planner prompt.

    Instruction sentences are omitted entirely (the planner must never see
    them as an option). A retake/duplicate sentence is kept but marked
    ``"skip"`` with the reason, so the planner understands the id exists in
    the transcript but must reference the target instead.
    """
    out: list[dict[str, Any]] = []
    for sent in clip_sentences:
        if sent.get("instruction"):
            continue
        item: dict[str, Any] = {
            "id": sent["id"],
            "s": sent["s"],
            "e": sent["e"],
            "text": sent["text"],
        }
        if sent.get("retake_of"):
            item["skip"] = f"retake, use {sent['retake_of']}"
        elif sent.get("duplicate_of"):
            item["skip"] = f"duplicate, use {sent['duplicate_of']}"
        out.append(item)
    return out


def compact_footage_log_for_planner(
    footage_log: Mapping[str, Any], catalogue: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace each clip's ``segments``/``takes`` with its numbered sentences.

    The rest of the footage log entry (summary, visual, hooks, numbers,
    background_music, instructions, location, ...) is left untouched — only
    the free-text segment/take detail the model used to reconstruct narration
    from is replaced by the sentence inventory it must reference by id.
    Clips with no transcribed speech (no catalogue entry, or an empty one)
    keep their original ``segments`` unchanged (pure visual b-roll still
    needs some description of what it shows).
    """
    by_clip = sentences_by_clip(catalogue)
    out = dict(footage_log)
    clips_raw = out.get("clips")
    if not isinstance(clips_raw, list):
        return out

    new_clips: list[Any] = []
    for entry in clips_raw:
        if not isinstance(entry, dict):
            new_clips.append(entry)
            continue
        clip_id = str(entry.get("clip") or entry.get("id") or "")
        clip_sentences = by_clip.get(clip_id)
        if not clip_sentences:
            new_clips.append(entry)
            continue
        new_entry = dict(entry)
        new_entry.pop("segments", None)
        new_entry.pop("takes", None)
        new_entry["sentences"] = compact_clip_sentences_for_prompt(clip_sentences)
        new_clips.append(new_entry)
    out["clips"] = new_clips
    return out


__all__ = [
    "SENTENCE_PAUSE_MAX",
    "DUPLICATE_JACCARD",
    "DUPLICATE_MIN_WORDS",
    "SentencesError",
    "build_clip_sentences",
    "build_sentence_catalogue",
    "compact_clip_sentences_for_prompt",
    "compact_footage_log_for_planner",
    "flag_duplicates",
    "flag_instructions",
    "flag_retakes",
    "load_sentence_index",
    "load_sentences",
    "render_sentences_md",
    "sentence_index",
    "sentences_by_clip",
    "sentences_path",
    "split_words_into_sentences",
    "write_sentences",
]
