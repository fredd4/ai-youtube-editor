"""Stage 2: clip audio -> ``transcripts/<clip>.json`` (+ ``.srt``, ``.txt``).

Primary engine is ElevenLabs Scribe v2 (word timestamps, audio events,
diarization, ``no_verbatim=False`` so repeated takes survive into the text).
When ``models.stt`` points at ``openrouter/<model>``, or when ElevenLabs fails
with a network/5xx error, the OpenRouter transcription endpoint is used instead
and the engine actually used is recorded in the transcript.

Besides the transcript itself this module ships :func:`detect_takes`, a purely
deterministic near-duplicate detector: normalised token n-grams inside a sliding
time window.  Its output is only a *hint* handed to the analyst LLM, which makes
the final call about which attempt to keep (playbook 3, "last-take rule").
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..costs import charge, check_budget, estimate
from ..log import get_logger
from ..project import Project, utcnow
from .elevenlabs import ElevenLabs, ElevenLabsError, Transcript
from .openrouter import OpenRouter

log = get_logger(__name__)

#: Concurrent STT requests (ElevenLabs tolerates more, but 2 keeps the ledger
#: writes and the cost log readable and is polite on a payg plan).
WORKERS = 2

#: Fallback engine when ElevenLabs is unreachable / 5xx.
FALLBACK_STT_MODEL = "openai/whisper-large-v3"

#: ISO-639-1 -> ISO-639-2/T, the form ElevenLabs expects in ``language_code``.
ISO1_TO_ISO3: dict[str, str] = {
    "pl": "pol", "en": "eng", "de": "deu", "fr": "fra", "es": "spa",
    "it": "ita", "pt": "por", "nl": "nld", "cs": "ces", "sk": "slk",
    "uk": "ukr", "ru": "rus", "sv": "swe", "no": "nor", "da": "dan",
    "fi": "fin", "tr": "tur", "el": "ell", "hu": "hun", "ro": "ron",
    "hr": "hrv", "sr": "srp", "bg": "bul", "lt": "lit", "lv": "lav",
    "et": "est", "ja": "jpn", "ko": "kor", "zh": "zho", "ar": "ara",
}
ISO3_TO_ISO1: dict[str, str] = {v: k for k, v in ISO1_TO_ISO3.items()}

#: Confidence above which a detected language is trusted for the mismatch flag.
LANGUAGE_CONFIDENCE = 0.6

# -- SRT shape ---------------------------------------------------------------
SRT_MIN_WORDS = 3
SRT_MAX_WORDS = 5
SRT_MIN_SECONDS = 1.0
SRT_MAX_SECONDS = 4.0

#: Sentence-final punctuation used by the SRT/TXT splitters.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”»)\]]*$")

#: Number words folded onto digits so "dwa" and "2" count as the same token.
NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "jeden": "1", "jedna": "1", "jedno": "1", "dwa": "2", "dwie": "2",
    "trzy": "3", "cztery": "4", "piec": "5", "szesc": "6", "siedem": "7",
    "osiem": "8", "dziewiec": "9", "dziesiec": "10", "sto": "100", "tysiac": "1000",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "hundred": "100",
    "thousand": "1000",
}


class TranscribeError(RuntimeError):
    """A clip could not be transcribed by any configured engine."""


@dataclass(slots=True)
class TranscribeResult:
    """Outcome of one clip's transcription."""

    clip_id: str
    status: str = "pending"  # done | skipped | stub | error
    engine: str = ""
    words: int = 0
    duration: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    take_hints: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# language helpers
# --------------------------------------------------------------------------- #


def to_iso3(language: str | None) -> str | None:
    """``"pl"`` -> ``"pol"``; pass through anything already 3-letter."""
    if not language:
        return None
    code = str(language).strip().lower().replace("_", "-").split("-")[0]
    if len(code) == 3:
        return code
    return ISO1_TO_ISO3.get(code, code or None)


def to_iso1(language: str | None) -> str | None:
    """``"pol"`` -> ``"pl"``; pass through anything already 2-letter."""
    if not language:
        return None
    code = str(language).strip().lower().replace("_", "-").split("-")[0]
    if len(code) == 2:
        return code
    return ISO3_TO_ISO1.get(code, code or None)


def language_mismatch(
    detected: str | None,
    project_language: str,
    probability: float | None,
    word_count: int | None = None,
) -> bool:
    """True when a confidently detected language differs from the project's.

    Clips with fewer than three words (engine noise, crowd chatter) never
    count: the detector's language guess is meaningless there.
    """
    if not detected or probability is None:
        return False
    if word_count is not None and word_count < 3:
        return False
    if float(probability) <= LANGUAGE_CONFIDENCE:
        return False
    return to_iso1(detected) != to_iso1(project_language)


# --------------------------------------------------------------------------- #
# take detection (deterministic hints for the analyst LLM)
# --------------------------------------------------------------------------- #


def normalize_token(text: str) -> str:
    """Fold a word to a comparable token: lowercase, no diacritics, no punctuation.

    Digits and small number words collapse onto the same representation so
    ``"trzy euro"`` and ``"3 euro"`` compare equal.
    """
    decomposed = unicodedata.normalize("NFKD", str(text or "").lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    token = re.sub(r"[^0-9a-z]+", "", stripped)
    return NUMBER_WORDS.get(token, token)


def _word_time(word: Mapping[str, Any], key: str) -> float:
    try:
        return float(word.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def detect_takes(
    words: Sequence[Mapping[str, Any]],
    window_s: float = 60.0,
    n: int = 4,
) -> list[dict[str, Any]]:
    """Find near-duplicate spoken spans (repeated takes) in a word list.

    The detector is deterministic and cheap: words are normalised, every
    ``n``-gram is indexed, and consecutive occurrences of the same n-gram that
    lie within ``window_s`` of each other are grown forwards and backwards for
    as long as the tokens keep matching.  Overlapping candidates are collapsed
    into groups.

    Args:
        words: ``[{"t", "s", "e"}, ...]`` from a transcript.
        window_s: Maximum time between two attempts at the same statement.
        n: n-gram length used as the repetition seed.

    Returns:
        ``[{"topic_hint": str, "attempts": [{"s", "e", "text"}, ...]}, ...]``
        in chronological order. These are candidates only: the analyst LLM
        verifies them and decides which attempt to keep.
    """
    kept: list[int] = []
    norm: list[str] = []
    for i, word in enumerate(words or []):
        token = normalize_token(word.get("t", ""))
        if token:
            kept.append(i)
            norm.append(token)
    if len(norm) < 2 * n:
        return []

    start_time = [_word_time(words[k], "s") for k in kept]
    end_time = [_word_time(words[k], "e") for k in kept]

    grams: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(norm) - n + 1):
        grams.setdefault(tuple(norm[i : i + n]), []).append(i)

    # (start_a, start_b, length) for every grown repetition.
    raw: list[tuple[int, int, int]] = []
    for positions in grams.values():
        if len(positions) < 2:
            continue
        for a, b in zip(positions, positions[1:]):
            if b - a < n:  # overlapping occurrences of a repetitive phrase
                continue
            if start_time[b] - start_time[a] > window_s:
                continue
            length = n
            while a + length < b and b + length < len(norm) and norm[a + length] == norm[b + length]:
                length += 1
            back = 0
            while (
                a - back - 1 >= 0
                and b - back - 1 >= a + length
                and norm[a - back - 1] == norm[b - back - 1]
            ):
                back += 1
            raw.append((a - back, b - back, length + back))

    if not raw:
        return []

    # Keep the longest candidates; drop any whose spans are already covered.
    covered: set[int] = set()
    spans: list[tuple[int, int]] = []  # (first_index, last_index) in compact space
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for a, b, length in sorted(raw, key=lambda c: -c[2]):
        span_a = set(range(a, a + length))
        span_b = set(range(b, b + length))
        if span_a <= covered and span_b <= covered:
            continue
        covered |= span_a | span_b
        pairs.append(((a, a + length - 1), (b, b + length - 1)))
        spans.extend([(a, a + length - 1), (b, b + length - 1)])

    # Union overlapping spans into groups (one group == one "take").
    parent: dict[tuple[int, int], tuple[int, int]] = {s: s for s in spans}

    def _find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: tuple[int, int], y: tuple[int, int]) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[ry] = rx

    for left, right in pairs:
        _union(left, right)
    ordered = sorted(set(spans))
    for left, right in zip(ordered, ordered[1:]):
        if right[0] <= left[1]:  # spans overlap in the word stream
            _union(left, right)

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for span in ordered:
        groups.setdefault(_find(span), []).append(span)

    def _text(first: int, last: int) -> str:
        return " ".join(str(words[kept[i]].get("t", "")).strip() for i in range(first, last + 1)).strip()

    out: list[dict[str, Any]] = []
    for members in groups.values():
        members = sorted(set(members))
        if len(members) < 2:
            continue
        attempts = [
            {
                "s": round(start_time[first], 3),
                "e": round(end_time[last], 3),
                "text": _text(first, last),
            }
            for first, last in members
        ]
        longest = max(members, key=lambda m: m[1] - m[0])
        hint = " ".join(_text(*longest).split()[:8])
        out.append({"topic_hint": hint, "attempts": attempts})
    out.sort(key=lambda g: g["attempts"][0]["s"])
    return out


# --------------------------------------------------------------------------- #
# SRT / TXT writers
# --------------------------------------------------------------------------- #


def srt_timestamp(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS,mmm``."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_cues(
    words: Sequence[Mapping[str, Any]],
    min_words: int = SRT_MIN_WORDS,
    max_words: int = SRT_MAX_WORDS,
    min_seconds: float = SRT_MIN_SECONDS,
    max_seconds: float = SRT_MAX_SECONDS,
) -> list[dict[str, Any]]:
    """Group words into short subtitle cues.

    Cues carry ``min_words``..``max_words`` words and last
    ``min_seconds``..``max_seconds``; a sentence end closes a cue early once it
    holds at least ``min_words`` words. Short cues are stretched up to
    ``min_seconds`` without overlapping the next one.
    """
    usable = [w for w in (words or []) if str(w.get("t", "")).strip()]
    if not usable:
        return []

    cues: list[dict[str, Any]] = []
    chunk: list[Mapping[str, Any]] = []

    def _flush() -> None:
        if not chunk:
            return
        cues.append(
            {
                "s": _word_time(chunk[0], "s"),
                "e": max(_word_time(chunk[-1], "e"), _word_time(chunk[0], "s")),
                "text": " ".join(str(w.get("t", "")).strip() for w in chunk).strip(),
            }
        )
        chunk.clear()

    for word in usable:
        chunk.append(word)
        span = _word_time(chunk[-1], "e") - _word_time(chunk[0], "s")
        sentence_end = bool(_SENTENCE_END.search(str(word.get("t", "")).strip()))
        if len(chunk) >= max_words or span >= max_seconds:
            _flush()
        elif sentence_end and len(chunk) >= min_words:
            _flush()
    _flush()

    for i, cue in enumerate(cues):
        limit = cues[i + 1]["s"] if i + 1 < len(cues) else cue["e"] + min_seconds
        if cue["e"] - cue["s"] < min_seconds:
            cue["e"] = min(max(cue["e"], cue["s"] + min_seconds), max(limit, cue["e"]))
        cue["s"] = round(cue["s"], 3)
        cue["e"] = round(max(cue["e"], cue["s"] + 0.05), 3)
    return cues


def write_srt(path: Path | str, words: Sequence[Mapping[str, Any]]) -> Path:
    """Write an SRT file of 3-5 word cues and return its path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        f"{i}\n{srt_timestamp(cue['s'])} --> {srt_timestamp(cue['e'])}\n{cue['text']}\n"
        for i, cue in enumerate(build_cues(words), start=1)
    ]
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out


def build_sentences(
    words: Sequence[Mapping[str, Any]], max_gap: float = 1.2, max_words: int = 40
) -> list[dict[str, Any]]:
    """Split a word list into sentences (punctuation, long pauses, hard cap)."""
    usable = [w for w in (words or []) if str(w.get("t", "")).strip()]
    sentences: list[dict[str, Any]] = []
    chunk: list[Mapping[str, Any]] = []

    def _flush() -> None:
        if not chunk:
            return
        sentences.append(
            {
                "s": round(_word_time(chunk[0], "s"), 3),
                "e": round(_word_time(chunk[-1], "e"), 3),
                "text": " ".join(str(w.get("t", "")).strip() for w in chunk).strip(),
            }
        )
        chunk.clear()

    for i, word in enumerate(usable):
        if chunk and _word_time(word, "s") - _word_time(chunk[-1], "e") > max_gap:
            _flush()
        chunk.append(word)
        if _SENTENCE_END.search(str(word.get("t", "")).strip()) or len(chunk) >= max_words:
            _flush()
    _flush()
    return sentences


def write_txt(path: Path | str, words: Sequence[Mapping[str, Any]], text: str = "") -> Path:
    """Write plain text with a ``[mm:ss.s]`` stamp in front of every sentence."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sentences = build_sentences(words)
    if not sentences:
        out.write_text((text or "").strip() + "\n" if text else "", encoding="utf-8")
        return out
    lines = []
    for sentence in sentences:
        minutes, seconds = divmod(sentence["s"], 60)
        lines.append(f"[{int(minutes):02d}:{seconds:04.1f}] {sentence['text']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# cost plumbing
# --------------------------------------------------------------------------- #


def _units_for_ledger(units: Any) -> Any:
    """Turn a client's unit *string* back into something ``costs`` can label."""
    if isinstance(units, (int, float)):
        return float(units)
    text = str(units or "").strip()
    seconds = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*s", text)
    if seconds:
        return float(seconds.group(1))
    chars = re.fullmatch(r"([0-9]+)\s*chars", text)
    if chars:
        return int(chars.group(1))
    tokens = re.fullmatch(r"([0-9]+)\s*\+\s*([0-9]+)\s*tok", text)
    if tokens:
        return {"in": int(tokens.group(1)), "out": int(tokens.group(2))}
    return {"units": text or "unknown"}


def cost_recorder(project: Project, stage: str, clip_id: str | None = None):
    """Build a ``cost_callback`` that charges the project ledger."""

    def _callback(
        service: str,
        op: str,
        model: str | None = None,
        units: Any = None,
        usd: float | None = None,
        **extra: Any,
    ) -> None:
        charge(
            project,
            service=service,
            op=op,
            units=_units_for_ledger(units),
            usd=usd,
            model=model,
            stage=stage,
            **({"clip": clip_id} if clip_id else {}),
            **extra,
        )

    return _callback


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #


def _keyterms(project: Project) -> list[str]:
    raw = project.config.get("keyterms") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(term) for term in raw if str(term).strip()][:1000]


def _language_code(project: Project) -> str | None:
    """Language hint for the STT engine.

    Default is auto-detection (``None``): travel footage routinely contains
    other languages (locals, vendors) and forcing the project language makes
    Scribe report a wrong ``language_code`` with probability 1.0, hiding the
    mismatch from the analyst. Set ``language_detect: force`` in
    ``project.yaml`` to pin the project language, or ``language_detect: <iso>``
    to pin another one.
    """
    detect = str(project.config.get("language_detect", "auto")).strip().lower()
    if detect in ("", "auto"):
        return None
    if detect == "force":
        return to_iso3(project.language)
    return to_iso3(detect)


def _is_transient(exc: ElevenLabsError) -> bool:
    """Network error (status 0) or a server-side 5xx -> worth falling back."""
    return exc.status == 0 or exc.status >= 500


def _elevenlabs_payload(transcript: Transcript) -> dict[str, Any]:
    return {
        "language": transcript.language,
        "language_probability": transcript.language_probability,
        "text": transcript.text,
        "words": transcript.words,
        "events": transcript.events,
        "speakers": transcript.speakers,
        "engine": transcript.engine,
        "duration": transcript.duration_s,
    }


def _openrouter_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "language": raw.get("language"),
        "language_probability": None,
        "text": raw.get("text", ""),
        "words": list(raw.get("words") or []),
        "events": [],
        "speakers": [],
        "engine": raw.get("engine", "openrouter"),
        "duration": float(raw.get("duration") or 0.0),
    }


def _run_elevenlabs(project: Project, clip_id: str, audio: Path) -> dict[str, Any]:
    client = ElevenLabs(
        api_key=project.settings.require_key("elevenlabs"),
        cost_callback=cost_recorder(project, "transcribe", clip_id),
    )
    try:
        transcript = client.transcribe(
            audio,
            language_code=_language_code(project),
            diarize=True,
            tag_audio_events=True,
            keyterms=_keyterms(project),
        )
    finally:
        client.close()
    return _elevenlabs_payload(transcript)


def _run_openrouter(
    project: Project, clip_id: str, audio: Path, model: str = FALLBACK_STT_MODEL
) -> dict[str, Any]:
    client = OpenRouter(
        api_key=project.settings.require_key("openrouter"),
        cost_callback=cost_recorder(project, "transcribe", clip_id),
    )
    try:
        raw = client.transcribe_audio(
            audio, model=model, language=to_iso1(project.language)
        )
    finally:
        client.close()
    return _openrouter_payload(raw)


def transcribe_clip(project: Project, clip: Mapping[str, Any]) -> dict[str, Any]:
    """Transcribe one clip with the configured engine, falling back on failure."""
    clip_id = str(clip["id"])
    audio = project.audio_path(clip_id)
    if not audio.exists():
        raise TranscribeError(f"{clip_id}: missing work audio at {audio}")

    configured = str(project.settings.get("models.stt", "elevenlabs/scribe_v2"))
    if configured.startswith("openrouter/"):
        return _run_openrouter(project, clip_id, audio, configured.split("/", 1)[1])
    try:
        return _run_elevenlabs(project, clip_id, audio)
    except ElevenLabsError as exc:
        if not _is_transient(exc):
            raise
        log.warning(
            "[clip]%s[/]: ElevenLabs unavailable (%s), falling back to %s",
            clip_id,
            exc,
            FALLBACK_STT_MODEL,
        )
        return _run_openrouter(project, clip_id, audio)


# --------------------------------------------------------------------------- #
# transcript documents
# --------------------------------------------------------------------------- #


def build_document(
    project: Project, clip: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble the ``transcripts/<clip>.json`` document."""
    clip_id = str(clip["id"])
    words = list(payload.get("words") or [])
    detected = payload.get("language")
    probability = payload.get("language_probability")
    duration = float(payload.get("duration") or 0.0) or float(clip.get("duration") or 0.0)
    return {
        "clip": clip_id,
        "language": detected,
        "language_probability": probability,
        "project_language": project.language,
        "language_mismatch": language_mismatch(
            detected, project.language, probability, len(words)
        ),
        "has_audio": True,
        "duration": round(duration, 3),
        "text": payload.get("text", ""),
        "words": words,
        "events": list(payload.get("events") or []),
        "speakers": list(payload.get("speakers") or []),
        "engine": payload.get("engine", ""),
        "take_hints": detect_takes(words),
        "created": utcnow(),
    }


def stub_document(project: Project, clip: Mapping[str, Any]) -> dict[str, Any]:
    """Transcript document for a clip that carries no audio track."""
    return {
        "clip": str(clip["id"]),
        "language": None,
        "language_probability": None,
        "project_language": project.language,
        "language_mismatch": False,
        "has_audio": False,
        "duration": round(float(clip.get("duration") or 0.0), 3),
        "text": "",
        "words": [],
        "events": [],
        "speakers": [],
        "engine": "none/no-audio",
        "take_hints": [],
        "created": utcnow(),
    }


def write_transcript(project: Project, document: Mapping[str, Any]) -> Path:
    """Write the JSON transcript plus its ``.srt`` and ``.txt`` siblings."""
    clip_id = str(document["clip"])
    path = project.transcript_path(clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    words = list(document.get("words") or [])
    write_srt(path.with_suffix(".srt"), words)
    write_txt(path.with_suffix(".txt"), words, str(document.get("text") or ""))
    return path


def load_transcript(project: Project, clip_id: str) -> dict[str, Any] | None:
    """Read one transcript document, or ``None`` when it does not exist."""
    path = project.transcript_path(clip_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupted artifact
        log.error("unreadable transcript %s: %s", path, exc)
        return None


def all_transcripts(project: Project) -> dict[str, dict[str, Any]]:
    """Every transcript in the project, keyed by clip id, in clip order."""
    out: dict[str, dict[str, Any]] = {}
    for clip in project.clips_in_order():
        clip_id = str(clip.get("id", ""))
        document = load_transcript(project, clip_id) if clip_id else None
        if document is not None:
            out[clip_id] = document
    return out


# --------------------------------------------------------------------------- #
# stage entry point
# --------------------------------------------------------------------------- #


def _needs_work(project: Project, clip: Mapping[str, Any], force: bool) -> bool:
    if force:
        return True
    if str(clip.get("stages", {}).get("transcribe", "")) != "done":
        return True
    return not project.transcript_path(str(clip.get("id", ""))).exists()


def transcribe(project: Project, force: bool = False) -> list[TranscribeResult]:
    """Transcribe every clip that still needs it.

    Args:
        project: Target project (must have been ingested).
        force: Re-transcribe clips already marked ``done``.

    Returns:
        One :class:`TranscribeResult` per clip that was considered.
    """
    clips = [c for c in project.clips_in_order() if _needs_work(project, c, force)]
    if not clips:
        log.info("[stage]transcribe[/]: nothing to do")
        project.set_stage("transcribe", "done", cost_usd=0.0)
        return []

    with_audio = [c for c in clips if c.get("has_audio")]
    total_seconds = sum(float(c.get("duration") or 0.0) for c in with_audio)
    upfront = estimate("elevenlabs", "stt", total_seconds, settings=project.settings)
    check_budget(project, upfront)
    log.info(
        "[stage]transcribe[/]: %d clip(s), %.1fs audio, estimated $%.4f",
        len(clips),
        total_seconds,
        upfront,
    )
    project.set_stage("transcribe", "running")
    before = _spent(project)

    def _one(clip: Mapping[str, Any]) -> TranscribeResult:
        clip_id = str(clip["id"])
        result = TranscribeResult(clip_id=clip_id, duration=float(clip.get("duration") or 0.0))
        try:
            if not clip.get("has_audio"):
                document = stub_document(project, clip)
                result.status = "stub"
            else:
                check_budget(
                    project,
                    estimate(
                        "elevenlabs", "stt", result.duration, settings=project.settings
                    ),
                )
                document = build_document(project, clip, transcribe_clip(project, clip))
                result.status = "done"
            write_transcript(project, document)
            result.engine = str(document.get("engine", ""))
            result.words = len(document.get("words") or [])
            result.take_hints = list(document.get("take_hints") or [])
            project.set_clip_stage(
                clip_id,
                "transcribe",
                "done",
                transcript=project.rel(project.transcript_path(clip_id)),
                transcript_engine=result.engine,
                transcript_words=result.words,
                language_mismatch=bool(document.get("language_mismatch")),
            )
            log.info(
                "[clip]%s[/] transcribed: %d words, %d take hint(s), engine=%s",
                clip_id,
                result.words,
                len(result.take_hints),
                result.engine,
            )
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the stage
            result.status = "error"
            result.error = str(exc)
            log.error("[clip]%s[/] transcription failed: %s", clip_id, exc)
            project.set_clip_stage(clip_id, "transcribe", "error", error=str(exc)[:2000])
        return result

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(_one, clips))

    spent = round(_spent(project) - before, 6)
    errors = [r for r in results if r.status == "error"]
    project.set_stage(
        "transcribe",
        "error" if errors else "done",
        cost_usd=spent,
        **({"error": f"{len(errors)} clip(s) failed"} if errors else {}),
    )
    log.info(
        "[stage]transcribe[/]: %d done, %d stub, %d error, [cost]$%.4f[/]",
        sum(1 for r in results if r.status == "done"),
        sum(1 for r in results if r.status == "stub"),
        len(errors),
        spent,
    )
    return results


def _spent(project: Project) -> float:
    return round(
        sum(float(e.get("usd", 0.0)) for e in project.load_state().get("costs", [])), 6
    )


__all__ = [
    "transcribe",
    "transcribe_clip",
    "load_transcript",
    "all_transcripts",
    "detect_takes",
    "normalize_token",
    "build_cues",
    "build_sentences",
    "write_srt",
    "write_txt",
    "srt_timestamp",
    "language_mismatch",
    "to_iso1",
    "to_iso3",
    "cost_recorder",
    "TranscribeResult",
    "TranscribeError",
]
