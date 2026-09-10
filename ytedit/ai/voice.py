"""Stage: ``voice`` — clean up home-recorded narration pickups.

The user records narration requests (``n001``...) and extra "gap pickups" on a
phone at home and drops the WAVs into ``voice/incoming/``. This module turns
each one into a ``voice`` beat of ``plan/cut.json``, doing by machine what the
editor-in-chief was doing by hand with scratch scripts:

1. Transcribe the WAV (ElevenLabs Scribe, word timestamps) — see
   :func:`_transcribe_voice_file`, reusing the cost-logged pattern of
   ``ytedit.ai.transcribe._run_elevenlabs`` under stage ``"voice"``.
2. Clean up the take: drop sentences that are editor instructions
   (:func:`is_instruction_sentence`) or a rejected retake inside the same
   recording (:func:`_clean_words`, Jaccard >= :data:`RETAKE_JACCARD`), cut
   stutter fragments, shorten long internal pauses
   (``voice.max_pause`` -> ``voice.pause_keep``), and trim to speech with the
   usual pads (``pacing.speech_pad_before``/``_after``). The kept ranges are
   cut from the source with one ffmpeg ``aselect``/``asetpts`` chain
   (:func:`_render_voice_wav`) into ``voice/<label>.wav``.
3. Place it (:func:`resolve_beat`, :func:`place_voice_beat`): a manifest entry
   either names an explicit ``after`` (a beat of ``plan/cut.json``) or a
   narration ``request`` id from ``plan/edit_plan.json``, whose
   ``place_after_segment`` text is matched to a beat best-effort. A ``voice``
   beat is inserted **immediately after** that beat and takes its picture from
   the ``broll`` beats that follow: they are moved out of the beat list and
   into the pickup's ``shots`` until they cover the WAV, keeping their own
   sound only if all of them had it. Nothing else moves —
   in cut v2 music, captions, chapters and markers are all positioned *by
   beat*, so inserting one shifts no other track's timing and there is no
   shift map to maintain.

Everything downstream is derived: ``ytedit.cut.resolve`` turns the edited cut
into ``plan/timeline.json`` at the end of the run. A cut with
``meta.edited_by_human`` is never overwritten directly — a fresh ``voice`` run
writes ``plan/cut.draft.json`` instead (playbook §1.5), unless ``--force``.

See ``docs/playbook/editing-playbook.md`` §5 for the manifest format and what
the user still reviews by hand (``voice/incoming/report.md``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ytedit import cut as cutlib
from ytedit.ai.sentences import load_sentence_index, split_words_into_sentences
from ytedit.cut import Beat, Cut, Shot
from ytedit.log import get_logger
from ytedit.media.ffmpeg import FFmpegError, ff
from ytedit.project import Project
from ytedit.words import Word

log = get_logger(__name__)

STAGE = "voice"

_EPS = 1e-6

#: Jaccard word-overlap at/above which two sentences in the *same* recording
#: count as one retaken twice — the earlier (or later, with ``keep_takes:
#: first``) is dropped. Looser than ``sentences.DUPLICATE_JACCARD`` (0.7)
#: because a home pickup re-does a line with more variation than an on-camera
#: retake ("zaczyna się tam koncert" / "zaczyna się występ, który...").
RETAKE_JACCARD: float = 0.6

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: Sentence-opening editor instructions, project-language aware. Easy to
#: extend: add a language key, or more patterns to an existing one. Matched
#: against the *start* of a sentence, case-insensitively.
INSTRUCTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "pl": (
        r"^to\s+wstaw",
        r"^to\s+na\b",
        r"^u[żz]yj",
        r"^wytnij",
        r"^to\s+wytnij",
        r"^to\s+wykorzystaj",
        r"^to\s+jako\b",
    ),
    "en": (
        r"^insert\s+this",
        r"^use\s+this\s+(?:as|for)",
        r"^cut\s+this",
        r"^discard\s+this",
    ),
}

_BEAT_ID_RE = re.compile(r"\bb\d{3,}\b")
_CLIP_ID_RE = re.compile(r"\bc\d{3,}\b")

#: How much of a beat's first sentence the draft manifest's beat listing shows.
_CATALOGUE_TEXT_CHARS = 64


class VoiceError(RuntimeError):
    """Raised when the voice stage cannot run or has nothing usable."""


# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------
@dataclass
class ManifestEntry:
    """One row of ``voice/incoming/manifest.yaml``.

    A pickup is placed *between* beats, so there is no ``offset``: ``after``
    names the beat the pickup follows and that is the only position a
    ``voice`` beat can have. (``anchor``, the v1 video-segment id, is gone
    with the segments it referenced.)
    """

    file: str
    request: str | None = None
    #: A beat of ``plan/cut.json``: its display id (``b012``) or its uid.
    after: str | None = None
    label: str | None = None
    cuts: list[tuple[float, float]] = field(default_factory=list)
    keep_takes: str = "last"

    @property
    def target_label(self) -> str:
        """Output basename (``voice/<label>.wav``): explicit, else request id, else the file stem."""
        return self.label or self.request or Path(self.file).stem


def manifest_path(project: Project) -> Path:
    """``voice/incoming/manifest.yaml``."""
    return project.voice_incoming_dir / "manifest.yaml"


def _draft_manifest_entries(project: Project) -> list[dict[str, Any]]:
    """One blank row per WAV sitting in ``voice/incoming/``."""
    incoming = project.voice_incoming_dir
    names = sorted(p.name for p in incoming.glob("*.wav")) if incoming.exists() else []
    return [{"file": name, "request": None, "after": None} for name in names]


def _beat_catalogue_lines(project: Project, cut: Cut) -> list[str]:
    """One readable line per beat, for the draft manifest's comment block.

    The user fills ``after:`` in by hand, so the draft has to say what the
    beats *are* — a bare ``b047`` is unanswerable without opening the editor.
    A speech beat is named by the text of its first sentence, a B-roll beat by
    its clip range, a pickup by its file.
    """
    index = load_sentence_index(project)
    lines: list[str] = []
    for beat in cut.beats:
        label = beat.id or beat.uid
        clip = beat.clip or "-"
        if beat.kind == "speech":
            what = _first_sentence_text(beat, index)
            body = f"{clip:<6} {what}"
        elif beat.kind == "broll":
            body = f"{clip:<6} {float(beat.in_ or 0.0):.2f}-{float(beat.out or 0.0):.2f}s"
        else:
            body = f"{'-':<6} {beat.file or '(no file)'}"
        lines.append(f"{label}  {beat.kind:<6} {body}".rstrip())
    return lines


def _first_sentence_text(beat: Beat, index: Mapping[str, Mapping[str, Any]]) -> str:
    """The quoted opening of a speech beat, truncated for the beat listing."""
    if beat.sentences:
        sent = index.get(beat.sentences[0])
        text = str((sent or {}).get("text", "")).strip()
        if not text:
            return f"({beat.sentences[0]})"
        if len(text) > _CATALOGUE_TEXT_CHARS:
            text = text[: _CATALOGUE_TEXT_CHARS - 1].rstrip() + "…"
        return f'"{text}"'
    if beat.words is not None:
        return f"words {int(beat.words[0])}-{int(beat.words[1])}"
    return "(no sentences)"


def _draft_manifest_text(project: Project, cut: Cut | None) -> str:
    """The full text of a drafted ``manifest.yaml``: help, beat listing, rows."""
    rows = _draft_manifest_entries(project)
    head = [
        "# voice/incoming/manifest.yaml — one entry per WAV in voice/incoming/.",
        "#",
        "#   request:    a narration request id from plan/edit_plan.json (nNNN)",
        "#   after:      a beat of plan/cut.json (b012, or a beat uid) — the pickup",
        "#               is inserted right after it and takes its picture from the",
        "#               B-roll beats that follow. `after` wins over `request`.",
        "#   label:      output basename -> voice/<label>.wav (default: the request",
        "#               id, else the file stem)",
        "#   cuts:       [[s, e], ...] extra manual cuts, in source seconds",
        "#   keep_takes: last (default) | first",
        "#",
        "# There is no `offset:` — a voice beat sits *between* beats, so a pickup",
        "# can only start where a beat starts.",
        "#",
    ]
    if cut is not None and cut.beats:
        head.append("# Beats of plan/cut.json:")
        head.append("#")
        head.extend(f"#   {line}" for line in _beat_catalogue_lines(project, cut))
        head.append("#")
    body = yaml.safe_dump(rows, allow_unicode=True, sort_keys=False) if rows else "[]\n"
    return "\n".join(head) + "\n" + body


def _as_pair(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None
    return None


def load_manifest(project: Project, cut: Cut | None = None) -> list[ManifestEntry]:
    """Read ``voice/incoming/manifest.yaml``, writing a draft when it's missing.

    Args:
        project: The owning project.
        cut: The cut the pickups will be placed on. Only used to list the
            beats the user can choose from in the drafted manifest's comment
            block; ``None`` simply omits that listing.

    Raises:
        VoiceError: When the manifest does not exist yet (a draft listing
            every WAV in ``voice/incoming/`` is written first) or is not a
            YAML list.
    """
    incoming = project.voice_incoming_dir
    incoming.mkdir(parents=True, exist_ok=True)
    path = manifest_path(project)
    if not path.exists():
        drafted = _draft_manifest_entries(project)
        path.write_text(_draft_manifest_text(project, cut), encoding="utf-8")
        raise VoiceError(
            f"no manifest at {path} — wrote a draft listing {len(drafted)} wav(s) with "
            "`request: null` and `after: null`, above a listing of the cut's beats. "
            "Fill in `request: nNNN` (a narration request id from plan/edit_plan.json) "
            "or `after: bNNN` (the beat of plan/cut.json the pickup follows) + "
            "`label:` for each file, then re-run "
            f"`ytedit voice {project.slug}`. There is no `offset:` any more: a voice "
            "beat sits between beats."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except yaml.YAMLError as exc:
        raise VoiceError(f"unreadable manifest {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise VoiceError(f"{path} must be a YAML list of entries")

    entries: list[ManifestEntry] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("file"):
            continue
        cuts = [p for p in (_as_pair(c) for c in (item.get("cuts") or [])) if p is not None]
        entries.append(
            ManifestEntry(
                file=str(item["file"]),
                request=str(item["request"]) if item.get("request") else None,
                after=str(item["after"]) if item.get("after") else None,
                label=str(item["label"]) if item.get("label") else None,
                cuts=cuts,
                keep_takes=str(item.get("keep_takes", "last") or "last").strip().lower(),
            )
        )
    return entries


# ----------------------------------------------------------------------
# per-file processing state (hash -> output, for skip-unless-changed)
# ----------------------------------------------------------------------
def _incoming_state_path(project: Project) -> Path:
    return project.voice_incoming_dir / "state.json"


def _load_incoming_state(project: Project) -> dict[str, Any]:
    path = _incoming_state_path(project)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - corrupted state
        return {}
    return data if isinstance(data, dict) else {}


def _save_incoming_state(project: Project, state: dict[str, Any]) -> None:
    _incoming_state_path(project).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# transcription (reuses the ElevenLabs cost-logging pattern of transcribe.py)
# ----------------------------------------------------------------------
def _transcribe_voice_file(project: Project, wav_path: Path) -> dict[str, Any]:
    """Transcribe one narration WAV with ElevenLabs, charged under stage ``voice``.

    Mirrors ``ytedit.ai.transcribe._run_elevenlabs``, but for an arbitrary
    file outside the clip registry (so it can't reuse that function directly,
    which resolves its audio from ``project.audio_path(clip_id)``).
    """
    from ytedit.ai.elevenlabs import ElevenLabs
    from ytedit.ai.transcribe import _keyterms, _language_code, cost_recorder

    client = ElevenLabs(
        api_key=project.settings.require_key("elevenlabs"),
        cost_callback=cost_recorder(project, STAGE),
    )
    try:
        transcript = client.transcribe(
            wav_path,
            language_code=_language_code(project),
            diarize=False,
            tag_audio_events=False,
            keyterms=_keyterms(project),
        )
    finally:
        client.close()
    return {
        "language": transcript.language,
        "language_probability": transcript.language_probability,
        "text": transcript.text,
        "words": transcript.words,
        "duration": transcript.duration_s,
        "engine": transcript.engine,
    }


def _transcript_sidecar_path(project: Project, filename: str) -> Path:
    return project.voice_incoming_dir / f"{filename}.transcript.json"


def _get_transcript(project: Project, wav_path: Path, filename: str, force: bool) -> dict[str, Any]:
    """Read the cached transcript sidecar, or transcribe and write it."""
    sidecar = _transcript_sidecar_path(project, filename)
    if sidecar.exists() and not force:
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):  # pragma: no cover - corrupted sidecar
            pass
    doc = _transcribe_voice_file(project, wav_path)
    sidecar.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def _words_from_doc(doc: Mapping[str, Any]) -> list[Word]:
    """Turn a transcript document's ``words`` (the ``t``/``s``/``e`` shape
    ``ytedit.ai.transcribe`` normalizes ElevenLabs output to) into :class:`Word`\\ s."""
    words: list[Word] = []
    for raw in doc.get("words") or []:
        if not isinstance(raw, dict):
            continue
        try:
            s, e = float(raw.get("s", 0.0)), float(raw.get("e", 0.0))
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        text = str(raw.get("t", raw.get("text", ""))).strip()
        if not text:
            continue
        words.append(Word(s, e, text))
    words.sort(key=lambda w: (w.s, w.e))
    return words


# ----------------------------------------------------------------------
# clean-up: instructions, retakes, manual cuts, stutters
# ----------------------------------------------------------------------
def is_instruction_sentence(text: str, language: str) -> bool:
    """True when ``text`` opens with an editor instruction (playbook §3)."""
    normalized = text.strip().lower()
    if not normalized:
        return False
    patterns = INSTRUCTION_PATTERNS.get(language, INSTRUCTION_PATTERNS["pl"])
    return any(re.search(pattern, normalized) for pattern in patterns)


def _normalized_words(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _norm_word(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _clean_words(
    words: Sequence[Word], entry: ManifestEntry, language: str
) -> tuple[list[bool], list[dict[str, Any]]]:
    """Decide which words survive, and log why each dropped range was cut.

    Order: editor instructions, then retakes (Jaccard >= :data:`RETAKE_JACCARD`
    between two sentences of the *same* recording), then the manifest's manual
    ``cuts``, then stutter fragments (a cut-off word, or an immediate repeat).

    Returns:
        ``(keep, cuts_log)`` — ``keep[i]`` is whether ``words[i]`` survives;
        ``cuts_log`` entries carry ``reason``/``text``/``s``/``e`` for the report.
    """
    keep = [True] * len(words)
    cuts_log: list[dict[str, Any]] = []
    if not words:
        return keep, cuts_log

    groups = split_words_into_sentences(list(words))
    spans: list[tuple[int, int]] = []
    idx = 0
    for group in groups:
        spans.append((idx, idx + len(group)))
        idx += len(group)

    def _text(a: int, b: int) -> str:
        return " ".join(w.text for w in words[a:b]).strip()

    dropped_reason: list[str | None] = [None] * len(spans)
    for i, (a, b) in enumerate(spans):
        if is_instruction_sentence(_text(a, b), language):
            dropped_reason[i] = "instruction"

    normalized = [_normalized_words(_text(a, b)) for a, b in spans]
    keep_first = entry.keep_takes == "first"
    for i in range(len(spans)):
        if dropped_reason[i] is not None or not normalized[i]:
            continue
        for j in range(i + 1, len(spans)):
            if dropped_reason[j] is not None or not normalized[j]:
                continue
            if _jaccard(normalized[i], normalized[j]) >= RETAKE_JACCARD:
                loser, kept_idx = (j, i) if keep_first else (i, j)
                dropped_reason[loser] = f"retake_of:{_text(*spans[kept_idx])[:80]}"
                if loser == i:
                    break

    for i, (a, b) in enumerate(spans):
        reason = dropped_reason[i]
        if reason is None:
            continue
        for k in range(a, b):
            keep[k] = False
        cuts_log.append(
            {
                "reason": reason.split(":", 1)[0],
                "text": _text(a, b),
                "s": words[a].s,
                "e": words[b - 1].e,
            }
        )

    for cut_s, cut_e in entry.cuts:
        removed_any = False
        for k, w in enumerate(words):
            if keep[k] and w.s < cut_e - _EPS and w.e > cut_s + _EPS:
                keep[k] = False
                removed_any = True
        if removed_any:
            cuts_log.append({"reason": "manual_cut", "s": cut_s, "e": cut_e})

    prev_kept: int | None = None
    for k, w in enumerate(words):
        if not keep[k]:
            continue
        if w.text.rstrip().endswith("-"):
            keep[k] = False
            cuts_log.append({"reason": "stutter_cutoff", "text": w.text, "s": w.s, "e": w.e})
            continue
        norm = _norm_word(w.text)
        if prev_kept is not None and norm and norm == _norm_word(words[prev_kept].text):
            keep[k] = False
            cuts_log.append({"reason": "stutter_repeat", "text": w.text, "s": w.s, "e": w.e})
            continue
        prev_kept = k

    return keep, cuts_log


def _keep_ranges(
    words: Sequence[Word],
    keep: Sequence[bool],
    max_pause: float,
    pause_keep: float,
    pad_before: float,
    pad_after: float,
) -> list[tuple[float, float]]:
    """Turn a per-word keep mask into source-time ranges to select.

    A gap between two consecutive *kept* words longer than ``max_pause`` is
    shortened to ``pause_keep`` (still audible as a pause, not a splice). A
    gap that exists only because a word in between was dropped (instruction,
    retake, manual cut, stutter) is not padded at all — it's a real edit, so
    the two surviving ranges simply butt together. The whole file gets the
    usual speech pads at its very first/last kept range.
    """
    ranges: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end = 0.0
    for w, k in zip(words, keep):
        if not k:
            if cur_start is not None:
                ranges.append((cur_start, cur_end))
                cur_start = None
            continue
        if cur_start is None:
            cur_start, cur_end = w.s, w.e
            continue
        gap = w.s - cur_end
        if gap > max_pause + _EPS:
            ranges.append((cur_start, cur_end + pause_keep))
            cur_start, cur_end = w.s, w.e
        else:
            cur_end = w.e
    if cur_start is not None:
        ranges.append((cur_start, cur_end))

    if ranges:
        s0, e0 = ranges[0]
        ranges[0] = (max(0.0, s0 - pad_before), e0)
        sN, eN = ranges[-1]
        ranges[-1] = (sN, eN + pad_after)
    return ranges


def _render_voice_wav(src: Path, ranges: Sequence[tuple[float, float]], out_path: Path) -> None:
    """Cut ``ranges`` out of ``src`` into one 48k mono pcm_s16le wav (hard cuts, in order)."""
    if not ranges:
        raise VoiceError(f"no speech ranges to render from {src}")
    expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in ranges)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ff(
        "-i", str(src),
        "-af", f"aselect='{expr}',asetpts=N/SR/TB",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    )


# ----------------------------------------------------------------------
# placement: narration request / explicit `after` -> a beat of the cut
# ----------------------------------------------------------------------
def _load_edit_plan(project: Project) -> dict[str, Any]:
    path = project.edit_plan_file
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # pragma: no cover - corrupted plan
        return {}


def _narration_requests(edit_plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    plan_obj = edit_plan.get("plan") if isinstance(edit_plan.get("plan"), dict) else edit_plan
    out: dict[str, dict[str, Any]] = {}
    for item in plan_obj.get("narration_requests") or []:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = item
    return out


def _target_beat(cut: Cut, text: str) -> Beat | None:
    """Best-effort: the beat a narration request's prose describes.

    Tried in order: an explicit beat id (``b012``) in the text, the first beat
    cut from the clip it names (``c048``), and finally the beat carrying a
    marker whose label occurs in the text ("bridge to Porto"). The planner
    writes ``place_after_segment`` as free prose, so this stays deliberately
    forgiving — an unresolved entry is reported, never guessed at.
    """
    match = _BEAT_ID_RE.search(text)
    if match:
        beat = cut.beat_by_ref(match.group(0))
        if beat is not None:
            return beat
    match = _CLIP_ID_RE.search(text)
    if match:
        beat = next((b for b in cut.beats if b.clip == match.group(0)), None)
        if beat is not None:
            return beat
    lowered = text.lower()
    for marker in cut.markers:
        label = str(marker.label or "").strip().lower()
        if label and label in lowered:
            beat = cut.beat_by_ref(marker.beat)
            if beat is not None:
                return beat
    return None


def resolve_beat(
    cut: Cut, edit_plan: Mapping[str, Any], entry: ManifestEntry
) -> tuple[Beat | None, str]:
    """Resolve one manifest entry's placement to the beat the pickup follows.

    An explicit ``after`` wins over a ``request``: it is what the user wrote
    by hand after reading the report, and it must not be second-guessed by a
    text match.

    Args:
        cut: The cut to place against.
        edit_plan: ``plan/edit_plan.json`` (or ``{}``), for its narration
            requests.
        entry: The manifest row.

    Returns:
        ``(beat, reason)`` — ``beat`` is ``None`` when unresolved, and
        ``reason`` explains either the resolution or the failure (it goes
        straight into ``voice/incoming/report.md``).
    """
    if entry.after:
        beat = cut.beat_by_ref(entry.after)
        if beat is None:
            return None, f"beat {entry.after!r} is not in plan/cut.json"
        return beat, f"explicit after {beat.id or beat.uid}"
    if entry.request:
        req = _narration_requests(edit_plan).get(entry.request)
        if req is None:
            return None, f"narration request {entry.request!r} not found in plan/edit_plan.json"
        text = str(req.get("place_after_segment", ""))
        target = _target_beat(cut, text)
        if target is None:
            return None, f"could not resolve place_after_segment {text!r} for {entry.request}"
        return target, f"{entry.request} -> after {target.id or target.uid} (matched {text!r})"
    return None, "manifest entry has neither `request` nor `after` yet"


def _shot_from_broll(beat: Beat) -> Shot:
    """Turn a ``broll`` beat into a picture-only shot of a pickup.

    Everything that describes the *picture* travels with it (clip, range,
    transform, grade, notes). The beat's ``audio`` does not: a shot has no
    say over sound, the pickup's own ``audio`` decides for all of them —
    see :func:`place_voice_beat`.
    """
    return Shot(
        clip=beat.clip or "",
        **{"in": float(beat.in_ or 0.0)},
        out=float(beat.out or 0.0),
        transform=beat.transform.model_copy(deep=True),
        grade=beat.grade,
        notes=beat.notes,
    )


def place_voice_beat(
    cut: Cut, target: Beat, file: str, length: float, notes: str = "", gain_db: float = 0.0
) -> tuple[Beat, list[str]]:
    """Insert (or refresh) the ``voice`` beat for one pickup, right after ``target``.

    The pickup's picture comes from the ``broll`` beats that already follow
    ``target``: they are *moved* into the new beat's ``shots`` — whole beats,
    in order — until they cover ``length``. Moving rather than copying is what
    keeps the programme's length honest: the same footage plays once, now
    under narration. The resolver trims the last shot to the WAV (and stretches
    it when the clip has room), so covering ``length`` is enough; nothing here
    cuts a broll beat in half.

    The beat's ``audio`` mirrors the picture it took over: it is set to
    ``ambient`` only when every absorbed beat was heard, so a pickup never
    silences sound the cut deliberately kept and never resurrects sound it
    deliberately muted (wind, a car radio). Otherwise it is left unset — a
    voice beat is silent by default, which is what a pickup recorded at home
    over a location wants.

    Re-running is idempotent: a ``voice`` beat that already carries ``file``
    is refreshed in place — same position, same shots — instead of a second
    one being inserted and more B-roll swallowed. A gain set by hand in the
    editor survives, since only its length can have changed.

    Args:
        cut: The cut to edit, in place.
        target: The beat the pickup follows.
        file: Project-relative path of the cleaned WAV (``voice/<label>.wav``).
        length: The WAV's length in seconds — how much picture to absorb.
        notes: Free text for the beat.
        gain_db: Manual level trim, used only for a newly inserted beat.

    Returns:
        ``(beat, absorbed)`` — the voice beat, and one description per B-roll
        beat that became one of its shots (empty on a refresh, and empty when
        no B-roll followed: then ``beat.shots`` is empty too and the cut has a
        ``voice_picture_short`` error for the user to fix in the editor).
    """
    existing = next((b for b in cut.beats if b.kind == "voice" and b.file == file), None)
    if existing is not None:
        existing.notes = notes or existing.notes
        return existing, []

    index = next((i for i, b in enumerate(cut.beats) if b.uid == target.uid), len(cut.beats) - 1)
    absorbed: list[str] = []
    shots: list[Shot] = []
    ambient = True
    covered = 0.0
    end = index + 1
    while covered < length - _EPS and end < len(cut.beats) and cut.beats[end].kind == "broll":
        donor = cut.beats[end]
        shots.append(_shot_from_broll(donor))
        covered += max(0.0, float(donor.out or 0.0) - float(donor.in_ or 0.0))
        ambient = ambient and donor.keeps_source_audio
        absorbed.append(
            f"{donor.id or donor.uid} {donor.clip} "
            f"{float(donor.in_ or 0.0):.2f}-{float(donor.out or 0.0):.2f}s "
            f"({'ambient' if donor.keeps_source_audio else 'mute'})"
        )
        end += 1
    del cut.beats[index + 1 : end]

    beat = Beat(
        kind="voice", file=file, shots=shots, gain_db=gain_db, role="b-roll", notes=notes,
        # Only a deliberate choice is written: a voice beat is silent by
        # default, so `audio` is set solely to *keep* ambience that was there.
        audio="ambient" if (shots and ambient) else None,
    )
    cut.beats.insert(index + 1, beat)
    return beat, absorbed


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def _write_report(project: Project, rows: list[dict[str, Any]]) -> Path:
    lines = [f"# Voice pickups — {project.slug}", ""]
    if not rows:
        lines.append("_no manifest entries_")
    for row in rows:
        lines.append(f"## {row.get('file')}")
        lines.append(f"- status: **{row.get('status', '?')}**")
        if row.get("error"):
            lines.append(f"- error: {row['error']}")
        if row.get("label"):
            lines.append(f"- label: `{row['label']}`")
        if row.get("duration") is not None:
            lines.append(f"- final duration: {row['duration']:.2f}s")
        if row.get("placement"):
            lines.append(f"- placement: {row['placement']}")
        if row.get("beat"):
            lines.append(f"- beat: `{row['beat']}`")
        cuts = row.get("cuts") or []
        if cuts:
            lines.append("- cuts:")
            for c in cuts:
                s, e = c.get("s"), c.get("e")
                where = f" {s:.2f}-{e:.2f}s" if isinstance(s, (int, float)) and isinstance(e, (int, float)) else ""
                text = c.get("text", "")
                suffix = f': "{text}"' if text else ""
                lines.append(f"  - {c.get('reason', '?')}{where}{suffix}")
        absorbed = row.get("absorbed") or []
        if absorbed:
            lines.append("- picture absorbed into the pickup (was B-roll of its own):")
            for change in absorbed:
                lines.append(f"  - {change}")
        if row.get("picture_audio"):
            lines.append(f"- picture audio under the pickup: {row['picture_audio']}")
        if row.get("action"):
            lines.append(f"- **action needed:** {row['action']}")
        lines.append("")
    path = project.voice_incoming_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------
def run_voice(project: Project, force: bool = False) -> dict[str, Any]:
    """Run the ``voice`` stage: manifest -> transcribe -> clean -> place.

    Args:
        project: Project whose ``plan/cut.json`` and
            ``voice/incoming/manifest.yaml`` are ready.
        force: Re-process every manifest entry (ignore the incoming-state
            cache) and overwrite a human-edited cut instead of writing
            ``cut.draft.json``.

    Returns:
        ``{"rows", "written", "backup", "edited_by_human", "report",
        "issues"}`` — ``written``/``backup`` are project-relative paths (or
        ``None`` when nothing changed), and ``issues`` are the cut validator's
        findings as strings.

    Raises:
        VoiceError: No ``plan/cut.json`` to place pickups against, or no
            manifest yet (a draft is written first).
    """
    settings = project.settings
    source = cutlib.cut_path(project)
    if not source.exists():
        raise VoiceError(
            f"no cut at {source} — run `ytedit plan {project.slug}` first"
        )
    cut = cutlib.load_cut(source)
    # The beat listing in a drafted manifest needs the cut, so it is read
    # first; a project with no cut at all cannot place a pickup anyway.
    entries = load_manifest(project, cut)
    human_edited = bool(cut.meta.edited_by_human)
    edit_plan = _load_edit_plan(project)
    state = _load_incoming_state(project)
    incoming = project.voice_incoming_dir

    pad_before = max(0.0, float(settings.get("pacing.speech_pad_before", 0.30)))
    pad_after = max(0.0, float(settings.get("pacing.speech_pad_after", 0.45)))
    max_pause = max(0.0, float(settings.get("voice.max_pause", 0.8)))
    pause_keep = max(0.0, float(settings.get("voice.pause_keep", 0.5)))

    rows: list[dict[str, Any]] = []
    processed_any = False
    cut_changed = False

    for entry in entries:
        row: dict[str, Any] = {"file": entry.file, "label": entry.target_label}
        wav_path = incoming / entry.file
        if not wav_path.exists():
            row["status"], row["error"] = "error", f"missing {wav_path}"
            rows.append(row)
            continue
        if not entry.request and not entry.after:
            row["status"] = "unassigned"
            row["error"] = "manifest entry has no `request` or `after` yet — fill it in"
            rows.append(row)
            continue

        file_hash = _hash_file(wav_path)
        cached = state.get(entry.file)
        if cached and cached.get("hash") == file_hash and not force:
            row["status"] = "skipped (cached)"
            row["output"] = cached.get("output")
            rows.append(row)
            continue

        try:
            transcript_doc = _get_transcript(project, wav_path, entry.file, force)
        except Exception as exc:  # transcription failures shouldn't abort the whole run
            row["status"], row["error"] = "error", f"transcription failed: {exc}"
            rows.append(row)
            continue

        words = _words_from_doc(transcript_doc)
        if not words:
            row["status"], row["error"] = "error", "no speech detected in transcript"
            rows.append(row)
            continue

        keep, cuts_log = _clean_words(words, entry, project.language)
        ranges = _keep_ranges(words, keep, max_pause, pause_keep, pad_before, pad_after)
        if not ranges:
            row["status"] = "error"
            row["error"] = "every sentence was cut (instruction/retake) — nothing left to keep"
            rows.append(row)
            continue

        label = entry.target_label
        out_wav = project.voice_dir / f"{label}.wav"
        try:
            _render_voice_wav(wav_path, ranges, out_wav)
        except FFmpegError as exc:
            row["status"], row["error"] = "error", f"ffmpeg failed: {exc}"
            rows.append(row)
            continue

        final_text = " ".join(w.text for w, k in zip(words, keep) if k)
        duration = round(sum(e - s for s, e in ranges), 3)
        sidecar = {
            "source": project.rel(wav_path),
            "ranges": [[round(s, 3), round(e, 3)] for s, e in ranges],
            "text": final_text,
            "duration": duration,
            "cuts": cuts_log,
        }
        (project.voice_dir / f"{label}.json").write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        row["duration"] = duration
        row["cuts"] = cuts_log
        target, placement_reason = resolve_beat(cut, edit_plan, entry)
        row["placement"] = placement_reason
        rel_file = project.rel(out_wav)

        if target is None:
            row["status"] = "unplaced"
            row["action"] = (
                "fill in `after: bNNN` in voice/incoming/manifest.yaml and re-run"
            )
        else:
            # The WAV on disk is the length the resolver will use, so absorb
            # picture against that rather than against the sum of the kept
            # ranges (which ignores ffmpeg's frame rounding).
            probed = float(cutlib.probe_voice_duration(out_wav)) or duration
            beat, absorbed = place_voice_beat(
                cut, target, rel_file, probed,
                notes=f"pickup {label}" + (f" ({entry.request})" if entry.request else ""),
            )
            if cut.beats.index(beat) != cut.beats.index(target) + 1:
                # An existing pickup is never dragged across the cut: the
                # picture it already owns would be stranded where it is.
                row["placement"] += (
                    " — but this pickup is already placed elsewhere and was left "
                    "there; delete its beat in the web editor to move it"
                )
            row["beat_uid"] = beat.uid
            row["absorbed"] = absorbed
            row["picture_audio"] = "ambient" if beat.keeps_source_audio else "mute"
            if beat.shots:
                row["status"] = "ok"
            else:
                row["status"] = "voice_picture_short"
                row["action"] = (
                    "no B-roll beat follows this pickup — pick its picture in the web "
                    "editor (the cut will not resolve until you do)"
                )
            cut_changed = True

        state[entry.file] = {"hash": file_hash, "label": label, "output": rel_file}
        processed_any = True
        rows.append(row)

    result: dict[str, Any] = {
        "rows": rows,
        "written": None,
        "backup": None,
        "edited_by_human": human_edited,
        "issues": [],
    }
    if processed_any:
        _save_incoming_state(project, state)
    if cut_changed:
        # Display ids only exist once, at the end: renumbering inside the loop
        # would invalidate every later entry's `after: bNNN`.
        cut.renumber()
        by_uid = {beat.uid: beat for beat in cut.beats}
        for row in rows:
            beat = by_uid.get(str(row.pop("beat_uid", "")))
            if beat is not None:
                row["beat"] = beat.id or beat.uid

        # One pass gives both halves: `validate()` *is* a resolve pass (see
        # ytedit/cut.py), so asking for the issues separately would only mean
        # resolving the same cut twice.
        timeline, issues = cutlib.resolve_verbose(project, cut)
        result["issues"] = [str(issue) for issue in issues]
        result["backup"] = cutlib.backup_cut(project)
        write_to_draft = human_edited and not force
        target_path = project.plan_dir / ("cut.draft.json" if write_to_draft else "cut.json")
        cutlib.save_cut(cut, target_path)
        result["written"] = project.rel(target_path)
        if write_to_draft:
            log.warning(
                "[clip]%s[/] cut.json is human-edited — voice pickups written to %s "
                "(diff it, or re-run with --force)",
                project.slug,
                target_path.name,
            )
        elif any(issue.severity == "error" for issue in issues):
            log.warning(
                "[clip]%s[/] the cut does not resolve yet — timeline.json left alone; "
                "see voice/incoming/report.md",
                project.slug,
            )
        else:
            timeline.save(project.timeline_file)
        project.set_stage(STAGE, "done", processed=len(rows))

    report_path = _write_report(project, rows)
    result["report"] = project.rel(report_path)
    return result


__all__ = [
    "VoiceError",
    "ManifestEntry",
    "INSTRUCTION_PATTERNS",
    "RETAKE_JACCARD",
    "is_instruction_sentence",
    "load_manifest",
    "manifest_path",
    "place_voice_beat",
    "resolve_beat",
    "run_voice",
]
