"""Stage: ``voice`` — clean up the user's home-recorded narration pickups.

the user records narration requests (``n001``...) and extra "gap pickups" on his
phone at home and drops the WAVs into ``voice/incoming/``. This module turns
each one into a placed :class:`~ytedit.timeline.VoiceItem`, doing by machine
what the editor-in-chief was doing by hand with scratch scripts:

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
3. Place it: a manifest entry either names an explicit ``anchor`` (a video
   segment id) or a narration ``request`` id from ``plan/edit_plan.json``,
   resolved against ``plan/timeline.json`` best-effort
   (:func:`resolve_anchor`). When the pickup runs longer than the muted
   picture underneath it, the last muted/ambient segment in the run is grown
   and, if still short, extra muted B-roll from the manifest's ``broll_pool``
   is inserted, until the narration is fully covered
   (:func:`_cover_voice_item`) — reusing
   :func:`ytedit.ai.tidy._shift_map`/``_retime_absolute_tracks`` so every
   other absolute-time track (captions, music, chapters, markers, other
   voice items) re-times exactly like a ``ytedit tidy`` pass would.

A timeline with ``meta.edited_by_human`` is never overwritten directly — a
fresh ``voice`` run writes ``plan/timeline.draft.json`` instead, same as
``plan``/``tidy`` (playbook §1.5), unless ``--force``.

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

from ytedit.ai.sentences import split_words_into_sentences
from ytedit.ai.tidy import Word, _retime_absolute_tracks, _shift_map, load_words
from ytedit.config import Settings
from ytedit.log import get_logger
from ytedit.media.ffmpeg import FFmpegError, ff
from ytedit.project import Project
from ytedit.timeline import Timeline, Transform, VideoSegment, VoiceAnchor, VoiceItem

log = get_logger(__name__)

STAGE = "voice"

_EPS = 1e-6

#: Jaccard word-overlap at/above which two sentences in the *same* recording
#: count as one retaken twice — the earlier (or later, with ``keep_takes:
#: first``) is dropped. Looser than ``sentences.DUPLICATE_JACCARD`` (0.7)
#: because a home pickup re-does a line with more variation than an on-camera
#: retake ("zaczyna się tam street party" / "zaczyna się święto, które...").
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

_SEGMENT_ID_RE = re.compile(r"\bs\d{3,}\b")
_CLIP_ID_RE = re.compile(r"\bc\d{3,}\b")


class VoiceError(RuntimeError):
    """Raised when the voice stage cannot run or has nothing usable."""


# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------
@dataclass
class ManifestEntry:
    """One row of ``voice/incoming/manifest.yaml``."""

    file: str
    request: str | None = None
    anchor: str | None = None
    offset: float = 0.0
    label: str | None = None
    cuts: list[tuple[float, float]] = field(default_factory=list)
    keep_takes: str = "last"
    broll_pool: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def target_label(self) -> str:
        """Output basename (``voice/<label>.wav``): explicit, else request id, else the file stem."""
        return self.label or self.request or Path(self.file).stem


def manifest_path(project: Project) -> Path:
    """``voice/incoming/manifest.yaml``."""
    return project.voice_incoming_dir / "manifest.yaml"


def _draft_manifest_entries(project: Project) -> list[dict[str, Any]]:
    incoming = project.voice_incoming_dir
    names = sorted(p.name for p in incoming.glob("*.wav")) if incoming.exists() else []
    return [{"file": name, "request": None} for name in names]


def _as_pair(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None
    return None


def _as_triple(raw: Any) -> tuple[str, float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return str(raw[0]), float(raw[1]), float(raw[2])
        except (TypeError, ValueError):
            return None
    return None


def load_manifest(project: Project) -> list[ManifestEntry]:
    """Read ``voice/incoming/manifest.yaml``, writing a draft when it's missing.

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
        path.write_text(
            yaml.safe_dump(drafted, allow_unicode=True, sort_keys=False) if drafted else "[]\n",
            encoding="utf-8",
        )
        raise VoiceError(
            f"no manifest at {path} — wrote a draft listing {len(drafted)} wav(s) with "
            "`request: null`. Fill in `request: nNNN` (a narration request id from "
            "plan/edit_plan.json) or `anchor: sNNN` + `label:` for each file, then "
            f"re-run `ytedit voice {project.slug}`."
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
        pool = [t for t in (_as_triple(c) for c in (item.get("broll_pool") or [])) if t is not None]
        entries.append(
            ManifestEntry(
                file=str(item["file"]),
                request=str(item["request"]) if item.get("request") else None,
                anchor=str(item["anchor"]) if item.get("anchor") else None,
                offset=float(item.get("offset", 0.0) or 0.0),
                label=str(item["label"]) if item.get("label") else None,
                cuts=cuts,
                keep_takes=str(item.get("keep_takes", "last") or "last").strip().lower(),
                broll_pool=pool,
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
# placement: narration request / explicit anchor -> VoiceAnchor
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


def _target_segment(timeline: Timeline, text: str) -> VideoSegment | None:
    """Best-effort: an exact segment id in ``text``, else its first clip's
    segment, else the segment starting at a beat marker named in ``text``."""
    video = timeline.tracks.video
    match = _SEGMENT_ID_RE.search(text)
    if match:
        seg = next((s for s in video if s.id == match.group(0)), None)
        if seg is not None:
            return seg
    match = _CLIP_ID_RE.search(text)
    if match:
        seg = next((s for s in video if s.clip == match.group(0)), None)
        if seg is not None:
            return seg
    lowered = text.lower()
    for marker in timeline.markers:
        label = str(marker.label or "").strip().lower()
        if label and label in lowered:
            pos = timeline.segment_at(marker.at)
            if pos is not None:
                return pos.segment
    return None


def _anchor_after(timeline: Timeline, target: VideoSegment) -> VoiceAnchor:
    video = timeline.tracks.video
    idx = next(i for i, s in enumerate(video) if s.id == target.id)
    if idx + 1 < len(video):
        return VoiceAnchor(segment=video[idx + 1].id, offset=0.0)
    return VoiceAnchor(segment=target.id, offset=round(target.duration, 3))


def resolve_anchor(
    timeline: Timeline, edit_plan: Mapping[str, Any], entry: ManifestEntry
) -> tuple[VoiceAnchor | None, str]:
    """Resolve one manifest entry's placement to a :class:`VoiceAnchor`.

    Returns:
        ``(anchor, reason)`` — ``anchor`` is ``None`` when unresolved, and
        ``reason`` explains either the resolution or the failure (for the
        report).
    """
    if entry.anchor:
        seg = next((s for s in timeline.tracks.video if s.id == entry.anchor), None)
        if seg is None:
            return None, f"anchor segment {entry.anchor!r} not found in timeline.json"
        return (
            VoiceAnchor(segment=entry.anchor, offset=entry.offset),
            f"explicit anchor {entry.anchor} +{entry.offset:.2f}s",
        )
    if entry.request:
        req = _narration_requests(edit_plan).get(entry.request)
        if req is None:
            return None, f"narration request {entry.request!r} not found in plan/edit_plan.json"
        text = str(req.get("place_after_segment", ""))
        target = _target_segment(timeline, text)
        if target is None:
            return None, f"could not resolve place_after_segment {text!r} for {entry.request}"
        anchor = _anchor_after(timeline, target)
        return anchor, f"{entry.request} -> after {target.id} (matched {text!r})"
    return None, "manifest entry has neither `request` nor `anchor` yet"


# ----------------------------------------------------------------------
# overlap: grow the muted picture under an overlong pickup
# ----------------------------------------------------------------------
def _is_coverable(project: Project, seg: VideoSegment) -> bool:
    """True when ``seg`` is muted/ambient — safe for a narration pickup to run over."""
    if seg.mute_source:
        return True
    if seg.audio_from is not None:
        return False
    words = load_words(project, seg.clip)
    return not any(w.s < seg.out - _EPS and w.e > seg.in_ + _EPS for w in words)


def _new_segment_id(video: Sequence[VideoSegment]) -> str:
    """A fresh ``sNNN`` id that collides with nothing already in ``video``.

    Deliberately does *not* renumber existing segments (unlike ``plan.py``'s
    passes) — an anchor elsewhere in the timeline may already reference one
    of their ids, and this stage runs after theirs, not before.
    """
    existing = {seg.id for seg in video}
    nums = [int(m.group(1)) for seg in video if (m := re.fullmatch(r"s(\d+)", seg.id))]
    n = (max(nums) + 1) if nums else 1
    while f"s{n:03d}" in existing:
        n += 1
    return f"s{n:03d}"


def _undo_growth(timeline: Timeline, growth: Mapping[str, Any] | None) -> None:
    """Revert a previous :func:`_cover_voice_item` result before recomputing it."""
    if not growth:
        return
    inserted = set(growth.get("inserted_segments") or [])
    if inserted:
        timeline.tracks.video = [s for s in timeline.tracks.video if s.id not in inserted]
    ext_id = growth.get("extended_segment")
    ext_by = float(growth.get("extended_by") or 0.0)
    if ext_id and ext_by > _EPS:
        seg = next((s for s in timeline.tracks.video if s.id == ext_id), None)
        if seg is not None:
            seg.out = round(seg.out - ext_by, 3)


def _cover_voice_item(
    project: Project,
    timeline: Timeline,
    settings: Settings,
    item: VoiceItem,
    broll_pool: Sequence[tuple[str, float, float]],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Grow the muted/ambient picture under ``item`` until it covers its length.

    Walks forward from the segment ``item`` starts under, summing the
    duration of each further segment that is muted or has no transcribed
    speech of its own (:func:`_is_coverable`). If the run isn't long enough
    before hitting a segment with its own narration (or the end of the
    timeline), the last coverable segment is extended (clamped to its own
    clip's duration) and, if still short, muted B-roll from ``broll_pool``
    (``[clip, in, out]`` triples) is inserted until the shortfall is covered
    or the pool runs out.

    Every absolute-time track is re-timed exactly like a ``ytedit tidy`` pass
    would (:func:`ytedit.ai.tidy._shift_map`/``_retime_absolute_tracks``), and
    anchored voice items (including ``item`` itself) are re-resolved.

    Returns:
        ``(covered, changes, growth_record)`` — ``growth_record`` is what
        :func:`_undo_growth` needs to revert this on a later re-run.
    """
    changes: list[str] = []
    record: dict[str, Any] = {"extended_segment": None, "extended_by": 0.0, "inserted_segments": []}
    if item.end is None:
        return True, changes, record
    voice_duration = item.end - item.at
    if voice_duration <= _EPS:
        return True, changes, record

    before = timeline.segment_positions()
    if not before:
        return False, ["timeline has no video segments"], record

    start_idx = None
    for i, pos in enumerate(before):
        if pos.start - _EPS <= item.at < pos.end + _EPS:
            start_idx = i
            break
    if start_idx is None:
        if item.at >= before[-1].end - _EPS:
            return True, [
                "voice item starts after the last video segment — nothing to grow "
                "into; add picture manually"
            ], record
        return False, ["voice item start does not fall on any video segment"], record

    covered = 0.0
    last_ok = None
    i = start_idx
    while i < len(before):
        seg = before[i].segment
        if not _is_coverable(project, seg):
            break
        covered += before[i].end - max(before[i].start, item.at)
        last_ok = i
        if covered >= voice_duration - _EPS:
            return True, changes, record
        i += 1

    shortfall = voice_duration - covered
    video = timeline.tracks.video
    clips = project.load_state().get("clips", {})

    if last_ok is not None:
        seg = video[last_ok]
        clip_duration = float((clips.get(seg.clip) or {}).get("duration") or 0.0)
        room = max(0.0, clip_duration - seg.out) if clip_duration > 0 else shortfall
        extend = min(shortfall, room)
        if extend > _EPS:
            seg.out = round(seg.out + extend, 3)
            shortfall -= extend
            record["extended_segment"] = seg.id
            record["extended_by"] = round(extend, 3)
            changes.append(f"grew {seg.id} ({seg.clip}) by {extend:.2f}s to cover the pickup")

    insert_at = (last_ok + 1) if last_ok is not None else start_idx
    if insert_at is not None:
        from ytedit.ai.plan import _vertical_fit  # local: avoids a module-load cycle risk

        default_fit = str(settings.get("fit.default_mode", "blur-fill"))
        pool = list(broll_pool)
        while shortfall > _EPS and pool:
            clip_id, in_s, out_s = pool.pop(0)
            dur = out_s - in_s
            if dur <= _EPS:
                continue
            use_out = out_s if dur <= shortfall + _EPS else in_s + shortfall
            clip_meta = clips.get(clip_id) or {}
            new_seg = VideoSegment(
                id=_new_segment_id(video),
                clip=clip_id,
                **{"in": round(in_s, 3)},
                out=round(use_out, 3),
                role="b-roll",
                transform=Transform(fit=_vertical_fit(clip_meta, default_fit)),
                mute_source=True,
                notes="voice: broll_pool fill for an overlong pickup",
            )
            video.insert(insert_at, new_seg)
            insert_at += 1
            added = new_seg.duration
            shortfall -= added
            record["inserted_segments"].append(new_seg.id)
            changes.append(
                f"inserted {clip_id}[{in_s:.2f}-{use_out:.2f}] ({added:.2f}s) from broll_pool"
            )

    remap = _shift_map(before, timeline.segment_positions())
    _retime_absolute_tracks(timeline, remap)
    timeline.resolve_voice_anchors()

    ok = shortfall <= _EPS
    if not ok:
        changes.append(
            f"still short by {shortfall:.2f}s after the broll_pool ran out — "
            "overlap unresolved, add more B-roll or trim the pickup"
        )
    return ok, changes, record


def _next_voice_id(timeline: Timeline) -> str:
    nums = [int(m.group(1)) for item in timeline.tracks.voice if (m := re.fullmatch(r"v(\d+)", item.id))]
    n = (max(nums) + 1) if nums else 1
    return f"v{n:03d}"


def _upsert_voice_item(
    timeline: Timeline, rel_file: str, duration: float, anchor: VoiceAnchor
) -> str:
    """Place (or replace, by ``file``) the :class:`VoiceItem` for one pickup."""
    starts = {pos.segment.id: pos.start for pos in timeline.segment_positions()}
    at = round(starts.get(anchor.segment, 0.0) + anchor.offset, 3)
    end = round(at + duration, 3)
    existing = next((v for v in timeline.tracks.voice if v.file == rel_file), None)
    if existing is not None:
        existing.anchor = anchor
        existing.at = at
        existing.end = end
        return existing.id
    item_id = _next_voice_id(timeline)
    timeline.tracks.voice.append(
        VoiceItem(id=item_id, file=rel_file, at=at, end=end, anchor=anchor)
    )
    return item_id


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
        if row.get("voice_item"):
            lines.append(f"- voice item: `{row['voice_item']}`")
        cuts = row.get("cuts") or []
        if cuts:
            lines.append("- cuts:")
            for c in cuts:
                s, e = c.get("s"), c.get("e")
                where = f" {s:.2f}-{e:.2f}s" if isinstance(s, (int, float)) and isinstance(e, (int, float)) else ""
                text = c.get("text", "")
                suffix = f': "{text}"' if text else ""
                lines.append(f"  - {c.get('reason', '?')}{where}{suffix}")
        overlap = row.get("overlap") or []
        if overlap:
            lines.append("- overlap handling:")
            for change in overlap:
                lines.append(f"  - {change}")
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
        project: Project whose ``plan/timeline.json`` and
            ``voice/incoming/manifest.yaml`` are ready.
        force: Re-process every manifest entry (ignore the incoming-state
            cache) and overwrite a human-edited timeline instead of writing
            ``timeline.draft.json``.

    Returns:
        ``{"rows", "written", "backup", "edited_by_human", "report", "issues"}``.

    Raises:
        VoiceError: No manifest yet (a draft is written first), or no
            ``plan/timeline.json`` to place pickups against.
    """
    settings = project.settings
    entries = load_manifest(project)
    if not project.timeline_file.exists():
        raise VoiceError(
            f"no timeline at {project.timeline_file} — run `ytedit plan {project.slug}` first"
        )
    timeline = Timeline.load(project.timeline_file)
    human_edited = bool(timeline.meta.edited_by_human)
    edit_plan = _load_edit_plan(project)
    state = _load_incoming_state(project)
    incoming = project.voice_incoming_dir

    pad_before = max(0.0, float(settings.get("pacing.speech_pad_before", 0.30)))
    pad_after = max(0.0, float(settings.get("pacing.speech_pad_after", 0.45)))
    max_pause = max(0.0, float(settings.get("voice.max_pause", 0.8)))
    pause_keep = max(0.0, float(settings.get("voice.pause_keep", 0.5)))

    rows: list[dict[str, Any]] = []
    processed_any = False
    timeline_changed = False

    for entry in entries:
        row: dict[str, Any] = {"file": entry.file, "label": entry.target_label}
        wav_path = incoming / entry.file
        if not wav_path.exists():
            row["status"], row["error"] = "error", f"missing {wav_path}"
            rows.append(row)
            continue
        if not entry.request and not entry.anchor:
            row["status"] = "unassigned"
            row["error"] = "manifest entry has no `request` or `anchor` yet — fill it in"
            rows.append(row)
            continue

        file_hash = _hash_file(wav_path)
        cached = state.get(entry.file)
        if cached and cached.get("hash") == file_hash and not force:
            row["status"] = "skipped (cached)"
            row["output"] = cached.get("output")
            rows.append(row)
            continue

        _undo_growth(timeline, (cached or {}).get("growth"))

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
        anchor, placement_reason = resolve_anchor(timeline, edit_plan, entry)
        row["placement"] = placement_reason
        rel_file = project.rel(out_wav)

        growth_record: dict[str, Any] | None = None
        if anchor is None:
            row["status"] = "unplaced"
        else:
            item_id = _upsert_voice_item(timeline, rel_file, duration, anchor)
            item = next(v for v in timeline.tracks.voice if v.id == item_id)
            ok, overlap_changes, growth_record = _cover_voice_item(
                project, timeline, settings, item, entry.broll_pool
            )
            row["voice_item"] = item_id
            row["overlap"] = overlap_changes
            row["status"] = "ok" if ok else "overlap_unresolved"
            timeline_changed = True

        state[entry.file] = {
            "hash": file_hash,
            "label": label,
            "output": rel_file,
            "growth": growth_record,
        }
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
    if timeline_changed:
        from ytedit.ai.tidy import backup_timeline

        result["issues"] = timeline.validate(project)
        result["backup"] = backup_timeline(project)
        write_to_draft = human_edited and not force
        target = project.plan_dir / ("timeline.draft.json" if write_to_draft else "timeline.json")
        timeline.save(target)
        result["written"] = project.rel(target)
        if write_to_draft:
            log.warning(
                "[clip]%s[/] timeline.json is human-edited — voice pickups written to %s "
                "(diff it, or re-run with --force)",
                project.slug,
                target.name,
            )
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
    "resolve_anchor",
    "run_voice",
]
