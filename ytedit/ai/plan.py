"""Edit planning: ``analysis/footage_log.json`` -> ``plan/edit_plan.json`` + timeline.

The planner model (``models.planner``) reads the whole footage log and returns one
compact JSON object (``plan.system`` in prompts.md): a story structure, a segment
list and the cue sheets. It is never asked for a ``timeline.json`` document —
:func:`build_timeline` derives that here — because everything the model returns is
a *proposal*, re-derived deterministically so the non-negotiable rules of the
playbook always hold, whatever the model said.

Deterministic post-processing, in order:

0. Consecutive planner segments that both carry ``voice_over`` for the same
   clip, separated by a small gap, are merged into one segment first — a
   planner that splits one narration take into two proposals must not produce
   two overlapping audio extractions (see the merge pre-pass in
   :func:`build_timeline`).
1. Segments referencing unknown clips, or with ``in >= out``, are dropped;
   ``in``/``out`` are clamped to the clip's real duration from ``state.json``.
2. Every range the footage log marked as an **editor instruction** is excised
   from every segment (segments split around it) — a spoken "put this at the
   end" must never survive into the cut.
3. The **last-take rule** is enforced: attempts a clip's analysis did not keep
   are excised too, unless the planner deliberately kept that range as silent
   B-roll (``mute_source: true``).
4. Vertical clips get ``transform.fit = "blur-fill"`` unless the planner asked
   for a specific fit.
5. Structural **markers** come from ``pacing.markers`` in the settings, never
   from the model (a negative ``at`` is relative to the end).
6. Captions / music cues / chapters are clamped and de-overlapped, then
   :meth:`Timeline.validate` runs and trivially fixable issues are fixed.
7. :func:`ytedit.ai.tidy.pad_segments_to_speech` gives every speech cut ~0.3 s of
   air before the first word and ~0.45 s after the last one, snaps a cut that
   still lands mid-sentence out to that sentence's own boundary, and merges
   same-clip jump cuts closer than ``pacing.merge_gap``.
7b. :func:`ytedit.ai.overlay.overlay_cutaways` finds a cutaway (or a run of them)
   dropped between two contiguous pieces of one take and gives it ``audio_from``
   so the narration keeps running underneath the cutaway instead of stopping
   dead; the continuation is moved (or, when too little is left of it, dropped).
8. A segment carrying ``voice_over`` (playbook §3: post-trip narration clips) has
   its own clip audio extracted into ``tracks.voice`` and is replaced on screen by
   its ``picture`` cuts (``mute_source: true``); the voice item's position is
   resolved only after step 7, since padding upstream can shift it. Speech
   padding makes the narration audio longer than its picture cuts add up to;
   the shortfall is covered by extending the picture cuts themselves (last cut
   first) and only falls back to the narration clip's own face-cam picture
   when the remainder is still a full shot (``pacing.min_shot_seconds``) —
   never a sub-second flash back to the narrator (see
   :func:`_build_voice_over_segment`).

A timeline with ``meta.edited_by_human`` never gets overwritten: the fresh draft
goes to ``plan/timeline.draft.json`` for diffing (playbook §1.5).
"""

from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ytedit.ai.openrouter import OpenRouter, OpenRouterError
from ytedit.ai.prompts import render
from ytedit.ai.ledger import dedupe_audio
from ytedit.ai.overlay import overlay_cutaways
from ytedit.ai.sentences import (
    compact_footage_log_for_planner,
    load_sentence_index,
    load_sentences,
    write_sentences,
)
from ytedit.ai.tidy import pad_segments_to_speech, sentence_snap_count
from ytedit.config import Settings
from ytedit.costs import charge
from ytedit.log import get_logger
from ytedit.media.ffmpeg import FFmpegError, ff
from ytedit.project import Project, utcnow
from ytedit.timeline import (
    AudioFrom,
    Caption,
    Chapter,
    Duck,
    Marker,
    MuteRange,
    MusicCue,
    Timeline,
    Transform,
    Transition,
    VideoSegment,
    VoiceAnchor,
    VoiceItem,
    new_timeline,
)

log = get_logger(__name__)

STAGE = "plan"

#: Wrap-up phrases that must not appear before the payoff (research rule 8).
ENDING_GUARD_RE = re.compile(
    r"dzi[eę]k(?:i|uj[eę])\s+za\s+ogl[aą]danie"
    r"|do\s+zobaczenia"
    r"|na\s+tym\s+ko[nń]czymy"
    r"|do\s+nast[eę]pnego"
    r"|trzymajcie\s+si[eę]"
    r"|thanks?\s+for\s+watching"
    r"|see\s+you\s+(?:next|soon)",
    re.IGNORECASE,
)

#: Roles that count as on-camera narration for the A-roll run rule.
AROLL_ROLE_RE = re.compile(r"a-?roll|narration|talking|piece-to-camera|ptc", re.IGNORECASE)

_EPS = 1e-6


class PlanError(RuntimeError):
    """Raised when the plan stage cannot run (missing inputs, unusable output)."""


# ----------------------------------------------------------------------
# footage log
# ----------------------------------------------------------------------
def load_footage_log(project: Project) -> dict[str, Any]:
    """Read ``analysis/footage_log.json``.

    Args:
        project: Project whose analysis stage has already run.

    Returns:
        The raw document as written by the analyze stage.

    Raises:
        PlanError: When the file is missing or unparseable.
    """
    path = project.analysis_dir / "footage_log.json"
    if not path.exists():
        raise PlanError(
            f"no footage log at {path} — run `ytedit analyze {project.slug}` first "
            "(the plan stage reads analysis/footage_log.json)"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"corrupted footage log {path}: {exc}") from exc
    if isinstance(data, list):
        return {"clips": data}
    if not isinstance(data, dict):
        raise PlanError(f"footage log {path} must be a JSON object or array")
    return data


def footage_entries(footage_log: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-clip analysis dicts from a footage log, in log order.

    Tolerates the three shapes the analyze stage may produce: ``{"clips": [...]}``,
    ``{"clips": {"c001": {...}}}`` and a bare list at the top level.
    """
    clips = footage_log.get("clips", footage_log.get("entries", footage_log.get("log")))
    if isinstance(clips, dict):
        items = [dict(v, clip=v.get("clip", k)) for k, v in clips.items()]
    elif isinstance(clips, list):
        items = [c for c in clips if isinstance(c, dict)]
    else:
        items = [v for k, v in footage_log.items() if isinstance(v, dict) and "clip" in v]
    return items


def _entries_by_clip(footage_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index footage-log entries by clip id."""
    out: dict[str, dict[str, Any]] = {}
    for entry in footage_entries(footage_log):
        clip_id = str(entry.get("clip") or entry.get("id") or "")
        if clip_id:
            out[clip_id] = entry
    return out


def instruction_ranges(entry: dict[str, Any]) -> list[tuple[float, float]]:
    """Time ranges of spoken editor instructions in one clip (clip time)."""
    ranges: list[tuple[float, float]] = []
    for item in entry.get("instructions") or []:
        if not isinstance(item, dict):
            continue
        s, e = _span(item)
        if e > s:
            ranges.append((s, e))
    return ranges


def rejected_take_ranges(entry: dict[str, Any]) -> list[tuple[float, float]]:
    """Ranges of take attempts the analysis did **not** keep (clip time)."""
    ranges: list[tuple[float, float]] = []
    for take in entry.get("takes") or []:
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
        for i, attempt in enumerate(attempts):
            if i == keep:
                continue
            s, e = _span(attempt)
            if e > s:
                ranges.append((s, e))
    return ranges


def background_music_ranges(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Background-music flags that must become ``mute_ranges`` (suggest != keep)."""
    out: list[dict[str, Any]] = []
    clip_id = str(entry.get("clip") or "")
    for item in entry.get("background_music") or []:
        if not isinstance(item, dict):
            continue
        suggest = str(item.get("suggest", "mute")).lower()
        if suggest == "keep":
            continue
        s, e = _span(item)
        if e <= s:
            continue
        out.append(
            {
                "clip": clip_id,
                "s": s,
                "e": e,
                "gain_db": -60.0 if suggest == "mute" else -14.0,
                "reason": f"footage log: background music ({suggest}, "
                f"confidence {item.get('confidence', '?')})",
            }
        )
    return out


def _span(item: dict[str, Any]) -> tuple[float, float]:
    """Read a ``(start, end)`` pair from an ``s``/``e`` or ``in``/``out`` dict."""
    start = item.get("s", item.get("in", item.get("start", item.get("at", 0.0))))
    end = item.get("e", item.get("out", item.get("end", 0.0)))
    try:
        return float(start or 0.0), float(end or 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0


# ----------------------------------------------------------------------
# LLM output model
# ----------------------------------------------------------------------
class _Lenient(BaseModel):
    """Base model for LLM output: unknown keys are kept, aliases accepted."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class Beat(_Lenient):
    """One structural story beat with its target timecode."""

    label: str = ""
    at_s_target: float = 0.0
    clips: list[str] = Field(default_factory=list)
    description: str = ""


class Story(_Lenient):
    """The narrative spine the planner proposes."""

    title_working: str = ""
    premise: str = ""
    hook_idea: str = ""
    beats: list[Beat] = Field(default_factory=list)


class ColdOpenPick(_Lenient):
    """A cold-open montage pick (clip range)."""

    clip: str = ""
    in_: float = Field(0.0, alias="in")
    out: float = 0.0
    why: str = ""


class VoiceOverPicture(_Lenient):
    """One picture cut shown on screen while a voice-over clip's own audio plays."""

    clip: str = ""
    in_: float = Field(0.0, alias="in")
    out: float = 0.0


class VoiceOver(_Lenient):
    """Optional voice-over routing for a segment (playbook §3: post-trip narration).

    When present, :func:`build_timeline` uses the segment's own clip audio as a
    narration pickup (``tracks.voice``) and shows ``picture`` cuts instead of the
    segment's own footage — the narrator is heard, not seen, except where the
    picture cuts fall short of the narration length.
    """

    picture: list[VoiceOverPicture] = Field(default_factory=list)


class PlanCutaway(_Lenient):
    """A picture-only insert placed after one sentence of a speech segment.

    The narration audio underneath keeps running (see
    :func:`_build_sentence_run_segments`): the cutaway's own audio is
    replaced by the stretch of the speech segment's clip that would
    otherwise have been skipped, exactly the semantics
    ``ytedit/ai/overlay.py`` produces for a planner that (in the legacy,
    raw-seconds regime) split one take into two contiguous pieces around a
    cutaway.
    """

    clip: str = ""
    in_: float = Field(0.0, alias="in")
    out: float = 0.0
    after_sentence: str = ""


class PlanAudioFrom(_Lenient):
    """Internal: where a synthetic (sentence-expanded) segment's audio comes from.

    Not part of the planner-facing schema — set only by
    :func:`_build_sentence_run_segments` when it expands a ``cutaways`` entry
    into its own :class:`PlanSegment`.
    """

    clip: str = ""
    in_: float = Field(0.0, alias="in")
    out: float = 0.0


class PlanSegment(_Lenient):
    """A proposed video segment, before deterministic post-processing."""

    clip: str = ""
    in_: float = Field(0.0, alias="in")
    out: float = 0.0
    role: str = ""
    #: ``None`` means "planner did not care" — the vertical rule then applies.
    fit: str | None = None
    zoom: float = 1.0
    transition_in: dict[str, Any] = Field(default_factory=dict)
    mute_source: bool = False
    source_audio_gain_db: float = 0.0
    grade: str = "default"
    speed: float = 1.0
    #: Post-trip narration routing (playbook §3); see :class:`VoiceOver`.
    voice_over: VoiceOver | None = None
    #: Script-first speech routing: contiguous sentence ids of one clip: when
    #: set, ``in``/``out`` above are ignored — :func:`build_timeline` derives
    #: them from the sentence catalogue instead (see
    #: :func:`_expand_sentence_segments`).
    sentences: list[str] = Field(default_factory=list)
    #: Optional picture-only inserts along ``sentences`` (see :class:`PlanCutaway`).
    cutaways: list[PlanCutaway] = Field(default_factory=list)
    #: Internal only (never planner-facing): set by sentence-cutaway expansion
    #: so a synthetic cutaway segment carries its borrowed audio through the
    #: normal per-segment pipeline in :func:`build_timeline`.
    audio_from: PlanAudioFrom | None = None
    notes: str = ""


class PlanCaption(_Lenient):
    """A proposed burned-in caption (absolute timeline time)."""

    at: float = 0.0
    end: float = 0.0
    text: str = ""
    style: str = "location"
    position: str = "lower-left"


class PlanMusicCue(_Lenient):
    """A proposed music cue; the file is generated later by the music stage."""

    id: str = ""
    style: str = ""
    mood: str = ""
    section: str = ""
    at: float = 0.0
    end: float = 0.0
    length_s: float = 0.0
    gain_db: float = -18.0
    duck_amount_db: float = -12.0


class PlanMuteRange(_Lenient):
    """A proposed source-audio mute, in clip time."""

    clip: str = ""
    s: float = 0.0
    e: float = 0.0
    gain_db: float = -60.0
    reason: str = ""


class PlanMarker(_Lenient):
    """A proposed marker (advisory — real markers come from settings)."""

    at: float = 0.0
    label: str = ""


class PlanChapter(_Lenient):
    """A proposed YouTube chapter."""

    at: float = 0.0
    title: str = ""


class NarrationRequest(_Lenient):
    """One pickup line for the user to record (playbook §5)."""

    id: str = ""
    purpose: str = "bridge"
    place_after_segment: str = ""
    target_seconds: float = 5.0
    script: str = ""
    why: str = ""
    tone: str = ""


class ThumbnailConcept(_Lenient):
    """A thumbnail idea tied to a real frame in the footage."""

    concept: str = ""
    frame_clip: str = ""
    frame_t: float = 0.0
    text: str = ""
    colors: list[str] = Field(default_factory=list)


class CTA(_Lenient):
    """The narratively-integrated subscribe call to action."""

    at_s: float = 0.0
    script: str = ""


class EditPlan(_Lenient):
    """The planner's full proposal, normalized from whatever shape it returned."""

    story: Story = Field(default_factory=Story)
    cold_open: list[ColdOpenPick] = Field(default_factory=list)
    segments: list[PlanSegment] = Field(default_factory=list)
    captions: list[PlanCaption] = Field(default_factory=list)
    music_cues: list[PlanMusicCue] = Field(default_factory=list)
    mute_ranges: list[PlanMuteRange] = Field(default_factory=list)
    markers: list[PlanMarker] = Field(default_factory=list)
    chapters: list[PlanChapter] = Field(default_factory=list)
    narration_requests: list[NarrationRequest] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    title_candidates: list[str] = Field(default_factory=list)
    thumbnail_concepts: list[ThumbnailConcept] = Field(default_factory=list)
    cta: CTA = Field(default_factory=CTA)

    @classmethod
    def from_llm(cls, raw: Any) -> "EditPlan":
        """Normalize a planner answer into an :class:`EditPlan`.

        Accepts both the shape documented in ``prompts.md`` (a timeline draft
        plus a sibling ``edit_plan`` object) and the flatter shape described in
        ``docs/ARCHITECTURE.md``. Missing pieces default to empty rather than
        raising — the post-processor is what guarantees a usable timeline.
        """
        if not isinstance(raw, dict):
            raise PlanError(f"planner returned {type(raw).__name__}, expected a JSON object")

        inner = raw.get("edit_plan") if isinstance(raw.get("edit_plan"), dict) else {}
        tl = raw.get("timeline") if isinstance(raw.get("timeline"), dict) else raw
        tracks = tl.get("tracks") if isinstance(tl.get("tracks"), dict) else {}

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                for source in (raw, inner, tl):
                    value = source.get(key)
                    if value not in (None, [], {}, ""):
                        return value
            return default

        story = pick("story", default=None)
        if not isinstance(story, dict):
            story = {}
        beats = story.get("beats") or pick("story_outline", "beats", default=[]) or []
        story = dict(story)
        story["beats"] = [_norm_beat(b) for b in beats if isinstance(b, dict)]
        story.setdefault("title_working", str(pick("title_working", "working_title", default="")))
        story.setdefault("premise", str(pick("premise", default="")))
        story.setdefault("hook_idea", str(pick("hook_idea", "hook", default="")))

        segments = pick("segments", default=None) or tracks.get("video") or []
        captions = pick("captions", default=None) or tracks.get("captions") or []
        music = pick("music_cues", default=None) or tracks.get("music") or []

        return cls.model_validate(
            {
                "story": story,
                "cold_open": [
                    _norm_cold_open(c)
                    for c in (pick("cold_open", "cold_open_picks", default=[]) or [])
                    if isinstance(c, dict)
                ],
                "segments": [_norm_segment(s) for s in segments if isinstance(s, dict)],
                "captions": [dict(c) for c in captions if isinstance(c, dict)],
                "music_cues": [_norm_music(m) for m in music if isinstance(m, dict)],
                "mute_ranges": [
                    dict(m) for m in (pick("mute_ranges", default=[]) or []) if isinstance(m, dict)
                ],
                "markers": [
                    dict(m) for m in (pick("markers", default=[]) or []) if isinstance(m, dict)
                ],
                "chapters": [
                    _norm_chapter(c)
                    for c in (pick("chapters", default=[]) or [])
                    if isinstance(c, dict)
                ],
                "narration_requests": [
                    _norm_narration(n, i)
                    for i, n in enumerate(pick("narration_requests", default=[]) or [])
                    if isinstance(n, dict)
                ],
                "risks": [
                    str(r) for r in (pick("risks", "risk_flags", default=[]) or []) if r
                ],
                "title_candidates": _norm_titles(
                    pick("title_candidates", default=None)
                    or (tl.get("meta") or {}).get("title_candidates")
                    or []
                ),
                "thumbnail_concepts": [
                    _norm_thumbnail(t)
                    for t in (pick("thumbnail_concepts", "thumbnail_prompts", default=[]) or [])
                    if isinstance(t, dict)
                ],
                "cta": _norm_cta(pick("cta", "subscribe_cta", default={})),
            }
        )


def _norm_beat(raw: dict[str, Any]) -> dict[str, Any]:
    """Map ``{beat, target_time, notes}`` onto ``{label, at_s_target, description}``."""
    out = dict(raw)
    out.setdefault("label", raw.get("beat") or raw.get("name") or "")
    out.setdefault("at_s_target", _num(raw.get("target_time", raw.get("at", 0.0))))
    out.setdefault("description", raw.get("notes") or raw.get("description") or "")
    out["clips"] = [str(c) for c in (raw.get("clips") or []) if c]
    return out


def _norm_cold_open(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    s, e = _span(raw)
    out["in"] = s
    out["out"] = e
    out.pop("in_", None)
    return out


def _norm_segment(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a timeline-shaped segment into a :class:`PlanSegment` payload."""
    out = dict(raw)
    s, e = _span(raw)
    out["in"] = s
    out["out"] = e
    out.pop("in_", None)
    transform = raw.get("transform")
    if isinstance(transform, dict):
        if transform.get("fit"):
            out["fit"] = str(transform["fit"])
        out["zoom"] = _num(transform.get("zoom", 1.0), 1.0)
    elif isinstance(transform, str):
        out["fit"] = transform
    transition = raw.get("transition_in")
    out["transition_in"] = dict(transition) if isinstance(transition, dict) else {}
    out["mute_source"] = bool(raw.get("mute_source", False))
    return out


def _norm_music(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    s, e = _span(raw)
    out["at"] = _num(raw.get("at", s))
    out["end"] = _num(raw.get("end", e))
    duck = raw.get("duck")
    if isinstance(duck, dict) and "amount_db" in duck:
        out.setdefault("duck_amount_db", _num(duck["amount_db"], -12.0))
    return out


def _norm_chapter(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    out["at"] = _num(raw.get("at", raw.get("at_seconds", raw.get("t", 0.0))))
    return out


def _norm_narration(raw: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(raw)
    out.setdefault("id", f"n{index + 1:03d}")
    out.setdefault("purpose", raw.get("kind") or _guess_purpose(raw))
    out.setdefault("place_after_segment", str(raw.get("where") or raw.get("place") or ""))
    out["target_seconds"] = _num(raw.get("target_seconds", raw.get("length_s", 5.0)), 5.0)
    out.setdefault("why", raw.get("why") or raw.get("reason") or "")
    out.setdefault("tone", raw.get("tone") or raw.get("delivery") or "")
    return out


def _guess_purpose(raw: dict[str, Any]) -> str:
    text = f"{raw.get('where', '')} {raw.get('why', '')}".lower()
    if "intro" in text or "cold open" in text or "hook" in text:
        return "intro"
    if "outro" in text or "ending" in text or "payoff" in text:
        return "outro"
    if "cta" in text or "subscri" in text or "subskry" in text:
        return "cta"
    return "bridge"


def _norm_titles(raw: Any) -> list[str]:
    titles: list[str] = []
    for item in raw or []:
        if isinstance(item, str):
            titles.append(item)
        elif isinstance(item, dict):
            value = item.get("text") or item.get("title")
            if value:
                titles.append(str(value))
    return titles


def _norm_thumbnail(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    out.setdefault("concept", raw.get("description") or raw.get("concept") or "")
    out.setdefault("text", raw.get("on_image_text") or raw.get("text_on_image") or "")
    out["colors"] = [str(c) for c in (raw.get("colors") or []) if c]
    hint = str(raw.get("source_frame_hint") or raw.get("frame") or "")
    clip, t = _parse_frame_hint(hint)
    out.setdefault("frame_clip", raw.get("frame_clip") or clip)
    out.setdefault("frame_t", _num(raw.get("frame_t", t)))
    return out


def _parse_frame_hint(hint: str) -> tuple[str, float]:
    """Parse ``"c003@12.4"`` / ``"clip c003 at 12.4s"`` into ``(clip, seconds)``."""
    clip = ""
    seconds = 0.0
    clip_match = re.search(r"\b(c\d{3,})\b", hint)
    if clip_match:
        clip = clip_match.group(1)
    time_match = re.search(r"(?:@|\bat\s+|\bt=)\s*(\d+(?:\.\d+)?)", hint)
    if time_match:
        seconds = float(time_match.group(1))
    return clip, seconds


def _norm_cta(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        out = dict(raw)
        out["at_s"] = _num(raw.get("at_s", raw.get("at", 0.0)))
        out.setdefault("script", str(raw.get("script") or raw.get("text") or ""))
        return out
    if isinstance(raw, str):
        return {"at_s": 0.0, "script": raw}
    return {"at_s": 0.0, "script": ""}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def units_for_ledger(units: Any) -> Any:
    """Normalize an API client's ``units`` string into what the ledger expects.

    The API clients report human strings (``"1234+567 tok"``, ``"12.3s"``,
    ``"2 images"``); :func:`ytedit.costs.record` wants a number or a mapping.
    """
    if units is None or isinstance(units, (int, float, dict)):
        return units if units is not None else 0
    text = str(units).strip()
    tokens = re.fullmatch(r"(\d+)\s*\+\s*(\d+)\s*tok", text)
    if tokens:
        return {"in": int(tokens.group(1)), "out": int(tokens.group(2))}
    number = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(?:s|chars?|images?|tok)?", text)
    if number:
        return float(number.group(1))
    return {"units": text or "unknown"}


# ----------------------------------------------------------------------
# range algebra
# ----------------------------------------------------------------------
def subtract_ranges(
    span: tuple[float, float], cuts: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Remove ``cuts`` from ``span``, returning the surviving pieces in order."""
    pieces = [span]
    for cut_s, cut_e in cuts:
        if cut_e <= cut_s:
            continue
        nxt: list[tuple[float, float]] = []
        for s, e in pieces:
            if cut_e <= s + _EPS or cut_s >= e - _EPS:
                nxt.append((s, e))
                continue
            if cut_s > s + _EPS:
                nxt.append((s, min(cut_s, e)))
            if cut_e < e - _EPS:
                nxt.append((max(cut_e, s), e))
        pieces = nxt
    return [(s, e) for s, e in pieces if e - s > _EPS]


def _merge_consecutive_voice_over(
    segments: Sequence[PlanSegment], cfg: Settings
) -> tuple[list[PlanSegment], int]:
    """Merge consecutive same-clip ``voice_over`` segments into one.

    A planner that proposes one narration take as two (or more) adjacent
    segments — e.g. ``c211 1.7-12.2`` then ``c211 12.7-22.1`` — would otherwise
    have each padded independently by :func:`_build_voice_over_segment`, and
    the resulting WAVs overlap: the same syllable is extracted twice and heard
    twice in the render. Segments are merged here, before any padding happens,
    whenever they share a clip, both carry ``voice_over``, and the gap between
    them is small enough that they clearly belong to the same take.

    Args:
        segments: The planner's segment list, in order.
        cfg: Settings supplying the speech-padding and jump-cut-merge amounts
            that size the merge gap.

    Returns:
        ``(merged_segments, merge_count)`` — ``merge_count`` is how many
        segments were folded into a predecessor (a chain of three merged into
        one counts as two), for ``stats["voice_over_merged"]``.
    """
    pad_before = max(0.0, float(cfg.get("pacing.speech_pad_before", 0.30)))
    pad_after = max(0.0, float(cfg.get("pacing.speech_pad_after", 0.45)))
    merge_gap = max(0.0, float(cfg.get("pacing.merge_gap", 0.15)))
    threshold = max(1.0, pad_before + pad_after + merge_gap)

    merged: list[PlanSegment] = []
    count = 0
    for seg in segments:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.voice_over is not None
            and seg.voice_over is not None
            and prev.clip == seg.clip
            and (seg.in_ - prev.out) <= threshold + _EPS
        ):
            pictures = list(prev.voice_over.picture) + list(seg.voice_over.picture)
            prev.voice_over = VoiceOver(picture=pictures)
            prev.out = max(prev.out, seg.out)
            prev.notes = " / ".join(n for n in (prev.notes, seg.notes) if n)
            count += 1
            continue
        merged.append(seg)
    return merged, count


# ----------------------------------------------------------------------
# script-first speech: sentence-id segments (see ytedit/ai/sentences.py)
# ----------------------------------------------------------------------
def _sentence_n(sentence_id: str) -> int:
    """Parse the ``n`` out of ``<clip>#<n>``; ``-1`` when unparseable."""
    try:
        return int(sentence_id.rsplit("#", 1)[1])
    except (IndexError, ValueError):
        return -1


def _contiguous_sentence_runs(
    ids: Sequence[str], sentence_index: Mapping[str, dict[str, Any]]
) -> list[list[str]]:
    """Split ``ids`` (already known-valid) into maximal same-clip, ``n+1`` runs."""
    runs: list[list[str]] = []
    current: list[str] = []
    prev_clip: str | None = None
    prev_n: int | None = None
    for sid in ids:
        info = sentence_index[sid]
        clip = str(info.get("clip", ""))
        n = _sentence_n(sid)
        if current and clip == prev_clip and prev_n is not None and n == prev_n + 1:
            current.append(sid)
        else:
            if current:
                runs.append(current)
            current = [sid]
        prev_clip, prev_n = clip, n
    if current:
        runs.append(current)
    return runs


def _build_sentence_run_segments(
    seg: PlanSegment,
    run: list[str],
    sentence_index: Mapping[str, dict[str, Any]],
    cutaways: Sequence[PlanCutaway],
    pad_before: float,
    pad_after: float,
    stats: dict[str, Any],
) -> tuple[list[PlanSegment], set[str]]:
    """Expand one contiguous sentence run (plus any cutaways inside it).

    Splits ``run`` at every ``cutaway.after_sentence`` that falls inside it,
    in run order, emitting: a speech piece ending exactly at the split
    sentence (no trailing pad — the narration continues under the cutaway), a
    synthetic cutaway :class:`PlanSegment` carrying ``audio_from`` over the
    stretch of ``seg.clip`` the cutaway covers, then the next piece resuming
    exactly where that borrowed audio ends. Only the very first piece gets a
    leading pad; only the very last piece gets a trailing one.

    Returns:
        ``(segments, matched_after_sentence_ids)`` — the second lets the
        caller report any of ``cutaways`` that named a sentence outside this
        run (or this segment altogether) as dropped.
    """
    clip_id = str(sentence_index[run[0]].get("clip", seg.clip))
    ordered_cutaways = sorted(
        (c for c in cutaways if c.after_sentence in run),
        key=lambda c: run.index(c.after_sentence),
    )
    matched = {c.after_sentence for c in ordered_cutaways}

    result: list[PlanSegment] = []
    remaining = list(run)
    audio_cursor = 0.0
    first_piece = True

    def emit_piece(ids: list[str], is_last: bool) -> None:
        nonlocal first_piece
        first_s = float(sentence_index[ids[0]]["s"])
        last_e = float(sentence_index[ids[-1]]["e"])
        piece_in = round(max(0.0, first_s - pad_before), 3) if first_piece else round(audio_cursor, 3)
        piece_out = round(last_e + pad_after, 3) if is_last else round(last_e, 3)
        new_seg = seg.model_copy(deep=True)
        new_seg.clip = clip_id
        new_seg.in_ = piece_in
        new_seg.out = piece_out
        new_seg.sentences = list(ids)
        new_seg.cutaways = []
        new_seg.audio_from = None
        result.append(new_seg)
        first_piece = False

    for cutaway in ordered_cutaways:
        split_at = remaining.index(cutaway.after_sentence)
        piece_ids, remaining = remaining[: split_at + 1], remaining[split_at + 1 :]
        if not piece_ids:
            continue
        emit_piece(piece_ids, is_last=False)
        audio_cursor = result[-1].out
        cut_clip = str(cutaway.clip)
        cut_in, cut_out = float(cutaway.in_), float(cutaway.out)
        if not cut_clip or cut_out <= cut_in + _EPS:
            stats["dropped_cutaways"].append(
                f"after {cutaway.after_sentence}: bad cutaway ({cut_clip!r} "
                f"{cut_in:.2f}-{cut_out:.2f})"
            )
            continue
        duration = cut_out - cut_in
        result.append(
            PlanSegment(
                clip=cut_clip,
                **{"in": cut_in},
                out=cut_out,
                role="cutaway",
                audio_from=PlanAudioFrom(
                    clip=clip_id, **{"in": audio_cursor}, out=round(audio_cursor + duration, 3)
                ),
                notes=f"cutaway after {cutaway.after_sentence}",
            )
        )
        audio_cursor = round(audio_cursor + duration, 3)

    if remaining:
        emit_piece(remaining, is_last=True)
    return result, matched


def _expand_sentence_segments(
    segments: Sequence[PlanSegment],
    sentence_index: Mapping[str, dict[str, Any]],
    cfg: Settings,
    stats: dict[str, Any],
) -> list[PlanSegment]:
    """Turn every ``sentences``-carrying segment into raw-seconds segments.

    A segment without ``sentences`` passes through unchanged (the legacy,
    raw-``in``/``out`` shape keeps working exactly as before). One with
    ``sentences`` is validated against ``sentence_index`` (unknown, already
    used, or excluded — instruction / retake / duplicate — ids are dropped
    with a stat) and split into contiguous same-clip runs; each run becomes
    one or more :class:`PlanSegment`\\ s via
    :func:`_build_sentence_run_segments`.

    ``used_sentence_ids`` is tracked across the *whole* call (not per
    segment) so a sentence referenced twice anywhere in the plan is only kept
    the first time — see ``stats["duplicate_sentence_refs"]``.
    """
    pad_before = max(0.0, float(cfg.get("pacing.speech_pad_before", 0.30)))
    pad_after = max(0.0, float(cfg.get("pacing.speech_pad_after", 0.45)))
    used: set[str] = set()
    out: list[PlanSegment] = []

    for seg in segments:
        if not seg.sentences:
            out.append(seg)
            continue

        valid_ids: list[str] = []
        for sid in seg.sentences:
            info = sentence_index.get(sid)
            if info is None:
                stats["unknown_sentence_refs"].append(sid)
                continue
            if sid in used:
                stats["duplicate_sentence_refs"].append(sid)
                continue
            if info.get("instruction"):
                stats["excluded_sentence_refs"].append(f"{sid} (instruction)")
                continue
            if info.get("retake_of"):
                stats["excluded_sentence_refs"].append(
                    f"{sid} (retake_of {info['retake_of']})"
                )
                continue
            if info.get("duplicate_of"):
                stats["excluded_sentence_refs"].append(
                    f"{sid} (duplicate_of {info['duplicate_of']})"
                )
                continue
            used.add(sid)
            valid_ids.append(sid)

        if not valid_ids:
            continue

        runs = _contiguous_sentence_runs(valid_ids, sentence_index)
        if len(runs) > 1:
            stats["non_contiguous_splits"] += len(runs) - 1

        matched_after: set[str] = set()
        for run in runs:
            built, matched = _build_sentence_run_segments(
                seg, run, sentence_index, seg.cutaways, pad_before, pad_after, stats
            )
            out.extend(built)
            matched_after |= matched
        for cutaway in seg.cutaways:
            if cutaway.after_sentence and cutaway.after_sentence not in matched_after:
                stats["dropped_cutaways"].append(
                    f"after {cutaway.after_sentence}: sentence not in this segment"
                )
        stats["sentence_segments"] += 1

    return out


# ----------------------------------------------------------------------
# timeline construction
# ----------------------------------------------------------------------
def build_timeline(
    plan_obj: EditPlan,
    project: Project,
    footage_log: dict[str, Any],
    settings: Settings | None = None,
) -> tuple[Timeline, dict[str, Any]]:
    """Turn a planner proposal into a validated :class:`Timeline`.

    Args:
        plan_obj: The normalized planner output.
        project: Project supplying the clip registry (durations, orientation).
        footage_log: Merged analysis used for instruction / take / music rules.
        settings: Settings override (defaults to ``project.settings``).

    Returns:
        ``(timeline, stats)`` where ``stats`` records what the post-processor
        changed — dropped segments, excised instruction ranges, take cuts.
    """
    cfg = settings or project.settings
    width, height, fps = cfg.canvas
    timeline = new_timeline(width=width, height=height, fps=fps, language=project.language)

    state = project.load_state()
    clips: dict[str, dict[str, Any]] = state.get("clips", {})
    entries = _entries_by_clip(footage_log)
    min_shot = float(cfg.get("pacing.min_shot_seconds", 0.8))
    default_fit = str(cfg.get("fit.default_mode", "blur-fill"))

    stats: dict[str, Any] = {
        "segments_in": len(plan_obj.segments),
        "dropped_unknown_clip": [],
        "dropped_empty": [],
        "clamped": [],
        "instruction_cuts": 0,
        "take_cuts": 0,
        "split_segments": 0,
        "vertical_fixed": [],
        "dropped_too_short": [],
        "voice_over_segments": 0,
        "voice_over_merged": 0,
        "voice_over_dropped": [],
        "voice_over_dropped_picture": 0,
        # -- script-first (sentence-referenced) speech, see ai/sentences.py --
        "sentence_segments": 0,
        "unknown_sentence_refs": [],
        "duplicate_sentence_refs": [],
        "excluded_sentence_refs": [],
        "non_contiguous_splits": 0,
        "dropped_cutaways": [],
    }

    #: One entry per ``voice_over`` segment: the extracted-audio metadata plus the
    #: (mutable) list of picture-cut ``VideoSegment`` objects standing in for it.
    #: Positions are resolved from these object references *after* tidy's speech
    #: padding, because padding upstream can shift every later segment's start.
    voice_groups: list[dict[str, Any]] = []

    sentence_index = load_sentence_index(project)
    segments, stats["voice_over_merged"] = _merge_consecutive_voice_over(
        plan_obj.segments, cfg
    )
    segments = _expand_sentence_segments(segments, sentence_index, cfg, stats)

    video: list[VideoSegment] = []
    for index, seg in enumerate(segments):
        clip_id = str(seg.clip)
        clip = clips.get(clip_id)
        if clip is None:
            stats["dropped_unknown_clip"].append(clip_id or f"#{index}")
            continue

        duration = float(clip.get("duration") or 0.0)
        start, end = float(seg.in_), float(seg.out)
        if duration > 0:
            clamped_s = max(0.0, min(start, duration))
            clamped_e = max(0.0, min(end, duration))
            if abs(clamped_s - start) > 1e-3 or abs(clamped_e - end) > 1e-3:
                stats["clamped"].append(
                    f"{clip_id} {start:.2f}-{end:.2f} -> {clamped_s:.2f}-{clamped_e:.2f}"
                )
            start, end = clamped_s, clamped_e
        if end - start <= _EPS:
            stats["dropped_empty"].append(f"{clip_id} {seg.in_:.2f}-{seg.out:.2f}")
            continue

        entry = entries.get(clip_id, {})

        if seg.voice_over is not None:
            added = _build_voice_over_segment(
                seg=seg,
                clip_id=clip_id,
                clip=clip,
                clips=clips,
                seg_in=start,
                seg_out=end,
                entry=entry,
                cfg=cfg,
                project=project,
                default_fit=default_fit,
                video=video,
                voice_groups=voice_groups,
                stats=stats,
            )
            if not added:
                stats["voice_over_dropped"].append(f"{clip_id} {start:.2f}-{end:.2f}")
            continue

        cuts: list[tuple[float, float]] = []
        if seg.audio_from is None:
            # A synthetic cutaway carrying borrowed audio is already an exact,
            # deterministically-computed range (see
            # _build_sentence_run_segments) — excising instructions/rejected
            # takes from it would risk splitting its audio_from mapping.
            cuts = list(instruction_ranges(entry))
            if cuts:
                stats["instruction_cuts"] += 1
            # Silent B-roll keeps rejected takes on purpose (playbook §3).
            if not seg.mute_source:
                take_cuts = rejected_take_ranges(entry)
                if take_cuts:
                    stats["take_cuts"] += 1
                cuts += take_cuts

        pieces = subtract_ranges((start, end), cuts)
        if len(pieces) > 1:
            stats["split_segments"] += 1

        fit = seg.fit
        if fit is None:
            vertical = str(clip.get("orientation", "")).lower() == "vertical" or (
                int(clip.get("height") or 0) > int(clip.get("width") or 0) > 0
            )
            fit = default_fit if vertical else "cover"
            if vertical:
                stats["vertical_fixed"].append(clip_id)
        if fit not in ("cover", "contain", "blur-fill", "crop-pan"):
            fit = "cover"

        # A single piece (guaranteed when audio_from is set, since cuts is
        # then empty) carries the whole borrowed-audio range unchanged; a
        # segment excised into several pieces would otherwise need the range
        # split proportionally, which never happens here in practice.
        audio_from = None
        if seg.audio_from is not None and len(pieces) == 1:
            audio_from = AudioFrom(
                clip=str(seg.audio_from.clip),
                **{"in": float(seg.audio_from.in_)},
                out=float(seg.audio_from.out),
            )

        for piece_s, piece_e in pieces:
            if piece_e - piece_s < min_shot:
                stats["dropped_too_short"].append(f"{clip_id} {piece_s:.2f}-{piece_e:.2f}")
                continue
            video.append(
                VideoSegment(
                    id="",  # assigned below, once the final order is known
                    clip=clip_id,
                    **{"in": round(piece_s, 3)},
                    out=round(piece_e, 3),
                    role=seg.role or "",
                    transform=Transform(fit=fit, zoom=max(1.0, _num(seg.zoom, 1.0))),
                    grade=seg.grade or "default",
                    transition_in=_transition(seg.transition_in),
                    speed=seg.speed if seg.speed and seg.speed > 0 else 1.0,
                    mute_source=bool(seg.mute_source),
                    source_audio_gain_db=_num(seg.source_audio_gain_db, 0.0),
                    audio_from=audio_from,
                    sentence_ids=list(seg.sentences) if seg.sentences else [],
                    notes=seg.notes or "",
                )
            )

    for i, segment in enumerate(video):
        segment.id = f"s{i + 1:03d}"
    # A transition cannot be longer than the shot it enters, and the first
    # segment never transitions in from anything.
    if video:
        video[0].transition_in = Transition(type="cut", duration=0.0)
    for segment in video:
        if segment.transition_in.duration > segment.duration:
            segment.transition_in = Transition(type="cut", duration=0.0)
    timeline.tracks.video = video

    # 7. Give every speech cut its air back and merge same-clip jump cuts; the
    #    model is told to leave room but the deterministic pass is what enforces
    #    it (see ytedit/ai/tidy.py). Ids are re-assigned because a merge drops one.
    timeline, padded = pad_segments_to_speech(timeline, project, cfg)
    stats["padded"] = padded
    stats["sentence_snapped"] = sentence_snap_count(padded)
    for i, segment in enumerate(timeline.tracks.video):
        segment.id = f"s{i + 1:03d}"

    # 7b. A cutaway dropped between two contiguous pieces of one take keeps the
    #     narration running underneath it (ytedit/ai/overlay.py). Ids are
    #     re-assigned again because the pass can drop the continuation segment.
    timeline, overlaid = overlay_cutaways(timeline, project, cfg)
    stats["overlaid"] = overlaid
    # 7c. No audio twice: whatever the planner and the passes above produced,
    #     a stretch of narration already heard is muted or advanced past.
    timeline, dedupe_changes = dedupe_audio(timeline, project, cfg)
    stats["deduped"] = len(dedupe_changes)
    for i, segment in enumerate(timeline.tracks.video):
        segment.id = f"s{i + 1:03d}"
    video = timeline.tracks.video

    total = timeline.duration()
    stats["segments_out"] = len(video)
    stats["duration_s"] = total

    # Voice items are placed only now: their absolute position is wherever their
    # picture-cut segments (tracked by object identity) landed after padding, not
    # where they were provisionally laid out before earlier segments could grow.
    timeline.tracks.voice, stats["voice_over_items"] = _build_voice_items(timeline, voice_groups)
    # The items above are already placed at their anchor segment's current
    # position, but resolving here too means a plan that later gets loaded and
    # re-tidied is bootstrapped in the same anchored state (harmless no-op now).
    timeline.resolve_anchors()

    timeline.mute_ranges = _build_mute_ranges(plan_obj, footage_log, clips)
    timeline.markers = _build_markers(cfg, total, plan_obj)
    timeline.tracks.captions = _build_captions(plan_obj, total)
    timeline.tracks.music = _build_music(plan_obj, total, cfg)
    timeline.chapters = _build_chapters(plan_obj, total)

    timeline.meta.generated_by = f"plan@{utcnow()}"
    timeline.meta.edited_by_human = False
    timeline.meta.title_candidates = list(plan_obj.title_candidates)
    timeline.meta.notes = plan_obj.story.premise or ""

    issues = timeline.validate(project)
    fixed, remaining = _fix_trivial_issues(timeline, project, issues)
    stats["validation_fixed"] = fixed
    stats["validation_issues"] = remaining
    return timeline, stats


# ----------------------------------------------------------------------
# voice-over segments (playbook §3: post-trip narration clips)
# ----------------------------------------------------------------------
def _vertical_fit(clip: dict[str, Any], default_fit: str) -> str:
    """Apply the vertical-clip fit rule (playbook §3) to one clip registry entry."""
    vertical = str(clip.get("orientation", "")).lower() == "vertical" or (
        int(clip.get("height") or 0) > int(clip.get("width") or 0) > 0
    )
    return default_fit if vertical else "cover"


def _voice_over_audio_range(
    entry: dict[str, Any],
    clip_duration: float,
    seg_in: float,
    seg_out: float,
    pad_before: float,
    pad_after: float,
) -> tuple[float, float]:
    """Compute the clip-time ``[start, end]`` range to extract as narration audio.

    Pads ``[seg_in, seg_out]`` by the configured speech padding, clamped to the
    clip and never crossing an excised editor instruction or a rejected take —
    the same ranges :func:`build_timeline` cuts around for ordinary segments.
    """
    start = max(0.0, seg_in - pad_before)
    end = seg_out + pad_after
    if clip_duration > 0:
        end = min(end, clip_duration)
    if end - start <= _EPS:
        return round(max(0.0, seg_in), 3), round(max(0.0, seg_out), 3)

    cuts = list(instruction_ranges(entry)) + list(rejected_take_ranges(entry))
    pieces = subtract_ranges((start, end), cuts)
    for piece_s, piece_e in pieces:
        if piece_s <= seg_in + _EPS and piece_e >= seg_out - _EPS:
            return round(piece_s, 3), round(piece_e, 3)

    best: tuple[float, float] | None = None
    best_overlap = 0.0
    for piece_s, piece_e in pieces:
        overlap = min(piece_e, seg_out) - max(piece_s, seg_in)
        if overlap > best_overlap:
            best_overlap = overlap
            best = (piece_s, piece_e)
    if best is not None:
        return round(best[0], 3), round(best[1], 3)
    # Nothing survived the cuts (the whole padded span was excised) — fall back to
    # the segment's own bounds, unpadded, rather than dropping the narration.
    return round(max(0.0, seg_in), 3), round(max(0.0, seg_out), 3)


def _voice_source_audio(project: Project, clip: dict[str, Any], clip_id: str) -> Path:
    """The work WAV to cut narration audio from — denoised when active.

    Mirrors :func:`ytedit.media.render.denoised_audio`, re-read here rather than
    imported to keep the plan stage free of the heavier render-module imports.
    """
    if clip.get("use_denoised") and clip.get("denoised"):
        candidate = project.path / str(clip["denoised"])
        if candidate.exists():
            return candidate
    return project.audio_path(clip_id)


def _extract_voice_clip(audio_src: Path, start: float, end: float, out_path: Path) -> None:
    """Cut ``[start, end)`` from ``audio_src`` into ``out_path`` (idempotent)."""
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    ff(
        "-ss", f"{start:.6f}",
        "-i", str(audio_src),
        "-t", f"{duration:.6f}",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    )


def _extend_picture_cuts(
    group_segments: list[VideoSegment], shortfall: float, clips: dict[str, Any]
) -> float:
    """Extend real picture cuts in place to absorb a shortfall; never shrinks.

    Tries the LAST cut first, up to its own clip's duration, then walks
    backwards through the earlier cuts if the last one could not absorb it
    all — this is always tried before ever falling back to the narrator's own
    face-cam picture (see :func:`_build_voice_over_segment`).

    Args:
        group_segments: The picture-cut run, screen order (mutated in place).
        shortfall: Seconds still missing from the narration length.
        clips: Clip registry, for each cut's own clip duration.

    Returns:
        Whatever shortfall remains after extending (``0`` when fully covered).
    """
    remaining = shortfall
    for seg in reversed(group_segments):
        if remaining <= _EPS:
            break
        clip_duration = float((clips.get(seg.clip) or {}).get("duration") or 0.0)
        room = (clip_duration - seg.out) if clip_duration > 0 else remaining
        extend = min(remaining, max(0.0, room))
        if extend > _EPS:
            seg.out = round(seg.out + extend, 3)
            remaining -= extend
    return max(0.0, remaining)


def _normalize_picture_run(
    group_segments: list[VideoSegment], voice_duration: float, clips: dict[str, Any]
) -> float:
    """Trim or extend ``group_segments`` in place to match ``voice_duration``.

    A run longer than the narration is trimmed from its last cut (dropping it
    entirely if trimming would leave nothing); a run still short after the
    fallback picture was added is extended on the last cut, clamped to that
    clip's own duration — the narration must never run past its own picture.

    Returns:
        The final total duration of ``group_segments``.
    """
    total = sum(s.duration for s in group_segments)
    while group_segments and total > voice_duration + _EPS:
        last = group_segments[-1]
        overshoot = total - voice_duration
        new_dur = last.duration - overshoot
        if new_dur > _EPS:
            last.out = round(last.in_ + new_dur, 3)
            total = voice_duration
        else:
            total -= last.duration
            group_segments.pop()

    if group_segments and total < voice_duration - _EPS:
        last = group_segments[-1]
        clip_duration = float((clips.get(last.clip) or {}).get("duration") or 0.0)
        new_out = last.out + (voice_duration - total)
        if clip_duration > 0:
            new_out = min(new_out, clip_duration)
        if new_out > last.out + _EPS:
            total += new_out - last.out
            last.out = round(new_out, 3)
    return total


def _build_voice_over_segment(
    seg: PlanSegment,
    clip_id: str,
    clip: dict[str, Any],
    clips: dict[str, dict[str, Any]],
    seg_in: float,
    seg_out: float,
    entry: dict[str, Any],
    cfg: Settings,
    project: Project,
    default_fit: str,
    video: list[VideoSegment],
    voice_groups: list[dict[str, Any]],
    stats: dict[str, Any],
) -> bool:
    """Turn one ``voice_over`` segment into a voice pickup plus picture cuts.

    Appends the resolved picture-cut ``VideoSegment``\\ s to ``video`` and records
    the extracted-audio metadata in ``voice_groups`` for :func:`_build_voice_items`
    to place once the timeline's final segment positions are known.

    Returns:
        ``True`` when a voice-over pickup was produced, ``False`` when there was
        nothing usable to extract (the caller then drops the segment entirely, as
        it would an unknown clip).
    """
    pad_before = max(0.0, float(cfg.get("pacing.speech_pad_before", 0.30)))
    pad_after = max(0.0, float(cfg.get("pacing.speech_pad_after", 0.45)))
    clip_duration = float(clip.get("duration") or 0.0)

    a_start, a_end = _voice_over_audio_range(
        entry, clip_duration, seg_in, seg_out, pad_before, pad_after
    )
    voice_duration = round(a_end - a_start, 3)
    if voice_duration <= _EPS:
        return False

    fname = f"vo_{clip_id}_{a_start:06.2f}_{a_end:06.2f}.wav"
    rel_path = f"voice/{fname}"
    out_path = project.voice_dir / fname
    audio_src = _voice_source_audio(project, clip, clip_id)
    try:
        _extract_voice_clip(audio_src, a_start, a_end, out_path)
    except FFmpegError as exc:
        log.warning(
            "voice-over extraction failed for %s %.2f-%.2f: %s", clip_id, a_start, a_end, exc
        )
        return False

    group_segments: list[VideoSegment] = []
    picks = list(seg.voice_over.picture) if seg.voice_over is not None else []
    for pick in picks:
        pick_clip_id = str(pick.clip)
        pick_clip = clips.get(pick_clip_id)
        if pick_clip is None:
            stats["voice_over_dropped_picture"] += 1
            continue
        pick_duration = float(pick_clip.get("duration") or 0.0)
        p_start, p_end = float(pick.in_), float(pick.out)
        if pick_duration > 0:
            p_start = max(0.0, min(p_start, pick_duration))
            p_end = max(0.0, min(p_end, pick_duration))
        if p_end - p_start <= _EPS:
            stats["voice_over_dropped_picture"] += 1
            continue
        group_segments.append(
            VideoSegment(
                id="",
                clip=pick_clip_id,
                **{"in": round(p_start, 3)},
                out=round(p_end, 3),
                role="b-roll",
                transform=Transform(fit=_vertical_fit(pick_clip, default_fit)),
                mute_source=True,
                notes=f"VO picture for {clip_id}",
            )
        )

    total = sum(s.duration for s in group_segments)
    shortfall = voice_duration - total
    if shortfall > _EPS:
        # Speech padding routinely makes the narration ~0.75s longer than the
        # picture cuts the planner picked for it. Cover that first by growing
        # the real picture cuts (last one first) — never by flashing back to
        # the narrator's own face for a fraction of a second.
        if group_segments:
            shortfall = _extend_picture_cuts(group_segments, shortfall, clips)
        min_shot = max(0.0, float(cfg.get("pacing.min_shot_seconds", 0.8)))
        if shortfall > _EPS:
            if not group_segments or shortfall >= min_shot - _EPS:
                # Either there was nothing to extend, or what's left is a full
                # shot on its own — the VO clip's own picture fills the gap,
                # muted (its audio is already carried by the voice item).
                group_segments.append(
                    VideoSegment(
                        id="",
                        clip=clip_id,
                        **{"in": round(seg_in, 3)},
                        out=round(seg_out, 3),
                        role="b-roll",
                        transform=Transform(fit=_vertical_fit(clip, default_fit)),
                        mute_source=True,
                        notes=f"VO picture for {clip_id}",
                    )
                )
            else:
                # Too small to extend a real cut with and too small to justify
                # a fallback shot of its own — absorb it as a last resort by
                # pulling the first cut's in-point back.
                first = group_segments[0]
                first.in_ = round(max(0.0, first.in_ - shortfall), 3)

    _normalize_picture_run(group_segments, voice_duration, clips)
    if not group_segments:
        return False

    video.extend(group_segments)
    voice_groups.append(
        {
            "file": rel_path,
            "clip": clip_id,
            "in": a_start,
            "out": a_end,
            "duration": voice_duration,
            "segments": group_segments,
        }
    )
    stats["voice_over_segments"] += 1
    return True


def _build_voice_items(
    timeline: Timeline, voice_groups: list[dict[str, Any]]
) -> tuple[list[VoiceItem], list[dict[str, Any]]]:
    """Place each voice-over group at its picture cuts' final timeline position.

    Segment padding (:func:`ytedit.ai.tidy.pad_segments_to_speech`) can shift the
    start of everything after a padded cut, so the voice item's absolute ``at``
    is only resolved here, from where its (identity-tracked) picture segments
    actually landed — never from a position computed before padding ran.

    Returns:
        ``(voice_items, voice_over_report)`` — the second is the same information
        in a shape :func:`render_edit_plan_md` can list.
    """
    starts: dict[int, float] = {
        id(pos.segment): pos.start for pos in timeline.segment_positions()
    }

    items: list[VoiceItem] = []
    report: list[dict[str, Any]] = []
    for i, group in enumerate(voice_groups):
        start: float | None = None
        anchor_segment: VideoSegment | None = None
        for segment in group["segments"]:
            candidate = starts.get(id(segment))
            if candidate is not None and (start is None or candidate < start):
                start = candidate
                anchor_segment = segment
        if start is None:
            log.warning(
                "voice-over pickup %s lost all its picture segments during tidy — dropped",
                group["file"],
            )
            continue
        end = round(start + group["duration"], 3)
        item_id = f"v{i + 1:03d}"
        # Anchor to the first picture segment of the group, offset 0, so the
        # pickup follows it through any later pass instead of staying pinned
        # to the absolute time computed here.
        anchor = VoiceAnchor(segment=anchor_segment.id, offset=0.0) if anchor_segment else None
        items.append(
            VoiceItem(id=item_id, file=group["file"], at=round(start, 3), end=end, anchor=anchor)
        )
        report.append(
            {
                "id": item_id,
                "file": group["file"],
                "clip": group["clip"],
                "in": group["in"],
                "out": group["out"],
                "at": round(start, 3),
                "end": end,
            }
        )
    return items, report


def _transition(raw: dict[str, Any]) -> Transition:
    """Build a :class:`Transition` from a loose dict; default is a hard cut."""
    kind = str(raw.get("type", "cut")).lower()
    if kind not in ("cut", "fade", "xfade"):
        kind = "cut"
    duration = _num(raw.get("duration", 0.0))
    if kind == "cut":
        duration = 0.0
    return Transition(type=kind, duration=max(0.0, duration), name=str(raw.get("name", "fade")))


def _build_mute_ranges(
    plan_obj: EditPlan, footage_log: dict[str, Any], clips: dict[str, Any]
) -> list[MuteRange]:
    """Merge the planner's mute ranges with every footage-log music flag."""
    out: list[MuteRange] = []
    seen: set[tuple[str, float, float]] = set()

    def add(clip: str, s: float, e: float, gain_db: float, reason: str) -> None:
        if clip not in clips or e <= s:
            return
        key = (clip, round(s, 2), round(e, 2))
        if key in seen:
            return
        seen.add(key)
        out.append(MuteRange(clip=clip, s=round(s, 3), e=round(e, 3), gain_db=gain_db, reason=reason))

    for item in plan_obj.mute_ranges:
        add(str(item.clip), float(item.s), float(item.e), _num(item.gain_db, -60.0),
            item.reason or "plan")
    for entry in footage_entries(footage_log):
        for flag in background_music_ranges(entry):
            add(flag["clip"], flag["s"], flag["e"], flag["gain_db"], flag["reason"])
    return sorted(out, key=lambda m: (m.clip, m.s))


def _build_markers(cfg: Settings, total: float, plan_obj: EditPlan) -> list[Marker]:
    """Structural markers from ``pacing.markers`` (negative ``at`` = from the end)."""
    markers: list[Marker] = []
    for spec in cfg.get("pacing.markers", []) or []:
        if not isinstance(spec, dict):
            continue
        at = _num(spec.get("at", 0.0))
        label = str(spec.get("label", ""))
        if at < 0:
            at = total + at
        if at < -_EPS or at > total + _EPS:
            continue
        markers.append(Marker(at=round(max(0.0, at), 3), label=label))
    # Subscribe CTA rides at ~30% of runtime unless the planner placed it.
    cta_at = plan_obj.cta.at_s if plan_obj.cta.at_s > 0 else round(total * 0.30, 1)
    if total > 0 and 0 < cta_at < total:
        markers.append(Marker(at=round(cta_at, 3), label="subscribe-cta"))
    return sorted(markers, key=lambda m: m.at)


def _build_captions(plan_obj: EditPlan, total: float) -> list[Caption]:
    """Clamp, de-overlap and id the planner's captions."""
    ordered = sorted(plan_obj.captions, key=lambda c: _num(c.at))
    out: list[Caption] = []
    for item in ordered:
        text = (item.text or "").strip()
        if not text:
            continue
        at = max(0.0, _num(item.at))
        end = _num(item.end, at + 2.5)
        if end <= at:
            end = at + 2.5
        if at >= total - _EPS:
            continue
        end = min(end, total)
        if end - at < 0.4:
            continue
        style = item.style if item.style in ("location", "hook", "subtitle") else "location"
        position = item.position if item.position in (
            "lower-left", "lower-right", "lower-center", "center",
            "upper-left", "upper-right", "upper-center",
        ) else "lower-left"
        if out and at < out[-1].end - _EPS and out[-1].style != "subtitle":
            # Trim the previous card rather than dropping either of them.
            out[-1].end = round(max(out[-1].at + 0.4, at - 0.05), 3)
            if out[-1].end <= out[-1].at:
                out.pop()
        out.append(
            Caption(id="", at=round(at, 3), end=round(end, 3), text=text, style=style,
                    position=position)  # type: ignore[arg-type]
        )
    for i, caption in enumerate(out):
        caption.id = f"t{i + 1:03d}"
    return out


def _build_music(plan_obj: EditPlan, total: float, cfg: Settings) -> list[MusicCue]:
    """Convert plan cues to timeline cues; files are filled in by the music stage."""
    amount = _num(cfg.get("ducking.amount_db", -12.0), -12.0)
    attack = _num(cfg.get("ducking.attack", 0.15), 0.15)
    release = _num(cfg.get("ducking.release", 0.6), 0.6)
    out: list[MusicCue] = []
    for i, cue in enumerate(sorted(plan_obj.music_cues, key=lambda c: _num(c.at))):
        at = max(0.0, _num(cue.at))
        end = _num(cue.end, at + _num(cue.length_s, 60.0))
        if end <= at:
            end = at + max(10.0, _num(cue.length_s, 60.0))
        if at >= total - _EPS:
            continue
        end = min(end, total)
        if out and at < out[-1].end - _EPS:
            at = out[-1].end
        if end - at < 5.0:
            continue
        cue_id = cue.id or f"m{i + 1:03d}"
        out.append(
            MusicCue(
                id=cue_id,
                file=f"music/{cue_id}.mp3",
                at=round(at, 3),
                end=round(end, 3),
                gain_db=_num(cue.gain_db, -18.0),
                fade_in=2.0,
                fade_out=3.0,
                duck=Duck(
                    mode="auto",
                    amount_db=_num(cue.duck_amount_db, amount),
                    attack=attack,
                    release=release,
                ),
            )
        )
    for i, cue in enumerate(out):
        cue.id = f"m{i + 1:03d}"
        cue.file = f"music/{cue.id}.mp3"
    return out


def _build_chapters(plan_obj: EditPlan, total: float) -> list[Chapter]:
    """Enforce YouTube's chapter rules: start at 0, ascending, >= 10 s apart."""
    ordered = sorted(plan_obj.chapters, key=lambda c: _num(c.at))
    out: list[Chapter] = []
    for item in ordered:
        title = (item.title or "").strip()
        at = max(0.0, _num(item.at))
        if not title or at > max(0.0, total - 10.0) + _EPS:
            continue
        if out and at - out[-1].at < 10.0:
            continue
        out.append(Chapter(at=round(at, 3), title=title))
    if out and out[0].at > _EPS:
        out[0].at = 0.0
    return out


def _fix_trivial_issues(
    timeline: Timeline, project: Project, issues: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Fix what can be fixed mechanically; return ``(fixed, remaining)``.

    Music files legitimately do not exist yet at plan time (the music stage
    generates them from the cue sheet), so those issues are filtered out rather
    than reported as problems.
    """
    fixed: list[str] = []
    total = timeline.duration()
    for caption in timeline.tracks.captions:
        if caption.end > total + _EPS:
            fixed.append(f"caption {caption.id}: trimmed {caption.end:.2f}s -> {total:.2f}s")
            caption.end = round(total, 3)
    timeline.tracks.captions = [c for c in timeline.tracks.captions if c.end - c.at > 0.3]
    for cue in timeline.tracks.music:
        if cue.end > total + _EPS:
            fixed.append(f"music {cue.id}: trimmed {cue.end:.2f}s -> {total:.2f}s")
            cue.end = round(total, 3)
    timeline.markers = [m for m in timeline.markers if m.at <= total + _EPS]

    remaining = [
        issue
        for issue in timeline.validate(project)
        if "missing file music/" not in issue
    ]
    return fixed, remaining


# ----------------------------------------------------------------------
# pacing report (research rules 5-9)
# ----------------------------------------------------------------------
def pacing_report(timeline: Timeline, settings: Settings) -> list[str]:
    """Check a timeline against the pacing rules of the production playbook.

    Implements research rules 5-9:

    * 5 — max shot length by zone (4 s before 6:00, 7 s after) plus a rolling
      30 s average that warns above 5 s in the early zone;
    * 6 — dead time: more than 12 s with no cut, caption or transform change;
    * 8 — ending guard: a wrap-up phrase in the last caption or segment notes;
    * 9 — A-roll runs longer than 10 s without a cutaway, and the overall
      A-roll share of the runtime (target 30-45%).

    Args:
        timeline: Timeline to check.
        settings: Source of ``pacing.*`` thresholds.

    Returns:
        Human-readable warnings, empty when the timeline passes.
    """
    early = float(settings.get("pacing.max_shot_seconds_early", 4.0))
    late = float(settings.get("pacing.max_shot_seconds_late", 7.0))
    switch_at = float(settings.get("pacing.pacing_switch_at", 360.0))
    warnings: list[str] = []

    positions = timeline.segment_positions()
    if not positions:
        return ["timeline has no video segments"]
    total = timeline.duration()

    # --- rule 5: shot-length ceilings ---------------------------------
    for pos in positions:
        limit = early if pos.start < switch_at else late
        if pos.segment.duration > limit + _EPS:
            warnings.append(
                f"rule 5: shot {pos.segment.id} ({pos.segment.clip}) is "
                f"{pos.segment.duration:.1f}s at {_mmss(pos.start)} — max {limit:.0f}s "
                f"{'before' if pos.start < switch_at else 'after'} {_mmss(switch_at)}"
            )

    # --- rule 5: rolling 30 s average in the early zone ----------------
    window = 30.0
    start = 0.0
    while start < min(switch_at, total) - _EPS:
        end = start + window
        in_window = [p for p in positions if p.start < end and p.end > start]
        if len(in_window) >= 2:
            average = sum(p.segment.duration for p in in_window) / len(in_window)
            if average > 5.0:
                warnings.append(
                    f"rule 5: rolling average shot length {average:.1f}s in "
                    f"{_mmss(start)}-{_mmss(end)} — target under 5s before {_mmss(switch_at)}"
                )
        start = end

    # --- rule 6: dead time --------------------------------------------
    changes = sorted(
        {round(p.start, 3) for p in positions}
        | {round(c.at, 3) for c in timeline.tracks.captions}
        | {0.0, round(total, 3)}
    )
    for prev, cur in zip(changes, changes[1:]):
        if cur - prev > 12.0 + _EPS:
            warnings.append(
                f"rule 6: {cur - prev:.1f}s with no visual change between "
                f"{_mmss(prev)} and {_mmss(cur)} (limit 12s)"
            )

    # --- rule 9: A-roll runs and ratio ---------------------------------
    run_start: float | None = None
    run_end = 0.0
    aroll_total = 0.0
    for pos in positions:
        # Keyed on the role alone (plus mute): an overlay cutaway carrying
        # ``audio_from`` is still a cutaway on screen, so it deliberately keeps
        # failing this regex and keeps breaking the A-roll run.
        is_aroll = bool(AROLL_ROLE_RE.search(pos.segment.role or "")) and not pos.segment.mute_source
        if is_aroll:
            aroll_total += pos.segment.duration
            if run_start is None:
                run_start = pos.start
            run_end = pos.end
        else:
            if run_start is not None and run_end - run_start > 10.0 + _EPS:
                warnings.append(
                    f"rule 9: A-roll run of {run_end - run_start:.1f}s from "
                    f"{_mmss(run_start)} with no cutaway (limit 10s)"
                )
            run_start = None
    if run_start is not None and run_end - run_start > 10.0 + _EPS:
        warnings.append(
            f"rule 9: A-roll run of {run_end - run_start:.1f}s from "
            f"{_mmss(run_start)} with no cutaway (limit 10s)"
        )
    if total > 0 and aroll_total > 0:
        share = aroll_total / total
        if not 0.30 <= share <= 0.45:
            warnings.append(
                f"rule 9: A-roll is {share * 100:.0f}% of the runtime (target 30-45%)"
            )

    # --- rule 8: ending guard ------------------------------------------
    tail_texts: list[tuple[str, str]] = []
    last = positions[-1].segment
    if last.notes:
        tail_texts.append((f"segment {last.id} notes", last.notes))
    for caption in timeline.tracks.captions:
        if caption.end >= total - 5.0:
            tail_texts.append((f"caption {caption.id}", caption.text))
    for where, text in tail_texts:
        match = ENDING_GUARD_RE.search(text or "")
        if match:
            warnings.append(
                f"rule 8: wrap-up phrase {match.group(0)!r} in {where} — cut it, "
                "the video must end on the payoff with a hard cut"
            )

    # --- rule 4: markers need a beat ------------------------------------
    for marker in timeline.markers:
        if timeline.segment_at(marker.at) is None and marker.at < total - _EPS:
            warnings.append(f"rule 4: marker {marker.label!r} at {_mmss(marker.at)} has no shot")

    return warnings


def _mmss(seconds: float) -> str:
    """Format seconds as ``m:ss`` (``h:mm:ss`` past an hour)."""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# ----------------------------------------------------------------------
# markdown reports
# ----------------------------------------------------------------------
def render_edit_plan_md(
    plan_obj: EditPlan,
    timeline: Timeline,
    project: Project,
    stats: dict[str, Any],
    warnings: Sequence[str],
    sentence_index: Mapping[str, dict[str, Any]] | None = None,
) -> str:
    """Render the human-readable ``plan/edit_plan.md``.

    Args:
        sentence_index: The flat sentence catalogue (``<clip>#<n>`` -> sentence
            dict, see ``ytedit/ai/sentences.py``) used to show the first words
            of a speech segment's script next to its sentence ids. Omitted
            (``None``) just skips that text — the ids alone still print.
    """
    sentence_index = sentence_index or {}
    total = timeline.duration()
    lines: list[str] = [
        f"# Edit plan — {project.slug}",
        "",
        f"Generated: {utcnow()} · language: `{project.language}` · "
        f"estimated runtime: **{_mmss(total)}** "
        f"({len(timeline.tracks.video)} shots, {len(timeline.clip_ids())} clips)",
        "",
    ]
    if plan_obj.story.title_working:
        lines += [f"**Working title:** {plan_obj.story.title_working}", ""]
    if plan_obj.story.premise:
        lines += [f"**Premise:** {plan_obj.story.premise}", ""]
    if plan_obj.story.hook_idea:
        lines += [f"**Hook:** {plan_obj.story.hook_idea}", ""]

    # -- story outline --------------------------------------------------
    lines += ["## Story outline", ""]
    if plan_obj.story.beats:
        lines += ["| Target | Beat | Clips | Notes |", "|---|---|---|---|"]
        for beat in plan_obj.story.beats:
            clips = ", ".join(beat.clips) or "—"
            lines.append(
                f"| {_mmss(beat.at_s_target)} | {beat.label} | {clips} | "
                f"{_md_cell(beat.description)} |"
            )
    else:
        lines.append("_The planner returned no story beats._")
    lines.append("")

    # -- cold open -------------------------------------------------------
    if plan_obj.cold_open:
        lines += ["## Cold open picks", ""]
        for pick in plan_obj.cold_open:
            lines.append(
                f"- `{pick.clip}` {pick.in_:.1f}–{pick.out:.1f}s "
                f"({pick.out - pick.in_:.1f}s) — {pick.why}"
            )
        lines.append("")

    # -- markers ---------------------------------------------------------
    lines += ["## Structural markers", ""]
    for marker in timeline.markers:
        shot = timeline.segment_at(marker.at)
        where = f"`{shot.segment.id}` / `{shot.segment.clip}`" if shot else "**no shot**"
        lines.append(f"- **{_mmss(marker.at)}** {marker.label} → {where}")
    lines.append("")

    # -- segment table ---------------------------------------------------
    lines += [
        "## Segments",
        "",
        "| # | at | clip | in–out | dur | role | fit | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, pos in enumerate(timeline.segment_positions(), start=1):
        seg = pos.segment
        mute = " 🔇" if seg.mute_source else ""
        vo = " 🎙" if seg.notes.startswith("VO picture for ") else ""
        overlay = " 🎞" if seg.audio_from is not None else ""
        note = _md_cell(seg.notes)
        if seg.audio_from is not None:
            borrowed = (
                f"audio: `{seg.audio_from.clip}` "
                f"{seg.audio_from.in_:.2f}–{seg.audio_from.out:.2f}"
            )
            note = f"{note} · {borrowed}" if note else borrowed
        if seg.sentence_ids:
            first_sentence = sentence_index.get(seg.sentence_ids[0])
            words = str((first_sentence or {}).get("text", "")).split()
            preview = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
            script = f"`{seg.sentence_ids[0]}`" + (
                f"–`{seg.sentence_ids[-1]}`" if len(seg.sentence_ids) > 1 else ""
            )
            script += f" “{preview}”" if preview else ""
            note = f"{note} · {script}" if note else script
        lines.append(
            f"| {i} | {_mmss(pos.start)} | `{seg.clip}` | "
            f"{seg.in_:.1f}–{seg.out:.1f} | {seg.duration:.1f}s | "
            f"{seg.role or '—'}{mute}{vo}{overlay} | "
            f"{seg.transform.fit} | {note} |"
        )
    lines.append("")

    # -- voice-over ---------------------------------------------------------
    voice_over_items = stats.get("voice_over_items") or []
    if voice_over_items:
        lines += [
            "## Voice-over (🎙)",
            "",
            "| voice | at | end | source clip | in–out |",
            "|---|---|---|---|---|",
        ]
        for item in voice_over_items:
            lines.append(
                f"| `{item['id']}` | {_mmss(item['at'])} | {_mmss(item['end'])} | "
                f"`{item['clip']}` | {item['in']:.1f}–{item['out']:.1f} |"
            )
        lines.append("")

    # -- captions --------------------------------------------------------
    if timeline.tracks.captions:
        lines += ["## Captions", "", "| at | end | style | text |", "|---|---|---|---|"]
        for caption in timeline.tracks.captions:
            lines.append(
                f"| {_mmss(caption.at)} | {_mmss(caption.end)} | {caption.style} | "
                f"{_md_cell(caption.text)} |"
            )
        lines.append("")

    # -- music cue sheet --------------------------------------------------
    lines += ["## Music cue sheet", ""]
    if plan_obj.music_cues:
        lines += ["| cue | section | mood / style | at–end | length | gain | duck |",
                  "|---|---|---|---|---|---|---|"]
        for cue, planned in zip(timeline.tracks.music, plan_obj.music_cues):
            lines.append(
                f"| `{cue.id}` | {planned.section or '—'} | "
                f"{_md_cell(planned.mood or planned.style)} | "
                f"{_mmss(cue.at)}–{_mmss(cue.end)} | {cue.duration:.0f}s | "
                f"{cue.gain_db:.0f} dB | {cue.duck.amount_db:.0f} dB |"
            )
    else:
        lines.append("_No music cues planned._")
    lines.append("")

    # -- mute ranges ------------------------------------------------------
    if timeline.mute_ranges:
        lines += ["## Mute ranges (source audio, clip time)", ""]
        for mute in timeline.mute_ranges:
            lines.append(
                f"- `{mute.clip}` {mute.s:.1f}–{mute.e:.1f}s @ {mute.gain_db:.0f} dB — {mute.reason}"
            )
        lines.append("")

    # -- chapters ---------------------------------------------------------
    if timeline.chapters:
        lines += ["## Chapters", ""]
        for chapter in timeline.chapters:
            lines.append(f"- {_mmss(chapter.at)} {chapter.title}")
        lines.append("")

    # -- narration --------------------------------------------------------
    lines += ["## Narration requests (checklist)", ""]
    if plan_obj.narration_requests:
        for request in plan_obj.narration_requests:
            lines += [
                f"- [ ] **{request.id} · {request.purpose}** "
                f"({request.target_seconds:.0f} s) — {request.place_after_segment or '—'}",
                f"  - Why: {request.why}",
                f"  - Script: “{request.script}”",
            ]
            if request.tone:
                lines.append(f"  - Tone: {request.tone}")
    else:
        lines.append("_No narration pickups needed._")
    lines.append("")

    if plan_obj.cta.script:
        lines += [
            "## Subscribe CTA",
            "",
            f"- **{_mmss(plan_obj.cta.at_s or total * 0.3)}** — “{plan_obj.cta.script}”",
            "",
        ]

    # -- risks & pacing ----------------------------------------------------
    lines += ["## Risks", ""]
    if plan_obj.risks:
        lines += [f"- {risk}" for risk in plan_obj.risks]
    else:
        lines.append("_No risks flagged by the planner._")
    lines.append("")

    lines += ["## Pacing report", ""]
    if warnings:
        lines += [f"- ⚠️ {warning}" for warning in warnings]
    else:
        lines.append("_Clean: no pacing rule violated._")
    lines.append("")

    # -- titles / thumbnails ------------------------------------------------
    lines += ["## Title candidates", ""]
    for title in plan_obj.title_candidates:
        lines.append(f"- ({len(title)} chars) {title}")
    if not plan_obj.title_candidates:
        lines.append("_None returned._")
    lines.append("")

    lines += ["## Thumbnail concepts", ""]
    for concept in plan_obj.thumbnail_concepts:
        frame = f"`{concept.frame_clip}` @ {concept.frame_t:.1f}s" if concept.frame_clip else "—"
        colors = ", ".join(concept.colors) or "—"
        lines.append(f"- **“{concept.text}”** — {concept.concept} (frame {frame}; colors: {colors})")
    if not plan_obj.thumbnail_concepts:
        lines.append("_None returned._")
    lines.append("")

    # -- post-processing log -------------------------------------------------
    lines += [
        "## Deterministic post-processing",
        "",
        f"- segments proposed: {stats.get('segments_in', 0)} → kept: {stats.get('segments_out', 0)}",
        f"- clips with instruction ranges excised: {stats.get('instruction_cuts', 0)}",
        f"- clips with rejected takes excised: {stats.get('take_cuts', 0)}",
        f"- segments split around excised ranges: {stats.get('split_segments', 0)}",
        f"- speech-air / jump-cut edits: {len(stats.get('padded') or [])}",
        f"- segments snapped to a sentence boundary: {stats.get('sentence_snapped', 0)}",
        f"- overlay-cutaway edits (🎞): {len(stats.get('overlaid') or [])}",
    ]
    padded = list(stats.get("padded") or [])
    if padded:
        lines.append("")
        lines.append("Speech padding (segment ids as of the pass, before re-numbering):")
        lines.append("")
        lines += [f"  - {change}" for change in padded[:20]]
        if len(padded) > 20:
            lines.append(f"  - ... and {len(padded) - 20} more")
        lines.append("")
    overlaid = list(stats.get("overlaid") or [])
    if overlaid:
        lines.append("")
        lines.append("Overlay cutaways (narration kept running underneath):")
        lines.append("")
        lines += [f"  - {change}" for change in overlaid[:20]]
        if len(overlaid) > 20:
            lines.append(f"  - ... and {len(overlaid) - 20} more")
        lines.append("")
    for key, label in (
        ("dropped_unknown_clip", "dropped (unknown clip)"),
        ("dropped_empty", "dropped (in >= out)"),
        ("dropped_too_short", "dropped (below min shot length)"),
        ("clamped", "clamped to clip duration"),
        ("vertical_fixed", "vertical clips set to blur-fill"),
        ("validation_fixed", "auto-fixed validation issues"),
        ("validation_issues", "remaining validation issues"),
    ):
        values = stats.get(key) or []
        if values:
            lines.append(f"- {label}: {', '.join(str(v) for v in values)}")
    lines.append("")

    # -- script check (script-first planning, ytedit/ai/sentences.py) --------
    lines += ["## Script check", ""]
    script_rows = (
        ("sentence_segments", "segments referencing sentence ids"),
        ("non_contiguous_splits", "non-contiguous sentence runs auto-split"),
        ("unknown_sentence_refs", "unknown sentence ids dropped"),
        ("duplicate_sentence_refs", "sentence ids reused elsewhere (kept first use only)"),
        ("excluded_sentence_refs", "instruction/retake/duplicate ids dropped"),
        ("dropped_cutaways", "cutaways dropped (bad clip/range or unmatched sentence)"),
    )
    any_script_row = False
    for key, label in script_rows:
        value = stats.get(key)
        if isinstance(value, list):
            if value:
                any_script_row = True
                lines.append(f"- {label}: {', '.join(str(v) for v in value)}")
        elif value:
            any_script_row = True
            lines.append(f"- {label}: {value}")
    if not any_script_row:
        lines.append("_Clean: every sentence reference resolved cleanly._")
    lines.append("")
    return "\n".join(lines)


def render_narration_requests_md(plan_obj: EditPlan, project: Project) -> str:
    """Render ``plan/narration_requests.md`` — headings English, scripts in-language."""
    lines = [
        f"# Narration requests — {project.slug}",
        "",
        f"Record these yourself and drop the files into `voice/` "
        f"(scripts are in `{project.language}`; read them as-is or riff on them).",
        "",
    ]
    if not plan_obj.narration_requests:
        lines.append("_Nothing to record — every beat has usable existing audio._")
        return "\n".join(lines) + "\n"

    for request in plan_obj.narration_requests:
        lines += [
            f"## {request.id} — {request.purpose}",
            "",
            f"- **Where:** {request.place_after_segment or '—'}",
            f"- **Why:** {request.why or '—'}",
            f"- **Length:** ~{request.target_seconds:.0f} s",
        ]
        if request.tone:
            lines.append(f"- **Tone:** {request.tone}")
        lines += [
            f"- **File:** `voice/{request.id}.wav`",
            "",
            "> " + (request.script or "").replace("\n", "\n> "),
            "",
        ]
    return "\n".join(lines) + "\n"


def _md_cell(text: str) -> str:
    """Make a string safe for a Markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip() or "—"


# ----------------------------------------------------------------------
# stage entry point
# ----------------------------------------------------------------------
#: Output-token budget for the planner. ``plan.system`` asks for exactly one
#: compact JSON object (no timeline document), which fits comfortably here; a
#: model that still runs long gets one automatic retry at double this budget.
#: Overridable via ``plan.max_tokens`` in the settings.
DEFAULT_PLAN_MAX_TOKENS = 32000

#: Ceiling for the automatic retry after a ``finish_reason == "length"`` answer.
#: Past this, a truncating model is a prompt problem, not a budget problem.
MAX_PLAN_MAX_TOKENS = 64000


def plan(
    project: Project,
    force: bool = False,
    notes: str | None = None,
    max_tokens: int | None = None,
    from_response: bool = False,
) -> dict[str, Any]:
    """Run the edit-planning stage.

    Args:
        project: Project whose ``analysis/footage_log.json`` is ready.
        force: Overwrite ``plan/timeline.json`` even when a human edited it.
        notes: Free-form editorial direction appended to the user prompt as
            "Editor notes"; use it to steer a re-plan without editing prompts.
            Ignored when ``from_response`` is set — there is no prompt to
            append it to.
        max_tokens: Output-token cap for the planner call. Defaults to
            ``plan.max_tokens`` in the settings, else
            :data:`DEFAULT_PLAN_MAX_TOKENS`. Ignored when ``from_response``.
        from_response: Skip the OpenRouter call entirely and rebuild from the
            previous run's ``plan/planner_response.json`` instead — useful to
            re-run the deterministic post-processing (bug fixes, settings
            tweaks) without paying for the LLM again. Everything downstream
            (``EditPlan.from_llm`` -> :func:`build_timeline` -> tidy ->
            artifact writing, including the human-edited-timeline guard) runs
            exactly as it would for a fresh planner answer. No cost is logged.

    Returns:
        ``{"timeline", "edit_plan", "markdown", "narration", "draft",
        "duration_s", "warnings", "stats", "cost_usd", "finish_reason",
        "completion_tokens"}``.

    Raises:
        PlanError: When the footage log is missing, the planner returns
            something unusable, or (``from_response``) there is no previous
            ``plan/planner_response.json`` to rebuild from.
    """
    settings = project.settings
    write_sentences(project)  # script-first: refresh the sentence catalogue every plan run
    footage_log = load_footage_log(project)
    entries = footage_entries(footage_log)
    if not entries:
        raise PlanError(
            f"footage log {project.analysis_dir / 'footage_log.json'} has no clip entries"
        )

    state = project.load_state()
    clips = state.get("clips", {})
    total_seconds = sum(float(c.get("duration") or 0.0) for c in clips.values())

    existing: Timeline | None = None
    human_edited = False
    if project.timeline_file.exists():
        try:
            existing = Timeline.load(project.timeline_file)
            human_edited = bool(existing.meta.edited_by_human)
        except Exception as exc:  # a broken timeline must not block a re-plan
            log.warning("could not read %s: %s", project.timeline_file, exc)

    project.set_stage(STAGE, "running")
    try:
        raw_path = project.plan_dir / "planner_response.json"
        if from_response:
            if not raw_path.exists():
                raise PlanError(
                    f"no planner response at {raw_path} — run `ytedit plan {project.slug}` "
                    "at least once (without --from-response) before rebuilding from it"
                )
            try:
                raw_json = json.loads(raw_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise PlanError(f"corrupted planner response {raw_path}: {exc}") from exc
            result = {
                "json": raw_json,
                "model": f"cached:{raw_path.name}",
                "cost_usd": 0.0,
                "finish_reason": "cached",
                "completion_tokens": 0,
                "prompt_tokens": 0,
            }
        else:
            result = _ask_planner(
                project=project,
                settings=settings,
                footage_log=footage_log,
                existing=existing,
                clip_count=len(entries),
                total_minutes=total_seconds / 60.0,
                notes=notes,
                max_tokens=int(
                    max_tokens
                    if max_tokens is not None
                    else settings.get("plan.max_tokens", DEFAULT_PLAN_MAX_TOKENS)
                ),
            )
            # Dump the raw answer first: a truncated or off-schema reply is only
            # debuggable if it survived to disk, and the model was already paid for.
            project.plan_dir.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps(result["json"], indent=2, ensure_ascii=False), encoding="utf-8"
            )

        plan_obj = EditPlan.from_llm(result["json"])
        timeline, stats = build_timeline(plan_obj, project, footage_log, settings)
        if not timeline.tracks.video:
            truncated = (
                " The reply hit the output-token cap (finish_reason=length), so it "
                "stops mid-JSON: raise plan.max_tokens or plan in two passes."
                if result.get("finish_reason") == "length"
                else ""
            )
            raise PlanError(
                f"the planner ({result['model']}) produced no usable video segments "
                f"({len(plan_obj.segments)} proposed, "
                f"{len(stats.get('dropped_unknown_clip', []))} referenced unknown clips, "
                f"{len(stats.get('dropped_empty', []))} were empty) — its raw answer is in "
                f"{raw_path}.{truncated}"
            )
        warnings = pacing_report(timeline, settings)

        # --- write artifacts -------------------------------------------
        project.plan_dir.mkdir(parents=True, exist_ok=True)
        write_to_draft = human_edited and not force
        target = project.plan_dir / ("timeline.draft.json" if write_to_draft else "timeline.json")
        timeline.save(target)
        if write_to_draft:
            log.warning(
                "[clip]%s[/] timeline.json is human-edited — wrote the new draft to %s "
                "(diff it, or re-run with --force to overwrite)",
                project.slug,
                target.name,
            )

        edit_plan_doc = {
            "project": project.slug,
            "generated": utcnow(),
            "model": result["model"],
            "language": project.language,
            "notes": notes or "",
            "plan": plan_obj.model_dump(by_alias=True, mode="json"),
            "raw": result["json"],
            "stats": stats,
            "pacing_warnings": warnings,
            "timeline_file": project.rel(target),
            "estimated_runtime_s": timeline.duration(),
        }
        project.edit_plan_file.write_text(
            json.dumps(edit_plan_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        md_path = project.plan_dir / "edit_plan.md"
        md_path.write_text(
            render_edit_plan_md(
                plan_obj, timeline, project, stats, warnings, load_sentence_index(project)
            ),
            encoding="utf-8",
        )
        narration_path = project.plan_dir / "narration_requests.md"
        narration_path.write_text(render_narration_requests_md(plan_obj, project), encoding="utf-8")

        summary: dict[str, Any] = {
            "timeline": str(target),
            "edit_plan": str(project.edit_plan_file),
            "markdown": str(md_path),
            "narration": str(narration_path),
            "draft": write_to_draft,
            "duration_s": timeline.duration(),
            "segments": len(timeline.tracks.video),
            "warnings": warnings,
            "stats": stats,
            "cost_usd": result["cost_usd"],
            "finish_reason": result.get("finish_reason", ""),
            "completion_tokens": result.get("completion_tokens", 0),
        }
        project.set_stage(
            STAGE,
            "done",
            cost_usd=round(result["cost_usd"], 6),
            duration_s=timeline.duration(),
            segments=len(timeline.tracks.video),
            draft=write_to_draft,
            warnings=len(warnings),
        )
        log.info(
            "[clip]%s[/] plan: %d shots, %s runtime, %d pacing warnings, "
            "%s output tokens, $%.4f",
            project.slug,
            len(timeline.tracks.video),
            _mmss(timeline.duration()),
            len(warnings),
            result.get("completion_tokens", 0),
            result["cost_usd"],
        )
        return summary
    except Exception as exc:
        project.set_stage(STAGE, "error", error=str(exc)[:500])
        raise


def _ask_planner(
    project: Project,
    settings: Settings,
    footage_log: dict[str, Any],
    existing: Timeline | None,
    clip_count: int,
    total_minutes: float,
    notes: str | None,
    max_tokens: int = DEFAULT_PLAN_MAX_TOKENS,
    client: OpenRouter | None = None,
) -> dict[str, Any]:
    """Render the plan prompts and call the planner model once."""
    model = settings.model("planner")
    system_prompt = render("plan.system", project_language=project.language)
    catalogue = load_sentences(project)
    prompt_footage_log = (
        compact_footage_log_for_planner(footage_log, catalogue) if catalogue else footage_log
    )
    user_prompt = render(
        "plan.user",
        project_slug=project.slug,
        project_language=project.language,
        clip_count=clip_count,
        total_footage_minutes=f"{total_minutes:.1f}",
        footage_log_json=json.dumps(
            prompt_footage_log, ensure_ascii=False, separators=(",", ":")
        ),
        existing_timeline_json_or_null=(
            json.dumps(existing.to_dict(), ensure_ascii=False, separators=(",", ":"))
            if existing is not None
            else "null"
        ),
    )
    if notes:
        user_prompt += f"\n\nEditor notes: {notes.strip()}"

    spend: list[float] = []

    def cost_callback(
        service: str, op: str, model: str, units: Any = None, usd: float = 0.0, **extra: Any
    ) -> None:
        spend.append(float(usd))
        charge(
            project,
            service,
            op,
            units_for_ledger(units),
            usd=usd,
            model=model,
            stage=STAGE,
            **extra,
        )

    owned = client is None
    api = client or OpenRouter(
        api_key=settings.require_key("openrouter"), cost_callback=cost_callback
    )
    _install_result_recorder(api)

    budget = max(1, int(max_tokens))
    answer: Any = None
    finish_reason = ""
    error: OpenRouterError | None = None
    try:
        for attempt in range(2):
            answer, finish_reason, error = _call_planner(
                api, model, system_prompt, user_prompt, budget
            )
            if finish_reason != "length":
                break
            # The model ran out of output tokens mid-JSON. Keep the partial answer
            # on disk (it is paid for and it is the only way to see *where* it
            # stopped), then retry once with double the budget.
            dump = _dump_truncated(project, api, attempt)
            retry = min(budget * 2, MAX_PLAN_MAX_TOKENS)
            if attempt or retry <= budget:
                log.error(
                    "[clip]%s[/] planner %s hit the %d-token output cap again — "
                    "the partial answer is in %s. Shorten the footage log, plan in "
                    "two passes, or raise plan.max_tokens past %d.",
                    project.slug, model, budget, dump, MAX_PLAN_MAX_TOKENS,
                )
                break
            log.error(
                "[clip]%s[/] planner %s stopped at the %d-token output cap "
                "(finish_reason=length) — partial answer dumped to %s; retrying "
                "once with max_tokens=%d.",
                project.slug, model, budget, dump, retry,
            )
            budget = retry
    finally:
        if owned:
            api.close()

    if answer is None:
        raise PlanError(
            f"the planner ({model}) returned no usable JSON "
            f"(finish_reason={finish_reason or 'unknown'}): {error}"
        ) from error

    result = getattr(api, "last_result", None)
    return {
        "json": answer,
        "model": model,
        "cost_usd": round(sum(spend), 6),
        "finish_reason": finish_reason,
        "max_tokens": budget,
        "completion_tokens": getattr(result, "completion_tokens", 0),
        "prompt_tokens": getattr(result, "prompt_tokens", 0),
    }


def _call_planner(
    api: Any, model: str, system_prompt: str, user_prompt: str, max_tokens: int
) -> tuple[Any, str, OpenRouterError | None]:
    """One ``ask_json`` round trip, reporting *why* the model stopped.

    Returns ``(answer_or_None, finish_reason, error_or_None)``. A truncated reply
    usually fails to parse, so the error is carried rather than raised: the caller
    decides whether a bigger budget is worth one more call.
    """
    answer: Any = None
    error: OpenRouterError | None = None
    try:
        answer = api.ask_json(
            model,
            system_prompt,
            user_prompt,
            SCHEMA_HINT,
            temperature=0.3,
            max_tokens=max_tokens,
        )
    except OpenRouterError as exc:
        error = exc
    result = getattr(api, "last_result", None)
    return answer, getattr(result, "finish_reason", "") or "", error


def _install_result_recorder(api: Any) -> None:
    """Expose the most recent :class:`ChatResult` as ``api.last_result``.

    ``ask_json`` hands back only the parsed JSON, so this is how the stage sees
    ``finish_reason`` (and the real token counts). Clients without a ``chat``
    method — the test double — are left untouched.
    """
    inner = getattr(api, "chat", None)
    if inner is None or getattr(api, "_records_results", False):
        return

    def chat(*args: Any, **kwargs: Any) -> Any:
        result = inner(*args, **kwargs)
        api.last_result = result
        return result

    api.last_result = None
    api.chat = chat
    api._records_results = True


def _dump_truncated(project: Project, api: Any, attempt: int) -> str:
    """Write the raw (truncated) model text next to the plan artifacts."""
    result = getattr(api, "last_result", None)
    text = getattr(result, "text", "") or ""
    project.plan_dir.mkdir(parents=True, exist_ok=True)
    path = project.plan_dir / f"planner_response.truncated.{attempt + 1}.txt"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:  # never let a debug dump kill the stage
        log.warning("could not write %s: %s", path, exc)
    return str(path)


#: Appended to the system prompt by ``ask_json``. ``plan.system`` in prompts.md
#: already carries the one and only schema, so this is a pointer, not a second
#: copy — restating the schema here used to cost thousands of output tokens.
SCHEMA_HINT = "Return ONLY the JSON object described in the system prompt."


__all__ = [
    "plan",
    "MAX_PLAN_MAX_TOKENS",
    "pacing_report",
    "load_footage_log",
    "footage_entries",
    "build_timeline",
    "subtract_ranges",
    "units_for_ledger",
    "DEFAULT_PLAN_MAX_TOKENS",
    "EditPlan",
    "PlanError",
    "render_edit_plan_md",
    "render_narration_requests_md",
]
