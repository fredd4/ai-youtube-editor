"""Edit planning: ``analysis/footage_log.json`` -> ``plan/edit_plan.json`` + ``plan/cut.json``.

The planner model (``models.planner``) reads the whole footage log and returns one
compact JSON object (``plan.system`` in prompts.md): a story structure, a segment
list and the cue sheets. It is never asked for a timeline document —
:func:`build_cut` derives ``plan/cut.json`` here — because everything the model
returns is a *proposal*, re-derived deterministically so the non-negotiable rules
of the playbook always hold, whatever the model said.

Since cut v2 (``docs/ARCHITECTURE.md``) the planner addresses speech by sentence
id and never by seconds, and this module owns exactly one translation step:
**planner segments -> beats**. Everything that used to police seconds here (speech
padding, sentence snapping, overlay cutaways, the audio ledger, anchors, the
``vo_`` WAV extraction) is gone: :func:`ytedit.cut.resolve` is now the only place
where a sentence becomes a time, and it is what writes ``plan/timeline.json``.

What :func:`build_cut` does, in order:

1. A segment carrying ``sentences`` becomes one or more **speech beats** — one per
   contiguous run of ids, so a planner that jumps backwards or leaves a gap gets
   split instead of silently mis-cut. ``cutaways[]`` become the beat's ``shots``
   (``after`` = the sentence they follow); ``voice_over.picture[]`` becomes the
   shots of an ``on_camera: false`` beat (the narrator is heard, not seen).
2. A segment with no ``sentences`` is B-roll: a **broll beat** in raw seconds,
   ``audio: "mute"`` when the planner asked for silence. Editor-instruction and
   rejected-take ranges are still excised from those seconds (a segment split by
   an instruction becomes several beats), and ranges are clamped to the clip.
3. A segment that *looks* like speech but carries no ``sentences`` — words in its
   range, not muted — is a **planner error**: :func:`plan` shows the model exactly
   which segments broke the rule and asks once more (see
   :class:`SpeechSecondsError`). There is no raw-seconds speech path any more.
4. Captions, chapters, markers and music cues arrive from the planner in absolute
   seconds; they are attached to the **beat** playing at that second
   (:func:`ytedit.cut.beat_spans`), because a beat is the only position that
   survives a re-cut.

A cut with ``meta.edited_by_human`` never gets overwritten: the fresh draft goes to
``plan/cut.draft.json`` for diffing (playbook §1.5).
"""

from __future__ import annotations

import json
import re

from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ytedit.ai.openrouter import OpenRouter, OpenRouterError
from ytedit.ai.prompts import render
from ytedit.ai.sentences import (
    compact_footage_log_for_planner,
    load_sentence_index,
    load_sentences,
    write_sentences,
)
from ytedit.config import Settings
from ytedit.costs import charge
from ytedit.cut import (
    Beat as CutBeat,
    Caption as CutCaption,
    Chapter as CutChapter,
    Cut,
    Marker as CutMarker,
    MusicCue as CutMusicCue,
    MuteRange,
    Shot,
    beat_at,
    beat_spans,
    cut_path,
    load_cut,
    resolve_verbose,
    save_cut,
)
from ytedit.log import get_logger
from ytedit.project import Project, utcnow
from ytedit.timeline import Duck, Timeline, Transform, Transition
from ytedit.words import load_words

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


class SpeechSecondsError(PlanError):
    """The planner gave speech as raw seconds instead of sentence ids.

    Since cut v2 there is no seconds path for narration: a segment whose range
    covers transcript words but carries neither ``sentences`` nor
    ``mute_source`` cannot be turned into a beat without guessing where a word
    starts. Rather than guess, :func:`plan` shows the model exactly which
    segments broke the rule (:meth:`prompt_note`) and asks once more; a second
    failure surfaces as this error, which is a :class:`PlanError`.

    Attributes:
        segments: One human-readable line per offending segment.
    """

    def __init__(self, segments: Sequence[str]) -> None:
        self.segments = list(segments)
        super().__init__(
            f"{len(self.segments)} planner segment(s) contain speech but no "
            "`sentences` (cut v2 addresses narration by sentence id, never by "
            "seconds): " + "; ".join(self.segments)
        )

    def prompt_note(self) -> str:
        """The correction appended to the planner's user prompt on the retry."""
        return (
            "Your previous answer broke the hard rule about speech. These segments "
            "give raw in/out seconds over a stretch of the clip where the narrator "
            "is talking, but no `sentences`:\n"
            + "\n".join(f"- {line}" for line in self.segments)
            + "\nEvery segment carrying dialogue must list the contiguous sentence "
            "ids from that clip's `sentences` inventory and omit in/out. Use in/out "
            "only for B-roll, silent-broll, audio-only and cold-open picks — or set "
            '"mute_source": true when you deliberately want that picture silent. '
            "Return the whole JSON object again, fixed."
        )


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

    When present the segment becomes an ``on_camera: false`` speech beat: the
    clip's own audio still carries the narration, but ``picture`` cuts are what
    the viewer sees. The narrator's own face only appears in a tail no picture
    cut covers, which the resolver flags (``narrator_visible``).
    """

    picture: list[VoiceOverPicture] = Field(default_factory=list)


class PlanCutaway(_Lenient):
    """A picture-only insert placed after one sentence of a speech segment.

    Becomes a :class:`ytedit.cut.Shot` on the beat that owns
    ``after_sentence``. The narration underneath never stops: a shot carries no
    audio of its own by definition, so the take keeps playing while the picture
    cuts away — which is why the planner must never fake a cutaway by splitting
    one take into two segments with a gap.
    """

    clip: str = ""
    in_: float = Field(0.0, alias="in")
    out: float = 0.0
    after_sentence: str = ""


class PlanSegment(_Lenient):
    """One proposal from the planner, on its way to becoming one or more beats.

    ``sentences`` (speech) and ``in``/``out`` (B-roll, silent, cold-open) are
    mutually exclusive: a segment carrying dialogue but no sentence ids is a
    :class:`SpeechSecondsError`, not something to interpret.
    """

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
    #: Script-first speech routing: contiguous sentence ids of one clip. When
    #: set, ``in``/``out`` above are ignored entirely — the beat carries the ids
    #: and :func:`ytedit.cut.resolve` is what turns them into seconds.
    sentences: list[str] = Field(default_factory=list)
    #: Optional picture-only inserts along ``sentences`` (see :class:`PlanCutaway`).
    cutaways: list[PlanCutaway] = Field(default_factory=list)
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


# ----------------------------------------------------------------------
# planner segments -> beats
# ----------------------------------------------------------------------
def _clip_is_vertical(clip: Mapping[str, Any]) -> bool:
    """True when a clip registry entry describes portrait footage."""
    return str(clip.get("orientation", "")).lower() == "vertical" or (
        int(clip.get("height") or 0) > int(clip.get("width") or 0) > 0
    )


def _fit_for(
    seg: PlanSegment, clip_id: str, clip: Mapping[str, Any], default_fit: str,
    stats: dict[str, Any],
) -> str:
    """The planner's ``fit``, or the vertical-clip rule when it did not care."""
    fit = seg.fit
    if fit is None:
        if _clip_is_vertical(clip):
            fit = default_fit
            if clip_id not in stats["vertical_fixed"]:
                stats["vertical_fixed"].append(clip_id)
        else:
            fit = "cover"
    if fit not in ("cover", "contain", "blur-fill", "crop-pan"):
        fit = "cover"
    return fit


def _beat_common(
    seg: PlanSegment, fit: str, first: bool
) -> dict[str, Any]:
    """The fields every beat of one planner segment shares.

    ``transition_in`` rides on the *first* beat a segment produces only: a
    segment split into several beats (a non-contiguous sentence run, an
    instruction range excised out of B-roll) is one editorial move on screen,
    so the fade belongs to its opening frame and nowhere else.
    """
    return {
        "role": seg.role or "",
        "notes": seg.notes or "",
        "transform": Transform(fit=fit, zoom=max(1.0, _num(seg.zoom, 1.0))),
        "grade": seg.grade or "default",
        "transition_in": _transition(seg.transition_in) if first else Transition(),
    }


def _shot_from_cutaway(
    cutaway: PlanCutaway, clips: Mapping[str, Any], default_fit: str,
    after: str | None, stats: dict[str, Any],
) -> Shot | None:
    """One ``cutaways[]``/``voice_over.picture[]`` entry as a :class:`Shot`.

    Returns ``None`` (with a ``dropped_cutaways`` stat) for an unknown clip or
    an empty range — a shot that cannot show a frame is worse than no shot,
    because the resolver would stretch its neighbour over the hole.
    """
    clip_id = str(cutaway.clip)
    clip = clips.get(clip_id)
    where = f"after {after}" if after else "at the beat start"
    if clip is None:
        stats["dropped_cutaways"].append(f"{where}: unknown clip {clip_id!r}")
        return None
    start, end = float(cutaway.in_), float(cutaway.out)
    duration = float(clip.get("duration") or 0.0)
    if duration > 0:
        start, end = max(0.0, min(start, duration)), max(0.0, min(end, duration))
    if end - start <= _EPS:
        stats["dropped_cutaways"].append(
            f"{where}: empty range {clip_id} {cutaway.in_:.2f}-{cutaway.out:.2f}"
        )
        return None
    return Shot(
        clip=clip_id,
        **{"in": round(start, 3)},
        out=round(end, 3),
        after=after,
        transform=Transform(fit=_clip_is_vertical(clip) and default_fit or "cover"),
    )


def _valid_sentences(
    seg: PlanSegment,
    sentence_index: Mapping[str, dict[str, Any]],
    used: dict[str, str],
    label: str,
    stats: dict[str, Any],
) -> list[str]:
    """Filter one segment's sentence references down to the usable ones.

    Unknown ids and ids already spent elsewhere in the plan are dropped (the
    resolver would error on a sentence used twice, and a plan that errors is a
    plan the user cannot look at). An ``instruction`` id is dropped outright — a
    spoken "cut this" must never reach the cut. A ``retake_of``/``duplicate_of``
    id is **kept**: the catalogue's detection of those is heuristic, so
    overriding it is the editor's call, and dropping the sentence silently
    would delete narration the planner deliberately chose. It is recorded in
    ``flagged_sentence_refs`` instead, which the edit-plan report prints.
    """
    valid: list[str] = []
    for sid in seg.sentences:
        info = sentence_index.get(sid)
        if info is None:
            stats["unknown_sentence_refs"].append(sid)
            continue
        if sid in used:
            stats["duplicate_sentence_refs"].append(f"{sid} (already in {used[sid]})")
            continue
        if info.get("instruction"):
            stats["excluded_sentence_refs"].append(f"{sid} (instruction)")
            continue
        if info.get("retake_of"):
            stats["flagged_sentence_refs"].append(f"{sid} (retake_of {info['retake_of']})")
        elif info.get("duplicate_of"):
            stats["flagged_sentence_refs"].append(
                f"{sid} (duplicate_of {info['duplicate_of']})"
            )
        used[sid] = label
        valid.append(sid)
    return valid


def _speech_beats(
    seg: PlanSegment,
    valid_ids: list[str],
    sentence_index: Mapping[str, dict[str, Any]],
    clips: Mapping[str, Any],
    default_fit: str,
    stats: dict[str, Any],
) -> list[CutBeat]:
    """Turn one planner speech segment into its beats.

    One beat per contiguous run of sentence ids: the resolver refuses a beat
    whose ids skip a number, and a jump backwards in one segment is the
    planner mis-reading its own inventory, not an edit.

    ``voice_over`` flips the whole segment off camera — the narrator is heard
    while its ``picture`` cuts play — and those cuts become the shots of the
    *first* beat, because picture cuts carry no sentence to hang the rest off.
    Otherwise each ``cutaways[]`` entry becomes a shot on whichever run owns
    the sentence it names.
    """
    clip_id = str(sentence_index[valid_ids[0]].get("clip", seg.clip))
    clip = clips.get(clip_id, {})
    fit = _fit_for(seg, clip_id, clip, default_fit, stats)
    off_camera = seg.voice_over is not None

    runs = _contiguous_sentence_runs(valid_ids, sentence_index)
    if len(runs) > 1:
        stats["non_contiguous_splits"] += len(runs) - 1

    beats: list[CutBeat] = []
    matched: set[str] = set()
    for index, run in enumerate(runs):
        shots: list[Shot] = []
        if off_camera:
            if index == 0:
                for picture in seg.voice_over.picture:  # type: ignore[union-attr]
                    shot = _shot_from_cutaway(
                        PlanCutaway(
                            clip=picture.clip, **{"in": picture.in_}, out=picture.out
                        ),
                        clips, default_fit, None, stats,
                    )
                    if shot is not None:
                        shots.append(shot)
            elif len(runs) > 1:
                stats["voice_over_split"] += 1
        else:
            for cutaway in seg.cutaways:
                if cutaway.after_sentence not in run:
                    continue
                matched.add(cutaway.after_sentence)
                shot = _shot_from_cutaway(
                    cutaway, clips, default_fit, cutaway.after_sentence, stats
                )
                if shot is not None:
                    shots.append(shot)
            shots.sort(key=lambda s: run.index(str(s.after)))

        beats.append(
            CutBeat(
                kind="speech",
                clip=clip_id,
                sentences=list(run),
                on_camera=not off_camera,
                shots=shots,
                gain_db=_num(seg.source_audio_gain_db, 0.0),
                **_beat_common(seg, fit, first=index == 0),
            )
        )

    if off_camera:
        stats["voice_over_segments"] += 1
        for cutaway in seg.cutaways:
            stats["dropped_cutaways"].append(
                f"after {cutaway.after_sentence}: voice-over segment "
                "(picture comes from voice_over.picture)"
            )
    else:
        for cutaway in seg.cutaways:
            if cutaway.after_sentence and cutaway.after_sentence not in matched:
                stats["dropped_cutaways"].append(
                    f"after {cutaway.after_sentence}: sentence not in this segment"
                )
    stats["sentence_segments"] += 1
    return beats


def _broll_beats(
    seg: PlanSegment,
    clip_id: str,
    clip: Mapping[str, Any],
    entry: Mapping[str, Any],
    default_fit: str,
    min_shot: float,
    stats: dict[str, Any],
) -> list[CutBeat]:
    """Turn one planner B-roll segment into beats, minus what must never air.

    Spoken editor instructions are excised from every range (a "put this at the
    end" that survives into the cut is the one defect nobody forgives), and so
    are the take attempts the analysis rejected — unless the planner asked for
    the picture silent on purpose, which is exactly how a rejected take is
    legitimately reused as B-roll (playbook §3). One range cut in two produces
    two beats.
    """
    start, end = float(seg.in_), float(seg.out)
    duration = float(clip.get("duration") or 0.0)
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
        return []

    cuts = list(instruction_ranges(entry))
    if cuts:
        stats["instruction_cuts"] += 1
    if not seg.mute_source:
        take_cuts = rejected_take_ranges(entry)
        if take_cuts:
            stats["take_cuts"] += 1
        cuts += take_cuts

    pieces = subtract_ranges((start, end), cuts)
    if len(pieces) > 1:
        stats["split_segments"] += 1

    fit = _fit_for(seg, clip_id, clip, default_fit, stats)
    beats: list[CutBeat] = []
    for piece_s, piece_e in pieces:
        if piece_e - piece_s < min_shot:
            stats["dropped_too_short"].append(f"{clip_id} {piece_s:.2f}-{piece_e:.2f}")
            continue
        beats.append(
            CutBeat(
                kind="broll",
                clip=clip_id,
                **{"in": round(piece_s, 3)},
                out=round(piece_e, 3),
                audio="mute" if seg.mute_source else "ambient",
                **_beat_common(seg, fit, first=not beats),
            )
        )
    return beats


def _has_words(project: Project, clip_id: str, start: float, end: float) -> bool:
    """True when a clip's transcript has any word inside ``[start, end)``."""
    return any(
        word.e > start + _EPS and word.s < end - _EPS
        for word in load_words(project, clip_id)
    )


# ----------------------------------------------------------------------
# cut construction
# ----------------------------------------------------------------------
def build_cut(
    plan_obj: EditPlan,
    project: Project,
    footage_log: dict[str, Any],
    settings: Settings | None = None,
) -> tuple[Cut, dict[str, Any]]:
    """Turn a planner proposal into ``plan/cut.json`` — the source of truth.

    This is the only translation between what the model proposed and the edit
    model of ``docs/ARCHITECTURE.md``. It produces beats, never seconds of
    speech: the sentence ids travel through untouched and
    :func:`ytedit.cut.resolve` decides where the words start.

    Args:
        plan_obj: The normalized planner output.
        project: Project supplying the clip registry, the sentence catalogue
            and the transcripts.
        footage_log: Merged analysis used for the instruction / take / music
            rules.
        settings: Settings override (defaults to ``project.settings``).

    Returns:
        ``(cut, stats)`` where ``stats`` records every deterministic decision —
        dropped segments, excised instruction ranges, flagged sentence ids,
        the resolver's own findings — for ``plan/edit_plan.md``.

    Raises:
        SpeechSecondsError: When a segment covers transcript words but carries
            no ``sentences`` and is not deliberately muted. :func:`plan` turns
            this into one retry of the planner call.
    """
    cfg = settings or project.settings
    width, height, fps = cfg.canvas

    state = project.load_state()
    clips: dict[str, dict[str, Any]] = state.get("clips", {})
    entries = _entries_by_clip(footage_log)
    min_shot = float(cfg.get("pacing.min_shot_seconds", 0.8))
    default_fit = str(cfg.get("fit.default_mode", "blur-fill"))
    sentence_index = load_sentence_index(project)

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
        "voice_over_split": 0,
        # -- script-first (sentence-referenced) speech, see ai/sentences.py --
        "sentence_segments": 0,
        "unknown_sentence_refs": [],
        "duplicate_sentence_refs": [],
        "excluded_sentence_refs": [],
        "flagged_sentence_refs": [],
        "non_contiguous_splits": 0,
        "dropped_cutaways": [],
    }

    beats: list[CutBeat] = []
    used_sentences: dict[str, str] = {}
    speech_in_seconds: list[str] = []

    for index, seg in enumerate(plan_obj.segments):
        label = f"segment #{index + 1}"
        if seg.sentences:
            valid_ids = _valid_sentences(seg, sentence_index, used_sentences, label, stats)
            if not valid_ids:
                continue
            beats += _speech_beats(
                seg, valid_ids, sentence_index, clips, default_fit, stats
            )
            continue

        clip_id = str(seg.clip)
        clip = clips.get(clip_id)
        if clip is None:
            stats["dropped_unknown_clip"].append(clip_id or f"#{index}")
            continue
        if not seg.mute_source and _has_words(project, clip_id, float(seg.in_), float(seg.out)):
            speech_in_seconds.append(
                f"{label}: {clip_id} {seg.in_:.2f}-{seg.out:.2f} "
                f"(role {seg.role or '—'}, notes {seg.notes or '—'})"
            )
            continue
        beats += _broll_beats(
            seg, clip_id, clip, entries.get(clip_id, {}), default_fit, min_shot, stats
        )

    if speech_in_seconds:
        raise SpeechSecondsError(speech_in_seconds)

    cut = Cut(
        fps=fps,
        width=width,
        height=height,
        language=project.language,
        beats=beats,
        mute_ranges=_build_mute_ranges(plan_obj, footage_log, clips),
    )
    cut.renumber()
    if cut.beats:
        # Nothing to fade in from at the very top of the programme.
        cut.beats[0].transition_in = Transition(type="cut", duration=0.0)

    # Everything below is positioned by beat, so the geometry has to exist
    # first: one resolve pass answers "what is playing at 41.2 s" for the
    # captions, the chapters, the markers and the music beds alike.
    spans, issues = beat_spans(project, cut)
    total = max((end for _start, end in spans.values()), default=0.0)
    stats["validation_issues"] = [str(issue) for issue in issues]
    for beat in cut.beats:
        span = spans.get(beat.uid)
        if span is not None and beat.transition_in.duration > span[1] - span[0]:
            beat.transition_in = Transition(type="cut", duration=0.0)

    cut.captions = _build_captions(plan_obj, cut, spans, total)
    cut.music = _build_music(plan_obj, cut, spans, total, cfg)
    cut.chapters = _build_chapters(plan_obj, cut, spans, total)
    cut.markers = _build_markers(cfg, cut, spans, total, plan_obj)

    cut.meta.generated_by = f"plan@{utcnow()}"
    cut.meta.edited_by_human = False
    cut.meta.title_candidates = list(plan_obj.title_candidates)
    cut.meta.notes = plan_obj.story.premise or ""

    stats["beats_out"] = len(cut.beats)
    stats["duration_s"] = total
    return cut, stats


# ----------------------------------------------------------------------
# absolute tracks -> beat references
# ----------------------------------------------------------------------
def _vertical_fit(clip: dict[str, Any], default_fit: str) -> str:
    """Apply the vertical-clip fit rule (playbook §3) to one clip registry entry.

    Kept as a public-ish helper because ``ytedit.ai.voice`` picks the fit of a
    pickup's shots the same way and must not re-derive the rule.
    """
    return default_fit if _clip_is_vertical(clip) else "cover"


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


def _beat_under(
    cut: Cut, spans: Mapping[str, tuple[float, float]], t: float
) -> CutBeat | None:
    """The beat playing at absolute second ``t`` (``None`` for an empty cut)."""
    return beat_at(dict(spans), cut, max(0.0, float(t)))


def _beat_ending_at(
    cut: Cut, spans: Mapping[str, tuple[float, float]], t: float
) -> CutBeat | None:
    """The last beat that is still playing *just before* ``t``.

    A music cue's ``end`` is exclusive — it is the second the next section
    starts on — so a cue ending exactly on a beat boundary must claim the beat
    that finished there, not the one that opens the next chapter. Anywhere
    else this is simply the beat under ``t``.
    """
    beat = _beat_under(cut, spans, t)
    if beat is None:
        return None
    span = spans.get(beat.uid)
    if span is not None and abs(span[0] - t) <= 0.05:
        previous: CutBeat | None = None
        for candidate in cut.beats:
            if candidate.uid == beat.uid:
                return previous or beat
            if candidate.uid in spans:
                previous = candidate
    return beat


def _build_markers(
    cfg: Settings,
    cut: Cut,
    spans: Mapping[str, tuple[float, float]],
    total: float,
    plan_obj: EditPlan,
) -> list[CutMarker]:
    """Structural markers from ``pacing.markers``, pinned to the beat under each.

    The times in the settings (negative ``at`` = from the end) are targets for
    *this* runtime; once the beat under one is known the marker travels with
    it, so a later re-cut moves the "hook" flag with the hook instead of
    leaving it stranded at 0:07.
    """
    markers: list[CutMarker] = []
    seen: set[tuple[str, str]] = set()

    def add(at: float, label: str) -> None:
        beat = _beat_under(cut, spans, at)
        if beat is None:
            return
        key = (beat.uid, label)
        if key in seen:
            return
        seen.add(key)
        markers.append(CutMarker(beat=beat.uid, label=label))

    for spec in cfg.get("pacing.markers", []) or []:
        if not isinstance(spec, dict):
            continue
        at = _num(spec.get("at", 0.0))
        if at < 0:
            at = total + at
        if at < -_EPS or at > total + _EPS:
            continue
        add(max(0.0, at), str(spec.get("label", "")))
    # Subscribe CTA rides at ~30% of runtime unless the planner placed it.
    cta_at = plan_obj.cta.at_s if plan_obj.cta.at_s > 0 else round(total * 0.30, 1)
    if total > 0 and 0 < cta_at < total:
        add(cta_at, "subscribe-cta")
    return markers


def _build_captions(
    plan_obj: EditPlan,
    cut: Cut,
    spans: Mapping[str, tuple[float, float]],
    total: float,
) -> list[CutCaption]:
    """Clamp and de-overlap the planner's captions, then anchor them to beats.

    The planner answers in absolute seconds, which is the only frame it can
    reason in, but a caption stored that way drifts the moment anything
    upstream of it changes length. Overlap is resolved here, in absolute time,
    where "which card is on screen" is still meaningful; what gets stored is
    the beat plus an offset into it.
    """
    ordered = sorted(plan_obj.captions, key=lambda c: _num(c.at))
    placed: list[tuple[float, float, PlanCaption]] = []
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
        if placed and at < placed[-1][1] - _EPS and placed[-1][2].style != "subtitle":
            # Trim the previous card rather than dropping either of them.
            prev_at, _prev_end, prev_item = placed[-1]
            trimmed = round(max(prev_at + 0.4, at - 0.05), 3)
            if trimmed <= prev_at:
                placed.pop()
            else:
                placed[-1] = (prev_at, trimmed, prev_item)
        placed.append((at, end, item))

    out: list[CutCaption] = []
    for at, end, item in placed:
        beat = _beat_under(cut, spans, at)
        if beat is None:
            continue
        span = spans[beat.uid]
        style = item.style if item.style in ("location", "hook", "subtitle") else "location"
        position = item.position if item.position in (
            "lower-left", "lower-right", "lower-center", "center",
            "upper-left", "upper-right", "upper-center",
        ) else "lower-left"
        out.append(
            CutCaption(
                id="",
                beat=beat.uid,
                offset=round(max(0.0, at - span[0]), 3),
                duration=round(max(0.4, end - at), 3),
                text=(item.text or "").strip(),
                style=style,
                position=position,  # type: ignore[arg-type]
            )
        )
    for i, caption in enumerate(out):
        caption.id = f"t{i + 1:03d}"
    return out


def _build_music(
    plan_obj: EditPlan,
    cut: Cut,
    spans: Mapping[str, tuple[float, float]],
    total: float,
    cfg: Settings,
) -> list[CutMusicCue]:
    """Convert the planner's cue sheet into inclusive beat ranges.

    A bed that is stored as "b012 through b031" keeps covering the section it
    was written for when the cut around it moves; the files themselves are
    generated later by the music stage from these very ids.
    """
    amount = _num(cfg.get("ducking.amount_db", -12.0), -12.0)
    attack = _num(cfg.get("ducking.attack", 0.15), 0.15)
    release = _num(cfg.get("ducking.release", 0.6), 0.6)
    order = {beat.uid: i for i, beat in enumerate(cut.beats)}

    out: list[CutMusicCue] = []
    cursor = 0.0
    for cue in sorted(plan_obj.music_cues, key=lambda c: _num(c.at)):
        at = max(0.0, _num(cue.at))
        end = _num(cue.end, at + _num(cue.length_s, 60.0))
        if end <= at:
            end = at + max(10.0, _num(cue.length_s, 60.0))
        if at >= total - _EPS:
            continue
        end = min(end, total)
        at = max(at, cursor)
        if end - at < 5.0:
            continue
        first = _beat_under(cut, spans, at)
        last = _beat_ending_at(cut, spans, end)
        if first is None or last is None:
            continue
        if order.get(last.uid, 0) < order.get(first.uid, 0):
            last = first
        cursor = end
        out.append(
            CutMusicCue(
                id="",
                file="",
                **{"from": first.uid},
                to=last.uid,
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


def _build_chapters(
    plan_obj: EditPlan,
    cut: Cut,
    spans: Mapping[str, tuple[float, float]],
    total: float,
) -> list[CutChapter]:
    """Enforce YouTube's chapter rules against the *resolved* beat starts.

    "First at 0:00, at least 10 s apart, none in the last 10 s" is a rule about
    what the viewer sees, so it is checked on where each chapter's beat
    actually lands — the planner's own guess at a timecode is only used to pick
    the beat.
    """
    out: list[CutChapter] = []
    starts: list[float] = []
    for item in sorted(plan_obj.chapters, key=lambda c: _num(c.at)):
        title = (item.title or "").strip()
        if not title:
            continue
        beat = _beat_under(cut, spans, max(0.0, _num(item.at)))
        if beat is None:
            continue
        start = spans[beat.uid][0]
        if start > max(0.0, total - 10.0) + _EPS:
            continue
        if any(chapter.beat == beat.uid for chapter in out):
            continue
        if starts and start - starts[-1] < 10.0:
            continue
        out.append(CutChapter(beat=beat.uid, title=title))
        starts.append(start)
    if out and cut.beats and out[0].beat != cut.beats[0].uid:
        # YouTube ignores a chapter list whose first entry is not 0:00.
        out[0].beat = cut.beats[0].uid
    return out


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
def spans_of(timeline: Timeline) -> dict[str, tuple[float, float]]:
    """``beat uid -> (start, end)`` read back off a resolved timeline.

    The same answer :func:`ytedit.cut.beat_spans` gives, without paying for a
    second resolve pass: the plan stage already has the timeline in hand.
    """
    spans: dict[str, tuple[float, float]] = {}
    for pos in timeline.segment_positions():
        uid = pos.segment.beat
        if not uid:
            continue
        start, _end = spans.get(uid, (pos.start, pos.end))
        spans[uid] = (start, pos.end)
    return spans


def _beat_script(
    beat: CutBeat, sentence_index: Mapping[str, dict[str, Any]]
) -> str:
    """``c048#3–c048#5 "first words…"`` for a speech beat, ``""`` otherwise."""
    if not beat.sentences:
        return ""
    script = f"`{beat.sentences[0]}`" + (
        f"–`{beat.sentences[-1]}`" if len(beat.sentences) > 1 else ""
    )
    words = str((sentence_index.get(beat.sentences[0]) or {}).get("text", "")).split()
    if words:
        script += " “" + " ".join(words[:6]) + ("…" if len(words) > 6 else "") + "”"
    return script


def render_edit_plan_md(
    plan_obj: EditPlan,
    cut: Cut,
    timeline: Timeline,
    project: Project,
    stats: dict[str, Any],
    warnings: Sequence[str],
    sentence_index: Mapping[str, dict[str, Any]] | None = None,
) -> str:
    """Render the human-readable ``plan/edit_plan.md``.

    Args:
        plan_obj: The planner's proposal, for everything the cut does not
            carry (story beats, risks, thumbnails, narration requests).
        cut: The cut that was built from it — the beat table is the report.
        timeline: That cut resolved, which is where every timecode comes from.
        project: The owning project.
        stats: What :func:`build_cut` recorded about its own decisions.
        warnings: :func:`pacing_report` output.
        sentence_index: The flat sentence catalogue (``<clip>#<n>`` -> sentence
            dict, see ``ytedit/ai/sentences.py``) used to show the first words
            of a speech beat's script next to its sentence ids. Omitted
            (``None``) just skips that text — the ids alone still print.
    """
    sentence_index = sentence_index or {}
    total = timeline.duration()
    spans = spans_of(timeline)
    lines: list[str] = [
        f"# Edit plan — {project.slug}",
        "",
        f"Generated: {utcnow()} · language: `{project.language}` · "
        f"estimated runtime: **{_mmss(total)}** "
        f"({len(cut.beats)} beats, {len(timeline.tracks.video)} shots, "
        f"{len(timeline.clip_ids())} clips)",
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
    for marker in cut.markers:
        beat = cut.beat_by_ref(marker.beat)
        span = spans.get(marker.beat)
        where = f"`{beat.id}` / `{beat.clip or beat.file or '—'}`" if beat else "**no beat**"
        when = _mmss(span[0]) if span else "—"
        lines.append(f"- **{when}** {marker.label} → {where}")
    lines.append("")

    # -- beat table ------------------------------------------------------
    lines += [
        "## Beats",
        "",
        "| beat | at | kind | clip | source | dur | role | fit | shots | notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for beat in cut.beats:
        span = spans.get(beat.uid)
        at = _mmss(span[0]) if span else "—"
        duration = f"{span[1] - span[0]:.1f}s" if span else "**unresolved**"
        if beat.kind == "speech":
            source = _beat_script(beat, sentence_index)
            if beat.words:
                source = f"words {beat.words[0]}–{beat.words[1]}"
            camera = "" if beat.on_camera else " 🎙"
        elif beat.kind == "voice":
            source = f"`{beat.file}`"
            camera = " 🎙"
        else:
            source = f"{_num(beat.in_):.1f}–{_num(beat.out):.1f}"
            camera = " 🔇" if beat.audio == "mute" else ""
        shots = ", ".join(
            f"`{shot.clip}`" + (f" after {shot.after}" if shot.after else "")
            for shot in beat.shots
        ) or "—"
        lines.append(
            f"| `{beat.id}` | {at} | {beat.kind} | `{beat.clip or '—'}` | {source} | "
            f"{duration} | {beat.role or '—'}{camera} | {beat.transform.fit} | "
            f"{shots} | {_md_cell(beat.notes)} |"
        )
    lines.append("")

    # -- captions --------------------------------------------------------
    if cut.captions:
        lines += [
            "## Captions",
            "",
            "| at | beat | dur | style | text |",
            "|---|---|---|---|---|",
        ]
        for caption in cut.captions:
            beat = cut.beat_by_ref(caption.beat)
            span = spans.get(caption.beat)
            at = _mmss(span[0] + caption.offset) if span else "—"
            lines.append(
                f"| {at} | `{beat.id if beat else caption.beat}` +{caption.offset:.1f}s | "
                f"{caption.duration:.1f}s | {caption.style} | {_md_cell(caption.text)} |"
            )
        lines.append("")

    # -- music cue sheet --------------------------------------------------
    lines += ["## Music cue sheet", ""]
    if cut.music:
        lines += ["| cue | section | mood / style | beats | at–end | gain | duck |",
                  "|---|---|---|---|---|---|---|"]
        for cue, planned in zip(cut.music, plan_obj.music_cues):
            first, last = cut.beat_by_ref(cue.from_), cut.beat_by_ref(cue.to)
            start = spans.get(cue.from_)
            end = spans.get(cue.to)
            when = (
                f"{_mmss(start[0])}–{_mmss(end[1])}" if start and end else "—"
            )
            lines.append(
                f"| `{cue.id}` | {planned.section or '—'} | "
                f"{_md_cell(planned.mood or planned.style)} | "
                f"`{first.id if first else cue.from_}`–`{last.id if last else cue.to}` | "
                f"{when} | {cue.gain_db:.0f} dB | {cue.duck.amount_db:.0f} dB |"
            )
    else:
        lines.append("_No music cues planned._")
    lines.append("")

    # -- mute ranges ------------------------------------------------------
    if cut.mute_ranges:
        lines += ["## Mute ranges (source audio, clip time)", ""]
        for mute in cut.mute_ranges:
            lines.append(
                f"- `{mute.clip}` {mute.s:.1f}–{mute.e:.1f}s @ {mute.gain_db:.0f} dB — {mute.reason}"
            )
        lines.append("")

    # -- chapters ---------------------------------------------------------
    if cut.chapters:
        lines += ["## Chapters", ""]
        for chapter in cut.chapters:
            beat = cut.beat_by_ref(chapter.beat)
            span = spans.get(chapter.beat)
            when = _mmss(span[0]) if span else "—"
            lines.append(
                f"- {when} ({'`' + beat.id + '`' if beat else chapter.beat}) {chapter.title}"
            )
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
        f"- segments proposed: {stats.get('segments_in', 0)} → beats: "
        f"{stats.get('beats_out', 0)}",
        f"- clips with instruction ranges excised: {stats.get('instruction_cuts', 0)}",
        f"- clips with rejected takes excised: {stats.get('take_cuts', 0)}",
        f"- B-roll segments split around excised ranges: {stats.get('split_segments', 0)}",
        f"- voice-over (off-camera) segments: {stats.get('voice_over_segments', 0)}",
    ]
    for key, label in (
        ("dropped_unknown_clip", "dropped (unknown clip)"),
        ("dropped_empty", "dropped (in >= out)"),
        ("dropped_too_short", "dropped (below min shot length)"),
        ("clamped", "clamped to clip duration"),
        ("vertical_fixed", "vertical clips set to blur-fill"),
        ("validation_issues", "resolver findings"),
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
        ("excluded_sentence_refs", "instruction ids dropped"),
        ("flagged_sentence_refs", "retake/duplicate ids kept — check these"),
        ("dropped_cutaways", "cutaways dropped (bad clip/range or unmatched sentence)"),
        ("voice_over_split", "voice-over runs left without picture cuts"),
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


def _existing_cut_is_human_edited(project: Project) -> bool:
    """Has the user (or the web editor) touched ``plan/cut.json``?

    The flag lives in the cut, not in the derived timeline: ``timeline.json``
    is regenerated from scratch by the resolver and carries no authorship.
    """
    path = cut_path(project)
    if not path.exists():
        return False
    try:
        return bool(load_cut(path).meta.edited_by_human)
    except Exception as exc:  # a broken cut must not block a re-plan
        log.warning("could not read %s: %s", path, exc)
        return False


def plan(
    project: Project,
    force: bool = False,
    notes: str | None = None,
    max_tokens: int | None = None,
    from_response: bool = False,
) -> dict[str, Any]:
    """Run the edit-planning stage.

    Writes ``plan/cut.json`` — the source of truth — and, when it resolves
    cleanly, the derived ``plan/timeline.json`` next to the two Markdown
    reports the user actually reads.

    Args:
        project: Project whose ``analysis/footage_log.json`` is ready.
        force: Overwrite ``plan/cut.json`` even when a human edited it.
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
            (``EditPlan.from_llm`` -> :func:`build_cut` -> resolve -> artifact
            writing, including the human-edited-cut guard) runs exactly as it
            would for a fresh planner answer, except that there is no prompt to
            retry against. No cost is logged.

    Returns:
        ``{"cut", "timeline", "edit_plan", "markdown", "narration", "draft",
        "duration_s", "beats", "segments", "warnings", "stats", "cost_usd",
        "finish_reason", "completion_tokens"}``.

    Raises:
        PlanError: When the footage log is missing, the planner returns
            something unusable (including speech given as raw seconds twice in
            a row — see :class:`SpeechSecondsError`), or (``from_response``)
            there is no previous ``plan/planner_response.json`` to rebuild
            from.
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
    if project.timeline_file.exists():
        try:
            existing = Timeline.load(project.timeline_file)
        except Exception as exc:  # a broken timeline must not block a re-plan
            log.warning("could not read %s: %s", project.timeline_file, exc)
    human_edited = _existing_cut_is_human_edited(project)

    project.set_stage(STAGE, "running")
    try:
        raw_path = project.plan_dir / "planner_response.json"

        def ask(retry_note: str | None = None) -> dict[str, Any]:
            """One planner round trip, dumped to disk before anything parses it."""
            answer = _ask_planner(
                project=project,
                settings=settings,
                footage_log=footage_log,
                existing=existing,
                clip_count=len(entries),
                total_minutes=total_seconds / 60.0,
                notes=notes,
                retry_note=retry_note,
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
                json.dumps(answer["json"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return answer

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
            result = ask()

        spent = float(result["cost_usd"])
        try:
            plan_obj = EditPlan.from_llm(result["json"])
            cut, stats = build_cut(plan_obj, project, footage_log, settings)
        except SpeechSecondsError as exc:
            # Speech given in seconds cannot be turned into a beat without
            # guessing where a word starts, and guessing is what cut v2 exists
            # to stop. Show the model its own offending segments and ask once
            # more; a second failure is a prompt problem, not a fluke.
            if from_response:
                raise
            log.warning(
                "[clip]%s[/] the planner gave %d speech segment(s) as raw seconds — "
                "asking it once more with the offending segments named",
                project.slug,
                len(exc.segments),
            )
            result = ask(retry_note=exc.prompt_note())
            spent += float(result["cost_usd"])
            plan_obj = EditPlan.from_llm(result["json"])
            cut, stats = build_cut(plan_obj, project, footage_log, settings)
        result["cost_usd"] = round(spent, 6)

        if not cut.beats:
            truncated = (
                " The reply hit the output-token cap (finish_reason=length), so it "
                "stops mid-JSON: raise plan.max_tokens or plan in two passes."
                if result.get("finish_reason") == "length"
                else ""
            )
            raise PlanError(
                f"the planner ({result['model']}) produced no usable beats "
                f"({len(plan_obj.segments)} segments proposed, "
                f"{len(stats.get('dropped_unknown_clip', []))} referenced unknown clips, "
                f"{len(stats.get('dropped_empty', []))} were empty) — its raw answer is in "
                f"{raw_path}.{truncated}"
            )

        # --- write artifacts -------------------------------------------
        project.plan_dir.mkdir(parents=True, exist_ok=True)
        write_to_draft = human_edited and not force
        target = cut_path(project).with_name("cut.draft.json" if write_to_draft else "cut.json")
        save_cut(cut, target)
        if write_to_draft:
            log.warning(
                "[clip]%s[/] cut.json is human-edited — wrote the new draft to %s "
                "(diff it, or re-run with --force to overwrite)",
                project.slug,
                target.name,
            )

        timeline, issues = resolve_verbose(project, cut)
        errors = [issue for issue in issues if issue.severity == "error"]
        stats["validation_issues"] = [str(issue) for issue in issues]
        if write_to_draft:
            timeline_path = project.timeline_file  # untouched: the draft is not the cut
        elif errors:
            # A timeline written from a cut that does not validate would hide
            # the breakage behind a fresh mtime; leave it stale so `render`
            # re-resolves and refuses out loud.
            for issue in errors:
                log.error("[clip]%s[/] %s", project.slug, issue)
            timeline_path = project.timeline_file
        else:
            timeline_path = timeline.save(project.timeline_file)

        warnings = pacing_report(timeline, settings)
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
            "cut_file": project.rel(target),
            "timeline_file": project.rel(timeline_path),
            "estimated_runtime_s": timeline.duration(),
        }
        project.edit_plan_file.write_text(
            json.dumps(edit_plan_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        md_path = project.plan_dir / "edit_plan.md"
        md_path.write_text(
            render_edit_plan_md(
                plan_obj, cut, timeline, project, stats, warnings, load_sentence_index(project)
            ),
            encoding="utf-8",
        )
        narration_path = project.plan_dir / "narration_requests.md"
        narration_path.write_text(render_narration_requests_md(plan_obj, project), encoding="utf-8")

        summary: dict[str, Any] = {
            "cut": str(target),
            "timeline": str(timeline_path),
            "edit_plan": str(project.edit_plan_file),
            "markdown": str(md_path),
            "narration": str(narration_path),
            "draft": write_to_draft,
            "duration_s": timeline.duration(),
            "beats": len(cut.beats),
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
            beats=len(cut.beats),
            segments=len(timeline.tracks.video),
            draft=write_to_draft,
            warnings=len(warnings),
        )
        log.info(
            "[clip]%s[/] plan: %d beats, %d shots, %s runtime, %d pacing warnings, "
            "%s output tokens, $%.4f",
            project.slug,
            len(cut.beats),
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
    retry_note: str | None = None,
    max_tokens: int = DEFAULT_PLAN_MAX_TOKENS,
    client: OpenRouter | None = None,
) -> dict[str, Any]:
    """Render the plan prompts and call the planner model once.

    Args:
        retry_note: A correction appended to the user prompt after an answer
            that broke a hard rule (see :class:`SpeechSecondsError`) — the
            convention every AI stage here follows: show the model exactly what
            was wrong with its own output and ask for the whole object again.
    """
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
    if retry_note:
        user_prompt += f"\n\n{retry_note.strip()}"

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
    "instruction_ranges",
    "rejected_take_ranges",
    "build_cut",
    "spans_of",
    "subtract_ranges",
    "units_for_ledger",
    "DEFAULT_PLAN_MAX_TOKENS",
    "EditPlan",
    "PlanError",
    "SpeechSecondsError",
    "render_edit_plan_md",
    "render_narration_requests_md",
]
