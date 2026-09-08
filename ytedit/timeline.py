"""The timeline (EDL) data model — ``plan/timeline.json``.

The timeline is the single source of truth for rendering. Video segments are
laid end to end (an ``xfade`` transition overlaps the previous segment by its
duration); voice, music, captions, sfx, markers and chapters all use absolute
timeline time. ``mute_ranges`` are in *clip* time and apply wherever that clip
range is used.

Segment geometry is computed in **whole frames** at the timeline ``fps``: a
segment's ``in``/``out`` may be any float, but it renders as
:meth:`VideoSegment.frames` frames and the next segment starts right after
them. :meth:`Timeline.segment_positions` and :meth:`Timeline.duration` report
those frame-snapped values, so absolute placements match the rendered file.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import secrets
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, NamedTuple, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .log import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from .project import Project

log = get_logger(__name__)

FitMode = Literal["cover", "contain", "blur-fill", "crop-pan"]
TransitionType = Literal["cut", "fade", "xfade"]
DuckMode = Literal["auto", "manual", "off"]
CaptionPosition = Literal[
    "lower-left", "lower-right", "lower-center", "center", "upper-left", "upper-right",
    "upper-center",
]


class _Model(BaseModel):
    """Base model: allows population by field name and keeps unknown keys."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")


def frames_for(seconds: float, fps: float) -> int:
    """Round a duration to the nearest whole frame (never negative).

    Half a frame rounds up, so the result is stable and matches the JavaScript
    mirror in the web editor (``Math.floor(x + 0.5)``).
    """
    if fps <= 0:
        return 0
    return max(0, int(math.floor(float(seconds) * float(fps) + 0.5)))


def frames_to_seconds(frames: int, fps: float) -> float:
    """Convert a whole number of frames back to seconds (6 decimals)."""
    if fps <= 0:
        return 0.0
    return round(int(frames) / float(fps), 6)


class Transform(_Model):
    """How a source frame is fitted into the canvas."""

    fit: FitMode = "cover"
    zoom: float = 1.0
    #: Ken Burns start/end rectangles as (x, y, w, h) fractions, for ``crop-pan``.
    pan_from: tuple[float, float, float, float] | None = None
    pan_to: tuple[float, float, float, float] | None = None


class Transition(_Model):
    """Transition into a segment. ``cut`` has zero duration."""

    type: TransitionType = "cut"
    duration: float = 0.0
    #: ffmpeg ``xfade`` transition name when ``type == "xfade"``.
    name: str = "fade"

    def frames(self, fps: float) -> int:
        """Overlap length in whole frames at ``fps`` (0 for a cut)."""
        if self.type == "cut":
            return 0
        return frames_for(self.duration, fps)


class Duck(_Model):
    """Music ducking behaviour under speech."""

    mode: DuckMode = "auto"
    amount_db: float = -12.0
    attack: float = 0.15
    release: float = 0.6
    #: Explicit ``(start, end)`` ranges in timeline time when ``mode == "manual"``.
    ranges: list[tuple[float, float]] = Field(default_factory=list)


class AudioFrom(_Model):
    """Audio-only override: where a segment's *sound* comes from.

    A cutaway placed in the middle of a take keeps its own picture but should
    not interrupt the narration underneath it. Setting ``audio_from`` makes the
    render take the segment's audio from ``clip[in, out]`` instead of the
    segment's own clip range, trimmed/padded to the segment's exact frame count
    (see :func:`ytedit.ai.overlay.overlay_cutaways`).
    """

    clip: str
    in_: float = Field(0.0, alias="in", serialization_alias="in")
    out: float = 0.0

    @property
    def duration(self) -> float:
        """Length of the borrowed audio range in seconds."""
        return max(0.0, self.out - self.in_)


class VideoSegment(_Model):
    """One cut from a normalized source clip."""

    id: str
    #: Stable identity, assigned once (random on creation) and NEVER
    #: reassigned or renumbered by any pass — unlike :attr:`id`, the display
    #: label that ``ytedit.ai.plan``/``ytedit tidy`` renumber to ``s001,
    #: s002, ...`` whenever a segment is inserted, dropped or merged.
    #: :class:`VoiceAnchor` / :class:`CaptionAnchor` key on this so a
    #: narration pickup or caption keeps following its picture even after
    #: ids shift under it — see :meth:`Timeline.resolve_anchors`. A file
    #: saved before this field existed gets one filled in deterministically
    #: on load (see :func:`ensure_segment_uids`) rather than a fresh random
    #: one every time it is read.
    uid: str = Field(default_factory=lambda: secrets.token_hex(4))
    clip: str
    in_: float = Field(0.0, alias="in", serialization_alias="in")
    out: float = 0.0
    role: str = ""
    transform: Transform = Field(default_factory=Transform)
    grade: str = "default"
    transition_in: Transition = Field(default_factory=Transition)
    speed: float = 1.0
    mute_source: bool = False
    source_audio_gain_db: float = 0.0
    notes: str = ""
    #: Take the audio from another clip range instead of this segment's own
    #: (``mute_source`` still wins and renders silence).
    audio_from: AudioFrom | None = None
    #: Script-first speech: the sentence ids (``<clip>#<n>``, see
    #: ``ytedit/ai/sentences.py``) this segment's ``in``/``out`` were derived
    #: from. Empty for B-roll/legacy raw-seconds segments and for the
    #: picture-only side of an overlay cutaway.
    sentence_ids: list[str] = Field(default_factory=list)

    @property
    def audio_source(self) -> tuple[str, float, float]:
        """``(clip, in, out)`` the segment's audio is actually read from."""
        if self.audio_from is not None:
            return self.audio_from.clip, self.audio_from.in_, self.audio_from.out
        return self.clip, self.in_, self.out

    @property
    def source_duration(self) -> float:
        """Length of the cut in source time (seconds)."""
        return max(0.0, self.out - self.in_)

    @property
    def duration(self) -> float:
        """Length of the cut on the timeline, after ``speed``."""
        speed = self.speed if self.speed > 0 else 1.0
        return self.source_duration / speed

    def frames(self, fps: float) -> int:
        """Whole frames the segment occupies when rendered at ``fps``.

        The render cuts every segment on frame boundaries, so this — not the
        fractional :attr:`duration` — determines where the next segment
        starts. A segment always renders at least one frame.
        """
        return max(1, frames_for(self.duration, fps))

    def rendered_duration(self, fps: float) -> float:
        """Length of the cut once snapped to whole frames at ``fps``."""
        return frames_to_seconds(self.frames(fps), fps)


class AnchorSignature(_Model):
    """A fingerprint of the segment a :class:`VoiceAnchor`/:class:`CaptionAnchor`
    pointed at when it was created, used to repair the anchor if its ``uid``
    is ever lost or stops resolving (see :meth:`Timeline.resolve_anchors`).
    """

    clip: str = ""
    in_: float = Field(0.0, alias="in", serialization_alias="in")


class VoiceAnchor(_Model):
    """Pins a :class:`VoiceItem` to a video segment instead of absolute time.

    ``offset`` seconds after the start of the anchored segment (its
    :attr:`SegmentPosition.start`, see :meth:`Timeline.segment_positions`).
    Re-resolved by :meth:`Timeline.resolve_voice_anchors` whenever the segment
    may have moved — speech padding, sentence snapping, overlay cutaways,
    audio dedupe, or a hand edit in the web editor all shift absolute time,
    but a pickup anchored this way follows its picture instead of drifting.

    ``segment`` is the segment's display ``id`` at the time it was last
    resolved — kept for readability (log lines, the web editor) — but
    resolution itself uses ``uid`` when present, since ``id`` gets
    renumbered whenever segments are inserted, dropped or merged elsewhere
    (see ``ytedit/ai/plan.py``). ``signature`` records that segment's
    ``clip``/``in`` at anchoring time, so a lost or stale ``uid`` can be
    repaired by finding the (hopefully unique) segment still carrying that
    fingerprint, rather than silently resolving to the wrong picture.
    """

    segment: str
    offset: float = 0.0
    uid: str | None = None
    signature: AnchorSignature | None = None


class CaptionAnchor(_Model):
    """Pins a :class:`Caption` to a video segment instead of absolute time.

    ``offset`` seconds after the start of the anchored segment (its
    :attr:`SegmentPosition.start`, see :meth:`Timeline.segment_positions`).
    Re-resolved by :meth:`Timeline.resolve_anchors` whenever the segment may
    have moved — speech padding, sentence snapping, overlay cutaways, audio
    dedupe, or a hand edit in the web editor all shift absolute time, but a
    location card or hook line anchored this way follows its picture instead
    of drifting under the wrong shot (see ``ytedit/ai/locations.py``).

    See :class:`VoiceAnchor` for why ``uid``/``signature`` exist alongside
    the readable ``segment`` display id.
    """

    segment: str
    offset: float = 0.0
    uid: str | None = None
    signature: AnchorSignature | None = None


class VoiceItem(_Model):
    """A narration pickup placed at an absolute timeline position.

    ``at``/``end`` are always the resolved absolute values render and the
    editor read. When :attr:`anchor` is set, they are recomputed from it by
    :meth:`Timeline.resolve_voice_anchors` rather than edited directly.
    """

    id: str
    file: str
    at: float = 0.0
    gain_db: float = 0.0
    #: Optional explicit end; when absent the file length is used at render time.
    end: float | None = None
    #: When set, this item is pinned to a video segment instead of absolute time.
    anchor: VoiceAnchor | None = None


class MusicCue(_Model):
    """A music bed placed at an absolute timeline position."""

    id: str
    file: str
    at: float = 0.0
    end: float = 0.0
    gain_db: float = -18.0
    fade_in: float = 2.0
    fade_out: float = 3.0
    duck: Duck = Field(default_factory=Duck)

    @property
    def duration(self) -> float:
        """Length of the cue in seconds."""
        return max(0.0, self.end - self.at)


class Caption(_Model):
    """A burned-in text card or subtitle cue."""

    id: str
    at: float = 0.0
    end: float = 0.0
    text: str = ""
    style: str = "location"
    position: CaptionPosition = "lower-left"
    #: When set, this cue is pinned to a video segment instead of absolute
    #: time (see :class:`CaptionAnchor`).
    anchor: CaptionAnchor | None = None

    @property
    def duration(self) -> float:
        """Length of the cue in seconds."""
        return max(0.0, self.end - self.at)


class SfxItem(_Model):
    """A one-shot sound effect."""

    id: str
    file: str
    at: float = 0.0
    gain_db: float = -6.0
    end: float | None = None


class MuteRange(_Model):
    """Source-audio attenuation, expressed in **clip** time."""

    clip: str
    s: float = 0.0
    e: float = 0.0
    gain_db: float = -60.0
    reason: str = ""


class Marker(_Model):
    """A structural beat marker (hook, promise, re-engagement, ...)."""

    at: float = 0.0
    label: str = ""


class Chapter(_Model):
    """A YouTube chapter."""

    at: float = 0.0
    title: str = ""


class Meta(_Model):
    """Timeline provenance and editorial metadata."""

    title_candidates: list[str] = Field(default_factory=list)
    generated_by: str = ""
    edited_by_human: bool = False
    notes: str = ""
    #: Anchors :meth:`Timeline.resolve_anchors` could not resolve (uid not
    #: found and no unique signature match either) on its most recent call —
    #: cleared and rebuilt every time it runs. See ``ytedit qc`` rule 34.
    anchor_issues: list[str] = Field(default_factory=list)


class Tracks(_Model):
    """The five parallel tracks of the EDL."""

    video: list[VideoSegment] = Field(default_factory=list)
    voice: list[VoiceItem] = Field(default_factory=list)
    music: list[MusicCue] = Field(default_factory=list)
    captions: list[Caption] = Field(default_factory=list)
    sfx: list[SfxItem] = Field(default_factory=list)


class SegmentPosition(NamedTuple):
    """A video segment with its absolute timeline placement."""

    segment: VideoSegment
    start: float
    end: float


class Timeline(_Model):
    """The full EDL document (``plan/timeline.json``)."""

    version: int = 1
    fps: int = 30
    width: int = 1920
    height: int = 1080
    language: str = "pl"
    tracks: Tracks = Field(default_factory=Tracks)
    mute_ranges: list[MuteRange] = Field(default_factory=list)
    markers: list[Marker] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    meta: Meta = Field(default_factory=Meta)

    # ------------------------------------------------------------------
    # io
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> "Timeline":
        """Read a timeline JSON file.

        Args:
            path: Path to ``timeline.json``.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ensure_segment_uids(data)
        return cls.model_validate(data)

    @classmethod
    def model_validate_migrated(cls, data: dict[str, Any]) -> "Timeline":
        """Like :meth:`model_validate`, but fills any missing segment ``uid``
        first (see :func:`ensure_segment_uids`).

        Use this instead of ``model_validate`` for a raw dict that might
        predate ``uid`` and did not come through :meth:`load` — e.g. the web
        editor validating/saving a posted payload.
        """
        ensure_segment_uids(data)
        return cls.model_validate(data)

    def save(self, path: Path | str) -> Path:
        """Write the timeline atomically and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".timeline-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return target

    def to_dict(self) -> dict[str, Any]:
        """Serialize with JSON aliases (``in_`` becomes ``in``)."""
        return self.model_dump(by_alias=True, mode="json")

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def segment_positions(self, fade_overlaps: bool = False) -> list[SegmentPosition]:
        """Return every video segment with its absolute start/end.

        Placement is computed in whole frames at :attr:`fps` — each segment
        occupies :meth:`VideoSegment.frames` frames and an overlapping
        transition removes :meth:`Transition.frames` — then converted back to
        seconds. That is exactly how the render lays the programme out, so a
        caption, voice pickup or music cue placed at ``positions[i].start``
        lands on the first frame of segment ``i``.

        A segment whose ``transition_in`` is an ``xfade`` overlaps the previous
        segment by the transition length; ``cut`` and ``fade`` do not, unless
        ``fade_overlaps`` is set (the render treats ``fade`` as an ``xfade``
        too — see :func:`ytedit.media.render.render_positions`).
        """
        overlapping = ("fade", "xfade") if fade_overlaps else ("xfade",)
        fps = self.fps
        out: list[SegmentPosition] = []
        cursor = 0
        previous = 0
        for i, seg in enumerate(self.tracks.video):
            length = seg.frames(fps)
            overlap = 0
            if i > 0 and seg.transition_in.type in overlapping:
                overlap = min(seg.transition_in.frames(fps), previous, length)
            start = max(0, cursor - overlap)
            end = start + length
            out.append(
                SegmentPosition(seg, frames_to_seconds(start, fps), frames_to_seconds(end, fps))
            )
            cursor = end
            previous = length
        return out

    def frame_count(self, fade_overlaps: bool = False) -> int:
        """Programme length in whole frames — the end of the video track."""
        positions = self.segment_positions(fade_overlaps=fade_overlaps)
        return frames_for(positions[-1].end, self.fps) if positions else 0

    def duration(self) -> float:
        """Programme length in seconds — the end of the video track.

        The video track defines the length of the render; audio or caption
        items that reach past it are trimmed. Use :meth:`content_end` to see how
        far any track extends.
        """
        positions = self.segment_positions()
        return round(positions[-1].end if positions else 0.0, 6)

    def content_end(self) -> float:
        """The furthest point reached by any track (video, music, voice, captions)."""
        tails: list[float] = [self.duration()]
        tails += [m.end for m in self.tracks.music]
        tails += [c.end for c in self.tracks.captions]
        tails += [v.end for v in self.tracks.voice if v.end is not None]
        tails += [s.end for s in self.tracks.sfx if s.end is not None]
        return round(max(tails) if tails else 0.0, 6)

    def segment_at(self, t: float) -> SegmentPosition | None:
        """Return the segment covering timeline time ``t``, if any."""
        for pos in self.segment_positions():
            if pos.start <= t < pos.end:
                return pos
        return None

    def resolve_anchors(self) -> int:
        """Recompute ``at``/``end`` for every anchored voice item and caption.

        For each :class:`VoiceItem` or :class:`Caption` carrying an anchor
        (:class:`VoiceAnchor` / :class:`CaptionAnchor`), the anchored segment
        is located by ``anchor.uid`` first (never renumbered — see
        :attr:`VideoSegment.uid`); a legacy anchor with no ``uid`` yet falls
        back to its readable ``segment`` display id. If the ``uid`` lookup
        fails, or resolves to a segment whose ``(clip, in)`` no longer
        matches ``anchor.signature`` within 0.5 s (someone renumbered ids and
        a *different* segment now happens to hold that uid — should not
        normally happen, but this is the safety net for it), a repair is
        attempted by searching every segment for a unique ``(clip, in)``
        match to the signature. Whichever way a segment is found, ``at`` is
        set to ``offset`` seconds after its current start (from
        :meth:`segment_positions`), ``end`` moves with it so the item's
        length never changes (``length = old end - old at`` for a voice
        item, or left ``None`` when it already was; always ``end - at`` for
        a caption), and the anchor's ``uid``/``segment``/``signature`` are
        refreshed to match.

        An item that cannot be resolved at all keeps its current absolute
        time untouched, is logged, and is recorded in
        :attr:`Meta.anchor_issues` (cleared and rebuilt on every call — see
        ``ytedit qc`` rule 34) — better a stale but sane position than a
        crash or a silent teleport to time zero.

        Returns:
            The number of items (voice + captions combined) successfully
            resolved against their anchor (items with no anchor, or one that
            could not be resolved, are not counted).
        """
        positions = self.segment_positions()
        by_uid: dict[str, SegmentPosition] = {}
        by_display_id: dict[str, SegmentPosition] = {}
        for pos in positions:
            if pos.segment.uid:
                by_uid.setdefault(pos.segment.uid, pos)
            by_display_id.setdefault(pos.segment.id, pos)

        self.meta.anchor_issues = []
        resolved = 0

        def _locate(anchor: VoiceAnchor | CaptionAnchor, kind: str, item_id: str) -> SegmentPosition | None:
            pos: SegmentPosition | None = None
            if anchor.uid:
                pos = by_uid.get(anchor.uid)
                sig = anchor.signature
                if pos is not None and sig is not None and (
                    pos.segment.clip != sig.clip or abs(pos.segment.in_ - sig.in_) > 0.5
                ):
                    pos = None  # uid resolved, but no longer the segment we anchored to
            if pos is None and anchor.signature is not None:
                sig = anchor.signature
                candidates = [
                    p for p in positions
                    if p.segment.clip == sig.clip and abs(p.segment.in_ - sig.in_) <= 0.5
                ]
                if len(candidates) == 1:
                    pos = candidates[0]
                    log.warning(
                        "%s %s: anchor uid %r lost or stale — repaired via signature "
                        "(clip=%s in=%.2f) to segment %s",
                        kind, item_id, anchor.uid, sig.clip, sig.in_, pos.segment.id,
                    )
            if pos is None and not anchor.uid and anchor.segment:
                # Legacy anchor predating uid/signature: fall back to the
                # display id it always used. Self-heals into uid+signature
                # below once found, so this branch only fires once per item.
                pos = by_display_id.get(anchor.segment)
            if pos is None:
                ref = anchor.segment or anchor.uid or "?"
                issue = f"{kind} {item_id}: anchor to segment {ref!r} could not be resolved"
                self.meta.anchor_issues.append(issue)
                log.warning(issue)
                return None
            anchor.uid = pos.segment.uid
            anchor.segment = pos.segment.id
            anchor.signature = AnchorSignature(clip=pos.segment.clip, **{"in": pos.segment.in_})
            return pos

        for item in self.tracks.voice:
            if item.anchor is None:
                continue
            pos = _locate(item.anchor, "voice", item.id)
            if pos is None:
                continue
            length = None if item.end is None else item.end - item.at
            item.at = round(pos.start + item.anchor.offset, 3)
            if length is not None:
                item.end = round(item.at + length, 3)
            resolved += 1
        for cap in self.tracks.captions:
            if cap.anchor is None:
                continue
            pos = _locate(cap.anchor, "caption", cap.id)
            if pos is None:
                continue
            length = cap.end - cap.at
            cap.at = round(pos.start + cap.anchor.offset, 3)
            cap.end = round(cap.at + length, 3)
            resolved += 1
        return resolved

    def resolve_voice_anchors(self) -> int:
        """Alias for :meth:`resolve_anchors` (kept for backward compatibility).

        The name predates caption anchoring; every call site now wants both
        tracks resolved together, so this simply forwards.
        """
        return self.resolve_anchors()

    def clip_ids(self) -> list[str]:
        """Distinct source clip ids referenced by the video track, in order."""
        seen: list[str] = []
        for seg in self.tracks.video:
            if seg.clip not in seen:
                seen.append(seg.clip)
        return seen

    # ------------------------------------------------------------------
    # edit API — insert/remove/replace video segments
    # ------------------------------------------------------------------
    def renumber_segment_ids(self) -> None:
        """Reassign every video segment's display ``id`` to ``s001, s002, ...``.

        Never touches :attr:`VideoSegment.uid` — anything anchored by uid
        (see :meth:`resolve_anchors`) is unaffected by a renumbering.
        """
        for i, seg in enumerate(self.tracks.video):
            seg.id = f"s{i + 1:03d}"

    def insert_segments(
        self,
        index: int,
        segments: Sequence[VideoSegment],
        *,
        before: Sequence[SegmentPosition] | None = None,
        renumber: bool = True,
    ) -> None:
        """Splice new segments into the video track at ``index``.

        This is the one supported way to grow the video track: every new
        segment gets a ``uid`` if it doesn't already have one, everything
        downstream — unanchored voice items, captions, music cues, markers
        and chapters — is re-timed by the frame-exact duration the insertion
        adds at that point (the same shift-map machinery
        ``ytedit.ai.tidy.pad_segments_to_speech`` uses for speech padding),
        display ids are renumbered, and every anchor is re-resolved.

        Args:
            index: Position in ``tracks.video`` the new segments are
                inserted before (``len(tracks.video)`` appends).
            segments: The new segments, in order.
            before: The positions to diff against for re-timing, for a
                caller that already captured them earlier — e.g. before a
                resize it applied directly to a surviving segment's
                ``in``/``out`` — so one combined re-time covers both edits,
                exactly like a single before/after diff would have.
                Defaults to a fresh snapshot taken right now.
            renumber: Reassign every segment's display ``id`` afterward
                (default on). Pass ``False`` for a caller that must keep its
                own numbering scheme across the edit (e.g. ``ytedit.ai.voice``,
                which never renumbers segments other stages already anchored
                against).
        """
        if before is None:
            before = self.segment_positions()
        for seg in segments:
            if not seg.uid:
                seg.uid = secrets.token_hex(4)
        self.tracks.video[index:index] = list(segments)
        remap = _shift_map(before, self.segment_positions())
        _retime_absolute_tracks(self, remap)
        if renumber:
            self.renumber_segment_ids()
        self.resolve_anchors()

    def remove_segments(
        self,
        predicate: Callable[[VideoSegment], bool] | Iterable[str],
        *,
        before: Sequence[SegmentPosition] | None = None,
        renumber: bool = True,
    ) -> int:
        """Drop every video segment ``predicate`` accepts, re-timing what follows.

        See :meth:`insert_segments` for ``before``/``renumber``.

        Args:
            predicate: ``callable(segment) -> bool`` (true means drop it), or
                an iterable of ids/uids to drop (matched against either).

        Returns:
            How many segments were dropped.
        """
        if before is None:
            before = self.segment_positions()
        if callable(predicate):
            pred: Callable[[VideoSegment], bool] = predicate
        else:
            keys = set(predicate)
            pred = lambda s: s.uid in keys or s.id in keys  # noqa: E731
        kept = [s for s in self.tracks.video if not pred(s)]
        removed = len(self.tracks.video) - len(kept)
        self.tracks.video = kept
        remap = _shift_map(before, self.segment_positions())
        _retime_absolute_tracks(self, remap)
        if renumber:
            self.renumber_segment_ids()
        self.resolve_anchors()
        return removed

    def replace_segment(
        self,
        uid: str,
        new_segments: Sequence[VideoSegment],
        *,
        before: Sequence[SegmentPosition] | None = None,
        renumber: bool = True,
    ) -> bool:
        """Replace one video segment (by ``uid``) with one or more new ones.

        The usual case is a split in the web editor: the original segment's
        ``uid`` disappears and two fresh ones take its place. See
        :meth:`insert_segments` for ``before``/``renumber``.

        Returns:
            ``False`` when ``uid`` is not found (nothing changed); ``True``
            otherwise.
        """
        idx = next((i for i, s in enumerate(self.tracks.video) if s.uid == uid), None)
        if idx is None:
            return False
        if before is None:
            before = self.segment_positions()
        for seg in new_segments:
            if not seg.uid:
                seg.uid = secrets.token_hex(4)
        self.tracks.video[idx:idx + 1] = list(new_segments)
        remap = _shift_map(before, self.segment_positions())
        _retime_absolute_tracks(self, remap)
        if renumber:
            self.renumber_segment_ids()
        self.resolve_anchors()
        return True

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(
        self,
        project: "Project | None" = None,
        *,
        skip_music: bool = False,
        skip_voice: bool = False,
    ) -> list[str]:
        """Check the timeline for structural problems.

        Args:
            project: When given, segment ``clip`` ids and referenced media files
                are checked against the project's registry and disk, and every
                segment's (and ``audio_from``'s) range is checked against that
                clip's known duration (``0 <= in < out <= duration + 0.05``).
            skip_music: Skip every music-track check (a ``--no-music`` render
                ignores the cues entirely, so a missing music file or bad cue
                is not a reason to refuse).
            skip_voice: Skip every voice-track check, symmetrically.

        Returns:
            A list of human-readable issues; empty means the timeline is sane.
        """
        issues: list[str] = []
        known_clips: set[str] = set()
        clip_records: dict[str, Any] = {}
        if project is not None:
            clip_records = project.load_state().get("clips", {})
            known_clips = set(clip_records)

        def _range_issues(seg_id: str, label: str, clip: str, in_: float, out: float) -> None:
            """Check one ``[in, out)`` range against ``clip``'s known duration.

            Whether the clip has an on-disk normalized source is a render-time
            concern checked separately (see
            :func:`ytedit.media.render.preflight`) — this only looks at the
            duration recorded in the project's clip registry.

            Args:
                label: ``"clip"`` or ``"audio_from clip"`` — how the range is
                    described in the issue text.
            """
            if project is None or not known_clips or clip not in known_clips:
                return
            duration = clip_records.get(clip, {}).get("duration")
            if duration is None:
                return
            dur = float(duration)
            if out > dur + 0.05:
                issues.append(
                    f"{seg_id}: {label} {clip!r} range {in_:.2f}-{out:.2f}s exceeds "
                    f"clip duration {dur:.2f}s"
                )

        # --- video track ------------------------------------------------
        seen_ids: set[str] = set()
        for seg in self.tracks.video:
            if seg.id in seen_ids:
                issues.append(f"duplicate segment id {seg.id!r}")
            seen_ids.add(seg.id)
            if seg.out <= seg.in_:
                issues.append(f"{seg.id}: in ({seg.in_}) >= out ({seg.out})")
            if seg.in_ < 0:
                issues.append(f"{seg.id}: negative in ({seg.in_})")
            if seg.speed <= 0:
                issues.append(f"{seg.id}: speed must be > 0 (got {seg.speed})")
            if project is not None and known_clips and seg.clip not in known_clips:
                issues.append(f"{seg.id}: missing clip {seg.clip!r} in project registry")
            elif seg.out > seg.in_:
                _range_issues(seg.id, "clip", seg.clip, seg.in_, seg.out)
            if seg.audio_from is not None:
                if (
                    project is not None
                    and known_clips
                    and seg.audio_from.clip not in known_clips
                ):
                    issues.append(
                        f"{seg.id}: missing audio_from clip {seg.audio_from.clip!r} "
                        "in project registry"
                    )
                elif seg.audio_from.out <= seg.audio_from.in_:
                    issues.append(
                        f"{seg.id}: audio_from in ({seg.audio_from.in_}) >= "
                        f"out ({seg.audio_from.out})"
                    )
                elif seg.audio_from.in_ < 0:
                    issues.append(f"{seg.id}: negative audio_from in ({seg.audio_from.in_})")
                else:
                    _range_issues(
                        seg.id, "audio_from clip", seg.audio_from.clip,
                        seg.audio_from.in_, seg.audio_from.out,
                    )
            if seg.transition_in.duration < 0:
                issues.append(f"{seg.id}: negative transition duration")
            if seg.transition_in.duration > seg.duration + 1e-6:
                issues.append(
                    f"{seg.id}: transition ({seg.transition_in.duration}s) longer than "
                    f"the segment ({seg.duration:.2f}s)"
                )

        positions = self.segment_positions()
        for prev, cur in zip(positions, positions[1:]):
            if cur.start < prev.start - 1e-6:
                issues.append(f"{cur.segment.id}: starts before {prev.segment.id}")
            overlap = prev.end - cur.start
            if overlap > 1e-6 and cur.segment.transition_in.type != "xfade":
                issues.append(
                    f"{cur.segment.id}: overlaps {prev.segment.id} by {overlap:.3f}s "
                    "without an xfade transition"
                )

        total = self.duration()

        # --- other tracks -----------------------------------------------
        issues += _check_overlaps(
            [(c.id, c.at, c.end) for c in self.tracks.captions if c.style != "subtitle"],
            "caption",
        )
        if not skip_music:
            issues += _check_overlaps(
                [(m.id, m.at, m.end) for m in self.tracks.music], "music cue"
            )

        for cap in self.tracks.captions:
            if cap.end <= cap.at:
                issues.append(f"caption {cap.id}: at ({cap.at}) >= end ({cap.end})")
            if cap.at < -1e-6 or cap.end > total + 1e-6:
                issues.append(
                    f"caption {cap.id}: {cap.at:.2f}-{cap.end:.2f}s outside programme "
                    f"duration (0-{total:.2f}s)"
                )
            if not cap.text.strip():
                issues.append(f"caption {cap.id}: empty text")

        if not skip_music:
            for cue in self.tracks.music:
                if cue.end <= cue.at:
                    issues.append(f"music {cue.id}: at ({cue.at}) >= end ({cue.end})")
                if cue.at > total + 1e-6:
                    issues.append(f"music {cue.id}: starts after the programme ends")
                if project is not None and not (project.path / cue.file).exists():
                    issues.append(f"music {cue.id}: missing file {cue.file}")

        if not skip_voice:
            for voice in self.tracks.voice:
                if voice.at < -1e-6:
                    issues.append(f"voice {voice.id}: negative position")
                if project is not None and not (project.path / voice.file).exists():
                    issues.append(f"voice {voice.id}: missing file {voice.file}")

        for item in self.tracks.sfx:
            if item.at < -1e-6:
                issues.append(f"sfx {item.id}: negative position")
            if project is not None and not (project.path / item.file).exists():
                issues.append(f"sfx {item.id}: missing file {item.file}")

        for mute in self.mute_ranges:
            if mute.e <= mute.s:
                issues.append(f"mute_range on {mute.clip}: s ({mute.s}) >= e ({mute.e})")
            if project is not None and known_clips and mute.clip not in known_clips:
                issues.append(f"mute_range: missing clip {mute.clip!r}")

        for marker in self.markers:
            if marker.at > total + 1e-6:
                issues.append(f"marker {marker.label!r} at {marker.at}s is past the end")

        last = -1.0
        for chapter in self.chapters:
            if chapter.at <= last:
                issues.append(f"chapter {chapter.title!r}: not in ascending order")
            last = chapter.at
        if self.chapters and abs(self.chapters[0].at) > 1e-6:
            issues.append("chapters: YouTube requires the first chapter at 0:00")

        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            issues.append("canvas: fps/width/height must be positive")

        return issues


def _check_overlaps(items: Iterable[tuple[str, float, float]], kind: str) -> list[str]:
    """Return issues for overlapping ``(id, start, end)`` spans."""
    ordered = sorted(items, key=lambda it: it[1])
    issues: list[str] = []
    for (aid, _astart, aend), (bid, bstart, _bend) in zip(ordered, ordered[1:]):
        if bstart < aend - 1e-6:
            issues.append(f"{kind} {bid!r} overlaps {aid!r} by {aend - bstart:.3f}s")
    return issues


# ----------------------------------------------------------------------
# stable segment identity — migration for files predating ``uid``
# ----------------------------------------------------------------------
def _deterministic_uid(clip: str, in_: float, out: float, index: int) -> str:
    """Stable fallback uid for a video segment loaded from a pre-``uid`` file.

    Derived from ``(clip, in, out, index)`` so two loads of the same
    not-yet-migrated file agree on a segment's identity instead of each
    minting a fresh random one (which :class:`VideoSegment`'s default
    factory would otherwise do for every field missing from the JSON).
    """
    seed = f"{clip}|{in_!r}|{out!r}|{index}".encode("utf-8")
    return hashlib.sha1(seed).hexdigest()[:8]


def ensure_segment_uids(data: dict[str, Any]) -> None:
    """Fill a missing ``uid`` on every video segment in raw timeline JSON.

    Mutates ``data`` in place. Used by :meth:`Timeline.load` and
    :meth:`Timeline.model_validate_migrated` (the web editor's timeline
    endpoints go through the latter) so a segment that never had a ``uid`` —
    an old file predating the field, or a brand-new piece a client forgot to
    stamp, such as one half of a split — gets one deterministically instead
    of a fresh random uid on every read.
    """
    tracks = data.get("tracks")
    if not isinstance(tracks, dict):
        return
    segments = tracks.get("video")
    if not isinstance(segments, list):
        return
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict) or seg.get("uid"):
            continue
        try:
            in_ = float(seg.get("in", 0.0) or 0.0)
            out = float(seg.get("out", 0.0) or 0.0)
        except (TypeError, ValueError):  # pragma: no cover - malformed timeline
            in_, out = 0.0, 0.0
        seg["uid"] = _deterministic_uid(str(seg.get("clip", "")), in_, out, i)


# ----------------------------------------------------------------------
# re-timing absolute-time tracks after an edit shifts the video track
# ----------------------------------------------------------------------
def _shift_map(
    before: Sequence[SegmentPosition], after: Sequence[SegmentPosition]
) -> Callable[[float], float]:
    """Build a step function mapping an old absolute time to its new one.

    An edit that grows, shrinks, inserts or removes video segments shifts
    the absolute start of everything after the change by the same amount;
    this reduces that to one lookup: the shift in force at time ``t`` is
    that of the last segment (matched by :attr:`VideoSegment.uid`, stable
    across the edit) starting at or before ``t`` that is still in the
    timeline — or, when it was dropped, that of the nearest earlier
    surviving one.

    Used by :meth:`Timeline.insert_segments`/``remove_segments``/
    ``replace_segment`` and by ``ytedit.ai.tidy``'s speech padding.

    Args:
        before: ``segment_positions()`` captured before the edit.
        after: ``segment_positions()`` of the result.

    Returns:
        A function from an old absolute time to its new one.
    """
    after_by_uid = {pos.segment.uid: pos.start for pos in after if pos.segment.uid}
    starts: list[float] = []
    shifts: list[float] = []
    last_shift = 0.0
    for pos in before:
        new_start = after_by_uid.get(pos.segment.uid) if pos.segment.uid else None
        if new_start is not None:
            last_shift = new_start - pos.start
        starts.append(pos.start)
        shifts.append(last_shift)

    def remap(t: float) -> float:
        if not starts:
            return t
        i = bisect.bisect_right(starts, t) - 1
        return round(t + (shifts[i] if i >= 0 else 0.0), 3)

    return remap


def _retime_absolute_tracks(timeline: "Timeline", remap: Callable[[float], float]) -> None:
    """Shift every absolute-time item downstream of a video-track edit.

    Anchored captions and voice items are skipped — they are pinned to a
    segment, not absolute time, and :meth:`Timeline.resolve_anchors`
    re-derives their position from wherever that segment ends up once every
    pass has settled.
    """
    for cap in timeline.tracks.captions:
        if cap.anchor is not None:
            continue
        cap.at = remap(cap.at)
        cap.end = remap(cap.end)
    for cue in timeline.tracks.music:
        cue.at = remap(cue.at)
        cue.end = remap(cue.end)
    for item in timeline.tracks.voice:
        if item.anchor is not None:
            continue
        # A pickup is a fixed-length file: move it, never stretch it.
        length = None if item.end is None else item.end - item.at
        item.at = remap(item.at)
        if length is not None:
            item.end = round(item.at + length, 3)
    # Structural markers and chapters are pinned to absolute time as well; a
    # chapter list that lags the picture by a minute is worse than none.
    for marker in timeline.markers:
        marker.at = remap(marker.at)
    for chapter in timeline.chapters:
        chapter.at = remap(chapter.at)


# ----------------------------------------------------------------------
# speech ranges
# ----------------------------------------------------------------------
def _word_span(word: dict[str, Any]) -> tuple[float, float] | None:
    """Extract ``(start, end)`` from a transcript word in either key spelling."""
    if word.get("type") not in (None, "word"):
        return None
    start = word.get("s", word.get("start"))
    end = word.get("e", word.get("end"))
    if start is None or end is None:
        return None
    try:
        return float(start), float(end)
    except (TypeError, ValueError):  # pragma: no cover - malformed transcript
        return None


def merge_ranges(
    ranges: Iterable[tuple[float, float]], merge_gap: float = 0.4, pad: float = 0.15
) -> list[tuple[float, float]]:
    """Pad and merge time ranges.

    Args:
        ranges: Unsorted ``(start, end)`` pairs.
        merge_gap: Ranges closer than this are joined.
        pad: Seconds added on both sides of every range before merging.

    Returns:
        Sorted, non-overlapping ranges clamped at zero.
    """
    padded = sorted(
        (max(0.0, s - pad), e + pad) for s, e in ranges if e > s
    )
    out: list[tuple[float, float]] = []
    for start, end in padded:
        if out and start - out[-1][1] <= merge_gap:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return [(round(s, 3), round(e, 3)) for s, e in out]


def speech_ranges_from_transcripts(
    project: "Project",
    timeline: Timeline | None = None,
    merge_gap: float = 0.4,
    pad: float = 0.15,
) -> list[tuple[float, float]]:
    """Map transcript word spans into timeline time.

    Every video segment contributes the words of its clip that fall inside
    ``[in, out)``, translated to the segment's absolute position (and divided by
    ``speed``). A segment with :attr:`VideoSegment.audio_from` contributes the
    words of *that* clip range instead, still placed at the picture segment's
    position. Ranges shorter than ``merge_gap`` apart are merged and padded by
    ``pad`` — the result drives ``duck.mode == "auto"`` music automation.

    Args:
        project: Project holding ``transcripts/<clip>.json``.
        timeline: Timeline to map onto; loaded from ``plan/timeline.json`` when
            omitted.
        merge_gap: Gap under which two ranges become one.
        pad: Padding applied on both sides of each word run.

    Returns:
        Sorted, non-overlapping ``(start, end)`` ranges in timeline seconds.
    """
    tl = timeline if timeline is not None else Timeline.load(project.timeline_file)
    cache: dict[str, list[tuple[float, float]]] = {}
    spans: list[tuple[float, float]] = []

    for pos in tl.segment_positions():
        seg = pos.segment
        if seg.mute_source:
            continue
        # An overlay cutaway is heard as the clip underneath it, not as its own
        # picture clip: read the words from wherever the audio really comes from.
        src_clip, src_in, src_out = seg.audio_source
        if src_clip not in cache:
            path = project.transcript_path(src_clip)
            words: list[tuple[float, float]] = []
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:  # pragma: no cover - bad transcript
                    log.warning("unreadable transcript %s", path)
                    data = {}
                for word in data.get("words", []):
                    span = _word_span(word)
                    if span:
                        words.append(span)
            else:
                log.debug("no transcript for clip %s", src_clip)
            cache[src_clip] = words

        speed = seg.speed if seg.speed > 0 else 1.0
        for w_start, w_end in cache[src_clip]:
            if w_end <= src_in or w_start >= src_out:
                continue
            s = max(w_start, src_in)
            e = min(w_end, src_out)
            spans.append((pos.start + (s - src_in) / speed, pos.start + (e - src_in) / speed))

    return merge_ranges(spans, merge_gap=merge_gap, pad=pad)


def new_timeline(
    width: int = 1920, height: int = 1080, fps: int = 30, language: str = "pl"
) -> Timeline:
    """Create an empty timeline with the given canvas."""
    return Timeline(width=width, height=height, fps=fps, language=language)
