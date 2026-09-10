"""The timeline (EDL) data model — ``plan/timeline.json``.

The timeline is a **derived** artifact: ``plan/cut.json`` is the source of
truth for the edit and :func:`ytedit.cut.resolve` is the only thing that
writes a timeline (see ``docs/ARCHITECTURE.md``). Nothing edits one in place —
it is safe to delete and regenerate — so this module is a data model plus its
structural checks, with no edit API.

Video segments are
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

import json
import math
import os
import secrets
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from .log import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from .project import Project

log = get_logger(__name__)

#: Only cut v2 timelines are readable; a v1 file must go through
#: ``ytedit migrate`` once (see :meth:`Timeline.load`).
TIMELINE_VERSION = 2

FitMode = Literal["cover", "contain", "blur-fill", "crop-pan"]
TransitionType = Literal["cut", "fade", "xfade"]
DuckMode = Literal["auto", "manual", "off"]
CaptionPosition = Literal[
    "lower-left", "lower-right", "lower-center", "center", "upper-left", "upper-right",
    "upper-center",
]


class TimelineVersionError(RuntimeError):
    """Raised by :meth:`Timeline.load` for a timeline that is not cut v2."""


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

    A shot placed in the middle of a speech beat keeps its own picture but must
    not interrupt the narration underneath it. Setting ``audio_from`` makes the
    render take the segment's audio from ``clip[in, out]`` instead of the
    segment's own clip range, trimmed/padded to the segment's exact frame count.
    This is what :func:`ytedit.cut.resolve` writes for every shot: a shot never
    carries sound of its own, it borrows the beat's.
    """

    clip: str
    in_: float = Field(0.0, alias="in", serialization_alias="in")
    out: float = 0.0

    @property
    def duration(self) -> float:
        """Length of the borrowed audio range in seconds."""
        return max(0.0, self.out - self.in_)


class AudioWindow(_Model):
    """The part of a segment's audio that is actually **heard** (v2).

    Expressed in the time base of the segment's audio source
    (:attr:`VideoSegment.audio_source` — the ``audio_from`` clip when one is
    set, otherwise the segment's own clip), exactly like
    :class:`AudioFrom`/:class:`MuteRange`. Everything inside the segment but
    outside ``[in, out]`` renders as silence.

    ``ytedit.cut.resolve`` sets this when a speech beat's picture had to be
    extended past the air its audio may safely carry: a neighbouring word
    sits close enough that the audio window has to stop short of the full
    ``pacing.speech_pad_before/after``, so the picture shows the extra
    fraction of a second while the word itself is never heard. That is what
    makes "a pause here" a guarantee rather than a hope.
    """

    in_: float = Field(0.0, alias="in", serialization_alias="in")
    out: float = 0.0

    @property
    def duration(self) -> float:
        """Length of the audible window in seconds."""
        return max(0.0, self.out - self.in_)


class VideoSegment(_Model):
    """One cut from a normalized source clip."""

    id: str
    #: Stable identity, derived once from the owning beat by
    #: :func:`ytedit.cut.resolve` (``sha1(beat uid + index)``) and never
    #: renumbered — unlike :attr:`id`, the display label reassigned to
    #: ``s001, s002, ...`` on every resolve. It is part of the render's
    #: per-segment cache key, so re-resolving an unchanged cut must not
    #: change it.
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
    #: The ``uid`` of the ``cut.json`` beat this segment was resolved from
    #: (:func:`ytedit.cut.resolve`). Every segment of one beat carries the
    #: same value and they are audio-contiguous.
    beat: str | None = None
    #: v2: the audible part of this segment's audio (see :class:`AudioWindow`).
    #: ``None`` means the whole segment is heard.
    audio_window: AudioWindow | None = None

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


class VoiceItem(_Model):
    """A narration pickup placed at an absolute timeline position.

    ``at``/``end`` are produced by :func:`ytedit.cut.resolve` from the voice
    beat's position; nothing edits them in place.
    """

    id: str
    file: str
    at: float = 0.0
    gain_db: float = 0.0
    #: Optional explicit end; when absent the file length is used at render time.
    end: float | None = None


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
    #: v2: what this (derived) timeline was resolved from, e.g.
    #: ``"cut.json@2026-09-09T10:11:12+00:00"`` — see ``ytedit.cut.resolve``.
    #: Empty for a v1 timeline that is still the source of truth.
    source: str = ""


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

    version: int = TIMELINE_VERSION
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
        """Read a v2 timeline JSON file.

        A timeline is a *derived* artifact of ``plan/cut.json`` since cut v2
        (:func:`ytedit.cut.resolve`); a v1 file carries no beat identity and
        cannot be rendered, inspected or re-resolved, so it is refused rather
        than half-understood.

        Args:
            path: Path to ``timeline.json``.

        Raises:
            TimelineVersionError: When the file is not ``version: 2``.
        """
        target = Path(path)
        data = json.loads(target.read_text(encoding="utf-8"))
        version = data.get("version")
        if version != TIMELINE_VERSION:
            raise TimelineVersionError(
                f"{target} is a v{version} timeline; plan/cut.json is the source of "
                f"truth since cut v2. Run `ytedit migrate <slug>` once to convert it."
            )
        return cls.model_validate(data)

    @classmethod
    def load_legacy(cls, path: Path | str) -> "Timeline":
        """Read a timeline of **any** version, version gate bypassed.

        The one legitimate caller is :mod:`ytedit.migrate`, which exists
        precisely to read a v1 file. The fields v2 dropped (the v1 anchors and
        the per-segment sentence-id list) survive as pydantic extras, which is
        all a migration needs. Nothing else may use this: a v1 timeline is not
        renderable and has no beat identity.

        Args:
            path: Path to the timeline JSON.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
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

        Never touches :attr:`VideoSegment.uid`, the identity the render
        cache and ``ytedit at`` key on.
        """
        for i, seg in enumerate(self.tracks.video):
            seg.id = f"s{i + 1:03d}"

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
            if seg.audio_window is not None:
                win = seg.audio_window
                src_clip, src_in, src_out = seg.audio_source
                if win.out <= win.in_:
                    issues.append(
                        f"{seg.id}: audio_window in ({win.in_}) >= out ({win.out})"
                    )
                elif win.in_ < src_in - 1e-6 or win.out > src_out + 1e-6:
                    issues.append(
                        f"{seg.id}: audio_window {win.in_:.3f}-{win.out:.3f}s falls outside "
                        f"the segment's audio range {src_clip} {src_in:.3f}-{src_out:.3f}s"
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
        # A shot is heard as the beat underneath it, not as its own picture
        # clip: read the words from wherever the audio really comes from.
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
        # v2: a word the segment shows but silences (see AudioWindow) is not
        # speech as far as the mix is concerned — it must not duck the music.
        heard_in, heard_out = src_in, src_out
        if seg.audio_window is not None:
            heard_in = max(heard_in, seg.audio_window.in_)
            heard_out = min(heard_out, seg.audio_window.out)
        for w_start, w_end in cache[src_clip]:
            if w_end <= heard_in or w_start >= heard_out:
                continue
            s = max(w_start, heard_in)
            e = min(w_end, heard_out)
            spans.append((pos.start + (s - src_in) / speed, pos.start + (e - src_in) / speed))

    return merge_ranges(spans, merge_gap=merge_gap, pad=pad)


def new_timeline(
    width: int = 1920, height: int = 1080, fps: int = 30, language: str = "pl"
) -> Timeline:
    """Create an empty timeline with the given canvas."""
    return Timeline(width=width, height=height, fps=fps, language=language)
