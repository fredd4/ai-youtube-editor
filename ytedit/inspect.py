"""Timecode inspector — ``ytedit at <slug> <time>``.

Turns a timecode from the user's feedback ("at 4:27 the sentence is cut") into an
exact reference into the **cut**: which beat of ``plan/cut.json`` is playing,
which of its sentences it carries, which shot of that beat is on screen, and —
from the derived ``plan/timeline.json`` — which video segment that is, where
its audio comes from, which transcript words are heard around that instant (or
which voice pickup is playing, or whether it is music-only), which captions
are burned in, which music cue is active and which chapter it falls under.

The beat half is what makes a note actionable: a correction is applied to the
cut (a sentence dropped, a shot moved), never to the timeline, which is a
derived artifact. When a project has no ``plan/cut.json`` the beat block is
simply absent and everything else still works.

Times are in **render** time — the position in the actual rendered file
(:meth:`~ytedit.timeline.Timeline.segment_positions` with
``fade_overlaps=True``, exactly what :func:`ytedit.media.render.render_positions`
uses) — since that is what the user is looking at when they timestamp feedback
against a preview/master/draft. Captions, voice pickups, music cues and
chapters are stored in *timeline* time (pre-transition-remap) and are mapped
forward through :func:`ytedit.media.render.build_time_map` to compare against
a render-time instant.

Nothing here writes anything or shells out to ffmpeg for encoding — at most a
cheap ``ffprobe`` to learn an un-ended voice pickup's file length, and the
re-resolve :func:`load_timeline` performs when the cut is newer than the
timeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .ai.sentences import load_sentences, sentences_by_clip
from .media import audio as audio_mod
from .media.render import SegmentPlacement, build_time_map, render_positions
from .project import Project
from .timeline import Timeline, VideoSegment, _word_span

if TYPE_CHECKING:  # pragma: no cover
    from .cut import Beat, Cut

#: Default window (seconds, in the audio clip's own time base) searched for
#: transcript words/sentences around the requested instant.
DEFAULT_WINDOW: float = 3.0


class TimecodeError(ValueError):
    """A timecode string could not be parsed."""


def parse_timecode(text: str) -> float:
    """Parse ``mm:ss``, ``mm:ss.s``, ``hh:mm:ss`` or a bare number of seconds.

    Args:
        text: User-supplied timecode, e.g. ``"4:27"``, ``"1:04:27.5"``, ``"267"``.

    Returns:
        Seconds as a float.

    Raises:
        TimecodeError: When ``text`` matches none of the accepted shapes.
    """
    raw = text.strip()
    if not raw:
        raise TimecodeError("empty timecode")
    if ":" not in raw:
        try:
            return float(raw)
        except ValueError as exc:
            raise TimecodeError(f"cannot parse timecode {text!r}") from exc
    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise TimecodeError(f"cannot parse timecode {text!r}")
    try:
        *head, secs = parts
        seconds = float(secs)
        for i, part in enumerate(reversed(head)):
            seconds += int(part) * (60 if i == 0 else 3600)
        return seconds
    except ValueError as exc:
        raise TimecodeError(f"cannot parse timecode {text!r}") from exc


def format_timecode(seconds: float) -> str:
    """Render seconds as ``mm:ss.s`` (``hh:mm:ss.s`` past an hour)."""
    seconds = max(0.0, seconds)
    whole = int(seconds)
    frac = seconds - whole
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s + frac:04.1f}"
    return f"{m}:{s + frac:04.1f}"


@dataclass
class WordHit:
    """One transcript word within the search window."""

    text: str
    start: float
    end: float


@dataclass
class MomentInfo:
    """Everything :func:`inspect_moment` found for one render-time instant."""

    at: float
    segment: VideoSegment | None
    segment_start: float | None
    segment_end: float | None
    audio_clip: str | None
    audio_clip_time: float | None
    audio_description: str
    voice_pickup: str | None
    words: list[WordHit] = field(default_factory=list)
    sentences_heard: list[str] = field(default_factory=list)
    captions: list[dict[str, Any]] = field(default_factory=list)
    music: dict[str, Any] | None = None
    chapter: str | None = None
    #: The ``cut.json`` beat the segment was resolved from (see
    #: :func:`beat_block`); ``None`` when the project has no cut yet.
    beat: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": round(self.at, 3),
            "at_tc": format_timecode(self.at),
            "segment": (
                {
                    "id": self.segment.id,
                    "uid": self.segment.uid,
                    "clip": self.segment.clip,
                    "in": round(self.segment.in_, 3),
                    "out": round(self.segment.out, 3),
                    "role": self.segment.role,
                    "mute_source": self.segment.mute_source,
                    "audio_from": (
                        {
                            "clip": self.segment.audio_from.clip,
                            "in": round(self.segment.audio_from.in_, 3),
                            "out": round(self.segment.audio_from.out, 3),
                        }
                        if self.segment.audio_from is not None
                        else None
                    ),
                    "render_start": round(self.segment_start, 3) if self.segment_start is not None else None,
                    "render_end": round(self.segment_end, 3) if self.segment_end is not None else None,
                }
                if self.segment is not None
                else None
            ),
            "audio_clip": self.audio_clip,
            "audio_clip_time": round(self.audio_clip_time, 3) if self.audio_clip_time is not None else None,
            "audio_description": self.audio_description,
            "voice_pickup": self.voice_pickup,
            "words": [{"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)} for w in self.words],
            "sentences_heard": self.sentences_heard,
            "captions": self.captions,
            "music": self.music,
            "chapter": self.chapter,
            "beat": self.beat,
        }


# ----------------------------------------------------------------------
# the cut half: which beat, which of its sentences, which shot
# ----------------------------------------------------------------------
def load_cut_for(project: Project) -> "Cut | None":
    """Load ``plan/cut.json``, or ``None`` when the project has none (or it is broken).

    The inspector reports, it never gates: a project that has not been
    migrated to cut v2 yet still answers "which segment is at 4:27", just
    without the beat half.
    """
    from .cut import cut_path, load_cut

    path = cut_path(project)
    if not path.exists():
        return None
    try:
        return load_cut(path)
    except Exception:  # pragma: no cover - unreadable/invalid cut
        return None


def load_timeline(project: Project) -> Timeline:
    """Re-resolve the cut when it is newer, then load ``plan/timeline.json``.

    The timeline is derived, so reading it without this can answer a question
    about a cut that no longer exists. Kept here (rather than in the caller)
    so every entry point into the inspector gets the same guarantee.

    Raises:
        ytedit.cut.CutError: When the cut no longer resolves.
        FileNotFoundError: When there is no timeline at all.
    """
    from .cut import ensure_resolved

    ensure_resolved(project)
    return Timeline.load(project.timeline_file)


def shot_index_by_segment(beat: "Beat", segments: Sequence[VideoSegment]) -> dict[str, int | None]:
    """Map each of a beat's segment uids to the index of the shot it shows.

    The resolver interleaves a beat's own picture with its ``shots`` in order
    and can trim or drop a shot, so the mapping is recovered by walking both
    lists forward together: a segment is the beat's own picture when it plays
    the beat's clip and the next unconsumed shot does not start there,
    otherwise it is the next shot on that clip.

    Returns:
        ``{segment uid: shot index or None}``, ``None`` meaning the beat's own
        picture.
    """
    mapping: dict[str, int | None] = {}
    shots = list(beat.shots)
    cursor = 0
    for seg in segments:
        own = bool(beat.clip) and seg.clip == beat.clip
        candidate = next(
            (i for i in range(cursor, len(shots)) if shots[i].clip == seg.clip), None
        )
        if candidate is not None and own:
            # Ambiguous only when a shot reuses the beat's own clip: the shot
            # then has to start where the segment does.
            if abs(float(shots[candidate].in_) - float(seg.in_)) > 0.05:
                candidate = None
        if candidate is None:
            mapping[seg.uid] = None
            continue
        mapping[seg.uid] = candidate
        cursor = candidate + 1
    return mapping


def beat_block(cut: "Cut", segment: VideoSegment, segments: Sequence[VideoSegment]) -> dict[str, Any] | None:
    """Describe the beat ``segment`` was resolved from, and its shot on screen.

    Args:
        cut: The loaded ``plan/cut.json``.
        segment: The segment on screen at the inspected instant.
        segments: Every segment of the timeline, in order (used to work out
            which of the beat's shots this one is).

    Returns:
        ``{"id", "uid", "kind", "role", "clip", "file", "sentences", "words",
        "on_screen"}`` — ``sentences`` being the ids of the *whole* beat, not
        just the words heard in the search window — or ``None`` when the
        segment carries no beat uid (a pre-cut timeline) or the beat is gone.
    """
    if not segment.beat:
        return None
    beat = cut.beat_by_ref(segment.beat)
    if beat is None:
        return None
    mine = [seg for seg in segments if seg.beat == beat.uid]
    shot_index = shot_index_by_segment(beat, mine).get(segment.uid)
    if shot_index is None:
        on_screen = {
            "kind": "beat",
            "shot": None,
            "clip": segment.clip,
            "in": round(segment.in_, 3),
            "out": round(segment.out, 3),
            "label": (
                f"the beat's own picture ({segment.clip} "
                f"{segment.in_:.2f}-{segment.out:.2f})"
            ),
        }
    else:
        on_screen = {
            "kind": "shot",
            "shot": shot_index,
            "clip": segment.clip,
            "in": round(segment.in_, 3),
            "out": round(segment.out, 3),
            "label": (
                f"shot {shot_index + 1}/{len(beat.shots)} ({segment.clip} "
                f"{segment.in_:.2f}-{segment.out:.2f})"
            ),
        }
    return {
        "id": beat.id or beat.uid,
        "uid": beat.uid,
        "kind": beat.kind,
        "role": beat.role,
        "clip": beat.clip,
        "file": beat.file,
        "sentences": list(beat.sentences),
        "words": list(beat.words) if beat.words is not None else None,
        "on_screen": on_screen,
    }


def _find_placement(placements: list[SegmentPlacement], at: float) -> SegmentPlacement | None:
    """Return the segment placement covering render time ``at``.

    Falls back to the last segment when ``at`` is past the programme end (a
    timecode a hair past the final frame due to rounding), and to the first
    when it is before the start.
    """
    if not placements:
        return None
    for placement in placements:
        if placement.start <= at < placement.end:
            return placement
    return placements[-1] if at >= placements[-1].end else placements[0]


def _active_voice_item(
    project: Project, timeline: Timeline, to_render, at: float
) -> tuple[Any, float, float] | None:
    """Return ``(item, render_start, render_end)`` for a voice pickup covering ``at``."""
    for item in timeline.tracks.voice:
        start = to_render(item.at)
        if item.end is not None:
            end = to_render(item.end)
        else:
            length = audio_mod.audio_duration(project.path / item.file)
            end = start + length
        if start <= at < end:
            return item, start, end
    return None


def _active_music_cue(timeline: Timeline, to_render, at: float) -> dict[str, Any] | None:
    """Return the active music cue (as a small dict) covering render time ``at``."""
    for cue in timeline.tracks.music:
        start, end = to_render(cue.at), to_render(cue.end)
        if start <= at < end:
            return {"id": cue.id, "file": cue.file, "gain_db": cue.gain_db}
    return None


def _active_captions(timeline: Timeline, to_render, at: float) -> list[dict[str, Any]]:
    """Return every caption cue (as small dicts) active at render time ``at``."""
    out: list[dict[str, Any]] = []
    for cue in timeline.tracks.captions:
        start, end = to_render(cue.at), to_render(cue.end)
        if start <= at < end:
            out.append({"id": cue.id, "text": cue.text, "style": cue.style})
    return out


def _active_chapter(timeline: Timeline, to_render, at: float) -> str | None:
    """Return the title of the chapter active at render time ``at``, if any."""
    current: str | None = None
    for chapter in sorted(timeline.chapters, key=lambda c: c.at):
        if to_render(chapter.at) <= at:
            current = chapter.title
        else:
            break
    return current


def _words_in_window(
    project: Project, clip_id: str, center: float, window: float
) -> list[WordHit]:
    """Transcript words of ``clip_id`` overlapping ``[center - window, center + window]``."""
    path = project.transcript_path(clip_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt transcript
        return []
    lo, hi = center - window, center + window
    hits: list[WordHit] = []
    for word in data.get("words", []):
        span = _word_span(word)
        if span is None:
            continue
        s, e = span
        if e < lo or s > hi:
            continue
        text = str(word.get("t", word.get("text", ""))).strip()
        if text:
            hits.append(WordHit(text, s, e))
    return hits


def _sentences_in_window(
    project: Project, clip_id: str, center: float, window: float
) -> list[str]:
    """Sentence catalogue ids for ``clip_id`` overlapping the search window."""
    doc = load_sentences(project)
    if not doc:
        return []
    by_clip = sentences_by_clip(doc)
    lo, hi = center - window, center + window
    return [
        str(s["id"])
        for s in by_clip.get(clip_id, [])
        if float(s.get("e", 0)) >= lo and float(s.get("s", 0)) <= hi
    ]


def inspect_moment(
    project: Project, timeline: Timeline, at: float, window: float = DEFAULT_WINDOW
) -> MomentInfo:
    """Resolve everything happening at one render-time instant.

    Args:
        project: Owning project.
        timeline: The loaded timeline (``plan/timeline.json``) — use
            :func:`load_timeline` to get one that is guaranteed current.
        at: Render-time instant in seconds (what a viewer's video-player
            timecode shows).
        window: Seconds either side of the audio clip's mapped instant to
            search for transcript words / sentence ids.

    Returns:
        A :class:`MomentInfo`. ``segment`` is ``None`` only for an empty
        timeline; ``beat`` is ``None`` when the project has no ``cut.json``.
    """
    cut = load_cut_for(project)
    placements = render_positions(timeline)
    to_render = build_time_map(timeline)
    placement = _find_placement(placements, at)

    voice_hit = _active_voice_item(project, timeline, to_render, at)
    music = _active_music_cue(timeline, to_render, at)
    captions = _active_captions(timeline, to_render, at)
    chapter = _active_chapter(timeline, to_render, at)

    if placement is None:
        if voice_hit is not None:
            voice_name = Path(voice_hit[0].file).name
            description = f"voice pickup: {voice_name}"
        elif music:
            voice_name, description = None, "music only"
        else:
            voice_name, description = None, "empty timeline"
        return MomentInfo(
            at=at, segment=None, segment_start=None, segment_end=None,
            audio_clip=None, audio_clip_time=None,
            audio_description=description, voice_pickup=voice_name,
            captions=captions, music=music, chapter=chapter,
        )

    seg = placement.segment
    audio_clip, audio_in, audio_out = seg.audio_source
    speed = seg.speed if seg.speed > 0 else 1.0
    # Offset into the segment's own render-time span, converted back to the
    # audio clip's original (pre-speed) time base.
    local = max(0.0, min(at - placement.start, placement.end - placement.start))
    audio_clip_time = min(audio_out, audio_in + local * speed)

    words: list[WordHit] = []
    sentences_heard: list[str] = []
    is_silent = seg.mute_source or not bool(
        project.load_state().get("clips", {}).get(audio_clip, {}).get("has_audio", True)
    )

    if voice_hit is not None:
        item, *_ = voice_hit
        description = f"voice pickup: {Path(item.file).name}"
        voice_name: str | None = Path(item.file).name
    elif is_silent:
        description = "music only" if music else "silence"
        voice_name = None
    else:
        words = _words_in_window(project, audio_clip, audio_clip_time, window)
        sentences_heard = _sentences_in_window(project, audio_clip, audio_clip_time, window)
        voice_name = None
        if words:
            snippet = " ".join(w.text for w in sorted(words, key=lambda w: w.start))
            description = f'"{snippet}"'
        else:
            description = "music only" if music else "ambient audio (no speech nearby)"

    # ``sentences_heard`` stays the window search (what is *heard* around this
    # instant); the beat block carries the ids of the whole beat, which is
    # what an edit is expressed against.
    beat = beat_block(cut, seg, timeline.tracks.video) if cut is not None else None

    return MomentInfo(
        at=at,
        segment=seg,
        segment_start=placement.start,
        segment_end=placement.end,
        audio_clip=audio_clip,
        audio_clip_time=audio_clip_time,
        audio_description=description,
        voice_pickup=voice_name,
        words=sorted(words, key=lambda w: w.start),
        sentences_heard=sentences_heard,
        captions=captions,
        music=music,
        chapter=chapter,
        beat=beat,
    )


def inspect_range(
    project: Project, timeline: Timeline, at: float, around: float
) -> list[dict[str, Any]]:
    """List every video segment within ``±around`` seconds of render time ``at``.

    Returns:
        Small dicts (``id``, ``uid``, ``clip``, ``in``, ``out``, ``role``,
        ``render_start``, ``render_end``), in timeline order.
    """
    placements = render_positions(timeline)
    lo, hi = at - around, at + around
    out: list[dict[str, Any]] = []
    for placement in placements:
        if placement.end < lo or placement.start > hi:
            continue
        seg = placement.segment
        out.append(
            {
                "id": seg.id,
                "uid": seg.uid,
                "clip": seg.clip,
                "in": round(seg.in_, 3),
                "out": round(seg.out, 3),
                "role": seg.role,
                "render_start": round(placement.start, 3),
                "render_end": round(placement.end, 3),
            }
        )
    return out
