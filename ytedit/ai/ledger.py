"""Audio ledger dedupe pass: no audio may ever play twice.

The planner and the overlay/tidy passes each reason about one stretch of the
timeline at a time; nothing keeps a global record of which slice of which
source clip's *audio* has already been heard once a human has also poked at
the cut in the web editor. Two independent segments can end up drawing on the
same seconds of the same clip — either directly (two ``a-roll``/own-audio
segments overlapping) or through an ``audio_from`` cutaway that keeps playing
narration a later a-roll segment already covers — and the viewer hears the
same sentence twice.

:func:`dedupe_audio` is the safety net: it walks every video segment in
timeline order, keeps a per-clip ledger of the audio ranges already
committed (a segment's own ``[in, out)`` when it is not muted, an
``audio_from`` range, and every ``voice/vo_<clip>_<in>_<out>.wav`` pickup —
the file name ``ytedit.ai.plan._build_voice_over_segment`` writes when it
extracts narration straight from a source clip), and fixes any segment whose
audio overlaps the ledger by more than ``pacing.audio_dupe_tolerance``:

* an own-audio (no ``audio_from``) segment has its ``in`` — and so its
  picture, since picture and audio are the same source — advanced past the
  end of what has already played; too little left (< ``min_shot_seconds``)
  and the segment is dropped instead;
* a segment whose *entire* own audio is already used is muted rather than
  dropped — the picture is worth keeping, the sound is not;
* an ``audio_from`` segment has ``audio_from.in`` advanced the same way;
  when what is left is shorter than the picture it carries, the picture is
  shortened to match (when that still clears ``min_shot_seconds``) or the
  narration is dropped (``audio_from = None``, ``mute_source = True``) and
  the picture plays silent under the music — never both problems at once.

Ambient repeats (no transcript word anywhere in the overlap — a repeated
splash of crowd noise, wind, water) are harmless and, by default
(``allow_ambient_repeat``), left alone; they are still counted in the
returned ``changes`` so a report can show how many were tolerated.

Dropping or resizing a segment shifts everything after it, so captions,
music cues, voice pickups, markers and chapters are re-timed with the same
``ytedit.ai.tidy`` machinery :mod:`ytedit.ai.overlay` uses.

The pass runs at the end of ``ytedit tidy`` (:func:`ytedit.ai.tidy.tidy`,
right after :func:`ytedit.ai.overlay.overlay_cutaways`) and can be called
directly to fix a timeline that was hand-edited into a duplicate.

:func:`find_duplicate_audio` is the read-only counterpart used by
``ytedit qc`` (rule 31): it reports every collision it finds — including
ones a fix pass cannot resolve on its own, such as two narration pickups
that were independently extracted from overlapping clip ranges — without
touching the timeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ytedit.ai.tidy import Word, _retime_absolute_tracks, _shift_map, load_words
from ytedit.config import Settings
from ytedit.log import get_logger
from ytedit.project import Project
from ytedit.timeline import Timeline, VideoSegment

log = get_logger(__name__)

_EPS = 1e-6

#: Filename pattern written by ``ytedit.ai.plan._build_voice_over_segment``:
#: ``voice/vo_<clip>_<in>_<out>.wav`` — narration cut straight from a source
#: clip's own audio track. Any other voice file (a recorded pickup,
#: ``n001.wav``, ``pickup_*.wav``...) carries no clip-audio-range information
#: and plays no part in this ledger.
VO_FILE_RE = re.compile(
    r"(?:^|/)vo_(?P<clip>[A-Za-z0-9]+)_(?P<in>\d+\.\d+)_(?P<out>\d+\.\d+)\.wav$"
)

#: Tag used in the change log for an allowed ambient repeat (see
#: :func:`ambient_repeat_count`).
AMBIENT_TAG = "ambient-repeat"


@dataclass
class DuplicateFinding:
    """One audio range played more than once, found by :func:`find_duplicate_audio`."""

    clip: str
    start: float
    end: float
    speech: bool
    ids: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ----------------------------------------------------------------------
# interval helpers
# ----------------------------------------------------------------------
def _merge_into(intervals: list[tuple[float, float]], s: float, e: float) -> None:
    """Insert ``[s, e)`` into a sorted, merged interval list, in place."""
    if e <= s + _EPS:
        return
    intervals.append((s, e))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for cs, ce in intervals:
        if merged and cs <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ce))
        else:
            merged.append((cs, ce))
    intervals[:] = merged


def committed_audio_ranges(
    timeline: Timeline, video_slice: slice | None = None
) -> dict[str, list[tuple[float, float]]]:
    """Per-clip merged audio ranges the timeline already claims right now.

    The same notion :func:`dedupe_audio` polices *after* the fact, exposed
    here so :mod:`ytedit.ai.tidy` (sentence-boundary snapping) and
    :mod:`ytedit.ai.overlay` (the cutaway rewrite) can consult it *before*
    moving a cut, instead of only cleaning up once a move has already
    created a duplicate: a segment's own ``[in, out)`` audio when it is not
    muted, every ``audio_from`` range, and every
    ``voice/vo_<clip>_<in>_<out>.wav`` pickup extracted straight from a
    clip's own audio.

    Unlike :func:`dedupe_audio`'s chronological walk, this is a plain
    snapshot — "what does the timeline claim right now" — with no notion of
    priority between overlapping claims. ``video_slice`` lets a caller scope
    that snapshot to one side of the segment it is about to move: a cut
    reaching *backward* should only be blocked by claims already staked
    *earlier* in the video track (``slice(0, index)``) — exactly the ones
    the ledger's chronological walk would have given priority to — and a cut
    reaching *forward* only by claims staked *later* (``slice(index + 1,
    None)``). Passing ``None`` (the default) uses the whole track. Voice
    pickups are always included regardless of ``video_slice``, since they
    carry no position in the video track to scope by.

    Returns:
        ``{clip: [(start, end), ...]}``, each list sorted and merged.
    """
    segments = timeline.tracks.video if video_slice is None else timeline.tracks.video[video_slice]
    ranges: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        if seg.mute_source:
            continue
        clip, s, e = seg.audio_source
        _merge_into(ranges.setdefault(clip, []), s, e)
    for item in timeline.tracks.voice:
        m = VO_FILE_RE.search(item.file)
        if not m:
            continue
        clip = m.group("clip")
        s, e = float(m.group("in")), float(m.group("out"))
        _merge_into(ranges.setdefault(clip, []), s, e)
    return ranges


def _consumed_overlap(intervals: Sequence[tuple[float, float]], s: float, e: float) -> float:
    """Total overlap of ``[s, e)`` against a merged interval list."""
    total = 0.0
    for cs, ce in intervals:
        total += max(0.0, min(e, ce) - max(s, cs))
    return total


def _advance_past(intervals: Sequence[tuple[float, float]], start: float, end: float) -> float:
    """Furthest point up to which anything in ``[start, end)`` is already used.

    Safety trumps neatness here: every interval overlapping ``[start, end)`` is
    folded in, even across a gap — a segment is better trimmed a little more
    than strictly necessary than left repeating a fragment of audio.
    """
    cur = start
    for cs, ce in intervals:
        if ce <= start + _EPS or cs >= end - _EPS:
            continue
        cur = max(cur, ce)
    return cur


def _has_words(
    project: Project, clip: str, s: float, e: float, cache: dict[str, list[Word]]
) -> bool:
    """True when ``clip`` has a transcript word inside ``[s, e)``."""
    if clip not in cache:
        cache[clip] = load_words(project, clip)
    return any(w.e > s + _EPS and w.s < e - _EPS for w in cache[clip])


def _fmt(value: float) -> str:
    return f"{value:.2f}"


# ----------------------------------------------------------------------
# the fix pass
# ----------------------------------------------------------------------
def dedupe_audio(
    timeline: Timeline,
    project: Project,
    settings: Settings | None = None,
    allow_ambient_repeat: bool = True,
) -> tuple[Timeline, list[str]]:
    """Walk the audio ledger and stop any range from playing twice.

    Args:
        timeline: The EDL to fix; modified in place and returned.
        project: Project supplying transcripts and clip durations.
        settings: Settings override (defaults to ``project.settings``); reads
            ``pacing.audio_dupe_tolerance`` and ``pacing.min_shot_seconds``.
        allow_ambient_repeat: When true (the default), a repeated range with no
            transcript word in it is left alone — still counted in ``changes``
            (see :func:`ambient_repeat_count`) but not rewritten.

    Returns:
        ``(timeline, changes)`` — ``changes`` is a human-readable list, empty
        when nothing overlapped.
    """
    cfg = settings or project.settings
    tolerance = max(0.0, float(cfg.get("pacing.audio_dupe_tolerance", 0.25)))
    min_shot = max(0.0, float(cfg.get("pacing.min_shot_seconds", 0.8)))

    words_cache: dict[str, list[Word]] = {}
    used: dict[str, list[tuple[float, float]]] = {}
    changes: list[str] = []
    dropped: set[int] = set()
    mutated = False

    before_positions = timeline.segment_positions()
    starts = {id(pos.segment): pos.start for pos in before_positions}
    video = timeline.tracks.video

    # A single chronological walk over both tracks: a video segment's position
    # is its rendered start; a VO pickup's is where it sits on the voice track.
    # Interleaving them means a narration pickup extracted from a clip counts
    # as "already used" for any picture segment that comes after it, and vice
    # versa.
    events: list[tuple[float, int, str, object]] = []
    for order, seg in enumerate(video):
        events.append((starts.get(id(seg), 0.0), order, "segment", seg))
    base = len(video)
    for order, item in enumerate(timeline.tracks.voice):
        m = VO_FILE_RE.search(item.file)
        if m:
            events.append((item.at, base + order, "voice", m))
    events.sort(key=lambda e: (e[0], e[1]))

    for _start, _order, kind, payload in events:
        if kind == "voice":
            m = payload
            clip = m.group("clip")
            v_in, v_out = float(m.group("in")), float(m.group("out"))
            _merge_into(used.setdefault(clip, []), v_in, v_out)
            continue

        seg = payload
        assert isinstance(seg, VideoSegment)
        if seg.mute_source:
            continue
        clip, a_in, a_out = seg.audio_source
        if a_out <= a_in + _EPS:
            continue
        bucket = used.setdefault(clip, [])
        overlap = _consumed_overlap(bucket, a_in, a_out)
        if overlap <= tolerance + _EPS:
            _merge_into(bucket, a_in, a_out)
            continue

        speech = _has_words(project, clip, a_in, a_out, words_cache)
        if not speech and allow_ambient_repeat:
            changes.append(
                f"{seg.id} {AMBIENT_TAG}: {clip} {_fmt(a_in)}-{_fmt(a_out)} repeats "
                f"{_fmt(overlap)}s of ambient audio already used — left as is (no speech)"
            )
            _merge_into(bucket, a_in, a_out)
            continue

        if seg.audio_from is None:
            # --- own-audio: picture and audio are the same range ---------
            new_in = min(_advance_past(bucket, a_in, a_out), a_out)
            remaining = a_out - new_in
            if remaining <= _EPS:
                changes.append(
                    f"{seg.id} muted: {clip} {_fmt(a_in)}-{_fmt(a_out)} is entirely a "
                    f"repeat of audio already used ({_fmt(overlap)}s) — kept the picture, "
                    "silenced the audio"
                )
                seg.mute_source = True
                mutated = True
                continue
            speed = seg.speed if seg.speed > 0 else 1.0
            if remaining / speed < min_shot - _EPS:
                changes.append(
                    f"{seg.id} dropped: only {_fmt(remaining)}s of {clip} is left after "
                    f"skipping {_fmt(overlap)}s already used — under the "
                    f"{min_shot:.2f}s minimum shot"
                )
                dropped.add(id(seg))
                mutated = True
                continue
            changes.append(
                f"{seg.id} in {_fmt(seg.in_)}→{_fmt(new_in)} ({clip}: skipped "
                f"{_fmt(overlap)}s already used)"
            )
            seg.in_ = round(new_in, 3)
            mutated = True
            _merge_into(bucket, seg.in_, seg.out)
            continue

        # --- audio_from: picture and audio are independent ----------------
        new_in = min(_advance_past(bucket, a_in, a_out), a_out)
        remaining = a_out - new_in
        if remaining <= _EPS:
            changes.append(
                f"{seg.id} audio_from muted: all of {clip} {_fmt(a_in)}-{_fmt(a_out)} is "
                f"already used ({_fmt(overlap)}s) — picture kept, narration silenced "
                "under music"
            )
            seg.audio_from = None
            seg.mute_source = True
            mutated = True
            continue

        picture_duration = seg.duration
        if remaining + _EPS < picture_duration:
            if remaining >= min_shot - _EPS:
                speed = seg.speed if seg.speed > 0 else 1.0
                new_out = round(seg.in_ + remaining * speed, 3)
                changes.append(
                    f"{seg.id} out {_fmt(seg.out)}→{_fmt(new_out)} and audio_from.in "
                    f"{_fmt(a_in)}→{_fmt(new_in)} ({clip}: skipped {_fmt(overlap)}s "
                    "already used)"
                )
                seg.out = new_out
                seg.audio_from.in_ = round(new_in, 3)
                seg.audio_from.out = round(new_in + remaining, 3)
                mutated = True
                _merge_into(bucket, *seg.audio_source[1:])
                continue
            changes.append(
                f"{seg.id} audio_from muted: only {_fmt(remaining)}s of {clip} narration "
                f"left after skipping {_fmt(overlap)}s already used — under the "
                f"{min_shot:.2f}s minimum, picture kept silent"
            )
            seg.audio_from = None
            seg.mute_source = True
            mutated = True
            continue

        changes.append(
            f"{seg.id} audio_from.in {_fmt(a_in)}→{_fmt(new_in)} ({clip}: skipped "
            f"{_fmt(overlap)}s already used)"
        )
        seg.audio_from.in_ = round(new_in, 3)
        seg.audio_from.out = round(min(a_out, new_in + picture_duration), 3)
        mutated = True
        _merge_into(bucket, *seg.audio_source[1:])

    if dropped:
        timeline.tracks.video = [s for s in video if id(s) not in dropped]

    if mutated:
        remap = _shift_map(before_positions, timeline.segment_positions())
        _retime_absolute_tracks(timeline, remap)
        log.info("audio ledger: %d fix(es) applied", sum(1 for c in changes if AMBIENT_TAG not in c))

    return timeline, changes


def ambient_repeat_count(changes: Sequence[str]) -> int:
    """How many entries in a change log are tolerated ambient repeats."""
    return sum(1 for line in changes if AMBIENT_TAG in line)


# ----------------------------------------------------------------------
# the read-only scan (used by ``ytedit qc`` rule 31)
# ----------------------------------------------------------------------
def find_duplicate_audio(
    timeline: Timeline, project: Project, settings: Settings | None = None
) -> list[DuplicateFinding]:
    """Report every audio range used more than once, without touching anything.

    Unlike :func:`dedupe_audio` this also flags collisions it could not have
    fixed on its own — e.g. two narration pickups (``tracks.voice``) that were
    independently extracted from overlapping clip ranges — so ``ytedit qc``
    can catch a bad hand-edit even when a fresh ``ytedit tidy`` was never run
    over it.

    Args:
        timeline: The EDL to scan.
        project: Project supplying transcripts.
        settings: Settings override; reads ``pacing.audio_dupe_tolerance``.

    Returns:
        Every collision found, each carrying whether it falls on speech.
    """
    cfg = settings or project.settings
    tolerance = max(0.0, float(cfg.get("pacing.audio_dupe_tolerance", 0.25)))
    words_cache: dict[str, list[Word]] = {}
    used: dict[str, list[tuple[float, float, str]]] = {}
    findings: list[DuplicateFinding] = []

    positions = {id(pos.segment): pos.start for pos in timeline.segment_positions()}
    events: list[tuple[float, int, str, object]] = []
    for order, seg in enumerate(timeline.tracks.video):
        events.append((positions.get(id(seg), 0.0), order, "segment", seg))
    base = len(timeline.tracks.video)
    for order, item in enumerate(timeline.tracks.voice):
        m = VO_FILE_RE.search(item.file)
        if m:
            events.append((item.at, base + order, "voice", (item, m)))
    events.sort(key=lambda e: (e[0], e[1]))

    for _start, _order, kind, payload in events:
        if kind == "voice":
            item, m = payload
            clip = m.group("clip")
            s, e = float(m.group("in")), float(m.group("out"))
            ident = item.id
        else:
            seg = payload
            assert isinstance(seg, VideoSegment)
            if seg.mute_source:
                continue
            clip, s, e = seg.audio_source
            ident = seg.id
        if e <= s + _EPS:
            continue

        pieces = used.setdefault(clip, [])
        collisions = [
            (max(s, cs), min(e, ce), oid)
            for cs, ce, oid in pieces
            if min(e, ce) - max(s, cs) > tolerance + _EPS
        ]
        if collisions:
            speech = _has_words(project, clip, s, e, words_cache)
            for os_, oe_, oid in collisions:
                findings.append(
                    DuplicateFinding(
                        clip=clip, start=round(os_, 3), end=round(oe_, 3),
                        speech=speech, ids=[oid, ident],
                    )
                )
        pieces.append((s, e, ident))

    return findings


__all__ = [
    "AMBIENT_TAG",
    "DuplicateFinding",
    "VO_FILE_RE",
    "ambient_repeat_count",
    "committed_audio_ranges",
    "dedupe_audio",
    "find_duplicate_audio",
]
