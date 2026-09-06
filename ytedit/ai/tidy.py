"""Deterministic cut tidying: air around speech, and jump-cut merging.

The planner (and a human dragging handles in the web editor) tends to cut
*exactly* on the first and last word of a sentence. That sounds clipped: the
breath before the first syllable and the decay of the last one are missing.
:func:`pad_segments_to_speech` fixes that after the fact, from the transcript,
without asking a model anything:

* a segment whose first word starts within ``pacing.speech_snap_window`` of its
  ``in`` gets ``in`` moved back to ``first_word.start - pacing.speech_pad_before``;
* a segment whose last word ends within that window of its ``out`` gets ``out``
  moved forward to ``last_word.end + pacing.speech_pad_after``;
* a cut that lands *inside* a word snaps to that word's boundary first;
* a cut that still lands **mid-sentence** is then extended to the sentence's own
  boundary: forward through the following words (while consecutive gaps stay
  under ``pacing.sentence_gap_max``) up to and including the first word ending
  in ``.?!…``, or backward to the first word of the sentence the cut opens in —
  plus the usual pad. The extra reach is capped at ``pacing.sentence_extend_max``
  per side. Cutting a narrator off in the middle of "Obecnie na wysokości cztery
  tysiące…" is the one thing a word-level pad cannot fix by itself;
* the move never crosses a neighbouring word (``GUARD`` seconds of clearance),
  an excised editor instruction, a rejected take, clip 0, the clip duration, or
  another segment of the same clip;
* B-roll (no words nearby) and ``mute_source`` segments are left alone.

Padding only ever *adds* air — a segment that already has more than the pad is
never trimmed back to it.

Afterwards consecutive segments cut from the same clip whose clip-time gap is
under ``pacing.merge_gap`` are merged: such a jump cut is not perceived as an
edit, only as a glitch.

The pass runs at the end of :func:`ytedit.ai.plan.build_timeline` (so every
fresh plan is tidy) and can be re-run over an existing EDL with ``ytedit tidy``.
"""

from __future__ import annotations

import bisect
import json
import shutil
from typing import Any, Callable, NamedTuple, Sequence

from ytedit.config import Settings
from ytedit.log import get_logger
from ytedit.project import Project, utcnow
from ytedit.timeline import SegmentPosition, Timeline, VideoSegment

log = get_logger(__name__)

STAGE = "tidy"

_EPS = 1e-6

#: Minimum clearance kept from a neighbouring word when padding (seconds).
GUARD: float = 0.05

#: How far a cut may sit inside a word before it counts as cutting the word.
_INSIDE_WORD: float = 0.02

#: A transcript word ending in one of these closes a sentence.
SENTENCE_END: str = ".?!…"

#: Marker used in the change log for a sentence-boundary snap (see
#: :func:`sentence_snap_count`).
SENTENCE_SNAP_TAG: str = "sentence-snap"

#: Backups of ``plan/timeline.json`` kept in ``plan/history/`` (mirrors server/app.py).
MAX_HISTORY = 20


class TidyError(RuntimeError):
    """Raised when the tidy pass has nothing to work on."""


class Word(NamedTuple):
    """One transcript word in clip time."""

    s: float
    e: float
    text: str


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------
def load_words(project: Project, clip_id: str) -> list[Word]:
    """Read ``transcripts/<clip>.json`` as sorted :class:`Word` spans.

    Args:
        project: Project holding the transcripts.
        clip_id: Clip id such as ``c004``.

    Returns:
        Words in clip time, sorted by start; empty when there is no transcript.
    """
    path = project.transcript_path(clip_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - bad transcript
        log.warning("unreadable transcript %s: %s", path, exc)
        return []
    words: list[Word] = []
    for raw in data.get("words", []):
        if not isinstance(raw, dict):
            continue
        if raw.get("type") not in (None, "word"):
            continue
        start = raw.get("s", raw.get("start"))
        end = raw.get("e", raw.get("end"))
        if start is None or end is None:
            continue
        try:
            s, e = float(start), float(end)
        except (TypeError, ValueError):  # pragma: no cover - malformed transcript
            continue
        if e <= s:
            continue
        words.append(Word(s, e, str(raw.get("t", raw.get("text", ""))).strip()))
    words.sort(key=lambda w: (w.s, w.e))
    return words


def excised_ranges(project: Project, clip_id: str) -> list[tuple[float, float]]:
    """Ranges of ``analysis/<clip>.json`` that must never reappear in the cut.

    Spoken editor instructions plus every take attempt the analysis rejected —
    the same rules :func:`ytedit.ai.plan.build_timeline` cuts around, re-read
    here so ``ytedit tidy`` enforces them on a hand-edited timeline too.
    """
    # Imported lazily: plan.py imports this module.
    from ytedit.ai.plan import instruction_ranges, rejected_take_ranges

    path = project.analysis_path(clip_id)
    if not path.exists():
        return []
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - bad analysis
        log.warning("unreadable analysis %s: %s", path, exc)
        return []
    if not isinstance(entry, dict):
        return []
    ranges = list(instruction_ranges(entry)) + list(rejected_take_ranges(entry))
    return sorted((s, e) for s, e in ranges if e > s)


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def _clip_bounds(
    video: Sequence[VideoSegment], index: int
) -> tuple[float, float]:
    """Return ``(low, high)`` clip-time limits set by other cuts of the same clip.

    ``low`` is the end of the nearest earlier segment of that clip (a padded
    ``in`` must not run back into it) and ``high`` the start of the nearest
    later one.
    """
    seg = video[index]
    low = 0.0
    high = float("inf")
    for i, other in enumerate(video):
        if i == index or other.clip != seg.clip:
            continue
        if other.out <= seg.in_ + _EPS:
            low = max(low, float(other.out))
        if other.in_ >= seg.out - _EPS:
            high = min(high, float(other.in_))
    return low, high


def _straddling(words: Sequence[Word], t: float) -> Word | None:
    """Return the word the instant ``t`` falls inside, if any."""
    for word in words:
        if word.s < t - _INSIDE_WORD and word.e > t + _INSIDE_WORD:
            return word
    return None


def _fmt(value: float) -> str:
    """Format a timecode for the change log."""
    return f"{value:.2f}"


def _pad_in(
    seg: VideoSegment,
    words: Sequence[Word],
    cuts: Sequence[tuple[float, float]],
    low: float,
    pad_before: float,
    window: float,
) -> tuple[float, Word, bool] | None:
    """Compute the new ``in`` for one segment, or ``None`` to leave it alone.

    Returns:
        ``(new_in, anchor_word, snapped)`` where ``snapped`` says the old ``in``
        sat inside the anchor word.
    """
    snapped = False
    anchor = _straddling(words, seg.in_)
    if anchor is not None:
        snapped = True
    else:
        anchor = next(
            (w for w in words if w.s >= seg.in_ - _EPS and w.s < seg.out - _EPS), None
        )
        if anchor is None or anchor.s - seg.in_ > window:
            return None

    desired = anchor.s - pad_before
    limit = max(0.0, float(low))
    previous = [w for w in words if w.e <= anchor.s + _EPS and w is not anchor]
    if previous:
        limit = max(limit, previous[-1].e + GUARD)
    for cut_s, cut_e in cuts:
        if cut_e <= anchor.s + _EPS:
            limit = max(limit, cut_e)
    new_in = max(desired, limit)
    if new_in >= seg.in_ - 1e-3:
        return None
    return round(new_in, 3), anchor, snapped


def _pad_out(
    seg: VideoSegment,
    words: Sequence[Word],
    cuts: Sequence[tuple[float, float]],
    high: float,
    duration: float,
    pad_after: float,
    window: float,
) -> tuple[float, Word, bool] | None:
    """Compute the new ``out`` for one segment, or ``None`` to leave it alone."""
    snapped = False
    anchor = _straddling(words, seg.out)
    if anchor is not None:
        snapped = True
    else:
        tail = [w for w in words if w.e <= seg.out + _EPS and w.e > seg.in_ + _EPS]
        if not tail:
            return None
        anchor = tail[-1]
        if seg.out - anchor.e > window:
            return None

    desired = anchor.e + pad_after
    limit = float(high)
    if duration > 0:
        limit = min(limit, duration)
    following = [w for w in words if w.s >= anchor.e - _EPS and w is not anchor]
    if following:
        limit = min(limit, following[0].s - GUARD)
    for cut_s, cut_e in cuts:
        if cut_s >= anchor.e - _EPS:
            limit = min(limit, cut_s)
    new_out = min(desired, limit)
    if new_out <= seg.out + 1e-3:
        return None
    return round(new_out, 3), anchor, snapped


# ----------------------------------------------------------------------
# sentence-boundary snapping
# ----------------------------------------------------------------------
def ends_sentence(word: Word) -> bool:
    """True when the word's text closes a sentence (``.``, ``?``, ``!``, ``…``)."""
    text = word.text.strip()
    return bool(text) and text[-1] in SENTENCE_END


def has_sentence_marks(words: Sequence[Word]) -> bool:
    """True when a transcript punctuates at all, so sentences can be located.

    A transcript that never writes a full stop (a hand-made one, or an STT run
    with punctuation disabled) carries no sentence information: snapping on it
    would glue every word within ``sentence_gap_max`` of the cut into one
    "sentence" and swallow whole takes. Sentence snapping is skipped for such a
    clip and only the word-level padding applies.
    """
    return any(ends_sentence(w) for w in words)


def _inside(seg: VideoSegment, words: Sequence[Word]) -> list[int]:
    """Indices of the words lying wholly inside ``[seg.in, seg.out]``."""
    return [
        i
        for i, w in enumerate(words)
        if w.s >= seg.in_ - _EPS and w.e <= seg.out + _EPS
    ]


def _snap_out_to_sentence(
    seg: VideoSegment,
    words: Sequence[Word],
    cuts: Sequence[tuple[float, float]],
    high: float,
    duration: float,
    pad_after: float,
    gap_max: float,
    extend_max: float,
) -> tuple[float, Word] | None:
    """Extend ``out`` to the end of the sentence the cut lands inside.

    Returns ``(new_out, last_word_of_the_sentence)``, or ``None`` when the cut
    already sits on a sentence boundary, the next word is further than
    ``gap_max`` away, or nothing can be gained without crossing a guard.
    """
    indices = _inside(seg, words)
    if not indices:
        return None
    last = indices[-1]
    if ends_sentence(words[last]):
        return None

    tail: list[Word] = []
    previous = words[last]
    for word in words[last + 1:]:
        if word.s - previous.e > gap_max + _EPS:
            break
        tail.append(word)
        previous = word
        if ends_sentence(word):
            break
    if not tail:
        return None
    anchor = tail[-1]

    desired = anchor.e + pad_after
    limit = min(float(high), seg.out + extend_max)
    if duration > 0:
        limit = min(limit, duration)
    following = [w for w in words if w.s >= anchor.e - _EPS]
    if following:
        limit = min(limit, following[0].s - GUARD)
    for cut_s, _cut_e in cuts:
        if cut_s >= words[last].e - _EPS:
            limit = min(limit, cut_s)
    new_out = min(desired, limit)
    if new_out <= seg.out + 1e-3:
        return None
    return round(new_out, 3), anchor


def _snap_in_to_sentence(
    seg: VideoSegment,
    words: Sequence[Word],
    cuts: Sequence[tuple[float, float]],
    low: float,
    pad_before: float,
    gap_max: float,
    extend_max: float,
) -> tuple[float, Word] | None:
    """Move ``in`` back to the start of the sentence the cut opens inside."""
    indices = _inside(seg, words)
    if not indices:
        return None
    first = indices[0]
    if first == 0:
        return None
    if ends_sentence(words[first - 1]):
        return None
    if words[first].s - words[first - 1].e > gap_max + _EPS:
        return None

    head: list[Word] = []
    current = words[first]
    for i in range(first - 1, -1, -1):
        word = words[i]
        if current.s - word.e > gap_max + _EPS or ends_sentence(word):
            break
        head.append(word)
        current = word
    if not head:
        return None
    anchor = head[-1]

    desired = anchor.s - pad_before
    limit = max(0.0, float(low), seg.in_ - extend_max)
    previous = [w for w in words if w.e <= anchor.s + _EPS]
    if previous:
        limit = max(limit, previous[-1].e + GUARD)
    for _cut_s, cut_e in cuts:
        if cut_e <= words[first].s + _EPS:
            limit = max(limit, cut_e)
    new_in = max(desired, limit)
    if new_in >= seg.in_ - 1e-3:
        return None
    return round(new_in, 3), anchor


def sentence_snap_count(changes: Sequence[str]) -> int:
    """How many distinct segments a change log sentence-snapped."""
    return len({line.split(" ", 1)[0] for line in changes if SENTENCE_SNAP_TAG in line})


# ----------------------------------------------------------------------
# the pass
# ----------------------------------------------------------------------
def pad_segments_to_speech(
    timeline: Timeline, project: Project, settings: Settings | None = None
) -> tuple[Timeline, list[str]]:
    """Give every speech segment air, then merge jump cuts inside one clip.

    Args:
        timeline: The EDL to tidy; it is modified in place and returned.
        project: Project supplying transcripts, analyses and clip durations.
        settings: Settings override (defaults to ``project.settings``); reads
            ``pacing.speech_pad_before``, ``pacing.speech_pad_after``,
            ``pacing.speech_snap_window`` and ``pacing.merge_gap``.

    Padding/merging only ever touches ``tracks.video``, but growing a segment's
    duration shifts the absolute start of everything after it. Captions, music
    cues and voice pickups are pinned to absolute timeline time, so once this
    pass has moved anything, they are re-timed by the same amount (see
    :func:`_shift_map`) — otherwise a plan re-tidied a second time (a human
    edit, then ``ytedit tidy``) would drift out of sync with its own audio.

    Returns:
        ``(timeline, changes)`` — ``changes`` is a human-readable list such as
        ``["s004 in 3.10→2.82 (pad before 'Yo')"]``, empty when nothing moved.
    """
    cfg = settings or project.settings
    pad_before = max(0.0, float(cfg.get("pacing.speech_pad_before", 0.30)))
    pad_after = max(0.0, float(cfg.get("pacing.speech_pad_after", 0.45)))
    window = max(0.0, float(cfg.get("pacing.speech_snap_window", 0.6)))
    merge_gap = max(0.0, float(cfg.get("pacing.merge_gap", 0.15)))
    gap_max = max(0.0, float(cfg.get("pacing.sentence_gap_max", 1.2)))
    extend_max = max(0.0, float(cfg.get("pacing.sentence_extend_max", 8.0)))

    clips = project.load_state().get("clips", {})
    words_cache: dict[str, list[Word]] = {}
    cuts_cache: dict[str, list[tuple[float, float]]] = {}

    def words_for(clip_id: str) -> list[Word]:
        if clip_id not in words_cache:
            words_cache[clip_id] = load_words(project, clip_id)
        return words_cache[clip_id]

    def cuts_for(clip_id: str) -> list[tuple[float, float]]:
        if clip_id not in cuts_cache:
            cuts_cache[clip_id] = excised_ranges(project, clip_id)
        return cuts_cache[clip_id]

    changes: list[str] = []
    video = timeline.tracks.video
    before_positions = timeline.segment_positions()

    for index, seg in enumerate(video):
        if seg.mute_source:
            continue
        words = words_for(seg.clip)
        if not words:
            continue
        cuts = cuts_for(seg.clip)
        low, high = _clip_bounds(video, index)
        duration = float((clips.get(seg.clip) or {}).get("duration") or 0.0)

        moved_in = _pad_in(seg, words, cuts, low, pad_before, window)
        if moved_in is not None:
            new_in, anchor, snapped = moved_in
            changes.append(
                f"{seg.id} in {_fmt(seg.in_)}→{_fmt(new_in)} "
                f"({'snap+pad' if snapped else 'pad'} before '{anchor.text}')"
            )
            seg.in_ = new_in

        moved_out = _pad_out(seg, words, cuts, high, duration, pad_after, window)
        if moved_out is not None:
            new_out, anchor, snapped = moved_out
            changes.append(
                f"{seg.id} out {_fmt(seg.out)}→{_fmt(new_out)} "
                f"({'snap+pad' if snapped else 'pad'} after '{anchor.text}')"
            )
            seg.out = new_out

        # The word-level pad can still leave the cut in the middle of a
        # sentence; finish the thought before handing over to the next shot.
        # Bounds are recomputed because in/out just moved.
        if not has_sentence_marks(words):
            continue
        low, high = _clip_bounds(video, index)
        snap_out = _snap_out_to_sentence(
            seg, words, cuts, high, duration, pad_after, gap_max, extend_max
        )
        if snap_out is not None:
            new_out, anchor = snap_out
            changes.append(
                f"{seg.id} out {_fmt(seg.out)}→{_fmt(new_out)} "
                f"({SENTENCE_SNAP_TAG} to '{anchor.text}')"
            )
            seg.out = new_out
        snap_in = _snap_in_to_sentence(
            seg, words, cuts, low, pad_before, gap_max, extend_max
        )
        if snap_in is not None:
            new_in, anchor = snap_in
            changes.append(
                f"{seg.id} in {_fmt(seg.in_)}→{_fmt(new_in)} "
                f"({SENTENCE_SNAP_TAG} to '{anchor.text}')"
            )
            seg.in_ = new_in

    snapped_segments = sentence_snap_count(changes)
    if snapped_segments:
        log.info("sentence-snapped %d segment(s) to their sentence boundaries",
                 snapped_segments)

    timeline.tracks.video = _merge_jump_cuts(video, cuts_cache, merge_gap, changes)

    if changes:
        remap = _shift_map(before_positions, timeline.segment_positions())
        _retime_absolute_tracks(timeline, remap)

    return timeline, changes


def _shift_map(
    before: Sequence[SegmentPosition], after: Sequence[SegmentPosition]
) -> Callable[[float], float]:
    """Build a step function mapping an old absolute time to its new one.

    Padding only ever grows a segment (never shrinks it — "padding only ever
    adds air"), which shifts every later segment's absolute start by the same
    amount; merging drops a segment but keeps the one it merged into, never
    reordering anything. Both effects reduce to one lookup: the shift in force
    at time ``t`` is that of the last original segment starting at or before
    ``t`` that is still in the timeline — or, when it was merged away, that of
    the nearest earlier surviving one (equal in practice, since a jump-cut
    merge only joins segments a few hundred milliseconds apart).

    Args:
        before: ``segment_positions()`` captured before this pass touched
            anything.
        after: ``segment_positions()`` of the padded/merged result.

    Returns:
        A function from an old absolute time to its new one.
    """
    after_by_id = {id(pos.segment): pos.start for pos in after}
    starts: list[float] = []
    shifts: list[float] = []
    last_shift = 0.0
    for pos in before:
        new_start = after_by_id.get(id(pos.segment))
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


def _retime_absolute_tracks(timeline: Timeline, remap: Callable[[float], float]) -> None:
    """Shift every absolute-time item downstream of a padding-induced move."""
    for cap in timeline.tracks.captions:
        cap.at = remap(cap.at)
        cap.end = remap(cap.end)
    for cue in timeline.tracks.music:
        cue.at = remap(cue.at)
        cue.end = remap(cue.end)
    for item in timeline.tracks.voice:
        item.at = remap(item.at)
        if item.end is not None:
            item.end = remap(item.end)


def _mergeable(prev: VideoSegment, seg: VideoSegment) -> bool:
    """True when two segments differ only in their in/out points."""
    return (
        prev.clip == seg.clip
        and prev.mute_source == seg.mute_source
        and abs(prev.speed - seg.speed) < 1e-6
        and prev.grade == seg.grade
        and prev.transform.fit == seg.transform.fit
        and abs(prev.source_audio_gain_db - seg.source_audio_gain_db) < 1e-6
        and seg.transition_in.type == "cut"
    )


def _merge_jump_cuts(
    video: Sequence[VideoSegment],
    cuts_cache: dict[str, list[tuple[float, float]]],
    merge_gap: float,
    changes: list[str],
) -> list[VideoSegment]:
    """Join consecutive same-clip segments separated by less than ``merge_gap``."""
    out: list[VideoSegment] = []
    for seg in video:
        if out:
            prev = out[-1]
            gap = seg.in_ - prev.out
            if _mergeable(prev, seg) and -_EPS <= gap < merge_gap - _EPS:
                excised = any(
                    cut_s < seg.in_ - _EPS and cut_e > prev.out + _EPS
                    for cut_s, cut_e in cuts_cache.get(seg.clip, [])
                )
                if not excised:
                    changes.append(
                        f"{seg.id} merged into {prev.id} "
                        f"({gap * 1000:.0f} ms gap in {seg.clip})"
                    )
                    prev.out = round(max(prev.out, seg.out), 3)
                    continue
        out.append(seg)
    return list(out)


# ----------------------------------------------------------------------
# the ``ytedit tidy`` stage
# ----------------------------------------------------------------------
def backup_timeline(project: Project) -> str | None:
    """Copy ``plan/timeline.json`` into ``plan/history/``, pruning old backups.

    Uses the same naming as the web editor (``timeline_<stamp>.json``) so both
    write into one history.
    """
    src = project.timeline_file
    if not src.exists():
        return None
    history = project.plan_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().replace(":", "").replace("-", "").replace("+0000", "Z")
    target = history / f"timeline_{stamp}.json"
    n = 1
    while target.exists():
        target = history / f"timeline_{stamp}_{n}.json"
        n += 1
    shutil.copy2(src, target)
    backups = sorted(history.glob("timeline_*.json"), key=lambda p: p.stat().st_mtime)
    for old in backups[:-MAX_HISTORY]:
        old.unlink(missing_ok=True)
    return target.name


def tidy(
    project: Project, dry_run: bool = False, force: bool = False
) -> dict[str, Any]:
    """Run :func:`pad_segments_to_speech` over ``plan/timeline.json``.

    Args:
        project: Project whose timeline is tidied.
        dry_run: Report the changes without writing anything.
        force: Overwrite a human-edited ``timeline.json`` instead of writing
            ``timeline.draft.json`` next to it.

    Returns:
        ``{"changes", "written", "backup", "dry_run", "edited_by_human",
        "duration_before", "duration_after", "issues", "sentence_snapped",
        "overlaid"}``.

    Raises:
        TidyError: When the project has no timeline yet.
    """
    # Imported lazily: overlay.py imports this module for its re-timing helpers.
    from ytedit.ai.overlay import overlay_cutaways

    if not project.timeline_file.exists():
        raise TidyError(
            f"no timeline at {project.timeline_file} — run `ytedit plan {project.slug}` first"
        )
    timeline = Timeline.load(project.timeline_file)
    human_edited = bool(timeline.meta.edited_by_human)
    before = timeline.duration()

    timeline, changes = pad_segments_to_speech(timeline, project)
    snapped = sentence_snap_count(changes)
    timeline, overlaid = overlay_cutaways(timeline, project)
    changes = changes + overlaid
    result: dict[str, Any] = {
        "changes": changes,
        "written": None,
        "backup": None,
        "dry_run": dry_run,
        "edited_by_human": human_edited,
        "duration_before": before,
        "duration_after": timeline.duration(),
        "issues": [],
        "sentence_snapped": snapped,
        "overlaid": len(overlaid),
    }
    if not changes or dry_run:
        return result

    result["issues"] = timeline.validate(project)
    result["backup"] = backup_timeline(project)
    write_to_draft = human_edited and not force
    target = project.plan_dir / ("timeline.draft.json" if write_to_draft else "timeline.json")
    timeline.save(target)
    result["written"] = project.rel(target)
    if write_to_draft:
        log.warning(
            "[clip]%s[/] timeline.json is human-edited — tidied copy written to %s "
            "(diff it, or re-run with --force)",
            project.slug,
            target.name,
        )
    project.set_stage(STAGE, "done", changes=len(changes))
    return result


__all__ = [
    "GUARD",
    "SENTENCE_END",
    "SENTENCE_SNAP_TAG",
    "TidyError",
    "Word",
    "backup_timeline",
    "ends_sentence",
    "has_sentence_marks",
    "excised_ranges",
    "load_words",
    "pad_segments_to_speech",
    "sentence_snap_count",
    "tidy",
]
