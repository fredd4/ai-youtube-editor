"""Overlay cutaways: picture from one clip, narration from the one underneath.

A take split into two contiguous pieces around a cutaway is the standard way to
hide a jump cut, but rendering it literally means the narration stops dead for
the length of the cutaway and resumes mid-breath. :func:`overlay_cutaways`
rewrites that pattern deterministically, after the speech padding in
:mod:`ytedit.ai.tidy` has settled the cut points:

``A`` (a speech segment) → ``C_1..C_k`` (cutaways on other clips) → ``A2`` (the
same clip resuming where ``A`` left off) becomes ``A`` → the same cutaways, each
carrying ``audio_from`` over the stretch of ``A``'s clip that would otherwise
have been skipped → ``A2`` starting after that stretch. The narration runs
unbroken; only the picture cuts away.

The pattern is only rewritten when ``A2`` really is a continuation: its ``in``
must sit within ``pacing.sentence_gap_max`` of ``A``'s ``out`` (the same
threshold the sentence snapping uses), and the cutaways must be long enough to
cover the gap between them. An ``A2`` that jumps somewhere else in the clip —
the planner deliberately dropping a whole sentence — is left exactly as it is.

Before committing to a rewrite, the stretch of ``A``'s clip the cutaways would
carry is checked against :func:`ytedit.ai.ledger.committed_audio_ranges` for
everything *earlier* in the track: if that stretch is already claimed there
(typically an ``audio_from`` :mod:`ytedit.ai.ledger` muted in a previous
``ytedit tidy`` round for being a duplicate), the whole run is left as it is
rather than re-creating the exact hand-off the ledger will just remove again.
Reassigning a cutaway's ``audio_from`` (or moving ``A2``) to the value it
already has is a no-op — nothing is logged and nothing counts as a change —
so a converged timeline stays quiet on a repeat run.

Dropping or shortening a segment moves everything after it, so captions, music
cues and voice pickups are re-timed with the same machinery the padding pass
uses (:func:`ytedit.ai.tidy._shift_map`).

The pass runs at the end of :func:`ytedit.ai.plan.build_timeline` and again on
every ``ytedit tidy``.

:func:`close_silent_interruptions` handles the pattern :func:`overlay_cutaways`
deliberately leaves alone: ``A`` → ``C_1..C_k`` (cutaways carrying no audio at
all — muted, no ``audio_from``) → ``A2`` (the same clip resuming *later* than
``A`` left off, and further than the cutaways or ``sentence_gap_max`` would
excuse). That used to mean the narrator was silenced for however long the
cutaways lasted and then resumed mid-breath — the user's fourth-round verdict: "if
you did not mute me... or if the cutaway came at the end of the thought".
Story continuity now wins over the shot-length ceilings:

* a skip of at most ``pacing.story_fill_max_s`` is a **fill** — the cutaways
  carry ``A``'s clip continuously from ``A.out``, so every word in the gap is
  still heard (just under different pictures), and ``A2`` resumes wherever
  that continuous narration lands, not at its original ``in``;
* a longer skip is a **J-cut** — ``A`` is first extended (or, failing that,
  retracted) to a real sentence end with the same machinery
  :mod:`ytedit.ai.tidy` uses for padding, so the thought actually finishes;
  the cutaways then carry ``A2``'s own opening words early, and ``A2``'s
  picture starts once its audio has caught up. Either way the audio never
  stops.

A cutaway that already carries ``audio_from``, a ``VO picture for`` montage
segment, or a run sitting under a ``tracks.voice`` pickup is left exactly as
it is — this pass only closes a run that is genuinely silent right now.
:func:`find_silent_interruptions` is the read-only counterpart (``ytedit qc``
rule 36): it reports any such pattern still standing, the way
:func:`ytedit.ai.ledger.find_duplicate_audio` reports rule 31.
"""

from __future__ import annotations

import re
from typing import Sequence

from ytedit.ai.ledger import _consumed_overlap, committed_audio_ranges
from ytedit.ai.tidy import (
    Word,
    _clip_bounds,
    _out_tail,
    _retract_out_to_previous_sentence,
    _snap_out_to_sentence,
    excised_ranges,
    has_sentence_marks,
    load_words,
)
from ytedit.config import Settings
from ytedit.log import get_logger
from ytedit.project import Project
from ytedit.timeline import AudioFrom, SegmentPosition, Timeline, VideoSegment

log = get_logger(__name__)

_EPS = 1e-6

#: Roles that may serve as the picture of an overlay cutaway (mirrors the style
#: of ``ai.plan.AROLL_ROLE_RE``; deliberately disjoint from it, so an overlay
#: segment still breaks an A-roll run in the pacing report).
CUTAWAY_ROLE_RE = re.compile(r"cutaway|b-?roll", re.IGNORECASE)

#: Prefix :func:`ytedit.ai.plan._build_voice_over_segment` puts on the notes of a
#: picture cut standing in for a narration pickup. Those already get their audio
#: from ``tracks.voice``, so they never become overlay cutaways.
VO_PICTURE_NOTE = "VO picture for "


def _has_speech(project: Project, seg: VideoSegment, cache: dict[str, list[Word]]) -> bool:
    """True when the segment's own clip has a transcript word inside the cut."""
    if seg.clip not in cache:
        cache[seg.clip] = load_words(project, seg.clip)
    return any(w.e > seg.in_ + _EPS and w.s < seg.out - _EPS for w in cache[seg.clip])


def _is_cutaway(seg: VideoSegment, clip: str) -> bool:
    """True when ``seg`` can serve as the picture of an overlay cutaway."""
    if seg.clip == clip:
        return False
    if seg.notes.startswith(VO_PICTURE_NOTE):
        return False
    return bool(CUTAWAY_ROLE_RE.search(seg.role or ""))


def _fmt(value: float) -> str:
    """Format a timecode for the change log."""
    return f"{value:.2f}"


def overlay_cutaways(
    timeline: Timeline, project: Project, settings: Settings | None = None
) -> tuple[Timeline, list[str]]:
    """Keep the narration running under cutaways that split one take in two.

    Args:
        timeline: The EDL to rewrite; it is modified in place and returned.
        project: Project supplying transcripts and clip durations.
        settings: Settings override (defaults to ``project.settings``); reads
            ``pacing.sentence_gap_max`` and ``pacing.min_shot_seconds``.

    Returns:
        ``(timeline, changes)`` — ``changes`` is a human-readable list, empty
        when no overlay pattern was found.
    """
    cfg = settings or project.settings
    gap_max = max(0.0, float(cfg.get("pacing.sentence_gap_max", 1.2)))
    min_shot = max(0.0, float(cfg.get("pacing.min_shot_seconds", 0.8)))
    tolerance = max(0.0, float(cfg.get("pacing.audio_dupe_tolerance", 0.25)))

    clips = project.load_state().get("clips", {})
    words_cache: dict[str, list[Word]] = {}
    changes: list[str] = []
    patterns = 0
    mutations = 0

    video: Sequence[VideoSegment] = timeline.tracks.video
    before_positions = timeline.segment_positions()
    dropped: set[int] = set()

    # Per-clip "audio consumed up to" cursor, carried across the whole pass —
    # not just within one A → cutaways → A2 pattern. A clip can reappear later
    # in the timeline outside any cutaway run (a stray a-roll piece the
    # planner or a human edit left behind); without this, that reappearance
    # would be treated as a fresh take even though a chain resolved earlier in
    # this same call already committed its audio range.
    consumed: dict[str, float] = {}

    i = 0
    while i < len(video) - 1:
        a = video[i]
        if a.mute_source or not _has_speech(project, a, words_cache):
            i += 1
            continue

        # --- catch a stray reappearance before treating A as a fresh take --
        cursor_for_a = consumed.get(a.clip, 0.0)
        if a.audio_from is None and a.in_ < cursor_for_a - _EPS:
            new_a_in = min(cursor_for_a, a.out)
            speed_a = a.speed if a.speed > 0 else 1.0
            if (a.out - new_a_in) / speed_a < min_shot - _EPS:
                changes.append(
                    f"{a.id} dropped: {a.clip} {_fmt(a.in_)}-{_fmt(a.out)} is entirely "
                    f"covered by narration an earlier cutaway run already carried "
                    f"(up to {_fmt(cursor_for_a)}s)"
                )
                dropped.add(id(a))
                mutations += 1
                i += 1
                continue
            changes.append(
                f"{a.id} in {_fmt(a.in_)}→{_fmt(new_a_in)} (an earlier cutaway run "
                f"already carried {a.clip} audio up to {_fmt(cursor_for_a)}s)"
            )
            a.in_ = round(new_a_in, 3)
            mutations += 1

        # --- the run of cutaways immediately after A ----------------------
        j = i + 1
        while j < len(video) and _is_cutaway(video[j], a.clip):
            j += 1
        if j == i + 1 or j >= len(video):
            i += 1
            continue

        a2 = video[j]
        if a2.clip != a.clip or a2.mute_source:
            i = j
            continue

        # --- detection: is this the same take resuming? --------------------
        if a2.in_ > a.out + gap_max + _EPS:
            log.debug(
                "%s → %s: %.2fs jump in %s is a deliberate skip, not a continuation",
                a.id, a2.id, a2.in_ - a.out, a.clip,
            )
            i = j
            continue

        run = list(video[i + 1:j])
        covered = sum(seg.duration for seg in run)
        if a2.in_ > a.out + covered + _EPS:
            # The continuation starts past everything the cutaways could cover:
            # stretching the narration would replay audio that was cut on
            # purpose. Leave the whole run alone.
            log.debug(
                "%s → %s: continuation starts %.2fs past the %.2fs of cutaways — left alone",
                a.id, a2.id, a2.in_ - a.out, covered,
            )
            i = j
            continue

        # --- rewrite --------------------------------------------------------
        # Never start the narration handoff behind what an earlier, unrelated
        # pattern already committed for this clip.
        cursor = max(a.out, consumed.get(a.clip, 0.0))

        # Guard against re-creating an audio_from the ledger already removed
        # (in an earlier ``ytedit tidy`` round) for being a duplicate: if the
        # stretch this run would hand to the cutaways is already claimed by
        # something earlier in the track (another segment's own audio,
        # another audio_from, or a voice pickup), leave the run exactly as it
        # is — muted stays muted — instead of flip-flopping with
        # dedupe_audio every round. Only *earlier* claims count: a later
        # segment reappearing over this same stretch is the "stray
        # reappearance" case handled above, which yields to this pattern, not
        # the other way round.
        committed = committed_audio_ranges(timeline, video_slice=slice(0, i)).get(a.clip, [])
        already_used = _consumed_overlap(committed, cursor, cursor + covered)
        if already_used > tolerance + _EPS:
            log.debug(
                "%s → %s: %.2fs of the %.2fs %s stretch this run would carry is "
                "already used earlier in the track — left as is",
                a.id, a2.id, already_used, covered, a.clip,
            )
            i = j
            continue

        pattern_changed = False
        for cutaway in run:
            end = round(cursor + cutaway.duration, 3)
            already_set = (
                not cutaway.mute_source
                and cutaway.audio_from is not None
                and cutaway.audio_from.clip == a.clip
                and abs(cutaway.audio_from.in_ - cursor) <= _EPS
                and abs(cutaway.audio_from.out - end) <= _EPS
            )
            if not already_set:
                cutaway.audio_from = AudioFrom(clip=a.clip, **{"in": round(cursor, 3)}, out=end)
                cutaway.mute_source = False
                changes.append(
                    f"{cutaway.id} audio_from {a.clip} {_fmt(cutaway.audio_from.in_)}→"
                    f"{_fmt(end)} (narration continues under the cutaway)"
                )
                pattern_changed = True
            cursor = end

        new_in = round(cursor, 3)
        speed = a2.speed if a2.speed > 0 else 1.0
        if (a2.out - new_in) / speed < min_shot - _EPS:
            # What is left of the continuation is a sub-second flash; show the
            # last cutaway for that bit longer instead and drop it. The
            # narration ends exactly where the planner ended it (A2.out) —
            # extending the picture further than that would manufacture
            # audio nobody wrote.
            last = run[-1]
            audio_available = max(0.0, a2.out - last.audio_from.in_)
            wanted = last.duration + a2.duration
            clip_duration = float((clips.get(last.clip) or {}).get("duration") or 0.0)
            last_speed = last.speed if last.speed > 0 else 1.0
            new_out = last.in_ + wanted * last_speed
            new_out = min(new_out, last.in_ + audio_available * last_speed)
            if clip_duration > 0:
                new_out = min(new_out, clip_duration)
            # Never shrink the picture the earlier cutaways in this run already
            # got: a run whose cumulative coverage already overshot A2.out (a
            # pre-existing inconsistency — usually a hand edit desyncing an
            # already-overlaid chain) leaves nothing to extend into, but that
            # is a reason to leave ``last`` exactly as it is, not to shorten it.
            new_out = max(new_out, last.out)
            if abs(new_out - last.out) > 1e-3:
                changes.append(
                    f"{last.id} out {_fmt(last.out)}→{_fmt(new_out)} and {a2.id} dropped "
                    f"({_fmt((a2.out - new_in) / speed)}s left of it is under the "
                    f"{min_shot:.2f}s minimum shot)"
                )
                last.out = round(new_out, 3)
                pattern_changed = True
            capped_audio_out = min(last.audio_from.in_ + last.duration, a2.out)
            if capped_audio_out > last.audio_from.in_ + _EPS:
                if abs(capped_audio_out - last.audio_from.out) > 1e-3:
                    last.audio_from.out = round(capped_audio_out, 3)
                    pattern_changed = True
            # else: capping to A2.out would make the range invalid — the run's
            # earlier cutaways already used more of the clip than the
            # planner's sentence end allows. Leave ``audio_from.out`` at what
            # the main loop above computed; ``ytedit.ai.ledger`` catches and
            # resolves any resulting overlap as a last resort.
            consumed[a.clip] = max(consumed.get(a.clip, 0.0), last.audio_from.out)
            if id(a2) not in dropped:
                dropped.add(id(a2))
                pattern_changed = True
            i = j + 1
        else:
            if abs(a2.in_ - new_in) > 1e-3:
                changes.append(
                    f"{a2.id} in {_fmt(a2.in_)}→{_fmt(new_in)} "
                    f"(resumes after {_fmt(covered)}s of overlay cutaway)"
                )
                a2.in_ = new_in
                pattern_changed = True
            consumed[a.clip] = max(consumed.get(a.clip, 0.0), new_in)
            i = j
        if pattern_changed:
            patterns += 1
            mutations += 1

    if mutations:
        # Route every drop through the timeline's own edit API — it re-times
        # everything downstream by uid (stable across the drop) instead of
        # this pass doing its own shift-map bookkeeping, and re-resolves
        # anchors. ``renumber=False``: ids are re-assigned by the caller
        # (``ytedit.ai.plan.build_timeline`` or ``ytedit.ai.tidy.tidy``)
        # once every pass in the round has run, not after each one.
        dropped_uids = {s.uid for s in video if id(s) in dropped}
        timeline.remove_segments(
            lambda s: s.uid in dropped_uids, before=before_positions, renumber=False,
        )
        if patterns:
            changes.append(
                f"{patterns} cutaway run(s) now carry the narration from underneath"
            )
        log.info(
            "overlaid %d cutaway run(s), %d stray-reappearance fix(es)",
            patterns, mutations - patterns,
        )

    return timeline, changes


# ----------------------------------------------------------------------
# story continuity: close a silent A -> muted cutaway(s) -> A2 interruption
# ----------------------------------------------------------------------
#: Tags used in the change log (see :func:`close_silent_interruptions`) and by
#: any caller wanting to count one kind of rewrite without parsing prose.
FILL_TAG = "fill:"
JCUT_TAG = "J-cut:"


def _is_silent_cutaway(seg: VideoSegment, clip: str) -> bool:
    """True when ``seg`` is a cutaway carrying no audio at all right now.

    The candidate picture for :func:`close_silent_interruptions`: a genuine
    cutaway (see :func:`_is_cutaway`) that is muted with no ``audio_from`` —
    i.e. dead air, not merely a cutaway playing its own ambience (which is not
    the defect this pass exists to fix) or one another pass has already
    handed narration to.
    """
    return _is_cutaway(seg, clip) and seg.mute_source and seg.audio_from is None


def _voice_covers_span(
    timeline: Timeline, positions: Sequence[SegmentPosition], segs: Sequence[VideoSegment]
) -> bool:
    """True when a ``tracks.voice`` pickup overlaps the absolute span of ``segs``.

    ``positions`` should be a snapshot taken before this round's mutations
    (segment identity, not index, is what is looked up) — a pickup already
    covering this stretch of the picture is a voice-pickup cover this pass
    must never touch, not a silent interruption.
    """
    wanted = {id(s) for s in segs}
    spans = [(p.start, p.end) for p in positions if id(p.segment) in wanted]
    if not spans:
        return False
    lo = min(s for s, _e in spans)
    hi = max(e for _s, e in spans)
    for item in timeline.tracks.voice:
        end = item.end if item.end is not None else item.at
        if end <= item.at:
            continue
        if end > lo + _EPS and item.at < hi - _EPS:
            return True
    return False


def _shrink_or_drop_continuation(
    run: Sequence[VideoSegment],
    a2: VideoSegment,
    clips: dict,
) -> None:
    """Shared fallback for both fill and J-cut: too little of ``a2`` survives.

    Mirrors the min-shot shrink in :func:`overlay_cutaways`: the last cutaway
    in ``run`` grows its picture (and, capped to what audio is actually
    available, its ``audio_from``) to cover the rest of what ``a2`` would have
    shown, instead of leaving a sub-``min_shot`` sliver of ``a2`` on the
    timeline. ``a2`` itself is left for the caller to drop.
    """
    last = run[-1]
    assert last.audio_from is not None
    audio_available = max(0.0, a2.out - last.audio_from.in_)
    wanted = last.duration + a2.duration
    clip_duration = float((clips.get(last.clip) or {}).get("duration") or 0.0)
    last_speed = last.speed if last.speed > 0 else 1.0
    new_out = last.in_ + wanted * last_speed
    new_out = min(new_out, last.in_ + audio_available * last_speed)
    if clip_duration > 0:
        new_out = min(new_out, clip_duration)
    new_out = max(new_out, last.out)
    last.out = round(new_out, 3)
    capped_audio_out = min(last.audio_from.in_ + last.duration, a2.out)
    if capped_audio_out > last.audio_from.in_ + _EPS:
        last.audio_from.out = round(capped_audio_out, 3)


def close_silent_interruptions(
    timeline: Timeline, project: Project, settings: Settings | None = None
) -> tuple[Timeline, list[str]]:
    """Never leave the narrator muted between two speech pieces of one take.

    Args:
        timeline: The EDL to rewrite; it is modified in place and returned.
        project: Project supplying transcripts, analyses and clip durations.
        settings: Settings override (defaults to ``project.settings``); reads
            ``pacing.story_fill_max_s``, ``pacing.min_shot_seconds``,
            ``pacing.audio_dupe_tolerance``, ``pacing.speech_pad_after``,
            ``pacing.sentence_gap_max`` and ``pacing.sentence_extend_max``.

    Returns:
        ``(timeline, changes)`` — one line per rewrite, tagged ``fill:`` or
        ``J-cut:`` (see :data:`FILL_TAG`/:data:`JCUT_TAG`); empty when no
        silent-interruption pattern was found.
    """
    cfg = settings or project.settings
    fill_max = max(0.0, float(cfg.get("pacing.story_fill_max_s", 10.0)))
    min_shot = max(0.0, float(cfg.get("pacing.min_shot_seconds", 0.8)))
    tolerance = max(0.0, float(cfg.get("pacing.audio_dupe_tolerance", 0.25)))
    pad_after = max(0.0, float(cfg.get("pacing.speech_pad_after", 0.45)))
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
    video: Sequence[VideoSegment] = timeline.tracks.video
    before_positions = timeline.segment_positions()
    dropped: set[int] = set()
    mutations = 0

    i = 0
    while i < len(video) - 1:
        a = video[i]
        if a.mute_source or not _has_speech(project, a, words_cache):
            i += 1
            continue

        j = i + 1
        while j < len(video) and _is_silent_cutaway(video[j], a.clip):
            j += 1
        if j == i + 1 or j >= len(video):
            i += 1
            continue

        a2 = video[j]
        if a2.clip != a.clip or a2.mute_source or not _has_speech(project, a2, words_cache):
            i = j
            continue
        if a2.in_ <= a.out + _EPS:
            # No gap to close — a.out already reaches a2.in (or past it); not
            # this pass's concern.
            i = j
            continue

        run = list(video[i + 1:j])
        if _voice_covers_span(timeline, before_positions, run):
            log.debug(
                "%s -> %s: cutaway run sits under a voice pickup — left alone", a.id, a2.id
            )
            i = j
            continue

        skip = a2.in_ - a.out
        total_cutaway = sum(seg.duration for seg in run)

        if skip <= fill_max + _EPS:
            # ---- fill: continuous narration under the cutaways -----------
            committed = committed_audio_ranges(timeline, video_slice=slice(0, i)).get(a.clip, [])
            if _consumed_overlap(committed, a.out, a.out + total_cutaway) > tolerance + _EPS:
                log.debug(
                    "%s -> %s: fill range already claimed earlier in the track — left alone",
                    a.id, a2.id,
                )
                i = j
                continue

            old_a2_in = a2.in_
            cursor = a.out
            for cutaway in run:
                end = round(cursor + cutaway.duration, 3)
                cutaway.audio_from = AudioFrom(clip=a.clip, **{"in": round(cursor, 3)}, out=end)
                cutaway.mute_source = False
                cursor = end
            new_in = round(cursor, 3)
            speed2 = a2.speed if a2.speed > 0 else 1.0
            if (a2.out - new_in) / speed2 < min_shot - _EPS:
                _shrink_or_drop_continuation(run, a2, clips)
                dropped.add(id(a2))
                changes.append(
                    f"{a.id} {FILL_TAG} {len(run)} cutaway(s) carry {a.clip} audio "
                    f"{_fmt(a.out)}→{_fmt(run[-1].audio_from.out)} continuously, closing a "
                    f"{skip:.2f}s silent gap; {a2.id} dropped (too little of it left)"
                )
            else:
                a2.in_ = new_in
                changes.append(
                    f"{a.id} {FILL_TAG} {len(run)} cutaway(s) carry {a.clip} audio "
                    f"{_fmt(a.out)}→{_fmt(new_in)} continuously, closing a {skip:.2f}s silent "
                    f"gap; {a2.id} in {_fmt(old_a2_in)}→{_fmt(new_in)}"
                )
            mutations += 1
            i = j + (1 if id(a2) in dropped else 0)
            continue

        # ---- J-cut: finish the sentence, then start the next one --------
        words = words_for(a.clip)
        if words and has_sentence_marks(words):
            tail = _out_tail(a, words, gap_max)
            if tail is not None:
                _low, high = _clip_bounds(video, i)
                cuts = cuts_for(a.clip)
                duration = float((clips.get(a.clip) or {}).get("duration") or 0.0)
                snap = _snap_out_to_sentence(
                    a, words, cuts, high, duration, pad_after, gap_max, extend_max
                )
                if snap is not None:
                    new_out, anchor = snap
                    changes.append(
                        f"{a.id} out {_fmt(a.out)}→{_fmt(new_out)} ({JCUT_TAG} extend to the "
                        f"end of the sentence at '{anchor.text}' before the cutaway)"
                    )
                    a.out = new_out
                    mutations += 1
                else:
                    speed_a = a.speed if a.speed > 0 else 1.0
                    retracted = _retract_out_to_previous_sentence(a, words, pad_after)
                    if (
                        retracted is not None
                        and retracted[0] < a.out - 1e-3
                        and (retracted[0] - a.in_) / speed_a >= min_shot - _EPS
                    ):
                        new_out, anchor = retracted
                        changes.append(
                            f"{a.id} out {_fmt(a.out)}→{_fmt(new_out)} ({JCUT_TAG} retract to "
                            f"the sentence already finished at '{anchor.text}' — the next one "
                            "is unreachable before the cutaway)"
                        )
                        a.out = new_out
                        mutations += 1
                    # else: extension and retraction are both blocked (a guard,
                    # a neighbouring cut, min_shot); leave a.out mid-sentence —
                    # ytedit qc rule 37 reports it, the same way rule 32
                    # reports a pad that hit the same wall.

        committed = committed_audio_ranges(timeline, video_slice=slice(0, j)).get(a.clip, [])
        if _consumed_overlap(committed, a2.in_, a2.in_ + total_cutaway) > tolerance + _EPS:
            log.debug(
                "%s -> %s: J-cut range already claimed earlier in the track — left alone",
                a.id, a2.id,
            )
            i = j
            continue

        old_a2_in = a2.in_
        cursor = old_a2_in
        for cutaway in run:
            end = round(cursor + cutaway.duration, 3)
            cutaway.audio_from = AudioFrom(clip=a.clip, **{"in": round(cursor, 3)}, out=end)
            cutaway.mute_source = False
            cursor = end
        new_in = round(cursor, 3)
        speed2 = a2.speed if a2.speed > 0 else 1.0
        if (a2.out - new_in) / speed2 < min_shot - _EPS:
            _shrink_or_drop_continuation(run, a2, clips)
            dropped.add(id(a2))
            changes.append(
                f"{a.id} {JCUT_TAG} {len(run)} cutaway(s) carry {a.clip} audio from {a2.id}'s "
                f"own start ({_fmt(old_a2_in)}→{_fmt(run[-1].audio_from.out)}); {a2.id} dropped "
                "(too little of it left) — the thought finishes, then the next starts clean"
            )
        else:
            a2.in_ = new_in
            changes.append(
                f"{a.id} {JCUT_TAG} {len(run)} cutaway(s) carry {a2.id}'s own audio "
                f"{_fmt(old_a2_in)}→{_fmt(new_in)} early; {a2.id} in {_fmt(old_a2_in)}→"
                f"{_fmt(new_in)} — the thought finishes, then the next starts clean under "
                "the cutaway"
            )
        mutations += 1
        i = j + (1 if id(a2) in dropped else 0)

    if mutations:
        # Same edit-API pattern as ``overlay_cutaways`` — see its own comment
        # on this: run unconditionally (even when nothing was dropped) so
        # every mutation gets a proper before/after re-time and every anchor
        # is re-resolved.
        dropped_uids = {s.uid for s in video if id(s) in dropped}
        timeline.remove_segments(
            lambda s: s.uid in dropped_uids, before=before_positions, renumber=False,
        )
        log.info("closed %d silent-interruption pattern(s)", mutations)

    return timeline, changes


def find_silent_interruptions(
    timeline: Timeline, project: Project, settings: Settings | None = None
) -> list[str]:
    """Report every silent ``A -> muted cutaway(s) -> A2`` pattern, unfixed.

    The read-only counterpart to :func:`close_silent_interruptions` — the
    exact defect it exists to fix (``ytedit qc`` rule 36). A finding here
    after a fresh ``ytedit tidy`` means the pattern could not be closed (a
    voice pickup covers it, or the ledger already claims the range) rather
    than that the pass was never run.

    Args:
        timeline: The EDL to scan.
        project: Project supplying transcripts.
        settings: Unused; accepted for symmetry with the other rule scanners.

    Returns:
        One message per pattern found; empty when the timeline is clean.
    """
    words_cache: dict[str, list[Word]] = {}
    video = timeline.tracks.video
    issues: list[str] = []

    i = 0
    while i < len(video) - 1:
        a = video[i]
        if a.mute_source or not _has_speech(project, a, words_cache):
            i += 1
            continue
        j = i + 1
        while j < len(video) and _is_silent_cutaway(video[j], a.clip):
            j += 1
        if j == i + 1 or j >= len(video):
            i += 1
            continue
        a2 = video[j]
        if a2.clip != a.clip or a2.mute_source or not _has_speech(project, a2, words_cache):
            i = j
            continue
        run_ids = ", ".join(s.id for s in video[i + 1:j])
        issues.append(
            f"silent interruption of a take: {a.id} ({a.clip}) is followed by "
            f"{j - i - 1} muted cutaway(s) [{run_ids}] with no audio, then {a2.id} resumes "
            f"{a.clip} — the narrator goes silent for the length of the cutaway"
        )
        i = j

    return issues


__all__ = [
    "CUTAWAY_ROLE_RE",
    "FILL_TAG",
    "JCUT_TAG",
    "close_silent_interruptions",
    "find_silent_interruptions",
    "overlay_cutaways",
]
