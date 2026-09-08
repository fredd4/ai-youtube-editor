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
"""

from __future__ import annotations

import re
from typing import Sequence

from ytedit.ai.ledger import _consumed_overlap, committed_audio_ranges
from ytedit.ai.tidy import Word, load_words
from ytedit.config import Settings
from ytedit.log import get_logger
from ytedit.project import Project
from ytedit.timeline import AudioFrom, Timeline, VideoSegment

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


__all__ = ["CUTAWAY_ROLE_RE", "overlay_cutaways"]
