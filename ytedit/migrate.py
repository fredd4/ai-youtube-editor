"""One-shot migration of a v1 ``plan/timeline.json`` into a v2 ``plan/cut.json``.

See ``docs/ARCHITECTURE.md`` §"Migration (ytedit migrate)". v1 stored the edit
as seconds; v2 stores speech as sentence ids and lets the resolver produce the
seconds. This module reads a v1 timeline plus the sentence catalogue and
reconstructs the editorial intent behind it:

* a run of consecutive segments reading the *same clip's* audio end to end —
  the narrator's own picture plus any cutaways borrowing that audio via
  ``audio_from`` — becomes one ``speech`` beat whose ``sentences`` are the
  catalogue sentences that range covered, and whose cutaways become ``shots``;
* everything muted or wordless becomes a ``broll`` beat;
* a ``voice/vo_<clip>_<in>_<out>.wav`` pickup (v1 cut the narrator's own audio
  out to a WAV) becomes an off-camera ``speech`` beat of that clip, and any
  other pickup becomes a ``voice`` beat; the picture segments underneath the
  pickup become its shots;
* captions, music, chapters and markers move from absolute seconds to beat
  references.

Nothing here is lossless by construction — v1 boundaries drifted, cutaway
hand-offs overlapped by a few frames, and a v1 cut could end mid-sentence.
Every such decision is recorded in :attr:`MigrationReport.issues` so the user
can review it in the editor rather than discover it in the render.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ytedit import cut as cut_module
from ytedit.ai.sentences import (
    build_sentence_catalogue,
    load_sentences,
    sentences_by_clip,
    sentences_path,
    write_sentences,
)
from ytedit.words import Word, load_words
from ytedit.cut import Beat, Caption, Chapter, Cut, Marker, Meta, MusicCue, Shot
from ytedit.log import get_logger
from ytedit.project import Project
from ytedit.timeline import Timeline, Transform, Transition, VideoSegment

log = get_logger(__name__)

STAGE = "migrate"

_EPS = 1e-6

#: ``voice/vo_<clip>_<in>_<out>.wav`` — narration v1 cut straight out of a
#: source clip's own audio track. In v2 that is not a pickup at all but an
#: off-camera ``speech`` beat, so the range in the filename is what the beat's
#: sentences are selected from.
VO_FILE_RE = re.compile(
    r"(?:^|/)vo_(?P<clip>[A-Za-z0-9]+)_(?P<in>\d+\.\d+)_(?P<out>\d+\.\d+)\.wav$"
)

#: Two audio ranges of the same clip count as abutting (one continuous take)
#: when the second starts no later than one frame after the first ends and
#: overlaps it by less than this. v1's cutaway hand-offs habitually replayed
#: 0.05–0.15 s (``s124`` audio ``c141 7.00–11.00`` handing over to ``s125``
#: ``c141 10.91–12.00``); that is drift, not a deliberate second reading.
MAX_HANDOFF_OVERLAP: float = 0.25

#: A sentence must be covered by this fraction of its own span to survive into
#: the beat; below it the sentence is dropped and reported ``edge_trimmed``.
SENTENCE_KEEP_RATIO: float = 0.5

#: How far a shot's borrowed audio start may sit from the sentence end it is
#: attached to before the placement is reported as approximate.
SHOT_AFTER_TOLERANCE: float = 0.3

#: Absolute-time slop when deciding whether a voice item covers a segment or
#: needs it split (seconds — well under one frame at 30 fps is not a split).
_COVER_EPS: float = 0.05

#: A resolved beat whose length differs from the v1 segments it came from by
#: more than this is listed in the report's duration section.
DURATION_DRIFT: float = 0.5

#: A beat a v1 music cue covers for less than this is not part of the cue's
#: beat range: v1 boundaries habitually landed a few tenths into the next beat.
MIN_CUE_OVERLAP: float = 0.5

#: A leftover shorter than this, produced by splitting a segment at a pickup
#: boundary, is reported so the user can delete it instead of shipping a sliver.
MIN_REMAINDER: float = 0.5

#: v1 roles that mean "picture, not the narrator talking to camera".
_PICTURE_ROLES: frozenset[str] = frozenset({"b-roll", "cold-open", "outro", "cutaway"})


class MigrationError(RuntimeError):
    """Raised when there is no v1 timeline to migrate."""


# ----------------------------------------------------------------------
# working units — one v1 video segment, with its absolute placement
# ----------------------------------------------------------------------
@dataclass
class _Unit:
    """One v1 video segment (or a piece of one, after a split), placed."""

    seg_id: str
    clip: str
    pic_in: float
    pic_out: float
    start: float
    end: float
    muted: bool
    a_clip: str
    a_in: float
    a_out: float
    role: str
    transform: Transform
    grade: str
    transition_in: Transition
    gain_db: float
    notes: str
    #: ``audio_from`` was set: the picture is borrowed, the audio is not.
    borrowed: bool
    #: Index of the voice/vo beat that consumed this unit, if any.
    owner: int | None = None

    @property
    def span(self) -> float:
        return max(0.0, self.end - self.start)

    def split(self, at: float, suffix_a: str, suffix_b: str) -> tuple["_Unit", "_Unit"]:
        """Cut this unit in two at absolute time ``at`` (picture and audio)."""
        span = self.end - self.start
        frac = 0.0 if span <= _EPS else (at - self.start) / span
        frac = min(1.0, max(0.0, frac))
        pic_cut = self.pic_in + (self.pic_out - self.pic_in) * frac
        a_cut = self.a_in + (self.a_out - self.a_in) * frac
        first = _replace_unit(
            self, seg_id=f"{self.seg_id}{suffix_a}", pic_out=pic_cut, end=at, a_out=a_cut
        )
        second = _replace_unit(
            self,
            seg_id=f"{self.seg_id}{suffix_b}",
            pic_in=pic_cut,
            start=at,
            a_in=a_cut,
            transition_in=Transition(),
        )
        return first, second


def _replace_unit(unit: _Unit, **changes: Any) -> _Unit:
    """``dataclasses.replace`` without deep-copying the pydantic sub-models."""
    data = {
        "seg_id": unit.seg_id, "clip": unit.clip, "pic_in": unit.pic_in,
        "pic_out": unit.pic_out, "start": unit.start, "end": unit.end,
        "muted": unit.muted, "a_clip": unit.a_clip, "a_in": unit.a_in,
        "a_out": unit.a_out, "role": unit.role, "transform": unit.transform,
        "grade": unit.grade, "transition_in": unit.transition_in,
        "gain_db": unit.gain_db, "notes": unit.notes, "borrowed": unit.borrowed,
        "owner": unit.owner,
    }
    data.update(changes)
    return _Unit(**data)


def _units_from(timeline: Timeline) -> list[_Unit]:
    """Every v1 video segment as a placed :class:`_Unit`, in order."""
    units: list[_Unit] = []
    for pos in timeline.segment_positions():
        seg: VideoSegment = pos.segment
        a_clip, a_in, a_out = seg.audio_source
        units.append(
            _Unit(
                seg_id=seg.id,
                clip=seg.clip,
                pic_in=seg.in_,
                pic_out=seg.out,
                start=pos.start,
                end=pos.end,
                muted=bool(seg.mute_source),
                a_clip=a_clip,
                a_in=a_in,
                a_out=a_out,
                role=seg.role,
                transform=seg.transform,
                grade=seg.grade,
                transition_in=seg.transition_in,
                gain_db=seg.source_audio_gain_db,
                notes=seg.notes,
                borrowed=seg.audio_from is not None,
            )
        )
    return units


# ----------------------------------------------------------------------
# sentence selection
# ----------------------------------------------------------------------
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _select_sentences(
    clip_sentences: Sequence[Mapping[str, Any]], a_in: float, a_out: float
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], float]]]:
    """Split a clip's sentences into kept and edge-trimmed for ``[a_in, a_out]``.

    Returns:
        ``(kept, trimmed)`` — ``kept`` are the sentences covered by at least
        :data:`SENTENCE_KEEP_RATIO` of their own span, ``trimmed`` pairs each
        partially covered sentence with the fraction that was covered.
    """
    kept: list[dict[str, Any]] = []
    trimmed: list[tuple[dict[str, Any], float]] = []
    for sent in clip_sentences:
        s = float(sent.get("s") or 0.0)
        e = float(sent.get("e") or 0.0)
        own = e - s
        if own <= _EPS:
            continue
        covered = _overlap(s, e, a_in, a_out)
        if covered <= _EPS:
            continue
        ratio = covered / own
        if ratio >= SENTENCE_KEEP_RATIO - _EPS:
            kept.append(dict(sent))
        else:
            trimmed.append((dict(sent), ratio))
    return kept, trimmed


def _words_in(words: Sequence[Word], s: float, e: float) -> str:
    """The transcript text between ``s`` and ``e`` (for the report)."""
    picked = [w.text for w in words if _overlap(w.s, w.e, s, e) > 0.01]
    return " ".join(t for t in picked if t).strip()


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
@dataclass
class MigrationReport:
    """Everything one ``ytedit migrate`` run produced and decided."""

    cut: Cut
    #: Decisions and ambiguities that need the editor's eye, in beat order.
    issues: list[str]
    #: Informational log lines (what was read, what was written).
    notes: list[str]
    #: ``str(Issue)`` lines from :func:`ytedit.cut.validate`, ``[]`` when the
    #: validator is not implemented yet.
    validation: list[str]
    duration_v1: float
    #: Resolved v2 length, ``None`` when :func:`ytedit.cut.resolve` is missing.
    duration_v2: float | None
    #: One line per beat: ``b012 speech c048 #3–#5, 1 shot, 11.2 s ← s022 s023``.
    beat_lines: list[str] = field(default_factory=list)
    #: Beats whose resolved length drifted from their v1 source segments.
    duration_drift: list[str] = field(default_factory=list)
    #: Files this run wrote (empty for a dry run without ``out_dir``).
    written: list[Path] = field(default_factory=list)

    def markdown(self) -> str:
        """Render ``plan/migrate_report.md``."""
        kinds: dict[str, int] = {}
        shots = 0
        for beat in self.cut.beats:
            kinds[beat.kind] = kinds.get(beat.kind, 0) + 1
            shots += len(beat.shots)
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) or "none"

        lines: list[str] = [
            "# Migration v1 timeline → cut.json",
            "",
            f"Beats: {len(self.cut.beats)} ({summary}) · shots: {shots} · "
            f"captions: {len(self.cut.captions)} · music: {len(self.cut.music)} · "
            f"chapters: {len(self.cut.chapters)} · markers: {len(self.cut.markers)}",
            "",
            "## Beats",
            "",
        ]
        lines += self.beat_lines or ["_no beats_"]
        lines += ["", "## Issues", ""]
        if self.issues:
            lines += [f"- {issue}" for issue in self.issues]
        else:
            lines.append("_none_")
        lines += ["", "## Validation", ""]
        if self.validation:
            lines += [f"- {line}" for line in self.validation]
        else:
            lines.append("_clean_")
        lines += ["", "## Duration", ""]
        if self.duration_v2 is None:
            lines.append(
                f"- v1: {self.duration_v1:.2f} s · v2: not resolved — see Notes"
            )
        else:
            delta = self.duration_v2 - self.duration_v1
            lines.append(
                f"- v1: {self.duration_v1:.2f} s · v2: {self.duration_v2:.2f} s "
                f"({delta:+.2f} s)"
            )
        if self.duration_drift:
            lines.append("")
            lines.append(f"Beats that changed by more than {DURATION_DRIFT:.1f} s:")
            lines.append("")
            lines += [f"- {line}" for line in self.duration_drift]
        if self.notes:
            lines += ["", "## Notes", ""]
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# the migration itself
# ----------------------------------------------------------------------
class _Migrator:
    """Builds one :class:`Cut` from one v1 :class:`Timeline`."""

    def __init__(self, project: Project, timeline: Timeline, catalogue: Mapping[str, Any]):
        self.project = project
        self.timeline = timeline
        self.by_clip = sentences_by_clip(catalogue)
        self.fps = float(timeline.fps or 30)
        self.frame = 1.0 / self.fps if self.fps > 0 else 0.033
        self.issues: list[str] = []
        self.notes: list[str] = []
        #: ``uid -> (start, end)`` in v1 absolute time.
        self.spans: dict[str, tuple[float, float]] = {}
        #: v1 segment ids each emitted beat was built from, beat order.
        self.sources: list[list[str]] = []
        self._words: dict[str, list[Word]] = {}

    # -- helpers -------------------------------------------------------
    def words(self, clip: str) -> list[Word]:
        if clip not in self._words:
            self._words[clip] = load_words(self.project, clip)
        return self._words[clip]

    def issue(self, code: str, where: str, message: str) -> None:
        self.issues.append(f"{code} [{where}]: {message}")

    def sentences_of(self, clip: str) -> list[Mapping[str, Any]]:
        return list(self.by_clip.get(clip) or [])

    # -- voice items ---------------------------------------------------
    def claim_voice(self, units: list[_Unit]) -> list[dict[str, Any]]:
        """Attach every v1 voice item to the units underneath it.

        Splits a unit that the item only partially covers so the item's
        picture is exactly the run of units it owns, and marks those units
        consumed. Returns one record per voice item, in timeline order.
        """
        records: list[dict[str, Any]] = []
        items = sorted(self.timeline.tracks.voice, key=lambda v: v.at)
        for owner, item in enumerate(items):
            at = float(item.at)
            end = item.end
            if end is None:
                end = at
                self.issue(
                    "voice_no_end", item.id,
                    f"{item.file} has no end in the v1 timeline — picture taken from the "
                    "single segment at its start",
                )
            end = float(end)

            window_end = max(end, at + _COVER_EPS)
            covered: list[int] = []
            for i, u in enumerate(units):
                if _overlap(u.start, u.end, at, window_end) <= _COVER_EPS:
                    continue
                if u.owner is not None:
                    if covered:
                        self.issue(
                            "voice_overlap", item.id,
                            f"{u.seg_id} is already used by an earlier pickup — "
                            f"{item.file} keeps only the picture before it",
                        )
                        break
                    continue
                if covered and i != covered[-1] + 1:
                    break
                covered.append(i)
            if not covered:
                # Nothing free underneath: either past the end of the video
                # track or every unit already belongs to an earlier pickup.
                self.issue(
                    "voice_unplaced", item.id,
                    f"{item.file} at {at:.2f}–{end:.2f}s covers no free picture segment",
                )
                records.append({"item": item, "units": [], "owner": owner})
                continue

            first, last = covered[0], covered[-1]
            # Trim the edges: the part of a segment before the pickup starts
            # (or after it ends) is still ordinary picture and stays broll.
            if at - units[first].start > _COVER_EPS:
                head, tail = units[first].split(at, "a", "b")
                units[first:first + 1] = [head, tail]
                self._report_sliver(item, head)
                first += 1
                last += 1
            if units[last].end - end > _COVER_EPS:
                head, tail = units[last].split(end, "a", "b")
                units[last:last + 1] = [head, tail]
                self._report_sliver(item, tail)

            owned = units[first:last + 1]
            for unit in owned:
                unit.owner = owner
                if unit.muted:
                    continue
                kept, _ = _select_sentences(
                    self.sentences_of(unit.a_clip), unit.a_in, unit.a_out
                )
                if kept:
                    self.issue(
                        "voice_over_speech", item.id,
                        f"{unit.seg_id} spoke {_sentence_range(kept)} of {unit.a_clip} "
                        f"under {item.file} in v1; the v2 shot is silent",
                    )
                else:
                    self.issue(
                        "voice_ambient_lost", item.id,
                        f"{unit.seg_id} ({unit.clip}) played its own sound under "
                        f"{item.file} in v1; the v2 shot is silent",
                    )
            records.append({"item": item, "units": owned, "owner": owner})
        return records

    def _report_sliver(self, item: Any, leftover: _Unit) -> None:
        """Flag a split leftover too short to be worth keeping as its own beat."""
        if leftover.span >= MIN_REMAINDER:
            return
        self.issue(
            "short_remainder", item.id,
            f"splitting {leftover.seg_id[:-1]} at the edge of {item.file} leaves "
            f"{leftover.span:.2f}s of {leftover.clip} as its own b-roll beat — "
            "delete it or fold it into a neighbour",
        )

    # -- beat builders -------------------------------------------------
    def _shot(self, unit: _Unit, after: str | None = None) -> Shot:
        return Shot(
            clip=unit.clip,
            **{"in": round(unit.pic_in, 3)},
            out=round(unit.pic_out, 3),
            after=after,
            transform=unit.transform,
            grade=unit.grade,
            notes=unit.notes,
        )

    def _register(self, beat: Beat, units: Sequence[_Unit]) -> Beat:
        self.spans[beat.uid] = (units[0].start, units[-1].end) if units else (0.0, 0.0)
        return beat

    def broll_beat(self, unit: _Unit) -> Beat:
        if unit.borrowed and unit.role == "cutaway":
            self.issue(
                "cutaway_orphan", unit.seg_id,
                f"cutaway over {unit.a_clip} {unit.a_in:.2f}–{unit.a_out:.2f}s has no "
                "speech beat to attach to — kept as b-roll with its own picture",
            )
        beat = Beat(
            kind="broll",
            clip=unit.clip,
            **{"in": round(unit.pic_in, 3)},
            out=round(unit.pic_out, 3),
            audio="mute" if unit.muted else "ambient",
            role=unit.role if unit.role != "cutaway" else "b-roll",
            transform=unit.transform,
            grade=unit.grade,
            transition_in=unit.transition_in,
            gain_db=unit.gain_db,
            notes=unit.notes,
        )
        return self._register(beat, [unit])

    def speech_beat(self, run: Sequence[_Unit], kept: Sequence[Mapping[str, Any]]) -> Beat:
        """One on/off-camera speech beat from a run of same-clip audio units."""
        clip = run[0].a_clip
        base = [u for u in run if not u.borrowed]
        anchor = base[0] if base else run[0]
        ids = [str(s["id"]) for s in kept]

        shots: list[Shot] = []
        chained: str | None = None
        for index, unit in enumerate(run):
            if not unit.borrowed:
                chained = None
                continue
            if index > 0 and run[index - 1].borrowed:
                # A run of back-to-back cutaways over one continuous take: v2
                # chains shots sharing the same ``after`` in list order, so
                # only the first of the chain names a sentence.
                shots.append(self._shot(unit, chained))
                continue
            after, delta = self._after_for(kept, unit.a_in)
            if after is None and base and index == 0:
                self.issue(
                    "shot_leads_beat", unit.seg_id,
                    f"cutaway opened the take in v1; in v2 the beat's own picture "
                    f"({clip}) plays first and the shot follows",
                )
            elif after is not None and delta > SHOT_AFTER_TOLERANCE:
                self.issue(
                    "shot_after_approx", unit.seg_id,
                    f"cutaway started at {unit.a_in:.2f}s, nearest sentence end "
                    f"{after} is {delta:.2f}s away",
                )
            chained = after
            shots.append(self._shot(unit, after))

        beat = Beat(
            kind="speech",
            clip=clip,
            sentences=ids,
            on_camera=bool(base),
            shots=shots,
            role=anchor.role if anchor.role != "cutaway" else "a-roll",
            transform=anchor.transform,
            grade=anchor.grade,
            transition_in=run[0].transition_in,
            gain_db=anchor.gain_db,
            notes=anchor.notes,
        )
        return self._register(beat, run)

    def _after_for(
        self, kept: Sequence[Mapping[str, Any]], a_in: float
    ) -> tuple[str | None, float]:
        """The sentence a shot starting at ``a_in`` hangs off, and the error."""
        if not kept:
            return None, 0.0
        if a_in <= float(kept[0].get("s") or 0.0) + _EPS:
            return None, 0.0
        best = min(kept, key=lambda s: abs(float(s.get("e") or 0.0) - a_in))
        return str(best["id"]), abs(float(best.get("e") or 0.0) - a_in)

    def voice_beat(self, record: Mapping[str, Any]) -> Beat:
        """A v1 voice item as either an off-camera speech beat or a voice beat."""
        item = record["item"]
        units: list[_Unit] = list(record["units"])
        match = VO_FILE_RE.search(str(item.file))
        if match:
            clip = match.group("clip")
            a_in = float(match.group("in"))
            a_out = float(match.group("out"))
            kept, trimmed = _select_sentences(self.sentences_of(clip), a_in, a_out)
            self._report_trimmed(item.id, clip, kept, trimmed, a_in, a_out)
            if not kept:
                self.issue(
                    "vo_no_sentences", item.id,
                    f"{item.file} covers no sentence of {clip} by "
                    f"{int(SENTENCE_KEEP_RATIO * 100)}% — left as a voice beat on the WAV",
                )
            else:
                self._flagged(item.id, kept)
                self._report_extension(item.id, kept, a_in, a_out)
                beat = Beat(
                    kind="speech",
                    clip=clip,
                    sentences=[str(s["id"]) for s in kept],
                    on_camera=False,
                    shots=[self._shot(u) for u in units],
                    role="a-roll",
                    transform=units[0].transform if units else Transform(),
                    grade=units[0].grade if units else "default",
                    transition_in=units[0].transition_in if units else Transition(),
                    gain_db=float(item.gain_db),
                    notes=f"v1 voice-over extract {item.file}",
                )
                self.issue(
                    "vo_extract", item.id,
                    f"{item.file} → off-camera speech beat {clip} "
                    f"{_sentence_range(kept)} over {len(units)} shot(s); "
                    "the extracted WAV is no longer used",
                )
                return self._register(beat, units)

        if not units:
            self.issue(
                "voice_no_picture", item.id,
                f"{item.file} has no picture — add shots in the editor "
                "(v2 refuses to render a voice beat without picture)",
            )
        beat = Beat(
            kind="voice",
            file=str(item.file),
            shots=[self._shot(u) for u in units],
            role=(units[0].role if units and units[0].role != "cutaway" else "b-roll"),
            transform=units[0].transform if units else Transform(),
            grade=units[0].grade if units else "default",
            transition_in=units[0].transition_in if units else Transition(),
            gain_db=float(item.gain_db),
            notes=f"v1 pickup {item.id}",
        )
        self.issue(
            "voice_pickup", item.id,
            f"{item.file} → voice beat over {len(units)} shot(s)",
        )
        return self._register(beat, units)

    def _report_trimmed(
        self,
        where: str,
        clip: str,
        kept: Sequence[Mapping[str, Any]],
        trimmed: Sequence[tuple[Mapping[str, Any], float]],
        a_in: float,
        a_out: float,
    ) -> None:
        for sent, ratio in trimmed:
            s = float(sent.get("s") or 0.0)
            e = float(sent.get("e") or 0.0)
            lost = _words_in(self.words(clip), max(s, a_in), min(e, a_out))
            message = (
                f"{sent['id']} was only {ratio * 100:.0f}% inside the v1 cut and is dropped"
            )
            if lost:
                message += f" — words lost: \"{lost}\""
            self.issue("edge_trimmed", where, message)

    def _report_extension(
        self, where: str, kept: Sequence[Mapping[str, Any]], a_in: float, a_out: float
    ) -> None:
        """Flag sentences that reach outside what v1 actually played.

        A sentence kept at 50–99 % coverage brings its missing head or tail
        back with it: v2 plays the whole sentence, so the beat gets longer and
        the viewer hears words the v1 draft never contained.
        """
        if not kept:
            return
        head = max(0.0, a_in - min(float(s.get("s") or 0.0) for s in kept))
        tail = max(0.0, max(float(s.get("e") or 0.0) for s in kept) - a_out)
        if head + tail <= DURATION_DRIFT:
            return
        parts = []
        if head > _EPS:
            parts.append(f"{head:.2f}s before the v1 in-point")
        if tail > _EPS:
            parts.append(f"{tail:.2f}s after the v1 out-point")
        self.issue(
            "sentence_extends", where,
            f"the kept sentences reach {' and '.join(parts)} — v2 plays speech the v1 "
            "draft cut off",
        )

    def _flagged(self, where: str, kept: Sequence[Mapping[str, Any]]) -> None:
        for sent in kept:
            reasons = []
            if sent.get("instruction"):
                reasons.append("editor instruction")
            if sent.get("retake_of"):
                reasons.append(f"retake of {sent['retake_of']}")
            if sent.get("duplicate_of"):
                reasons.append(f"duplicate of {sent['duplicate_of']}")
            if reasons:
                self.issue(
                    "flagged_sentence", where,
                    f"{sent['id']} is flagged ({', '.join(reasons)}) but was in the v1 "
                    "cut — v2 validation will refuse it until you drop or replace it",
                )

    # -- the walk ------------------------------------------------------
    def _abuts(self, prev: _Unit, nxt: _Unit) -> bool:
        if prev.a_clip != nxt.a_clip:
            return False
        delta = nxt.a_in - prev.a_out
        return -MAX_HANDOFF_OVERLAP < delta <= self.frame + _EPS

    def build_beats(
        self, units: list[_Unit], voice_records: Sequence[Mapping[str, Any]]
    ) -> list[Beat]:
        """Walk the placed units once and emit the beats they become."""
        beats: list[Beat] = []
        emitted_owners: set[int] = set()
        by_owner = {int(r["owner"]): r for r in voice_records}
        sources: list[list[str]] = []

        i = 0
        while i < len(units):
            unit = units[i]
            if unit.owner is not None:
                owner = int(unit.owner)
                if owner not in emitted_owners:
                    emitted_owners.add(owner)
                    beats.append(self.voice_beat(by_owner[owner]))
                    sources.append([u.seg_id for u in by_owner[owner]["units"]])
                i += 1
                continue

            run = [unit]
            if not unit.muted:
                j = i + 1
                while (
                    j < len(units)
                    and units[j].owner is None
                    and not units[j].muted
                    and self._abuts(run[-1], units[j])
                ):
                    run.append(units[j])
                    j += 1

            kept: list[dict[str, Any]] = []
            if not unit.muted:
                a_in = min(u.a_in for u in run)
                a_out = max(u.a_out for u in run)
                kept, trimmed = _select_sentences(self.sentences_of(unit.a_clip), a_in, a_out)
                if kept:
                    self._report_trimmed(
                        run[0].seg_id, unit.a_clip, kept, trimmed, a_in, a_out
                    )
                    self._flagged(run[0].seg_id, kept)

            if kept:
                if all(u.role in _PICTURE_ROLES for u in run):
                    self.issue(
                        "broll_became_speech", run[0].seg_id,
                        f"v1 role {run[0].role!r}, but its own sound carries "
                        f"{unit.a_clip} {_sentence_range(kept)} — now a speech beat "
                        "(mute it if that speech is background)",
                    )
                self._report_extension(run[0].seg_id, kept, a_in, a_out)
                beats.append(self.speech_beat(run, kept))
                sources.append([u.seg_id for u in run])
                i += len(run)
                continue

            beats.append(self.broll_beat(unit))
            sources.append([unit.seg_id])
            i += 1

        # any voice item that owns nothing still deserves a beat
        for owner in sorted(by_owner):
            if owner in emitted_owners:
                continue
            beats.append(self.voice_beat(by_owner[owner]))
            sources.append([])

        self.sources = sources
        return beats

    # -- absolute tracks ------------------------------------------------
    def beat_at(self, beats: Sequence[Beat], t: float) -> Beat | None:
        """The beat covering v1 absolute time ``t`` (last beat when past the end)."""
        placed = [(b, *self.spans.get(b.uid, (0.0, 0.0))) for b in beats]
        placed = [p for p in placed if p[2] > p[1]]
        if not placed:
            return None
        for beat, start, end in placed:
            if start - _EPS <= t < end - _EPS:
                return beat
        if t < placed[0][1]:
            return placed[0][0]
        return placed[-1][0]

    def beat_range(
        self, beats: Sequence[Beat], at: float, end: float, where: str
    ) -> tuple[Beat | None, Beat | None]:
        """The inclusive beat range a v1 absolute span ``[at, end)`` becomes.

        v1 music cues ended on a chapter time, which regularly falls a few
        tenths *inside* the beat that opens the next chapter. Taking the beat
        under ``end`` literally would hand that whole beat to the outgoing
        cue, so a beat the cue barely touches is dropped from the range in
        favour of its neighbour (reported ``music_range_approx``).
        """
        touched = [
            (beat, _overlap(*self.spans.get(beat.uid, (0.0, 0.0)), at, end))
            for beat in beats
            if _overlap(*self.spans.get(beat.uid, (0.0, 0.0)), at, end) > _EPS
        ]
        if not touched:
            return self.beat_at(beats, at), self.beat_at(beats, max(at, end - 0.001))
        solid = [b for b, covered in touched if covered >= MIN_CUE_OVERLAP]
        first, last = (solid[0], solid[-1]) if solid else (touched[0][0], touched[-1][0])
        if first is not touched[0][0] or last is not touched[-1][0]:
            mig_first, mig_last = touched[0][0], touched[-1][0]
            mig_start = self.spans.get(mig_first.uid, (0.0, 0.0))[0]
            mig_end = self.spans.get(mig_last.uid, (0.0, 0.0))[1]
            self.issue(
                "music_range_approx", where,
                f"v1 span {at:.2f}–{end:.2f}s only clipped the edge beats "
                f"({mig_start:.2f}–{mig_end:.2f}s); the cue now runs "
                f"{first.id or first.uid}–{last.id or last.uid}",
            )
        return first, last

    def line_for(self, beat: Beat, source_ids: Sequence[str]) -> str:
        start, end = self.spans.get(beat.uid, (0.0, 0.0))
        length = end - start
        shots = len(beat.shots)
        shot_txt = "" if not shots else f", {shots} shot{'s' if shots != 1 else ''}"
        if beat.kind == "speech":
            what = f"speech {beat.clip} {_sentence_range(beat.sentences)}"
            if not beat.on_camera:
                what += " (off-camera)"
        elif beat.kind == "voice":
            what = f"voice {beat.file}"
        else:
            what = (
                f"broll {beat.clip} {beat.in_:.2f}–{beat.out:.2f} "
                f"({'muted' if beat.audio == 'mute' else 'ambient'})"
            )
        src = " ".join(source_ids) or "—"
        return f"`{beat.id}` {what}{shot_txt}, {length:.1f} s ← {src}"


def _sentence_range(sentences: Sequence[Any]) -> str:
    """``#3–#5`` for a list of sentence dicts or ids."""
    numbers: list[int] = []
    for sent in sentences:
        sid = str(sent["id"]) if isinstance(sent, Mapping) else str(sent)
        _, _, tail = sid.partition("#")
        try:
            numbers.append(int(tail))
        except ValueError:
            continue
    if not numbers:
        return "(no sentences)"
    if len(numbers) == 1:
        return f"#{numbers[0]}"
    return f"#{min(numbers)}–#{max(numbers)}"


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------
def migrate_project(
    project: Project, dry_run: bool = False, out_dir: Path | None = None
) -> MigrationReport:
    """Turn a project's v1 ``plan/timeline.json`` into a v2 ``plan/cut.json``.

    Args:
        project: The project to migrate.
        dry_run: Never touch anything inside the project directory. The cut
            and the report are still built (and written to ``out_dir`` when
            one is given), so the migration can be reviewed before it runs.
        out_dir: Where a dry run writes ``cut.json`` / ``migrate_report.md``.
            Ignored for a real run, which always writes into ``plan/``.

    Returns:
        The :class:`MigrationReport`, whose :attr:`MigrationReport.cut` is the
        v2 cut whether or not it was written.

    Raises:
        MigrationError: When there is no v1 timeline to read.
    """
    if not project.timeline_file.exists():
        raise MigrationError(f"no v1 timeline at {project.timeline_file}")

    # ``load_legacy``, not ``load``: reading a v1 timeline is the whole point
    # of this module, and ``Timeline.load`` refuses one on purpose.
    timeline = Timeline.load_legacy(project.timeline_file)
    notes: list[str] = [
        f"read {project.rel(project.timeline_file)}: "
        f"{len(timeline.tracks.video)} segments, {len(timeline.tracks.voice)} voice items, "
        f"{len(timeline.tracks.music)} music cues, {len(timeline.tracks.captions)} captions"
    ]

    # sentence catalogue — built in memory for a dry run, written for a real one
    overlay_needed = False
    if sentences_path(project).exists():
        catalogue = load_sentences(project)
        notes.append(f"sentence catalogue: {project.rel(sentences_path(project))}")
    elif dry_run:
        catalogue = build_sentence_catalogue(project)
        overlay_needed = True
        notes.append("sentence catalogue built in memory (dry run writes nothing)")
    else:
        catalogue = write_sentences(project)
        notes.append(f"sentence catalogue written to {project.rel(sentences_path(project))}")

    mig = _Migrator(project, timeline, catalogue)
    units = _units_from(timeline)
    voice_records = mig.claim_voice(units)
    beats = mig.build_beats(units, voice_records)

    cut = Cut(
        version=cut_module.CUT_VERSION,
        fps=timeline.fps,
        width=timeline.width,
        height=timeline.height,
        language=timeline.language,
        beats=beats,
        mute_ranges=list(timeline.mute_ranges),
        meta=Meta(
            title_candidates=list(timeline.meta.title_candidates),
            generated_by=(
                f"{timeline.meta.generated_by} + migrate@{_today()}"
                if timeline.meta.generated_by
                else f"migrate@{_today()}"
            ),
            edited_by_human=bool(timeline.meta.edited_by_human),
            notes=timeline.meta.notes,
        ),
    )
    cut.renumber()

    _map_absolute_tracks(mig, timeline, cut)

    report_lines = [
        mig.line_for(beat, src) for beat, src in zip(cut.beats, mig.sources)
    ]

    with contextlib.ExitStack() as stack:
        # A dry run may never write the catalogue into the project, but the
        # validator reads it from ``analysis/sentences.json``; give it an
        # overlay of that directory with the in-memory catalogue in it.
        checked = project
        if overlay_needed:
            checked = stack.enter_context(_catalogue_overlay(project, catalogue))
        validation, resolved = _check(checked, cut, mig)

    duration_v1 = timeline.duration()
    duration_v2 = float(resolved.duration()) if resolved is not None else None
    drift = _duration_drift(mig, cut, resolved)

    report = MigrationReport(
        cut=cut,
        issues=mig.issues,
        notes=notes + mig.notes,
        validation=validation,
        duration_v1=duration_v1,
        duration_v2=duration_v2,
        beat_lines=report_lines,
        duration_drift=drift,
    )

    if dry_run:
        if out_dir is not None:
            report.written = _write_artifacts(Path(out_dir), cut, report)
        return report

    history = project.plan_dir / "history"
    archived = history / "timeline.v1.json"
    # The notes go in before the report is rendered, so the file on disk says
    # what this run did rather than what it was about to do.
    report.notes.append(f"v1 timeline archived as {project.rel(archived)}")
    if resolved is not None:
        report.notes.append(
            f"resolved v2 timeline written to {project.rel(project.timeline_file)}"
        )
    else:
        report.notes.append(
            f"{project.rel(project.timeline_file)} is gone until the cut validates — "
            "fix the errors above, then resolve again"
        )

    report.written = _write_artifacts(project.plan_dir, cut, report)
    history.mkdir(parents=True, exist_ok=True)
    shutil.move(str(project.timeline_file), str(archived))
    report.written.append(archived)
    if resolved is not None:
        resolved.save(project.timeline_file)
        report.written.append(project.timeline_file)
    project.set_stage(STAGE, "done", beats=len(cut.beats), issues=len(report.issues))
    return report


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class _OverlayProject(Project):
    """A project whose ``analysis/`` is read from somewhere else.

    Used only by a dry run: the validator loads the sentence catalogue from
    ``analysis/sentences.json``, and a dry run is not allowed to put one
    there, so it reads a temporary directory that shadows the real one.
    """

    def __init__(self, project: Project, analysis_dir: Path) -> None:
        super().__init__(project.path, project.settings)
        self._analysis_dir = analysis_dir

    @property
    def analysis_dir(self) -> Path:  # type: ignore[override]
        return self._analysis_dir


@contextlib.contextmanager
def _catalogue_overlay(project: Project, catalogue: Mapping[str, Any]) -> Any:
    """Yield a project view whose ``analysis/`` also holds ``catalogue``."""
    with tempfile.TemporaryDirectory(prefix="ytedit-migrate-") as tmp:
        overlay = Path(tmp)
        if project.analysis_dir.is_dir():
            for entry in project.analysis_dir.iterdir():
                if entry.name == "sentences.json":
                    continue
                with contextlib.suppress(OSError):
                    (overlay / entry.name).symlink_to(entry)
        (overlay / "sentences.json").write_text(
            json.dumps(catalogue, ensure_ascii=False), encoding="utf-8"
        )
        yield _OverlayProject(project, overlay)


def _check(project: Project, cut: Cut, mig: "_Migrator") -> tuple[list[str], Any]:
    """Run ``ytedit.cut``'s validator and resolver, tolerating either being absent."""
    validation: list[str] = []
    resolved: Any = None
    try:
        validation = [str(issue) for issue in cut_module.validate(project, cut)]
    except NotImplementedError:
        mig.notes.append("`ytedit.cut.validate` is not implemented yet — cut not validated")
    try:
        resolved = cut_module.resolve(project, cut)
    except NotImplementedError:
        mig.notes.append("`ytedit.cut.resolve` is not implemented yet — no v2 timeline written")
    except cut_module.CutError as exc:
        if not validation:
            validation = [str(issue) for issue in exc.issues]
        mig.notes.append(
            "the migrated cut does not resolve yet — fix the validation errors above "
            "in the editor, then re-resolve"
        )
    return validation, resolved


def _write_artifacts(directory: Path, cut: Cut, report: MigrationReport) -> list[Path]:
    """Write ``cut.json`` + ``migrate_report.md`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    cut_file = directory / "cut.json"
    try:
        cut_module.save_cut(cut, cut_file)
    except NotImplementedError:
        cut_file.write_text(
            json.dumps(cut.model_dump(by_alias=True, mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    report_file = directory / "migrate_report.md"
    report_file.write_text(report.markdown(), encoding="utf-8")
    return [cut_file, report_file]


def _map_absolute_tracks(mig: _Migrator, timeline: Timeline, cut: Cut) -> None:
    """Move captions/music/chapters/markers from seconds onto beat references."""
    for cap in timeline.tracks.captions:
        beat = mig.beat_at(cut.beats, cap.at)
        if beat is None:
            mig.issue("caption_dropped", cap.id, f"{cap.text!r} has no beat to sit on")
            continue
        start, _end = mig.spans[beat.uid]
        cut.captions.append(
            Caption(
                id=cap.id,
                beat=beat.id,
                offset=round(max(0.0, cap.at - start), 3),
                duration=round(max(0.1, cap.end - cap.at), 3),
                text=cap.text,
                style=cap.style,
                position=cap.position,
            )
        )

    for cue in timeline.tracks.music:
        first, last = mig.beat_range(cut.beats, cue.at, cue.end, cue.id)
        if first is None or last is None:
            mig.issue("music_dropped", cue.id, f"{cue.file} has no beat range")
            continue
        cut.music.append(
            MusicCue(
                id=cue.id,
                file=cue.file,
                **{"from": first.id},
                to=last.id,
                gain_db=cue.gain_db,
                fade_in=cue.fade_in,
                fade_out=cue.fade_out,
                duck=cue.duck,
            )
        )

    for chapter in timeline.chapters:
        beat = mig.beat_at(cut.beats, chapter.at)
        if beat is None:
            mig.issue("chapter_dropped", chapter.title, "no beat at this time")
            continue
        cut.chapters.append(Chapter(beat=beat.id, title=chapter.title))

    for marker in timeline.markers:
        beat = mig.beat_at(cut.beats, marker.at)
        if beat is None:
            mig.issue("marker_dropped", marker.label, "no beat at this time")
            continue
        cut.markers.append(Marker(beat=beat.id, label=marker.label))


def _duration_drift(mig: _Migrator, cut: Cut, resolved: Any) -> list[str]:
    """One line per beat whose resolved length left its v1 span by > 0.5 s."""
    if resolved is None:
        return []
    lengths: dict[str, float] = {}
    try:
        positions = resolved.segment_positions()
    except AttributeError:  # pragma: no cover - a resolver returning something else
        return []
    for pos in positions:
        uid = getattr(pos.segment, "beat", None)
        if not uid:
            continue
        lengths[str(uid)] = lengths.get(str(uid), 0.0) + (pos.end - pos.start)
    out: list[str] = []
    for beat in cut.beats:
        v1_start, v1_end = mig.spans.get(beat.uid, (0.0, 0.0))
        v1 = v1_end - v1_start
        v2 = lengths.get(beat.uid)
        if v2 is None:
            continue
        if abs(v2 - v1) > DURATION_DRIFT:
            out.append(
                f"`{beat.id}` {beat.kind}: v1 {v1:.2f} s → v2 {v2:.2f} s ({v2 - v1:+.2f} s)"
            )
    return out


__all__ = [
    "MAX_HANDOFF_OVERLAP",
    "SENTENCE_KEEP_RATIO",
    "MigrationError",
    "MigrationReport",
    "migrate_project",
]
