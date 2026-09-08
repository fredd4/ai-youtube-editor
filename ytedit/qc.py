"""Quality control: the playbook rule checker over a timeline and its master.

``qc(project)`` runs two families of checks and writes
``exports/qc_report.json`` plus a human-readable ``exports/qc_report.md``:

**Timeline rules** — pacing (delegated to :func:`ytedit.ai.plan.pacing_report`,
which implements research rules 5-9), the structural marker template (rule 4),
a clean final 20 s for the end screen, the Polish ending guard on the mapped
transcript (rule 8), vertical-clip handling (rule 15), caption bounds and the
safe-area font check (rule 21), Content-ID pre-flight (every
``background_music`` flag the footage log raised must be answered by a
``mute_range``, rule 19), the audio ledger dedupe check (no audio range may
play twice, rule 31), a true-break sentence check (a cut where the audio
actually stops landing mid-sentence, rule 32), a voice pickup overlapping a
segment's own narration (rule 33), an anchor that could not be resolved
against a stable segment (rule 34), and a voice pickup spilling past the
muted/ambient picture it was covered by (rule 35). Rules 31-35 are not in
``docs/research/youtube-production-playbook.md`` (1-30) — they encode rules
added afterwards; see :mod:`ytedit.ai.ledger` (31) and
:attr:`ytedit.timeline.VideoSegment.uid` (33-35, the stable-anchor fix).

**Rendered file** — container and codec conformance (rule 23: H.264 High,
yuv420p, faststart with ``moov`` ahead of ``mdat``, bt709 tags, AAC-LC
48 kHz), the measured programme loudness (rule 16: −14 LUFS ±1, true peak
≤ −1 dBTP) and the music-under-speech separation (rule 18: 14-18 LU).

Everything is advisory except the checks marked as errors; the return value is
``{"ok": bool, "errors": [...], "warnings": [...], "info": [...]}``.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from .log import console, get_logger
from .media import audio as audio_mod
from .media import captions as captions_mod
from .media.ffmpeg import FFmpegError, ff, ffprobe_json
from .project import Project, utcnow
from .timeline import Timeline, speech_ranges_from_transcripts

log = get_logger(__name__)

#: Polish wrap-up phrases that must not appear in the last seconds (rule 8).
ENDING_GUARD_RE = re.compile(
    r"dzięki\s+za\s+ogl[ąa]danie"
    r"|dzieki\s+za\s+ogl[ąa]danie"
    r"|do\s+zobaczenia"
    r"|na\s+tym\s+koń?cz[yę]m[ye]"
    r"|do\s+usłyszenia",
    re.IGNORECASE,
)

#: Roles that count as on-camera narration (mirrors ``ai.plan.AROLL_ROLE_RE``).
AROLL_ROLE_RE = re.compile(r"a-?roll|narration|talking|piece-to-camera|ptc", re.IGNORECASE)

#: Roles accepted as the opening beat of the programme.
OPENING_ROLE_RE = re.compile(r"cold-?open|hook|teaser", re.IGNORECASE)

#: How close a timeline marker must sit to the template position, in seconds.
MARKER_TOLERANCE: float = 5.0

#: Seconds at the end of the programme that must stay free of burned-in text.
END_SCREEN_SECONDS: float = 20.0

#: Window at the end scanned by the ending guard.
ENDING_GUARD_SECONDS: float = 3.0

#: Target integrated loudness tolerance in LU.
LOUDNESS_TOLERANCE: float = 1.0

#: Music must sit this far under the narration (rule 18).
#: Loudness of narration-bearing regions minus music-only regions, in LU.
#: Playbook: music in the clear sits at -18..-22 LUFS against narration at
#: -14, i.e. 4-8 LU under it; ducking under speech is enforced by the
#: render automation and cannot be isolated from a finished mix.
MUSIC_UNDER_SPEECH: tuple[float, float] = (3.0, 10.0)
#: Hard true-peak ceiling from YouTube's upload guidance.
YOUTUBE_TRUE_PEAK_DBTP = -1.0

#: Minimum audio needed in a window before its loudness is trustworthy.
MIN_MEASURE_SECONDS: float = 3.0

#: AAC bitrate floor for the master; below this it is only a warning down to...
AAC_BITRATE_TARGET: int = 320_000
#: ...this hard floor, under which it becomes an error.
AAC_BITRATE_FLOOR: int = 192_000

#: Rule 33: a voice pickup may overlap a segment carrying its own narration
#: by up to this much (float/frame noise) before it is an error.
VOICE_OVERLAP_TOLERANCE: float = 0.3


class QCError(RuntimeError):
    """QC could not run at all (no timeline, unreadable media)."""


class Report:
    """Accumulates the three severities and renders them."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.checks: dict[str, Any] = {}

    def error(self, message: str) -> None:
        """Record a hard failure."""
        self.errors.append(message)

    def warn(self, message: str) -> None:
        """Record an advisory finding."""
        self.warnings.append(message)

    def note(self, message: str) -> None:
        """Record a measurement worth showing."""
        self.info.append(message)

    def extend_warnings(self, messages: Iterable[str]) -> None:
        """Record several advisory findings."""
        self.warnings.extend(messages)

    @property
    def ok(self) -> bool:
        """True when nothing failed hard."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "info": list(self.info),
            "checks": dict(self.checks),
        }


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _mmss(seconds: float) -> str:
    """Format seconds as ``M:SS``."""
    minutes, secs = divmod(max(0.0, float(seconds)), 60)
    return f"{int(minutes)}:{secs:04.1f}"


def _clip_orientation(project: Project, clip_id: str) -> str:
    """Return the ingested orientation of a clip (``""`` when unknown)."""
    clip = project.load_state().get("clips", {}).get(clip_id) or {}
    return str(clip.get("orientation", ""))


def _clip_height(project: Project, clip_id: str) -> int:
    """Return the display height of a clip after rotation (0 when unknown)."""
    clip = project.load_state().get("clips", {}).get(clip_id) or {}
    width, height = int(clip.get("width") or 0), int(clip.get("height") or 0)
    if int(clip.get("rotation", 0)) % 360 in (90, 270):
        return width
    return height


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the overlap of two ranges in seconds."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _covered(span: tuple[float, float], ranges: Sequence[tuple[float, float]]) -> float:
    """Return the fraction of ``span`` covered by ``ranges`` (0..1)."""
    length = span[1] - span[0]
    if length <= 0:
        return 1.0
    return min(1.0, sum(_overlap(span, r) for r in ranges) / length)


def subtract(
    ranges: Sequence[tuple[float, float]], holes: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return ``ranges`` with every part of ``holes`` removed."""
    out: list[tuple[float, float]] = []
    for start, end in ranges:
        pieces = [(float(start), float(end))]
        for h0, h1 in holes:
            nxt: list[tuple[float, float]] = []
            for p0, p1 in pieces:
                if h1 <= p0 or h0 >= p1:
                    nxt.append((p0, p1))
                    continue
                if h0 > p0:
                    nxt.append((p0, h0))
                if h1 < p1:
                    nxt.append((h1, p1))
            pieces = nxt
        out += [(a, b) for a, b in pieces if b - a > 1e-6]
    return out


def total_length(ranges: Sequence[tuple[float, float]]) -> float:
    """Sum the length of a list of ranges."""
    return round(sum(max(0.0, e - s) for s, e in ranges), 3)


# ----------------------------------------------------------------------
# container inspection
# ----------------------------------------------------------------------
def top_level_atoms(path: Path | str, limit: int = 64) -> list[str]:
    """Return the MP4 top-level box types in file order.

    Reads only the box headers, so the cost is a handful of seeks even on a
    multi-gigabyte master.

    Args:
        path: MP4/MOV file.
        limit: Stop after this many boxes.

    Returns:
        Box types such as ``["ftyp", "moov", "mdat"]``; empty when the file is
        not an ISO base media file.
    """
    boxes: list[str] = []
    size_of = Path(path).stat().st_size
    offset = 0
    with open(path, "rb") as fh:
        while offset < size_of and len(boxes) < limit:
            fh.seek(offset)
            header = fh.read(8)
            if len(header) < 8:
                break
            (size,) = struct.unpack(">I", header[:4])
            kind = header[4:8].decode("ascii", "replace")
            if not kind.isprintable():
                break
            if size == 1:
                extended = fh.read(8)
                if len(extended) < 8:
                    break
                (size,) = struct.unpack(">Q", extended)
            elif size == 0:
                boxes.append(kind)
                break
            if size < 8:
                break
            boxes.append(kind)
            offset += size
    return boxes


def has_faststart(path: Path | str) -> bool | None:
    """True when ``moov`` precedes ``mdat`` (``None`` when neither is found)."""
    boxes = top_level_atoms(path)
    if "moov" not in boxes or "mdat" not in boxes:
        return None
    return boxes.index("moov") < boxes.index("mdat")


def has_edit_list(path: Path | str, probe_bytes: int = 4_000_000) -> bool:
    """True when an ``elst`` box appears in the head or tail of the file."""
    size = Path(path).stat().st_size
    window = min(probe_bytes, size)
    with open(path, "rb") as fh:
        head = fh.read(window)
        fh.seek(max(0, size - window))
        tail = fh.read(window)
    return b"elst" in head or b"elst" in tail


# ----------------------------------------------------------------------
# loudness of a set of ranges
# ----------------------------------------------------------------------
def measure_ranges_loudness(
    path: Path | str, ranges: Sequence[tuple[float, float]]
) -> dict[str, float] | None:
    """Measure integrated loudness over a selection of time ranges.

    The ranges are stitched together with ``aselect`` before ``loudnorm``
    analyses them, so gaps do not drag the integrated value down.

    Args:
        path: Media file with an audio stream.
        ranges: ``(start, end)`` ranges in seconds.

    Returns:
        The parsed loudnorm report, or ``None`` when there is too little audio
        (< :data:`MIN_MEASURE_SECONDS`) or the measurement failed.
    """
    usable = [(float(s), float(e)) for s, e in ranges if e - s > 1e-3]
    if total_length(usable) < MIN_MEASURE_SECONDS:
        return None
    expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in usable)
    chain = (
        f"aselect='{expr}',asetpts=N/SR/TB,"
        "loudnorm=I=-14:TP=-1:LRA=11:print_format=json"
    )
    try:
        stderr = ff("-i", str(path), "-map", "0:a:0", "-af", chain, "-f", "null", "-")
        return audio_mod.parse_loudnorm(stderr)
    except (FFmpegError, ValueError) as exc:
        log.warning("range loudness measurement failed: %s", exc)
        return None


# ----------------------------------------------------------------------
# timeline checks
# ----------------------------------------------------------------------
def check_timeline(project: Project, timeline: Timeline, report: Report) -> None:
    """Run every timeline-level playbook rule into ``report``."""
    # Resolve anchors against the timeline as it stands now (not whatever it
    # held when last saved) so rule 34 sees fresh anchor_issues and rules 33/
    # 35 check the pickups' actual current positions.
    timeline.resolve_anchors()
    total = timeline.duration()
    report.checks["duration"] = total
    report.checks["segments"] = len(timeline.tracks.video)
    report.note(f"programme {_mmss(total)} · {len(timeline.tracks.video)} segment(s)")

    for issue in timeline.validate(project):
        report.error(f"timeline: {issue}")

    # --- pacing (rules 5-9), delegated ---------------------------------
    report.extend_warnings(pacing_warnings(timeline, project.settings))

    # --- rule 4: structural markers ------------------------------------
    _check_markers(project, timeline, total, report)

    # --- clean end screen (final 20 s) ---------------------------------
    if total > END_SCREEN_SECONDS:
        window = (total - END_SCREEN_SECONDS, total)
        for cue in timeline.tracks.captions:
            if cue.style in ("subtitle", "hook") and _overlap((cue.at, cue.end), window) > 0:
                report.warn(
                    f"end screen: {cue.style} caption {cue.id!r} at {_mmss(cue.at)} sits in "
                    f"the last {END_SCREEN_SECONDS:.0f}s — keep it clean for the end screen"
                )

    # --- rule 8: ending guard on the transcript ------------------------
    _check_ending_guard(project, timeline, total, report)

    # --- rule 15: vertical handling ------------------------------------
    for pos in timeline.segment_positions():
        seg = pos.segment
        if _clip_orientation(project, seg.clip) != "vertical":
            continue
        if seg.transform.fit == "cover" and seg.transform.zoom <= 1.0 + 1e-6:
            height = _clip_height(project, seg.clip)
            suggestion = "crop-pan" if height >= 2160 else "blur-fill"
            report.warn(
                f"rule 15: vertical clip {seg.clip} in {seg.id} at {_mmss(pos.start)} uses "
                f"fit=cover with no zoom — it will be cropped hard; use {suggestion!r}"
            )

    # --- captions inside the programme ---------------------------------
    for cue in timeline.tracks.captions:
        if cue.at < -1e-6 or cue.end > total + 1e-3:
            report.error(
                f"caption {cue.id!r} ({_mmss(cue.at)}-{_mmss(cue.end)}) falls outside the "
                f"programme (0-{_mmss(total)})"
            )

    # --- rule 19: Content-ID pre-flight --------------------------------
    _check_music_flags(project, timeline, report)

    # --- opening beat ---------------------------------------------------
    segments = timeline.tracks.video
    if segments:
        role = segments[0].role or ""
        if not OPENING_ROLE_RE.search(role):
            report.warn(
                f"opening: first segment {segments[0].id} has role {role or '—'!r}; the "
                "playbook opens on a cold-open/hook within 2 s"
            )

    # --- A-roll share (measured; rule 9 warns via pacing_report) --------
    roles = [seg.role for seg in segments if seg.role]
    if roles and total > 0:
        # Role-keyed on purpose: an overlay cutaway (``audio_from`` set) is heard
        # as narration but seen as B-roll, and the share measures the picture.
        aroll = sum(
            seg.duration
            for seg in segments
            if AROLL_ROLE_RE.search(seg.role or "") and not seg.mute_source
        )
        share = aroll / total
        report.checks["aroll_share"] = round(share, 4)
        report.note(f"A-roll share {share * 100:.0f}% of the runtime (target 30-45%)")

    # --- rule 21: the caption font must carry Polish -------------------
    _check_font(project, timeline, report)

    # --- rule 31: no audio range may play twice -------------------------
    _check_duplicate_audio(project, timeline, report)

    # --- rule 32: a true cut landing mid-sentence ------------------------
    _check_sentence_breaks(project, timeline, report)

    # --- rule 33: a voice pickup over a segment's own narration ----------
    for issue in voice_pickup_overlap_issues(project, timeline):
        report.error(issue)

    # --- rule 34: an anchor that could not be resolved -------------------
    for issue in anchor_issue_messages(timeline):
        report.error(issue)

    # --- rule 35: a voice pickup spilling past its muted/ambient picture -
    for issue in voice_spill_issues(project, timeline):
        report.warn(issue)


def pacing_warnings(timeline: Timeline, settings: Any) -> list[str]:
    """Return :func:`ytedit.ai.plan.pacing_report` output, or ``[]``.

    The plan module is imported lazily so QC still runs in an installation
    where the AI layer is unavailable.
    """
    try:
        from .ai.plan import pacing_report
    except ImportError:  # pragma: no cover - AI layer not installed
        log.debug("ytedit.ai.plan is unavailable; skipping the pacing rules")
        return []
    try:
        return list(pacing_report(timeline, settings))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("pacing_report failed: %s", exc)
        return []


def _check_markers(project: Project, timeline: Timeline, total: float, report: Report) -> None:
    """Rule 4: the structural beat template must be marked."""
    template = project.settings.get("pacing.markers", []) or []
    present = [(float(m.at), str(m.label)) for m in timeline.markers]
    missing: list[str] = []
    for entry in template:
        at = float(entry.get("at", 0.0))
        label = str(entry.get("label", ""))
        target = total + at if at < 0 else at
        if target < 0 or target > total + 1e-6:
            continue  # the programme is too short for this beat
        hit = any(
            abs(pos - target) <= MARKER_TOLERANCE or lbl == label for pos, lbl in present
        )
        if not hit:
            missing.append(f"{label} @ {_mmss(target)}")
    report.checks["markers_missing"] = missing
    if missing:
        report.warn("rule 4: no beat marked for " + ", ".join(missing))


def _check_ending_guard(
    project: Project, timeline: Timeline, total: float, report: Report
) -> None:
    """Rule 8: no Polish wrap-up phrase in the final seconds."""
    if total <= 0:
        return
    window = (max(0.0, total - ENDING_GUARD_SECONDS), total)
    try:
        cues = captions_mod.srt_from_transcripts(project, timeline, project.settings)
    except Exception as exc:  # pragma: no cover - unreadable transcripts
        log.warning("cannot map the transcripts for the ending guard: %s", exc)
        return
    spoken = " ".join(
        cue.text.replace("\n", " ") for cue in cues if _overlap((cue.start, cue.end), window) > 0
    )
    match = ENDING_GUARD_RE.search(spoken)
    if match:
        report.warn(
            f"rule 8: wrap-up phrase {match.group(0)!r} in the last "
            f"{ENDING_GUARD_SECONDS:.0f}s — end on the payoff, then hard cut"
        )


def _check_music_flags(project: Project, timeline: Timeline, report: Report) -> None:
    """Rule 19: every flagged background-music range must be muted or ducked."""
    log_file = project.analysis_dir / "footage_log.json"
    if not log_file.exists():
        return
    try:
        document = json.loads(log_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - corrupt log
        log.warning("unreadable %s", log_file)
        return

    unanswered = 0
    for flag in document.get("music_flags", []) or []:
        suggest = str(flag.get("suggest", "mute"))
        if suggest == "keep":
            continue
        clip = str(flag.get("clip", ""))
        span = (float(flag.get("s", 0.0)), float(flag.get("e", 0.0)))
        if span[1] <= span[0]:
            continue
        used = [
            (max(span[0], pos.segment.in_), min(span[1], pos.segment.out))
            for pos in timeline.segment_positions()
            if pos.segment.clip == clip
        ]
        used = [r for r in used if r[1] > r[0]]
        if not used:
            continue  # the flagged range never made the cut
        mutes = [
            (float(m.s), float(m.e))
            for m in timeline.mute_ranges
            if m.clip == clip and m.gain_db <= -6.0
        ]
        for piece in used:
            coverage = _covered(piece, mutes)
            if coverage < 0.9:
                unanswered += 1
                report.warn(
                    f"rule 19: {clip} {_mmss(piece[0])}-{_mmss(piece[1])} was flagged as "
                    f"background music (suggest {suggest!r}) but only {coverage * 100:.0f}% "
                    "of it is covered by a mute_range — mute, duck or accept it"
                )
    report.checks["music_flags_unanswered"] = unanswered


def _check_font(project: Project, timeline: Timeline, report: Report) -> None:
    """Rule 21: hard-fail when the caption font lacks a needed glyph."""
    name, font_file = project.settings.caption_font()
    needed = captions_mod.POLISH_GLYPHS + "".join(c.text for c in timeline.tracks.captions)
    try:
        captions_mod.check_font(font_file, needed, name=name)
        report.checks["caption_font"] = str(font_file)
    except captions_mod.CaptionFontError as exc:
        report.error(f"rule 21: {exc}")


def _check_duplicate_audio(project: Project, timeline: Timeline, report: Report) -> None:
    """Rule 31 (not in the 1-30 research playbook — the audio ledger dedupe
    pass added afterwards): no audio range may be heard more than once.

    Speech playing twice is an ERROR — it is the exact defect
    :mod:`ytedit.ai.ledger` exists to fix, so seeing one here means a
    hand-edited ``timeline.json`` was never run back through ``ytedit tidy``.
    A repeated *ambient* range (no transcript word in it) is only a WARNING:
    the dedupe pass allows those by default since a repeated splash of crowd
    noise or wind is harmless.
    """
    try:
        from .ai.ledger import find_duplicate_audio
    except ImportError:  # pragma: no cover - AI layer not installed
        log.debug("ytedit.ai.ledger is unavailable; skipping the audio-dedupe rule")
        return
    try:
        findings = find_duplicate_audio(timeline, project)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("find_duplicate_audio failed: %s", exc)
        return
    for finding in findings:
        ids = " / ".join(finding.ids)
        message = (
            f"rule 31: {finding.clip} {_mmss(finding.start)}-{_mmss(finding.end)} "
            f"({finding.duration:.2f}s) plays twice — {ids}"
        )
        if finding.speech:
            report.error(message)
        else:
            report.warn(message + " (ambient — allowed by default, but re-run `ytedit tidy`)")


def _check_sentence_breaks(project: Project, timeline: Timeline, report: Report) -> None:
    """Rule 32 (not in the 1-30 research playbook): a true cut — the audio
    actually stops there, see :func:`ytedit.ai.tidy._is_continuous_handoff` —
    landing in the middle of a sentence.

    ``ytedit tidy`` already tries to fix every one of these (extend, or
    retract/advance to the nearest complete sentence when extension is
    blocked); a finding here means either ``tidy`` was never re-run after a
    hand edit, or the cap/guard left it capped mid-sentence on purpose.
    """
    try:
        from .ai.tidy import (
            _in_head,
            _is_continuous_handoff,
            _out_tail,
            has_sentence_marks,
            load_words,
        )
    except ImportError:  # pragma: no cover - AI layer not installed
        log.debug("ytedit.ai.tidy is unavailable; skipping the sentence-break rule")
        return

    cfg = project.settings
    gap_max = max(0.0, float(cfg.get("pacing.sentence_gap_max", 1.2)))
    merge_gap = max(0.0, float(cfg.get("pacing.merge_gap", 0.15)))
    words_cache: dict[str, list] = {}
    video = timeline.tracks.video

    def words_for(clip_id: str) -> list:
        if clip_id not in words_cache:
            words_cache[clip_id] = load_words(project, clip_id)
        return words_cache[clip_id]

    for index, seg in enumerate(video):
        if seg.mute_source:
            continue
        words = words_for(seg.clip)
        if not words or not has_sentence_marks(words):
            continue

        next_seg = video[index + 1] if index + 1 < len(video) else None
        if next_seg is None or not _is_continuous_handoff(seg, next_seg, merge_gap):
            found = _out_tail(seg, words, gap_max)
            if found is not None:
                _last, tail = found
                report.warn(
                    f"rule 32: {seg.id} out at {_mmss(seg.out)} cuts mid-sentence "
                    f"— next word would be '{tail[0].text}'"
                )

        prev_seg = video[index - 1] if index > 0 else None
        if prev_seg is None or not _is_continuous_handoff(prev_seg, seg, merge_gap):
            found_in = _in_head(seg, words, gap_max)
            if found_in is not None:
                _first, head = found_in
                report.warn(
                    f"rule 32: {seg.id} in at {_mmss(seg.in_)} opens mid-sentence "
                    f"— previous word was '{head[0].text}'"
                )


def voice_pickup_overlap_issues(
    project: Project, timeline: Timeline, tolerance: float = VOICE_OVERLAP_TOLERANCE
) -> list[str]:
    """Rule 33 (ERROR): a voice pickup must not run over a segment carrying
    its own narration.

    This is the exact failure mode that motivated stable segment identity
    (see :attr:`ytedit.timeline.VideoSegment.uid`): a hand-edit or a stale
    anchor can leave a pickup's absolute position sitting on top of a shot
    that has its own spoken narration — the CTA plays over someone else's
    sentence, or a narration pickup starts before the picture it was written
    for. "Carries its own narration" means the segment is not muted and
    either its own ``[in, out)`` (no ``audio_from``) or its ``audio_from``
    range has a transcript word in it.

    Args:
        project: Project supplying transcripts.
        timeline: The EDL to check.
        tolerance: Overlap under this many seconds is not reported (frame
            rounding, a deliberate handoff at the very edge of a cut).

    Returns:
        One message per offending (voice item, segment) pair.
    """
    words_cache: dict[str, list] = {}

    def has_words(clip: str, s: float, e: float) -> bool:
        if clip not in words_cache:
            from .ai.tidy import load_words
            words_cache[clip] = load_words(project, clip)
        return any(w.e > s + 1e-6 and w.s < e - 1e-6 for w in words_cache[clip])

    issues: list[str] = []
    positions = timeline.segment_positions()
    for item in timeline.tracks.voice:
        end = item.end if item.end is not None else item.at
        if end <= item.at:
            continue
        for pos in positions:
            overlap = _overlap((item.at, end), (pos.start, pos.end))
            if overlap <= tolerance + 1e-9:
                continue
            seg = pos.segment
            if seg.mute_source:
                continue
            if seg.audio_from is not None:
                clip, s, e = seg.audio_from.clip, seg.audio_from.in_, seg.audio_from.out
            else:
                clip, s, e = seg.clip, seg.in_, seg.out
            if not has_words(clip, s, e):
                continue
            issues.append(
                f"rule 33: voice {item.id} ({_mmss(item.at)}-{_mmss(end)}) overlaps "
                f"{seg.id} ({clip} {_mmss(pos.start)}-{_mmss(pos.end)}) by {overlap:.2f}s, "
                "which carries its own narration"
            )
    return issues


def anchor_issue_messages(timeline: Timeline) -> list[str]:
    """Rule 34 (ERROR): an anchor :meth:`Timeline.resolve_anchors` could not
    resolve — uid not found and no unique signature match either.

    Call ``timeline.resolve_anchors()`` before this so
    :attr:`ytedit.timeline.Meta.anchor_issues` reflects the timeline as it
    stands now, not whatever it held when the file was last saved.
    """
    return [f"rule 34: {issue}" for issue in timeline.meta.anchor_issues]


def voice_spill_issues(project: Project, timeline: Timeline) -> list[str]:
    """Rule 35 (WARNING): a voice pickup whose end runs past the last
    muted/ambient picture under it — the narration spills into the next
    scene instead of ending inside the B-roll it was covered by.

    Walks forward from the segment the pickup starts under, for as long as
    each further segment is muted or ambient (reuses
    :func:`ytedit.ai.voice._is_coverable`, the same notion the pickup-growth
    pass uses to decide what it may run over); if the pickup's end reaches
    past where that run stops, it is spilling into a segment that was never
    grown/inserted to cover it.
    """
    try:
        from .ai.voice import _is_coverable
    except ImportError:  # pragma: no cover - AI layer not installed
        log.debug("ytedit.ai.voice is unavailable; skipping the voice-spill rule")
        return []

    issues: list[str] = []
    positions = timeline.segment_positions()
    for item in timeline.tracks.voice:
        end = item.end if item.end is not None else item.at
        if end <= item.at:
            continue
        # The segment that is on screen when the pickup starts: a pickup
        # anchored to a cut starts exactly at one segment's end / the next
        # one's start, so the end bound is exclusive.
        idx = next(
            (i for i, pos in enumerate(positions) if pos.start - 1e-6 <= item.at < pos.end - 1e-6),
            None,
        )
        if idx is None:
            continue
        run_end = positions[idx].start
        i = idx
        while i < len(positions) and _is_coverable(project, positions[i].segment):
            run_end = positions[i].end
            i += 1
        if end > run_end + 0.05:
            issues.append(
                f"rule 35: voice {item.id} ends at {_mmss(end)}, {end - run_end:.2f}s past "
                f"the muted/ambient picture under it (ends {_mmss(run_end)}) — spills into "
                "the next scene"
            )
    return issues


# ----------------------------------------------------------------------
# rendered-file checks
# ----------------------------------------------------------------------
def find_rendered(project: Project) -> Path | None:
    """Return the newest master, falling back to the preview."""
    candidates = sorted(project.exports_dir.glob("master_*.mp4"))
    candidates += [project.renders_dir / "preview.mp4"]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def check_rendered(
    project: Project, timeline: Timeline, rendered: Path, report: Report, master: bool
) -> None:
    """Run every file-level playbook rule over a rendered programme."""
    report.checks["rendered"] = project.rel(rendered)
    report.checks["rendered_master"] = master
    try:
        probe = ffprobe_json(rendered)
    except FFmpegError as exc:
        report.error(f"cannot probe {rendered.name}: {exc}")
        return

    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = probe.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    report.checks["rendered_duration"] = round(duration, 3)

    expected = timeline.duration()
    if expected and abs(duration - expected) > 0.25:
        report.warn(
            f"rendered duration {_mmss(duration)} differs from the timeline "
            f"({_mmss(expected)}) by {abs(duration - expected):.2f}s"
        )

    _check_video_stream(video, timeline, report, master)
    _check_audio_stream(audio, report, master)
    _check_container(rendered, streams, report, master)
    _check_loudness(project, timeline, rendered, report, master)


def _check_video_stream(
    video: dict[str, Any] | None, timeline: Timeline, report: Report, master: bool
) -> None:
    """Rule 23: codec, profile, pixel format, colour tags, geometry."""
    if video is None:
        report.error("the rendered file has no video stream")
        return
    codec = str(video.get("codec_name", ""))
    profile = str(video.get("profile", ""))
    pix_fmt = str(video.get("pix_fmt", ""))
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    report.checks["video"] = {
        "codec": codec, "profile": profile, "pix_fmt": pix_fmt,
        "width": width, "height": height,
        "color_primaries": video.get("color_primaries"),
        "color_transfer": video.get("color_transfer"),
        "color_space": video.get("color_space"),
    }
    report.note(f"video {codec} {profile} {width}x{height} {pix_fmt}")

    if codec != "h264":
        report.error(f"rule 23: video codec is {codec!r}, expected h264")
    if pix_fmt != "yuv420p":
        report.error(f"rule 23: pixel format is {pix_fmt!r}, expected yuv420p")
    if master and profile.lower() != "high":
        report.error(f"rule 23: H.264 profile is {profile!r}, expected High")

    fps = _fps(video)
    report.checks["fps"] = fps
    if timeline.fps and fps and abs(fps - timeline.fps) > 0.05:
        report.warn(f"frame rate is {fps:g} fps, the timeline asks for {timeline.fps}")

    if master and (width, height) != (timeline.width, timeline.height):
        report.warn(
            f"master resolution {width}x{height} differs from the timeline canvas "
            f"{timeline.width}x{timeline.height}"
        )

    tags = {
        "color_primaries": str(video.get("color_primaries") or "unknown"),
        "color_trc": str(video.get("color_transfer") or "unknown"),
        "colorspace": str(video.get("color_space") or "unknown"),
    }
    wrong = {k: v for k, v in tags.items() if v != "bt709"}
    if wrong and master:
        report.error(
            "rule 23: colour tags are not bt709: "
            + ", ".join(f"{k}={v}" for k, v in sorted(wrong.items()))
        )
    elif wrong:
        report.warn(
            "colour tags are not bt709: "
            + ", ".join(f"{k}={v}" for k, v in sorted(wrong.items()))
        )


def _fps(video: dict[str, Any]) -> float:
    """Return the average frame rate of a probed video stream."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(video.get(key) or "")
        if "/" in raw:
            num, _, den = raw.partition("/")
            try:
                if float(den):
                    return round(float(num) / float(den), 3)
            except ValueError:  # pragma: no cover
                continue
    return 0.0


def _check_audio_stream(audio: dict[str, Any] | None, report: Report, master: bool) -> None:
    """Rule 20: AAC-LC, 48 kHz, stereo, 384 kbps (320 kbps in practice)."""
    if audio is None:
        report.error("the rendered file has no audio stream")
        return
    codec = str(audio.get("codec_name", ""))
    profile = str(audio.get("profile", ""))
    rate = int(audio.get("sample_rate") or 0)
    channels = int(audio.get("channels") or 0)
    bitrate = int(audio.get("bit_rate") or 0)
    report.checks["audio"] = {
        "codec": codec, "profile": profile, "sample_rate": rate,
        "channels": channels, "bit_rate": bitrate,
    }
    report.note(f"audio {codec} {profile} {rate} Hz {channels}ch {bitrate / 1000:.0f} kbps")

    if codec != "aac":
        report.error(f"rule 20: audio codec is {codec!r}, expected aac")
    elif profile and profile.upper() not in ("LC", "AAC-LC"):
        report.warn(f"rule 20: AAC profile is {profile!r}, expected LC")
    if rate != 48000:
        report.error(f"rule 20: sample rate is {rate} Hz, expected 48000")
    if channels != 2:
        report.warn(f"rule 20: {channels} channel(s), expected stereo")
    if master and bitrate:
        if bitrate < AAC_BITRATE_FLOOR:
            report.error(
                f"rule 20: audio bitrate is {bitrate / 1000:.0f} kbps, far below the "
                f"{AAC_BITRATE_TARGET / 1000:.0f} kbps floor"
            )
        elif bitrate < AAC_BITRATE_TARGET:
            report.warn(
                f"rule 20: audio bitrate is {bitrate / 1000:.0f} kbps; the playbook asks "
                "for 384 kbps (ffmpeg's native aac encoder caps near 256 kbps — build "
                "with libfdk_aac or use aac_at)"
            )


def _check_container(
    rendered: Path, streams: Sequence[dict[str, Any]], report: Report, master: bool
) -> None:
    """Rule 23: faststart and no *shifting* edit lists.

    ffmpeg's mov muxer always writes an ``elst`` box (an identity one, plus the
    AAC priming delay), which is harmless. Only an edit list that actually
    shifts a stream — a non-zero ``start_time`` — is worth reporting.
    """
    faststart = has_faststart(rendered)
    report.checks["faststart"] = faststart
    if faststart is False:
        message = "rule 23: moov comes after mdat — re-mux with -movflags +faststart"
        report.error(message) if master else report.warn(message)
    elif faststart is None:
        report.warn("rule 23: cannot find moov/mdat; is this an MP4?")

    report.checks["edit_list"] = has_edit_list(rendered)
    shifted = {
        str(s.get("codec_type", "?")): float(s.get("start_time") or 0.0)
        for s in streams
        if abs(float(s.get("start_time") or 0.0)) > 1e-3
    }
    report.checks["stream_start_times"] = shifted
    if shifted:
        report.warn(
            "rule 23: an edit list shifts "
            + ", ".join(f"{k} by {v:+.3f}s" for k, v in sorted(shifted.items()))
            + " — strip the edit lists before upload"
        )


def _check_loudness(
    project: Project, timeline: Timeline, rendered: Path, report: Report, master: bool
) -> None:
    """Rules 16 and 18: programme loudness and music-under-speech separation."""
    loud_cfg = project.settings.section("audio").get("loudnorm", {})
    target_i = float(loud_cfg.get("I", -14))
    target_tp = float(loud_cfg.get("TP", -1))
    try:
        measured = audio_mod.measure_loudness(rendered, I=target_i, TP=target_tp)
    except (FFmpegError, ValueError) as exc:
        report.error(f"cannot measure the loudness of {rendered.name}: {exc}")
        return

    integrated = measured.get("input_i", 0.0)
    true_peak = measured.get("input_tp", 0.0)
    report.checks["loudness"] = {
        "integrated_lufs": round(integrated, 2),
        "true_peak_dbtp": round(true_peak, 2),
        "lra": round(measured.get("input_lra", 0.0), 2),
    }
    report.note(
        f"loudness {integrated:.2f} LUFS · true peak {true_peak:.2f} dBTP · "
        f"LRA {measured.get('input_lra', 0.0):.1f} LU"
    )

    tolerance = LOUDNESS_TOLERANCE if master else LOUDNESS_TOLERANCE + 0.5
    if abs(integrated - target_i) > tolerance:
        message = (
            f"rule 16: integrated loudness {integrated:.2f} LUFS is "
            f"{integrated - target_i:+.2f} LU off the {target_i:g} LUFS target"
        )
        report.error(message) if master else report.warn(message)
    # YouTube's hard ceiling is -1 dBTP; the configured TP is a safety target.
    if true_peak > YOUTUBE_TRUE_PEAK_DBTP + 0.05:
        message = (
            f"rule 16: true peak {true_peak:.2f} dBTP exceeds YouTube's "
            f"{YOUTUBE_TRUE_PEAK_DBTP:g} dBTP ceiling"
        )
        report.error(message) if master else report.warn(message)
    elif true_peak > target_tp + 0.05:
        report.warn(
            f"rule 16: true peak {true_peak:.2f} dBTP is above the {target_tp:g} dBTP "
            "safety target (still within YouTube's -1 dBTP ceiling)"
        )

    _check_music_separation(project, timeline, rendered, report)


def _check_music_separation(
    project: Project, timeline: Timeline, rendered: Path, report: Report
) -> None:
    """Rule 18: verify the music sits 14-18 LU under the narration."""
    if not timeline.tracks.music:
        return
    duck_cfg = project.settings.section("ducking")
    try:
        speech = speech_ranges_from_transcripts(
            project,
            timeline,
            merge_gap=float(duck_cfg.get("merge_gap", 0.4)),
            pad=float(duck_cfg.get("pad", 0.15)),
        )
    except Exception as exc:  # pragma: no cover - unreadable transcripts
        log.warning("cannot derive speech ranges: %s", exc)
        return
    speech += [
        (float(v.at), float(v.end) if v.end is not None else float(v.at) + 1.0)
        for v in timeline.tracks.voice
    ]
    speech = sorted(r for r in speech if r[1] > r[0])
    if not speech:
        return

    music_spans = [(float(c.at), float(c.end)) for c in timeline.tracks.music if c.end > c.at]
    music_only = subtract(music_spans, speech)
    speech_under_music = [
        (max(s, m0), min(e, m1))
        for s, e in speech
        for m0, m1 in music_spans
        if min(e, m1) - max(s, m0) > 1e-3
    ]

    report.checks["speech_seconds"] = total_length(speech_under_music)
    report.checks["music_only_seconds"] = total_length(music_only)

    with_speech = measure_ranges_loudness(rendered, speech_under_music)
    without_speech = measure_ranges_loudness(rendered, music_only)
    if not with_speech or not without_speech:
        report.note(
            "rule 18: not enough material to measure the music-under-speech separation "
            f"(speech {total_length(speech_under_music):.1f}s, "
            f"music-only {total_length(music_only):.1f}s)"
        )
        return

    delta = with_speech.get("input_i", 0.0) - without_speech.get("input_i", 0.0)
    report.checks["music_under_speech_lu"] = round(delta, 2)
    report.note(f"music-only sections sit {delta:.1f} LU under the narration sections (target 4-8 LU)")
    low, high = MUSIC_UNDER_SPEECH
    if delta < low:
        report.warn(
            f"rule 18: music-only sections are only {delta:.1f} LU under narration "
            f"(target {low:.0f}-{high:.0f} LU) — duck it harder"
        )
    elif delta > high:
        report.warn(
            f"rule 18: music-only sections are {delta:.1f} LU under narration "
            f"(target {low:.0f}-{high:.0f} LU) — it will be inaudible"
        )


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def render_markdown(project: Project, report: Report) -> str:
    """Render the report as Markdown for ``exports/qc_report.md``."""
    checks = report.checks
    lines = [
        f"# QC report — {project.slug}",
        "",
        f"*Generated {utcnow()} — "
        f"**{'PASS' if report.ok else 'FAIL'}**, {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s).*",
        "",
    ]
    rendered = checks.get("rendered")
    if rendered:
        lines += [f"Rendered file: `{rendered}`", ""]

    for title, items in (
        ("Errors", report.errors),
        ("Warnings", report.warnings),
        ("Measurements", report.info),
    ):
        lines.append(f"## {title}")
        lines.append("")
        if items:
            lines += [f"- {item}" for item in items]
        else:
            lines.append("- none")
        lines.append("")

    lines += ["## Raw checks", "", "```json",
              json.dumps(checks, indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def print_report(report: Report) -> None:
    """Print the report as a rich table on the shared console."""
    from rich.table import Table

    table = Table(title="QC", show_lines=False, header_style="bold")
    table.add_column("severity", style="bold", no_wrap=True)
    table.add_column("finding")
    for message in report.errors:
        table.add_row("[bold red]error[/]", message)
    for message in report.warnings:
        table.add_row("[yellow]warning[/]", message)
    for message in report.info:
        table.add_row("[green]info[/]", message)
    if not (report.errors or report.warnings or report.info):
        table.add_row("[green]ok[/]", "nothing to report")
    console.print(table)
    verdict = "[green]PASS[/]" if report.ok else "[bold red]FAIL[/]"
    console.print(
        f"{verdict} — {len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------
def qc(
    project: Project,
    rendered: Path | str | None = None,
    timeline_path: Path | str | None = None,
    show_table: bool = True,
) -> dict[str, Any]:
    """Check a project's timeline (and its rendered master) against the playbook.

    Args:
        project: Project to check.
        rendered: Rendered file to inspect. When omitted the newest
            ``exports/master_*.mp4`` is used, falling back to
            ``renders/preview.mp4``; when nothing is rendered only the
            timeline rules run.
        timeline_path: Timeline to check (default ``plan/timeline.json``).
        show_table: Print the rich summary table.

    Returns:
        ``{"ok": bool, "errors": [...], "warnings": [...], "info": [...],
        "checks": {...}}``. The same document is written to
        ``exports/qc_report.json`` alongside ``exports/qc_report.md``.

    Raises:
        QCError: When there is no timeline to check.
    """
    source = Path(timeline_path) if timeline_path else project.timeline_file
    if not source.exists():
        raise QCError(f"no timeline at {source} — run the plan stage first")

    project.ensure_dirs()
    project.set_stage("qc", "running")
    report = Report()
    try:
        timeline = Timeline.load(source)
        log.info("qc [stage]timeline[/] · %s", project.rel(source))
        check_timeline(project, timeline, report)

        target = Path(rendered) if rendered else find_rendered(project)
        if target is not None and target.exists():
            master = target.parent == project.exports_dir
            log.info("qc [stage]rendered[/] · %s", project.rel(target))
            check_rendered(project, timeline, target, report, master=master)
        else:
            report.warn("no rendered file found — run `ytedit render` before the file checks")

        document = report.to_dict()
        document["project"] = project.slug
        document["timeline"] = project.rel(source)
        document["generated"] = utcnow()

        json_path = project.exports_dir / "qc_report.json"
        json_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        md_path = project.exports_dir / "qc_report.md"
        md_path.write_text(render_markdown(project, report), encoding="utf-8")

        if show_table:
            print_report(report)
        log.info(
            "qc done: %d error(s), %d warning(s) -> %s",
            len(report.errors), len(report.warnings), project.rel(md_path),
        )
        project.set_stage(
            "qc",
            "done",
            ok=report.ok,
            errors=len(report.errors),
            warnings=len(report.warnings),
            report=project.rel(json_path),
        )
        return document
    except QCError:
        raise
    except Exception as exc:
        project.set_stage("qc", "error", error=str(exc)[:2000])
        raise
