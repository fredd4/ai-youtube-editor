"""Render verification — ``ytedit check-render <slug>``.

Transcribes the *finished* video's audio and lines every word up against the
timeline, so a cut can be judged by what it actually sounds like rather than by
what the EDL claims. This is the productised version of the hand-rolled pass
that found the the reference project pickup ending inside "początkowo": the plan was internally
consistent, the render was not.

What it checks, for every place where the audio really changes source:

* **source side** — the margin between the cut and the nearest transcript word
  of the source clip (``out - last word end`` before, ``first word start - in``
  after), whether ``in``/``out`` land *inside* a word, and whether the two
  sides replay the same stretch of the same clip;
* **render side** — the gap between the last word transcribed before the
  boundary and the first one after it, and any word that straddles the
  boundary (a word chopped in half by the cut, which is what a listener hears
  as a swallowed syllable).

Boundaries are:

* consecutive video segments whose :attr:`~ytedit.timeline.VideoSegment.audio_source`
  is not the same clip continuing within one frame (a continuous hand-off —
  a jump cut, or a shot cutting away over running narration — is not a cut at
  all and is skipped, see :func:`is_continuous`). Two segments of the *same*
  beat are contiguous by construction since cut v2: the resolver lays a beat
  out as one unbroken audio range, so ``prev.audio_out == next.audio_in``
  always holds and a within-beat picture change is never an audio boundary;
* the start and the end of every ``tracks.voice`` pickup.

Every boundary is labelled with the beats on both sides (``b012 -> b013``, or
``b012 (shot 2) -> b012 (own picture)`` for a hand-off inside one beat),
because a correction is applied to ``plan/cut.json`` — the timeline is
derived and nothing edits it.

Everything is reported, nothing fails: this is a report, not a gate
(``ytedit qc`` is the gate). Render times come from
:meth:`~ytedit.timeline.Timeline.segment_positions` with ``fade_overlaps=True``
— exactly what :func:`ytedit.media.render.render_positions` lays out — and
timeline-time items (voice pickups) are mapped forward through
:func:`ytedit.media.render.build_time_map`, the same convention ``ytedit at``
uses.

The render transcript costs one ElevenLabs Scribe call per render (~$0.006/min)
and is cached in ``renders/check/<name>.transcript.json`` against the render
file's mtime and size, so re-running after a report tweak is free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .words import Word, load_words
from .ai.transcribe import cost_recorder, to_iso3
from .log import get_logger
from .media.ffmpeg import ff
from .media.render import build_time_map
from .project import Project
from .timeline import Timeline, VideoSegment

if TYPE_CHECKING:  # pragma: no cover
    from .cut import Cut

log = get_logger(__name__)

STAGE = "check-render"

_EPS = 1e-9

#: Render "words" longer than this are Scribe artefacts (music mistaken for
#: speech) and are dropped before any alignment.
MAX_RENDER_WORD: float = 3.0

#: How far either side of a boundary a render word is looked for.
SEARCH_WINDOW: float = 2.5

#: Slack allowed when deciding whether a render word sits before/after a
#: boundary (a word may start a frame or two early).
EDGE_TOLERANCE: float = 0.05

#: A render word overlapping the boundary by more than this on *both* sides was
#: chopped by the cut.
STRADDLE: float = 0.08

#: Less air than this between the two sides in the render reads as a hard join.
MIN_RENDER_GAP: float = 0.35

#: A source ``in``/``out`` this far inside a word counts as cutting the word.
INSIDE_WORD: float = 0.02

#: Same-clip audio replayed by more than this on both sides of a boundary.
MIN_REPLAY: float = 0.05

class VerifyError(RuntimeError):
    """The render (or its transcript) could not be prepared."""


# ----------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------
def timecode(seconds: float) -> str:
    """``mm:ss.ss`` (``h:mm:ss.ss`` past an hour)."""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"
    return f"{int(minutes):02d}:{secs:05.2f}"


# ----------------------------------------------------------------------
# the report model
# ----------------------------------------------------------------------
@dataclass
class Side:
    """One side of a boundary, as the *source* transcript sees it."""

    #: ``"segment"`` or ``"voice"``.
    kind: str
    #: Display id of the segment (or the voice item).
    id: str
    clip: str | None = None
    in_: float = 0.0
    out: float = 0.0
    #: The last word before the cut (leading side) / the first after it
    #: (trailing side), when the source clip has a transcript.
    word: str | None = None
    word_s: float | None = None
    word_e: float | None = None
    #: ``out - word.e`` on a leading side, ``word.s - in`` on a trailing one.
    margin: float | None = None
    #: The cut lands inside :attr:`word`.
    inside_word: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["in"] = data.pop("in_")
        return data


@dataclass
class BoundaryReport:
    """Everything known about one audio boundary."""

    #: Render time of the boundary (seconds).
    at: float
    #: ``"cut"``, ``"voice-in"`` or ``"voice-out"``.
    kind: str
    #: Short label for the table, e.g. ``s031 -> s032`` or ``v002 start``.
    label: str
    before: Side | None = None
    after: Side | None = None
    #: The cut beat on each side, e.g. ``b012 (shot 2)`` — where an edit goes.
    beat_before: str | None = None
    beat_after: str | None = None
    #: Seconds of the same clip's audio played on both sides (0 when none).
    replayed: float = 0.0
    #: Silence between the last render word before and the first after.
    render_gap: float | None = None
    render_before: str | None = None
    render_after: str | None = None
    #: A render word chopped by the boundary, as ``{"text", "s", "e"}``.
    straddle: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def beats(self) -> str:
        """``b012 -> b013`` — the cut beats this boundary sits between."""
        if self.beat_before is None and self.beat_after is None:
            return "-"
        return f"{self.beat_before or '-'} -> {self.beat_after or '-'}"

    def render_phrase(self) -> str:
        """``...last [0.42s] first...`` — what the boundary sounds like."""
        if self.render_gap is None:
            return "(no words both sides)"
        return (
            f"...{self.render_before} [{self.render_gap:.2f}s] {self.render_after}..."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": round(self.at, 3),
            "at_tc": timecode(self.at),
            "kind": self.kind,
            "label": self.label,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "beat_before": self.beat_before,
            "beat_after": self.beat_after,
            "beats": self.beats,
            "replayed": round(self.replayed, 3),
            "render_gap": None if self.render_gap is None else round(self.render_gap, 3),
            "render_before": self.render_before,
            "render_after": self.render_after,
            "straddle": self.straddle,
            "flags": list(self.flags),
        }


# ----------------------------------------------------------------------
# render files
# ----------------------------------------------------------------------
def resolve_render(project: Project, timeline: Timeline, spec: str = "draft") -> Path:
    """Turn ``draft``/``preview``/``master`` (or a path) into a real file.

    Raises:
        VerifyError: When the file does not exist.
    """
    spec = (spec or "draft").strip()
    if spec == "draft":
        path = project.renders_dir / "draft.mp4"
    elif spec == "preview":
        path = project.renders_dir / "preview.mp4"
    elif spec == "master":
        path = project.exports_dir / f"master_{int(timeline.height)}p.mp4"
        if not path.exists():
            found = sorted(project.exports_dir.glob("master_*.mp4"))
            if found:
                path = found[-1]
    else:
        path = Path(spec)
        if not path.is_absolute():
            candidate = project.path / spec
            path = candidate if candidate.exists() else path
    if not path.exists():
        raise VerifyError(f"no render at {path} — render it first")
    return path


def check_dir(project: Project) -> Path:
    """``renders/check/``, created on demand."""
    path = project.renders_dir / "check"
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_audio(project: Project, render: Path) -> Path:
    """Pull mono 16 kHz PCM out of ``render`` into ``renders/check/<name>.wav``."""
    out = check_dir(project) / f"{render.stem}.wav"
    ff(
        "-i", str(render),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out),
    )
    return out


def _source_key(render: Path) -> dict[str, Any]:
    stat = render.stat()
    return {"mtime": round(stat.st_mtime, 3), "size": stat.st_size}


def render_transcript(
    project: Project, render: Path, force: bool = False
) -> tuple[dict[str, Any], bool]:
    """Transcribe the render's audio, caching against its mtime and size.

    Returns:
        ``(transcript, cached)`` — the ElevenLabs ``Transcript.to_dict()``
        payload and whether it came from the cache (i.e. cost nothing).
    """
    cache = check_dir(project) / f"{render.stem}.transcript.json"
    key = _source_key(render)
    if cache.exists() and not force:
        try:
            stored = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt cache
            stored = {}
        if stored.get("source") == key and isinstance(stored.get("transcript"), dict):
            return stored["transcript"], True

    # Imported here so the module stays importable (and unit-testable) without
    # an API key or the HTTP client.
    from .ai.elevenlabs import ElevenLabs

    wav = extract_audio(project, render)
    client = ElevenLabs(
        api_key=project.settings.require_key("elevenlabs"),
        cost_callback=cost_recorder(project, STAGE),
    )
    try:
        transcript = client.transcribe(
            wav,
            language_code=to_iso3(project.language),
            diarize=False,
            keyterms=keyterms(project),
        )
    finally:
        client.close()
    payload = transcript.to_dict()
    cache.write_text(
        json.dumps({"source": key, "transcript": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload, False


def keyterms(project: Project) -> list[str]:
    """Place names from ``project.yaml`` that bias recognition (max 1000)."""
    raw = project.config.get("keyterms") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(term) for term in raw if str(term).strip()][:1000]


def transcript_words(payload: Mapping[str, Any]) -> list[Word]:
    """Transcript payload -> sorted :class:`Word`\\ s, artefacts dropped."""
    words: list[Word] = []
    for raw in payload.get("words") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") not in (None, "word"):
            continue
        text = str(raw.get("t", raw.get("text", ""))).strip()
        start, end = raw.get("s", raw.get("start")), raw.get("e", raw.get("end"))
        if not text or start is None or end is None:
            continue
        try:
            s, e = float(start), float(end)
        except (TypeError, ValueError):  # pragma: no cover - malformed payload
            continue
        if e <= s or e - s > MAX_RENDER_WORD:
            continue
        words.append(Word(s, e, text))
    words.sort(key=lambda w: (w.s, w.e))
    return words


# ----------------------------------------------------------------------
# alignment
# ----------------------------------------------------------------------
def _leading_side(
    kind: str, item_id: str, clip: str, in_: float, out: float, words: Sequence[Word]
) -> Side:
    """The side whose audio *stops* at the boundary."""
    side = Side(kind=kind, id=item_id, clip=clip, in_=round(in_, 3), out=round(out, 3))
    spoken = [w for w in words if w.e > in_ + _EPS and w.s < out - _EPS]
    if not spoken:
        return side
    last = spoken[-1]
    side.word, side.word_s, side.word_e = last.text, last.s, last.e
    side.margin = round(out - last.e, 3)
    side.inside_word = last.s < out - INSIDE_WORD and last.e > out + INSIDE_WORD
    return side


def _trailing_side(
    kind: str, item_id: str, clip: str, in_: float, out: float, words: Sequence[Word]
) -> Side:
    """The side whose audio *starts* at the boundary."""
    side = Side(kind=kind, id=item_id, clip=clip, in_=round(in_, 3), out=round(out, 3))
    spoken = [w for w in words if w.e > in_ + _EPS and w.s < out - _EPS]
    if not spoken:
        return side
    first = spoken[0]
    side.word, side.word_s, side.word_e = first.text, first.s, first.e
    side.margin = round(first.s - in_, 3)
    side.inside_word = first.s < in_ - INSIDE_WORD and first.e > in_ + INSIDE_WORD
    return side


def _render_context(report: BoundaryReport, render_words: Sequence[Word]) -> None:
    """Fill in the render-side gap / straddling word for one boundary."""
    at = report.at
    before = [
        w for w in render_words if w.e <= at + EDGE_TOLERANCE and w.e > at - SEARCH_WINDOW
    ]
    after = [
        w for w in render_words if w.s >= at - EDGE_TOLERANCE and w.s < at + SEARCH_WINDOW
    ]
    if before and after:
        report.render_before = before[-1].text
        report.render_after = after[0].text
        report.render_gap = round(after[0].s - before[-1].e, 3)
    straddle = next(
        (w for w in render_words if w.s < at - STRADDLE and w.e > at + STRADDLE), None
    )
    if straddle is not None:
        report.straddle = {
            "text": straddle.text,
            "s": round(straddle.s, 3),
            "e": round(straddle.e, 3),
        }


def _flags(report: BoundaryReport) -> list[str]:
    """Everything worth the user's attention at this boundary."""
    flags: list[str] = []
    if report.straddle is not None:
        flags.append("chopped word")
    if report.render_gap is not None and report.render_gap < MIN_RENDER_GAP:
        flags.append("no air")
    if report.before is not None and report.before.inside_word:
        flags.append("out inside word")
    if report.after is not None and report.after.inside_word:
        flags.append("in inside word")
    if report.replayed > MIN_REPLAY:
        flags.append(f"replayed {report.replayed:.2f}s")
    return flags


def beat_labels(cut: "Cut", segments: Sequence[VideoSegment]) -> dict[str, str]:
    """``{segment uid: "b012 (shot 2)"}`` — where a boundary's fix belongs.

    A correction is applied to ``plan/cut.json``; the segment ids in the
    report are of the derived timeline and are renumbered on every resolve, so
    the beat (and which of its shots is on screen) is the only durable address
    a boundary can be given.

    Args:
        cut: The loaded ``plan/cut.json``.
        segments: Every video segment of the resolved timeline, in order.

    Returns:
        One label per segment that carries a beat uid still present in the
        cut; segments of a pre-cut timeline are simply absent.
    """
    from .inspect import shot_index_by_segment

    by_beat: dict[str, list[VideoSegment]] = {}
    for seg in segments:
        if seg.beat:
            by_beat.setdefault(seg.beat, []).append(seg)

    labels: dict[str, str] = {}
    for uid, mine in by_beat.items():
        beat = cut.beat_by_ref(uid)
        if beat is None:
            continue
        name = beat.id or beat.uid
        mapping = shot_index_by_segment(beat, mine)
        for seg in mine:
            index = mapping.get(seg.uid)
            labels[seg.uid] = (
                f"{name} (shot {index + 1})" if index is not None else f"{name} (own picture)"
            )
    return labels


def _label_around(
    positions: Sequence[Any], labels: Mapping[str, str], at: float
) -> tuple[str | None, str | None]:
    """The beat labels on either side of render time ``at``.

    Used for a voice pickup's start and end, which are placed in timeline time
    rather than at a segment boundary: the picture under a pickup can change
    mid-WAV, so the sides are read off the programme itself.
    """
    before = after = None
    for pos in positions:
        if pos.start < at - EDGE_TOLERANCE:
            before = labels.get(pos.segment.uid, before)
        if pos.end > at + EDGE_TOLERANCE:
            after = labels.get(pos.segment.uid)
            break
    return before, after


def is_continuous(a: VideoSegment, b: VideoSegment, fps: float) -> bool:
    """True when ``b``'s audio simply continues ``a``'s, so there is no cut.

    The same clip picked up where it left off — the pieces of one beat around
    a shot, which carry the narration underneath the insert's picture — is a
    hand-off, not a boundary, and is skipped. The tolerance is the render's
    own one frame.
    """
    if a.mute_source or b.mute_source:
        return False
    a_clip, _a_in, a_out = a.audio_source
    b_clip, b_in, _b_out = b.audio_source
    tolerance = 1.0 / fps if fps > 0 else 0.033
    return a_clip == b_clip and abs(b_in - a_out) <= tolerance


def analyze(
    timeline: Timeline,
    clip_words: Mapping[str, Sequence[Word]],
    render_words: Sequence[Word],
    cut: "Cut | None" = None,
) -> list[BoundaryReport]:
    """Line every audio boundary of ``timeline`` up against the render transcript.

    Args:
        timeline: The loaded ``plan/timeline.json``.
        clip_words: Source transcript words per clip id.
        render_words: Words transcribed from the rendered file (render time).
        cut: The ``plan/cut.json`` the timeline was resolved from, so every
            boundary can be labelled with the beats on both sides. ``None``
            leaves ``beat_before``/``beat_after`` unset.

    Returns:
        One :class:`BoundaryReport` per real boundary, sorted by render time.
        Boundaries where both sides are silent are left out entirely.
    """
    reports: list[BoundaryReport] = []
    positions = timeline.segment_positions(fade_overlaps=True)
    fps = float(timeline.fps)
    labels = beat_labels(cut, timeline.tracks.video) if cut is not None else {}

    for i in range(len(positions) - 1):
        a, b = positions[i].segment, positions[i + 1].segment
        if is_continuous(a, b, fps):
            continue
        a_clip, a_in, a_out = a.audio_source
        b_clip, b_in, b_out = b.audio_source
        before = (
            None
            if a.mute_source
            else _leading_side("segment", a.id, a_clip, a_in, a_out, clip_words.get(a_clip, ()))
        )
        after = (
            None
            if b.mute_source
            else _trailing_side("segment", b.id, b_clip, b_in, b_out, clip_words.get(b_clip, ()))
        )
        if before is None and after is None:
            continue  # muted picture on both sides: no audio boundary here
        replayed = 0.0
        if before is not None and after is not None and a_clip == b_clip:
            replayed = max(0.0, round(a_out - b_in, 3))
        report = BoundaryReport(
            at=positions[i].end,
            kind="cut",
            label=f"{a.id} -> {b.id}",
            before=before,
            after=after,
            beat_before=labels.get(a.uid),
            beat_after=labels.get(b.uid),
            replayed=replayed,
        )
        _render_context(report, render_words)
        report.flags = _flags(report)
        reports.append(report)

    to_render = build_time_map(timeline)
    for item in timeline.tracks.voice:
        end = item.end if item.end is not None else item.at
        for kind, at in (("voice-in", item.at), ("voice-out", end)):
            when = to_render(at)
            beat_before, beat_after = _label_around(positions, labels, when)
            report = BoundaryReport(
                at=when,
                kind=kind,
                label=f"{item.id} {'start' if kind == 'voice-in' else 'end'}",
                beat_before=beat_before,
                beat_after=beat_after,
            )
            _render_context(report, render_words)
            report.flags = _flags(report)
            reports.append(report)

    reports.sort(key=lambda r: (r.at, r.kind))
    return reports


def summarize(reports: Sequence[BoundaryReport]) -> dict[str, int]:
    """Count boundaries and how often each flag fired."""
    counts: dict[str, int] = {"boundaries": len(reports), "flagged": 0}
    for report in reports:
        if report.flags:
            counts["flagged"] += 1
        for flag in report.flags:
            key = flag.split(" ")[0] if flag.startswith("replayed") else flag
            counts[key] = counts.get(key, 0) + 1
    return counts


# ----------------------------------------------------------------------
# reports on disk
# ----------------------------------------------------------------------
def _cell(text: str) -> str:
    """One Markdown table cell — a pipe in a transcript word must not split it."""
    return (text or "-").replace("|", "\\|")


def report_markdown(
    project: Project, render: Path, reports: Sequence[BoundaryReport], counts: Mapping[str, int]
) -> str:
    """Render the report as Markdown (``renders/check/<name>.report.md``)."""
    lines = [
        f"# Render check — {project.slug}",
        "",
        f"Render: `{render.name}`  ",
        f"Boundaries: {counts['boundaries']} — flagged: {counts['flagged']}",
        "",
    ]
    extra = [f"- {k}: {v}" for k, v in counts.items() if k not in ("boundaries", "flagged")]
    if extra:
        lines += sorted(extra) + [""]
    lines += [
        "| at | boundary | beats | flags | source | render |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        source = []
        if report.before is not None and report.before.word:
            source.append(
                f"A {report.before.clip}@{report.before.out:.2f} "
                f"'{report.before.word}' {report.before.margin:+.2f}"
            )
        if report.after is not None and report.after.word:
            source.append(
                f"B {report.after.clip}@{report.after.in_:.2f} "
                f"'{report.after.word}' {report.after.margin:+.2f}"
            )
        flags = ", ".join(report.flags) or "-"
        phrase = report.render_phrase()
        if report.straddle is not None:
            phrase += f" STRADDLE {report.straddle['text']!r}"
        lines.append(
            f"| {timecode(report.at)} | {report.label} | {_cell(report.beats)} | {flags} | "
            f"{_cell('; '.join(source))} | {_cell(phrase)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_cut(project: Project) -> "Cut | None":
    """The project's ``plan/cut.json``, or ``None`` when it has none/is broken.

    The check reports, it never gates: a render is still worth checking when
    the cut cannot be read — it just loses the beat labels.
    """
    from .cut import cut_path, load_cut

    path = cut_path(project)
    if not path.exists():
        return None
    try:
        return load_cut(path)
    except Exception as exc:  # pragma: no cover - unreadable/invalid cut
        log.warning("could not read %s: %s", path, exc)
        return None


def check_render(
    project: Project,
    render: str = "draft",
    force: bool = False,
    write_json: bool = False,
) -> dict[str, Any]:
    """Transcribe a render and align it against the timeline.

    Args:
        project: The project to check.
        render: ``draft``/``preview``/``master`` or a path to a rendered file.
        force: Re-transcribe even when the cached transcript still matches.
        write_json: Also write ``renders/check/<name>.report.json``.

    Returns:
        ``{"render", "cached", "boundaries", "counts", "report_md", ...}``.

    Raises:
        VerifyError: When the render or the timeline is missing.
    """
    if not project.timeline_file.exists():
        raise VerifyError(f"no timeline at {project.timeline_file}")
    timeline = Timeline.load(project.timeline_file)
    path = resolve_render(project, timeline, render)

    payload, cached = render_transcript(project, path, force=force)
    render_words = transcript_words(payload)

    clips = {seg.audio_source[0] for seg in timeline.tracks.video}
    clip_words = {clip: load_words(project, clip) for clip in sorted(clips)}

    reports = analyze(timeline, clip_words, render_words, cut=_load_cut(project))
    counts = summarize(reports)

    md_path = check_dir(project) / f"{path.stem}.report.md"
    md_path.write_text(report_markdown(project, path, reports, counts), encoding="utf-8")

    result: dict[str, Any] = {
        "render": str(path),
        "cached": cached,
        "render_words": len(render_words),
        "counts": counts,
        "boundaries": [r.to_dict() for r in reports],
        "report_md": str(md_path),
    }
    if write_json:
        json_path = check_dir(project) / f"{path.stem}.report.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["report_json"] = str(json_path)
    log.info(
        "check-render %s: %d boundaries, %d flagged (%s transcript)",
        path.name,
        counts["boundaries"],
        counts["flagged"],
        "cached" if cached else "fresh",
    )
    return result


__all__ = [
    "BoundaryReport",
    "Side",
    "VerifyError",
    "analyze",
    "beat_labels",
    "check_render",
    "is_continuous",
    "render_transcript",
    "report_markdown",
    "resolve_render",
    "summarize",
    "timecode",
    "transcript_words",
]
