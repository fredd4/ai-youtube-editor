"""Cut v2 — the script-first edit model (``plan/cut.json``).

See ``docs/ARCHITECTURE.md`` ("plan/cut.json", "The resolver"). This module owns:

* the pydantic models of ``cut.json`` (:class:`Cut` and its beats),
* :func:`validate` — every structural/editorial rule, returned as issues,
* :func:`resolve` — the **only** place where speech becomes seconds: turns a
  :class:`Cut` into a render-ready :class:`ytedit.timeline.Timeline`.

Speech is addressed by sentence ids (``analysis/sentences.json``) or word
indices, never by seconds; picture-only ``shots`` never carry audio of their
own; every sentence/word range is used at most once. ``plan/timeline.json``
is a derived artifact written by :func:`resolve` and never edited.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ytedit.log import get_logger
from ytedit.words import Word, load_words
from ytedit.timeline import (
    AudioFrom,
    AudioWindow,
    CaptionPosition,
    Duck,
    MuteRange,
    Timeline,
    Transform,
    Transition,
    VideoSegment,
    VoiceItem,
)
from ytedit.timeline import Caption as TimelineCaption
from ytedit.timeline import Chapter as TimelineChapter
from ytedit.timeline import Marker as TimelineMarker
from ytedit.timeline import MusicCue as TimelineMusicCue

if TYPE_CHECKING:  # pragma: no cover
    from ytedit.project import Project

log = get_logger(__name__)

CUT_VERSION = 2

_EPS = 1e-6

BeatKind = Literal["speech", "broll", "voice"]
BeatAudio = Literal["ambient", "mute"]
Severity = Literal["error", "warning"]


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


def new_uid() -> str:
    """Stable beat identity: 8 hex chars, assigned once, never renumbered."""
    return secrets.token_hex(4)


class Shot(_Model):
    """A picture-only insert over a beat's audio (never carries its own sound)."""

    clip: str
    in_: float = Field(0.0, alias="in", serialization_alias="in")
    out: float = 0.0
    #: Sentence id (``c048#3``) of the owning speech beat — the shot starts at
    #: that sentence's end; an int is a word index when the beat uses
    #: ``words``; ``None`` = the start of the beat (voice beats, off-camera).
    after: str | int | None = None
    transform: Transform | None = None
    grade: str | None = None
    notes: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.out - self.in_)


class Beat(_Model):
    """One editorial unit of the cut. Which fields apply depends on ``kind``."""

    id: str = ""
    uid: str = Field(default_factory=new_uid)
    kind: BeatKind
    role: str = ""
    notes: str = ""
    transform: Transform = Field(default_factory=Transform)
    grade: str = "default"
    transition_in: Transition = Field(default_factory=Transition)

    # speech + broll
    clip: str | None = None
    # speech: exactly one of ``sentences`` / ``words``
    sentences: list[str] = Field(default_factory=list)
    #: ``[first_index, last_index]`` inclusive into ``transcripts/<clip>.json.words``.
    words: tuple[int, int] | None = None
    on_camera: bool = True
    # speech + voice
    shots: list[Shot] = Field(default_factory=list)
    gain_db: float = 0.0
    # broll
    in_: float | None = Field(None, alias="in", serialization_alias="in")
    out: float | None = None
    #: Source-audio routing for the beat's own picture. ``None`` means "not
    #: chosen" and takes the kind's default: a ``broll`` beat plays its own
    #: sound, a ``voice`` beat's shots are silent under the pickup. Only a
    #: deliberate choice is written to disk, so a file stays byte-stable.
    audio: BeatAudio | None = None
    # voice
    file: str | None = None

    @property
    def keeps_source_audio(self) -> bool:
        """True when this beat's picture is heard as well as seen.

        A ``speech`` beat is always heard — that is what it is for. A
        ``broll`` beat is heard unless it was muted. A ``voice`` beat's shots
        are silent unless the editor deliberately kept the ambience under the
        narration (the cold open, a street party a pickup talks over): a pickup
        recorded at home has no business fighting the location sound by
        accident, only on purpose.
        """
        if self.kind == "speech":
            return True
        default = "ambient" if self.kind == "broll" else "mute"
        return (self.audio or default) == "ambient"


class MusicCue(_Model):
    id: str
    file: str
    from_: str = Field(alias="from", serialization_alias="from")
    to: str
    gain_db: float = -18.0
    fade_in: float = 2.0
    fade_out: float = 3.0
    duck: Duck = Field(default_factory=Duck)


class Caption(_Model):
    id: str
    beat: str
    offset: float = 0.3
    duration: float = 3.0
    text: str = ""
    style: str = "location"
    position: CaptionPosition = "lower-left"


class Chapter(_Model):
    beat: str
    title: str = ""


class Marker(_Model):
    beat: str
    label: str = ""


class Meta(_Model):
    title_candidates: list[str] = Field(default_factory=list)
    generated_by: str = ""
    edited_by_human: bool = False
    notes: str = ""


class Cut(_Model):
    """``plan/cut.json`` — the source of truth for the edit."""

    version: int = CUT_VERSION
    fps: int = 30
    width: int = 1920
    height: int = 1080
    language: str = "pl"
    beats: list[Beat] = Field(default_factory=list)
    music: list[MusicCue] = Field(default_factory=list)
    captions: list[Caption] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    markers: list[Marker] = Field(default_factory=list)
    mute_ranges: list[MuteRange] = Field(default_factory=list)
    meta: Meta = Field(default_factory=Meta)

    # --- identity helpers -------------------------------------------------
    def beat_by_ref(self, ref: str) -> Beat | None:
        """Look a beat up by ``uid`` first, then by display ``id``."""
        if not ref:
            return None
        for beat in self.beats:
            if beat.uid == ref:
                return beat
        for beat in self.beats:
            if beat.id == ref:
                return beat
        return None

    def renumber(self) -> None:
        """Assign display ids ``b001``… in order (uids are never touched)."""
        for i, beat in enumerate(self.beats, 1):
            beat.id = f"b{i:03d}"


@dataclass
class Issue:
    """One validation finding."""

    severity: Severity
    code: str
    beat: str | None
    message: str

    def __str__(self) -> str:
        where = f" [{self.beat}]" if self.beat else ""
        return f"{self.severity}: {self.code}{where}: {self.message}"


def _analysis_span(item: dict[str, Any]) -> tuple[float, float]:
    """Read a ``(start, end)`` pair from an ``s``/``e`` (or ``in``/``out``) dict."""
    start = item.get("s", item.get("in", item.get("start", 0.0)))
    end = item.get("e", item.get("out", item.get("end", 0.0)))
    try:
        return float(start or 0.0), float(end or 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0


def instruction_ranges(entry: dict[str, Any]) -> list[tuple[float, float]]:
    """Time ranges of spoken editor instructions in one clip's analysis (clip time)."""
    ranges: list[tuple[float, float]] = []
    for item in entry.get("instructions") or []:
        if isinstance(item, dict):
            s, e = _analysis_span(item)
            if e > s:
                ranges.append((s, e))
    return ranges


def rejected_take_ranges(entry: dict[str, Any]) -> list[tuple[float, float]]:
    """Ranges of take attempts the analysis did **not** keep (clip time)."""
    ranges: list[tuple[float, float]] = []
    for take in entry.get("takes") or []:
        if not isinstance(take, dict):
            continue
        attempts = [a for a in (take.get("attempts") or []) if isinstance(a, dict)]
        if len(attempts) < 2:
            continue
        try:
            keep = int(take.get("keep", len(attempts) - 1))
        except (TypeError, ValueError):
            keep = len(attempts) - 1
        if keep < 0:
            keep += len(attempts)
        for i, attempt in enumerate(attempts):
            if i == keep:
                continue
            s, e = _analysis_span(attempt)
            if e > s:
                ranges.append((s, e))
    return ranges


class CutError(RuntimeError):
    """Raised by :func:`resolve` when the cut has validation **errors**.

    Attributes:
        issues: Every issue found, errors and warnings alike.
    """

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues = list(issues)
        errors = [i for i in self.issues if i.severity == "error"]
        head = f"{len(errors)} error(s) in cut.json"
        super().__init__("\n".join([head] + [f"  - {i}" for i in errors]))


# ----------------------------------------------------------------------
# io
# ----------------------------------------------------------------------
def cut_path(project: "Project") -> Path:
    """``plan/cut.json`` — the source of truth for the edit."""
    return project.plan_dir / "cut.json"


def _normalize_refs(cut: Cut) -> None:
    """Rewrite every beat reference to the beat's ``uid``.

    ``music.from``/``music.to``/``captions.beat``/``chapters.beat``/
    ``markers.beat`` may be written as the readable display id (``b003``) or
    as the uid; internally only the uid is ever used, since display ids are
    renumbered on every save. A reference that matches no beat is left alone
    so :func:`validate` can report it.
    """
    for cue in cut.music:
        beat = cut.beat_by_ref(cue.from_)
        if beat is not None:
            cue.from_ = beat.uid
        beat = cut.beat_by_ref(cue.to)
        if beat is not None:
            cue.to = beat.uid
    for item in [*cut.captions, *cut.chapters, *cut.markers]:
        beat = cut.beat_by_ref(item.beat)
        if beat is not None:
            item.beat = beat.uid


def load_cut(path: Path | str) -> Cut:
    """Read ``plan/cut.json``.

    Beat references (``music.from``/``to``, ``captions``/``chapters``/
    ``markers`` ``beat``) are re-resolved through the beats' ``uid`` on the
    way in, so a file that spells them as display ids stays correct even
    after the ids are renumbered.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cut = Cut.model_validate(data)
    for beat in cut.beats:
        if not beat.uid:
            beat.uid = new_uid()
    _normalize_refs(cut)
    return cut


def save_cut(cut: Cut, path: Path | str) -> Path:
    """Write ``cut`` atomically and return the path.

    Display ids are renumbered ``b001``… in list order first, and every beat
    reference is normalized to the referenced beat's ``uid`` — the only
    identity that survives a reorder. Keys keep the models' declaration
    order and ``None`` values are omitted, so two saves of the same cut
    produce byte-identical files.
    """
    for beat in cut.beats:
        if not beat.uid:
            beat.uid = new_uid()
    cut.renumber()
    _normalize_refs(cut)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        cut.model_dump(by_alias=True, mode="json", exclude_none=True),
        indent=2,
        ensure_ascii=False,
    )
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".cut-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return target


#: Backups of ``plan/cut.json`` kept in ``plan/history/`` (mirrors server/app.py).
MAX_HISTORY = 20


def backup_cut(project: "Project") -> str | None:
    """Copy ``plan/cut.json`` into ``plan/history/``, pruning old backups.

    Every stage that rewrites the cut in place (``voice``, ``captions``,
    ``music``) calls this first, so a bad automatic edit is always one file
    copy away from being undone.

    Args:
        project: The owning project.

    Returns:
        The backup's filename, or ``None`` when there was no cut to copy.
    """
    import shutil

    from ytedit.project import utcnow

    src = cut_path(project)
    if not src.exists():
        return None
    history = project.plan_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().replace(":", "").replace("-", "").replace("+0000", "Z")
    target = history / f"cut_{stamp}.json"
    n = 1
    while target.exists():
        target = history / f"cut_{stamp}_{n}.json"
        n += 1
    shutil.copy2(src, target)
    backups = sorted(history.glob("cut_*.json"), key=lambda p: p.stat().st_mtime)
    for old in backups[:-MAX_HISTORY]:
        old.unlink(missing_ok=True)
    return target.name


def probe_voice_duration(path: Path | str) -> float:
    """Length of a narration WAV in seconds, cached in ``<file>.probe.json``.

    The cache is keyed by the file's mtime, so a re-recorded pickup is
    re-probed and everything else is free. Tests monkeypatch this function
    rather than shipping real audio.

    Args:
        path: The WAV (or any media file ffprobe understands).

    Returns:
        The duration in seconds, or ``0.0`` when the file cannot be probed.
    """
    source = Path(path)
    try:
        mtime = source.stat().st_mtime_ns
    except OSError:
        return 0.0
    cache = source.with_name(source.name + ".probe.json")
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("mtime_ns") == mtime:
                return float(data["duration"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass  # fall through and re-probe
    from ytedit.media.audio import audio_duration  # local: keeps ffmpeg off the import path

    duration = float(audio_duration(source))
    try:
        cache.write_text(
            json.dumps({"mtime_ns": mtime, "duration": round(duration, 6)}), encoding="utf-8"
        )
    except OSError:  # pragma: no cover - best-effort cache
        pass
    return duration


# ----------------------------------------------------------------------
# resolver
# ----------------------------------------------------------------------
def _r(value: float) -> float:
    """Round to microseconds — the shared quantum for every produced time."""
    return round(float(value), 6)


def _segment_uid(beat_uid: str, index: int) -> str:
    """Deterministic segment identity, so re-resolving never busts the cache."""
    return hashlib.sha1(f"{beat_uid}#{index}".encode("utf-8")).hexdigest()[:8]


@dataclass
class _Piece:
    """One planned video segment of a beat, before it becomes a segment."""

    clip: str
    in_: float
    out: float
    #: Clip time of the *beat's* audio this piece plays under; ``None`` for a
    #: piece that carries no audio at all (voice beats, muted B-roll).
    audio_in: float | None = None
    audio_out: float | None = None
    #: The shot this piece came from (``None`` = the beat's own picture).
    shot: Shot | None = None
    borrows_audio: bool = False
    mute: bool = False


@dataclass
class _Speech:
    """A speech beat's resolved geometry, all in the beat clip's time base."""

    clip: str
    first_s: float
    last_e: float
    #: Picture range — the full ``speech_pad_before/after`` air, clip-clamped.
    seg_in: float
    seg_out: float
    #: Audible range — the same, pulled back off any neighbouring word.
    a0: float
    a1: float
    #: Legal ``shot.after`` references of this beat -> the clip time they cut at.
    ends: dict[Any, float]


class _Resolver:
    """One resolve pass: turns a :class:`Cut` into a v2 :class:`Timeline`.

    Validation and resolution are the same walk — a rule like
    ``voice_picture_short`` or ``air_short`` can only be checked by laying the
    beat out — so :func:`validate` runs this and keeps the issues, while
    :func:`resolve` runs it and keeps the timeline.
    """

    def __init__(self, project: "Project", cut: Cut) -> None:
        from ytedit.ai.sentences import load_sentence_index

        self.project = project
        self.cut = cut
        self.issues: list[Issue] = []

        settings = project.settings
        self.pad_before = float(settings.get("pacing.speech_pad_before", 0.30))
        self.pad_after = float(settings.get("pacing.speech_pad_after", 0.45))
        self.guard = float(settings.get("pacing.word_guard", 0.05))
        self.min_shot = float(settings.get("pacing.min_shot_seconds", 0.8))
        self.air_warn = float(settings.get("pacing.air_warn_below", 0.35))

        self.clips: dict[str, Any] = project.load_state().get("clips", {}) or {}
        self.sentences = load_sentence_index(project)
        self._words: dict[str, list[Word]] = {}
        self._protected: dict[str, list[tuple[float, float]]] = {}
        #: sentence id -> the beat that already used it
        self._claimed_sentences: dict[str, str] = {}
        #: clip -> [(first_word_index, last_word_index, beat label)]
        self._claimed_words: dict[str, list[tuple[int, int, str]]] = {}

    # -- issue helpers --------------------------------------------------
    @staticmethod
    def _label(beat: Beat | None) -> str | None:
        """How a beat is named in an issue: its display id, else its uid."""
        return None if beat is None else (beat.id or beat.uid)

    def error(self, code: str, beat: Beat | None, message: str) -> None:
        self.issues.append(Issue("error", code, self._label(beat), message))

    def warn(self, code: str, beat: Beat | None, message: str) -> None:
        self.issues.append(Issue("warning", code, self._label(beat), message))

    # -- project data ---------------------------------------------------
    def words(self, clip: str) -> list[Word]:
        """Transcript words of a clip, cached."""
        if clip not in self._words:
            self._words[clip] = load_words(self.project, clip)
        return self._words[clip]

    def clip_duration(self, clip: str) -> float | None:
        """Duration from the clip registry, ``None`` when unknown."""
        record = self.clips.get(clip)
        if not isinstance(record, dict):
            return None
        try:
            return float(record.get("duration"))
        except (TypeError, ValueError):
            return None

    def picture_limit(self, clip: str) -> float | None:
        """The last clip time a cut may end at, ``None`` when unknown.

        The registry duration is the container's; the last decodable frame
        ends one frame earlier (one measured clip: 15.806 s registered, 474 frames
        = 15.800 s), and a cut reaching past it renders one frame short and
        is refused by the segment check. Every picture range is clamped to
        this, not to the raw duration.
        """
        duration = self.clip_duration(clip)
        if duration is None:
            return None
        fps = float(self.cut.fps or 30)
        return max(0.0, _r(duration - 1.0 / fps))

    def protected_ranges(self, clip: str) -> list[tuple[float, float]]:
        """Instruction + rejected-take ranges of a clip, in clip time."""
        if clip not in self._protected:
            path = self.project.analysis_path(clip)
            entry: dict[str, Any] = {}
            if path.exists():
                try:
                    entry = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):  # pragma: no cover - bad analysis
                    log.warning("unreadable analysis %s", path)
                    entry = {}
            self._protected[clip] = sorted(
                instruction_ranges(entry) + rejected_take_ranges(entry)
            )
        return self._protected[clip]

    def known_clip(self, clip: str | None, beat: Beat, what: str = "clip") -> bool:
        """True when ``clip`` is in the registry; records an error when not."""
        if not clip:
            self.error("unknown_clip", beat, f"{what} is missing")
            return False
        if self.clips and clip not in self.clips:
            self.error("unknown_clip", beat, f"{what} {clip!r} is not in the clip registry")
            return False
        return True

    def check_range(self, beat: Beat, clip: str, in_: float, out: float, what: str) -> bool:
        """Check a raw seconds range against the clip's duration."""
        if out <= in_:
            self.error("bad_range", beat, f"{what}: in ({in_}) >= out ({out})")
            return False
        if in_ < -_EPS:
            self.error("bad_range", beat, f"{what}: negative in ({in_})")
            return False
        duration = self.clip_duration(clip)
        if duration is not None and out > duration + 0.05:
            self.error(
                "bad_range", beat,
                f"{what}: {in_:.2f}-{out:.2f}s exceeds clip {clip} duration {duration:.2f}s",
            )
            return False
        return True

    def clamp_out(self, clip: str, in_: float, out: float) -> float:
        """Pull a raw ``out`` back to the clip's last whole frame."""
        limit = self.picture_limit(clip)
        if limit is None or out <= limit + _EPS:
            return out
        return max(_r(limit), _r(in_ + 1.0 / float(self.cut.fps or 30)))

    # -- speech geometry ------------------------------------------------
    def speech_geometry(self, beat: Beat) -> _Speech | None:
        """Resolve a speech beat's sentence/word range into clip seconds."""
        clip = beat.clip or ""
        if not self.known_clip(clip, beat):
            return None
        has_sentences = bool(beat.sentences)
        has_words = beat.words is not None
        if has_sentences == has_words:
            self.error(
                "beat_range", beat,
                "a speech beat needs exactly one of 'sentences' or 'words'",
            )
            return None

        words = self.words(clip)
        ends: dict[Any, float] = {}
        if has_sentences:
            picked = self._pick_sentences(beat, clip)
            if picked is None:
                return None
            first_s, last_e = picked[0]["s"], picked[-1]["e"]
            ends = {str(sent["id"]): float(sent["e"]) for sent in picked}
            first_i, last_i = self._word_span_indices(words, first_s, last_e)
        else:
            assert beat.words is not None
            first_i, last_i = int(beat.words[0]), int(beat.words[1])
            if first_i > last_i:
                first_i, last_i = last_i, first_i
            if not words or first_i < 0 or last_i >= len(words):
                self.error(
                    "unknown_word_index", beat,
                    f"words [{first_i}, {last_i}] outside {clip}'s transcript "
                    f"({len(words)} word(s))",
                )
                return None
            first_s, last_e = float(words[first_i].s), float(words[last_i].e)
            ends = {i: float(words[i].e) for i in range(first_i, last_i + 1)}
            for s, e in self.protected_ranges(clip):
                if first_s < e - _EPS and last_e > s + _EPS:
                    self.error(
                        "words_in_instruction", beat,
                        f"words [{first_i}, {last_i}] overlap an instruction/rejected take "
                        f"at {s:.2f}-{e:.2f}s",
                    )
                    return None

        if first_i is not None and last_i is not None:
            if not self._claim_words(beat, clip, first_i, last_i):
                return None

        duration = self.picture_limit(clip)
        limit = duration if duration is not None else last_e + self.pad_after

        prev_e = max(
            (w.e for w in words if w.e <= first_s + _EPS), default=None
        )
        next_s = min(
            (w.s for w in words if w.s >= last_e - _EPS), default=None
        )

        # Picture: the full pad, clip-clamped.
        seg_in = max(0.0, first_s - self.pad_before)
        seg_out = min(limit, last_e + self.pad_after)
        # Audio: the same, but never reaching a neighbouring word or a range
        # the analysis marked as an instruction / a rejected take.
        a0, a1 = seg_in, seg_out
        if prev_e is not None:
            a0 = max(a0, prev_e + self.guard)
        if next_s is not None:
            a1 = min(a1, next_s - self.guard)
        for s, e in self.protected_ranges(clip):
            if e <= first_s + _EPS and e > a0:
                a0 = min(e, first_s)
            if s >= last_e - _EPS and s < a1:
                a1 = max(s, last_e)
        a0 = min(max(a0, 0.0), first_s)
        a1 = max(min(a1, limit), last_e)

        return _Speech(
            clip=clip,
            first_s=_r(first_s), last_e=_r(last_e),
            seg_in=_r(seg_in), seg_out=_r(seg_out),
            a0=_r(a0), a1=_r(a1),
            ends=ends,
        )

    def _pick_sentences(self, beat: Beat, clip: str) -> list[dict[str, Any]] | None:
        """Validate a speech beat's sentence ids and return them in order."""
        picked: list[dict[str, Any]] = []
        for sid in beat.sentences:
            sent = self.sentences.get(sid)
            if sent is None:
                self.error("unknown_sentence", beat, f"no sentence {sid!r} in the catalogue")
                return None
            if str(sent.get("clip")) != clip:
                self.error(
                    "unknown_sentence", beat,
                    f"sentence {sid!r} belongs to clip {sent.get('clip')!r}, not {clip!r}",
                )
                return None
            if sent.get("instruction"):
                self.error(
                    "instruction_sentence", beat,
                    f"sentence {sid!r} is a spoken editor instruction and must never be used",
                )
                return None
            # A retake / duplicate flag is advisory: the catalogue's detection
            # is heuristic (a short "Zobaczcie." or a song chorus trips it), so
            # using such a sentence is the editor's call and only warns. A
            # spoken instruction above is the one hard rule.
            if sent.get("retake_of"):
                self.warn(
                    "retake_sentence", beat,
                    f"sentence {sid!r} is flagged as a rejected take of {sent['retake_of']!r}",
                )
            elif sent.get("duplicate_of"):
                self.warn(
                    "duplicate_sentence", beat,
                    f"sentence {sid!r} is flagged as a near-duplicate of the later "
                    f"{sent['duplicate_of']!r}",
                )
            owner = self._claimed_sentences.get(sid)
            if owner is not None:
                self.error(
                    "sentence_reused", beat,
                    f"sentence {sid!r} is already used by beat {owner}",
                )
                return None
            picked.append(sent)

        for prev, cur in zip(picked, picked[1:]):
            if int(cur["n"]) != int(prev["n"]) + 1:
                self.error(
                    "sentences_not_contiguous", beat,
                    f"{prev['id']} -> {cur['id']} skips a sentence; split this into two beats",
                )
                return None

        for sent in picked:
            self._claimed_sentences[str(sent["id"])] = self._label(beat) or ""
        return picked

    @staticmethod
    def _word_span_indices(
        words: Sequence[Word], start: float, end: float
    ) -> tuple[int | None, int | None]:
        """Indices of the first/last transcript word inside ``[start, end]``."""
        inside = [i for i, w in enumerate(words) if w.e > start + _EPS and w.s < end - _EPS]
        if not inside:
            return None, None
        return inside[0], inside[-1]

    def _claim_words(self, beat: Beat, clip: str, first_i: int, last_i: int) -> bool:
        """Record a beat's word-index range, refusing an overlap with another."""
        claims = self._claimed_words.setdefault(clip, [])
        for start, end, owner in claims:
            if first_i <= end and last_i >= start:
                self.error(
                    "words_reused", beat,
                    f"words [{first_i}, {last_i}] of {clip} overlap beat {owner}'s "
                    f"[{start}, {end}] — no audio may play twice",
                )
                return False
        claims.append((first_i, last_i, self._label(beat) or ""))
        return True

    # -- beats ----------------------------------------------------------
    def speech_pieces(self, beat: Beat, sp: _Speech) -> list[_Piece]:
        """Lay a speech beat out as own-picture pieces and shot inserts."""
        groups = self._shot_groups(beat, sp)
        pieces: list[_Piece] = []
        cursor = sp.seg_in

        for point, shots in groups:
            point = max(point, cursor)
            if point > cursor + _EPS:
                pieces.append(
                    _Piece(sp.clip, cursor, _r(point), audio_in=cursor, audio_out=_r(point))
                )
                cursor = _r(point)
            for shot in shots:
                if not self.known_clip(shot.clip, beat, "shot clip"):
                    continue
                if not self.check_range(
                    beat, shot.clip, shot.in_, shot.out, f"shot on {shot.clip}"
                ):
                    continue
                available = sp.seg_out - cursor
                if available <= _EPS:
                    self.warn(
                        "shot_trimmed", beat,
                        f"shot on {shot.clip} starts after the beat's audio ends; dropped",
                    )
                    continue
                length = min(self.clamp_out(shot.clip, shot.in_, shot.out) - shot.in_, available)
                if length < shot.duration - _EPS:
                    self.warn(
                        "shot_trimmed", beat,
                        f"shot on {shot.clip} trimmed from {shot.duration:.2f}s to "
                        f"{length:.2f}s to end with the beat",
                    )
                end = _r(cursor + length)
                pieces.append(
                    _Piece(
                        shot.clip, _r(shot.in_), _r(shot.in_ + length),
                        audio_in=cursor, audio_out=end, shot=shot, borrows_audio=True,
                    )
                )
                cursor = end

        self._close_tail(beat, sp, pieces, cursor)
        return pieces

    def _shot_groups(self, beat: Beat, sp: _Speech) -> list[tuple[float, list[Shot]]]:
        """Group a beat's shots by the clip time they cut in at, in time order."""
        grouped: dict[float, list[Shot]] = {}
        order: list[float] = []
        for shot in beat.shots:
            point = sp.seg_in
            after = shot.after
            if after is not None and beat.words is not None and isinstance(after, str):
                try:  # a word index written as a string
                    after = int(after)
                except ValueError:
                    pass
            if after is not None and beat.on_camera:
                if after not in sp.ends:
                    self.error(
                        "unknown_shot_after", beat,
                        f"shot on {shot.clip}: after={after!r} is not part of this beat",
                    )
                    continue
                point = sp.ends[after]
            elif after is not None and not beat.on_camera:
                # Off camera the narrator is never on screen, so shots simply
                # chain from the beat start; an ``after`` is meaningless here.
                self.warn(
                    "shot_after_ignored", beat,
                    f"shot on {shot.clip}: 'after' is ignored on an off-camera beat "
                    "(shots chain from the beat start)",
                )
            if point not in grouped:
                grouped[point] = []
                order.append(point)
            grouped[point].append(shot)
        return [(point, grouped[point]) for point in sorted(order)]

    def _close_tail(
        self, beat: Beat, sp: _Speech, pieces: list[_Piece], cursor: float
    ) -> None:
        """Fill (or absorb) whatever of the beat no shot covered."""
        tail = sp.seg_out - cursor
        if tail <= _EPS:
            return
        last = pieces[-1] if pieces else None
        stub = tail < self.min_shot - _EPS
        if stub and last is not None and last.shot is not None:
            # Rather than flash back to the narrator for a fraction of a
            # second, hold the last shot to the end of the beat.
            room = self.picture_limit(last.clip)
            wanted = _r(last.out + tail)
            if room is None or wanted <= room + _EPS:
                last.out = wanted
                last.audio_out = sp.seg_out
                return
            self.warn(
                "short_piece", beat,
                f"only {tail:.2f}s of picture is left after the last shot and {last.clip} "
                f"has no footage left to cover it (min_shot_seconds={self.min_shot})",
            )
            stub = False    # already reported, with the reason
        pieces.append(
            _Piece(sp.clip, cursor, sp.seg_out, audio_in=cursor, audio_out=sp.seg_out)
        )
        if not beat.on_camera:
            self.warn(
                "narrator_visible", beat,
                f"{tail:.2f}s of this off-camera beat is not covered by a shot — the "
                "narrator is on screen; add a shot",
            )
        elif stub:
            self.warn(
                "short_piece", beat,
                f"the resumed own-picture piece is only {tail:.2f}s "
                f"(min_shot_seconds={self.min_shot})",
            )

    def voice_pieces(self, beat: Beat, length: float) -> list[_Piece] | None:
        """Lay a voice beat's picture out under a WAV of ``length`` seconds."""
        pieces: list[_Piece] = []
        cursor = 0.0
        for shot in beat.shots:
            if not self.known_clip(shot.clip, beat, "shot clip"):
                continue
            if not self.check_range(beat, shot.clip, shot.in_, shot.out, f"shot on {shot.clip}"):
                continue
            available = length - cursor
            if available <= _EPS:
                self.warn(
                    "shot_trimmed", beat,
                    f"shot on {shot.clip} starts after the pickup ends; dropped",
                )
                continue
            take = min(shot.duration, available)
            if take < shot.duration - _EPS:
                self.warn(
                    "shot_trimmed", beat,
                    f"shot on {shot.clip} trimmed from {shot.duration:.2f}s to {take:.2f}s "
                    "to end with the pickup",
                )
            pieces.append(
                _Piece(
                    shot.clip, _r(shot.in_), _r(shot.in_ + take), shot=shot,
                    mute=not beat.keeps_source_audio,
                )
            )
            cursor = _r(cursor + take)

        missing = length - cursor
        if missing > _EPS and pieces:
            last = pieces[-1]
            room = self.picture_limit(last.clip)
            wanted = _r(last.out + missing)
            if room is None or wanted <= room + _EPS:
                last.out = wanted
                cursor = _r(length)
                missing = 0.0
        if missing > _EPS:
            self.error(
                "voice_picture_short", beat,
                f"the pickup is {length:.2f}s but its shots only cover {cursor:.2f}s — "
                "add a shot",
            )
            return None
        return pieces

    def broll_pieces(self, beat: Beat) -> list[_Piece] | None:
        """One picture piece with (or without) its own ambient sound."""
        clip = beat.clip or ""
        if not self.known_clip(clip, beat):
            return None
        in_ = float(beat.in_ or 0.0)
        out = float(beat.out or 0.0)
        if not self.check_range(beat, clip, in_, out, "broll range"):
            return None
        out = self.clamp_out(clip, in_, out)
        mute = not beat.keeps_source_audio
        if not mute:
            spoken = [w for w in self.words(clip) if w.e > in_ + _EPS and w.s < out - _EPS]
            if spoken:
                text = " ".join(w.text for w in spoken[:6])
                self.warn(
                    "speech_in_broll", beat,
                    f"{len(spoken)} transcript word(s) inside this ambient B-roll "
                    f"({text!r}…) — make it a speech beat or mute it",
                )
        return [_Piece(clip, _r(in_), _r(out), mute=mute)]

    # -- pieces -> segments ---------------------------------------------
    def segments_for(
        self, beat: Beat, pieces: Sequence[_Piece], window: tuple[float, float] | None
    ) -> list[VideoSegment]:
        """Turn a beat's planned pieces into video segments."""
        segments: list[VideoSegment] = []
        for i, piece in enumerate(pieces):
            shot = piece.shot
            audio_from = None
            if piece.borrows_audio and piece.audio_in is not None:
                audio_from = AudioFrom(
                    clip=beat.clip or piece.clip,
                    **{"in": piece.audio_in},
                    out=piece.audio_out or piece.audio_in,
                )
            audio_window = None
            mute = piece.mute
            if window is not None and piece.audio_in is not None and not mute:
                src_in = piece.audio_in if audio_from is not None else piece.in_
                src_out = piece.audio_out if audio_from is not None else piece.out
                w0 = max(window[0], src_in)
                w1 = min(window[1], src_out)
                if w1 <= w0 + _EPS:
                    mute = True
                elif w0 > src_in + _EPS or w1 < src_out - _EPS:
                    audio_window = AudioWindow(**{"in": _r(w0)}, out=_r(w1))
            segments.append(
                VideoSegment(
                    id="",
                    uid=_segment_uid(beat.uid, i),
                    clip=piece.clip,
                    **{"in": piece.in_},
                    out=piece.out,
                    role=beat.role,
                    transform=(
                        shot.transform if shot is not None and shot.transform else beat.transform
                    ).model_copy(deep=True),
                    grade=(shot.grade if shot is not None and shot.grade else beat.grade),
                    transition_in=(
                        beat.transition_in.model_copy(deep=True) if i == 0 else Transition()
                    ),
                    mute_source=mute,
                    source_audio_gain_db=float(beat.gain_db),
                    notes=(shot.notes if shot is not None else beat.notes),
                    audio_from=audio_from,
                    audio_window=audio_window,
                    beat=beat.uid,
                )
            )
        return segments

    # -- the walk -------------------------------------------------------
    def build(self) -> Timeline:
        """Resolve the whole cut; issues land in :attr:`issues`."""
        self._check_identity()
        timeline = Timeline(
            version=CUT_VERSION,
            fps=self.cut.fps,
            width=self.cut.width,
            height=self.cut.height,
            language=self.cut.language,
        )
        voice_lengths: dict[str, float] = {}
        air: dict[str, tuple[float, float]] = {}
        self._joined: set[str] = set()
        # The previous beat when it was speech: (beat, geometry, its segments).
        prev_take: tuple[Beat, _Speech, list[VideoSegment]] | None = None
        for beat in self.cut.beats:
            pieces: list[_Piece] | None
            window: tuple[float, float] | None = None
            segments: list[VideoSegment] = []
            if beat.kind == "speech":
                geometry = self.speech_geometry(beat)
                if geometry is None:
                    prev_take = None
                    continue
                if prev_take is not None and self._join_takes(prev_take, beat, geometry, air):
                    self._joined.add(beat.uid)
                window = (geometry.a0, geometry.a1)
                air[beat.uid] = (
                    _r(geometry.first_s - geometry.seg_in),
                    _r(geometry.seg_out - geometry.last_e),
                )
                pieces = self.speech_pieces(beat, geometry)
            elif beat.kind == "broll":
                pieces = self.broll_pieces(beat)
            elif beat.kind == "voice":
                length = self._voice_length(beat)
                if length is None:
                    continue
                voice_lengths[beat.uid] = length
                pieces = self.voice_pieces(beat, length)
            else:  # pragma: no cover - the Literal keeps this unreachable
                self.error("unknown_kind", beat, f"unknown beat kind {beat.kind!r}")
                continue
            if not pieces:
                prev_take = None
                continue
            segments = self.segments_for(beat, pieces, window)
            timeline.tracks.video.extend(segments)
            prev_take = (beat, geometry, segments) if beat.kind == "speech" else None

        timeline.renumber_segment_ids()
        self._check_air(air)
        self._place_absolute_tracks(timeline, voice_lengths)
        timeline.mute_ranges = [m.model_copy(deep=True) for m in self.cut.mute_ranges]
        timeline.meta.title_candidates = list(self.cut.meta.title_candidates)
        timeline.meta.generated_by = self.cut.meta.generated_by
        timeline.meta.edited_by_human = self.cut.meta.edited_by_human
        timeline.meta.notes = self.cut.meta.notes
        timeline.meta.source = _source_stamp(self.project)
        return timeline

    def _join_takes(
        self,
        prev_take: tuple[Beat, _Speech, list[VideoSegment]],
        beat: Beat,
        geometry: _Speech,
        air: dict[str, tuple[float, float]],
    ) -> bool:
        """Make two consecutive speech beats of one take continuous.

        When the next beat continues the same clip so closely that the two
        pads would overlap (a take split only so a shot could sit between its
        sentences), there is no cut to air: the previous beat's picture would
        jump back by up to ``pad_before + pad_after`` and the same room tone
        would play twice. Instead both meet at the midpoint of the gap between
        the last word heard and the first word to come — exactly the
        hand-off an overlay cutaway makes — and no silence is windowed in.

        Returns:
            True when the beats were joined.
        """
        prev_beat, prev, segments = prev_take
        if prev.clip != geometry.clip or not segments:
            return False
        gap = geometry.first_s - prev.last_e
        if gap < -_EPS or gap >= self.pad_before + self.pad_after - _EPS:
            return False
        boundary = _r((prev.last_e + geometry.first_s) / 2.0)
        frame = 1.0 / float(self.cut.fps or 30)
        last = segments[-1]
        if last.audio_from is None:
            if last.clip != prev.clip or boundary - last.in_ < frame - _EPS:
                return False
            last.out = boundary
        else:
            delta = _r(last.audio_from.out - boundary)
            if last.audio_from.clip != prev.clip or last.out - delta - last.in_ < frame - _EPS:
                return False
            last.audio_from.out = boundary
            last.out = _r(last.out - delta)
        if last.audio_window is not None:
            last.audio_window.out = boundary
            if last.audio_window.out - last.audio_window.in_ <= _EPS:
                last.audio_window = None
        geometry.seg_in = boundary
        geometry.a0 = boundary
        air[prev_beat.uid] = (air[prev_beat.uid][0], _r(boundary - prev.last_e))
        log.info(
            "%s continues %s on %s: joined at %.2fs (%.2fs gap between words)",
            beat.id or beat.uid, prev_beat.id or prev_beat.uid, prev.clip, boundary, gap,
        )
        return True

    def _check_identity(self) -> None:
        """Duplicate beat uids would make every reference ambiguous."""
        seen: set[str] = set()
        for beat in self.cut.beats:
            if beat.uid in seen:
                self.error("duplicate_uid", beat, f"beat uid {beat.uid!r} is used twice")
            seen.add(beat.uid)

    def _voice_length(self, beat: Beat) -> float | None:
        """Probe a voice beat's WAV, reporting a missing/unreadable file."""
        if not beat.file:
            self.error("voice_file_missing", beat, "a voice beat needs a 'file'")
            return None
        path = self.project.path / beat.file
        if not path.exists():
            self.error("voice_file_missing", beat, f"missing narration file {beat.file}")
            return None
        length = float(probe_voice_duration(path))
        if length <= 0:
            self.error(
                "voice_file_missing", beat, f"{beat.file} has no readable duration"
            )
            return None
        return _r(length)

    def _check_air(self, air: dict[str, tuple[float, float]]) -> None:
        """Warn where two speech beats meet with too little silence between."""
        speech = [b for b in self.cut.beats if b.uid in air]
        index = {b.uid: i for i, b in enumerate(self.cut.beats)}
        for prev, cur in zip(speech, speech[1:]):
            if index[cur.uid] != index[prev.uid] + 1:
                continue  # something else sits between them; that is the air
            if cur.uid in getattr(self, "_joined", set()):
                continue  # one take continuing across two beats: no cut there
            total = air[prev.uid][1] + air[cur.uid][0]
            if total < self.air_warn - _EPS:
                self.warn(
                    "air_short", cur,
                    f"only {total:.2f}s of air at the cut from {prev.id or prev.uid} "
                    f"(want {self.air_warn:.2f}s) — the clip boundary is in the way",
                )

    # -- absolute tracks ------------------------------------------------
    def _place_absolute_tracks(
        self, timeline: Timeline, voice_lengths: dict[str, float]
    ) -> None:
        """Convert every beat-relative placement into absolute seconds."""
        spans: dict[str, tuple[float, float]] = {}
        for pos in timeline.segment_positions():
            uid = pos.segment.beat
            if not uid:
                continue
            start, _end = spans.get(uid, (pos.start, pos.end))
            spans[uid] = (start, pos.end)

        for beat in self.cut.beats:
            if beat.kind != "voice" or beat.uid not in spans:
                continue
            start = spans[beat.uid][0]
            length = voice_lengths.get(beat.uid, 0.0)
            timeline.tracks.voice.append(
                VoiceItem(
                    id=f"v{len(timeline.tracks.voice) + 1:03d}",
                    file=str(beat.file),
                    at=_r(start),
                    end=_r(start + length),
                    gain_db=float(beat.gain_db),
                )
            )

        for cue in self.cut.music:
            start_beat = self.cut.beat_by_ref(cue.from_)
            end_beat = self.cut.beat_by_ref(cue.to)
            if start_beat is None and end_beat is None:
                self.issues.append(Issue(
                    "error", "music_beat_missing", None,
                    f"music {cue.id}: neither 'from' ({cue.from_}) nor 'to' ({cue.to}) "
                    "is a beat of this cut",
                ))
                continue
            if start_beat is None or end_beat is None:
                surviving = self.cut.beats[0] if start_beat is None else self.cut.beats[-1]
                self.issues.append(Issue(
                    "warning", "music_beat_snapped", None,
                    f"music {cue.id}: "
                    f"{'from' if start_beat is None else 'to'} beat is gone — snapped to "
                    f"{surviving.id or surviving.uid}",
                ))
                start_beat = start_beat or surviving
                end_beat = end_beat or surviving
            if start_beat.uid not in spans or end_beat.uid not in spans:
                self.issues.append(Issue(
                    "warning", "music_beat_snapped", None,
                    f"music {cue.id}: its beats produced no picture; cue dropped",
                ))
                continue
            at = spans[start_beat.uid][0]
            end = spans[end_beat.uid][1]
            timeline.tracks.music.append(
                TimelineMusicCue(
                    id=cue.id, file=cue.file,
                    at=_r(min(at, end)), end=_r(max(at, end)),
                    gain_db=cue.gain_db, fade_in=cue.fade_in, fade_out=cue.fade_out,
                    duck=cue.duck.model_copy(deep=True),
                )
            )

        # Music beds never overlap: when two cues meet inside one beat (a v1
        # cue that changed mid-beat, or a hand edit), the later cue's start
        # wins and the earlier one ends there. Inclusive ``to`` ranges make
        # this the common case, so it is a note, not a warning.
        cues = sorted(timeline.tracks.music, key=lambda c: (c.at, c.end))
        for earlier, later in zip(cues, cues[1:]):
            if earlier.end > later.at + _EPS:
                if later.at <= earlier.at + _EPS:
                    self.issues.append(Issue(
                        "warning", "music_overlap", None,
                        f"music {later.id} starts at the same beat as {earlier.id}; "
                        f"{earlier.id} is dropped",
                    ))
                    earlier.end = earlier.at
                    continue
                log.info(
                    "music %s ends at %.2fs where %s starts (was %.2fs)",
                    earlier.id, later.at, later.id, earlier.end,
                )
                earlier.end = _r(later.at)
        timeline.tracks.music = [c for c in cues if c.end > c.at + _EPS]

        for i, caption in enumerate(self.cut.captions, 1):
            span = self._span_for(spans, caption.beat, "caption", caption.id or f"t{i:03d}")
            if span is None:
                continue
            at = _r(span[0] + caption.offset)
            end = _r(min(at + caption.duration, span[1] + 0.5))
            timeline.tracks.captions.append(
                TimelineCaption(
                    id=caption.id or f"t{i:03d}", at=at, end=max(end, _r(at + 0.1)),
                    text=caption.text, style=caption.style, position=caption.position,
                )
            )

        for chapter in self.cut.chapters:
            span = self._span_for(spans, chapter.beat, "chapter", chapter.title)
            if span is None:
                continue
            timeline.chapters.append(TimelineChapter(at=_r(span[0]), title=chapter.title))
        timeline.chapters.sort(key=lambda c: c.at)

        for marker in self.cut.markers:
            span = self._span_for(spans, marker.beat, "marker", marker.label)
            if span is None:
                continue
            timeline.markers.append(TimelineMarker(at=_r(span[0]), label=marker.label))
        timeline.markers.sort(key=lambda m: m.at)

    def _span_for(
        self, spans: dict[str, tuple[float, float]], ref: str, kind: str, label: str
    ) -> tuple[float, float] | None:
        """Look a beat span up for a caption/chapter/marker, warning when gone."""
        beat = self.cut.beat_by_ref(ref)
        if beat is None or beat.uid not in spans:
            self.issues.append(Issue(
                "warning", f"{kind}_dropped", None,
                f"{kind} {label!r}: beat {ref!r} is not in the cut (or produced no "
                "picture); dropped",
            ))
            return None
        return spans[beat.uid]


def _source_stamp(project: "Project") -> str:
    """``cut.json@<mtime iso>`` for :attr:`ytedit.timeline.Meta.source`."""
    path = cut_path(project)
    try:
        when = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        when = datetime.now(timezone.utc)
    return "cut.json@" + when.replace(microsecond=0).isoformat()


# ----------------------------------------------------------------------
# public api
# ----------------------------------------------------------------------
def validate(project: "Project", cut: Cut) -> list[Issue]:
    """Check a cut against the project: every rule of the cut model.

    Validation *is* a resolve pass — a rule such as ``voice_picture_short``
    or ``air_short`` only exists once the beat has been laid out — so this
    runs the resolver and keeps its findings instead of its timeline.

    Args:
        project: The owning project (clip registry, transcripts, analyses,
            the sentence catalogue and the narration files).
        cut: The cut to check.

    Returns:
        Every :class:`Issue`, errors and warnings, in the order they were
        found. An empty list means the cut resolves cleanly.
    """
    resolver = _Resolver(project, cut)
    resolver.build()
    return resolver.issues


def resolve_verbose(project: "Project", cut: Cut) -> tuple[Timeline, list[Issue]]:
    """Resolve a cut and hand back its issues instead of raising on an error.

    The producers need both halves of one pass: the geometry (to place a
    caption or size a music bed) *and* the findings (to report them). Nothing
    renders from this — :func:`resolve` is still the gate a render goes
    through.

    Args:
        project: The owning project.
        cut: The cut to resolve.

    Returns:
        ``(timeline, issues)``. The timeline is whatever the resolver could
        lay out; beats that errored contribute no segments to it.
    """
    resolver = _Resolver(project, cut)
    timeline = resolver.build()
    return timeline, resolver.issues


def beat_spans(
    project: "Project", cut: Cut
) -> tuple[dict[str, tuple[float, float]], list[Issue]]:
    """Absolute ``(start, end)`` of every beat that produced picture.

    Keyed by beat ``uid``. This is how a producer holding a *time* — a
    planner caption at 41.2 s, a music cue ending at 3:20 — finds the beat to
    attach it to, and how the music stage sizes a bed to its beat range.

    Args:
        project: The owning project.
        cut: The cut to lay out.

    Returns:
        ``(spans, issues)``; a beat the resolver could not lay out is absent
        from ``spans``.
    """
    timeline, issues = resolve_verbose(project, cut)
    spans: dict[str, tuple[float, float]] = {}
    for pos in timeline.segment_positions():
        uid = pos.segment.beat
        if not uid:
            continue
        start, _end = spans.get(uid, (pos.start, pos.end))
        spans[uid] = (start, pos.end)
    return spans, issues


def beat_at(
    spans: dict[str, tuple[float, float]], cut: Cut, t: float
) -> Beat | None:
    """The beat playing at absolute time ``t`` (the last one starting at or before).

    Args:
        spans: The mapping :func:`beat_spans` returned.
        cut: The cut those spans came from.
        t: Absolute timeline seconds.

    Returns:
        The beat under ``t``, or ``None`` when the cut produced no picture.
    """
    best: Beat | None = None
    for beat in cut.beats:
        span = spans.get(beat.uid)
        if span is None:
            continue
        if span[0] <= t + _EPS:
            best = beat
        if span[0] > t + _EPS:
            break
    if best is None:
        for beat in cut.beats:
            if beat.uid in spans:
                return beat
    return best


def resolve(project: "Project", cut: Cut) -> Timeline:
    """Turn a cut into the render-ready v2 :class:`Timeline`.

    This is the **only** place in the system where speech becomes seconds.
    Deterministic and side-effect free apart from the cached WAV probes: the
    same cut and the same project always produce the same timeline, down to
    the segment uids.

    Args:
        project: The owning project.
        cut: The cut to resolve.

    Returns:
        A :class:`ytedit.timeline.Timeline` with ``version = 2``, ready to be
        written to ``plan/timeline.json``.

    Raises:
        CutError: When :func:`validate` reports any ``error`` — a broken cut
            is never half-rendered.
    """
    resolver = _Resolver(project, cut)
    timeline = resolver.build()
    if any(issue.severity == "error" for issue in resolver.issues):
        raise CutError(resolver.issues)
    for issue in resolver.issues:
        log.warning("%s", issue)
    return timeline


def ensure_resolved(project: "Project") -> Path:
    """Re-resolve ``plan/timeline.json`` when it is older than ``plan/cut.json``.

    Called at the start of every command that reads the timeline (``render``,
    ``qc``, ``at``, ``check-render``) so a hand-edited or freshly generated
    cut is never rendered from a stale derived file. A project that has no
    ``cut.json`` yet (still v1) is left completely alone.

    Args:
        project: The owning project.

    Returns:
        The path of ``plan/timeline.json``.

    Raises:
        CutError: When the cut does not validate.
    """
    source = cut_path(project)
    target = project.timeline_file
    if not source.exists():
        return target
    try:
        stale = target.stat().st_mtime_ns < source.stat().st_mtime_ns
    except OSError:
        stale = True
    if not stale:
        return target
    log.info("cut.json is newer than timeline.json — resolving")
    timeline = resolve(project, load_cut(source))
    timeline.save(target)
    return target
