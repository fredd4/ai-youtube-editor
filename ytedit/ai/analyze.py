"""Stage 3: transcript + annotated frames -> ``analysis/<clip>.json`` + footage log.

One LLM call per clip (prompts from ``docs/playbook/prompts.md``): the compacted
word-level transcript, the STT audio events, the deterministic take hints from
:func:`ytedit.ai.transcribe.detect_takes`, and up to
:data:`MAX_FRAMES` evenly spaced annotated frames.  The answer is validated with
:class:`ClipAnalysis` (lenient: unknown keys allowed, every list optional) and
retried once with the validation error appended.

Afterwards every per-clip analysis is merged into ``analysis/footage_log.json``
(chronological, with a header summarising instructions, music flags, language
mismatches and locations) and a human-readable ``analysis/footage_log.md``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..costs import check_budget, estimate
from ..log import get_logger
from ..media.frames import annotate_frame
from ..project import Project, utcnow
from . import prompts
from .openrouter import OpenRouter
from .transcribe import cost_recorder, detect_takes, load_transcript

log = get_logger(__name__)

#: Concurrent analyst calls.
WORKERS = 3

#: Frames handed to the vision model per clip.
MAX_FRAMES = 8

#: Soft cap for the transcript embedded in the prompt (~4 chars per token).
TRANSCRIPT_TOKEN_BUDGET = 6000
_CHARS_PER_TOKEN = 4

#: Output-token allowance for one analysis answer.
MAX_OUTPUT_TOKENS = 6000

KINDS = ("a-roll", "b-roll", "silent-broll", "audio-only", "instruction")

SCHEMA_HINT = """{
  "clip": "string", "summary": "string",
  "location": {"name": "string|null", "city": "string|null", "country": "string|null", "confidence": 0.0},
  "kind": "a-roll|b-roll|silent-broll|audio-only|instruction",
  "instructions": [{"s": 0.0, "e": 0.0, "text": "string", "action": "move_to_end|use_for_intro|use_for_outro|discard|other"}],
  "takes": [{"topic": "string", "attempts": [{"s": 0.0, "e": 0.0}], "keep": 0, "reason": "string"}],
  "segments": [{"s": 0.0, "e": 0.0, "role": "narration|ambient|instruction|silence", "keep": true, "text": "string", "quality": 0.0}],
  "background_music": [{"s": 0.0, "e": 0.0, "confidence": 0.0, "suggest": "mute|duck|keep"}],
  "visual": {"quality": 0.0, "issues": ["string"], "best_frames": [0.0], "thumbnail_candidate": true},
  "hooks": ["string"], "numbers": ["string"], "topics": ["string"]
}"""


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


class _Lenient(BaseModel):
    """Base model: unknown keys are kept, ``null`` collapses to the default."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Location(_Lenient):
    name: str | None = None
    city: str | None = None
    country: str | None = None
    confidence: float = 0.0


class Instruction(_Lenient):
    s: float = 0.0
    e: float = 0.0
    text: str = ""
    action: str = "other"


class Attempt(_Lenient):
    s: float = 0.0
    e: float = 0.0
    text: str | None = None


class Take(_Lenient):
    topic: str = ""
    attempts: list[Attempt] = Field(default_factory=list)
    keep: int = 0
    reason: str = ""


class Segment(_Lenient):
    s: float = 0.0
    e: float = 0.0
    role: str = "narration"
    keep: bool = True
    text: str = ""
    quality: float = 0.0


class MusicFlag(_Lenient):
    s: float = 0.0
    e: float = 0.0
    confidence: float = 0.0
    suggest: str = "mute"


class Visual(_Lenient):
    quality: float = 0.0
    issues: list[str] = Field(default_factory=list)
    best_frames: list[float] = Field(default_factory=list)
    thumbnail_candidate: bool = False


class ClipAnalysis(_Lenient):
    """The ``analysis/<clip>.json`` schema from ``docs/ARCHITECTURE.md``."""

    clip: str = ""
    summary: str = ""
    location: Location = Field(default_factory=Location)
    kind: str = "b-roll"
    instructions: list[Instruction] = Field(default_factory=list)
    takes: list[Take] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    background_music: list[MusicFlag] = Field(default_factory=list)
    visual: Visual = Field(default_factory=Visual)
    hooks: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    @field_validator(
        "instructions", "takes", "segments", "background_music", "hooks",
        "numbers", "topics", mode="before",
    )
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("location", "visual", mode="before")
    @classmethod
    def _none_to_object(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("kind", mode="before")
    @classmethod
    def _known_kind(cls, value: Any) -> Any:
        text = str(value or "").strip().lower().replace("_", "-")
        aliases = {
            "aroll": "a-roll", "broll": "b-roll", "silent": "silent-broll",
            "silent-b-roll": "silent-broll", "audio": "audio-only",
        }
        text = aliases.get(text, text)
        return text if text in KINDS else "b-roll"


def clamp_times(analysis: ClipAnalysis, duration: float) -> ClipAnalysis:
    """Clamp every time field into ``[0, duration]`` (and keep ``e >= s``)."""
    limit = max(0.0, float(duration or 0.0))

    def _fix(obj: Any) -> None:
        start = max(0.0, float(getattr(obj, "s", 0.0) or 0.0))
        end = max(0.0, float(getattr(obj, "e", 0.0) or 0.0))
        if limit:
            start = min(start, limit)
            end = min(end, limit)
        obj.s = round(start, 3)
        obj.e = round(max(end, start), 3)

    for item in analysis.instructions + analysis.segments + analysis.background_music:
        _fix(item)
    for take in analysis.takes:
        for attempt in take.attempts:
            _fix(attempt)
        if take.attempts:
            take.keep = max(0, min(int(take.keep or 0), len(take.attempts) - 1))
    # A "take" only means something when the narrator tried more than once;
    # single-attempt entries are just the segment list restated.
    analysis.takes = [t for t in analysis.takes if len(t.attempts) >= 2]
    analysis.visual.best_frames = [
        round(max(0.0, min(float(t or 0.0), limit if limit else float(t or 0.0))), 3)
        for t in analysis.visual.best_frames
    ]
    return analysis


# --------------------------------------------------------------------------- #
# prompt inputs
# --------------------------------------------------------------------------- #


def compact_transcript(
    transcript: Mapping[str, Any] | None, token_budget: int = TRANSCRIPT_TOKEN_BUDGET
) -> str:
    """Render the transcript for the prompt as ``[["word", s, e], ...]``.

    Confidence (``p``) and speaker are dropped and times are rounded to two
    decimals.  When the word list still exceeds ``token_budget``, it degrades to
    sentence-level triples, then to plain text.
    """
    if not transcript:
        return "[]"
    words = [
        [str(w.get("t", "")), round(float(w.get("s") or 0.0), 2), round(float(w.get("e") or 0.0), 2)]
        for w in (transcript.get("words") or [])
        if str(w.get("t", "")).strip()
    ]
    if not words:
        return json.dumps({"text": transcript.get("text", "")}, ensure_ascii=False)

    limit = token_budget * _CHARS_PER_TOKEN
    payload = json.dumps(words, ensure_ascii=False, separators=(",", ":"))
    if len(payload) <= limit:
        return payload

    from .transcribe import build_sentences  # local import: avoids a cycle at import time

    sentences = [
        [s["text"], round(float(s["s"]), 2), round(float(s["e"]), 2)]
        for s in build_sentences(transcript.get("words") or [])
    ]
    payload = json.dumps(sentences, ensure_ascii=False, separators=(",", ":"))
    if len(payload) <= limit:
        log.info("transcript compacted to sentence level for the prompt")
        return payload
    text = str(transcript.get("text") or "")[: limit - 64]
    return json.dumps({"text": text, "truncated": True}, ensure_ascii=False)


def take_hints(transcript: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Deterministic take candidates: from the transcript, or recomputed."""
    if not transcript:
        return []
    hints = transcript.get("take_hints")
    if isinstance(hints, list):
        return hints
    return detect_takes(transcript.get("words") or [])


def select_frames(
    project: Project, clip_id: str, max_frames: int = MAX_FRAMES
) -> list[tuple[float, bytes]]:
    """Pick up to ``max_frames`` evenly spaced sampled frames for a clip.

    Frames were written by ingest into ``media/thumbs/frames/<clip>/NNN.jpg``
    together with an ``index.json`` holding their timestamps.
    """
    directory = project.clip_frames_dir(clip_id)
    files = sorted(directory.glob("[0-9]*.jpg")) if directory.is_dir() else []
    if not files:
        return []
    times: list[float] = []
    index = directory / "index.json"
    if index.exists():
        try:
            times = [float(t) for t in (json.loads(index.read_text(encoding="utf-8")).get("times") or [])]
        except (json.JSONDecodeError, TypeError, ValueError):  # pragma: no cover
            times = []
    if len(times) < len(files):
        interval = float(project.settings.get("ingest.frames.interval", 3.0))
        times = [round(i * interval, 3) for i in range(len(files))]

    count = min(max_frames, len(files))
    if count == len(files):
        picks = list(range(len(files)))
    elif count == 1:
        picks = [len(files) // 2]
    else:
        step = (len(files) - 1) / (count - 1)
        picks = sorted({int(round(i * step)) for i in range(count)})
    return [(times[i], files[i].read_bytes()) for i in picks]


def annotate(frames: Sequence[tuple[float, bytes]]) -> tuple[list[bytes], list[str]]:
    """Label frames with their timestamp; returns ``(images, labels)``."""
    images: list[bytes] = []
    labels: list[str] = []
    for i, (t, jpeg) in enumerate(frames):
        try:
            images.append(annotate_frame(jpeg, i, t, grid=False, label=f"t={t:.1f}s"))
        except Exception as exc:  # noqa: BLE001 - a broken JPEG must not kill the clip
            log.warning("frame %d annotation failed (%s), using the raw frame", i, exc)
            images.append(jpeg)
        labels.append(f"Frame {i} (t={t:.1f}s):")
    return images, labels


def build_user_prompt(
    project: Project,
    clip: Mapping[str, Any],
    transcript: Mapping[str, Any] | None,
    frame_times: Sequence[float],
) -> str:
    """Render ``analyze.user`` and append the deterministic take hints."""
    clip_id = str(clip["id"])
    events = list((transcript or {}).get("events") or [])
    text = prompts.render(
        "analyze.user",
        clip_id=clip_id,
        project_language=project.language,
        duration_seconds=f"{float(clip.get('duration') or 0.0):.2f}",
        transcript_json=compact_transcript(transcript),
        audio_events_json=json.dumps(events, ensure_ascii=False),
        annotated_frames_placeholder=(
            "(see attached images)" if frame_times else "(no frames available)"
        ),
    )
    extra: list[str] = [text]
    hints = take_hints(transcript)
    if hints:
        extra.append(
            "\nDeterministic take hints (verify): repeated word sequences found by a "
            "non-AI n-gram matcher. They are candidates, not decisions — confirm or "
            "reject each one against the transcript, and apply the last-take rule.\n"
            + json.dumps(hints, ensure_ascii=False, indent=1)
        )
    if transcript is not None and not transcript.get("has_audio", True):
        extra.append(
            "\nThis clip has NO audio track at all: judge it visually only "
            '(kind is almost certainly "silent-broll").'
        )
    if (transcript or {}).get("language_mismatch"):
        extra.append(
            f"\nNote: the detected transcript language "
            f"({(transcript or {}).get('language')}) differs from the project "
            f"language ({project.language})."
        )
    clip_facts = {
        "orientation": clip.get("orientation"),
        "has_audio": bool(clip.get("has_audio")),
        "width": clip.get("width"),
        "height": clip.get("height"),
        "hdr": clip.get("hdr"),
        "source_file": clip.get("source_file"),
    }
    extra.append("\nClip technical facts: " + json.dumps(clip_facts, ensure_ascii=False))
    return "\n".join(extra)


# --------------------------------------------------------------------------- #
# per-clip analysis
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AnalyzeResult:
    """Outcome of analysing one clip."""

    clip_id: str
    status: str = "pending"  # done | skipped | error
    model: str = ""
    kind: str = ""
    instructions: int = 0
    takes: int = 0
    error: str | None = None


def _estimate_call(project: Project, model: str, prompt: str, frames: int) -> float:
    """Rough token estimate for the budget check before an analyst call."""
    tokens_in = len(prompt) // _CHARS_PER_TOKEN + frames * 400 + 1200
    return estimate(
        "openrouter",
        "chat",
        {"in": tokens_in, "out": MAX_OUTPUT_TOKENS // 3},
        settings=project.settings,
        model=model,
    )


def analyze_clip(
    project: Project,
    clip: Mapping[str, Any],
    client: OpenRouter | None = None,
) -> dict[str, Any]:
    """Analyze one clip and return the validated analysis document.

    Clips without usable audio go to the cheaper ``models.vision`` model; every
    other clip goes to ``models.analyst``.
    """
    clip_id = str(clip["id"])
    transcript = load_transcript(project, clip_id)
    has_speech = bool(clip.get("has_audio")) and bool((transcript or {}).get("words"))
    role = "analyst" if has_speech else "vision"
    model = project.settings.model(role)

    frames = select_frames(project, clip_id)
    images, labels = annotate(frames)
    system_prompt = prompts.render("analyze.system", project_language=project.language)
    user_prompt = build_user_prompt(project, clip, transcript, [t for t, _ in frames])

    check_budget(project, _estimate_call(project, model, user_prompt, len(images)))

    owned = client is None
    client = client or OpenRouter(
        api_key=project.settings.require_key("openrouter"),
        cost_callback=cost_recorder(project, "analyze", clip_id),
    )
    try:
        payload = client.ask_json(
            model=model,
            system_prompt=system_prompt,
            user=user_prompt,
            schema_hint=SCHEMA_HINT,
            images=images,
            labels=labels,
            detail="low",
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        try:
            analysis = ClipAnalysis.model_validate(payload)
        except ValidationError as exc:
            log.warning("[clip]%s[/]: analysis failed validation, retrying once", clip_id)
            payload = client.ask_json(
                model=model,
                system_prompt=system_prompt,
                user=(
                    f"{user_prompt}\n\nYour previous answer did not validate against "
                    f"the schema. Fix exactly these problems and return the whole "
                    f"object again:\n{exc}"
                ),
                schema_hint=SCHEMA_HINT,
                images=images,
                labels=labels,
                detail="low",
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            analysis = ClipAnalysis.model_validate(payload)
    finally:
        if owned:
            client.close()

    analysis.clip = clip_id
    analysis = clamp_times(analysis, float(clip.get("duration") or 0.0))
    document = analysis.model_dump()
    document.update(
        {
            "clip": clip_id,
            "model": model,
            "has_audio": bool(clip.get("has_audio")),
            "duration": round(float(clip.get("duration") or 0.0), 3),
            "language": (transcript or {}).get("language"),
            "language_mismatch": bool((transcript or {}).get("language_mismatch")),
            "take_hints": take_hints(transcript),
            "frames_used": [round(t, 3) for t, _ in frames],
            "created": utcnow(),
        }
    )
    return document


def write_analysis(project: Project, document: Mapping[str, Any]) -> Path:
    """Write ``analysis/<clip>.json``."""
    path = project.analysis_path(str(document["clip"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_analysis(project: Project, clip_id: str) -> dict[str, Any] | None:
    """Read one per-clip analysis document, or ``None``."""
    path = project.analysis_path(clip_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupted artifact
        log.error("unreadable analysis %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# footage log
# --------------------------------------------------------------------------- #


def footage_log_path(project: Project) -> Path:
    """``analysis/footage_log.json``."""
    return project.analysis_dir / "footage_log.json"


def build_footage_log(project: Project) -> dict[str, Any]:
    """Merge every per-clip analysis into the chronological footage log."""
    clips_out: list[dict[str, Any]] = []
    instructions_found: list[dict[str, Any]] = []
    music_flags: list[dict[str, Any]] = []
    language_mismatches: list[str] = []
    locations: list[str] = []
    total_duration = 0.0

    for clip in project.clips_in_order():
        clip_id = str(clip.get("id", ""))
        analysis = load_analysis(project, clip_id) or {}
        transcript = load_transcript(project, clip_id) or {}
        duration = round(float(clip.get("duration") or 0.0), 3)
        total_duration += duration
        location = analysis.get("location") or {}

        entry = {
            "id": clip_id,
            "order": int(clip.get("order", 0) or 0),
            "source_file": clip.get("source_file", ""),
            "duration": duration,
            "orientation": clip.get("orientation", ""),
            "kind": analysis.get("kind", ""),
            "has_audio": bool(clip.get("has_audio")),
            "language": transcript.get("language") or analysis.get("language"),
            "summary": analysis.get("summary", ""),
            "location": location,
            "instructions": analysis.get("instructions") or [],
            "takes": analysis.get("takes") or [],
            "segments": analysis.get("segments") or [],
            "background_music": analysis.get("background_music") or [],
            "visual": analysis.get("visual") or {},
            "hooks": analysis.get("hooks") or [],
            "numbers": analysis.get("numbers") or [],
            "topics": analysis.get("topics") or [],
            "recorded_at": clip.get("recorded_at", ""),
        }
        clips_out.append(entry)

        for instruction in entry["instructions"]:
            instructions_found.append({"clip": clip_id, **dict(instruction)})
        for flag in entry["background_music"]:
            if str(flag.get("suggest", "mute")) != "keep":
                music_flags.append({"clip": clip_id, **dict(flag)})
        if transcript.get("language_mismatch") or analysis.get("language_mismatch"):
            language_mismatches.append(clip_id)
        name = str(location.get("name") or location.get("city") or "").strip()
        if name and name not in locations:
            locations.append(name)

    return {
        "project": project.slug,
        "language": project.language,
        "total_duration": round(total_duration, 3),
        "clips_count": len(clips_out),
        "instructions_found": instructions_found,
        "music_flags": music_flags,
        "language_mismatches": language_mismatches,
        "locations": locations,
        "generated": utcnow(),
        "clips": clips_out,
    }


def _hms(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, float(seconds)), 60)
    return f"{int(minutes)}:{secs:04.1f}"


def render_footage_log_md(log_doc: Mapping[str, Any]) -> str:
    """Render the human-readable ``analysis/footage_log.md``."""
    lines: list[str] = [
        f"# Footage log — {log_doc.get('project', '')}",
        "",
        f"Language: `{log_doc.get('language', '')}` · clips: "
        f"{log_doc.get('clips_count', 0)} · total duration: "
        f"{_hms(float(log_doc.get('total_duration') or 0.0))}",
        "",
        "| clip | order | dur | kind | audio | location | summary |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for clip in log_doc.get("clips", []):
        location = clip.get("location") or {}
        place = " / ".join(
            str(x) for x in (location.get("name"), location.get("city")) if x
        )
        summary = str(clip.get("summary", "")).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {clip.get('id','')} | {clip.get('order','')} | "
            f"{float(clip.get('duration') or 0.0):.1f}s | {clip.get('kind','')} | "
            f"{'yes' if clip.get('has_audio') else 'no'} | {place or '-'} | {summary} |"
        )

    lines += ["", "## Editor instructions", ""]
    instructions = log_doc.get("instructions_found") or []
    if not instructions:
        lines.append("_none detected_")
    for item in instructions:
        lines.append(
            f"- **{item.get('clip','')}** {_hms(float(item.get('s') or 0.0))}–"
            f"{_hms(float(item.get('e') or 0.0))} · `{item.get('action','other')}` — "
            f"\"{str(item.get('text','')).strip()}\""
        )

    lines += ["", "## Copyright-music flags", ""]
    flags = log_doc.get("music_flags") or []
    if not flags:
        lines.append("_none detected_")
    for flag in flags:
        lines.append(
            f"- **{flag.get('clip','')}** {_hms(float(flag.get('s') or 0.0))}–"
            f"{_hms(float(flag.get('e') or 0.0))} · suggest "
            f"`{flag.get('suggest','mute')}` (confidence "
            f"{float(flag.get('confidence') or 0.0):.2f})"
        )

    lines += ["", "## Takes", ""]
    any_take = False
    for clip in log_doc.get("clips", []):
        for take in clip.get("takes") or []:
            any_take = True
            attempts = take.get("attempts") or []
            keep = int(take.get("keep") or 0)
            rendered = ", ".join(
                f"{'**' if i == keep else ''}#{i} "
                f"{_hms(float(a.get('s') or 0.0))}–{_hms(float(a.get('e') or 0.0))}"
                f"{'**' if i == keep else ''}"
                for i, a in enumerate(attempts)
            )
            lines.append(
                f"- **{clip.get('id','')}** “{take.get('topic','')}”: {rendered} "
                f"→ keep #{keep}"
                + (f" ({take.get('reason')})" if take.get("reason") else "")
            )
    if not any_take:
        lines.append("_no repeated takes detected_")

    mismatches = log_doc.get("language_mismatches") or []
    if mismatches:
        lines += ["", "## Language mismatches", "", "- " + ", ".join(mismatches)]
    locations = log_doc.get("locations") or []
    if locations:
        lines += ["", "## Locations, in order", "", "- " + "\n- ".join(locations)]
    return "\n".join(lines) + "\n"


def write_footage_log(project: Project) -> dict[str, Any]:
    """Build and write ``footage_log.json`` and ``footage_log.md``."""
    document = build_footage_log(project)
    footage_log_path(project).write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (project.analysis_dir / "footage_log.md").write_text(
        render_footage_log_md(document), encoding="utf-8"
    )
    return document


def load_footage_log(project: Project) -> dict[str, Any]:
    """Read ``analysis/footage_log.json`` (``{}`` when absent/unreadable)."""
    path = footage_log_path(project)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupted artifact
        log.error("unreadable footage log %s: %s", path, exc)
        return {}


# --------------------------------------------------------------------------- #
# stage entry point
# --------------------------------------------------------------------------- #


def _needs_work(project: Project, clip: Mapping[str, Any], force: bool) -> bool:
    if force:
        return True
    if str(clip.get("stages", {}).get("analyze", "")) != "done":
        return True
    return not project.analysis_path(str(clip.get("id", ""))).exists()


def analyze(project: Project, force: bool = False) -> list[AnalyzeResult]:
    """Analyze every clip that still needs it, then rebuild the footage log.

    Args:
        project: Target project (transcribed).
        force: Re-analyze clips already marked ``done``.

    Returns:
        One :class:`AnalyzeResult` per clip that was considered.
    """
    clips = [c for c in project.clips_in_order() if _needs_work(project, c, force)]
    if not clips:
        log.info("[stage]analyze[/]: nothing to do, refreshing footage log")
        write_footage_log(project)
        project.set_stage("analyze", "done", cost_usd=0.0)
        return []

    log.info("[stage]analyze[/]: %d clip(s) via %s", len(clips), project.settings.model("analyst"))
    project.set_stage("analyze", "running")
    before = _spent(project)

    def _one(clip: Mapping[str, Any]) -> AnalyzeResult:
        clip_id = str(clip["id"])
        result = AnalyzeResult(clip_id=clip_id)
        try:
            document = analyze_clip(project, clip)
            write_analysis(project, document)
            result.status = "done"
            result.model = str(document.get("model", ""))
            result.kind = str(document.get("kind", ""))
            result.instructions = len(document.get("instructions") or [])
            result.takes = len(document.get("takes") or [])
            project.set_clip_stage(
                clip_id,
                "analyze",
                "done",
                analysis=project.rel(project.analysis_path(clip_id)),
                analysis_kind=result.kind,
            )
            log.info(
                "[clip]%s[/] analyzed: kind=%s instructions=%d takes=%d",
                clip_id, result.kind, result.instructions, result.takes,
            )
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the stage
            result.status = "error"
            result.error = str(exc)
            log.error("[clip]%s[/] analysis failed: %s", clip_id, exc)
            project.set_clip_stage(clip_id, "analyze", "error", error=str(exc)[:2000])
        return result

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(_one, clips))

    document = write_footage_log(project)
    spent = round(_spent(project) - before, 6)
    errors = [r for r in results if r.status == "error"]
    project.set_stage(
        "analyze",
        "error" if errors else "done",
        cost_usd=spent,
        **({"error": f"{len(errors)} clip(s) failed"} if errors else {}),
    )
    log.info(
        "[stage]analyze[/]: %d done, %d error, %d instruction(s), %d music flag(s), "
        "[cost]$%.4f[/]",
        sum(1 for r in results if r.status == "done"),
        len(errors),
        len(document.get("instructions_found") or []),
        len(document.get("music_flags") or []),
        spent,
    )
    return results


def _spent(project: Project) -> float:
    return round(
        sum(float(e.get("usd", 0.0)) for e in project.load_state().get("costs", [])), 6
    )


__all__ = [
    "analyze",
    "analyze_clip",
    "ClipAnalysis",
    "clamp_times",
    "compact_transcript",
    "select_frames",
    "build_user_prompt",
    "build_footage_log",
    "write_footage_log",
    "load_footage_log",
    "load_analysis",
    "render_footage_log_md",
    "AnalyzeResult",
    "SCHEMA_HINT",
]
