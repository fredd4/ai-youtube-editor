"""Stage 5: music cue sheet -> ``music/<cue>.mp3`` (+ sidecars, manifest).

Cues come from ``plan/cut.json: music[]`` when there is a cut; otherwise from
an explicit ``styles`` argument, otherwise from a single default style guessed
from the footage log's moods/locations against the tags in
``config/music_styles.yaml``.

A cue in the cut is a **beat range** (``from``/``to``, inclusive), not a pair
of absolute seconds, so a bed is generated exactly as long as the beats it
covers actually resolve to (:func:`ytedit.cut.beat_spans`). That is the whole
point of the cut being the source of truth: re-plan, add a narration pickup or
hand-edit the cut and the next ``ytedit music`` run sizes the bed to the new
cut instead of to the length some earlier producer wrote down.

Every track is instrumental and generated in ``loop`` mode so a bed can be
trimmed or extended to any cue length without an audible ending (see
``docs/research/technical-stack.md`` 2).  Generation is billed at
:data:`MUSIC_USD_PER_MINUTE`; the estimate is shown and budget-checked before
anything is generated.
"""

from __future__ import annotations

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rich.table import Table

from ..costs import check_budget, estimate
from ..log import console, get_logger
from ..project import Project, utcnow
from .elevenlabs import MUSIC_USD_PER_MINUTE, ElevenLabs
from .transcribe import cost_recorder

log = get_logger(__name__)

#: ElevenLabs music concurrency limit on Starter/Creator/Pro.
WORKERS = 2

#: Fallback cue length when neither the plan nor the caller says otherwise.
DEFAULT_LENGTH_S = 90

#: ``/v1/music`` accepts 3-600 s.
MIN_LENGTH_MS = 3_000
MAX_LENGTH_MS = 600_000

#: Style used when nothing in the footage log matches a preset.
FALLBACK_STYLE = "arrival-warm"


@dataclass(slots=True)
class MusicCue:
    """One bed to generate."""

    id: str
    style: str
    length_s: float = DEFAULT_LENGTH_S
    mood: str = ""
    section: str = ""
    prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "style": self.style,
            "length_s": round(float(self.length_s), 2),
            "mood": self.mood,
            "section": self.section,
            "prompt": self.prompt,
        }


@dataclass(slots=True)
class MusicResultInfo:
    """Outcome of one cue."""

    cue_id: str
    status: str = "pending"  # done | skipped | error
    path: str = ""
    style: str = ""
    length_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- #
# style presets
# --------------------------------------------------------------------------- #


def styles_table(settings: Any) -> dict[str, dict[str, Any]]:
    """Return the ``styles:`` mapping from ``config/music_styles.yaml``.

    Both layouts are accepted: the nested ``{"styles": {...}}`` file shipped in
    ``config/`` and a flat ``{name: preset}`` mapping.
    """
    raw = dict(settings.music_styles or {})
    nested = raw.get("styles")
    if isinstance(nested, Mapping):
        return {str(k): dict(v) for k, v in nested.items() if isinstance(v, Mapping)}
    return {
        str(k): dict(v)
        for k, v in raw.items()
        if isinstance(v, Mapping) and "prompt" in v
    }


def music_defaults(settings: Any) -> dict[str, Any]:
    """Return the ``defaults:`` block of ``config/music_styles.yaml``."""
    raw = dict(settings.music_styles or {})
    block = raw.get("defaults")
    return dict(block) if isinstance(block, Mapping) else {}


def get_preset(settings: Any, style: str) -> dict[str, Any]:
    """Look up one style preset.

    Raises:
        KeyError: When the style is not defined in ``music_styles.yaml``.
    """
    table = styles_table(settings)
    if style not in table:
        raise KeyError(f"unknown music style {style!r}; known: {sorted(table)}")
    return table[style]


def _tokens(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", str(text or "").lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return {t for t in re.split(r"[^a-z0-9]+", folded) if len(t) > 2}


def pick_style_for(mood_keywords: Sequence[str], settings: Any) -> str:
    """Choose the best-matching style preset for a list of mood keywords.

    Scoring is a plain token overlap against each preset's key, ``label`` and
    ``mood`` tags — deterministic and cheap, no LLM call.
    """
    table = styles_table(settings)
    if not table:
        return FALLBACK_STYLE
    wanted: set[str] = set()
    for keyword in mood_keywords or ():
        wanted |= _tokens(keyword)
    if not wanted:
        return FALLBACK_STYLE if FALLBACK_STYLE in table else sorted(table)[0]

    best, best_score = "", -1.0
    for name, preset in sorted(table.items()):
        tags = _tokens(name) | _tokens(preset.get("label", ""))
        for tag in preset.get("mood") or ():
            tags |= _tokens(tag)
        score = float(len(tags & wanted))
        # A weak tie-breaker: substring hits on the style name itself.
        score += 0.5 * sum(1 for w in wanted if w in name.replace("-", ""))
        if score > best_score:
            best, best_score = name, score
    if best_score <= 0:
        return FALLBACK_STYLE if FALLBACK_STYLE in table else sorted(table)[0]
    return best


def default_mood_keywords(project: Project) -> list[str]:
    """Mood hints for style selection: project.yaml plus the footage log."""
    keywords: list[str] = []
    mood = project.config.get("music_mood")
    if mood:
        keywords.append(str(mood))
    keywords.append(str(project.config.get("style", "")))

    from .analyze import load_footage_log  # local import: avoids an import cycle

    document = load_footage_log(project)
    for location in (document.get("locations") or [])[:5]:
        keywords.append(str(location))
    for clip in (document.get("clips") or [])[:12]:
        keywords.extend(str(t) for t in (clip.get("topics") or [])[:3])
    return [k for k in keywords if k.strip()]


# --------------------------------------------------------------------------- #
# cue sources
# --------------------------------------------------------------------------- #


def _slug(text: str, fallback: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text or "").lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return slug or fallback


def normalize_cue(raw: Mapping[str, Any], index: int, settings: Any) -> MusicCue:
    """Turn one cue dict into a :class:`MusicCue`.

    Accepts the documented ``{id, style, prompt, length_s, mood, section}``
    shape — what :func:`cues_from_plan` builds out of the cut — and, for cues
    typed on the command line or left over in an old plan, the planner's
    ``{section, s, e, mood}`` cue-sheet shape.
    """
    section = str(raw.get("section") or raw.get("where") or "")
    mood = str(raw.get("mood") or "")
    cue_id = str(raw.get("id") or "").strip() or f"{index + 1:02d}-{_slug(section or mood, 'cue')}"

    length = raw.get("length_s")
    if length is None and raw.get("s") is not None and raw.get("e") is not None:
        length = float(raw.get("e") or 0.0) - float(raw.get("s") or 0.0)
    if not length or float(length) <= 0:
        length = DEFAULT_LENGTH_S

    style = str(raw.get("style") or "").strip()
    if not style or style not in styles_table(settings):
        # The planner describes the cue in prose (often in the project language);
        # keep that description as the director's note and pick the closest
        # preset as the musical base.
        if style:
            mood = "; ".join(x for x in (style, mood) if x)
        style = pick_style_for([mood, section, style], settings)
    # Loop-mode beds shorter than ~30 s sound abrupt; render trims/loops anyway.
    length = max(float(length), float(music_defaults(settings).get("min_length_s", 30)))
    return MusicCue(
        id=cue_id,
        style=style,
        length_s=float(length),
        mood=mood,
        section=section,
        prompt=(str(raw["prompt"]) if raw.get("prompt") else None),
    )


#: Cue metadata the planner may have written next to a cut cue. ``MusicCue``
#: is ``extra="allow"``, so these survive a ``cut.json`` round trip and are the
#: cue's musical direction; the *length* never comes from here.
CUE_META_KEYS: tuple[str, ...] = ("style", "mood", "section", "prompt")


def _edit_plan_cues(project: Project) -> list[Mapping[str, Any]]:
    """Raw ``music_cues`` out of ``plan/edit_plan.json`` (``[]`` when absent).

    Only the *style/mood/section* wording is ever taken from here — a cue's
    length is a fact about the cut, and the edit plan's ``s``/``e`` are the
    seconds some earlier planning pass guessed at.
    """
    path = project.edit_plan_file
    if not path.exists():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupted artifact
        log.error("unreadable edit plan %s: %s", path, exc)
        return []
    if isinstance(document, Mapping):
        for key in ("music_cues", "edit_plan", "plan"):
            value = document.get(key)
            if isinstance(value, list):
                return [c for c in value if isinstance(c, Mapping)]
            if isinstance(value, Mapping) and isinstance(value.get("music_cues"), list):
                return [c for c in value["music_cues"] if isinstance(c, Mapping)]
    return []


def _plan_cue_meta(project: Project) -> dict[str, Mapping[str, Any]]:
    """``{cue id: edit-plan cue}`` — the fallback source of a cue's wording."""
    out: dict[str, Mapping[str, Any]] = {}
    for cue in _edit_plan_cues(project):
        cue_id = str(cue.get("id") or "").strip()
        if cue_id:
            out[cue_id] = cue
    return out


def cues_from_plan(project: Project) -> list[dict[str, Any]]:
    """Read the cue sheet from ``plan/cut.json``, sized against the cut.

    Each cue's ``length_s`` is the real duration of its beat range — the start
    of the ``from`` beat to the end of the ``to`` beat (inclusive), as
    :func:`ytedit.cut.beat_spans` resolves them. A cue whose beats no longer
    resolve (both ends deleted, or the range collapsed to nothing) is dropped
    with a warning: an ElevenLabs generation is paid for by the minute, and a
    bed of the wrong length is worse than no bed at all.

    The musical direction (``style``/``mood``/``section``/``prompt``) rides
    along on the cut's cue as extra fields; when the cut carries none of them —
    a cut produced by ``ytedit migrate``, which keeps only file/gain/fades —
    it falls back to the same-id cue in ``plan/edit_plan.json``.

    Returns:
        Cue dicts for :func:`normalize_cue`, in cut order (``[]`` when the
        project has no cut or no cues in it).
    """
    from ..cut import beat_spans, cut_path, load_cut

    path = cut_path(project)
    if not path.exists():
        return []
    try:
        cut = load_cut(path)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        log.error("unreadable cut %s: %s", path, exc)
        return []
    if not cut.music:
        return []

    spans, _issues = beat_spans(project, cut)
    plan_meta = _plan_cue_meta(project)

    out: list[dict[str, Any]] = []
    for cue in cut.music:
        first = spans.get(cue.from_)
        last = spans.get(cue.to)
        if first is None or last is None:
            log.warning(
                "music cue %s: beat range %s→%s produces no picture — skipped",
                cue.id, cue.from_, cue.to,
            )
            continue
        length = round(last[1] - first[0], 3)
        if length <= 0:
            log.warning(
                "music cue %s: beat range %s→%s is %.2f s long — skipped",
                cue.id, cue.from_, cue.to, length,
            )
            continue

        item: dict[str, Any] = {
            "id": cue.id,
            "file": cue.file,
            "gain_db": cue.gain_db,
            "length_s": length,
        }
        extra = cue.model_extra or {}
        fallback = plan_meta.get(cue.id, {})
        for key in CUE_META_KEYS:
            value = extra.get(key) or fallback.get(key)
            if value:
                item[key] = str(value)
        out.append(item)
    return out


def resolve_cues(
    project: Project,
    cues: Sequence[Mapping[str, Any]] | None = None,
    styles: Sequence[str] | None = None,
    length_s: int | float | None = None,
) -> list[MusicCue]:
    """Decide what to generate: explicit cues > cut cues > styles > default.

    ``length_s`` overrides every cue's length, whatever its source — the one
    way to ask for a bed that is not the length of its beat range.
    """
    settings = project.settings
    default_length = float(
        length_s
        or float(music_defaults(settings).get("default_length_ms", 90_000)) / 1000.0
    )

    raw: Sequence[Mapping[str, Any]] = cues if cues is not None else cues_from_plan(project)
    if raw:
        resolved = [normalize_cue(cue, i, settings) for i, cue in enumerate(raw)]
        if length_s:
            for cue in resolved:
                cue.length_s = float(length_s)
        return resolved

    if styles:
        return [
            MusicCue(
                id=f"{i + 1:02d}-{_slug(style, 'bed')}",
                style=style if style in styles_table(settings) else pick_style_for([style], settings),
                length_s=default_length,
                mood=style,
                section="manual",
            )
            for i, style in enumerate(styles)
        ]

    style = pick_style_for(default_mood_keywords(project), settings)
    log.info("no cue sheet: falling back to a single %r bed", style)
    return [
        MusicCue(
            id=f"01-{_slug(style, 'bed')}",
            style=style,
            length_s=default_length,
            mood=str(project.config.get("music_mood") or ""),
            section="default",
        )
    ]


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def build_prompt(preset: Mapping[str, Any], cue: MusicCue) -> str:
    """Compose the ElevenLabs prompt: preset formula + negatives + cue hint."""
    if cue.prompt:
        return cue.prompt.strip()
    parts = [" ".join(str(preset.get("prompt", "")).split())]
    hint = "; ".join(x for x in (cue.section, cue.mood) if x).strip()
    if hint:
        parts.append(
            f"Director's note for this cue (may be in another language, follow it): {hint}. "
            "Instrumental only, loopable, consistent energy, no vocals."
        )
    negative = str(preset.get("negative") or "").strip()
    if negative:
        parts.append(negative)
    return " ".join(p for p in parts if p)


def clamp_length_ms(length_s: float) -> int:
    """Clamp a cue length into the API's 3-600 s window."""
    return max(MIN_LENGTH_MS, min(MAX_LENGTH_MS, int(round(float(length_s) * 1000))))


def sidecar_path(project: Project, cue_id: str) -> Path:
    """``music/<cue>.json`` next to the generated mp3."""
    return project.music_dir / f"{cue_id}.json"


def track_path(project: Project, cue_id: str) -> Path:
    """``music/<cue>.mp3``."""
    return project.music_dir / f"{cue_id}.mp3"


def manifest_path(project: Project) -> Path:
    """``music/manifest.json``."""
    return project.music_dir / "manifest.json"


def list_music(project: Project) -> list[dict[str, Any]]:
    """Every generated bed, read back from its sidecar, sorted by cue id."""
    out: list[dict[str, Any]] = []
    if not project.music_dir.is_dir():
        return out
    for sidecar in sorted(project.music_dir.glob("*.json")):
        if sidecar.name == "manifest.json":
            continue
        try:
            out.append(json.loads(sidecar.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupted artifact
            log.error("unreadable music sidecar %s: %s", sidecar, exc)
    return out


def write_manifest(project: Project) -> Path:
    """Rewrite ``music/manifest.json`` from the sidecars on disk."""
    tracks = list_music(project)
    path = manifest_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "project": project.slug,
                "generated": utcnow(),
                "count": len(tracks),
                "total_cost_usd": round(sum(float(t.get("cost_usd") or 0.0) for t in tracks), 6),
                "tracks": tracks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def estimate_cues(cues: Sequence[MusicCue]) -> float:
    """Total USD for a cue list at $0.15 per generated minute."""
    return round(
        sum(clamp_length_ms(c.length_s) / 60_000.0 * MUSIC_USD_PER_MINUTE for c in cues), 6
    )


def _cue_table(cues: Sequence[MusicCue], total: float) -> Table:
    table = Table(title="music cues", header_style="bold cyan")
    for column in ("cue", "style", "length", "section", "est. $"):
        table.add_column(column)
    for cue in cues:
        table.add_row(
            cue.id,
            cue.style,
            f"{cue.length_s:.0f}s",
            cue.section or "-",
            f"{clamp_length_ms(cue.length_s) / 60_000.0 * MUSIC_USD_PER_MINUTE:.4f}",
        )
    table.caption = f"total estimated cost ${total:.4f}"
    return table


def generate_cue(
    project: Project, cue: MusicCue, force: bool = False
) -> MusicResultInfo:
    """Generate (or skip) one bed and write its mp3 plus sidecar."""
    settings = project.settings
    result = MusicResultInfo(cue_id=cue.id, style=cue.style, length_s=float(cue.length_s))
    mp3 = track_path(project, cue.id)
    if mp3.exists() and not force:
        result.status = "skipped"
        result.path = project.rel(mp3)
        log.info("music cue %s: %s already exists, skipping", cue.id, mp3.name)
        return result

    preset = get_preset(settings, cue.style)
    defaults = music_defaults(settings)
    prompt = build_prompt(preset, cue)
    length_ms = clamp_length_ms(cue.length_s)
    cost = length_ms / 60_000.0 * MUSIC_USD_PER_MINUTE
    check_budget(project, cost)

    client = ElevenLabs(
        api_key=settings.require_key("elevenlabs"),
        cost_callback=cost_recorder(project, "music", cue.id),
    )
    try:
        track = client.compose_music(
            prompt,
            length_ms,
            model_id=str(settings.model("music") or defaults.get("model_id", "music_v2")),
            force_instrumental=True,
            generation_mode=str(preset.get("mode") or defaults.get("generation_mode", "loop")),
            output_format=str(defaults.get("output_format", "mp3_44100_192")),
        )
    finally:
        client.close()

    mp3.parent.mkdir(parents=True, exist_ok=True)
    track.write(mp3)
    sidecar = {
        "id": cue.id,
        "file": project.rel(mp3),
        "prompt": prompt,
        "style": cue.style,
        "style_label": preset.get("label", cue.style),
        "mood": cue.mood,
        "section": cue.section,
        "song_id": track.song_id,
        "length_s": round(length_ms / 1000.0, 2),
        "length_ms": length_ms,
        "model_id": track.meta.get("model_id"),
        "generation_mode": track.meta.get("generation_mode"),
        "gain_db": defaults.get("gain_db", -18),
        "fade_in": defaults.get("fade_in", 2.0),
        "fade_out": defaults.get("fade_out", 3.0),
        "cost_usd": round(float(track.meta.get("cost_usd") or cost), 6),
        "bytes": len(track.audio),
        "generated_at": utcnow(),
    }
    sidecar_path(project, cue.id).write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result.status = "done"
    result.path = sidecar["file"]
    result.cost_usd = float(sidecar["cost_usd"])
    log.info(
        "music cue %s: %s (%s, %.0fs, [cost]$%.4f[/])",
        cue.id, mp3.name, cue.style, length_ms / 1000.0, result.cost_usd,
    )
    return result


def generate_music(
    project: Project,
    force: bool = False,
    cues: Sequence[Mapping[str, Any]] | None = None,
    styles: Sequence[str] | None = None,
    length_s: int | None = None,
) -> list[MusicResultInfo]:
    """Generate every music bed the project asks for.

    Args:
        project: Target project.
        force: Regenerate beds whose mp3 already exists.
        cues: Explicit cue dicts; defaults to ``plan/cut.json:music[]``.
        styles: Style names to generate when there is no cue sheet.
        length_s: Override every cue's length in seconds.

    Returns:
        One :class:`MusicResultInfo` per cue.
    """
    resolved = resolve_cues(project, cues=cues, styles=styles, length_s=length_s)
    if not resolved:
        log.info("[stage]music[/]: no cues to generate")
        project.set_stage("music", "done", cost_usd=0.0)
        return []

    pending = [c for c in resolved if force or not track_path(project, c.id).exists()]
    total = estimate_cues(pending)
    console.print(_cue_table(resolved, total))
    check_budget(project, total)

    project.set_stage("music", "running")
    before = _spent(project)

    def _one(cue: MusicCue) -> MusicResultInfo:
        try:
            return generate_cue(project, cue, force=force)
        except Exception as exc:  # noqa: BLE001 - one bad cue must not kill the stage
            log.error("music cue %s failed: %s", cue.id, exc)
            return MusicResultInfo(
                cue_id=cue.id, status="error", style=cue.style,
                length_s=float(cue.length_s), error=str(exc),
            )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(_one, resolved))

    write_manifest(project)
    spent = round(_spent(project) - before, 6)
    errors = [r for r in results if r.status == "error"]
    project.set_stage(
        "music",
        "error" if errors else "done",
        cost_usd=spent,
        **({"error": f"{len(errors)} cue(s) failed"} if errors else {}),
    )
    log.info(
        "[stage]music[/]: %d generated, %d skipped, %d error, [cost]$%.4f[/]",
        sum(1 for r in results if r.status == "done"),
        sum(1 for r in results if r.status == "skipped"),
        len(errors),
        spent,
    )
    return results


def _spent(project: Project) -> float:
    return round(
        sum(float(e.get("usd", 0.0)) for e in project.load_state().get("costs", [])), 6
    )


__all__ = [
    "generate_music",
    "generate_cue",
    "list_music",
    "pick_style_for",
    "resolve_cues",
    "normalize_cue",
    "cues_from_plan",
    "build_prompt",
    "estimate_cues",
    "styles_table",
    "get_preset",
    "write_manifest",
    "MusicCue",
    "MusicResultInfo",
]
