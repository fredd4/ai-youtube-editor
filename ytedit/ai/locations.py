"""Location cards: ``analysis/footage_log.json`` -> canonical places -> timeline captions.

``ytedit captions <slug>`` is the fix for a specific complaint: location
captions pinned to absolute time (``Caption.at``/``end``) drift under the
wrong picture every time a later pass changes segment lengths (speech
padding, sentence snapping, overlay cutaways, audio dedupe, a hand edit in
the web editor). Two things happen here:

1. **Normalize places once per project.** The per-clip footage log carries
   free-text, sometimes-hedged location strings ("Belém
   (prawdopodobnie)", "Vila d'Ouro / Belém", "Lisboa region"). A single
   writer-model call (see ``docs/playbook/prompts.md``
   ``captions.places.system``/``.user``) turns the distinct raw strings into
   canonical ``place_id``/``label``/``region`` triples, merging near-duplicates.
   The result is cached to ``analysis/places.json`` (``--force`` re-asks) so
   re-running ``ytedit captions`` after a re-plan/re-tidy never re-spends on
   this.
2. **Place a card at every new location, anchored to its segment.** Walking
   ``tracks.video`` in order, whenever the picture's place changes to one not
   carded in the last ``captions.min_gap_s``, a :class:`~ytedit.timeline.Caption`
   is created with a :class:`~ytedit.timeline.CaptionAnchor` pinning it to
   that segment — so every later pass that moves the segment moves the card
   with it (:meth:`~ytedit.timeline.Timeline.resolve_anchors`) instead of
   leaving it a fixed number of seconds into the programme.

The planner's own ``hook`` captions are kept (anchored to whatever segment
sits under them right now); its ``location`` captions are dropped and
replaced by the generated ones, unless ``--keep-existing``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import Settings
from ..costs import charge
from ..log import get_logger
from ..project import Project, utcnow
from ..timeline import Caption, CaptionAnchor, Timeline
from .openrouter import OpenRouter
from .plan import PlanError, footage_entries, load_footage_log, units_for_ledger
from .prompts import render
from .publish import format_timecode

log = get_logger(__name__)

STAGE = "captions"

#: Confidence below which a footage-log location is not trusted on its own —
#: see :func:`_needs_inheritance`.
LOW_CONFIDENCE = 0.5

DEFAULT_MIN_GAP_S = 90.0
DEFAULT_LOCATION_SECONDS = 2.4
DEFAULT_CARD_OFFSET_S = 0.3
DEFAULT_MIN_SEGMENT_S = 2.5
DEFAULT_SKIP_COLD_OPEN_S = 10.0
DEFAULT_END_SCREEN_S = 15.0
DEFAULT_MAX_TOKENS = 4000

#: Appended to the system prompt by ``ask_json``. ``captions.places.system`` in
#: prompts.md carries the one and only schema; this is a pointer to it.
SCHEMA_HINT = "Return ONLY the JSON object described in the system prompt."

#: A place_id no group actually resolved to (dangling anchor style bug guard).
_NO_PLACE = ""


class LocationsError(RuntimeError):
    """Raised when the captions stage cannot run."""


# ----------------------------------------------------------------------
# raw locations from the footage log
# ----------------------------------------------------------------------
def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def raw_locations(footage_log: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-clip raw location info, in footage-log (chronological) order.

    Returns:
        ``[{"clip", "order", "name", "city", "country", "confidence"}, ...]``
        — ``order`` is the 0-based position in the footage log, used later as
        the "clip order" for nearest-neighbour inheritance. A clip with no
        ``location`` block at all gets ``name=""``, ``confidence=0.0``.
    """
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(footage_entries(footage_log)):
        loc = entry.get("location")
        loc = loc if isinstance(loc, dict) else {}
        out.append(
            {
                "clip": str(entry.get("clip") or entry.get("id") or ""),
                "order": i,
                "name": str(loc.get("name") or "").strip(),
                "city": str(loc.get("city") or "").strip(),
                "country": str(loc.get("country") or "").strip(),
                "confidence": _num(loc.get("confidence"), 0.0),
            }
        )
    return out


def _needs_inheritance(raw: dict[str, Any]) -> bool:
    """True when a clip's own location is unusable and must borrow a neighbour's.

    Per the playbook: a clip with no name at all AND low confidence. A named
    location — even a hedged, low-confidence one ("Belém
    (prawdopodobnie)") — is still sent to the writer for normalization rather
    than discarded, since it usually *is* the same real place.
    """
    return not raw["name"] and raw["confidence"] < LOW_CONFIDENCE


def group_raw_locations(raws: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse named locations into distinct-string groups, in first-seen order.

    Returns:
        ``[{"group_id": 0, "name", "city", "country", "clips": [...]}, ...]``
        — one entry per distinct exact ``(name, city, country)`` triple. Near-
        duplicate strings ("Belém" vs "Belém (prawdopodobnie)")
        are intentionally NOT merged here — that is the writer model's job
        (see ``captions.places.system``); this pass only avoids sending the
        same exact string hundreds of times.
    """
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for raw in raws:
        if not raw["name"]:
            continue
        key = (raw["name"], raw["city"], raw["country"])
        if key not in groups:
            groups[key] = {
                "group_id": len(order),
                "name": raw["name"],
                "city": raw["city"],
                "country": raw["country"],
                "clips": [],
            }
            order.append(key)
        groups[key]["clips"].append(raw["clip"])
    return [groups[k] for k in order]


# ----------------------------------------------------------------------
# canonicalization (one writer call per project)
# ----------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """A stable, ASCII, hyphenated id from a place name."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", normalized.lower()).strip("-")
    return slug or "place"


def _ask_writer(
    project: Project,
    settings: Settings,
    groups: Sequence[dict[str, Any]],
    client: OpenRouter | None = None,
) -> dict[str, Any]:
    """Render the places prompts and call the writer model once."""
    model = settings.model("writer")
    system_prompt = render("captions.places.system", project_language=project.language)
    payload = [
        {
            "group_id": g["group_id"],
            "name": g["name"],
            "city": g["city"],
            "country": g["country"],
            "example_clips": g["clips"][:5],
        }
        for g in groups
    ]
    user_prompt = render(
        "captions.places.user",
        project_language=project.language,
        raw_locations_json=json.dumps(payload, ensure_ascii=False),
    )

    spend: list[float] = []

    def cost_callback(
        service: str, op: str, model: str, units: Any = None, usd: float = 0.0, **extra: Any
    ) -> None:
        spend.append(float(usd))
        charge(project, service, op, units_for_ledger(units), usd=usd, model=model,
               stage=STAGE, **extra)

    owned = client is None
    api = client or OpenRouter(
        api_key=settings.require_key("openrouter"), cost_callback=cost_callback
    )
    try:
        max_tokens = int(settings.get("captions.max_tokens", DEFAULT_MAX_TOKENS))
        answer = api.ask_json(
            model, system_prompt, user_prompt, SCHEMA_HINT,
            temperature=0.2, max_tokens=max_tokens,
        )
    finally:
        if owned:
            api.close()
    return {"json": answer, "model": model, "cost_usd": round(sum(spend), 6)}


def _places_from_answer(
    groups: Sequence[dict[str, Any]], answer: Any
) -> dict[int, dict[str, str]]:
    """Map ``group_id`` -> ``{"place_id", "label", "region"}`` from the writer's JSON.

    Any ``group_id`` the writer did not answer for falls back to a naive
    normalization (slug of the raw name, the name itself as the label) rather
    than being left unassigned — a partial/malformed answer should degrade,
    not drop places silently.
    """
    by_group: dict[int, dict[str, str]] = {}
    items = answer.get("places") if isinstance(answer, dict) else None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        place_id = str(item.get("place_id") or "").strip() or slugify(label or "place")
        region = str(item.get("region") or "").strip()
        info = {"place_id": place_id, "label": label or place_id, "region": region}
        group_ids = item.get("group_ids")
        if not isinstance(group_ids, list) or not group_ids:
            group_ids = [item.get("group_id")]
        for gid in group_ids:
            try:
                by_group[int(gid)] = info
            except (TypeError, ValueError):
                continue

    for group in groups:
        gid = group["group_id"]
        if gid not in by_group:
            log.warning(
                "captions: writer did not place group %d (%r) — using a naive normalization",
                gid, group["name"],
            )
            by_group[gid] = {
                "place_id": slugify(group["name"]),
                "label": group["name"],
                "region": group.get("city") or "",
            }
    return by_group


# ----------------------------------------------------------------------
# places.json (cached document)
# ----------------------------------------------------------------------
def places_path(project: Project) -> Path:
    """Where the canonical per-clip place assignments are cached."""
    return project.analysis_dir / "places.json"


def build_places_document(
    project: Project,
    footage_log: dict[str, Any],
    settings: Settings | None = None,
    client: OpenRouter | None = None,
    ask_writer: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize every clip's location into a canonical place (one writer call).

    Returns the full ``analysis/places.json`` document: ``{"project",
    "language", "generated", "model", "cost_usd", "clips": [{"clip",
    "place_id", "label", "region", "source_name", "confidence", "inherited"},
    ...]}``. A clip with nothing usable at all (no name, low confidence, and
    no earlier-or-later clip with a place to borrow from) gets
    ``place_id=""``.
    """
    cfg = settings or project.settings
    raws = raw_locations(footage_log)
    groups = group_raw_locations(raws)

    if groups:
        asker = ask_writer or _ask_writer
        result = asker(project, cfg, groups, client=client)
        by_group = _places_from_answer(groups, result["json"])
        model = result["model"]
        cost = result["cost_usd"]
    else:
        by_group = {}
        model = ""
        cost = 0.0

    assignments: dict[str, dict[str, str]] = {}
    for group in groups:
        info = by_group.get(group["group_id"])
        if info is None:
            continue
        for clip in group["clips"]:
            assignments[clip] = dict(info)

    # Nearest-chronological-neighbour inheritance for clips with no usable
    # location of their own. Candidates are the clips just normalized above
    # (real, named places) — inheriting from one of those directly gives the
    # same result as chasing a chain of inherited neighbours, without the
    # bookkeeping.
    candidates = [(raws[i]["order"], raws[i]["clip"]) for i in range(len(raws))
                  if raws[i]["clip"] in assignments]
    for raw in raws:
        clip = raw["clip"]
        if clip in assignments or not _needs_inheritance(raw):
            continue
        best_clip: str | None = None
        best_dist: int | None = None
        for order, cand_clip in candidates:
            dist = abs(order - raw["order"])
            if best_dist is None or dist < best_dist:
                best_dist, best_clip = dist, cand_clip
        if best_clip is not None:
            assignments[clip] = dict(assignments[best_clip])

    named_clips = {c for g in groups for c in g["clips"]}
    per_clip = [
        {
            "clip": raw["clip"],
            "place_id": assignments.get(raw["clip"], {}).get("place_id", _NO_PLACE),
            "label": assignments.get(raw["clip"], {}).get("label", ""),
            "region": assignments.get(raw["clip"], {}).get("region", ""),
            "source_name": raw["name"],
            "confidence": raw["confidence"],
            "inherited": raw["clip"] in assignments and raw["clip"] not in named_clips,
        }
        for raw in raws
    ]

    return {
        "project": project.slug,
        "language": project.language,
        "generated": utcnow(),
        "model": model,
        "cost_usd": cost,
        "clips": per_clip,
    }


def ensure_places(
    project: Project,
    footage_log: dict[str, Any],
    settings: Settings | None = None,
    client: OpenRouter | None = None,
    ask_writer: Callable[..., dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Load the cached ``analysis/places.json``, or build and cache it.

    Args:
        force: Re-ask the writer even when a cache file already exists.
    """
    path = places_path(project)
    if path.exists() and not force:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("corrupted places cache %s (%s) — regenerating", path, exc)

    document = build_places_document(
        project, footage_log, settings=settings, client=client, ask_writer=ask_writer
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return document


def places_by_clip(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a places document's ``clips`` by clip id, dropping unassigned ones."""
    return {
        c["clip"]: c
        for c in document.get("clips", [])
        if isinstance(c, dict) and c.get("place_id")
    }


# ----------------------------------------------------------------------
# card placement
# ----------------------------------------------------------------------
#: Sentinel distinct from any real place_id (str) or "no place" (None), used
#: to force the very first eligible segment to be treated as a place change —
#: nothing has actually been shown to the viewer yet, cold open included.
_UNSEEN = object()

#: Segment notes that mark muted picture cuts laid under a narration pickup or
#: a voice-over montage — B-roll from many places under one story.
MONTAGE_NOTE_PREFIXES: tuple[str, ...] = (
    "VO picture for", "pickup", "a location price pickup", "street party pickup",
    "n001 bridge", "n002 bridge", "n003 bridge", "n004 bridge", "n005 bridge",
)


def generate_location_captions(
    timeline: Timeline,
    places: dict[str, dict[str, Any]],
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    location_seconds: float = DEFAULT_LOCATION_SECONDS,
    card_offset_s: float = DEFAULT_CARD_OFFSET_S,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
    skip_cold_open_s: float = DEFAULT_SKIP_COLD_OPEN_S,
    exclude_clips: Sequence[str] | None = None,
    end_screen_s: float = DEFAULT_END_SCREEN_S,
    include_cold_open: bool = False,
) -> list[dict[str, Any]]:
    """Decide where to place a location card, walking ``tracks.video`` in order.

    A card fires whenever the picture's place changes to one not carded in
    the last ``min_gap_s``. If the segment where the change lands is shorter
    than ``min_segment_s``, placement is deferred to the next segment still
    of that same place (as long as the place doesn't change again first); if
    none is long enough before that, the occurrence is skipped rather than
    forced onto a too-short cut. The cold open (first ``skip_cold_open_s``,
    unless ``include_cold_open``) and the final ``end_screen_s`` never get a
    card.

    Returns:
        Report rows in timeline order: ``[{"segment", "place_id", "label",
        "region", "at", "end"}, ...]`` — ``at``/``end`` are the card's
        absolute position *right now* (before being turned into an anchored
        :class:`~ytedit.timeline.Caption` by the caller).
    """
    positions = timeline.segment_positions()
    total = timeline.duration()
    eligible_start = 0.0 if include_cold_open else max(0.0, skip_cold_open_s)
    eligible_end = max(eligible_start, total - max(0.0, end_screen_s))

    excluded = set(exclude_clips or ())

    def place_of(clip: str) -> dict[str, Any] | None:
        if clip in excluded:
            return None
        return places.get(clip)

    def is_montage(segment: Any) -> bool:
        """Muted picture cuts under a voice pickup or narration montage."""
        notes = str(getattr(segment, "notes", "") or "")
        return bool(getattr(segment, "mute_source", False)) and any(
            notes.startswith(prefix) for prefix in MONTAGE_NOTE_PREFIXES
        )

    rows: list[dict[str, Any]] = []
    last_shown: dict[str, float] = {}
    active_place: Any = _UNSEEN
    entered_eligible = False

    n = len(positions)
    i = 0
    while i < n:
        pos = positions[i]
        place = place_of(pos.segment.clip)
        place_id = place["place_id"] if place else None

        eligible = eligible_start <= pos.start < eligible_end
        if not eligible:
            active_place = place_id
            i += 1
            continue
        if not entered_eligible:
            active_place = _UNSEEN
            entered_eligible = True

        if place_id is None or place_id == active_place:
            active_place = place_id
            i += 1
            continue
        if is_montage(pos.segment) or str(getattr(pos.segment, "role", "")) == "cutaway":
            # Montage cuts (B-roll under a pickup or voice-over) and cutaways
            # inside another scene are illustrations, not visits: no card, and
            # they do not count as leaving the scene, so the real arrival at
            # that place still gets its card.
            i += 1
            continue

        # A genuine place change. Mark it active immediately so segments
        # visited during the forward search below (or later in the main
        # loop) are never re-detected as the same change again.
        change_time = pos.start
        active_place = place_id
        if change_time - last_shown.get(place_id, float("-inf")) < min_gap_s:
            i += 1
            continue

        target = None
        j = i
        while j < n:
            cand = positions[j]
            if cand.start >= eligible_end:
                break
            cand_place = place_of(cand.segment.clip)
            cand_place_id = cand_place["place_id"] if cand_place else None
            if cand_place_id != place_id:
                break
            if (cand.end - cand.start) >= min_segment_s - 1e-6:
                target = cand
                break
            j += 1

        if target is None:
            i += 1
            continue

        assert place is not None  # place_id is not None here, so place isn't either
        at = round(target.start + card_offset_s, 3)
        end = round(at + location_seconds, 3)
        rows.append(
            {
                "segment": target.segment.id,
                "place_id": place_id,
                "label": place["label"],
                "region": place.get("region", ""),
                "at": at,
                "end": end,
            }
        )
        last_shown[place_id] = at
        i += 1

    return rows


# ----------------------------------------------------------------------
# applying to the timeline
# ----------------------------------------------------------------------
def build_timeline_captions(
    timeline: Timeline,
    document: dict[str, Any],
    settings: Settings | None = None,
    include_cold_open: bool = False,
    keep_existing: bool = False,
    exclude_clips: Sequence[str] | None = None,
) -> tuple[Timeline, list[dict[str, Any]]]:
    """Replace the planner's location captions with generated, anchored cards.

    The planner's ``hook`` captions are kept but anchored to whatever segment
    is under them right now (``offset = at - segment.start``); its
    ``location`` captions are dropped unless ``keep_existing``. Every other
    caption style (e.g. a hand-added ``subtitle`` cue) is left untouched.

    Returns:
        ``(timeline, report_rows)`` — ``report_rows`` cover every caption now
        on the timeline, in time order: ``[{"id", "style", "text", "segment",
        "at", "end"}, ...]``. The timeline is modified in place and returned.
    """
    cfg = settings or Settings()
    places = places_by_clip(document)

    rows = generate_location_captions(
        timeline,
        places,
        min_gap_s=float(cfg.get("captions.min_gap_s", DEFAULT_MIN_GAP_S)),
        location_seconds=float(cfg.get("captions.location_seconds", DEFAULT_LOCATION_SECONDS)),
        card_offset_s=float(cfg.get("captions.card_offset_s", DEFAULT_CARD_OFFSET_S)),
        min_segment_s=float(cfg.get("captions.min_segment_s", DEFAULT_MIN_SEGMENT_S)),
        skip_cold_open_s=float(cfg.get("captions.skip_cold_open_s", DEFAULT_SKIP_COLD_OPEN_S)),
        end_screen_s=float(cfg.get("captions.end_screen_s", DEFAULT_END_SCREEN_S)),
        include_cold_open=include_cold_open,
        exclude_clips=exclude_clips,
    )
    card_offset_s = float(cfg.get("captions.card_offset_s", DEFAULT_CARD_OFFSET_S))
    new_cards = [
        Caption(
            id="", at=r["at"], end=r["end"], text=r["label"], style="location",
            position="lower-left",
            anchor=CaptionAnchor(segment=r["segment"], offset=card_offset_s),
        )
        for r in rows
    ]

    kept: list[Caption] = []
    for cap in timeline.tracks.captions:
        if cap.style == "location":
            if keep_existing:
                kept.append(cap)
            continue
        if cap.style == "hook" and cap.anchor is None:
            seg_pos = timeline.segment_at(cap.at)
            if seg_pos is not None:
                cap.anchor = CaptionAnchor(
                    segment=seg_pos.segment.id, offset=round(cap.at - seg_pos.start, 3)
                )
            kept.append(cap)
            continue
        kept.append(cap)

    combined = sorted(kept + new_cards, key=lambda c: c.at)
    for index, cap in enumerate(combined, start=1):
        cap.id = f"t{index:03d}"
    timeline.tracks.captions = combined
    timeline.resolve_anchors()

    report_rows = [
        {
            "id": cap.id,
            "style": cap.style,
            "text": cap.text,
            "segment": cap.anchor.segment if cap.anchor else "",
            "at": cap.at,
            "end": cap.end,
        }
        for cap in sorted(timeline.tracks.captions, key=lambda c: c.at)
    ]
    return timeline, report_rows


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def write_captions_report(
    project: Project, rows: Sequence[dict[str, Any]], document: dict[str, Any]
) -> Path:
    """Write ``analysis/captions_report.md`` — time, segment, style, text."""
    lines = [
        f"# Captions report — {project.slug}",
        "",
        f"Generated: {utcnow()} · places model: `{document.get('model') or '—'}` · "
        f"places cost: ${float(document.get('cost_usd') or 0.0):.4f}",
        "",
        "| time | segment | style | text |",
        "|---|---|---|---|",
    ]
    for row in rows:
        text = str(row["text"]).replace("|", "\\|")
        lines.append(
            f"| {format_timecode(row['at'])} | {row['segment'] or '—'} | {row['style']} | {text} |"
        )
    content = "\n".join(lines) + "\n"
    path = project.analysis_dir / "captions_report.md"
    path.write_text(content, encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# the ``ytedit captions`` stage
# ----------------------------------------------------------------------
def run_captions_stage(
    project: Project,
    force: bool = False,
    include_cold_open: bool = False,
    keep_existing: bool = False,
    settings: Settings | None = None,
    client: OpenRouter | None = None,
    ask_writer: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate location cards for ``project`` and save them onto its timeline.

    Args:
        force: Re-ask the writer for places (ignoring the ``places.json``
            cache) AND overwrite a human-edited ``timeline.json`` directly
            instead of writing ``timeline.draft.json``.
        include_cold_open: Also allow a card during the opening
            ``captions.skip_cold_open_s`` seconds.
        keep_existing: Keep the planner's own ``location`` captions instead of
            dropping them.
        ask_writer: Test seam — replaces the OpenRouter call for the places
            normalization (see :func:`_ask_writer`).

    Returns:
        ``{"written", "backup", "edited_by_human", "issues", "report",
        "report_path", "cards_added", "places_cost_usd", "places_path"}``.

    Raises:
        LocationsError: No timeline or no footage log yet.
    """
    # Imported lazily: ai.tidy imports ai.overlay/ai.ledger which are heavier
    # than this stage needs on every import, and this avoids any import cycle
    # with ai.plan (already imported above at module level).
    from .tidy import backup_timeline

    cfg = settings or project.settings
    if not project.timeline_file.exists():
        raise LocationsError(
            f"no timeline at {project.timeline_file} — run `ytedit plan {project.slug}` first"
        )
    try:
        footage_log = load_footage_log(project)
    except PlanError as exc:
        raise LocationsError(str(exc)) from exc

    document = ensure_places(
        project, footage_log, settings=cfg, client=client, ask_writer=ask_writer, force=force
    )

    timeline = Timeline.load(project.timeline_file)
    human_edited = bool(timeline.meta.edited_by_human)

    # Clips recorded after the trip (home narration) have no place to card.
    home_clips = [
        str(c.get("id")) for c in project.clips_in_order()
        if "post recording" in str(c.get("source_file", "")).lower()
    ]
    timeline, report_rows = build_timeline_captions(
        timeline, document, settings=cfg,
        include_cold_open=include_cold_open, keep_existing=keep_existing,
        exclude_clips=home_clips,
    )
    cards_added = sum(1 for r in report_rows if r["style"] == "location")
    issues = timeline.validate(project)

    backup = backup_timeline(project)
    write_to_draft = human_edited and not force
    target = project.plan_dir / ("timeline.draft.json" if write_to_draft else "timeline.json")
    timeline.save(target)

    report_path = write_captions_report(project, report_rows, document)
    project.set_stage(STAGE, "done", cards=cards_added)

    return {
        "written": project.rel(target),
        "backup": backup,
        "edited_by_human": human_edited,
        "issues": issues,
        "report": report_rows,
        "report_path": project.rel(report_path),
        "cards_added": cards_added,
        "places_cost_usd": document.get("cost_usd", 0.0),
        "places_path": project.rel(places_path(project)),
    }


__all__ = [
    "DEFAULT_CARD_OFFSET_S",
    "DEFAULT_END_SCREEN_S",
    "DEFAULT_LOCATION_SECONDS",
    "DEFAULT_MIN_GAP_S",
    "DEFAULT_MIN_SEGMENT_S",
    "DEFAULT_SKIP_COLD_OPEN_S",
    "LOW_CONFIDENCE",
    "LocationsError",
    "build_places_document",
    "build_timeline_captions",
    "ensure_places",
    "generate_location_captions",
    "group_raw_locations",
    "places_by_clip",
    "places_path",
    "raw_locations",
    "run_captions_stage",
    "slugify",
    "write_captions_report",
]
