"""Tests for ``ytedit.ai.locations``: places normalization + anchored location cards.

Everything here is offline: the writer call is replaced with a fixed JSON via
the ``ask_writer`` seam, and the ASS overlap check uses the project's real
font/styles (already exercised in ``tests/test_captions.py``) but no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from ytedit.ai import locations as L
from ytedit.config import Settings, load_settings
from ytedit.media.captions import build_ass
from ytedit.project import Project
from ytedit.timeline import (
    Caption,
    CaptionAnchor,
    Timeline,
    Tracks,
    VideoSegment,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def seg(seg_id: str, clip: str, start: float, end: float, **extra: Any) -> VideoSegment:
    return VideoSegment(id=seg_id, clip=clip, **{"in": start}, out=end, **extra)


def timeline_of(*segments: VideoSegment, **kw: Any) -> Timeline:
    return Timeline(tracks=Tracks(video=list(segments)), **kw)


def place(place_id: str, label: str | None = None, region: str = "") -> dict[str, Any]:
    return {"place_id": place_id, "label": label or place_id, "region": region}


def write_footage_log(project: Project, clips: list[dict[str, Any]]) -> None:
    (project.analysis_dir / "footage_log.json").write_text(
        json.dumps({"project": project.slug, "language": "pl", "clips": clips}, ensure_ascii=False),
        encoding="utf-8",
    )


def clip_entry(
    clip_id: str, name: str = "", city: str = "", country: str = "", confidence: float = 0.0
) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": clip_id, "clip": clip_id}
    if name or confidence:
        entry["location"] = {"name": name, "city": city, "country": country, "confidence": confidence}
    return entry


#: Zero-width cold-open/end-screen exclusion, for fixtures too short for the
#: real 10s/15s project defaults to leave any eligible window at all.
ZERO_WINDOW_SETTINGS = Settings(data={"captions": {"skip_cold_open_s": 0.0, "end_screen_s": 0.0}})


def fake_ask_writer(answer: dict[str, Any]) -> Any:
    """An ``ask_writer`` replacement returning a fixed JSON, no HTTP call."""

    def _asker(project: Project, settings: Any, groups: Any, client: Any = None) -> dict[str, Any]:
        return {"json": answer, "model": "test-writer", "cost_usd": 0.0}

    return _asker


# ----------------------------------------------------------------------
# raw locations + grouping
# ----------------------------------------------------------------------
def test_raw_locations_reads_footage_log_in_order(project: Project) -> None:
    write_footage_log(
        project,
        [
            clip_entry("c001", "Alfama", "Lisboa", "Portugal", 0.85),
            clip_entry("c002"),  # no location block at all
            clip_entry("c003", "Belém (prawdopodobnie)", confidence=0.4),
        ],
    )
    footage_log = L.load_footage_log(project)
    raws = L.raw_locations(footage_log)
    assert [r["clip"] for r in raws] == ["c001", "c002", "c003"]
    assert raws[0]["order"] == 0 and raws[0]["confidence"] == pytest.approx(0.85)
    assert raws[1]["name"] == "" and raws[1]["confidence"] == 0.0
    assert raws[2]["name"] == "Belém (prawdopodobnie)"


def test_group_raw_locations_collapses_exact_duplicates_only() -> None:
    raws = [
        {"clip": "c001", "order": 0, "name": "Alfama", "city": "Lisboa", "country": "Portugal", "confidence": 0.9},
        {"clip": "c002", "order": 1, "name": "Alfama", "city": "Lisboa", "country": "Portugal", "confidence": 0.8},
        {"clip": "c003", "order": 2, "name": "Alfama (prawdopodobnie)", "city": "", "country": "", "confidence": 0.3},
        {"clip": "c004", "order": 3, "name": "", "city": "", "country": "", "confidence": 0.0},
    ]
    groups = L.group_raw_locations(raws)
    # c001/c002 share the exact same string -> one group; the hedged variant
    # is a different exact string, so it's a separate group (merging near-
    # duplicates like this is the writer model's job, not this pass's).
    assert len(groups) == 2
    by_id = {g["group_id"]: g for g in groups}
    assert by_id[0]["clips"] == ["c001", "c002"]
    assert by_id[1]["clips"] == ["c003"]


def test_needs_inheritance_only_for_no_name_and_low_confidence() -> None:
    assert L._needs_inheritance({"name": "", "confidence": 0.0})
    assert L._needs_inheritance({"name": "", "confidence": 0.49})
    assert not L._needs_inheritance({"name": "", "confidence": 0.5})
    # A hedged but *named* low-confidence location is normalized, not inherited.
    assert not L._needs_inheritance({"name": "Belém (prawdopodobnie)", "confidence": 0.2})


# ----------------------------------------------------------------------
# places.json construction (writer call replaced by a fixed JSON)
# ----------------------------------------------------------------------
def test_build_places_document_normalizes_and_inherits(project: Project) -> None:
    write_footage_log(
        project,
        [
            clip_entry("c000", "Alfama", "Lisboa", "Portugal", 0.9),   # named -> group 0
            clip_entry("c001"),                                          # no location -> inherit c000
            clip_entry("c002"),                                          # no location -> inherit c003
            clip_entry("c003", "Belém", "Lisboa", "Portugal", 0.9),    # named -> group 1
        ],
    )
    footage_log = L.load_footage_log(project)
    answer = {
        "places": [
            {"group_ids": [0], "place_id": "alfama", "label": "Alfama · Lisboa", "region": "Lisboa"},
            {"group_ids": [1], "place_id": "belem", "label": "Belém · Lisboa", "region": "Lisboa"},
        ]
    }
    document = L.build_places_document(
        project, footage_log, settings=project.settings, ask_writer=fake_ask_writer(answer)
    )
    by_clip = {c["clip"]: c for c in document["clips"]}
    assert by_clip["c000"]["place_id"] == "alfama"
    assert by_clip["c000"]["inherited"] is False
    assert by_clip["c003"]["place_id"] == "belem"
    # c001 is next to c000 (distance 1) vs c003 (distance 2) -> inherits c000's place.
    assert by_clip["c001"]["place_id"] == "alfama"
    assert by_clip["c001"]["inherited"] is True
    # c002 is next to c003 (distance 1) vs c000 (distance 2) -> inherits c003's place.
    assert by_clip["c002"]["place_id"] == "belem"
    assert by_clip["c002"]["inherited"] is True
    assert document["model"] == "test-writer"


def test_build_places_document_falls_back_when_writer_skips_a_group(project: Project) -> None:
    write_footage_log(project, [clip_entry("c001", "Baixa", "Lisboa", "Portugal", 0.9)])
    footage_log = L.load_footage_log(project)
    document = L.build_places_document(
        project, footage_log, settings=project.settings, ask_writer=fake_ask_writer({"places": []})
    )
    entry = document["clips"][0]
    assert entry["place_id"] == "baixa"  # naive slugify fallback
    assert entry["label"] == "Baixa"


def test_ensure_places_caches_and_force_reasks(project: Project) -> None:
    write_footage_log(project, [clip_entry("c001", "Baixa", "Lisboa", "Portugal", 0.9)])
    footage_log = L.load_footage_log(project)
    calls = {"n": 0}

    def counting_asker(project: Project, settings: Any, groups: Any, client: Any = None) -> dict[str, Any]:
        calls["n"] += 1
        return {
            "json": {"places": [{"group_ids": [0], "place_id": "baixa", "label": "Baixa", "region": "Lisboa"}]},
            "model": "test-writer", "cost_usd": 0.001,
        }

    doc1 = L.ensure_places(project, footage_log, settings=project.settings, ask_writer=counting_asker)
    assert calls["n"] == 1
    assert L.places_path(project).exists()

    doc2 = L.ensure_places(project, footage_log, settings=project.settings, ask_writer=counting_asker)
    assert calls["n"] == 1  # cached, no re-ask
    assert doc2 == doc1

    L.ensure_places(project, footage_log, settings=project.settings, ask_writer=counting_asker, force=True)
    assert calls["n"] == 2  # --force re-asks


# ----------------------------------------------------------------------
# card placement
# ----------------------------------------------------------------------
def test_basic_place_change_emits_an_anchored_card() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),
        seg("s002", "c002", 10.0, 20.0),
    )
    places = {"c001": place("a", "Place A"), "c002": place("b", "Place B")}
    rows = L.generate_location_captions(
        tl, places, skip_cold_open_s=0.0, end_screen_s=0.0, card_offset_s=0.3,
    )
    # The very first eligible segment always cards (nothing shown yet), and
    # the change into c002's place cards too.
    assert [r["segment"] for r in rows] == ["s001", "s002"]
    assert rows[0]["at"] == pytest.approx(0.3)
    assert rows[1]["at"] == pytest.approx(10.3)
    assert rows[1]["label"] == "Place B"


def test_place_shown_recently_is_suppressed_within_min_gap() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),   # place A cards at 0.3
        seg("s002", "c002", 10.0, 20.0),  # place B cards at 10.3
        seg("s003", "c001", 20.0, 30.0),  # back to A after only 10s: suppressed (min_gap=90)
    )
    places = {"c001": place("a", "Place A"), "c002": place("b", "Place B")}
    rows = L.generate_location_captions(
        tl, places, skip_cold_open_s=0.0, end_screen_s=0.0, min_gap_s=90.0,
    )
    assert [r["segment"] for r in rows] == ["s001", "s002"]


def test_place_shown_long_ago_cards_again_past_min_gap() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),
        seg("s002", "c002", 10.0, 200.0),
        seg("s003", "c001", 200.0, 210.0),  # 200s after A's card: past min_gap
    )
    places = {"c001": place("a", "Place A"), "c002": place("b", "Place B")}
    rows = L.generate_location_captions(
        tl, places, skip_cold_open_s=0.0, end_screen_s=0.0, min_gap_s=90.0,
    )
    assert [r["segment"] for r in rows] == ["s001", "s002", "s003"]


def test_short_segment_defers_to_the_next_segment_of_the_same_place() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),
        seg("s002", "c002", 10.0, 11.5),  # place change, but only 1.5s: too short
        seg("s003", "c002", 11.5, 16.0),  # still place B, 4.5s: long enough
    )
    places = {"c001": place("a", "Place A"), "c002": place("b", "Place B")}
    rows = L.generate_location_captions(
        tl, places, skip_cold_open_s=0.0, end_screen_s=0.0, min_segment_s=2.5,
    )
    b_rows = [r for r in rows if r["place_id"] == "b"]
    assert len(b_rows) == 1
    assert b_rows[0]["segment"] == "s003"
    assert b_rows[0]["at"] == pytest.approx(11.5 + 0.3)


def test_short_segment_with_no_long_enough_followup_is_skipped() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),
        seg("s002", "c002", 10.0, 11.0),  # place B, short
        seg("s003", "c002", 11.0, 12.5),  # still B, still short (1.5s)
        seg("s004", "c003", 12.5, 20.0),  # place changes to C before B ever got long enough
    )
    places = {
        "c001": place("a", "Place A"), "c002": place("b", "Place B"), "c003": place("c", "Place C"),
    }
    rows = L.generate_location_captions(
        tl, places, skip_cold_open_s=0.0, end_screen_s=0.0, min_segment_s=2.5,
    )
    assert "b" not in [r["place_id"] for r in rows]
    assert [r["place_id"] for r in rows] == ["a", "c"]


def test_cold_open_and_end_screen_are_never_carded() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 5.0),    # cold open: place A, no card
        seg("s002", "c002", 5.0, 15.0),   # first eligible segment: cards even though
                                           # it's a genuinely new place vs. the cold open
        seg("s003", "c003", 15.0, 95.0),  # place C, eligible, cards
        seg("s004", "c004", 95.0, 100.0),  # inside the last 15s of a 100s programme: no card
    )
    places = {
        "c001": place("a", "Place A"), "c002": place("b", "Place B"),
        "c003": place("c", "Place C"), "c004": place("d", "Place D"),
    }
    rows = L.generate_location_captions(
        tl, places, skip_cold_open_s=5.0, end_screen_s=15.0,
    )
    assert [r["segment"] for r in rows] == ["s002", "s003"]


def test_unassigned_clip_does_not_crash_and_breaks_continuity() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),
        seg("s002", "c999", 10.0, 100.0),  # no place assigned at all
        seg("s003", "c001", 100.0, 110.0),  # back to A well past min_gap (90s)
    )
    places = {"c001": place("a", "Place A")}
    rows = L.generate_location_captions(
        tl, places, skip_cold_open_s=0.0, end_screen_s=0.0, min_gap_s=90.0,
    )
    assert [r["segment"] for r in rows] == ["s001", "s003"]


# ----------------------------------------------------------------------
# applying to the timeline: hook anchoring + location replacement
# ----------------------------------------------------------------------
def test_build_timeline_captions_anchors_hooks_and_replaces_locations() -> None:
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),
        seg("s002", "c002", 10.0, 20.0),
    )
    tl.tracks.captions = [
        Caption(id="t001", at=1.0, end=4.0, text="A HOOK", style="hook"),
        Caption(id="t002", at=11.0, end=13.0, text="stale location", style="location"),
    ]
    places = {"c001": place("a", "Place A"), "c002": place("b", "Place B")}
    document = {"clips": [
        {"clip": "c001", "place_id": "a", "label": "Place A", "region": ""},
        {"clip": "c002", "place_id": "b", "label": "Place B", "region": ""},
    ]}
    tl, rows = L.build_timeline_captions(
        tl, document, settings=ZERO_WINDOW_SETTINGS,
    )
    # planner's hook survives and is now anchored to s001, offset 1.0 - 0 = 1.0
    hook = next(c for c in tl.tracks.captions if c.text == "A HOOK")
    assert hook.anchor is not None
    assert hook.anchor.segment == "s001"
    assert hook.anchor.offset == pytest.approx(1.0)
    assert hook.at == pytest.approx(1.0)

    # planner's stale location caption is gone, replaced by generated cards
    assert "stale location" not in [c.text for c in tl.tracks.captions]
    labels = {c.text for c in tl.tracks.captions if c.style == "location"}
    assert labels == {"Place A", "Place B"}
    for c in tl.tracks.captions:
        if c.style == "location":
            assert c.anchor is not None

    # ids are sequential and report rows are in time order
    assert [c.id for c in tl.tracks.captions] == [f"t{i:03d}" for i in range(1, len(tl.tracks.captions) + 1)]
    ats = [r["at"] for r in rows]
    assert ats == sorted(ats)


def test_build_timeline_captions_keep_existing_preserves_old_location_cards() -> None:
    tl = timeline_of(seg("s001", "c001", 0.0, 10.0))
    tl.tracks.captions = [Caption(id="t001", at=1.0, end=2.0, text="old", style="location")]
    document = {"clips": [{"clip": "c001", "place_id": "a", "label": "Place A", "region": ""}]}
    tl, _rows = L.build_timeline_captions(
        tl, document, settings=ZERO_WINDOW_SETTINGS, keep_existing=True
    )
    assert "old" in [c.text for c in tl.tracks.captions]
    assert "Place A" in [c.text for c in tl.tracks.captions]


# ----------------------------------------------------------------------
# end-to-end: run_captions_stage against a real project + real ASS writer
# ----------------------------------------------------------------------
def test_run_captions_stage_writes_timeline_and_report(project: Project) -> None:
    write_footage_log(
        project,
        [
            clip_entry("c001", "Alfama", "Lisboa", "Portugal", 0.9),
            clip_entry("c002", "Belém", "Lisboa", "Portugal", 0.9),
        ],
    )
    tl = timeline_of(seg("s001", "c001", 0.0, 10.0), seg("s002", "c002", 10.0, 20.0))
    tl.tracks.captions = [Caption(id="t001", at=1.0, end=4.0, text="HOOK", style="hook")]
    tl.save(project.timeline_file)

    answer = {
        "places": [
            {"group_ids": [0], "place_id": "alfama", "label": "Alfama", "region": "Lisboa"},
            {"group_ids": [1], "place_id": "belem", "label": "Belém", "region": "Lisboa"},
        ]
    }
    # Override the (project-wide default) 10s cold-open / 15s end-screen
    # exclusion windows -- this fixture's programme is only 20s long, so the
    # defaults would leave no eligible window at all.
    result = L.run_captions_stage(
        project, ask_writer=fake_ask_writer(answer), settings=ZERO_WINDOW_SETTINGS
    )

    assert result["cards_added"] == 2
    assert Path(project.path / result["written"]).exists()
    assert Path(project.path / result["report_path"]).exists()
    assert L.places_path(project).exists()

    saved = Timeline.load(project.timeline_file)
    locations = [c for c in saved.tracks.captions if c.style == "location"]
    assert {c.text for c in locations} == {"Alfama", "Belém"}
    assert all(c.anchor is not None for c in locations)


def test_run_captions_stage_needs_a_timeline(project: Project) -> None:
    write_footage_log(project, [clip_entry("c001", "Baixa", "Lisboa", "Portugal", 0.9)])
    with pytest.raises(L.LocationsError):
        L.run_captions_stage(project, ask_writer=fake_ask_writer({"places": []}))


# ----------------------------------------------------------------------
# no overlap in the burned-in ASS output
# ----------------------------------------------------------------------
_DIALOGUE_RE = re.compile(r"^Dialogue: \d+,([\d:.]+),([\d:.]+),(\S+),")


def _ass_seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def test_generated_cards_never_overlap_in_the_ass_output() -> None:
    # A hook line anchored right where a location card would also land: build_ass
    # must shift the later cue rather than let them overlap on screen.
    tl = timeline_of(
        seg("s001", "c001", 0.0, 10.0),
        seg("s002", "c002", 10.0, 20.0),
    )
    tl.tracks.captions = [
        # Anchored 0.1s into s002, 3s long -- overlaps the generated location
        # card at s002+0.3s (2.4s long) once both are resolved to absolute time.
        Caption(id="t000", at=0.0, end=3.0, text="HOOK", style="hook",
                anchor=CaptionAnchor(segment="s002", offset=0.1)),
    ]
    document = {"clips": [
        {"clip": "c001", "place_id": "a", "label": "Place A", "region": ""},
        {"clip": "c002", "place_id": "b", "label": "Place B", "region": ""},
    ]}
    tl, _rows = L.build_timeline_captions(tl, document, settings=ZERO_WINDOW_SETTINGS)

    settings = load_settings()
    ass_text = build_ass(
        tl.tracks.captions, tl.width, tl.height, settings.caption_styles, settings.caption_font(),
        duration=tl.duration(),
    )
    cues = []
    for line in ass_text.splitlines():
        m = _DIALOGUE_RE.match(line)
        if m:
            cues.append((_ass_seconds(m.group(1)), _ass_seconds(m.group(2)), m.group(3)))
    cues.sort()
    assert len(cues) >= 2
    for (a_start, a_end, a_style), (b_start, b_end, b_style) in zip(cues, cues[1:]):
        if a_style == "subtitle" or b_style == "subtitle":
            continue
        assert b_start >= a_end - 1e-6, f"{a_style}@{a_end} overlaps {b_style}@{b_start}"
