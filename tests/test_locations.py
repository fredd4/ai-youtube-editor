"""Tests for ``ytedit.ai.locations``: places normalization + location cards by beat.

Everything here is offline: the writer call is replaced with a fixed JSON via
the ``ask_writer`` seam, and the ASS overlap check uses the project's real
font/styles (already exercised in ``tests/test_captions.py``) but no network.

Card placement is a pure function of a :class:`ytedit.cut.Cut` and the beats'
resolved spans, so most tests here build a cut plus a hand-written span map
instead of a project — the resolver has its own tests in ``test_cut.py``. Only
the stage tests need a real project on disk (see :func:`build_project`, which
mirrors ``tests/test_cut.py``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from ytedit.ai import locations as L
from ytedit.ai.sentences import sentences_path
from ytedit.config import Settings, load_settings
from ytedit.cut import Beat, Caption, Cut, Shot, cut_path, load_cut, save_cut
from ytedit.media.captions import build_ass
from ytedit.project import Project
from ytedit.timeline import Timeline


# ----------------------------------------------------------------------
# helpers: cuts and spans
# ----------------------------------------------------------------------
def broll(clip: str, **kwargs: Any) -> Beat:
    """A B-roll beat on ``clip`` (its length comes from the span map)."""
    return Beat(kind="broll", clip=clip, **{"in": 0.0}, out=10.0, **kwargs)


def voice(*clips: str, **kwargs: Any) -> Beat:
    """A narration beat whose picture is one shot per clip."""
    return Beat(
        kind="voice",
        file="voice/n001.wav",
        shots=[Shot(clip=c, **{"in": 0.0}, out=2.0) for c in clips],
        **kwargs,
    )


def cut_of(*beats: Beat) -> Cut:
    """A cut of ``beats`` with display ids already assigned."""
    cut = Cut(beats=list(beats))
    cut.renumber()
    return cut


def spans_of(cut: Cut, *lengths: float) -> dict[str, tuple[float, float]]:
    """Lay the cut's beats end to end, ``lengths[i]`` seconds each.

    A beat with no length left over is absent from the map — exactly what
    :func:`ytedit.cut.beat_spans` does for a beat that produced no picture.
    """
    out: dict[str, tuple[float, float]] = {}
    at = 0.0
    for beat, length in zip(cut.beats, lengths):
        out[beat.uid] = (at, at + length)
        at += length
    return out


def place(place_id: str, label: str | None = None, region: str = "") -> dict[str, Any]:
    return {"place_id": place_id, "label": label or place_id, "region": region}


PLACES = {
    "c001": place("a", "Place A"),
    "c002": place("b", "Place B"),
    "c003": place("c", "Place C"),
    "c004": place("d", "Place D"),
}


def cards(cut: Cut, spans: dict[str, tuple[float, float]], **kwargs: Any) -> list[dict[str, Any]]:
    """:func:`L.generate_location_cards` with the fixtures' usual defaults."""
    kwargs.setdefault("skip_cold_open_s", 0.0)
    kwargs.setdefault("end_screen_s", 0.0)
    return L.generate_location_cards(cut, spans, PLACES, **kwargs)


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
# a project the stage can actually run against (mirrors test_cut.py)
# ----------------------------------------------------------------------
CLIPS = ("c001", "c002", "c003", "c004")


def build_project(tmp_path: Path) -> Project:
    """A project of silent 60 s clips, one per place: no speech to resolve."""
    project = Project.create("t-loc", language="pl", root=tmp_path / "projects")
    for order, clip_id in enumerate(CLIPS, 1):
        project.add_clip({
            "id": clip_id, "order": order, "duration": 60.0,
            "width": 1920, "height": 1080, "orientation": "horizontal", "has_audio": True,
        })
        project.transcript_path(clip_id).write_text(
            json.dumps({"clip": clip_id, "language": "pl", "words": []}), encoding="utf-8"
        )
        project.analysis_path(clip_id).write_text(
            json.dumps({"clip": clip_id, "instructions": [], "takes": []}), encoding="utf-8"
        )
    sentences_path(project).write_text(
        json.dumps({
            "project": project.slug, "language": "pl", "clips_count": len(CLIPS),
            "sentences_count": 0,
            "clips": [{"id": c, "sentences": []} for c in CLIPS],
        }),
        encoding="utf-8",
    )
    return project


@pytest.fixture()
def loc_project(tmp_path: Path) -> Project:
    return build_project(tmp_path)


def write_cut(project: Project, cut: Cut) -> Cut:
    """Save ``cut`` as the project's ``plan/cut.json`` and read it back."""
    save_cut(cut, cut_path(project))
    return load_cut(cut_path(project))


def staged_cut(project: Project) -> Cut:
    """A four-beat cut, 10 s per beat, one clip (= one place) each."""
    cut = cut_of(*(Beat(kind="broll", clip=c, **{"in": 0.0}, out=10.0) for c in CLIPS))
    return write_cut(project, cut)


PLACES_ANSWER = {
    "places": [
        {"group_ids": [0], "place_id": "alfama", "label": "Alfama", "region": "Lisboa"},
        {"group_ids": [1], "place_id": "belem", "label": "Belém", "region": "Lisboa"},
        {"group_ids": [2], "place_id": "baixa", "label": "Baixa", "region": "Lisboa"},
        {"group_ids": [3], "place_id": "vila-douro", "label": "Vila d'Ouro", "region": "Lisboa"},
    ]
}


def write_places_footage_log(project: Project) -> None:
    write_footage_log(project, [
        clip_entry("c001", "Alfama", "Lisboa", "Portugal", 0.9),
        clip_entry("c002", "Belém", "Lisboa", "Portugal", 0.9),
        clip_entry("c003", "Baixa", "Lisboa", "Portugal", 0.9),
        clip_entry("c004", "Vila d'Ouro", "Lisboa", "Portugal", 0.9),
    ])


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
# which clip a beat's place comes from
# ----------------------------------------------------------------------
def test_a_beat_takes_its_place_from_its_main_clip() -> None:
    assert L.beat_clip(broll("c001")) == "c001"
    assert L.beat_clip(Beat(kind="speech", clip="c002", sentences=["c002#1"])) == "c002"
    # A voice beat is narration over borrowed picture: the first shot is what
    # the viewer actually sees when the card would appear.
    assert L.beat_clip(voice("c003", "c004")) == "c003"
    assert L.beat_clip(voice()) is None


# ----------------------------------------------------------------------
# card placement
# ----------------------------------------------------------------------
def test_a_new_place_gets_a_card_on_its_first_beat() -> None:
    cut = cut_of(broll("c001"), broll("c002"))
    rows = cards(cut, spans_of(cut, 10.0, 10.0), card_offset_s=0.3)
    # The very first eligible beat always cards (nothing shown yet), and the
    # change into c002's place cards too.
    assert [r["beat_id"] for r in rows] == ["b001", "b002"]
    assert [r["beat"] for r in rows] == [cut.beats[0].uid, cut.beats[1].uid]
    assert rows[0]["at"] == pytest.approx(0.3)
    assert rows[1]["at"] == pytest.approx(10.3)
    assert rows[1]["label"] == "Place B"


def test_a_new_place_mid_cut_gets_its_own_card() -> None:
    cut = cut_of(broll("c001"), broll("c001"), broll("c002"), broll("c002"))
    rows = cards(cut, spans_of(cut, 10.0, 10.0, 10.0, 10.0))
    # One card per place, each on the *first* beat at that place.
    assert [(r["beat_id"], r["place_id"]) for r in rows] == [("b001", "a"), ("b003", "b")]


def test_a_voice_beat_cards_the_place_of_its_first_shot() -> None:
    cut = cut_of(broll("c001"), voice("c002", "c003"))
    rows = cards(cut, spans_of(cut, 10.0, 10.0))
    assert [(r["beat_id"], r["label"]) for r in rows] == [("b001", "Place A"), ("b002", "Place B")]
    assert rows[1]["clip"] == "c002"


def test_place_shown_recently_is_suppressed_within_min_gap() -> None:
    cut = cut_of(broll("c001"), broll("c002"), broll("c001"))
    # back to A after only 20 s: suppressed (min_gap = 90 s)
    rows = cards(cut, spans_of(cut, 10.0, 10.0, 10.0), min_gap_s=90.0)
    assert [r["beat_id"] for r in rows] == ["b001", "b002"]


def test_place_shown_long_ago_cards_again_past_min_gap() -> None:
    cut = cut_of(broll("c001"), broll("c002"), broll("c001"))
    rows = cards(cut, spans_of(cut, 10.0, 190.0, 10.0), min_gap_s=90.0)
    assert [r["beat_id"] for r in rows] == ["b001", "b002", "b003"]


def test_a_short_beat_defers_to_the_next_beat_of_the_same_place() -> None:
    cut = cut_of(broll("c001"), broll("c002"), broll("c002"))
    rows = cards(cut, spans_of(cut, 10.0, 1.5, 4.5), min_segment_s=2.5)
    b_rows = [r for r in rows if r["place_id"] == "b"]
    assert len(b_rows) == 1
    assert b_rows[0]["beat_id"] == "b003"
    assert b_rows[0]["at"] == pytest.approx(11.5 + 0.3)


def test_a_short_beat_with_no_long_enough_followup_is_skipped() -> None:
    cut = cut_of(broll("c001"), broll("c002"), broll("c002"), broll("c003"))
    # Place B is on screen for 1.0 s + 1.5 s and then it's gone: no card at
    # all rather than a card flashed onto a 1 s cut.
    rows = cards(cut, spans_of(cut, 10.0, 1.0, 1.5, 7.5), min_segment_s=2.5)
    assert [r["place_id"] for r in rows] == ["a", "c"]


def test_the_cold_open_and_the_end_screen_are_never_carded() -> None:
    cut = cut_of(broll("c001"), broll("c002"), broll("c003"), broll("c004"))
    rows = L.generate_location_cards(
        cut, spans_of(cut, 5.0, 10.0, 80.0, 5.0), PLACES,
        skip_cold_open_s=5.0, end_screen_s=15.0,
    )
    # b001 is the cold open; b004 starts inside the last 15 s of a 100 s
    # programme. b002 cards even though it only differs from the cold open.
    assert [r["beat_id"] for r in rows] == ["b002", "b003"]


def test_include_cold_open_cards_the_opening_beat() -> None:
    cut = cut_of(broll("c001"), broll("c002"), broll("c003"), broll("c004"))
    rows = L.generate_location_cards(
        cut, spans_of(cut, 5.0, 10.0, 80.0, 5.0), PLACES,
        skip_cold_open_s=5.0, end_screen_s=15.0, include_cold_open=True,
    )
    assert [r["beat_id"] for r in rows] == ["b001", "b002", "b003"]


def test_a_clip_with_no_place_does_not_crash_and_breaks_continuity() -> None:
    cut = cut_of(broll("c001"), broll("c999"), broll("c001"))
    rows = cards(cut, spans_of(cut, 10.0, 90.0, 10.0), min_gap_s=90.0)
    assert [r["beat_id"] for r in rows] == ["b001", "b003"]


def test_an_excluded_clip_never_carries_a_place() -> None:
    cut = cut_of(broll("c001"), broll("c002"))
    rows = cards(cut, spans_of(cut, 10.0, 10.0), exclude_clips=["c002"])
    assert [r["beat_id"] for r in rows] == ["b001"]


def test_a_cutaway_beat_is_an_illustration_not_a_visit() -> None:
    cut = cut_of(broll("c001"), broll("c002", role=L.CUTAWAY_ROLE), broll("c001"))
    rows = cards(cut, spans_of(cut, 10.0, 10.0, 10.0), min_gap_s=90.0)
    # The cutaway gets no card of its own, and it does not count as leaving
    # place A, so returning to A is not a new arrival either.
    assert [r["beat_id"] for r in rows] == ["b001"]


def test_a_beat_that_produced_no_picture_is_skipped() -> None:
    cut = cut_of(broll("c001"), broll("c002"), broll("c003"))
    spans = spans_of(cut, 10.0, 10.0, 10.0)
    del spans[cut.beats[1].uid]          # b002 resolved to nothing
    rows = cards(cut, spans, min_gap_s=90.0)
    assert [r["beat_id"] for r in rows] == ["b001", "b003"]


# ----------------------------------------------------------------------
# applying to the cut
# ----------------------------------------------------------------------
def document_for(*clips: str) -> dict[str, Any]:
    return {"clips": [dict(PLACES[c], clip=c) for c in clips]}


def test_build_cut_captions_keeps_hooks_and_replaces_locations() -> None:
    cut = cut_of(broll("c001"), broll("c002"))
    hook_beat = cut.beats[0].uid
    cut.captions = [
        Caption(id="t001", beat=hook_beat, offset=1.0, duration=3.0, text="A HOOK", style="hook"),
        Caption(id="t002", beat=cut.beats[1].uid, offset=0.5, duration=2.0,
                text="stale location", style="location"),
    ]
    spans = spans_of(cut, 10.0, 10.0)
    cut, rows = L.build_cut_captions(cut, spans, document_for("c001", "c002"),
                                     settings=ZERO_WINDOW_SETTINGS)

    hook = next(c for c in cut.captions if c.text == "A HOOK")
    assert (hook.beat, hook.offset, hook.style) == (hook_beat, 1.0, "hook")
    assert "stale location" not in [c.text for c in cut.captions]
    assert {c.text for c in cut.captions if c.style == "location"} == {"Place A", "Place B"}

    # ids are sequential in screen order, and so are the report rows
    assert [c.id for c in cut.captions] == [f"t{i:03d}" for i in range(1, len(cut.captions) + 1)]
    assert [r["at"] for r in rows] == sorted(r["at"] for r in rows)
    card = next(r for r in rows if r["text"] == "Place B")
    assert (card["beat"], card["kind"], card["clip"]) == ("b002", "broll", "c002")
    assert card["where"] == "b002 broll c002"
    assert card["at"] == pytest.approx(10.3)


def test_a_card_takes_its_offset_and_duration_from_the_settings() -> None:
    cut = cut_of(broll("c001"))
    settings = Settings(data={"captions": {
        "skip_cold_open_s": 0.0, "end_screen_s": 0.0,
        "card_offset_s": 1.25, "location_seconds": 4.0,
    }})
    cut, rows = L.build_cut_captions(cut, spans_of(cut, 10.0), document_for("c001"),
                                     settings=settings)
    card = cut.captions[0]
    assert (card.offset, card.duration) == (1.25, 4.0)
    assert card.style == "location" and card.position == "lower-left"
    assert rows[0]["at"] == pytest.approx(1.25)
    assert rows[0]["end"] == pytest.approx(5.25)


def test_build_cut_captions_leaves_other_styles_alone() -> None:
    cut = cut_of(broll("c001"))
    cut.captions = [Caption(id="t001", beat=cut.beats[0].uid, offset=2.0, duration=1.0,
                            text="a note", style="subtitle")]
    cut, _rows = L.build_cut_captions(cut, spans_of(cut, 10.0), document_for("c001"),
                                      settings=ZERO_WINDOW_SETTINGS)
    note = next(c for c in cut.captions if c.style == "subtitle")
    assert (note.text, note.offset) == ("a note", 2.0)


def test_keep_existing_preserves_the_old_location_cards() -> None:
    cut = cut_of(broll("c001"))
    cut.captions = [Caption(id="t001", beat=cut.beats[0].uid, offset=1.0, duration=2.0,
                            text="old", style="location")]
    cut, _rows = L.build_cut_captions(cut, spans_of(cut, 10.0), document_for("c001"),
                                      settings=ZERO_WINDOW_SETTINGS, keep_existing=True)
    texts = [c.text for c in cut.captions]
    assert "old" in texts and "Place A" in texts


# ----------------------------------------------------------------------
# the stage: cut.json in, cut.json + timeline.json out
# ----------------------------------------------------------------------
def test_run_captions_stage_writes_the_cut_the_timeline_and_the_report(
    loc_project: Project,
) -> None:
    write_places_footage_log(loc_project)
    staged_cut(loc_project)

    result = L.run_captions_stage(
        loc_project, ask_writer=fake_ask_writer(PLACES_ANSWER), settings=ZERO_WINDOW_SETTINGS
    )

    assert result["written"] == "plan/cut.json"
    assert result["timeline"] == "plan/timeline.json"
    assert result["cards_added"] == 4
    assert result["backup"] is None or result["backup"].startswith("cut_")
    assert not [i for i in result["issues"] if i.severity == "error"]
    assert L.places_path(loc_project).exists()
    assert Path(loc_project.path / result["report_path"]).exists()

    saved = load_cut(cut_path(loc_project))
    labels = [c.text for c in saved.captions if c.style == "location"]
    assert labels == ["Alfama", "Belém", "Baixa", "Vila d'Ouro"]
    # every card names a beat of the cut, by uid, with no absolute time at all
    uids = {b.uid for b in saved.beats}
    assert all(c.beat in uids for c in saved.captions)
    assert all(not hasattr(c, "at") or c.at is None for c in saved.captions)

    timeline = Timeline.load(loc_project.timeline_file)
    resolved = [c for c in timeline.tracks.captions if c.style == "location"]
    assert [c.at for c in resolved] == [pytest.approx(x) for x in (0.3, 10.3, 20.3, 30.3)]

    report = (loc_project.analysis_dir / "captions_report.md").read_text(encoding="utf-8")
    assert "| time | beat | style | text |" in report
    assert "b002 broll c002" in report


def test_run_captions_stage_does_not_overwrite_a_human_edited_cut(
    loc_project: Project,
) -> None:
    write_places_footage_log(loc_project)
    cut = staged_cut(loc_project)
    cut.meta.edited_by_human = True
    save_cut(cut, cut_path(loc_project))

    result = L.run_captions_stage(
        loc_project, ask_writer=fake_ask_writer(PLACES_ANSWER), settings=ZERO_WINDOW_SETTINGS
    )
    assert result["edited_by_human"] is True
    assert result["written"] == "plan/cut.draft.json"
    assert result["timeline"] is None
    assert not loc_project.timeline_file.exists()

    assert not load_cut(cut_path(loc_project)).captions          # the real cut is untouched
    draft = load_cut(loc_project.plan_dir / "cut.draft.json")
    assert len([c for c in draft.captions if c.style == "location"]) == 4


def test_run_captions_stage_force_overwrites_a_human_edited_cut(
    loc_project: Project,
) -> None:
    write_places_footage_log(loc_project)
    cut = staged_cut(loc_project)
    cut.meta.edited_by_human = True
    save_cut(cut, cut_path(loc_project))

    result = L.run_captions_stage(
        loc_project, ask_writer=fake_ask_writer(PLACES_ANSWER),
        settings=ZERO_WINDOW_SETTINGS, force=True,
    )
    assert result["written"] == "plan/cut.json"
    assert len(load_cut(cut_path(loc_project)).captions) == 4


def test_run_captions_stage_needs_a_cut(loc_project: Project) -> None:
    write_places_footage_log(loc_project)
    with pytest.raises(L.LocationsError):
        L.run_captions_stage(loc_project, ask_writer=fake_ask_writer(PLACES_ANSWER))


# ----------------------------------------------------------------------
# no overlap in the burned-in ASS output
# ----------------------------------------------------------------------
_DIALOGUE_RE = re.compile(r"^Dialogue: \d+,([\d:.]+),([\d:.]+),(\S+),")


def _ass_seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def test_generated_cards_never_overlap_in_the_ass_output(loc_project: Project) -> None:
    # A hook line on the same beat as a generated location card: build_ass
    # must shift the later cue rather than let them overlap on screen.
    write_places_footage_log(loc_project)
    cut = staged_cut(loc_project)
    cut.captions = [
        Caption(id="t000", beat=cut.beats[1].uid, offset=0.1, duration=3.0,
                text="HOOK", style="hook"),
    ]
    save_cut(cut, cut_path(loc_project))
    L.run_captions_stage(
        loc_project, ask_writer=fake_ask_writer(PLACES_ANSWER), settings=ZERO_WINDOW_SETTINGS
    )

    timeline = Timeline.load(loc_project.timeline_file)
    settings = load_settings()
    ass_text = build_ass(
        timeline.tracks.captions, timeline.width, timeline.height,
        settings.caption_styles, settings.caption_font(), duration=timeline.duration(),
    )
    cues = []
    for line in ass_text.splitlines():
        m = _DIALOGUE_RE.match(line)
        if m:
            cues.append((_ass_seconds(m.group(1)), _ass_seconds(m.group(2)), m.group(3)))
    cues.sort()
    assert len(cues) >= 2
    for (_a_start, a_end, a_style), (b_start, _b_end, b_style) in zip(cues, cues[1:]):
        if a_style == "subtitle" or b_style == "subtitle":
            continue
        assert b_start >= a_end - 1e-6, f"{a_style}@{a_end} overlaps {b_style}@{b_start}"
