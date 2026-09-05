"""Tests for ``ytedit.project``, ``ytedit.config`` and ``ytedit.costs``."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ytedit.config import deep_merge, load_settings
from ytedit.costs import BudgetExceeded, charge, check_budget, estimate, remaining, summary
from ytedit.project import Project, ProjectError, validate_slug


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_defaults_load() -> None:
    cfg = load_settings()
    assert cfg.canvas == (1920, 1080, 30)
    assert cfg.model("planner") == "anthropic/claude-opus-5"
    assert cfg.model("vision") == "google/gemini-3.8-flash"
    assert cfg.get("audio.loudnorm.I") == -14
    assert cfg.encoding("master")["crf"] == 18
    assert cfg.encoding("master")["gop"] == 15


def test_deep_merge_replaces_lists_and_merges_dicts() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}, "l": [1, 2]}, {"a": {"c": 9}, "l": [3]})
    assert merged == {"a": {"b": 1, "c": 9}, "l": [3]}


def test_project_yaml_overrides_defaults(tmp_path: Path) -> None:
    project = Project.create("cfg", language="en", root=tmp_path / "projects")
    (project.path / "project.yaml").write_text(
        "language: en\nbudget_usd: 3.5\ncanvas:\n  fps: 24\n", encoding="utf-8"
    )
    cfg = load_settings(project.path)
    assert cfg.language == "en"
    assert cfg.budget_usd == 3.5
    assert cfg.canvas == (1920, 1080, 24)


def test_caption_font_exists_and_covers_polish() -> None:
    name, path = load_settings().caption_font()
    assert name
    assert path.exists(), f"caption font missing: {path}"


def test_music_styles_are_instrumental() -> None:
    styles = load_settings().music_styles
    assert len(styles["styles"]) == 8
    assert styles["defaults"]["force_instrumental"] is True
    for key, style in styles["styles"].items():
        assert "Instrumental." in style["prompt"], key
        assert "BPM" in style["prompt"], key


# ----------------------------------------------------------------------
# project
# ----------------------------------------------------------------------
def test_create_makes_the_full_tree(tmp_path: Path) -> None:
    project = Project.create("demo", language="pl", title="Demo", root=tmp_path / "projects")
    for directory in (
        project.input_dir, project.sources_dir, project.proxies_dir, project.audio_dir,
        project.peaks_dir, project.thumbs_dir, project.frames_dir, project.transcripts_dir,
        project.analysis_dir, project.plan_dir, project.music_dir, project.voice_dir,
        project.renders_dir, project.exports_dir, project.jobs_dir,
    ):
        assert directory.is_dir(), directory
    assert project.state_file.exists()
    assert project.title == "Demo"
    assert project.language == "pl"
    assert Project.list_projects(tmp_path / "projects") == ["demo"]


@pytest.mark.parametrize("slug", ["", "Bad Slug", "../escape", "UPPER"])
def test_bad_slugs_rejected(slug: str) -> None:
    with pytest.raises(ProjectError):
        validate_slug(slug)


def test_load_missing_project(tmp_path: Path) -> None:
    with pytest.raises(ProjectError):
        Project.load("nope", root=tmp_path)


def test_add_clip_merges_and_keeps_stages(project: Project) -> None:
    project.add_clip({"id": "c001", "order": 1, "duration": 4.0})
    project.set_clip_stage("c001", "ingest", "done")
    project.add_clip({"id": "c001", "duration": 5.0})
    clip = project.get_clip("c001")
    assert clip["duration"] == 5.0
    assert clip["order"] == 1
    assert clip["stages"]["ingest"] == "done"


def test_clips_in_order(project: Project) -> None:
    project.add_clip({"id": "c003", "order": 3})
    project.add_clip({"id": "c001", "order": 1})
    project.add_clip({"id": "c002", "order": 2})
    assert [c["id"] for c in project.clips_in_order()] == ["c001", "c002", "c003"]
    assert project.next_clip_id() == "c004"


def test_concurrent_state_writes_do_not_lose_updates(project: Project) -> None:
    def add(i: int) -> None:
        project.add_clip({"id": f"c{i:03d}", "order": i})

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(add, range(1, 25)))
    assert len(json.loads(project.state_file.read_text())["clips"]) == 24


def test_save_state_is_atomic(project: Project) -> None:
    project.save_state({"project": project.slug, "clips": {}, "costs": []})
    leftovers = list(project.path.glob(".state-*.json"))
    assert leftovers == []
    assert json.loads(project.state_file.read_text())["project"] == project.slug


def test_set_stage_records_timestamps(project: Project) -> None:
    project.set_stage("ingest", "running")
    project.set_stage("ingest", "done", clips=3)
    entry = project.load_state()["stages"]["ingest"]
    assert entry["status"] == "done"
    assert entry["started"] and entry["finished"]
    assert entry["clips"] == 3
    assert project.stage_status("transcribe") == "pending"


# ----------------------------------------------------------------------
# costs
# ----------------------------------------------------------------------
def test_estimate_elevenlabs() -> None:
    assert estimate("elevenlabs", "stt", 3600) == pytest.approx(0.22)
    assert estimate("elevenlabs", "music", 60) == pytest.approx(0.15)
    assert estimate("elevenlabs", "tts", 1000) == pytest.approx(0.10)
    assert estimate("elevenlabs", "isolation", 60) == pytest.approx(0.12)


def test_estimate_openrouter_uses_per_model_prices() -> None:
    usd = estimate("openrouter", "chat", {"in": 1_000_000, "out": 100_000},
                   model="anthropic/claude-opus-5")
    assert usd == pytest.approx(5.0 + 2.5)
    assert estimate("openrouter", "chat", {"in": 1000}, model="unknown/model") == 0.0


def test_estimate_fal() -> None:
    assert estimate("fal", "image", 4, model="fal-ai/nano-banana-pro/edit") == pytest.approx(0.60)


def test_charge_appends_to_the_ledger(project: Project) -> None:
    charge(project, "elevenlabs", "stt", 1800, clip="c001")
    charge(project, "openrouter", "chat", {"in": 10_000, "out": 2_000},
           model="anthropic/claude-sonnet-5", stage="analyze")
    ledger = project.load_state()["costs"]
    assert [e["service"] for e in ledger] == ["elevenlabs", "openrouter"]
    assert ledger[0]["units"] == "1800.0s"
    assert ledger[1]["model"] == "anthropic/claude-sonnet-5"
    spend = summary(project)
    assert spend["spent"] == pytest.approx(0.11 + 0.04)
    assert spend["remaining"] == pytest.approx(spend["budget"] - spend["spent"])


def test_budget_cap_blocks_overspend(project: Project) -> None:
    with project.edit_state() as state:
        state["budget_usd"] = 0.05
    charge(project, "elevenlabs", "stt", 360)  # $0.022
    assert remaining(project) == pytest.approx(0.028)
    with pytest.raises(BudgetExceeded):
        charge(project, "elevenlabs", "music", 600)  # $1.50
    assert len(project.load_state()["costs"]) == 1, "a refused charge is not recorded"


def test_check_budget_passes_under_the_cap(project: Project) -> None:
    check_budget(project, 1.0)
