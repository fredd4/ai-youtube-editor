"""Tests for the local web editor (``server.app`` + ``server.jobs``).

Everything runs against a temporary projects root, so no test ever touches a
real project. Media files are empty stubs — the API only cares that they exist
and reports URLs for them; the bytes are Starlette's problem.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.jobs import JobManager
from ytedit.project import Project
from ytedit.timeline import Timeline, new_timeline

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _touch(path: Path, payload: str = "") -> Path:
    """Create a file (and its parents) with optional content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """An empty projects root."""
    d = tmp_path / "projects"
    d.mkdir()
    return d


@pytest.fixture()
def demo(root: Path) -> Project:
    """A project with one fully-ingested clip and one bare clip."""
    project = Project.create("demo", language="pl", title="Demo", root=root)
    project.ensure_dirs()
    for clip_id, extras in (("c001", True), ("c002", False)):
        project.add_clip({
            "id": clip_id,
            "source_file": f"input/{clip_id}.mp4",
            "duration": 6.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "orientation": "horizontal",
            "has_audio": True,
        })
        _touch(project.proxy_path(clip_id))
        if extras:
            _touch(project.poster_path(clip_id))
            _touch(project.peaks_path(clip_id), json.dumps(
                {"duration": 6.0, "peaks": [-0.5, 0.5, -0.4, 0.4]}))
            _touch(project.transcript_path(clip_id), json.dumps({
                "language": "pl", "language_mismatch": False,
                "words": [{"t": "cześć", "s": 0.1, "e": 0.6}],
            }))
            _touch(project.analysis_path(clip_id), json.dumps({
                "kind": "a-roll",
                "summary": "Test",
                "instructions": [{"s": 1.0, "e": 1.5, "text": "jeszcze raz"}],
                "takes": [{"topic": "intro", "attempts": [{"s": 0, "e": 1}, {"s": 2, "e": 3}],
                           "keep": 1}],
                "background_music": [{"s": 4.0, "e": 5.0, "suggest": "mute"}],
                "visual": {"thumbnail_candidate": True},
            }))
    return project


@pytest.fixture()
def client(root: Path, demo: Project, tmp_path: Path) -> TestClient:
    """A TestClient whose job manager runs a harmless stand-in for ``ytedit``."""
    script = tmp_path / "fake-ytedit"
    script.write_text("#!/bin/sh\necho \"ran: $*\"\nexit 0\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    manager = JobManager(executable=script, cwd=tmp_path)
    return TestClient(create_app(root, manager=manager))


def _timeline_payload(clip: str = "c001") -> dict[str, Any]:
    """A minimal valid timeline document."""
    tl = new_timeline()
    tl.tracks.video.append(Timeline.model_validate({
        "tracks": {"video": [{"id": "s001", "clip": clip, "in": 0.0, "out": 3.0}]}
    }).tracks.video[0])
    return tl.to_dict()


def _multi_segment_timeline() -> dict[str, Any]:
    """Three segments, the last one overlapping via an xfade, plus a mute range."""
    return Timeline.model_validate({
        "tracks": {"video": [
            {"id": "s001", "clip": "c001", "in": 0.0, "out": 2.0},
            {"id": "s002", "clip": "c002", "in": 1.0, "out": 3.0, "mute_source": True},
            {"id": "s003", "clip": "c001", "in": 3.0, "out": 5.0,
             "transition_in": {"type": "xfade", "duration": 0.5}},
        ]},
        "mute_ranges": [{"clip": "c001", "s": 0.5, "e": 1.5, "gain_db": -60.0,
                         "reason": "bar music"}],
    }).to_dict()


# ----------------------------------------------------------------------
# projects + pages
# ----------------------------------------------------------------------
def test_project_list(client: TestClient) -> None:
    """``/api/projects`` reports slug, clip count, stages and spend."""
    body = client.get("/api/projects").json()
    assert [p["slug"] for p in body["projects"]] == ["demo"]
    entry = body["projects"][0]
    assert entry["title"] == "Demo"
    assert entry["language"] == "pl"
    assert entry["clips"] == 2
    assert entry["stages"]["ingest"]["status"] in ("pending", "done")
    assert entry["costs"]["budget"] > 0
    assert entry["costs"]["spent"] == 0


def test_create_project(client: TestClient, root: Path) -> None:
    """``POST /api/projects`` scaffolds a new tree; bad slugs are rejected."""
    res = client.post("/api/projects", json={"slug": "lisbon", "language": "en", "title": "L"})
    assert res.status_code == 201, res.text
    assert (root / "lisbon" / "project.yaml").exists()
    assert client.post("/api/projects", json={"slug": "lisbon"}).status_code == 409
    assert client.post("/api/projects", json={"slug": "../evil"}).status_code == 400


def test_pages_served(client: TestClient) -> None:
    """The SPA shell is served for both routes; the slug is still validated."""
    assert "<div id=\"app\">" in client.get("/").text
    assert client.get("/p/demo").status_code == 200
    assert client.get("/p/..%2Fetc").status_code in (400, 404)


# ----------------------------------------------------------------------
# state
# ----------------------------------------------------------------------
def test_state_enriches_clips(client: TestClient) -> None:
    """Each clip carries media URLs, mtimes, artefact flags and badge data."""
    body = client.get("/api/p/demo/state").json()
    assert body["language"] == "pl"
    first, second = body["clips"]
    assert first["proxy_url"].startswith("/media/demo/proxies/c001.mp4?v=")
    assert first["poster_url"] and first["peaks_url"]
    assert first["has_transcript"] and first["has_analysis"]
    assert first["mtimes"]["proxy"] > 0
    assert first["flags"] == {
        "kind": "a-roll", "instructions": 1, "takes": 1, "background_music": 1,
        "language_mismatch": False, "summary": "Test", "location": "",
        "thumbnail_candidate": True, "transcript_language": "pl",
    }
    assert second["poster_url"] is None
    assert second["has_transcript"] is False
    assert set(body["stages"]) >= {"ingest", "transcribe", "render_preview", "render_master"}
    assert body["costs"]["budget"] > 0
    assert body["files"]["timeline"] is False


def test_state_unknown_project(client: TestClient) -> None:
    """An unknown slug is a 404, a malformed one a 400."""
    assert client.get("/api/p/nope/state").status_code == 404
    assert client.get("/api/p/Bad Slug/state").status_code == 400


def test_next_step_state_machine(client: TestClient, demo: Project) -> None:
    """The hint walks the pipeline: transcribe -> analyze -> plan -> preview."""
    def hint() -> str:
        return client.get("/api/p/demo/state").json()["next_step"]

    demo.set_stage("ingest", "done")
    assert "Transcribe" in hint()

    demo.set_stage("transcribe", "done")
    assert "Analyze" in hint()

    demo.set_stage("analyze", "done")
    assert "Plan" in hint()

    client.put("/api/p/demo/timeline", json=_timeline_payload())
    assert "Render preview" in hint()

    _touch(demo.renders_dir / "preview.mp4", "x")
    os.utime(demo.renders_dir / "preview.mp4", (time.time() + 10, time.time() + 10))
    assert "Render master" in hint()


def test_next_step_without_clips(root: Path) -> None:
    """A brand-new project asks for footage first."""
    Project.create("fresh", root=root)
    client = TestClient(create_app(root))
    assert client.get("/api/p/fresh/state").json()["next_step"] == (
        "Drop clips into input/ and run Ingest.")


# ----------------------------------------------------------------------
# per-clip documents
# ----------------------------------------------------------------------
def test_clip_documents(client: TestClient, demo: Project) -> None:
    """Transcript / analysis / footage log endpoints, present and missing."""
    assert client.get("/api/p/demo/transcript/c001").json()["language"] == "pl"
    assert client.get("/api/p/demo/analysis/c001").json()["kind"] == "a-roll"
    assert client.get("/api/p/demo/transcript/c002").status_code == 404
    assert client.get("/api/p/demo/footage_log").status_code == 404

    _touch(demo.analysis_dir / "footage_log.json", json.dumps({"clips": []}))
    assert client.get("/api/p/demo/footage_log").json() == {"clips": []}


def test_plan_qc_publish_endpoints(client: TestClient, demo: Project) -> None:
    """The markdown-backed tabs return their documents once the files exist."""
    assert client.get("/api/p/demo/plan").status_code == 404
    _touch(demo.edit_plan_file, json.dumps({"structure": []}))
    _touch(demo.plan_dir / "edit_plan.md", "# Plan")
    _touch(demo.plan_dir / "narration_requests.md", "# Narration")
    plan = client.get("/api/p/demo/plan").json()
    assert plan["edit_plan"] == {"structure": []}
    assert plan["edit_plan_md"] == "# Plan"
    assert plan["narration_requests_md"] == "# Narration"

    assert client.get("/api/p/demo/qc").status_code == 404
    _touch(demo.exports_dir / "qc_report.md", "# QC")
    assert client.get("/api/p/demo/qc").json()["markdown"] == "# QC"

    assert client.get("/api/p/demo/publish").status_code == 404
    _touch(demo.exports_dir / "publish.json", json.dumps({"titles": ["A"]}))
    _touch(demo.exports_dir / "thumbnails" / "thumb_a.jpg")
    _touch(demo.exports_dir / "thumbnails" / "thumb_a_preview_120px.jpg")
    pub = client.get("/api/p/demo/publish").json()
    assert pub["publish"]["titles"] == ["A"]
    assert [t["name"] for t in pub["thumbnails"]] == [
        "thumb_a.jpg", "thumb_a_preview_120px.jpg"]
    assert [t["preview"] for t in pub["thumbnails"]] == [False, True]


def test_music_listing(client: TestClient, demo: Project) -> None:
    """Generated beds are offered to the cue editor with their sidecar data."""
    _touch(demo.music_dir / "bed_a.mp3", "x")
    _touch(demo.music_dir / "bed_a.json", json.dumps({"duration": 90, "prompt": "calm"}))
    music = client.get("/api/p/demo/music").json()["music"]
    assert music[0]["file"] == "music/bed_a.mp3"
    assert music[0]["url"].startswith("/music/demo/bed_a.mp3?v=")
    assert music[0]["duration"] == 90


# ----------------------------------------------------------------------
# timeline
# ----------------------------------------------------------------------
def test_timeline_put_validation_error(client: TestClient, demo: Project) -> None:
    """A schema break is 422; a semantic issue is 400 until ``force``."""
    bad_schema = _timeline_payload()
    bad_schema["tracks"]["video"][0]["out"] = "not a number"
    assert client.put("/api/p/demo/timeline", json=bad_schema).status_code == 422
    assert not demo.timeline_file.exists()

    bad_clip = _timeline_payload(clip="c999")
    res = client.put("/api/p/demo/timeline", json=bad_clip)
    assert res.status_code == 400
    assert "missing clip" in json.dumps(res.json())
    assert not demo.timeline_file.exists()

    forced = client.put("/api/p/demo/timeline?force=true", json=bad_clip)
    assert forced.status_code == 200
    assert forced.json()["issues"]
    assert demo.timeline_file.exists()


def test_timeline_put_marks_human_and_backs_up(client: TestClient, demo: Project) -> None:
    """Saving stamps ``edited_by_human`` and keeps a rolling backup."""
    first = client.put("/api/p/demo/timeline", json=_timeline_payload())
    assert first.status_code == 200, first.text
    assert first.json()["backup"] is None          # nothing to back up yet
    assert first.json()["duration"] == pytest.approx(3.0)
    assert first.json()["has_draft"] is False
    saved = json.loads(demo.timeline_file.read_text(encoding="utf-8"))
    assert saved["meta"]["edited_by_human"] is True

    second = client.put("/api/p/demo/timeline", json=_timeline_payload())
    assert second.json()["backup"].startswith("timeline_")
    assert list((demo.plan_dir / "history").glob("timeline_*.json"))

    got = client.get("/api/p/demo/timeline").json()
    assert got["timeline"]["tracks"]["video"][0]["clip"] == "c001"
    assert got["issues"] == []
    assert got["has_draft"] is False
    assert got["history"]


def test_timeline_backup_is_capped(client: TestClient, demo: Project) -> None:
    """``plan/history/`` never grows past MAX_HISTORY entries."""
    from server.app import MAX_HISTORY

    for _ in range(MAX_HISTORY + 4):
        assert client.put("/api/p/demo/timeline", json=_timeline_payload()).status_code == 200
    assert len(list((demo.plan_dir / "history").glob("timeline_*.json"))) == MAX_HISTORY


def test_timeline_validate_endpoint(client: TestClient) -> None:
    """``/timeline/validate`` reports issues without writing anything."""
    ok = client.post("/api/p/demo/timeline/validate", json=_timeline_payload()).json()
    assert ok == {"ok": True, "issues": [], "duration": 3.0}
    bad = client.post("/api/p/demo/timeline/validate", json=_timeline_payload("c999")).json()
    assert bad["ok"] is False and bad["issues"]


def test_timeline_draft_and_accept(client: TestClient, demo: Project) -> None:
    """A plan draft is reported, fetchable and promotable."""
    draft = demo.plan_dir / "timeline.draft.json"
    _touch(draft, json.dumps(_timeline_payload()))

    missing = client.get("/api/p/demo/timeline")
    assert missing.status_code == 404
    assert missing.json()["detail"]["has_draft"] is True
    assert client.get("/api/p/demo/timeline/draft").status_code == 200

    accepted = client.post("/api/p/demo/timeline/accept_draft")
    assert accepted.status_code == 200, accepted.text
    assert demo.timeline_file.exists()
    assert not draft.exists()
    assert client.post("/api/p/demo/timeline/accept_draft").status_code == 404


def test_timeline_positions_match_python(client: TestClient, demo: Project) -> None:
    """``/timeline/positions`` is exactly ``Timeline.segment_positions()``."""
    assert client.get("/api/p/demo/timeline/positions").status_code == 404

    payload = _multi_segment_timeline()
    assert client.put("/api/p/demo/timeline", json=payload).status_code == 200
    body = client.get("/api/p/demo/timeline/positions").json()

    expected = Timeline.load(demo.timeline_file).segment_positions()
    assert [(p["id"], p["start"], p["end"]) for p in body["positions"]] == [
        (pos.segment.id, pos.start, pos.end) for pos in expected]
    # the xfade pulls the third segment back over the second one
    assert body["positions"][2]["start"] == pytest.approx(3.5)
    assert body["duration"] == pytest.approx(5.5)
    assert body["content_end"] == pytest.approx(5.5)
    assert body["positions"][1]["mute_source"] is True
    assert body["positions"][2]["transition"] == "xfade"


def test_timeline_positions_map_mute_ranges(client: TestClient) -> None:
    """Clip-time mute ranges are projected onto programme time, once per use."""
    client.put("/api/p/demo/timeline", json=_multi_segment_timeline())
    mutes = client.get("/api/p/demo/timeline/positions").json()["mutes"]
    # c001 0.5-1.5 falls inside s001 (in=0) and is outside s003 (in=3).
    assert [(m["segment"], m["start"], m["end"]) for m in mutes] == [("s001", 0.5, 1.5)]
    assert mutes[0]["reason"] == "bar music"


def test_timeline_positions_invalid_file(client: TestClient, demo: Project) -> None:
    """A hand-broken timeline on disk is a 422, not a 500."""
    _touch(demo.timeline_file, json.dumps({"tracks": {"video": [{"id": 1}]}}))
    assert client.get("/api/p/demo/timeline/positions").status_code == 422


# ----------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------
@needs_ffmpeg
def test_frame_endpoint_extracts_and_caches(
    client: TestClient, demo: Project, media: dict[str, Path]
) -> None:
    """A JPEG is extracted from the proxy and reused from the on-disk cache."""
    shutil.copy(media["landscape.mp4"], demo.proxy_path("c001"))
    cache = demo.thumbs_dir / "cache"

    res = client.get("/api/p/demo/frame", params={"clip": "c001", "t": 1.5, "w": 160})
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content[:3] == b"\xff\xd8\xff"          # JPEG SOI marker

    cached = list(cache.glob("c001_1.50_160.jpg"))
    assert len(cached) == 1
    stamp = cached[0].stat().st_mtime_ns

    again = client.get("/api/p/demo/frame", params={"clip": "c001", "t": 1.5, "w": 160})
    assert again.status_code == 200
    assert again.content == res.content
    assert cached[0].stat().st_mtime_ns == stamp      # served from the cache

    other = client.get("/api/p/demo/frame", params={"clip": "c001", "t": 3.0, "w": 160})
    assert other.status_code == 200
    assert sorted(p.name for p in cache.glob("*.jpg")) == [
        "c001_1.50_160.jpg", "c001_3.00_160.jpg"]


@needs_ffmpeg
def test_frame_cache_follows_the_proxy(
    client: TestClient, demo: Project, media: dict[str, Path]
) -> None:
    """A re-ingested proxy invalidates the cached frames."""
    shutil.copy(media["landscape.mp4"], demo.proxy_path("c001"))
    params = {"clip": "c001", "t": 0.5, "w": 96}
    assert client.get("/api/p/demo/frame", params=params).status_code == 200
    cached = demo.thumbs_dir / "cache" / "c001_0.50_96.jpg"
    before = cached.stat().st_mtime_ns

    shutil.copy(media["vertical.mp4"], demo.proxy_path("c001"))
    future = time.time() + 5
    os.utime(demo.proxy_path("c001"), (future, future))
    assert client.get("/api/p/demo/frame", params=params).status_code == 200
    assert cached.stat().st_mtime_ns != before


def test_frame_endpoint_rejections(client: TestClient, demo: Project) -> None:
    """Bad clip names, widths and missing proxies never reach ffmpeg."""
    assert client.get("/api/p/demo/frame",
                      params={"clip": "../etc", "t": 1}).status_code == 400
    assert client.get("/api/p/demo/frame",
                      params={"clip": "c001", "t": 1, "w": 5000}).status_code == 400
    assert client.get("/api/p/demo/frame",
                      params={"clip": "c001", "t": -3}).status_code == 422
    demo.proxy_path("c001").unlink()
    assert client.get("/api/p/demo/frame",
                      params={"clip": "c001", "t": 1}).status_code == 404


# ----------------------------------------------------------------------
# mute ranges
# ----------------------------------------------------------------------
def test_mute_ranges_add_update_delete(client: TestClient, demo: Project) -> None:
    """The waveform tool creates a timeline when needed, then edits it."""
    assert not demo.timeline_file.exists()
    added = client.post("/api/p/demo/mute_ranges", json={
        "op": "add", "clip": "c001", "s": 4.0, "e": 5.0,
        "gain_db": -60, "reason": "bar music"})
    assert added.status_code == 200, added.text
    assert added.json()["mute_ranges"] == [
        {"clip": "c001", "s": 4.0, "e": 5.0, "gain_db": -60.0, "reason": "bar music"}]
    on_disk = json.loads(demo.timeline_file.read_text(encoding="utf-8"))
    assert on_disk["meta"]["edited_by_human"] is True
    assert on_disk["meta"]["generated_by"] == "web-editor"

    updated = client.post("/api/p/demo/mute_ranges", json={
        "op": "update", "index": 0, "clip": "c001", "s": 4.0, "e": 5.0,
        "gain_db": -12, "reason": "duck"})
    assert updated.json()["mute_ranges"][0]["gain_db"] == -12.0

    removed = client.post("/api/p/demo/mute_ranges", json={"op": "delete", "index": 0})
    assert removed.json()["mute_ranges"] == []


def test_mute_ranges_errors(client: TestClient) -> None:
    """Reversed ranges, unknown ops and deletes without a timeline are refused."""
    assert client.post("/api/p/demo/mute_ranges", json={
        "op": "add", "clip": "c001", "s": 5.0, "e": 4.0}).status_code == 400
    assert client.post("/api/p/demo/mute_ranges", json={"op": "nope"}).status_code == 400
    assert client.post("/api/p/demo/mute_ranges", json={
        "op": "delete", "index": 0}).status_code == 404
    assert client.post("/api/p/demo/mute_ranges", json={
        "op": "add", "clip": "../x", "s": 1, "e": 2}).status_code == 400


def test_mute_ranges_preserved_by_timeline_put(client: TestClient) -> None:
    """Ranges marked before planning survive a subsequent timeline save."""
    client.post("/api/p/demo/mute_ranges", json={
        "op": "add", "clip": "c001", "s": 1.0, "e": 2.0})
    payload = client.get("/api/p/demo/timeline").json()["timeline"]
    payload["tracks"]["video"] = _timeline_payload()["tracks"]["video"]
    assert client.put("/api/p/demo/timeline", json=payload).status_code == 200
    assert len(client.get("/api/p/demo/timeline").json()["timeline"]["mute_ranges"]) == 1


# ----------------------------------------------------------------------
# clips
# ----------------------------------------------------------------------
def test_update_clip(client: TestClient, demo: Project) -> None:
    """Editorial per-clip fields round-trip into ``state.json``."""
    res = client.post("/api/p/demo/clips/c001", json={
        "notes": "ładne ujęcie", "exclude": True, "kind_override": "b-roll"})
    assert res.status_code == 200, res.text
    assert res.json()["clip"]["notes"] == "ładne ujęcie"
    stored = demo.load_state()["clips"]["c001"]
    assert stored["exclude"] is True
    assert stored["kind_override"] == "b-roll"

    assert client.post("/api/p/demo/clips/c001", json={"bogus": 1}).status_code == 400
    assert client.post("/api/p/demo/clips/c999", json={"notes": "x"}).status_code == 404


# ----------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------
def test_job_lifecycle(client: TestClient, demo: Project) -> None:
    """A stage is spawned, its output captured and the record persisted."""
    res = client.post("/api/p/demo/jobs", json={"stage": "ingest"})
    assert res.status_code == 202, res.text
    job_id = res.json()["id"]
    assert res.json()["stage"] == "ingest"

    client.app.state.jobs.wait(demo, job_id, timeout=10)
    record = client.get(f"/api/p/demo/jobs/{job_id}").json()
    assert record["status"] == "done"
    assert record["returncode"] == 0
    assert "ran: ingest demo" in record["log"]
    assert (demo.jobs_dir / f"{job_id}.log").exists()
    assert json.loads((demo.jobs_dir / "jobs.json").read_text())[0]["id"] == job_id

    listing = client.get("/api/p/demo/jobs").json()
    assert [j["id"] for j in listing["jobs"]] == [job_id]
    assert listing["running"] is None


def test_job_argv_and_rejections(client: TestClient, demo: Project) -> None:
    """Render stages map onto ``--preview``/``--master``; junk is rejected."""
    manager: JobManager = client.app.state.jobs
    assert manager.build_argv("render_preview", "demo")[-2:] == ["demo", "--preview"]
    assert manager.build_argv("render_master", "demo")[-2:] == ["demo", "--master"]
    assert manager.build_argv("plan", "demo", {"force": True})[-1] == "--force"

    assert client.post("/api/p/demo/jobs", json={"stage": "rm -rf"}).status_code == 400
    assert client.post("/api/p/demo/jobs", json={"stage": "ingest", "args": []}).status_code == 400

    res = client.post("/api/p/demo/jobs", json={"stage": "render_preview", "args": {"force": True}})
    assert res.status_code == 202
    assert res.json()["cmd"].endswith("render demo --preview --force")
    manager.wait(demo, res.json()["id"], timeout=10)


def test_tidy_and_denoise_jobs(client: TestClient, demo: Project) -> None:
    """The editing helpers build the argv the CLI expects."""
    manager: JobManager = client.app.state.jobs
    assert manager.build_argv("tidy", "demo")[1:] == ["tidy", "demo"]
    assert manager.build_argv("denoise", "demo", {"clip": "c001"})[1:] == [
        "denoise", "demo", "--clip", "c001", "--engine", "local"]
    assert manager.build_argv(
        "denoise", "demo", {"clip": "c001", "engine": "elevenlabs"})[1:] == [
        "denoise", "demo", "--clip", "c001", "--engine", "elevenlabs"]
    assert manager.build_argv("denoise", "demo", {"clip": "c001", "off": True})[1:] == [
        "denoise", "demo", "--clip", "c001", "--off"]
    # the A/B only compares what an earlier run wrote, so no engine is passed
    assert manager.build_argv("denoise", "demo", {
        "clip": "c001", "preview": True})[1:] == [
        "denoise", "demo", "--clip", "c001", "--preview"]

    tidy = client.post("/api/p/demo/jobs", json={"stage": "tidy"})
    assert tidy.status_code == 202, tidy.text
    assert tidy.json()["cmd"].endswith("tidy demo")
    manager.wait(demo, tidy.json()["id"], timeout=10)

    den = client.post("/api/p/demo/jobs", json={
        "stage": "denoise", "args": {"clip": "c001", "engine": "elevenlabs"}})
    assert den.status_code == 202, den.text
    assert den.json()["cmd"].endswith("denoise demo --clip c001 --engine elevenlabs")
    manager.wait(demo, den.json()["id"], timeout=10)


def test_denoise_job_rejections(client: TestClient) -> None:
    """A denoise without a sane clip id or engine is a 400, not a spawn."""
    for args in ({}, {"clip": "../etc"}, {"clip": "c001", "engine": "sox"}):
        res = client.post("/api/p/demo/jobs", json={"stage": "denoise", "args": args})
        assert res.status_code == 400, args


def test_helper_stages_are_not_pipeline_stages(client: TestClient) -> None:
    """``tidy``/``denoise`` stay out of the stage bar and of state.json."""
    from server.jobs import STAGE_ARGV, STAGE_ORDER

    assert {"tidy", "denoise"} <= set(STAGE_ARGV)
    assert not {"tidy", "denoise"} & set(STAGE_ORDER)
    assert set(client.get("/api/p/demo/state").json()["stages"]) == set(STAGE_ORDER)


def test_state_reports_denoise(client: TestClient, demo: Project) -> None:
    """The clip view learns whether denoised audio is in use and can A/B it."""
    first = client.get("/api/p/demo/state").json()["clips"][0]
    assert first["denoise"] == {"use": False, "engine": None, "ab_url": None}

    with demo.edit_state() as state:
        state["clips"]["c001"].update({"use_denoised": True, "denoise_engine": "elevenlabs"})
    _touch(demo.audio_dir / "c001.denoise_ab.wav", "riff")

    first = client.get("/api/p/demo/state").json()["clips"][0]
    assert first["denoise"]["use"] is True
    assert first["denoise"]["engine"] == "elevenlabs"
    assert first["denoise"]["ab_url"].startswith("/media/demo/audio/c001.denoise_ab.wav?v=")
    assert client.get(first["denoise"]["ab_url"]).status_code == 200


def test_job_progress_and_unknown_id(client: TestClient, demo: Project) -> None:
    """Render progress documents are forwarded; unknown ids 404."""
    _touch(demo.jobs_dir / "render_preview.json", json.dumps({"percent": 42.0, "step": "encode"}))
    assert client.app.state.jobs.progress(demo)["percent"] == 42.0
    assert client.get("/api/p/demo/jobs/nope-1").status_code == 404
    assert client.post("/api/p/demo/jobs/nope-1/cancel").status_code == 404


def test_only_one_job_per_project(root: Path, demo: Project, tmp_path: Path) -> None:
    """A second concurrent stage for the same project is a 409."""
    script = tmp_path / "slow-ytedit"
    script.write_text("#!/bin/sh\nsleep 3\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    manager = JobManager(executable=script, cwd=tmp_path)
    client = TestClient(create_app(root, manager=manager))

    first = client.post("/api/p/demo/jobs", json={"stage": "ingest"})
    assert first.status_code == 202
    assert client.post("/api/p/demo/jobs", json={"stage": "analyze"}).status_code == 409

    cancelled = client.post(f"/api/p/demo/jobs/{first.json()['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    manager.wait(demo, first.json()["id"], timeout=10)


# ----------------------------------------------------------------------
# media + security
# ----------------------------------------------------------------------
def test_media_mounts(client: TestClient, demo: Project) -> None:
    """Every project sub-tree is reachable, with Range and no-cache."""
    demo.proxy_path("c001").write_bytes(b"0123456789")
    _touch(demo.renders_dir / "preview.mp4", "render")
    _touch(demo.exports_dir / "master.mp4", "master")
    _touch(demo.music_dir / "bed.mp3", "music")
    _touch(demo.voice_dir / "take.wav", "voice")

    for url in ("/media/demo/proxies/c001.mp4", "/renders/demo/preview.mp4",
                "/exports/demo/master.mp4", "/music/demo/bed.mp3", "/voice/demo/take.wav"):
        res = client.get(url)
        assert res.status_code == 200, url
        assert res.headers["cache-control"] == "no-cache, must-revalidate", url

    ranged = client.get("/media/demo/proxies/c001.mp4", headers={"Range": "bytes=2-5"})
    assert ranged.status_code == 206
    assert ranged.content == b"2345"


def test_static_assets_are_not_cached(client: TestClient) -> None:
    """The UI is edited in place; a cached app.js would look like a bug."""
    res = client.get("/static/app.js")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-cache, must-revalidate"


def test_media_path_traversal_blocked(client: TestClient, root: Path) -> None:
    """Nothing outside the project tree can be reached through a mount."""
    secret = root.parent / "secret.txt"
    secret.write_text("classified", encoding="utf-8")

    for url in ("/media/demo/../../secret.txt",
                "/media/demo/..%2f..%2fsecret.txt",
                "/media/demo/proxies/..%2F..%2F..%2Fsecret.txt",
                "/media/..%2Fsecret.txt",
                "/media/Bad Slug/x.mp4",
                "/renders/demo/../../../secret.txt"):
        res = client.get(url)
        assert res.status_code in (400, 404), f"{url} -> {res.status_code}"
        assert "classified" not in res.text

    assert client.get("/api/p/demo/transcript/..%2F..%2Fsecret").status_code in (400, 404)
    assert client.get("/api/p/demo/analysis/.%2E").status_code in (400, 404)
