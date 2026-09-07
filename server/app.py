"""The local web editor: FastAPI JSON API + static SPA + media serving.

Design notes:

* **Read-only about media, authoritative about the timeline.** The API never
  runs ffmpeg itself; it edits JSON (``plan/timeline.json``, ``state.json``) and
  delegates every heavy stage to :mod:`server.jobs`.
* **Media goes through Starlette's ``StaticFiles``** so ``<video>`` seeking gets
  real HTTP Range support. The mounts are slug-scoped views of the project tree
  (``/media/<slug>/...`` -> ``projects/<slug>/media/...``) and refuse anything
  that escapes the project directory.
* **Cache-Control: no-cache on media**, because a re-render reuses the same
  filename; the client additionally appends ``?v=<mtime>`` so the browser
  actually refetches.
* **Localhost only.** :func:`serve` binds ``127.0.0.1``; there is no auth, and
  the API happily reads and writes files under ``projects/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from ytedit import costs
from ytedit.config import PROJECTS_DIR
from ytedit.log import get_logger
from ytedit.project import Project, ProjectError, utcnow, validate_slug
from ytedit.timeline import MuteRange, Timeline, new_timeline

from .jobs import STAGE_ORDER, JobError, JobManager

log = get_logger(__name__)

try:  # Starlette >= 0.33 strips the mount prefix into scope["root_path"].
    from starlette._utils import get_route_path as _route_path
except ImportError:  # pragma: no cover - older Starlette keeps it in scope["path"]
    def _route_path(scope: dict[str, Any]) -> str:
        """Path relative to the mount point."""
        root = scope.get("root_path", "")
        path = scope.get("path", "")
        return path[len(root):] if root and path.startswith(root) else path

WEB_DIR = Path(__file__).resolve().parent / "web"

#: Clip ids and other file-name path parameters must match this.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: URL prefixes served straight off the project tree, and their sub-directory.
MEDIA_MOUNTS: dict[str, str] = {
    "/media": "media",
    "/renders": "renders",
    "/exports": "exports",
    "/music": "music",
    "/voice": "voice",
}

#: Backups of ``plan/timeline.json`` kept in ``plan/history/``.
MAX_HISTORY = 20

#: The ``ffmpeg`` binary used by ``GET /frame``. Resolved once at import.
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"

#: Width bounds accepted by ``GET /frame``.
FRAME_MIN_WIDTH, FRAME_MAX_WIDTH = 32, 1920


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _safe_name(value: str, what: str = "name") -> str:
    """Validate a single path segment coming from a URL.

    Raises:
        HTTPException: 400 for empty, traversing or otherwise odd names.
    """
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise HTTPException(400, f"invalid {what}: {value!r}")
    if not NAME_RE.match(value):
        raise HTTPException(400, f"invalid {what}: {value!r}")
    return value


def _inside(child: Path, parent: Path) -> Path:
    """Return ``child`` resolved, ensuring it stays under ``parent``.

    Raises:
        HTTPException: 400 when the path escapes ``parent``.
    """
    resolved = child.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise HTTPException(400, "path outside the project")
    return resolved


def _mtime(path: Path) -> int:
    """Integer mtime for cache-busting, ``0`` when the file is missing."""
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0


def _read_json(path: Path) -> Any:
    """Read a JSON file, returning ``None`` when missing or unparseable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("unreadable %s: %s", path, exc)
        return None


def _read_text(path: Path) -> str | None:
    """Read a text file, returning ``None`` when missing."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover
        return None


def _first_existing(*paths: Path) -> Path | None:
    """Return the first path that exists."""
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_frame(source: Path, target: Path, t: float, width: int) -> None:
    """Write a single JPEG frame of ``source`` at ``t`` seconds into ``target``.

    Fast-seeks (``-ss`` before ``-i``) because the program strip asks for one
    thumbnail per segment and the proxies are keyframe-dense H.264.

    Raises:
        HTTPException: 500 when ffmpeg is missing or produced nothing.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.jpg")
    cmd = [
        FFMPEG_BIN, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, t):.3f}", "-i", str(source),
        "-frames:v", "1", "-vf", f"scale={width}:-2:flags=bicubic", "-q:v", "4",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"ffmpeg failed: {exc}") from exc
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise HTTPException(500, f"ffmpeg could not extract a frame: {' / '.join(tail)}")
    os.replace(tmp, target)


class _JsonCache:
    """Tiny mtime-keyed cache so ``/state`` polling does not re-parse transcripts."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, path: Path) -> Any:
        """Return the parsed document for ``path`` (``None`` when missing)."""
        key = str(path)
        try:
            stamp = path.stat().st_mtime
        except OSError:
            self._entries.pop(key, None)
            return None
        hit = self._entries.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        data = _read_json(path)
        self._entries[key] = (stamp, data)
        return data


class ProjectStatic(StaticFiles):
    """A ``StaticFiles`` view of one sub-directory of every project.

    The first URL segment is the project slug: ``/media/lisbon/proxies/c001.mp4``
    resolves to ``projects/lisbon/media/proxies/c001.mp4``. ``StaticFiles``
    itself does the ``resolve()``/containment check against the projects root,
    and Range requests are handled by Starlette's ``FileResponse``.
    """

    def __init__(self, root: Path, subdir: str) -> None:
        super().__init__(directory=str(root), check_dir=False)
        self.subdir = subdir

    def get_path(self, scope: dict[str, Any]) -> str:  # type: ignore[override]
        """Rewrite ``<slug>/<rest>`` to ``<slug>/<subdir>/<rest>``.

        ``scope["path"]`` still carries the mount prefix on modern Starlette
        (which moves it into ``root_path``), so the path has to come from
        :func:`starlette._utils.get_route_path` exactly like the base class.
        """
        raw = PurePosixPath(_route_path(scope).lstrip("/"))
        parts = [p for p in raw.parts if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts):
            raise HTTPException(404, "not found")
        slug, rest = parts[0], parts[1:]
        try:
            validate_slug(slug)
        except ProjectError as exc:
            raise HTTPException(404, str(exc)) from exc
        return os.path.join(slug, self.subdir, *rest)


# ----------------------------------------------------------------------
# state enrichment
# ----------------------------------------------------------------------
def _clip_flags(analysis: Any, transcript: Any) -> dict[str, Any]:
    """Derive the badges the clip list shows from analysis + transcript."""
    flags: dict[str, Any] = {
        "kind": None,
        "instructions": 0,
        "takes": 0,
        "background_music": 0,
        "language_mismatch": False,
        "summary": "",
        "location": "",
        "thumbnail_candidate": False,
    }
    if isinstance(analysis, dict):
        flags["kind"] = analysis.get("kind")
        flags["instructions"] = len(analysis.get("instructions") or [])
        takes = analysis.get("takes") or []
        flags["takes"] = sum(max(0, len(t.get("attempts") or []) - 1) for t in takes
                             if isinstance(t, dict))
        flags["background_music"] = len(analysis.get("background_music") or [])
        flags["summary"] = str(analysis.get("summary") or "")
        loc = analysis.get("location") or {}
        if isinstance(loc, dict):
            flags["location"] = ", ".join(
                str(loc[k]) for k in ("name", "city") if loc.get(k)
            )
        visual = analysis.get("visual") or {}
        if isinstance(visual, dict):
            flags["thumbnail_candidate"] = bool(visual.get("thumbnail_candidate"))
    if isinstance(transcript, dict):
        flags["language_mismatch"] = bool(transcript.get("language_mismatch"))
        flags["transcript_language"] = transcript.get("language")
    return flags


def _next_step(
    project: Project,
    state: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    files: dict[str, Any],
) -> str:
    """Return the one sentence the top bar shows: what to do next.

    The old amazonia-studio editor's best idea — a single state-machine hint —
    reimplemented over this pipeline's artifacts. Order mirrors
    ``docs/playbook/editing-playbook.md`` §2.
    """
    clips = state.get("clips") or {}
    inputs = list(project.input_dir.glob("*")) if project.input_dir.exists() else []
    media_inputs = [p for p in inputs if p.is_file() and not p.name.startswith(".")
                    and p.suffix.lower() != ".txt"]

    if not clips:
        if media_inputs:
            return f"{len(media_inputs)} file(s) waiting in input/ — run Ingest."
        return "Drop clips into input/ and run Ingest."
    if stages["ingest"]["status"] not in ("done", "running"):
        return "Ingest has not finished — run Ingest."
    if len(media_inputs) > len(clips):
        return "New files in input/ — run Ingest to register them."
    if stages["transcribe"]["status"] != "done":
        return "Transcribe the clips (word timestamps drive ducking and subtitles)."
    if stages["analyze"]["status"] != "done":
        return "Analyze the transcripts and frames to build the footage log."
    if not files["timeline"] and not files["draft"]:
        return "Plan the edit — turn the footage log into a timeline."
    if not files["timeline"] and files["draft"]:
        return "A draft timeline exists — open the Timeline tab and accept it."
    if files["draft"] and files["draft_mtime"] > files["timeline_mtime"]:
        return "A newer plan draft is waiting — diff it, then accept or discard it."
    if files["missing_music"]:
        return "The timeline references music that has not been generated — run Music."
    if not files["preview"]:
        return "Review the timeline, then Render preview."
    if files["preview_mtime"] < files["timeline_mtime"]:
        return "The timeline changed since the preview — Render preview again."
    if not files["master"]:
        return "Watch the preview, adjust the timeline, then Render master."
    if files["master_mtime"] < files["timeline_mtime"]:
        return "The timeline changed since the master — Render master again."
    if stages["qc"]["status"] != "done":
        return "Run QC on the master (playbook rules + measured loudness)."
    if not files["publish"]:
        return "Run Publish to generate titles, description, chapters and thumbnails."
    return "Everything is done — copy the publish pack and upload."


def _stage_view(project: Project, state: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
    """Status per UI stage button (``render`` is split into preview/master)."""
    raw = state.get("stages") or {}
    out: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        key = "render" if stage.startswith("render_") else stage
        entry = dict(raw.get(key) or {})
        status = str(entry.get("status", "pending"))
        if stage == "render_preview":
            status = "done" if files["preview"] else ("pending" if status == "done" else status)
        elif stage == "render_master":
            status = "done" if files["master"] else ("pending" if status == "done" else status)
        out[stage] = {
            "status": status,
            "started": entry.get("started"),
            "finished": entry.get("finished"),
            "cost_usd": entry.get("cost_usd"),
            "error": entry.get("error"),
        }
    stale = False
    if files["preview"] and files["preview_mtime"] < files["timeline_mtime"]:
        stale = True
    out["render_preview"]["stale"] = stale
    out["render_master"]["stale"] = bool(
        files["master"] and files["master_mtime"] < files["timeline_mtime"]
    )
    return out


def _project_files(project: Project) -> dict[str, Any]:
    """Locate the artifacts the state machine and the tabs care about."""
    timeline = project.timeline_file
    draft = project.plan_dir / "timeline.draft.json"
    preview = project.renders_dir / "preview.mp4"
    masters = sorted(project.exports_dir.glob("master*.mp4"), key=_mtime)
    master = masters[-1] if masters else None
    publish = project.exports_dir / "publish.json"
    qc = _first_existing(
        project.exports_dir / "qc_report.md",
        project.path / "qc_report.md",
        project.renders_dir / "qc_report.md",
    )

    missing_music: list[str] = []
    if timeline.exists():
        data = _read_json(timeline) or {}
        for cue in ((data.get("tracks") or {}).get("music") or []):
            ref = str((cue or {}).get("file") or "")
            if ref and not (project.path / ref).exists():
                missing_music.append(ref)

    return {
        "timeline": timeline.exists(),
        "timeline_mtime": _mtime(timeline),
        "draft": draft.exists(),
        "draft_mtime": _mtime(draft),
        "preview": preview.exists(),
        "preview_mtime": _mtime(preview),
        "preview_url": f"/renders/{project.slug}/preview.mp4?v={_mtime(preview)}"
        if preview.exists() else None,
        "master": master.name if master else None,
        "master_mtime": _mtime(master) if master else 0,
        "master_url": f"/exports/{project.slug}/{master.name}?v={_mtime(master)}"
        if master else None,
        "publish": publish.exists(),
        "qc": bool(qc),
        "footage_log": (project.analysis_dir / "footage_log.json").exists(),
        "edit_plan": (project.plan_dir / "edit_plan.json").exists(),
        "missing_music": missing_music,
    }


# ----------------------------------------------------------------------
# application
# ----------------------------------------------------------------------
def create_app(
    projects_root: Path | str | None = None, manager: JobManager | None = None
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        projects_root: Directory holding ``<slug>/`` project trees. Defaults to
            the repository's ``projects/``; tests pass a temporary directory.
        manager: Job runner; a default one is created when omitted.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    root = Path(projects_root).resolve() if projects_root else PROJECTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    jobs = manager or JobManager()
    cache = _JsonCache()

    api = FastAPI(title="ytedit web editor", docs_url="/api/docs", openapi_url="/api/openapi.json")
    api.state.projects_root = root
    api.state.jobs = jobs

    # -- infrastructure ------------------------------------------------
    @api.middleware("http")
    async def no_cache_media(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Stop the browser from serving stale renders or stale UI code.

        Media reuses filenames across re-renders; ``/static`` is edited in place
        while the server runs, and a cached ``app.js`` looks exactly like a bug.
        """
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") or any(
            path.startswith(p + "/") for p in MEDIA_MOUNTS
        ):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    @api.exception_handler(ProjectError)
    async def _project_error(request: Request, exc: ProjectError):  # type: ignore[no-untyped-def]
        """Turn project lookup failures into 404s."""
        return JSONResponse({"detail": str(exc)}, status_code=404)

    def load(slug: str) -> Project:
        """Load a project by slug or raise 400/404."""
        try:
            validate_slug(slug)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        path = root / slug
        if not path.is_dir():
            raise HTTPException(404, f"project {slug!r} not found")
        return Project(path)

    def project_file(project: Project, base: Path, name: str, suffix: str) -> Path:
        """Resolve ``<base>/<name><suffix>`` after validating ``name``."""
        _safe_name(name, "clip")
        return _inside(base / f"{name}{suffix}", project.path)

    # ------------------------------------------------------------------
    # projects
    # ------------------------------------------------------------------
    @api.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        """List every project with clip count, stage status and spend."""
        out = []
        for slug in Project.list_projects(root):
            project = Project(root / slug)
            state = project.load_state()
            spend = costs.summary(project)
            files = _project_files(project)
            out.append({
                "slug": slug,
                "title": project.title,
                "language": project.language,
                "clips": len(state.get("clips") or {}),
                "stages": _stage_view(project, state, files),
                "costs": spend,
                "created": state.get("created"),
            })
        return {"projects": out, "root": str(root)}

    @api.post("/api/projects", status_code=201)
    def create_project(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Create a project directory tree (``ytedit new`` from the browser)."""
        slug = str(payload.get("slug") or "").strip()
        try:
            validate_slug(slug)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        if (root / slug).exists():
            raise HTTPException(409, f"project {slug!r} already exists")
        project = Project.create(
            slug,
            language=str(payload.get("language") or "pl"),
            title=(payload.get("title") or None),
            root=root,
        )
        return {"slug": project.slug, "title": project.title, "language": project.language}

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @api.get("/api/p/{slug}/state")
    def get_state(slug: str) -> dict[str, Any]:
        """``state.json`` enriched with media URLs, per-clip flags and the hint."""
        project = load(slug)
        state = project.load_state()
        files = _project_files(project)
        stages = _stage_view(project, state, files)

        clips = []
        for clip in project.clips_in_order(state):
            cid = str(clip.get("id") or "")
            if not cid:
                continue
            proxy = project.proxy_path(cid)
            poster = project.poster_path(cid)
            peaks = project.peaks_path(cid)
            transcript = project.transcript_path(cid)
            analysis = project.analysis_path(cid)
            ab = project.audio_dir / f"{cid}.denoise_ab.wav"
            enriched = dict(clip)
            enriched.update({
                "proxy_url": f"/media/{slug}/proxies/{cid}.mp4?v={_mtime(proxy)}"
                if proxy.exists() else None,
                "poster_url": f"/media/{slug}/thumbs/{cid}.jpg?v={_mtime(poster)}"
                if poster.exists() else None,
                "peaks_url": f"/media/{slug}/peaks/{cid}.json?v={_mtime(peaks)}"
                if peaks.exists() else None,
                "has_transcript": transcript.exists(),
                "has_analysis": analysis.exists(),
                # `ytedit denoise` flips use_denoised/denoise_engine on the clip
                # record and writes a 6 s original->denoised comparison wav.
                "denoise": {
                    "use": bool(clip.get("use_denoised")),
                    "engine": clip.get("denoise_engine"),
                    "ab_url": f"/media/{slug}/audio/{cid}.denoise_ab.wav?v={_mtime(ab)}"
                    if ab.exists() else None,
                },
                "mtimes": {
                    "proxy": _mtime(proxy),
                    "poster": _mtime(poster),
                    "peaks": _mtime(peaks),
                    "transcript": _mtime(transcript),
                    "analysis": _mtime(analysis),
                },
                "flags": _clip_flags(cache.get(analysis), cache.get(transcript)),
            })
            clips.append(enriched)

        running = jobs.running(project)
        return {
            "slug": slug,
            "title": project.title,
            "language": project.language,
            "budget_usd": state.get("budget_usd"),
            "clips": clips,
            "stages": stages,
            "raw_stages": state.get("stages") or {},
            "costs": costs.summary(project),
            "files": files,
            "next_step": _next_step(project, state, stages, files),
            "running_job": running.to_dict() if running else None,
            "music": _music_list(project),
        }

    def _music_list(project: Project) -> list[dict[str, Any]]:
        """Music beds available for cue assignment."""
        out = []
        if project.music_dir.is_dir():
            for path in sorted(project.music_dir.iterdir()):
                if path.suffix.lower() not in (".mp3", ".wav", ".m4a", ".aac", ".flac"):
                    continue
                sidecar = _read_json(path.with_suffix(".json")) or {}
                out.append({
                    "file": f"music/{path.name}",
                    "name": path.name,
                    "url": f"/music/{project.slug}/{path.name}?v={_mtime(path)}",
                    "duration": sidecar.get("duration"),
                    "prompt": sidecar.get("prompt"),
                })
        return out

    # ------------------------------------------------------------------
    # per-clip documents
    # ------------------------------------------------------------------
    @api.get("/api/p/{slug}/transcript/{clip}")
    def get_transcript(slug: str, clip: str) -> Any:
        """Word-level transcript for one clip."""
        project = load(slug)
        path = project_file(project, project.transcripts_dir, clip, ".json")
        data = _read_json(path)
        if data is None:
            raise HTTPException(404, f"no transcript for {clip}")
        return data

    @api.get("/api/p/{slug}/analysis/{clip}")
    def get_analysis(slug: str, clip: str) -> Any:
        """LLM analysis for one clip."""
        project = load(slug)
        path = project_file(project, project.analysis_dir, clip, ".json")
        data = _read_json(path)
        if data is None:
            raise HTTPException(404, f"no analysis for {clip}")
        return data

    @api.get("/api/p/{slug}/frame")
    def get_frame(
        slug: str,
        clip: str = Query(..., description="Clip id, e.g. c004."),
        t: float = Query(0.0, ge=0.0, description="Source time in seconds."),
        w: int = Query(320, description="Output width in pixels."),
    ) -> FileResponse:
        """Extract one JPEG frame from a clip's proxy, cached on disk.

        The program strip shows the frame a segment actually starts on rather
        than the clip's poster, which is what makes a cut recognisable. Frames
        land in ``media/thumbs/cache/<clip>_<t>_<w>.jpg`` and are reused until
        the proxy is re-ingested (the cache entry is regenerated when it is
        older than the proxy).
        """
        project = load(slug)
        _safe_name(clip, "clip")
        if not FRAME_MIN_WIDTH <= w <= FRAME_MAX_WIDTH:
            raise HTTPException(400, f"w must be {FRAME_MIN_WIDTH}..{FRAME_MAX_WIDTH}")
        source = _first_existing(project.proxy_path(clip), project.source_path(clip))
        if source is None:
            raise HTTPException(404, f"no proxy for {clip} — run Ingest")
        cached = _inside(
            project.thumbs_dir / "cache" / f"{clip}_{t:.2f}_{w}.jpg", project.path
        )
        if not cached.exists() or _mtime(cached) < _mtime(source):
            _extract_frame(source, cached, t, w)
        return FileResponse(cached, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    @api.get("/api/p/{slug}/footage_log")
    def get_footage_log(slug: str) -> Any:
        """The merged, chronological footage log."""
        project = load(slug)
        data = _read_json(project.analysis_dir / "footage_log.json")
        if data is None:
            raise HTTPException(404, "no footage_log.json — run Analyze")
        return data

    @api.get("/api/p/{slug}/plan")
    def get_plan(slug: str) -> dict[str, Any]:
        """``edit_plan.json`` plus the human-readable markdown companions."""
        project = load(slug)
        plan = _read_json(project.edit_plan_file)
        edit_plan_md = _read_text(project.plan_dir / "edit_plan.md")
        narration_md = _read_text(project.plan_dir / "narration_requests.md")
        if plan is None and edit_plan_md is None and narration_md is None:
            raise HTTPException(404, "no plan yet — run Plan")
        return {
            "edit_plan": plan,
            "edit_plan_md": edit_plan_md,
            "narration_requests_md": narration_md,
        }

    @api.get("/api/p/{slug}/qc")
    def get_qc(slug: str) -> dict[str, Any]:
        """The QC report (markdown + JSON when present)."""
        project = load(slug)
        md = _first_existing(
            project.exports_dir / "qc_report.md",
            project.path / "qc_report.md",
            project.renders_dir / "qc_report.md",
        )
        js = _first_existing(
            project.exports_dir / "qc_report.json",
            project.path / "qc_report.json",
            project.renders_dir / "qc_report.json",
        )
        if md is None and js is None:
            raise HTTPException(404, "no QC report — run QC")
        return {"markdown": _read_text(md) if md else None,
                "report": _read_json(js) if js else None}

    @api.get("/api/p/{slug}/publish")
    def get_publish(slug: str) -> dict[str, Any]:
        """The publish pack: titles/description/chapters plus the thumbnails."""
        project = load(slug)
        data = _read_json(project.exports_dir / "publish.json")
        md = _read_text(project.exports_dir / "publish.md")
        thumbs_dir = project.exports_dir / "thumbnails"
        thumbs = []
        if thumbs_dir.is_dir():
            for path in sorted(thumbs_dir.iterdir()):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    thumbs.append({
                        "name": path.name,
                        "url": f"/exports/{slug}/thumbnails/{path.name}?v={_mtime(path)}",
                        "preview": "preview" in path.name.lower(),
                    })
        if data is None and md is None and not thumbs:
            raise HTTPException(404, "no publish pack — run Publish")
        return {"publish": data, "markdown": md, "thumbnails": thumbs}

    @api.get("/api/p/{slug}/music")
    def get_music(slug: str) -> dict[str, Any]:
        """Music beds available on disk."""
        return {"music": _music_list(load(slug))}

    # ------------------------------------------------------------------
    # timeline
    # ------------------------------------------------------------------
    def _backup_timeline(project: Project) -> str | None:
        """Copy the current timeline into ``plan/history/``, pruning old ones."""
        src = project.timeline_file
        if not src.exists():
            return None
        history = project.plan_dir / "history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().replace(":", "").replace("-", "").replace("+0000", "Z")
        target = history / f"timeline_{stamp}.json"
        n = 1
        while target.exists():
            target = history / f"timeline_{stamp}_{n}.json"
            n += 1
        shutil.copy2(src, target)
        backups = sorted(history.glob("timeline_*.json"), key=_mtime)
        for old in backups[:-MAX_HISTORY]:
            old.unlink(missing_ok=True)
        return target.name

    @api.get("/api/p/{slug}/timeline")
    def get_timeline(slug: str) -> dict[str, Any]:
        """The EDL, with a note about whether a newer plan draft exists."""
        project = load(slug)
        draft = project.plan_dir / "timeline.draft.json"
        if not project.timeline_file.exists():
            raise HTTPException(
                404,
                {"detail": "no timeline yet — run Plan", "has_draft": draft.exists()}
                if draft.exists() else "no timeline yet — run Plan",
            )
        data = _read_json(project.timeline_file)
        issues: list[str] = []
        try:
            issues = Timeline.model_validate(data).validate(project)
        except ValidationError as exc:  # pragma: no cover - hand-edited file
            issues = [f"schema: {e['loc']}: {e['msg']}" for e in exc.errors()]
        return {
            "timeline": data,
            "mtime": _mtime(project.timeline_file),
            "has_draft": draft.exists(),
            "draft_mtime": _mtime(draft),
            "draft_newer": draft.exists() and _mtime(draft) > _mtime(project.timeline_file),
            "issues": issues,
            "history": sorted(
                p.name for p in (project.plan_dir / "history").glob("timeline_*.json")
            )[-MAX_HISTORY:],
        }

    @api.get("/api/p/{slug}/timeline/draft")
    def get_timeline_draft(slug: str) -> Any:
        """The plan's draft timeline, when a fresh plan run wrote one."""
        project = load(slug)
        data = _read_json(project.plan_dir / "timeline.draft.json")
        if data is None:
            raise HTTPException(404, "no draft timeline")
        return data

    @api.get("/api/p/{slug}/timeline/positions")
    def get_timeline_positions(slug: str) -> dict[str, Any]:
        """Absolute placement of every video segment, computed in Python.

        The program strip draws blocks from this instead of re-deriving the
        cumulative-duration-minus-xfade-overlap arithmetic in JavaScript, so the
        picture the user sees can never disagree with what the renderer will build.
        Mute ranges (stored in *clip* time) are mapped into programme time here
        for the same reason.
        """
        project = load(slug)
        data = _read_json(project.timeline_file)
        if data is None:
            raise HTTPException(404, "no timeline yet — run Plan")
        try:
            timeline = Timeline.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(422, {"detail": "timeline on disk is invalid",
                                      "errors": exc.errors(include_url=False)}) from exc

        positions = timeline.segment_positions()
        segments = [
            {
                "id": pos.segment.id,
                "clip": pos.segment.clip,
                "start": pos.start,
                "end": pos.end,
                "duration": round(pos.end - pos.start, 6),
                "in": pos.segment.in_,
                "out": pos.segment.out,
                "role": pos.segment.role,
                "speed": pos.segment.speed,
                "mute_source": pos.segment.mute_source,
                "fit": pos.segment.transform.fit,
                "transition": pos.segment.transition_in.type,
                "transition_duration": pos.segment.transition_in.duration,
                "notes": pos.segment.notes,
            }
            for pos in positions
        ]

        # Mute ranges live in clip time; a range shows up once per segment that
        # actually uses that slice of the clip.
        mutes: list[dict[str, Any]] = []
        for mute in timeline.mute_ranges:
            for pos in positions:
                seg = pos.segment
                if seg.clip != mute.clip or seg.mute_source:
                    continue
                lo, hi = max(mute.s, seg.in_), min(mute.e, seg.out)
                if hi <= lo:
                    continue
                speed = seg.speed if seg.speed > 0 else 1.0
                mutes.append({
                    "clip": mute.clip,
                    "segment": seg.id,
                    "start": round(pos.start + (lo - seg.in_) / speed, 6),
                    "end": round(pos.start + (hi - seg.in_) / speed, 6),
                    "gain_db": mute.gain_db,
                    "reason": mute.reason,
                })

        return {
            "positions": segments,
            "mutes": sorted(mutes, key=lambda m: m["start"]),
            "duration": timeline.duration(),
            "content_end": timeline.content_end(),
            "mtime": _mtime(project.timeline_file),
        }

    @api.put("/api/p/{slug}/timeline")
    def put_timeline(
        slug: str,
        payload: dict[str, Any] = Body(...),
        force: bool = Query(False, description="Save even when validation reports issues."),
    ) -> dict[str, Any]:
        """Validate and save the timeline, stamping ``meta.edited_by_human``.

        The previous file is copied into ``plan/history/timeline_<ts>.json``
        (at most :data:`MAX_HISTORY` kept) so a bad save is always recoverable.
        Schema errors are always fatal (422); semantic issues from
        :meth:`Timeline.validate` return 400 unless ``force=true``.
        """
        project = load(slug)
        try:
            timeline = Timeline.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, {"detail": "invalid timeline", "errors": exc.errors(
                include_url=False)}) from exc
        # A hand edit in the Program view can move the segments an anchored
        # voice pickup follows; re-pin it here before validating/saving rather
        # than trusting whatever absolute `at` the client happened to send.
        timeline.resolve_voice_anchors()
        issues = timeline.validate(project)
        if issues and not force:
            raise HTTPException(400, {"detail": "timeline has issues", "issues": issues})
        timeline.meta.edited_by_human = True
        backup = _backup_timeline(project)
        project.plan_dir.mkdir(parents=True, exist_ok=True)
        timeline.save(project.timeline_file)
        log.info("timeline saved for %s (backup=%s)", slug, backup)
        draft = project.plan_dir / "timeline.draft.json"
        return {
            "ok": True,
            "backup": backup,
            "issues": issues,
            "mtime": _mtime(project.timeline_file),
            "duration": timeline.duration(),
            "has_draft": draft.exists(),
        }

    @api.post("/api/p/{slug}/timeline/validate")
    def validate_timeline(
        slug: str, payload: dict[str, Any] | None = Body(None)
    ) -> dict[str, Any]:
        """Validate a posted timeline, or the one on disk when the body is empty."""
        project = load(slug)
        data = payload if payload else _read_json(project.timeline_file)
        if data is None:
            raise HTTPException(404, "no timeline to validate")
        try:
            timeline = Timeline.model_validate(data)
        except ValidationError as exc:
            return {"ok": False, "issues": [
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(
                    include_url=False)]}
        issues = timeline.validate(project)
        return {"ok": not issues, "issues": issues, "duration": timeline.duration()}

    @api.post("/api/p/{slug}/timeline/accept_draft")
    def accept_draft(slug: str) -> dict[str, Any]:
        """Promote ``plan/timeline.draft.json`` to ``plan/timeline.json``."""
        project = load(slug)
        draft = project.plan_dir / "timeline.draft.json"
        if not draft.exists():
            raise HTTPException(404, "no draft timeline to accept")
        data = _read_json(draft)
        try:
            timeline = Timeline.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(422, {"detail": "draft is not a valid timeline",
                                      "errors": exc.errors(include_url=False)}) from exc
        timeline.resolve_voice_anchors()
        backup = _backup_timeline(project)
        timeline.save(project.timeline_file)
        draft.unlink(missing_ok=True)
        return {"ok": True, "backup": backup, "issues": timeline.validate(project)}

    # ------------------------------------------------------------------
    # mute ranges
    # ------------------------------------------------------------------
    @api.post("/api/p/{slug}/mute_ranges")
    def edit_mute_ranges(slug: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Add, update or delete a source-audio mute/duck range.

        A convenience wrapper over ``PUT /timeline`` for the waveform tool.
        Body: ``{op: add|update|delete, clip, s, e, gain_db, reason, index}``.
        ``index`` addresses an existing entry for ``update``/``delete``; without
        it, delete matches on ``(clip, s, e)``.

        When the project has no timeline yet a minimal one is created, so bar
        music can be marked **before** planning (playbook §2.2). Saving here
        stamps ``meta.edited_by_human``, which is what protects those ranges:
        a later ``ytedit plan`` then writes ``plan/timeline.draft.json`` instead
        of overwriting ``timeline.json``, and hands the existing timeline —
        these ranges included — to the planner as context.
        """
        project = load(slug)
        op = str(payload.get("op") or "add").lower()
        if op not in ("add", "update", "delete"):
            raise HTTPException(400, f"unknown op {op!r}")

        if project.timeline_file.exists():
            data = _read_json(project.timeline_file) or {}
            try:
                timeline = Timeline.model_validate(data)
            except ValidationError as exc:
                raise HTTPException(422, {"detail": "timeline on disk is invalid",
                                          "errors": exc.errors(include_url=False)}) from exc
        else:
            if op != "add":
                raise HTTPException(404, "no timeline yet")
            width, height, fps = project.settings.canvas
            timeline = new_timeline(width=width, height=height, fps=fps,
                                    language=project.language)
            timeline.meta.generated_by = "web-editor"

        index = payload.get("index")
        ranges = list(timeline.mute_ranges)

        def _match() -> int:
            """Index of the entry addressed by the body, or -1."""
            if isinstance(index, int) and 0 <= index < len(ranges):
                return index
            clip = payload.get("clip")
            for i, mr in enumerate(ranges):
                if (mr.clip == clip and abs(mr.s - float(payload.get("s", -1))) < 1e-3
                        and abs(mr.e - float(payload.get("e", -1))) < 1e-3):
                    return i
            return -1

        if op == "delete":
            i = _match()
            if i < 0:
                raise HTTPException(404, "mute range not found")
            ranges.pop(i)
        else:
            clip = _safe_name(str(payload.get("clip") or ""), "clip")
            try:
                start, end = float(payload["s"]), float(payload["e"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, "s and e are required numbers") from exc
            if end <= start:
                raise HTTPException(400, f"s ({start}) must be < e ({end})")
            entry = {
                "clip": clip,
                "s": round(start, 3),
                "e": round(end, 3),
                "gain_db": float(payload.get("gain_db", -60.0)),
                "reason": str(payload.get("reason") or ""),
            }
            if op == "update":
                i = _match()
                if i < 0:
                    raise HTTPException(404, "mute range not found")
                ranges[i] = MuteRange.model_validate(entry)
            else:
                ranges.append(MuteRange.model_validate(entry))

        timeline.mute_ranges = ranges
        timeline.meta.edited_by_human = True
        _backup_timeline(project)
        timeline.save(project.timeline_file)
        return {"ok": True, "mute_ranges": [m.model_dump(mode="json") for m in ranges]}

    # ------------------------------------------------------------------
    # clips
    # ------------------------------------------------------------------
    @api.post("/api/p/{slug}/clips/{clip}")
    def update_clip(slug: str, clip: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Store editorial per-clip fields (``notes``, ``exclude``, ``kind_override``)."""
        project = load(slug)
        _safe_name(clip, "clip")
        allowed = {"notes", "exclude", "kind_override"}
        unknown = set(payload) - allowed
        if unknown:
            raise HTTPException(400, f"unknown fields: {', '.join(sorted(unknown))}")
        with project.edit_state() as state:
            record = (state.get("clips") or {}).get(clip)
            if record is None:
                raise HTTPException(404, f"unknown clip {clip!r}")
            if "notes" in payload:
                record["notes"] = str(payload["notes"] or "")
            if "exclude" in payload:
                record["exclude"] = bool(payload["exclude"])
            if "kind_override" in payload:
                value = payload["kind_override"]
                record["kind_override"] = str(value) if value else None
            result = dict(record)
        return {"ok": True, "clip": result}

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------
    @api.get("/api/p/{slug}/jobs")
    def list_jobs(slug: str) -> dict[str, Any]:
        """Every job for a project; running jobs carry a live log tail."""
        project = load(slug)
        progress = jobs.progress(project)
        records = [
            job.to_dict(tail=200 if job.running else 0,
                        progress=progress if job.running else None)
            for job in jobs.list(project)
        ]
        return {"jobs": records[-40:], "running": next(
            (r for r in records if r["status"] == "running"), None)}

    @api.post("/api/p/{slug}/jobs", status_code=202)
    def start_job(slug: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Spawn ``ytedit <stage> <slug>`` for this project."""
        project = load(slug)
        stage = str(payload.get("stage") or "")
        args = payload.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise HTTPException(400, "args must be an object")
        try:
            job = jobs.submit(project, stage, args)
        except JobError as exc:
            raise HTTPException(409 if "still running" in str(exc) else 400, str(exc)) from exc
        return job.to_dict(tail=10)

    @api.get("/api/p/{slug}/jobs/{job_id}")
    def get_job(slug: str, job_id: str) -> dict[str, Any]:
        """One job with its full captured log."""
        project = load(slug)
        _safe_name(job_id, "job id")
        job = jobs.get(project, job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id!r}")
        record = job.to_dict(tail=200, progress=jobs.progress(project) if job.running else None)
        record["log"] = jobs.log_text(project, job_id)
        return record

    @api.post("/api/p/{slug}/jobs/{job_id}/cancel")
    def cancel_job(slug: str, job_id: str) -> dict[str, Any]:
        """Terminate a running job's whole process group."""
        project = load(slug)
        _safe_name(job_id, "job id")
        try:
            job = jobs.cancel(project, job_id)
        except JobError as exc:
            raise HTTPException(404 if "unknown" in str(exc) else 409, str(exc)) from exc
        return job.to_dict()

    # ------------------------------------------------------------------
    # pages + media
    # ------------------------------------------------------------------
    @api.get("/")
    def index() -> FileResponse:
        """The single-page editor shell (project list view)."""
        return FileResponse(WEB_DIR / "index.html")

    @api.get("/p/{slug}")
    def project_page(slug: str) -> FileResponse:
        """The same shell; ``app.js`` reads the slug out of the URL."""
        try:
            validate_slug(slug)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return FileResponse(WEB_DIR / "index.html")

    @api.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness probe used by the dev harness."""
        return {"ok": True, "root": str(root)}

    api.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    for prefix, subdir in MEDIA_MOUNTS.items():
        api.mount(prefix, ProjectStatic(root, subdir), name=f"media-{subdir}")
    return api


app = create_app()


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Run the web editor with uvicorn.

    Args:
        host: Bind address; keep the ``127.0.0.1`` default — the API has no
            authentication and full read/write access to ``projects/``.
        port: TCP port.
        reload: Enable uvicorn's autoreload (development only).
    """
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("binding %s exposes the editor (no auth) beyond this machine", host)
    log.info("web editor on http://%s:%d", host, port)
    if reload:  # pragma: no cover - dev convenience
        uvicorn.run("server.app:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    """``python -m server.app`` entry point."""
    parser = argparse.ArgumentParser(description="Run the ytedit web editor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
