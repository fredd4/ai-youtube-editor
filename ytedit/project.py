"""Project directories, ``state.json`` and the clip registry.

A project is a directory under ``projects/<slug>``. All state lives in
``state.json``, written atomically (tmp + ``os.replace``) under an ``fcntl``
file lock so concurrent stages/workers cannot lose updates.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from .config import PROJECTS_DIR, Settings, load_settings
from .log import get_logger

log = get_logger(__name__)

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

#: Sub-directories created for every project.
SUBDIRS: tuple[str, ...] = (
    "input",
    "media/sources",
    "media/proxies",
    "media/audio",
    "media/peaks",
    "media/thumbs",
    "media/thumbs/frames",
    "transcripts",
    "analysis",
    "plan",
    "music",
    "voice",
    "renders",
    "exports",
    "jobs",
)

#: Pipeline stages in order. ``denoise`` and ``tidy`` are optional repair
#: stages: nothing downstream requires them, they just stay ``pending``.
STAGES: tuple[str, ...] = (
    "ingest",
    "denoise",
    "transcribe",
    "analyze",
    "plan",
    "tidy",
    "music",
    "render",
    "qc",
    "publish",
)


def utcnow() -> str:
    """Return an ISO-8601 UTC timestamp (seconds resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ProjectError(RuntimeError):
    """Raised for missing projects, bad slugs and registry misuse."""


def validate_slug(slug: str) -> str:
    """Validate and return a project slug.

    Args:
        slug: Lowercase name, e.g. ``my-video``.

    Raises:
        ProjectError: If the slug is empty or contains illegal characters.
    """
    if not slug or not SLUG_RE.match(slug):
        raise ProjectError(
            f"invalid slug {slug!r}: use lowercase letters, digits, '-', '_', '.'"
        )
    return slug


class Project:
    """One video project on disk.

    Args:
        path: The ``projects/<slug>`` directory.
        settings: Pre-loaded settings; loaded from disk when omitted.
    """

    def __init__(self, path: Path | str, settings: Settings | None = None) -> None:
        self.path = Path(path).resolve()
        self.slug = self.path.name
        self._settings = settings
        self._state: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        slug: str,
        language: str = "pl",
        title: str | None = None,
        root: Path | str | None = None,
        **project_yaml_extra: Any,
    ) -> "Project":
        """Create a project directory tree, ``project.yaml`` and ``state.json``.

        Args:
            slug: Project slug (directory name).
            language: Content language for narration/captions (ISO-639-1).
            title: Working title; defaults to the slug.
            root: Parent directory for projects (default ``projects/``).
            **project_yaml_extra: Extra keys merged into ``project.yaml``.

        Returns:
            The new :class:`Project`. Existing projects are returned as-is
            (directories are created idempotently).
        """
        validate_slug(slug)
        base = Path(root) if root else PROJECTS_DIR
        path = base / slug
        for sub in SUBDIRS:
            (path / sub).mkdir(parents=True, exist_ok=True)

        project = cls(path)
        cfg_file = path / "project.yaml"
        if not cfg_file.exists():
            cfg: dict[str, Any] = {
                "project": slug,
                "title": title or slug,
                "language": language,
                "created": utcnow(),
                "style": "travel-vlog",
                "music_mood": "arrival-warm",
                "output_preset": "master",
            }
            cfg.update(project_yaml_extra)
            cfg_file.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        if not project.state_file.exists():
            project.save_state(
                {
                    "project": slug,
                    "created": utcnow(),
                    "clips": {},
                    "stages": {},
                    "costs": [],
                    "budget_usd": float(project.settings.budget_usd),
                }
            )
        log.info("created project [clip]%s[/] at %s", slug, path)
        return project

    @classmethod
    def load(cls, slug: str, root: Path | str | None = None) -> "Project":
        """Load an existing project by slug.

        Raises:
            ProjectError: If the directory does not exist.
        """
        validate_slug(slug)
        base = Path(root) if root else PROJECTS_DIR
        path = base / slug
        if not path.is_dir():
            raise ProjectError(f"project {slug!r} not found at {path}")
        return cls(path)

    @staticmethod
    def list_projects(root: Path | str | None = None) -> list[str]:
        """Return the slugs of all projects under ``root``, sorted."""
        base = Path(root) if root else PROJECTS_DIR
        if not base.is_dir():
            return []
        return sorted(
            p.name for p in base.iterdir() if p.is_dir() and (p / "project.yaml").exists()
        )

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        """Settings with this project's ``project.yaml`` merged in."""
        if self._settings is None:
            self._settings = load_settings(self.path)
        return self._settings

    @property
    def config(self) -> dict[str, Any]:
        """Raw ``project.yaml`` contents."""
        f = self.path / "project.yaml"
        if not f.exists():
            return {}
        return yaml.safe_load(f.read_text(encoding="utf-8")) or {}

    @property
    def language(self) -> str:
        """Content language of this project."""
        return str(self.config.get("language", self.settings.language))

    @property
    def title(self) -> str:
        """Working title."""
        return str(self.config.get("title", self.slug))

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    @property
    def input_dir(self) -> Path:
        """Raw drop zone the user copies phone clips into."""
        return self.path / "input"

    @property
    def sources_dir(self) -> Path:
        """Normalized mezzanine files (CFR, rotation baked, SDR)."""
        return self.path / "media" / "sources"

    @property
    def proxies_dir(self) -> Path:
        """720p proxies for the browser editor and vision analysis."""
        return self.path / "media" / "proxies"

    @property
    def audio_dir(self) -> Path:
        """Mono 48 kHz work WAVs for STT and analysis."""
        return self.path / "media" / "audio"

    @property
    def peaks_dir(self) -> Path:
        """Pre-computed waveform peaks for wavesurfer."""
        return self.path / "media" / "peaks"

    @property
    def thumbs_dir(self) -> Path:
        """Poster frames."""
        return self.path / "media" / "thumbs"

    @property
    def frames_dir(self) -> Path:
        """Sampled frames per clip (``frames/<clip>/NNN.jpg``)."""
        return self.path / "media" / "thumbs" / "frames"

    @property
    def transcripts_dir(self) -> Path:
        """``<clip>.json`` / ``<clip>.srt`` transcripts."""
        return self.path / "transcripts"

    @property
    def analysis_dir(self) -> Path:
        """Per-clip LLM analysis plus ``footage_log.json``."""
        return self.path / "analysis"

    @property
    def plan_dir(self) -> Path:
        """``edit_plan.json`` and ``timeline.json``."""
        return self.path / "plan"

    @property
    def music_dir(self) -> Path:
        """Generated music beds and sidecar metadata."""
        return self.path / "music"

    @property
    def voice_dir(self) -> Path:
        """Narration pickups (recorded or TTS)."""
        return self.path / "voice"

    @property
    def renders_dir(self) -> Path:
        """Preview renders and the segment cache."""
        return self.path / "renders"

    @property
    def exports_dir(self) -> Path:
        """Masters, captions, thumbnails, publish pack."""
        return self.path / "exports"

    @property
    def jobs_dir(self) -> Path:
        """Job progress files written by the server."""
        return self.path / "jobs"

    @property
    def state_file(self) -> Path:
        """``state.json`` — clip registry, stage status, cost ledger."""
        return self.path / "state.json"

    @property
    def lock_file(self) -> Path:
        """Advisory lock guarding ``state.json`` writes."""
        return self.path / ".state.lock"

    @property
    def timeline_file(self) -> Path:
        """``plan/timeline.json`` — the EDL, source of truth for render."""
        return self.plan_dir / "timeline.json"

    @property
    def edit_plan_file(self) -> Path:
        """``plan/edit_plan.json`` — the LLM's reasoning/draft."""
        return self.plan_dir / "edit_plan.json"

    # per-clip paths -----------------------------------------------------
    def source_path(self, clip_id: str) -> Path:
        """Normalized mezzanine for a clip."""
        return self.sources_dir / f"{clip_id}.mp4"

    def proxy_path(self, clip_id: str) -> Path:
        """720p proxy for a clip."""
        return self.proxies_dir / f"{clip_id}.mp4"

    def audio_path(self, clip_id: str) -> Path:
        """Mono work WAV for a clip."""
        return self.audio_dir / f"{clip_id}.wav"

    def peaks_path(self, clip_id: str) -> Path:
        """Waveform peaks JSON for a clip."""
        return self.peaks_dir / f"{clip_id}.json"

    def poster_path(self, clip_id: str) -> Path:
        """Poster JPEG for a clip."""
        return self.thumbs_dir / f"{clip_id}.jpg"

    def clip_frames_dir(self, clip_id: str) -> Path:
        """Directory of sampled frames for a clip."""
        return self.frames_dir / clip_id

    def transcript_path(self, clip_id: str) -> Path:
        """Transcript JSON for a clip."""
        return self.transcripts_dir / f"{clip_id}.json"

    def analysis_path(self, clip_id: str) -> Path:
        """Analysis JSON for a clip."""
        return self.analysis_dir / f"{clip_id}.json"

    def rel(self, path: Path | str) -> str:
        """Return ``path`` relative to the project dir as a POSIX string."""
        p = Path(path)
        try:
            return p.resolve().relative_to(self.path).as_posix()
        except ValueError:
            return p.as_posix()

    def ensure_dirs(self) -> None:
        """Create any missing sub-directory (safe to call repeatedly)."""
        for sub in SUBDIRS:
            (self.path / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # state.json
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        """Hold an exclusive advisory lock on the project state."""
        self.path.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_file, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def default_state(self) -> dict[str, Any]:
        """A fresh, empty state document."""
        return {
            "project": self.slug,
            "created": utcnow(),
            "clips": {},
            "stages": {},
            "costs": [],
            "budget_usd": 20.0,
        }

    def load_state(self) -> dict[str, Any]:
        """Read ``state.json`` from disk (no caching)."""
        if not self.state_file.exists():
            return self.default_state()
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupted state
            raise ProjectError(f"corrupted {self.state_file}: {exc}") from exc

    def save_state(self, state: dict[str, Any]) -> None:
        """Write ``state.json`` atomically (tmp file + ``os.replace``)."""
        self.path.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, indent=2, ensure_ascii=False, sort_keys=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.path), prefix=".state-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.state_file)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        self._state = state

    @property
    def state(self) -> dict[str, Any]:
        """Cached state document (call :meth:`load_state` to refresh)."""
        if self._state is None:
            self._state = self.load_state()
        return self._state

    @contextlib.contextmanager
    def edit_state(self) -> Iterator[dict[str, Any]]:
        """Read-modify-write ``state.json`` under the project lock.

        Yields:
            The freshly-read state dict; it is saved when the block exits
            without an exception.
        """
        with self._lock():
            state = self.load_state()
            yield state
            self.save_state(state)

    # ------------------------------------------------------------------
    # clip registry
    # ------------------------------------------------------------------
    def add_clip(self, clip: dict[str, Any]) -> dict[str, Any]:
        """Insert or merge a clip record into the registry.

        Args:
            clip: Must contain ``id``; other keys are merged over any existing
                record (``stages`` is merged rather than replaced).

        Returns:
            The stored clip record.
        """
        clip_id = clip.get("id")
        if not clip_id:
            raise ProjectError("clip record needs an 'id'")
        with self.edit_state() as state:
            clips = state.setdefault("clips", {})
            existing = dict(clips.get(clip_id, {}))
            stages = dict(existing.get("stages", {}))
            stages.update(clip.get("stages", {}))
            existing.update(clip)
            existing["stages"] = stages
            clips[clip_id] = existing
            return dict(existing)

    def set_clip_stage(self, clip_id: str, stage: str, status: str, **extra: Any) -> None:
        """Set ``clips[<id>].stages[<stage>]`` and merge extra clip fields.

        Args:
            clip_id: Clip id such as ``c001``.
            stage: Stage name (``ingest``, ``transcribe``, ...).
            status: ``pending`` | ``running`` | ``done`` | ``error``.
            **extra: Additional clip fields to merge (paths, error text, ...).
        """
        with self.edit_state() as state:
            clip = state.setdefault("clips", {}).setdefault(clip_id, {"id": clip_id})
            clip.setdefault("stages", {})[stage] = status
            if extra:
                clip.update(extra)

    def get_clip(self, clip_id: str) -> dict[str, Any]:
        """Return one clip record.

        Raises:
            ProjectError: If the clip is unknown.
        """
        clip = self.load_state().get("clips", {}).get(clip_id)
        if clip is None:
            raise ProjectError(f"unknown clip {clip_id!r} in project {self.slug!r}")
        return clip

    def clips_in_order(self, state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return clip records sorted by ``order`` then id (recording order)."""
        src = state if state is not None else self.load_state()
        clips = list(src.get("clips", {}).values())
        return sorted(clips, key=lambda c: (int(c.get("order", 10**6)), str(c.get("id", ""))))

    def next_clip_id(self, state: dict[str, Any] | None = None) -> str:
        """Return the next free ``cNNN`` id."""
        src = state if state is not None else self.load_state()
        used = {c for c in src.get("clips", {})}
        i = 1
        while f"c{i:03d}" in used:
            i += 1
        return f"c{i:03d}"

    # ------------------------------------------------------------------
    # stage status
    # ------------------------------------------------------------------
    def set_stage(self, stage: str, status: str, **extra: Any) -> None:
        """Record a pipeline-level stage status in ``state.stages``.

        Args:
            stage: Stage name.
            status: ``running`` | ``done`` | ``error``.
            **extra: Extra fields (``error``, ``cost_usd``, ...).
        """
        with self.edit_state() as state:
            entry = state.setdefault("stages", {}).setdefault(stage, {})
            entry["status"] = status
            if status == "running":
                entry["started"] = utcnow()
                entry.pop("error", None)
            if status in ("done", "error"):
                entry["finished"] = utcnow()
            entry.update(extra)

    def stage_status(self, stage: str) -> str:
        """Return the status of a stage, ``"pending"`` when never run."""
        return str(self.load_state().get("stages", {}).get(stage, {}).get("status", "pending"))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Project(slug={self.slug!r}, path={str(self.path)!r})"
