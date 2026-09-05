"""Pipeline stages run as subprocesses on behalf of the web editor.

The editor never imports a stage module in-process: a stage is a long, noisy,
possibly-crashing piece of work, so it is spawned as ``ytedit <stage> <slug>``
with its merged stdout/stderr streamed to ``projects/<slug>/jobs/<id>.log`` by
a reader thread. Job records are persisted to ``projects/<slug>/jobs/jobs.json``
so they survive a server restart, and **only one job may run per project** —
the stages mutate ``state.json`` and the media tree, and two of them racing
would corrupt both.

Render progress is picked up from whatever ``jobs/render_*.json`` the render
stage writes (``{percent, eta, step}``); the job record simply forwards it.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ytedit.log import get_logger
from ytedit.project import Project, utcnow

log = get_logger(__name__)

#: Stage name (as the API accepts it) -> ``ytedit`` argv after the executable.
#: ``{slug}`` is substituted with the project slug, ``{clip}``/``{engine}`` with
#: the matching entries of the request's ``args`` object.
STAGE_ARGV: dict[str, tuple[str, ...]] = {
    "ingest": ("ingest", "{slug}"),
    "transcribe": ("transcribe", "{slug}"),
    "analyze": ("analyze", "{slug}"),
    "plan": ("plan", "{slug}"),
    "music": ("music", "{slug}"),
    "render_preview": ("render", "{slug}", "--preview"),
    "render_master": ("render", "{slug}", "--master"),
    "qc": ("qc", "{slug}"),
    "publish": ("publish", "{slug}"),
    # Editing helpers: not part of the linear pipeline, so they are absent from
    # STAGE_ORDER and never appear as stage buttons or in state.json's stage map.
    "tidy": ("tidy", "{slug}"),
    "denoise": ("denoise", "{slug}", "--clip", "{clip}", "--engine", "{engine}"),
}

#: ``denoise`` argv when ``args = {"off": true}`` — go back to the raw audio.
DENOISE_OFF_ARGV: tuple[str, ...] = ("denoise", "{slug}", "--clip", "{clip}", "--off")

#: ``denoise`` argv when ``args = {"preview": true}`` — write the 6 s
#: original-then-denoised A/B wav and stop. Free: it only compares files an
#: earlier denoise run already produced, so ``--engine`` is not passed.
DENOISE_AB_ARGV: tuple[str, ...] = ("denoise", "{slug}", "--clip", "{clip}", "--preview")

#: Accepted ``denoise --engine`` values. The API defaults to the free one; the
#: CLI's own default is the paid engine, so the editor is always explicit.
DENOISE_ENGINES: tuple[str, ...] = ("elevenlabs", "local")

#: Stages the UI shows as buttons, in pipeline order.
STAGE_ORDER: tuple[str, ...] = (
    "ingest", "transcribe", "analyze", "plan", "music",
    "render_preview", "render_master", "qc", "publish",
)

#: Clip ids accepted in job arguments.
CLIP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: How many log lines are kept in memory (and returned by the API).
LOG_TAIL_LINES = 200

#: Jobs kept per project in ``jobs.json``.
MAX_JOBS = 100


class JobError(RuntimeError):
    """Raised for unknown stages and for a second concurrent job."""


def default_executable() -> Path:
    """Return the ``ytedit`` console script next to the running interpreter."""
    return Path(sys.executable).parent / "ytedit"


@dataclass
class Job:
    """One spawned stage run.

    Attributes:
        id: ``<stage>-<epoch-ms>`` identifier, also the log file name.
        stage: Stage key from :data:`STAGE_ARGV`.
        slug: Project the job belongs to.
        cmd: The argv that was executed.
        status: ``running`` | ``done`` | ``error`` | ``cancelled`` | ``aborted``.
        started: ISO-8601 UTC start time.
        finished: ISO-8601 UTC end time, or ``None`` while running.
        returncode: Process exit status once finished.
        error: Spawn/runtime error text.
    """

    id: str
    stage: str
    slug: str
    cmd: list[str] = field(default_factory=list)
    status: str = "running"
    started: str = field(default_factory=utcnow)
    finished: str | None = None
    returncode: int | None = None
    error: str = ""

    #: Runtime-only state, never persisted.
    _proc: subprocess.Popen[bytes] | None = field(default=None, repr=False, compare=False)
    _tail: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES), repr=False)

    @property
    def running(self) -> bool:
        """Whether the job is still in flight."""
        return self.status == "running"

    def to_dict(self, tail: int = 0, progress: dict[str, Any] | None = None) -> dict[str, Any]:
        """Serialize for the API.

        Args:
            tail: How many trailing log lines to include (0 for none).
            progress: Optional render progress document to attach.
        """
        lines = list(self._tail)[-tail:] if tail else []
        return {
            "id": self.id,
            "stage": self.stage,
            "slug": self.slug,
            "cmd": " ".join(shlex.quote(c) for c in self.cmd),
            "status": self.status,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "error": self.error,
            "log_tail": lines,
            "progress": progress,
        }

    def persisted(self) -> dict[str, Any]:
        """The subset written to ``jobs.json``."""
        return {
            "id": self.id,
            "stage": self.stage,
            "slug": self.slug,
            "cmd": self.cmd,
            "status": self.status,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "error": self.error,
        }

    @classmethod
    def from_persisted(cls, data: dict[str, Any]) -> "Job":
        """Rebuild a record read back from ``jobs.json``."""
        job = cls(
            id=str(data.get("id", "")),
            stage=str(data.get("stage", "")),
            slug=str(data.get("slug", "")),
            cmd=list(data.get("cmd") or []),
            status=str(data.get("status", "done")),
            started=str(data.get("started", "")),
            finished=data.get("finished"),
            returncode=data.get("returncode"),
            error=str(data.get("error", "")),
        )
        # A "running" record read from disk belongs to a dead process: the
        # server that owned it is gone.
        if job.status == "running":
            job.status = "aborted"
            job.finished = job.finished or utcnow()
        return job


class JobManager:
    """Spawns and tracks stage subprocesses, one at a time per project.

    Args:
        executable: The ``ytedit`` entry point. Defaults to the console script
            next to ``sys.executable``; tests point it at a harmless command.
        cwd: Working directory for spawned processes (the repository root by
            default, so relative config paths resolve).
    """

    def __init__(self, executable: Path | str | None = None, cwd: Path | str | None = None) -> None:
        self.executable = Path(executable) if executable else default_executable()
        self.cwd = Path(cwd) if cwd else Path(__file__).resolve().parent.parent
        self._jobs: dict[str, list[Job]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    @staticmethod
    def jobs_file(project: Project) -> Path:
        """Path of the per-project job index."""
        return project.jobs_dir / "jobs.json"

    @staticmethod
    def log_file(project: Project, job_id: str) -> Path:
        """Path of one job's captured output."""
        return project.jobs_dir / f"{job_id}.log"

    def _load(self, project: Project) -> list[Job]:
        """Return the in-memory job list for a project, reading disk once."""
        with self._lock:
            jobs = self._jobs.get(project.slug)
            if jobs is not None:
                return jobs
            jobs = []
            path = self.jobs_file(project)
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    log.warning("unreadable %s, starting a fresh job list", path)
                    raw = []
                for entry in raw if isinstance(raw, list) else []:
                    job = Job.from_persisted(entry)
                    job._tail.extend(_read_tail(self.log_file(project, job.id)))
                    jobs.append(job)
            self._jobs[project.slug] = jobs
            return jobs

    def _persist(self, project: Project) -> None:
        """Write the job index (newest last, capped at :data:`MAX_JOBS`)."""
        jobs = self._jobs.get(project.slug, [])[-MAX_JOBS:]
        self._jobs[project.slug] = jobs
        project.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file(project).write_text(
            json.dumps([j.persisted() for j in jobs], indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def list(self, project: Project) -> list[Job]:
        """Return every known job for a project, oldest first."""
        with self._lock:
            return list(self._load(project))

    def get(self, project: Project, job_id: str) -> Job | None:
        """Return one job by id, or ``None``."""
        with self._lock:
            for job in self._load(project):
                if job.id == job_id:
                    return job
        return None

    def running(self, project: Project) -> Job | None:
        """Return the project's in-flight job, if any."""
        with self._lock:
            for job in reversed(self._load(project)):
                if job.running:
                    return job
        return None

    def progress(self, project: Project) -> dict[str, Any] | None:
        """Return the newest ``jobs/render_*.json`` progress document, if any."""
        files = sorted(project.jobs_dir.glob("render_*.json"), key=_mtime)
        if not files:
            return None
        try:
            data = json.loads(files[-1].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def log_text(self, project: Project, job_id: str, max_bytes: int = 200_000) -> str:
        """Return the tail of a job's log file as text."""
        path = self.log_file(project, job_id)
        if not path.exists():
            return ""
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            return fh.read().decode("utf-8", "replace")

    # ------------------------------------------------------------------
    # spawning
    # ------------------------------------------------------------------
    def build_argv(self, stage: str, slug: str, args: dict[str, Any] | None = None) -> list[str]:
        """Build the argv for a stage run.

        Args:
            stage: Key of :data:`STAGE_ARGV`.
            slug: Project slug.
            args: ``{"force": bool}`` for every stage; ``denoise`` additionally
                takes ``{"clip": "c001", "engine": "elevenlabs"|"local"}``,
                ``{"clip": "c001", "preview": true}`` for the 6 s A/B wav of an
                existing denoise, or ``{"clip": "c001", "off": true}`` to drop
                back to raw audio.

        Raises:
            JobError: For an unknown stage or bad ``denoise`` arguments.
        """
        if stage not in STAGE_ARGV:
            raise JobError(f"unknown stage {stage!r}; known: {', '.join(STAGE_ARGV)}")
        args = args or {}
        parts = STAGE_ARGV[stage]
        clip = engine = ""
        if stage == "denoise":
            clip = str(args.get("clip") or "")
            if not CLIP_RE.match(clip):
                raise JobError(f"denoise needs a valid clip id (got {clip!r})")
            if args.get("off"):
                parts = DENOISE_OFF_ARGV
            elif args.get("preview"):
                parts = DENOISE_AB_ARGV
            else:
                engine = str(args.get("engine") or "local")
                if engine not in DENOISE_ENGINES:
                    raise JobError(
                        f"unknown denoise engine {engine!r}; "
                        f"known: {', '.join(DENOISE_ENGINES)}"
                    )
        argv = [str(self.executable)]
        argv += [part.format(slug=slug, clip=clip, engine=engine) for part in parts]
        if args.get("force"):
            argv.append("--force")
        return argv

    def submit(self, project: Project, stage: str, args: dict[str, Any] | None = None) -> Job:
        """Spawn a stage for a project.

        Args:
            project: Target project.
            stage: Key of :data:`STAGE_ARGV`.
            args: ``{"force": true}`` appends ``--force``.

        Returns:
            The new :class:`Job` (already running, or ``error`` if the spawn
            itself failed).

        Raises:
            JobError: For an unknown stage or when a job is already running.
        """
        argv = self.build_argv(stage, project.slug, args)
        with self._lock:
            jobs = self._load(project)
            busy = self.running(project)
            if busy is not None:
                raise JobError(
                    f"job {busy.id} ({busy.stage}) is still running for {project.slug}"
                )
            job = Job(id=f"{stage}-{int(time.time() * 1000)}", stage=stage, slug=project.slug,
                      cmd=argv)
            jobs.append(job)
            project.jobs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_file(project, job.id)
            try:
                # start_new_session gives the child its own process group so a
                # cancel kills ffmpeg and friends too, not just the CLI.
                job._proc = subprocess.Popen(
                    argv,
                    cwd=str(self.cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1",
                         "TERM": "dumb", "COLUMNS": "160"},
                )
            except OSError as exc:
                job.status = "error"
                job.error = str(exc)
                job.finished = utcnow()
                job._tail.append(f"failed to start: {exc}")
                log_path.write_text(f"failed to start: {exc}\n", encoding="utf-8")
                self._persist(project)
                return job
            self._persist(project)

        threading.Thread(
            target=self._pump, args=(project, job, log_path), name=f"job-{job.id}", daemon=True
        ).start()
        log.info("job %s started: %s", job.id, " ".join(argv))
        return job

    def _pump(self, project: Project, job: Job, log_path: Path) -> None:
        """Stream the child's merged output into the log file and the tail ring."""
        proc = job._proc
        assert proc is not None and proc.stdout is not None
        try:
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(f"$ {' '.join(shlex.quote(c) for c in job.cmd)}\n")
                fh.flush()
                job._tail.append(f"$ {' '.join(shlex.quote(c) for c in job.cmd)}")
                for raw in proc.stdout:
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    job._tail.append(line)
                    fh.write(line + "\n")
                    fh.flush()
            proc.wait()
        except OSError as exc:  # pragma: no cover - log file vanished
            job.error = str(exc)
        finally:
            with self._lock:
                job.returncode = proc.returncode
                job.finished = utcnow()
                if job.status == "cancelled":
                    pass
                elif proc.returncode == 0:
                    job.status = "done"
                else:
                    job.status = "error"
                    if not job.error:
                        job.error = f"exit code {proc.returncode}"
                job._proc = None
                self._persist(project)
            log.info("job %s %s (rc=%s)", job.id, job.status, job.returncode)

    def cancel(self, project: Project, job_id: str) -> Job:
        """Terminate a running job's process group.

        Raises:
            JobError: If the job is unknown or already finished.
        """
        job = self.get(project, job_id)
        if job is None:
            raise JobError(f"unknown job {job_id!r}")
        if not job.running or job._proc is None:
            raise JobError(f"job {job_id} is not running")
        job.status = "cancelled"
        proc = job._proc
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib_suppress():
                proc.terminate()
        return job

    def wait(self, project: Project, job_id: str, timeout: float = 10.0) -> Job | None:
        """Block until a job leaves the ``running`` state (used by tests)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get(project, job_id)
            if job is None or not job.running:
                return job
            time.sleep(0.02)
        return self.get(project, job_id)


def contextlib_suppress():  # pragma: no cover - tiny local helper
    """``contextlib.suppress(Exception)`` without a module-level import cost."""
    import contextlib

    return contextlib.suppress(Exception)


def _mtime(path: Path) -> float:
    """Modification time, ``0.0`` when the file disappeared."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _read_tail(path: Path, lines: int = LOG_TAIL_LINES) -> Iterable[str]:
    """Return the last ``lines`` lines of a log file (empty when missing)."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return []
    return text.splitlines()[-lines:]
