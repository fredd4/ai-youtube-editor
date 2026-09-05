"""``ytedit`` command line interface.

Stage commands import their implementation lazily so the CLI keeps working
while other modules (``ytedit.ai.*``, ``ytedit.media.render``, ``server.app``)
are still being written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer
from rich.table import Table

from .log import console, get_logger, setup_logging
from .project import STAGES, Project, ProjectError

log = get_logger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AI YouTube editor: raw phone footage -> finished 16:9 master.",
)


def _load(slug: str) -> Project:
    """Load a project or exit with a friendly message."""
    try:
        return Project.load(slug)
    except ProjectError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=2) from exc


def _lazy(module: str, attr: str) -> Callable[..., Any] | None:
    """Import ``module.attr`` or return None when the module does not exist yet."""
    try:
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr)
    except (ImportError, AttributeError):
        return None


def _not_implemented(stage: str, module: str) -> None:
    """Report an unfinished stage without failing the whole CLI."""
    console.print(
        f"[yellow]{stage}: not implemented yet[/] (waiting for [bold]{module}[/])"
    )
    raise typer.Exit(code=0)


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Configure logging before any command runs."""
    setup_logging("DEBUG" if verbose else "INFO")


# ----------------------------------------------------------------------
# project management
# ----------------------------------------------------------------------
@app.command()
def new(
    slug: str = typer.Argument(..., help="Project slug, e.g. lisbon-day-1."),
    language: str = typer.Option("pl", "--language", "-l", help="Content language (ISO-639-1)."),
    title: str | None = typer.Option(None, "--title", "-t", help="Working title."),
) -> None:
    """Create a new project directory tree."""
    project = Project.create(slug, language=language, title=title)
    console.print(f"[green]created[/] {project.path}")
    console.print(f"drop clips into [bold]{project.input_dir}[/], then: make ingest NAME={slug}")


@app.command("list")
def list_projects() -> None:
    """List all projects."""
    slugs = Project.list_projects()
    if not slugs:
        console.print("[dim]no projects yet — run: ytedit new <slug>[/]")
        return
    table = Table(title="projects", header_style="bold cyan")
    table.add_column("slug")
    table.add_column("title")
    table.add_column("lang")
    table.add_column("clips", justify="right")
    table.add_column("stages")
    for slug in slugs:
        project = Project.load(slug)
        state = project.load_state()
        done = [s for s in STAGES if state.get("stages", {}).get(s, {}).get("status") == "done"]
        table.add_row(
            slug, project.title, project.language, str(len(state.get("clips", {}))), ",".join(done) or "-"
        )
    console.print(table)


@app.command()
def status(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Show the clip registry, stage status and spend for a project."""
    from .costs import summary
    from .media.ingest import clips_table

    project = _load(slug)
    state = project.load_state()
    console.print(clips_table(project))

    stages = Table(title="stages", header_style="bold cyan")
    for column in ("stage", "status", "finished", "cost"):
        stages.add_column(column)
    for stage in STAGES:
        entry = state.get("stages", {}).get(stage, {})
        stage_status = str(entry.get("status", "pending"))
        style = {"done": "green", "error": "red", "running": "yellow"}.get(stage_status, "dim")
        stages.add_row(
            stage,
            f"[{style}]{stage_status}[/]",
            str(entry.get("finished", "-")),
            f"${float(entry.get('cost_usd') or 0):.4f}",
        )
    console.print(stages)

    spend = summary(project)
    console.print(
        f"budget [bold]${spend['budget']:.2f}[/] · spent [cost]${spend['spent']:.4f}[/] "
        f"· remaining [green]${spend['remaining']:.4f}[/]"
    )


@app.command()
def probe(path: Path = typer.Argument(..., help="Media file to inspect.")) -> None:
    """Print probed media properties as JSON."""
    from .media.probe import probe as probe_file

    info = probe_file(path)
    console.print_json(json.dumps(info.to_dict(), indent=2))


@app.command()
def ingest(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(False, "--force", help="Rebuild assets that already exist."),
) -> None:
    """Normalize input clips and build proxies, audio, peaks and thumbnails."""
    from .media.ingest import ingest as run_ingest

    project = _load(slug)
    results = run_ingest(project, force=force)
    errors = [r for r in results if r.status == "error"]
    if errors:
        raise typer.Exit(code=1)


@app.command()
def validate(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Validate ``plan/timeline.json`` against the project."""
    from .timeline import Timeline

    project = _load(slug)
    if not project.timeline_file.exists():
        console.print(f"[yellow]no timeline at {project.timeline_file}[/]")
        raise typer.Exit(code=1)
    issues = Timeline.load(project.timeline_file).validate(project)
    if not issues:
        console.print("[green]timeline ok[/]")
        return
    for issue in issues:
        console.print(f"[red]-[/] {issue}")
    raise typer.Exit(code=1)


# ----------------------------------------------------------------------
# stages owned by other modules (stubs until those land)
# ----------------------------------------------------------------------
@app.command()
def transcribe(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(False, "--force", help="Re-transcribe existing clips."),
) -> None:
    """Transcribe clip audio to word-level JSON + SRT."""
    fn = _lazy("ytedit.ai.transcribe", "transcribe")
    if fn is None:
        _not_implemented("transcribe", "ytedit.ai.transcribe")
    fn(_load(slug), force=force)


@app.command()
def analyze(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(False, "--force", help="Re-analyze existing clips."),
) -> None:
    """Analyze transcripts and frames into per-clip analysis JSON."""
    fn = _lazy("ytedit.ai.analyze", "analyze")
    if fn is None:
        _not_implemented("analyze", "ytedit.ai.analyze")
    fn(_load(slug), force=force)


@app.command()
def plan(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing draft."),
    notes: str | None = typer.Option(
        None, "--notes", help="Editor notes appended to the planner prompt."
    ),
) -> None:
    """Turn the footage log into an edit plan and a draft timeline."""
    fn = _lazy("ytedit.ai.plan", "plan")
    if fn is None:
        _not_implemented("plan", "ytedit.ai.plan")
    fn(_load(slug), force=force, notes=notes)


@app.command()
def tidy(
    slug: str = typer.Argument(..., help="Project slug."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report changes without writing."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite a human-edited timeline instead of writing a draft."
    ),
) -> None:
    """Give every speech cut air (~0.3 s before / ~0.45 s after) and merge jump cuts."""
    from .ai.tidy import TidyError
    from .ai.tidy import tidy as run_tidy

    project = _load(slug)
    try:
        result = run_tidy(project, dry_run=dry_run, force=force)
    except TidyError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    changes = result["changes"]
    if not changes:
        console.print("[green]nothing to tidy[/] — every cut already has its air")
        return
    for change in changes:
        console.print(f"[dim]-[/] {change}")
    console.print(
        f"[bold]{len(changes)}[/] change(s) · duration "
        f"{result['duration_before']:.2f}s -> {result['duration_after']:.2f}s"
    )
    for issue in result["issues"]:
        console.print(f"[yellow]![/] {issue}")
    if result["dry_run"]:
        console.print("[yellow]dry run — nothing written[/]")
        return
    console.print(f"[green]wrote[/] {result['written']} (backup: {result['backup']})")
    if result["edited_by_human"] and not force:
        console.print(
            "[yellow]timeline.json is human-edited[/] — review the draft, "
            "or re-run with --force"
        )


@app.command()
def denoise(
    slug: str = typer.Argument(..., help="Project slug."),
    clip: list[str] = typer.Option(
        None, "--clip", "-c", help="Clip id to process; repeat for several (default: all)."
    ),
    engine: str = typer.Option(
        "elevenlabs", "--engine", "-e", help="elevenlabs (paid, $0.12/min) or local (free ffmpeg)."
    ),
    force: bool = typer.Option(False, "--force", help="Redo clips already denoised."),
    off: bool = typer.Option(
        False, "--off", help="Stop using the denoised audio (keeps the file on disk)."
    ),
    on: bool = typer.Option(False, "--on", help="Use the denoised audio again."),
    preview: bool = typer.Option(
        False, "--preview", help="Write a 6 s original + 6 s denoised A/B WAV and exit."
    ),
) -> None:
    """Remove wind / background noise from clip audio (used by render when enabled)."""
    from .media.audio import (
        DenoiseError,
        denoise_ab_preview,
        denoise_clips,
        set_use_denoised,
    )

    project = _load(slug)
    clips = list(clip or [])
    try:
        if off or on:
            if not clips:
                clips = sorted(project.load_state().get("clips", {}))
            touched = set_use_denoised(project, clips, enabled=on)
            console.print(
                f"[green]use_denoised={'true' if on else 'false'}[/] for {', '.join(touched)}"
            )
            return

        if preview:
            if not clips:
                console.print("[bold red]--preview needs --clip <id>[/]")
                raise typer.Exit(code=2)
            from .ai.tidy import load_words

            for clip_id in clips:
                starts = [w.s for w in load_words(project, clip_id)] or None
                path = denoise_ab_preview(project, clip_id, speech_starts=starts)
                console.print(f"[green]A/B[/] {path}")
            return

        results = denoise_clips(project, clip_ids=clips or None, engine=engine, force=force)
    except DenoiseError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    table = Table(title=f"denoise ({engine})", header_style="bold cyan")
    for column in ("clip", "status", "seconds", "file"):
        table.add_column(column)
    for item in results:
        style = {"done": "green", "cached": "dim", "error": "red"}.get(item["status"], "")
        table.add_row(
            item["clip"],
            f"[{style}]{item['status']}[/]" if style else item["status"],
            f"{item['seconds']:.1f}",
            str(item.get("error") or item["engine_file"]),
        )
    console.print(table)
    if any(item["status"] == "error" for item in results):
        raise typer.Exit(code=1)


@app.command()
def run(
    slug: str = typer.Argument(..., help="Project slug."),
    until: str = typer.Option(
        "plan", "--until", help="Last stage to run: ingest|transcribe|analyze|plan|music."
    ),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if done."),
) -> None:
    """Run ingest -> transcribe -> analyze -> plan (-> music) in order, skipping done stages."""
    order = ["ingest", "transcribe", "analyze", "plan", "music"]
    if until not in order:
        console.print(f"[bold red]--until must be one of {', '.join(order)}[/]")
        raise typer.Exit(code=2)
    project = _load(slug)
    modules = {
        "ingest": ("ytedit.media.ingest", "ingest"),
        "transcribe": ("ytedit.ai.transcribe", "transcribe"),
        "analyze": ("ytedit.ai.analyze", "analyze"),
        "plan": ("ytedit.ai.plan", "plan"),
        "music": ("ytedit.ai.music", "generate_music"),
    }
    for stage in order[: order.index(until) + 1]:
        if not force and project.stage_status(stage) == "done":
            console.print(f"[dim]{stage}: already done, skipping (use --force to redo)[/]")
            continue
        console.rule(f"[bold cyan]{stage}[/]")
        fn = _lazy(*modules[stage])
        if fn is None:
            _not_implemented(stage, modules[stage][0])
        fn(project, force=force)
        project = _load(slug)


@app.command()
def music(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(False, "--force", help="Regenerate existing tracks."),
) -> None:
    """Generate the music beds requested by the plan."""
    fn = _lazy("ytedit.ai.music", "generate_music")
    if fn is None:
        _not_implemented("music", "ytedit.ai.music")
    fn(_load(slug), force=force)


@app.command()
def render(
    slug: str = typer.Argument(..., help="Project slug."),
    preview: bool = typer.Option(False, "--preview", help="Fast 720p preview render."),
    master: bool = typer.Option(False, "--master", help="Full-quality master export."),
) -> None:
    """Render the timeline to a preview or a master file."""
    fn = _lazy("ytedit.media.render", "render")
    if fn is None:
        _not_implemented("render", "ytedit.media.render")
    fn(_load(slug), preview=preview, master=master or not preview)


@app.command()
def qc(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Check the timeline and the rendered master against the playbook rules."""
    fn = _lazy("ytedit.qc", "qc")
    if fn is None:
        _not_implemented("qc", "ytedit.qc")
    fn(_load(slug))


@app.command()
def publish(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Generate titles, description, chapters and thumbnail candidates."""
    fn = _lazy("ytedit.ai.publish", "publish")
    if fn is None:
        _not_implemented("publish", "ytedit.ai.publish")
    fn(_load(slug))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", help="Port for the web editor."),
) -> None:
    """Run the local web editor."""
    fn = _lazy("server.app", "serve")
    if fn is None:
        _not_implemented("serve", "server.app")
    fn(host=host, port=port)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
