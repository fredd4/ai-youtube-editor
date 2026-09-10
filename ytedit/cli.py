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
    proxies: bool | None = typer.Option(
        None, "--proxies/--no-proxies",
        help="Build 720p browser proxies too (default: config ingest.proxies, off — "
        "'ytedit serve' builds the ones the web editor needs).",
    ),
) -> None:
    """Build the mezzanine, audio, peaks and thumbnails for every input clip.

    A clip that already matches the project's `format` is remuxed (stream
    copy) instead of re-encoded; the summary says which clips were encoded
    and why.
    """
    from .media.ingest import ingest as run_ingest

    project = _load(slug)
    results = run_ingest(project, force=force, proxies=proxies)
    errors = [r for r in results if r.status == "error"]
    if errors:
        raise typer.Exit(code=1)


def _print_cut_issues(issues: list[Any]) -> int:
    """Print ``ytedit.cut`` validation issues; return how many are errors."""
    from rich.markup import escape

    errors = 0
    for issue in issues:
        colour = "red" if issue.severity == "error" else "yellow"
        if issue.severity == "error":
            errors += 1
        where = f" ({escape(issue.beat)})" if issue.beat else ""
        console.print(
            f"[{colour}]{issue.severity:<7}[/] {escape(issue.code)}{where}: "
            f"{escape(issue.message)}"
        )
    return errors


def _ensure_resolved(project: Project) -> None:
    """Re-resolve ``plan/timeline.json`` from ``plan/cut.json`` when it is stale.

    A project with no ``cut.json`` yet is left untouched — the command that
    follows reports the missing file itself. A cut that does not validate
    stops the command rather than letting it render a stale file.
    """
    from .cut import CutError, cut_path, ensure_resolved

    if not cut_path(project).exists():
        return
    try:
        ensure_resolved(project)
    except CutError as exc:
        console.print("[bold red]cut.json does not resolve[/] — fix it and try again:")
        _print_cut_issues(exc.issues)
        raise typer.Exit(code=1) from exc


@app.command()
def validate(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Validate ``plan/cut.json`` against the project: every rule, no writes."""
    from .cut import cut_path, load_cut
    from .cut import validate as validate_cut

    project = _load(slug)
    source = cut_path(project)
    if not source.exists():
        console.print(
            f"[bold red]no cut at {source}[/] — run 'ytedit plan {slug}', or "
            f"'ytedit migrate {slug}' on a project that still has a v1 timeline"
        )
        raise typer.Exit(code=2)
    issues = validate_cut(project, load_cut(source))
    errors = _print_cut_issues(issues)
    if errors:
        console.print(f"[bold red]{errors} error(s)[/] in {source}")
        raise typer.Exit(code=1)
    console.print(f"[green]cut ok[/] ({len(issues)} warning(s))")


@app.command()
def resolve(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Resolve ``plan/cut.json`` into the derived ``plan/timeline.json``.

    The cut is the source of truth; the timeline is a render artifact and is
    overwritten every time. Validation errors stop the write.
    """
    from .cut import cut_path, load_cut
    from .cut import resolve as resolve_cut
    from .cut import validate as validate_cut

    project = _load(slug)
    source = cut_path(project)
    if not source.exists():
        console.print(f"[bold red]no cut at {source}[/] — run 'ytedit migrate {slug}' "
                      "on a v1 project")
        raise typer.Exit(code=2)
    cut = load_cut(source)
    errors = _print_cut_issues(validate_cut(project, cut))
    if errors:
        console.print(f"[bold red]{errors} error(s)[/] — timeline not written")
        raise typer.Exit(code=1)
    timeline = resolve_cut(project, cut)
    timeline.save(project.timeline_file)
    console.print(
        f"[green]resolved[/] {len(cut.beats)} beat(s) -> "
        f"{len(timeline.tracks.video)} segment(s), {timeline.duration():.2f}s "
        f"-> {project.timeline_file}"
    )


@app.command()
def migrate(
    slug: str = typer.Argument(..., help="Project slug."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change without writing anything."
    ),
) -> None:
    """Convert a v1 ``plan/timeline.json`` into a v2 ``plan/cut.json``."""
    fn = _lazy("ytedit.migrate", "migrate_project")
    if fn is None:
        _not_implemented("migrate", "ytedit.migrate")
    project = _load(slug)
    report = fn(project, dry_run=dry_run)
    console.print(report.markdown(), markup=False, highlight=False)
    if report.issues:
        console.print(f"[yellow]{len(report.issues)} issue(s)[/] — see the report above")


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
def sentences(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Build the sentence catalogue (script-first planning) from transcripts + analysis."""
    from .ai.sentences import write_sentences

    document = write_sentences(_load(slug))
    console.print(
        f"[green]wrote[/] analysis/sentences.json — {document['clips_count']} clip(s), "
        f"{document['sentences_count']} sentence(s)"
    )


@app.command()
def plan(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing draft."),
    notes: str | None = typer.Option(
        None, "--notes", help="Editor notes appended to the planner prompt."
    ),
    from_response: bool = typer.Option(
        False,
        "--from-response",
        help="Skip the LLM call; rebuild from the last plan/planner_response.json "
        "(no cost, e.g. after a post-processing fix).",
    ),
) -> None:
    """Turn the footage log into ``plan/cut.json`` and resolve it to a timeline."""
    fn = _lazy("ytedit.ai.plan", "plan")
    if fn is None:
        _not_implemented("plan", "ytedit.ai.plan")
    fn(_load(slug), force=force, notes=notes, from_response=from_response)


@app.command()
def captions(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(
        False, "--force",
        help="Re-ask the writer for places and overwrite a human-edited cut.",
    ),
    include_cold_open: bool = typer.Option(
        False, "--include-cold-open", help="Also allow a location card during the cold open."
    ),
    keep_existing: bool = typer.Option(
        False, "--keep-existing",
        help="Keep the planner's own location captions instead of replacing them.",
    ),
) -> None:
    """Place a location card at every new place, on that place's first beat."""
    from .ai.locations import LocationsError
    from .ai.locations import run_captions_stage as _run_captions
    from .ai.publish import format_timecode

    project = _load(slug)
    try:
        result = _run_captions(
            project, force=force, include_cold_open=include_cold_open,
            keep_existing=keep_existing,
        )
    except LocationsError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"captions — {slug}")
    for col in ("time", "segment", "style", "text"):
        table.add_column(col)
    for row in result["report"]:
        table.add_row(format_timecode(row["at"]), row["segment"] or "—", row["style"], row["text"])
    console.print(table)
    console.print(
        f"[bold]{result['cards_added']}[/] location card(s) · "
        f"places cost ${result['places_cost_usd']:.4f} ({result['places_path']}) · "
        f"wrote {result['written']} (backup: {result['backup']}) · "
        f"report: {result['report_path']}"
    )
    for issue in result["issues"]:
        console.print(f"[yellow]![/] {issue}")
    if result["edited_by_human"] and not force:
        console.print(
            "[yellow]cut.json is human-edited[/] — review the draft, "
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
            from .words import load_words

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
def noise(
    slug: str = typer.Argument(..., help="Project slug."),
    used_only: bool = typer.Option(
        None,
        "--used-only/--all",
        help="Restrict to clips used unmuted in plan/timeline.json "
        "(default: on when a timeline exists, otherwise every clip is scanned).",
    ),
    denoise_flag: bool = typer.Option(
        False, "--denoise", help="Run denoise on every windy clip that isn't denoised yet."
    ),
    engine: str = typer.Option(
        "elevenlabs", "--engine", "-e", help="Engine for --denoise: elevenlabs (paid) or local (free)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the cost confirmation gate for --denoise."
    ),
    threshold_snr: float | None = typer.Option(
        None, "--threshold-snr", help="Override noise.snr_flag_db (windy cutoff) for this run."
    ),
    threshold_low: float | None = typer.Option(
        None, "--threshold-low", help="Override noise.low_band_flag (windy cutoff) for this run."
    ),
) -> None:
    """Scan clip audio for wind/noise and flag which clips need denoise."""
    from .media.audio import DenoiseError, denoise_clips
    from .media.noise import (
        NoiseError,
        estimate_denoise_cost,
        scan_project,
        windy_undenoised_clips,
        write_report,
    )

    project = _load(slug)
    resolved_used_only = project.timeline_file.exists() if used_only is None else used_only
    try:
        report = scan_project(
            project,
            used_only=resolved_used_only,
            snr_flag_db=threshold_snr,
            low_band_flag=threshold_low,
        )
    except NoiseError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    json_path, md_path = write_report(project, report)

    table = Table(title="noise scan (worst first)", header_style="bold cyan")
    for column in ("clip", "snr db", "gap db", "speech db", "low band", "gap s", "speech s", "flag", "denoised"):
        table.add_column(column)
    for row in report["clips"]:
        flag = row["flags"][0] if row["flags"] else "-"
        style = {"windy": "red", "noisy": "yellow"}.get(flag, "")
        table.add_row(
            f"[{style}]{row['clip']}[/]" if style else row["clip"],
            f"{row['snr_db']:.1f}",
            f"{row['gap_rms_db']:.1f}",
            f"{row['speech_rms_db']:.1f}",
            f"{row['low_band_ratio']:.2f}",
            f"{row['gap_seconds']:.1f}",
            f"{row['speech_seconds']:.1f}",
            f"[{style}]{flag}[/]" if style else flag,
            f"yes ({row['denoise_engine']})" if row["use_denoised"] else "-",
        )
    console.print(table)
    console.print(f"[dim]wrote[/] {project.rel(json_path)}, {project.rel(md_path)}")
    if report["skipped"]:
        console.print(
            f"[dim]skipped (not enough gap/speech signal): {', '.join(report['skipped'])}[/]"
        )

    windy = windy_undenoised_clips(report)
    if not windy:
        if denoise_flag:
            console.print("[green]nothing to denoise[/] — no un-denoised windy clips")
        return

    if not denoise_flag:
        console.print(
            f"[yellow]{len(windy)} windy clip(s) not yet denoised:[/] {', '.join(windy)} "
            "— re-run with --denoise to clean them up"
        )
        return

    estimated = estimate_denoise_cost(project, windy, engine)
    console.print(
        f"[bold]denoise estimate:[/] {len(windy)} clip(s) with [bold]{engine}[/], "
        f"~[cost]${estimated:.2f}[/]"
    )
    if estimated > 2.0 and not yes:
        console.print(
            "[bold red]estimated cost exceeds $2[/] — re-run with --yes to proceed"
        )
        raise typer.Exit(code=2)

    try:
        results = denoise_clips(project, clip_ids=windy, engine=engine)
    except DenoiseError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    for item in results:
        style = {"done": "green", "cached": "dim", "error": "red"}.get(item["status"], "")
        line = f"{item['clip']}: {item['status']}"
        console.print(f"[{style}]{line}[/]" if style else line)
    if any(item["status"] == "error" for item in results):
        raise typer.Exit(code=1)


@app.command()
def run(
    slug: str = typer.Argument(..., help="Project slug."),
    until: str = typer.Option(
        "plan", "--until",
        help="Last stage to run: ingest|transcribe|analyze|sentences|plan.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if done."),
) -> None:
    """Run ingest -> transcribe -> analyze -> sentences -> plan, skipping done stages.

    Music is deliberately not part of this chain: a bed is only worth
    generating once the cut is stable, and it costs real money per track (see
    the cost rules in CLAUDE.md). Run ``ytedit music`` yourself when the cut
    has settled.
    """
    order = ["ingest", "transcribe", "analyze", "sentences", "plan"]
    if until not in order:
        console.print(f"[bold red]--until must be one of {', '.join(order)}[/]")
        raise typer.Exit(code=2)
    project = _load(slug)
    modules = {
        "ingest": ("ytedit.media.ingest", "ingest"),
        "transcribe": ("ytedit.ai.transcribe", "transcribe"),
        "analyze": ("ytedit.ai.analyze", "analyze"),
        "sentences": ("ytedit.ai.sentences", "write_sentences"),
        "plan": ("ytedit.ai.plan", "plan"),
    }
    for stage in order[: order.index(until) + 1]:
        if not force and project.stage_status(stage) == "done":
            console.print(f"[dim]{stage}: already done, skipping (use --force to redo)[/]")
            continue
        console.rule(f"[bold cyan]{stage}[/]")
        fn = _lazy(*modules[stage])
        if fn is None:
            _not_implemented(stage, modules[stage][0])
        if stage == "sentences":
            fn(project)
        else:
            fn(project, force=force)
        project = _load(slug)
    # The cut is the source of truth; make sure the derived timeline matches
    # it before anyone renders or reviews.
    _ensure_resolved(project)


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
    draft: bool = typer.Option(
        False, "--draft",
        help="Very fast, very low quality 720p render (~1 MB/10s) from the ingest proxy — "
        "recommended for a first full-timeline review before spending 10-25 minutes on "
        "--preview. Audio (denoise/leveling/ducking/loudnorm) already matches the master.",
    ),
    preview: bool = typer.Option(
        False, "--preview", help="Fast 720p preview render (full mezzanine, hardware encoder)."
    ),
    master: bool = typer.Option(
        False, "--master",
        help="Full-quality master export (hardware-encoded by default; add --x264 for the "
        "slower libx264 tier). Default when no tier flag is given.",
    ),
    no_music: bool = typer.Option(
        False, "--no-music", help="Ignore music cues for this render (timeline untouched)."
    ),
    no_voice: bool = typer.Option(
        False, "--no-voice", help="Ignore voice pickups for this render, symmetrically."
    ),
    x264: bool = typer.Option(
        False, "--x264",
        help="Master only: use the slower libx264 tier instead of the hardware-encoded "
        "default — YouTube re-encodes on ingest anyway, so the hardware tier is visually "
        "equivalent and about 10x faster; reach for --x264 only when you want the extra "
        "quality margin for a final upload.",
    ),
) -> None:
    """Render the timeline to a draft, preview or master file."""
    if sum([draft, preview, master]) > 1:
        console.print("[bold red]pick one of --draft, --preview, --master[/]")
        raise typer.Exit(code=2)
    fn = _lazy("ytedit.media.render", "render")
    if fn is None:
        _not_implemented("render", "ytedit.media.render")
    project = _load(slug)
    _ensure_resolved(project)
    fn(
        project, preview=preview, draft=draft, master=master or not (preview or draft),
        no_music=no_music, no_voice=no_voice, x264=x264,
    )


@app.command()
def clean(
    slug: str = typer.Argument(..., help="Project slug."),
    segments_only: bool = typer.Option(
        False, "--segments", help="Remove unreferenced cached segments only."
    ),
    intermediates_only: bool = typer.Option(
        False, "--intermediates", help="Remove renders/ intermediates only."
    ),
    all_: bool = typer.Option(
        False, "--all", help="Remove both intermediates and unreferenced segments (default)."
    ),
) -> None:
    """Free disk space: drop render intermediates and unreferenced cached segments.

    With no flags (or ``--all``) both are removed. ``--segments``/
    ``--intermediates`` restrict the sweep to just that one. Never touches
    ``media/``, ``input/`` or ``exports/``.
    """
    from .media.render import clean as run_clean

    project = _load(slug)
    only_one = segments_only or intermediates_only
    do_segments = segments_only or all_ or not only_one
    do_intermediates = intermediates_only or all_ or not only_one

    result = run_clean(project, segments=do_segments, intermediates=do_intermediates)
    freed_mb = result["freed_bytes"] / 1_048_576
    console.print(
        f"[green]freed {freed_mb:.1f} MB[/] — "
        f"{len(result['removed_intermediates'])} intermediate file(s), "
        f"{len(result['removed_segments'])} segment file(s)"
    )


@app.command()
def at(
    slug: str = typer.Argument(..., help="Project slug."),
    times: list[str] = typer.Argument(
        ..., help="Timecode(s): mm:ss, mm:ss.s, hh:mm:ss or bare seconds."
    ),
    around: float | None = typer.Option(
        None, "--around", help="Also list every segment within +/- this many seconds."
    ),
    window: float = typer.Option(
        3.0, "--window", help="Seconds either side to search for transcript words/sentences."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON instead."),
) -> None:
    """Resolve a rendered timecode to its exact beat/segment/audio/caption/chapter.

    Turns a timecoded note ("at 4:27 the sentence is cut") into a precise
    reference into the edit: the plan/cut.json beat (id, kind, clip, its
    sentences and which of its shots is on screen) — which is what an edit is
    expressed against — plus the derived plan/timeline.json segment, where its
    audio actually comes from, the transcript words or voice pickup heard
    there, active captions, the music cue and the chapter.
    """
    from .inspect import TimecodeError, inspect_moment, inspect_range, parse_timecode
    from .timeline import Timeline

    project = _load(slug)
    _ensure_resolved(project)
    if not project.timeline_file.exists():
        console.print(f"[bold red]no timeline at {project.timeline_file}[/]")
        raise typer.Exit(code=2)
    timeline = Timeline.load(project.timeline_file)

    try:
        seconds = [parse_timecode(t) for t in times]
    except TimecodeError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    results = []
    for at_seconds in seconds:
        moment = inspect_moment(project, timeline, at_seconds, window=window)
        entry: dict[str, Any] = {"moment": moment.to_dict()}
        if around is not None:
            entry["around"] = inspect_range(project, timeline, at_seconds, around)
        results.append(entry)

    if as_json:
        console.print_json(json.dumps(results, ensure_ascii=False))
        return

    for entry in results:
        moment = entry["moment"]
        console.print(f"\n[bold cyan]{moment['at_tc']}[/] ({moment['at']}s)")
        beat = moment["beat"]
        if beat is not None:
            beat_table = Table(show_header=False, box=None, padding=(0, 1))
            beat_table.add_row("beat", f"{beat['id']} (uid {beat['uid']})")
            beat_table.add_row("kind", beat["kind"] + (f" / {beat['role']}" if beat["role"] else ""))
            beat_table.add_row("source", beat["clip"] or beat["file"] or "-")
            if beat["sentences"]:
                beat_table.add_row("sentences", ", ".join(beat["sentences"]))
            elif beat["words"]:
                beat_table.add_row("words", f"{beat['words'][0]}-{beat['words'][1]}")
            beat_table.add_row("on screen", beat["on_screen"]["label"])
            console.print(beat_table)
        seg = moment["segment"]
        if seg is None:
            console.print("  [dim]no video segment at this time[/]")
        else:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_row("segment", f"{seg['id']} (uid {seg['uid']})")
            table.add_row("clip", f"{seg['clip']}  [{seg['in']}-{seg['out']}]")
            table.add_row("role", seg["role"] or "-")
            table.add_row("mute_source", str(seg["mute_source"]))
            if seg["audio_from"]:
                af = seg["audio_from"]
                table.add_row("audio_from", f"{af['clip']} [{af['in']}-{af['out']}]")
            table.add_row(
                "render span", f"{seg['render_start']}s - {seg['render_end']}s"
            )
            console.print(table)
        console.print(f"  audio: [yellow]{moment['audio_description']}[/]")
        if moment["audio_clip"]:
            console.print(
                f"  audio source: {moment['audio_clip']} @ {moment['audio_clip_time']}s"
            )
        if moment["sentences_heard"]:
            console.print(f"  sentences heard: {', '.join(moment['sentences_heard'])}")
        if moment["words"]:
            words = " ".join(w["text"] for w in moment["words"])
            console.print(f"  words nearby: {words}")
        if moment["captions"]:
            for cap in moment["captions"]:
                console.print(f"  caption [{cap['style']}]: {cap['text']!r}")
        if moment["music"]:
            console.print(f"  music: {moment['music']['file']}")
        if moment["chapter"]:
            console.print(f"  chapter: {moment['chapter']}")

        if "around" in entry:
            around_table = Table(title=f"segments within +/-{around}s", header_style="bold cyan")
            for col in ("id", "uid", "clip", "in", "out", "role", "render_start", "render_end"):
                around_table.add_column(col)
            for seg_info in entry["around"]:
                around_table.add_row(*(str(seg_info[c]) for c in (
                    "id", "uid", "clip", "in", "out", "role", "render_start", "render_end"
                )))
            console.print(around_table)


@app.command(name="check-render")
def check_render(
    slug: str = typer.Argument(..., help="Project slug."),
    render: str = typer.Option(
        "draft", "--render", help="draft | preview | master, or a path to a rendered file."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-transcribe even when the cached transcript still matches."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print machine-readable JSON and write the .json report too."
    ),
) -> None:
    """Transcribe a finished render and check how its cuts actually sound.

    Lines every real audio boundary (a segment whose audio does not simply
    continue the previous one, plus every voice pickup's start and end) up
    against the render's own transcript: how much air there is on each side,
    whether a cut lands inside a word, whether the same audio is replayed.
    Always exits 0 — this is a report, not a gate (that is `ytedit qc`).
    """
    from rich.markup import escape

    from .verify import VerifyError, check_render as run_check, timecode

    project = _load(slug)
    _ensure_resolved(project)
    try:
        result = run_check(project, render=render, force=force, write_json=as_json)
    except VerifyError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    counts = result["counts"]
    table = Table(
        title=f"{Path(result['render']).name} — {counts['boundaries']} audio boundaries",
        header_style="bold cyan",
    )
    for col in ("at", "boundary", "beats", "flags", "source", "render"):
        table.add_column(col, overflow="fold")
    for row in result["boundaries"]:
        source = []
        for label, side, edge in (("A", row["before"], "out"), ("B", row["after"], "in")):
            if side and side["word"]:
                source.append(
                    f"{label} {side['clip']}@{side[edge]:.2f} "
                    f"{side['word']!r} {side['margin']:+.2f}"
                )
        phrase = "(no words both sides)"
        if row["render_gap"] is not None:
            phrase = (
                f"...{row['render_before']} [{row['render_gap']:.2f}s] "
                f"{row['render_after']}..."
            )
        if row["straddle"]:
            phrase += f" STRADDLE {row['straddle']['text']!r}"
        flags = ", ".join(row["flags"])
        table.add_row(
            timecode(row["at"]),
            row["label"],
            escape(row["beats"]),
            f"[bold red]{escape(flags)}[/]" if flags else "-",
            escape("; ".join(source)) or "-",
            escape(phrase),
        )
    console.print(table)

    extra = ", ".join(
        f"{k}: {v}" for k, v in sorted(counts.items()) if k not in ("boundaries", "flagged")
    )
    colour = "green" if not counts["flagged"] else "yellow"
    console.print(
        f"[{colour}]{counts['flagged']}/{counts['boundaries']} boundaries flagged[/]"
        + (f" ({extra})" if extra else "")
        + f" — transcript {'cached' if result['cached'] else 'fresh'}, "
        f"report: {result['report_md']}"
    )


@app.command()
def qc(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Check the timeline and the rendered master against the playbook rules."""
    fn = _lazy("ytedit.qc", "qc")
    if fn is None:
        _not_implemented("qc", "ytedit.qc")
    project = _load(slug)
    _ensure_resolved(project)
    fn(project)


@app.command()
def publish(slug: str = typer.Argument(..., help="Project slug.")) -> None:
    """Generate titles, description, chapters and thumbnail candidates."""
    fn = _lazy("ytedit.ai.publish", "publish")
    if fn is None:
        _not_implemented("publish", "ytedit.ai.publish")
    fn(_load(slug))


@app.command()
def serve(
    slug: str | None = typer.Argument(
        None, help="Build this project's missing proxies before serving."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", help="Port for the web editor."),
) -> None:
    """Run the local web editor.

    The editor streams the 720p proxies, which ingest no longer builds by
    default (see ``ingest.proxies``). Naming a project builds the ones it is
    missing first — a few minutes of ffmpeg for a whole trip, so it is never
    done for every project at once behind the user's back; without a SLUG the
    server starts immediately and a project whose proxies were never built
    shows no video until ``ytedit ingest <slug> --proxies`` has run.
    """
    from .media.ingest import ensure_proxies

    fn = _lazy("server.app", "serve")
    if fn is None:
        _not_implemented("serve", "server.app")

    if slug:
        project = _load(slug)
        built = ensure_proxies(project)
        if built:
            console.print(
                f"[dim]{slug}: built {len(built)} proxy/proxies for the editor "
                f"({', '.join(built)})[/]"
            )
    else:
        console.print(
            "[dim]serving without preparing proxies; a project with none shows no "
            "video — run 'ytedit serve <slug>' or 'ytedit ingest <slug> --proxies'.[/]"
        )
    fn(host=host, port=port)


@app.command()
def voice(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(
        False, "--force",
        help="Re-process every manifest entry (ignore the incoming-state cache) and "
        "overwrite a human-edited cut instead of writing a draft.",
    ),
) -> None:
    """Clean up narration pickups from voice/incoming/ and place them in the cut.

    Reads voice/incoming/manifest.yaml (written as a draft on the first run if
    missing, listing the cut's beats to choose from), transcribes each WAV,
    cuts retakes/instructions/stutters/pauses, trims to speech, then inserts a
    voice beat after the beat the manifest names (or the one its narration
    request points at), taking its picture from the B-roll beats that follow.
    See voice/incoming/report.md for what happened to each file.
    """
    from .ai.voice import VoiceError, run_voice

    project = _load(slug)
    try:
        result = run_voice(project, force=force)
    except VoiceError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    for row in result["rows"]:
        style = {
            "ok": "green", "skipped (cached)": "dim", "unassigned": "yellow",
            "unplaced": "yellow", "overlap_unresolved": "yellow", "error": "red",
        }.get(row.get("status", ""), "white")
        console.print(f"[{style}]{row.get('status', '?')}[/] {row['file']} ({row.get('label', '-')})")
        if row.get("error"):
            console.print(f"  [red]{row['error']}[/]")
        if row.get("placement"):
            console.print(f"  [dim]{row['placement']}[/]")
        for change in row.get("overlap") or []:
            console.print(f"  [dim]- {change}[/]")

    console.print(f"[bold]report:[/] {result['report']}")
    for issue in result.get("issues") or []:
        console.print(f"[yellow]![/] {issue}")
    if result["written"]:
        console.print(f"[green]wrote[/] {result['written']} (backup: {result['backup']})")
        if result["edited_by_human"] and not force:
            console.print(
                "[yellow]cut.json is human-edited[/] — review the draft, "
                "or re-run with --force"
            )


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
