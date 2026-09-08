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
    """Turn the footage log into an edit plan and a draft timeline."""
    fn = _lazy("ytedit.ai.plan", "plan")
    if fn is None:
        _not_implemented("plan", "ytedit.ai.plan")
    fn(_load(slug), force=force, notes=notes, from_response=from_response)


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
        f"[bold]{len(changes)}[/] change(s) · "
        f"{result['sentence_snapped']} sentence-snapped · "
        f"{result['overlaid']} overlay change(s) · "
        f"{result['deduped']} dedupe change(s) "
        f"({result['ambient_repeats']} ambient repeat(s) allowed) · duration "
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


@app.command("voice-anchor")
def voice_anchor(
    slug: str = typer.Argument(..., help="Project slug."),
    voice_id: str = typer.Argument(..., help="Voice item id (tracks.voice), e.g. v002."),
    segment_id: str = typer.Argument(..., help="Video segment id to pin it to, e.g. s014."),
    offset: float = typer.Option(
        0.0, "--offset", help="Seconds after the segment's start where the pickup begins."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite a human-edited timeline instead of writing a draft."
    ),
) -> None:
    """Pin a voice pickup to a video segment instead of an absolute time.

    The pickup then follows that segment through every later pass (speech
    padding, sentence snapping, overlay cutaways, audio dedupe, another
    ``ytedit tidy``) instead of drifting at a fixed timestamp.
    """
    from .ai.tidy import backup_timeline
    from .timeline import Timeline, VoiceAnchor

    project = _load(slug)
    if not project.timeline_file.exists():
        console.print(f"[bold red]no timeline at {project.timeline_file}[/]")
        raise typer.Exit(code=1)

    timeline = Timeline.load(project.timeline_file)
    human_edited = bool(timeline.meta.edited_by_human)

    item = next((v for v in timeline.tracks.voice if v.id == voice_id), None)
    if item is None:
        console.print(f"[bold red]no voice item {voice_id!r} in tracks.voice[/]")
        raise typer.Exit(code=1)
    if not any(s.id == segment_id for s in timeline.tracks.video):
        console.print(f"[bold red]no video segment {segment_id!r} in tracks.video[/]")
        raise typer.Exit(code=1)

    item.anchor = VoiceAnchor(segment=segment_id, offset=offset)
    timeline.resolve_anchors()

    backup = backup_timeline(project)
    write_to_draft = human_edited and not force
    target = project.plan_dir / ("timeline.draft.json" if write_to_draft else "timeline.json")
    timeline.save(target)
    console.print(
        f"[green]{voice_id}[/] anchored to {segment_id} +{offset:.2f}s "
        f"-> at {item.at:.2f}s · wrote {project.rel(target)} (backup: {backup})"
    )
    if write_to_draft:
        console.print(
            "[yellow]timeline.json is human-edited[/] — review the draft, "
            "or re-run with --force"
        )


@app.command()
def captions(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(
        False, "--force",
        help="Re-ask the writer for places and overwrite a human-edited timeline.",
    ),
    include_cold_open: bool = typer.Option(
        False, "--include-cold-open", help="Also allow a location card during the cold open."
    ),
    keep_existing: bool = typer.Option(
        False, "--keep-existing",
        help="Keep the planner's own location captions instead of replacing them.",
    ),
) -> None:
    """Place a location card at every new place, anchored to its segment."""
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
    no_music: bool = typer.Option(
        False, "--no-music", help="Ignore music cues for this render (timeline untouched)."
    ),
    no_voice: bool = typer.Option(
        False, "--no-voice", help="Ignore voice pickups for this render, symmetrically."
    ),
    fast: bool = typer.Option(
        False, "--fast",
        help="Master only: hardware-encoded (h264_videotoolbox) tier, several times "
        "faster than the default libx264 tier at some quality cost.",
    ),
) -> None:
    """Render the timeline to a preview or a master file."""
    fn = _lazy("ytedit.media.render", "render")
    if fn is None:
        _not_implemented("render", "ytedit.media.render")
    fn(
        _load(slug), preview=preview, master=master or not preview,
        no_music=no_music, no_voice=no_voice, fast=fast,
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


@app.command()
def voice(
    slug: str = typer.Argument(..., help="Project slug."),
    force: bool = typer.Option(
        False, "--force",
        help="Re-process every manifest entry (ignore the incoming-state cache) and "
        "overwrite a human-edited timeline instead of writing a draft.",
    ),
) -> None:
    """Clean up narration pickups from voice/incoming/ and place them on the timeline.

    Reads voice/incoming/manifest.yaml (written as a draft on the first run if
    missing), transcribes each WAV, cuts retakes/instructions/stutters/pauses,
    trims to speech, and places the result at its narration request's beat or
    an explicit anchor — growing the muted picture underneath (or pulling in
    manifest broll_pool clips) when the pickup runs long. See
    voice/incoming/report.md for what happened to each file.
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
                "[yellow]timeline.json is human-edited[/] — review the draft, "
                "or re-run with --force"
            )


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
