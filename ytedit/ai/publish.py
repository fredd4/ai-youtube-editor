"""Publish pack: titles, description, chapters, tags and thumbnail candidates.

Reads ``plan/edit_plan.json`` + ``plan/timeline.json``, asks the writer model for
the public-facing copy, then validates everything deterministically against the
research rules (25-29) before anything reaches the user:

* :func:`validate_title` — 55-70 characters, at least one of
  {number, curiosity gap, superlative, transformation}, no keyword repetition.
* :func:`validate_chapters` / :func:`format_chapters` — first chapter at 0:00,
  at least three, at least 10 s apart, strictly ascending.
* :func:`validate_thumbnail_text` — at most five words, Polish diacritics kept.

Thumbnails are produced twice over: a base frame is always cut out of the real
source clip with ffmpeg, a **local** PIL composite is always rendered (so there
is a usable candidate even with no fal key and no spend), and the fal
``nano-banana-pro/edit`` pass runs on top only when it is asked for and paid for.
A 120 px preview strip is written for the "legible on mobile" sign-off.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from ytedit.ai.fal import NANO_BANANA_USD_PER_IMAGE, Fal
from ytedit.ai.openrouter import OpenRouter
from ytedit.ai.plan import footage_entries, load_footage_log, units_for_ledger
from ytedit.ai.prompts import render
from ytedit.config import Settings
from ytedit.costs import charge, check_budget
from ytedit.log import get_logger
from ytedit.media.ffmpeg import ff
from ytedit.project import Project, utcnow
from ytedit.timeline import Timeline

log = get_logger(__name__)

STAGE = "publish"

THUMB_WIDTH = 1280
THUMB_HEIGHT = 720
PREVIEW_HEIGHT = 120

#: Output-token budget for the writer. A Polish description plus five titles,
#: chapters, tags and three thumbnail concepts does not fit in a small budget, and
#: a reply cut mid-JSON is unusable; overridable via ``publish.max_tokens``.
DEFAULT_PUBLISH_MAX_TOKENS = 12000

TITLE_MIN = 55
TITLE_MAX = 70
TITLE_HARD_MAX = 100
MAX_THUMB_WORDS = 5
MAX_TAGS = 15

#: Rule 27 formulas, with the Polish variants that actually occur in titles.
NUMBER_RE = re.compile(r"\d")
CURIOSITY_RE = re.compile(
    r"\bdlaczego\b|\bjak\s|\bco\s+si[eę]\b|\bnikt\b|\bsekret\w*|\bnie\s+uwierzysz\b"
    r"|\bnaprawd[eę]\b|\bczego\b|\bokazuje\s+si[eę]\b|\bnie\s+spodziewa\w*"
    r"|\bwhy\b|\bhow\b|\bwhat\s+happen\w*|\bnobody\b|\bsecret\b|\?",
    re.IGNORECASE,
)
SUPERLATIVE_RE = re.compile(
    r"\bnaj\w+|\bjedyn\w+|\bszalon\w+|\bekstremaln\w+|\bnigdy\b"
    r"|\bbest\b|\bworst\b|\bmost\b|\bcheapest\b|\bonly\b|\bever\b",
    re.IGNORECASE,
)
TRANSFORMATION_RE = re.compile(
    r"\bod\s+\w+\s+do\s+\w+|\bz\s+\w+\s+do\s+\w+|\bzmieni\w*|\bprzemian\w*"
    r"|\bw\s+\d+\s*(?:godzin\w*|dni|dzie[nń]|minut\w*)"
    r"|\bfrom\s+\w+\s+to\s+\w+|\bturned\s+into\b|\btransform\w*",
    re.IGNORECASE,
)

#: Words too common to count as a repeated keyword.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "your", "you",
    "jest", "sie", "się", "nie", "tak", "jak", "ale", "oraz", "przez", "dla",
    "czy", "juz", "już", "tym", "tego", "ktore", "które", "moze", "może", "byl",
    "był", "była", "byla", "jego", "jej", "ich", "sam", "tylko", "bardzo",
}


class PublishError(RuntimeError):
    """Raised when the publish stage cannot run or the model output is unusable."""


# ----------------------------------------------------------------------
# model output
# ----------------------------------------------------------------------
class _Lenient(BaseModel):
    """Base model for LLM output: unknown keys kept, aliases accepted."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class TitleCandidate(_Lenient):
    """One title option with the formula it leans on."""

    title: str = ""
    formula: str = ""
    chars: int = 0


class PublishChapter(_Lenient):
    """A YouTube chapter for the description."""

    at: float = 0.0
    title: str = ""


class ThumbnailVariant(_Lenient):
    """A thumbnail concept bound to a real frame of the footage."""

    text: str = ""
    concept: str = ""
    colors: list[str] = Field(default_factory=list)
    frame_clip: str = ""
    frame_t: float = 0.0
    image_prompt: str = ""


class PublishPack(_Lenient):
    """The full publish pack, normalized from whatever the writer returned."""

    titles: list[TitleCandidate] = Field(default_factory=list)
    description: str = ""
    chapters: list[PublishChapter] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    thumbnail_variants: list[ThumbnailVariant] = Field(default_factory=list)
    test_and_compare: str = ""

    @classmethod
    def from_llm(cls, raw: Any) -> "PublishPack":
        """Normalize the writer's answer (both prompts.md and flat shapes)."""
        if isinstance(raw, list):
            # A reply cut mid-JSON often salvages as a bare array; keep it if it
            # is clearly the title list, otherwise say what actually happened.
            if raw and all(isinstance(item, (str, dict)) for item in raw):
                raw = {"titles": raw}
            else:
                raise PublishError(
                    "the writer returned a bare JSON array, not the publish pack — "
                    "this usually means the answer was cut off; raise "
                    "publish.max_tokens in the settings"
                )
        if not isinstance(raw, dict):
            raise PublishError(f"writer returned {type(raw).__name__}, expected a JSON object")

        titles: list[dict[str, Any]] = []
        for item in raw.get("titles") or raw.get("title_candidates") or []:
            if isinstance(item, str):
                titles.append({"title": item, "formula": "", "chars": len(item)})
            elif isinstance(item, dict):
                text = str(item.get("title") or item.get("text") or "")
                titles.append(
                    {
                        "title": text,
                        "formula": str(item.get("formula") or item.get("rationale") or ""),
                        "chars": int(item.get("chars") or len(text)),
                    }
                )

        chapters: list[dict[str, Any]] = []
        for item in raw.get("chapters") or []:
            if not isinstance(item, dict):
                continue
            at = item.get("at", item.get("at_seconds", item.get("t", 0.0)))
            chapters.append({"at": _num(at), "title": str(item.get("title") or "")})

        variants: list[dict[str, Any]] = []
        for item in raw.get("thumbnail_variants") or raw.get("thumbnail_prompts") or []:
            if not isinstance(item, dict):
                continue
            hint = str(item.get("source_frame_hint") or item.get("frame") or "")
            clip, seconds = _parse_frame_hint(hint)
            variants.append(
                {
                    "text": str(item.get("text") or item.get("on_image_text") or ""),
                    "concept": str(item.get("concept") or item.get("description") or ""),
                    "colors": [str(c) for c in (item.get("colors") or []) if c],
                    "frame_clip": str(item.get("frame_clip") or clip),
                    "frame_t": _num(item.get("frame_t", seconds)),
                    "image_prompt": str(item.get("image_prompt") or ""),
                }
            )

        tags = [str(t).strip() for t in (raw.get("tags") or []) if str(t).strip()]
        return cls.model_validate(
            {
                "titles": titles,
                "description": str(raw.get("description") or ""),
                "chapters": chapters,
                "tags": tags[:MAX_TAGS],
                "thumbnail_variants": variants,
                "test_and_compare": str(
                    raw.get("test_and_compare") or raw.get("test_and_compare_note") or ""
                ),
            }
        )


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_frame_hint(hint: str) -> tuple[str, float]:
    """Parse ``"c003@12.4"`` / ``"clip c003 at 12.4s"`` into ``(clip, seconds)``."""
    clip_match = re.search(r"\b(c\d{3,})\b", hint)
    time_match = re.search(r"(?:@|\bat\s+|\bt=)\s*(\d+(?:\.\d+)?)", hint)
    return (clip_match.group(1) if clip_match else ""), (
        float(time_match.group(1)) if time_match else 0.0
    )


# ----------------------------------------------------------------------
# validators (also imported by ytedit.qc)
# ----------------------------------------------------------------------
def validate_title(title: str, language: str = "pl") -> list[str]:
    """Check one title against research rule 27.

    Args:
        title: The candidate title.
        language: Project language; Polish runs 15-20% longer than English, so
            the over-length message says so explicitly.

    Returns:
        A list of issues; empty means the title passes. Keyword *position*
        cannot be checked without knowing the primary keyword, so it is skipped
        here and left to the human review.
    """
    issues: list[str] = []
    text = (title or "").strip()
    if not text:
        return ["empty title"]

    length = len(text)
    if length < TITLE_MIN:
        issues.append(f"{length} chars — under the {TITLE_MIN}-{TITLE_MAX} target")
    elif length > TITLE_HARD_MAX:
        issues.append(f"{length} chars — over the {TITLE_HARD_MAX} hard maximum")
    elif length > TITLE_MAX:
        issues.append(
            f"{length} chars — over the {TITLE_MAX} target"
            + (f" ({language} runs long, aim shorter)" if language == "pl" else "")
        )

    formulas = [
        name
        for name, pattern in (
            ("number", NUMBER_RE),
            ("curiosity", CURIOSITY_RE),
            ("superlative", SUPERLATIVE_RE),
            ("transformation", TRANSFORMATION_RE),
        )
        if pattern.search(text)
    ]
    if not formulas:
        issues.append(
            "none of {number, curiosity gap, superlative, transformation} detected"
        )

    words = [w for w in re.findall(r"\w{4,}", text.lower()) if w not in _STOPWORDS]
    repeated = sorted({w for w in words if words.count(w) > 1})
    if repeated:
        issues.append(f"repeated keyword(s): {', '.join(repeated)}")
    return issues


def title_formulas(title: str) -> list[str]:
    """Return which of the four title formulas a title uses."""
    return [
        name
        for name, pattern in (
            ("number", NUMBER_RE),
            ("curiosity", CURIOSITY_RE),
            ("superlative", SUPERLATIVE_RE),
            ("transformation", TRANSFORMATION_RE),
        )
        if pattern.search(title or "")
    ]


def validate_chapters(
    chapters: Sequence[Any], duration: float | None = None
) -> list[str]:
    """Check a chapter list against research rule 25.

    Args:
        chapters: ``{"at": seconds, "title": str}`` mappings or objects with
            ``at``/``title`` attributes.
        duration: Programme length; when given, chapters past the end and a last
            chapter with under 10 s of runtime left are flagged.

    Returns:
        A list of issues; empty means the chapters are valid for YouTube.
    """
    items = [_chapter_pair(c) for c in chapters]
    issues: list[str] = []
    if len(items) < 3:
        issues.append(f"only {len(items)} chapters — YouTube needs at least 3")
    if not items:
        return issues

    if abs(items[0][0]) > 1e-6:
        issues.append(f"first chapter is at {format_timecode(items[0][0])}, must be 0:00")
    for i, (at, title) in enumerate(items):
        if not str(title).strip():
            issues.append(f"chapter {i + 1} has no title")
        elif re.fullmatch(r"(cz[eę][sś][cć]|part|rozdzia[lł])\s*\d+", str(title).strip(), re.I):
            issues.append(f"chapter {i + 1} {title!r} is a generic label — use keywords")
        if at < -1e-6:
            issues.append(f"chapter {i + 1} has a negative timecode")
        if duration is not None and at > duration + 1e-6:
            issues.append(
                f"chapter {i + 1} at {format_timecode(at)} is past the end "
                f"({format_timecode(duration)})"
            )
    for (a_at, _), (b_at, b_title) in zip(items, items[1:]):
        if b_at <= a_at:
            issues.append(f"chapter {b_title!r} at {format_timecode(b_at)} is not ascending")
        elif b_at - a_at < 10.0:
            issues.append(
                f"chapter {b_title!r} is only {b_at - a_at:.0f}s after the previous one "
                "(minimum 10s)"
            )
    if duration is not None and items[-1][0] > duration - 10.0:
        issues.append("last chapter leaves under 10s of runtime")
    return issues


def format_chapters(chapters: Sequence[Any]) -> str:
    """Render chapters as the ``m:ss Title`` block that goes in the description."""
    return "\n".join(
        f"{format_timecode(at)} {str(title).strip()}"
        for at, title in (_chapter_pair(c) for c in chapters)
    )


def _chapter_pair(chapter: Any) -> tuple[float, str]:
    """Coerce a chapter (mapping, pydantic model or tuple) into ``(at, title)``."""
    if isinstance(chapter, dict):
        return _num(chapter.get("at", chapter.get("at_seconds", 0.0))), str(
            chapter.get("title", "")
        )
    if isinstance(chapter, (tuple, list)) and len(chapter) >= 2:
        return _num(chapter[0]), str(chapter[1])
    return _num(getattr(chapter, "at", 0.0)), str(getattr(chapter, "title", ""))


def format_timecode(seconds: float) -> str:
    """Format seconds as ``m:ss`` (``h:mm:ss`` past an hour) for YouTube."""
    total = int(round(max(0.0, float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def validate_thumbnail_text(text: str) -> list[str]:
    """Check on-image thumbnail text against research rule 28.

    At most five words (fewer is better) and legible at 120 px. Polish
    diacritics are explicitly allowed — only characters no caption font can be
    trusted with (emoji, symbols) are flagged.
    """
    issues: list[str] = []
    stripped = (text or "").strip()
    if not stripped:
        return ["empty thumbnail text"]
    words = stripped.split()
    if len(words) > MAX_THUMB_WORDS:
        issues.append(f"{len(words)} words — maximum {MAX_THUMB_WORDS}")
    if len(stripped) > 30:
        issues.append(f"{len(stripped)} characters — too long to read at 120px")
    if any(len(w) > 14 for w in words):
        issues.append("a word longer than 14 characters will not fit legibly")
    illegal = sorted({c for c in stripped if not (c.isalnum() or c.isspace() or c in "!?.,:'-–—&%€$")})
    if illegal:
        issues.append(f"characters that may not render: {' '.join(illegal)}")
    return issues


# ----------------------------------------------------------------------
# thumbnails
# ----------------------------------------------------------------------
def extract_frame(source: Path, at: float, dest: Path) -> Path:
    """Cut a single 1280x720 JPEG out of a source clip with ffmpeg.

    Args:
        source: A normalized clip under ``media/sources``.
        at: Timestamp in seconds.
        dest: Output JPEG path.

    Returns:
        ``dest``.

    Raises:
        PublishError: When ffmpeg produced no file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    ff(
        "-ss", f"{max(0.0, float(at)):.3f}",
        "-i", str(source),
        "-frames:v", "1",
        "-vf",
        f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={THUMB_WIDTH}:{THUMB_HEIGHT}",
        "-q:v", "2",
        str(dest),
    )
    if not dest.exists():
        raise PublishError(f"ffmpeg produced no frame for {source.name} @ {at}s")
    return dest


def render_local_thumbnail(
    base: Path,
    text: str,
    dest: Path,
    font_file: Path | None = None,
) -> Path:
    """Compose the offline fallback thumbnail: punchier grade + bold headline.

    The text is drawn white with a heavy black outline in the upper-left third,
    clear of YouTube's bottom-right duration badge, at most two lines. The font
    comes from ``caption_styles.yaml`` so Polish diacritics render exactly as
    they will in the burned-in captions.

    Args:
        base: The extracted source frame.
        text: On-image text (already validated to <= 5 words).
        dest: Output JPEG path.
        font_file: Override the caption font (tests, non-macOS hosts).

    Returns:
        ``dest``, always ``1280x720``.
    """
    image = Image.open(base).convert("RGB")
    if image.size != (THUMB_WIDTH, THUMB_HEIGHT):
        image = image.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
    image = ImageEnhance.Color(image).enhance(1.18)
    image = ImageEnhance.Contrast(image).enhance(1.12)
    image = ImageEnhance.Brightness(image).enhance(1.02)

    words = (text or "").strip().split()
    lines = _wrap_words(words, max_lines=2)
    if lines:
        draw = ImageDraw.Draw(image)
        font_size = 132 if len(lines) == 1 else 108
        font = _load_font(font_file, font_size)
        # Shrink until the widest line fits inside a 5% side-safe area.
        max_width = int(THUMB_WIDTH * 0.86)
        while font_size > 40:
            widest = max(draw.textlength(line, font=font) for line in lines)
            if widest <= max_width:
                break
            font_size -= 6
            font = _load_font(font_file, font_size)

        y = int(THUMB_HEIGHT * 0.10)
        x = int(THUMB_WIDTH * 0.06)
        stroke = max(4, font_size // 14)
        for line in lines:
            draw.text(
                (x, y),
                line,
                font=font,
                fill=(255, 255, 255),
                stroke_width=stroke,
                stroke_fill=(0, 0, 0),
            )
            y += int(font_size * 1.12)

    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="JPEG", quality=92)
    return dest


def _wrap_words(words: Sequence[str], max_lines: int = 2) -> list[str]:
    """Split at most five words across up to ``max_lines`` balanced lines."""
    if not words:
        return []
    if len(words) <= 2 or max_lines <= 1:
        return [" ".join(words)]
    split = (len(words) + 1) // 2
    return [" ".join(words[:split]), " ".join(words[split:])]


def _load_font(font_file: Path | None, size: int) -> Any:
    """Load a bold TrueType face, falling back to PIL's bitmap default."""
    candidates = [font_file] if font_file else []
    candidates += [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:  # pragma: no cover - unusable font file
                continue
    return ImageFont.load_default()  # pragma: no cover - no TrueType on host


def preview_strip(images: Iterable[Path], dest: Path, height: int = PREVIEW_HEIGHT) -> Path:
    """Write the side-by-side 120 px preview used for the mobile legibility check."""
    tiles: list[Image.Image] = []
    for path in images:
        if not Path(path).exists():
            continue
        img = Image.open(path).convert("RGB")
        width = max(1, int(img.width * height / img.height))
        tiles.append(img.resize((width, height), Image.LANCZOS))
    if not tiles:
        raise PublishError("no thumbnails to build a preview strip from")

    gap = 8
    strip = Image.new(
        "RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), height), (16, 16, 16)
    )
    x = 0
    for tile in tiles:
        strip.paste(tile, (x, 0))
        x += tile.width + gap
    dest.parent.mkdir(parents=True, exist_ok=True)
    strip.save(dest, format="JPEG", quality=90)
    return dest


# ----------------------------------------------------------------------
# prompt inputs
# ----------------------------------------------------------------------
def _story_outline(edit_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Story beats from ``edit_plan.json`` (normalized ``plan`` section first)."""
    plan_section = edit_plan.get("plan") if isinstance(edit_plan.get("plan"), dict) else edit_plan
    story = plan_section.get("story") if isinstance(plan_section.get("story"), dict) else {}
    beats = story.get("beats") or plan_section.get("story_outline") or []
    return [b for b in beats if isinstance(b, dict)]


def _hooks_and_numbers(footage_log: dict[str, Any]) -> dict[str, list[str]]:
    """Collect every hook line and concrete number across the footage log."""
    hooks: list[str] = []
    numbers: list[str] = []
    topics: list[str] = []
    for entry in footage_entries(footage_log):
        hooks += [str(h) for h in (entry.get("hooks") or []) if h]
        numbers += [str(n) for n in (entry.get("numbers") or []) if n]
        topics += [str(t) for t in (entry.get("topics") or []) if t]
    return {
        "hooks": _dedupe(hooks),
        "numbers": _dedupe(numbers),
        "topics": _dedupe(topics),
    }


def _locations(footage_log: dict[str, Any]) -> list[dict[str, Any]]:
    """Locations in footage order, de-duplicated by name."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in footage_entries(footage_log):
        location = entry.get("location")
        if not isinstance(location, dict):
            continue
        name = str(location.get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(
            {
                "clip": entry.get("clip"),
                "name": name,
                "city": location.get("city"),
                "country": location.get("country"),
            }
        )
    return out


def candidate_frames(footage_log: dict[str, Any]) -> list[dict[str, Any]]:
    """Thumbnail-candidate frames from the footage log's ``visual`` blocks."""
    out: list[dict[str, Any]] = []
    for entry in footage_entries(footage_log):
        visual = entry.get("visual")
        if not isinstance(visual, dict):
            continue
        frames = [_num(f) for f in (visual.get("best_frames") or [])]
        if not frames:
            continue
        out.append(
            {
                "clip": entry.get("clip"),
                "best_frames": frames,
                "thumbnail_candidate": bool(visual.get("thumbnail_candidate")),
                "quality": visual.get("quality"),
                "summary": entry.get("summary", ""),
            }
        )
    out.sort(key=lambda item: (not item["thumbnail_candidate"], -_num(item.get("quality"), 0.0)))
    return out


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value.strip())
    return out


# ----------------------------------------------------------------------
# stage entry point
# ----------------------------------------------------------------------
def publish(
    project: Project,
    thumbnails: bool = True,
    n_thumbs: int = 3,
    notes: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Run the publish stage.

    Args:
        project: Project with a rendered plan (``plan/edit_plan.json`` and
            ``plan/timeline.json`` must exist).
        thumbnails: Run the paid fal ``nano-banana-pro/edit`` pass on top of the
            local composites. Base frames and local composites are always made.
        n_thumbs: Number of variants (3 is what Test & Compare needs).
        notes: Extra direction appended to the user prompt.
        max_tokens: Output-token cap for the writer call. Defaults to
            ``publish.max_tokens`` in the settings, else
            :data:`DEFAULT_PUBLISH_MAX_TOKENS`.

    Returns:
        A summary dict with the written files, validation results and spend.

    Raises:
        PublishError: When the plan or timeline is missing, or the writer model
            returns nothing usable.
    """
    settings = project.settings
    if not project.edit_plan_file.exists():
        raise PublishError(
            f"no edit plan at {project.edit_plan_file} — run `ytedit plan {project.slug}` first"
        )
    if not project.timeline_file.exists():
        raise PublishError(
            f"no timeline at {project.timeline_file} — run `ytedit plan {project.slug}` first"
        )
    edit_plan = json.loads(project.edit_plan_file.read_text(encoding="utf-8"))
    timeline = Timeline.load(project.timeline_file)
    try:
        footage_log = load_footage_log(project)
    except Exception as exc:  # publishing without the log is possible, just poorer
        log.warning("publish: no footage log (%s); hooks/locations will be empty", exc)
        footage_log = {"clips": []}

    project.set_stage(STAGE, "running")
    try:
        result = _ask_writer(
            project=project,
            settings=settings,
            edit_plan=edit_plan,
            timeline=timeline,
            footage_log=footage_log,
            notes=notes,
            max_tokens=int(
                max_tokens
                if max_tokens is not None
                else settings.get("publish.max_tokens", DEFAULT_PUBLISH_MAX_TOKENS)
            ),
        )
        # Dump the raw answer before parsing: the call is already paid for, and an
        # off-schema or truncated reply is only debuggable if it reached disk.
        project.exports_dir.mkdir(parents=True, exist_ok=True)
        (project.exports_dir / "writer_response.json").write_text(
            json.dumps(result["json"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pack = PublishPack.from_llm(result["json"])
        if not pack.titles:
            raise PublishError("the writer returned no titles")

        duration = timeline.duration()
        pack = _apply_fallbacks(pack, edit_plan, timeline, footage_log)

        title_issues = {
            candidate.title: validate_title(candidate.title, project.language)
            for candidate in pack.titles
        }
        chapter_issues = validate_chapters(pack.chapters, duration)
        thumb_issues = {
            variant.text: validate_thumbnail_text(variant.text)
            for variant in pack.thumbnail_variants
        }

        thumbs = _make_thumbnails(
            project=project,
            settings=settings,
            pack=pack,
            footage_log=footage_log,
            generate=thumbnails,
            n_thumbs=n_thumbs,
        )

        document = {
            "project": project.slug,
            "generated": utcnow(),
            "model": result["model"],
            "language": project.language,
            "duration_s": duration,
            "titles": [t.model_dump(mode="json") for t in pack.titles],
            "title_issues": title_issues,
            "description": pack.description,
            "chapters": [c.model_dump(mode="json") for c in pack.chapters],
            "chapters_block": format_chapters(pack.chapters),
            "chapter_issues": chapter_issues,
            "tags": pack.tags,
            "thumbnail_variants": [v.model_dump(mode="json") for v in pack.thumbnail_variants],
            "thumbnail_text_issues": thumb_issues,
            "thumbnails": thumbs,
            "test_and_compare": pack.test_and_compare or TEST_AND_COMPARE_DEFAULT,
            "cost_usd": round(result["cost_usd"] + thumbs.get("cost_usd", 0.0), 6),
        }
        project.exports_dir.mkdir(parents=True, exist_ok=True)
        json_path = project.exports_dir / "publish.json"
        json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path = project.exports_dir / "publish.md"
        md_path.write_text(render_publish_md(document, project), encoding="utf-8")

        project.set_stage(
            STAGE,
            "done",
            cost_usd=document["cost_usd"],
            titles=len(pack.titles),
            thumbnails=len(thumbs.get("variants", [])),
        )
        log.info(
            "[clip]%s[/] publish: %d titles, %d chapters, %d thumbnails, $%.4f",
            project.slug,
            len(pack.titles),
            len(pack.chapters),
            len(thumbs.get("variants", [])),
            document["cost_usd"],
        )
        return {
            "publish_json": str(json_path),
            "publish_md": str(md_path),
            "titles": [t.title for t in pack.titles],
            "title_issues": title_issues,
            "chapter_issues": chapter_issues,
            "thumbnails": thumbs,
            "cost_usd": document["cost_usd"],
        }
    except Exception as exc:
        project.set_stage(STAGE, "error", error=str(exc)[:500])
        raise


TEST_AND_COMPARE_DEFAULT = (
    "Upload 3 titles and 3 thumbnails to YouTube Test & Compare. Let it run about "
    "2 weeks or ~1,000-5,000 impressions per variant. The winner is the variant with "
    "the highest watch time per impression, not the highest raw CTR."
)


def _apply_fallbacks(
    pack: PublishPack,
    edit_plan: dict[str, Any],
    timeline: Timeline,
    footage_log: dict[str, Any],
) -> PublishPack:
    """Fill in anything the writer left out from the plan and the footage log."""
    if not pack.chapters and timeline.chapters:
        pack.chapters = [
            PublishChapter(at=chapter.at, title=chapter.title) for chapter in timeline.chapters
        ]
    if not pack.tags:
        extras = _hooks_and_numbers(footage_log)
        locations = [item["name"] for item in _locations(footage_log)]
        pack.tags = _dedupe(locations + extras["topics"])[:MAX_TAGS]

    plan_section = edit_plan.get("plan") if isinstance(edit_plan.get("plan"), dict) else {}
    concepts = plan_section.get("thumbnail_concepts") or []
    frames = candidate_frames(footage_log)
    for index, variant in enumerate(pack.thumbnail_variants):
        if not variant.text and index < len(concepts) and isinstance(concepts[index], dict):
            variant.text = str(concepts[index].get("text") or "")
        if not variant.frame_clip:
            if index < len(concepts) and isinstance(concepts[index], dict):
                variant.frame_clip = str(concepts[index].get("frame_clip") or "")
                variant.frame_t = _num(concepts[index].get("frame_t"), variant.frame_t)
        if not variant.frame_clip and frames:
            source = frames[min(index, len(frames) - 1)]
            variant.frame_clip = str(source["clip"])
            variant.frame_t = source["best_frames"][0]
    if not pack.thumbnail_variants and concepts:
        pack.thumbnail_variants = [
            ThumbnailVariant(
                text=str(c.get("text") or ""),
                concept=str(c.get("concept") or ""),
                colors=[str(x) for x in (c.get("colors") or [])],
                frame_clip=str(c.get("frame_clip") or ""),
                frame_t=_num(c.get("frame_t")),
            )
            for c in concepts
            if isinstance(c, dict)
        ]
    return pack


def _make_thumbnails(
    project: Project,
    settings: Settings,
    pack: PublishPack,
    footage_log: dict[str, Any],
    generate: bool,
    n_thumbs: int,
) -> dict[str, Any]:
    """Extract base frames, render local composites, optionally run fal.

    Returns:
        ``{"dir", "variants": [...], "preview", "cost_usd", "errors"}``.
    """
    out_dir = project.exports_dir / "thumbnails"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = pack.thumbnail_variants[: max(1, n_thumbs)]
    locations = {item["clip"]: item for item in _locations(footage_log)}
    _, font_file = settings.caption_font()

    fal_key = settings.keys.get("fal")
    fal_client: Fal | None = None
    fal_cost = 0.0
    if generate and variants:
        estimate = len(variants) * NANO_BANANA_USD_PER_IMAGE
        log.info(
            "publish: %d fal thumbnails at $%.2f each = $%.2f estimated",
            len(variants),
            NANO_BANANA_USD_PER_IMAGE,
            estimate,
        )
        if not fal_key:
            log.warning("publish: no FAL_KEY — local thumbnails only")
        else:
            check_budget(project, estimate)

            def cost_callback(
                service: str,
                op: str,
                model: str,
                units: Any = None,
                usd: float = 0.0,
                **extra: Any,
            ) -> None:
                charge(
                    project,
                    service,
                    op,
                    units_for_ledger(units),
                    usd=usd,
                    model=model,
                    stage=STAGE,
                    **extra,
                )

            fal_client = Fal(api_key=fal_key, cost_callback=cost_callback)

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    local_paths: list[Path] = []

    for index, variant in enumerate(variants, start=1):
        record: dict[str, Any] = {
            "n": index,
            "text": variant.text,
            "concept": variant.concept,
            "colors": variant.colors,
            "frame_clip": variant.frame_clip,
            "frame_t": variant.frame_t,
            "issues": validate_thumbnail_text(variant.text),
        }
        source = project.source_path(variant.frame_clip) if variant.frame_clip else None
        if source is None or not source.exists():
            errors.append(f"variant {index}: no source clip for {variant.frame_clip!r}")
            records.append(record)
            continue

        base = out_dir / f"base_{index}.jpg"
        try:
            extract_frame(source, variant.frame_t, base)
            record["base"] = project.rel(base)
        except Exception as exc:
            errors.append(f"variant {index}: frame extraction failed ({exc})")
            records.append(record)
            continue

        local = out_dir / f"local_{index}.jpg"
        try:
            render_local_thumbnail(base, variant.text, local, font_file)
            record["local"] = project.rel(local)
            local_paths.append(local)
        except Exception as exc:  # pragma: no cover - PIL failure
            errors.append(f"variant {index}: local render failed ({exc})")

        location = locations.get(variant.frame_clip, {})
        prompt = variant.image_prompt or render(
            "thumbnail.image_prompt",
            location_name=location.get("name") or project.title,
            country_name=location.get("country") or "",
            on_image_text=variant.text,
            project_language=project.language,
            text_position="upper-left, clear of the subject's face",
            colors_csv=", ".join(variant.colors) or "white, deep blue, warm yellow",
        )
        record["image_prompt"] = prompt

        if fal_client is not None:
            try:
                generated = fal_client.thumbnail_edit(
                    prompt=prompt,
                    image_paths=[base],
                    n=1,
                    aspect_ratio="16:9",
                    resolution="2K",
                    out_dir=out_dir,
                )
                if generated:
                    final = out_dir / f"fal_{index}.jpg"
                    generated[0].replace(final)
                    record["fal"] = project.rel(final)
                    fal_cost += NANO_BANANA_USD_PER_IMAGE
            except Exception as exc:
                errors.append(f"variant {index}: fal generation failed ({exc})")
        records.append(record)

    preview: str | None = None
    strip_sources = [
        Path(project.path / r["fal"]) if r.get("fal") else Path(project.path / r["local"])
        for r in records
        if r.get("fal") or r.get("local")
    ]
    if strip_sources:
        try:
            preview = project.rel(
                preview_strip(strip_sources, out_dir / "preview_120px.jpg")
            )
        except Exception as exc:  # pragma: no cover - PIL failure
            errors.append(f"preview strip failed ({exc})")

    return {
        "dir": project.rel(out_dir),
        "variants": records,
        "preview": preview,
        "cost_usd": round(fal_cost, 6),
        "errors": errors,
    }


def _ask_writer(
    project: Project,
    settings: Settings,
    edit_plan: dict[str, Any],
    timeline: Timeline,
    footage_log: dict[str, Any],
    notes: str | None,
    max_tokens: int = DEFAULT_PUBLISH_MAX_TOKENS,
    client: OpenRouter | None = None,
) -> dict[str, Any]:
    """Render the publish prompts and call the writer model once."""
    model = settings.model("writer")
    compact = {"separators": (",", ":"), "ensure_ascii": False}
    system_prompt = render("publish.system", project_language=project.language)
    user_prompt = render(
        "publish.user",
        project_language=project.language,
        story_outline_json=json.dumps(_story_outline(edit_plan), **compact),
        hooks_and_numbers_json=json.dumps(_hooks_and_numbers(footage_log), **compact),
        locations_json=json.dumps(_locations(footage_log), **compact),
        markers_and_chapters_json=json.dumps(
            {
                "duration_s": round(timeline.duration(), 2),
                "markers": [m.model_dump(mode="json") for m in timeline.markers],
                "chapters": [c.model_dump(mode="json") for c in timeline.chapters],
            },
            **compact,
        ),
        thumbnail_candidate_frames_json=json.dumps(candidate_frames(footage_log), **compact),
    )
    if notes:
        user_prompt += f"\n\nEditor notes: {notes.strip()}"

    spend: list[float] = []

    def cost_callback(
        service: str, op: str, model: str, units: Any = None, usd: float = 0.0, **extra: Any
    ) -> None:
        spend.append(float(usd))
        charge(
            project,
            service,
            op,
            units_for_ledger(units),
            usd=usd,
            model=model,
            stage=STAGE,
            **extra,
        )

    owned = client is None
    api = client or OpenRouter(
        api_key=settings.require_key("openrouter"), cost_callback=cost_callback
    )
    try:
        answer = api.ask_json(
            model,
            system_prompt,
            user_prompt,
            SCHEMA_HINT,
            temperature=0.6,
            max_tokens=max_tokens,
        )
    finally:
        if owned:
            api.close()
    return {"json": answer, "model": model, "cost_usd": round(sum(spend), 6)}


#: Appended to the system prompt by ``ask_json``. ``publish.system`` in prompts.md
#: carries the one and only schema; this is a pointer to it, not a second copy.
SCHEMA_HINT = "Return ONLY the JSON object described in the system prompt."


# ----------------------------------------------------------------------
# markdown report
# ----------------------------------------------------------------------
def render_publish_md(document: dict[str, Any], project: Project) -> str:
    """Render the paste-ready ``exports/publish.md``."""
    lines: list[str] = [
        f"# Publish pack — {project.slug}",
        "",
        f"Generated: {document['generated']} · language: `{document['language']}` · "
        f"runtime: {format_timecode(document['duration_s'])} · "
        f"model: `{document['model']}` · cost: ${document['cost_usd']:.4f}",
        "",
        "## Titles",
        "",
        "| # | chars | formulas | title | validator |",
        "|---|---|---|---|---|",
    ]
    for i, title in enumerate(document["titles"], start=1):
        text = title["title"]
        issues = document["title_issues"].get(text) or []
        verdict = "✅ ok" if not issues else "⚠️ " + "; ".join(issues)
        formulas = ", ".join(title_formulas(text)) or "—"
        lines.append(
            f"| {i} | {len(text)} | {formulas} | {text.replace('|', '\\|')} | {verdict} |"
        )
    lines.append("")

    lines += ["## Description (paste as-is)", "", "```text", document["description"].strip(), "```", ""]

    lines += ["## Chapters", ""]
    chapter_issues = document["chapter_issues"]
    lines.append("✅ valid" if not chapter_issues else "⚠️ " + "; ".join(chapter_issues))
    lines += ["", "```text", document["chapters_block"], "```", ""]

    lines += ["## Tags", "", ", ".join(document["tags"]) or "_none_", ""]

    lines += ["## Thumbnails", ""]
    thumbs = document["thumbnails"]
    for record in thumbs.get("variants", []):
        issues = record.get("issues") or []
        verdict = "✅ ok" if not issues else "⚠️ " + "; ".join(issues)
        lines += [
            f"### Variant {record['n']} — “{record.get('text', '')}” ({verdict})",
            "",
            f"- Concept: {record.get('concept') or '—'}",
            f"- Colors: {', '.join(record.get('colors') or []) or '—'}",
            f"- Source frame: `{record.get('frame_clip', '—')}` @ {record.get('frame_t', 0):.1f}s",
        ]
        for key, label in (("base", "Base frame"), ("local", "Local composite"), ("fal", "fal")):
            if record.get(key):
                lines.append(f"- {label}: `{record[key]}`")
        lines.append("")
    if thumbs.get("preview"):
        lines += [
            f"120 px legibility strip: `{thumbs['preview']}` — open it and confirm every "
            "variant still reads at mobile size before presenting them.",
            "",
        ]
    if thumbs.get("errors"):
        lines += ["**Thumbnail problems:**", ""] + [f"- {e}" for e in thumbs["errors"]] + [""]

    lines += ["## Test & Compare", "", document["test_and_compare"], ""]
    return "\n".join(lines)


__all__ = [
    "publish",
    "DEFAULT_PUBLISH_MAX_TOKENS",
    "validate_title",
    "validate_chapters",
    "format_chapters",
    "format_timecode",
    "validate_thumbnail_text",
    "title_formulas",
    "extract_frame",
    "render_local_thumbnail",
    "preview_strip",
    "candidate_frames",
    "PublishPack",
    "PublishError",
]
