"""ASS (burned-in) and SRT (upload) caption writers.

Two very different products come out of this module:

* :func:`build_ass` renders the timeline's ``captions`` track — location
  cards, hook statements, selective subtitles — into an ASS document that
  libass burns into the master. Styles come from ``config/caption_styles.yaml``
  and are scaled to the render canvas; every cue is kept inside the playbook
  safe area (5 % left/right/top, 12 % bottom).
* :func:`srt_from_transcripts` turns the word-level transcripts into a full
  Polish SRT for upload (toggleable, enables auto-translate). It is written to
  ``exports/captions.srt`` and is **never** burned in.

The font is hard-checked for the Polish glyph set before anything is rendered:
a missing ``ł`` only shows up as a tofu box three hours into a master render.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence, TYPE_CHECKING

from ..log import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ..project import Project
    from ..timeline import Caption, Timeline

log = get_logger(__name__)

#: The full Polish diacritic set, upper and lower case.
POLISH_GLYPHS: str = "ĄĆĘŁŃÓŚŹŻąćęłńóśźż"

#: Reference canvas the sizes in ``caption_styles.yaml`` are authored for.
REFERENCE_HEIGHT: int = 1080

#: ``position`` field -> ASS alignment number (numpad layout).
ALIGNMENT: dict[str, int] = {
    "lower-left": 1,
    "lower-center": 2,
    "lower-right": 3,
    "center": 5,
    "middle-left": 4,
    "middle-right": 6,
    "upper-left": 7,
    "upper-center": 8,
    "upper-right": 9,
}

#: Default fade when a style does not specify one, in milliseconds.
DEFAULT_FADE_MS: int = 300

_ASS_STYLE_FORMAT = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
    "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_ASS_EVENT_FORMAT = "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


class CaptionFontError(RuntimeError):
    """The caption font cannot render every glyph the captions need."""

    def __init__(self, font: str, missing: Sequence[str]) -> None:
        self.font = str(font)
        self.missing = list(missing)
        joined = " ".join(f"{c!r} (U+{ord(c):04X})" for c in missing)
        super().__init__(f"font {self.font} is missing {len(missing)} glyph(s): {joined}")


class Cue(NamedTuple):
    """One subtitle cue in absolute programme time."""

    start: float
    end: float
    text: str


# ----------------------------------------------------------------------
# time formatting
# ----------------------------------------------------------------------
def ass_time(t: float) -> str:
    """Format seconds as an ASS timestamp (``H:MM:SS.cc``)."""
    total = max(0.0, float(t))
    centis = int(round(total * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, centis = divmod(rest, 100)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centis:02d}"


def srt_time(t: float) -> str:
    """Format seconds as an SRT timestamp (``HH:MM:SS,mmm``)."""
    total = max(0.0, float(t))
    millis = int(round(total * 1000))
    hours, rest = divmod(millis, 3600000)
    minutes, rest = divmod(rest, 60000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# ----------------------------------------------------------------------
# font glyph coverage
# ----------------------------------------------------------------------
def missing_glyphs(font_file: Path | str, text: str = POLISH_GLYPHS) -> list[str]:
    """Return the characters of ``text`` the font cannot render.

    Uses ``fontTools`` to read the font's cmap when it is installed; otherwise
    falls back to a Pillow heuristic that compares each rendered glyph against
    the font's ``.notdef`` box (rendered from an unassigned private-use
    codepoint). Whitespace and control characters are ignored.

    Args:
        font_file: Path to a TTF/OTF/TTC file.
        text: Characters to check.

    Returns:
        The missing characters, in first-seen order and de-duplicated.
    """
    wanted: list[str] = []
    for char in text:
        if char in wanted or char.isspace() or unicodedata.category(char).startswith("C"):
            continue
        wanted.append(char)
    if not wanted:
        return []

    path = Path(font_file)
    if not path.exists():
        log.warning("caption font %s does not exist", path)
        return list(wanted)

    try:  # preferred: read the cmap directly
        from fontTools.ttLib import TTCollection, TTFont  # type: ignore[import-not-found]

        if path.suffix.lower() == ".ttc":
            fonts = list(TTCollection(str(path)).fonts)
        else:
            fonts = [TTFont(str(path), fontNumber=0, lazy=True)]
        covered: set[int] = set()
        for font in fonts:
            for table in font["cmap"].tables:
                covered |= set(table.cmap)
        return [c for c in wanted if ord(c) not in covered]
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - broken font file
        log.warning("cannot read the cmap of %s (%s); falling back to Pillow", path, exc)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        log.warning("neither fontTools nor Pillow is available; skipping the glyph check")
        return []

    try:
        face = ImageFont.truetype(str(path), 48)
    except OSError as exc:  # pragma: no cover - unreadable font
        log.warning("cannot open font %s: %s", path, exc)
        return list(wanted)

    def render(char: str) -> bytes:
        image = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(image).text((8, 8), char, font=face, fill=255)
        return image.tobytes()

    # U+E000 is private use and assigned by essentially no text font, so it
    # renders as whatever the face uses for .notdef.
    notdef = render("")
    return [c for c in wanted if render(c) == notdef]


def check_font(font_file: Path | str, text: str = POLISH_GLYPHS, name: str = "") -> None:
    """Raise :class:`CaptionFontError` when the font lacks any needed glyph.

    Args:
        font_file: Font to check.
        text: Characters that must render.
        name: Human-readable font name for the error message.

    Raises:
        CaptionFontError: Listing the missing characters.
    """
    missing = missing_glyphs(font_file, text)
    if missing:
        raise CaptionFontError(name or str(font_file), missing)


# ----------------------------------------------------------------------
# ASS
# ----------------------------------------------------------------------
def escape_ass_text(text: str) -> str:
    """Escape caption text for an ASS ``Dialogue`` line."""
    out = str(text).replace("\\", "∖")
    out = out.replace("{", "(").replace("}", ")")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out.replace("\n", "\\N").strip()


def _style_line(
    name: str,
    style: Mapping[str, Any],
    scale: float,
    font_name: str,
    margins: Mapping[str, int],
) -> str:
    """Render one ``Style:`` line, scaled to the canvas."""
    def px(key: str, default: float) -> int:
        return max(0, int(round(float(style.get(key, default)) * scale)))

    alignment = int(style.get("alignment", 2))
    margin_l = max(px("margin_l", 96), margins["left"])
    margin_r = max(px("margin_r", 96), margins["right"])
    margin_v = px("margin_v", 108)
    margin_v = max(margin_v, margins["top"] if alignment >= 7 else margins["bottom"])
    if 4 <= alignment <= 6:
        margin_v = px("margin_v", 0)

    fields = [
        name,
        str(style.get("fontname", font_name)),
        str(px("fontsize", 52)),
        str(style.get("primary_colour", "&H00FFFFFF")),
        str(style.get("secondary_colour", style.get("primary_colour", "&H00FFFFFF"))),
        str(style.get("outline_colour", "&H00000000")),
        str(style.get("back_colour", "&H80000000")),
        str(int(style.get("bold", 1))),
        str(int(style.get("italic", 0))),
        str(int(style.get("underline", 0))),
        str(int(style.get("strikeout", 0))),
        "100",
        "100",
        f"{float(style.get('spacing', 0.0)):.1f}",
        "0",
        str(int(style.get("border_style", 1))),
        f"{float(style.get('outline', 3)) * scale:.1f}",
        f"{float(style.get('shadow', 1)) * scale:.1f}",
        str(alignment),
        str(margin_l),
        str(margin_r),
        str(margin_v),
        "1",
    ]
    return "Style: " + ",".join(fields)


def build_ass(
    captions: Sequence["Caption"],
    width: int,
    height: int,
    styles: Mapping[str, Any],
    font: tuple[str, Path],
    duration: float = 0.0,
) -> str:
    """Render the caption track as a complete ASS document.

    Args:
        captions: Caption cues in absolute *render* time.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        styles: The parsed ``config/caption_styles.yaml`` document.
        font: ``(name, file)`` as returned by ``Settings.caption_font()``.
        duration: Programme length; cues are clamped to it when non-zero.

    Returns:
        The ASS document as a string.

    Raises:
        CaptionFontError: When the font cannot render the Polish glyph set or
            any character actually used by the captions.
    """
    font_name, font_file = font[0], Path(font[1])
    needed = POLISH_GLYPHS + "".join(c.text for c in captions)
    check_font(font_file, needed, name=font_name)

    reference = int(
        (styles.get("reference") or {}).get("height", REFERENCE_HEIGHT) or REFERENCE_HEIGHT
    )
    scale = float(height) / float(reference or REFERENCE_HEIGHT)

    safe = styles.get("safe_area") or {}
    margins = {
        "left": int(round(float(safe.get("left", 0.05)) * width)),
        "right": int(round(float(safe.get("right", 0.05)) * width)),
        "top": int(round(float(safe.get("top", 0.05)) * height)),
        "bottom": int(round(float(safe.get("bottom", 0.12)) * height)),
    }

    style_table: dict[str, Any] = dict(styles.get("styles") or {})
    if not style_table:
        style_table = {"subtitle": {}}

    lines: list[str] = [
        "[Script Info]",
        "; Generated by ytedit",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        f"Format: {_ASS_STYLE_FORMAT}",
    ]
    for name, style in style_table.items():
        lines.append(_style_line(name, style or {}, scale, font_name, margins))
    lines += ["", "[Events]", f"Format: {_ASS_EVENT_FORMAT}"]

    # Two burned-in cards must never overlap on screen: a location card and a
    # hook line attached to neighbouring beats can land a few frames apart
    # once the cut is resolved, and the timeline's own overlap check
    # (Timeline.validate) already treats "subtitle"-style cues as exempt (they
    # are dense, transcript-driven, and never actually burned in — see the
    # module docstring), so the same exemption applies here: only cards
    # (location/hook/...) are pushed apart, never subtitle cues.
    ordered = sorted(captions, key=lambda c: (float(c.at), str(c.id)))
    resolved: list[tuple["Caption", float, float]] = []
    last_card_end = 0.0
    for cue in ordered:
        start = max(0.0, float(cue.at))
        end = float(cue.end)
        if duration:
            end = min(end, float(duration))
        if cue.style != "subtitle" and start < last_card_end - 1e-6:
            shift = last_card_end - start
            start += shift
            end += shift
        if cue.style != "subtitle":
            last_card_end = max(last_card_end, end)
        if end <= start:
            log.warning("caption %s has no length after clamping; skipped", cue.id)
            continue
        resolved.append((cue, start, end))

    for cue, start, end in resolved:
        style_name = cue.style if cue.style in style_table else next(iter(style_table))
        style = style_table.get(style_name) or {}

        text = cue.text.upper() if style.get("uppercase") else cue.text
        fade_in = int(style.get("fade_in_ms", DEFAULT_FADE_MS))
        fade_out = int(style.get("fade_out_ms", DEFAULT_FADE_MS))
        override = f"\\fad({fade_in},{fade_out})" if (fade_in or fade_out) else ""

        alignment = ALIGNMENT.get(str(cue.position), int(style.get("alignment", 2)))
        if alignment != int(style.get("alignment", alignment)):
            override += f"\\an{alignment}"

        margin_l = margins["left"] if alignment in (1, 4, 7) else 0
        margin_r = margins["right"] if alignment in (3, 6, 9) else 0
        margin_v = 0
        if alignment >= 7:
            margin_v = margins["top"]
        elif alignment <= 3:
            margin_v = margins["bottom"]

        body = ("{" + override + "}" if override else "") + escape_ass_text(text)
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style_name},,"
            f"{margin_l},{margin_r},{margin_v},,{body}"
        )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# SRT
# ----------------------------------------------------------------------
def write_srt(cues: Iterable[Cue | Mapping[str, Any] | Sequence[Any]], path: Path | str) -> Path:
    """Write cues to an SRT file.

    Args:
        cues: ``Cue`` tuples, ``{start, end, text}`` mappings or 3-sequences.
        path: Destination ``.srt`` file.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    index = 0
    for cue in cues:
        if isinstance(cue, Mapping):
            start, end, text = cue["start"], cue["end"], cue["text"]
        else:
            start, end, text = cue[0], cue[1], cue[2]
        text = str(text).strip()
        if not text or float(end) <= float(start):
            continue
        index += 1
        blocks.append(f"{index}\n{srt_time(float(start))} --> {srt_time(float(end))}\n{text}\n")
    target.write_text("\n".join(blocks), encoding="utf-8")
    return target


def _wrap(text: str, max_chars: int, max_lines: int) -> str:
    """Soft-wrap a cue to at most ``max_lines`` lines of ``max_chars``."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars and len(lines) + 1 < max_lines:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def chunk_words(
    words: Sequence[tuple[float, float, str]],
    min_words: int = 3,
    max_words: int = 5,
    min_duration: float = 1.0,
    max_duration: float = 4.0,
    max_gap: float = 0.8,
    max_chars: int = 34,
    max_lines: int = 2,
) -> list[Cue]:
    """Group timed words into subtitle cues.

    A cue is closed when it reaches ``max_words``, would exceed
    ``max_duration``, or the next word is more than ``max_gap`` away. Short
    cues are stretched to ``min_duration`` without ever overlapping the next
    one.

    Args:
        words: ``(start, end, text)`` triples in ascending order.
        min_words: Preferred minimum words per cue.
        max_words: Hard maximum words per cue.
        min_duration: Minimum on-screen time in seconds.
        max_duration: Maximum on-screen time in seconds.
        max_gap: A pause longer than this always breaks the cue.
        max_chars: Soft line length before wrapping.
        max_lines: Maximum lines per cue.

    Returns:
        The cues, in order.
    """
    cues: list[Cue] = []
    bucket: list[tuple[float, float, str]] = []

    def flush() -> None:
        if not bucket:
            return
        start = bucket[0][0]
        end = bucket[-1][1]
        text = _wrap(" ".join(w[2] for w in bucket), max_chars, max_lines)
        cues.append(Cue(round(start, 3), round(end, 3), text))
        bucket.clear()

    for word in words:
        if bucket:
            gap = word[0] - bucket[-1][1]
            span = word[1] - bucket[0][0]
            if gap > max_gap or span > max_duration or len(bucket) >= max_words:
                flush()
            elif len(bucket) >= min_words and (word[2].endswith((".", "!", "?"))):
                bucket.append(word)
                flush()
                continue
        bucket.append(word)
    flush()

    out: list[Cue] = []
    for i, cue in enumerate(cues):
        end = cue.end
        if end - cue.start < min_duration:
            limit = cues[i + 1].start if i + 1 < len(cues) else end + min_duration
            end = min(cue.start + min_duration, max(end, limit))
        out.append(Cue(cue.start, round(min(end, cue.start + max_duration), 3), cue.text))
    return out


def srt_from_transcripts(
    project: "Project", timeline: "Timeline", settings: Any = None
) -> list[Cue]:
    """Map transcript words into timeline time and chunk them into SRT cues.

    Words are read from ``transcripts/<clip>.json`` in *clip* time, clipped to
    each segment's ``[in, out)`` window and translated to the segment's
    absolute position (divided by ``speed``). Segments with ``mute_source`` are
    skipped — nothing is audible there.

    Args:
        project: Project holding the transcripts.
        timeline: Timeline to map onto.
        settings: Settings used for the ``subtitle`` caption style limits;
            ``project.settings`` when omitted.

    Returns:
        Cues in ascending order, ready for :func:`write_srt`.
    """
    cfg = settings or project.settings
    style = (cfg.caption_styles.get("styles") or {}).get("subtitle") or {}

    cache: dict[str, list[tuple[float, float, str]]] = {}
    collected: list[tuple[float, float, str]] = []

    for pos in timeline.segment_positions():
        seg = pos.segment
        if seg.mute_source:
            continue
        if seg.clip not in cache:
            path = project.transcript_path(seg.clip)
            words: list[tuple[float, float, str]] = []
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:  # pragma: no cover - bad transcript
                    log.warning("unreadable transcript %s", path)
                    data = {}
                for word in data.get("words", []):
                    if word.get("type") not in (None, "word"):
                        continue
                    text = str(word.get("t", word.get("text", ""))).strip()
                    start = word.get("s", word.get("start"))
                    end = word.get("e", word.get("end"))
                    if not text or start is None or end is None:
                        continue
                    try:
                        words.append((float(start), float(end), text))
                    except (TypeError, ValueError):  # pragma: no cover
                        continue
            cache[seg.clip] = words

        speed = seg.speed if seg.speed > 0 else 1.0
        for w_start, w_end, text in cache[seg.clip]:
            if w_end <= seg.in_ or w_start >= seg.out:
                continue
            s = max(w_start, seg.in_)
            e = min(w_end, seg.out)
            collected.append(
                (
                    pos.start + (s - seg.in_) / speed,
                    pos.start + (e - seg.in_) / speed,
                    text,
                )
            )

    collected.sort()
    return chunk_words(
        collected,
        min_words=int(style.get("words_per_cue_min", 3)),
        max_words=int(style.get("words_per_cue_max", 5)),
        max_chars=int(style.get("max_chars_per_line", 34)),
        max_lines=int(style.get("max_lines", 2)),
    )
