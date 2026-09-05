"""Tests for ``ytedit.ai.publish`` — validators, thumbnails and the stage itself.

Everything here is offline: the writer model is a fake ``OpenRouter`` and the
paid fal pass is a fake ``Fal``, so no test spends money. Only ffmpeg and PIL do
real work (a two-second lavfi clip is generated per test that needs frames).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ytedit.ai.prompts import load as load_prompt
from ytedit.ai.publish import (
    SCHEMA_HINT,
    PublishError,
    PublishPack,
    candidate_frames,
    extract_frame,
    format_chapters,
    format_timecode,
    preview_strip,
    publish,
    render_local_thumbnail,
    title_formulas,
    validate_chapters,
    validate_thumbnail_text,
    validate_title,
)
from ytedit.project import Project
from ytedit.timeline import Chapter, Marker, Timeline, VideoSegment, new_timeline

# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
CLIPS: dict[str, dict[str, Any]] = {
    "c001": {"id": "c001", "order": 1, "duration": 60.0, "width": 1920, "height": 1080,
             "orientation": "horizontal"},
    "c002": {"id": "c002", "order": 2, "duration": 40.0, "width": 1920, "height": 1080,
             "orientation": "horizontal"},
}

FOOTAGE_LOG: dict[str, Any] = {
    "project": "t-pub",
    "clips": [
        {
            "clip": "c001",
            "summary": "Tramwaj 28 i widok na Alfamę.",
            "kind": "a-roll",
            "hooks": ["tramwaj był tak pełny, że poszliśmy pieszo"],
            "numbers": ["bilet za 3 EUR"],
            "topics": ["tramwaj 28", "Alfama"],
            "visual": {"quality": 0.9, "best_frames": [1.0], "thumbnail_candidate": True},
            "location": {"name": "Alfama", "city": "Lizbona", "country": "Portugalia"},
        },
        {
            "clip": "c002",
            "summary": "Pastéis de Belém.",
            "kind": "a-roll",
            "hooks": [],
            "numbers": ["1.20 EUR za pastel"],
            "topics": ["pastéis de nata"],
            "visual": {"quality": 0.6, "best_frames": [0.5], "thumbnail_candidate": False},
            "location": {"name": "Belém", "city": "Lizbona", "country": "Portugalia"},
        },
    ],
}

LLM_PUBLISH: dict[str, Any] = {
    "titles": [
        {"title": "Lizbona za 30 euro dziennie — czego nikt ci nie mówi o tramwaju",
         "formula": "number+curiosity", "chars": 0},
        {"title": "Najlepszy dzień w Lizbonie: 3 miejsca, o których nikt nie pisze",
         "formula": "superlative", "chars": 0},
        {"title": "Za krótki", "formula": "curiosity", "chars": 0},
    ],
    "description": (
        "Lizbona w jeden dzień za 30 euro — tramwaj 28, Alfama i pastéis de nata, "
        "czyli jak zobaczyć miasto bez wydawania fortuny.\n\n"
        "0:00 Przyjazd\n0:20 Alfama\n0:40 Belém\n\nLINKS: {links}"
    ),
    "chapters": [
        {"at": 0, "title": "Przyjazd do Lizbony"},
        {"at": 20, "title": "Tramwaj 28 i Alfama"},
        {"at": 40, "title": "Pastéis de Belém"},
    ],
    "tags": ["lizbona", "portugalia", "tramwaj 28"],
    "thumbnail_variants": [
        {"text": "3 EURO ZA DZIEŃ", "concept": "twarz w tramwaju",
         "colors": ["żółty", "granatowy"], "frame_clip": "c001", "frame_t": 1.0},
        {"text": "ŁÓDŹ CZY LIZBONA?", "concept": "widok z miradouro",
         "colors": ["biały"], "frame_clip": "c002", "frame_t": 0.5},
    ],
    "test_and_compare": "Wrzuć 3 tytuły i 3 miniatury do Test & Compare na 2 tygodnie.",
}


class FakeOpenRouter:
    """Stand-in for :class:`ytedit.ai.openrouter.OpenRouter` that never spends."""

    answer: Any = LLM_PUBLISH
    calls: list[dict[str, Any]] = []

    def __init__(self, api_key: str, cost_callback: Any = None, **_: Any) -> None:
        self.api_key = api_key
        self.cost_callback = cost_callback

    def ask_json(self, model: str, system_prompt: str, user: str, schema_hint: str, **kw: Any):
        FakeOpenRouter.calls.append({"model": model, "system": system_prompt, "user": user, **kw})
        if self.cost_callback:
            # The real client reports units as a human string, not a number.
            self.cost_callback(
                service="openrouter", op="chat", model=model, units="900+400 tok", usd=0.02
            )
        return FakeOpenRouter.answer

    def close(self) -> None:
        return None


class FakeFal:
    """Stand-in for :class:`ytedit.ai.fal.Fal`: copies the base frame, no network."""

    calls: list[dict[str, Any]] = []

    def __init__(self, api_key: str, cost_callback: Any = None, **_: Any) -> None:
        self.cost_callback = cost_callback

    def thumbnail_edit(
        self, prompt: str, image_paths: list[Path], out_dir: Path, **kw: Any
    ) -> list[Path]:
        FakeFal.calls.append({"prompt": prompt, "images": list(image_paths), **kw})
        dest = Path(out_dir) / f"fake_{len(FakeFal.calls)}.jpg"
        Image.open(image_paths[0]).save(dest, format="JPEG")
        if self.cost_callback:
            self.cost_callback(
                service="fal", op="thumbnail", model="fal-ai/nano-banana-pro/edit",
                units="1 images", usd=0.15,
            )
        return [dest]


def _make_clip(dest: Path, seconds: float = 2.0, size: str = "1920x1080") -> Path:
    """Render a tiny lavfi clip so ffmpeg frame extraction has something real."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
            "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-an", str(dest),
        ],
        check=True,
    )
    return dest


def _timeline() -> Timeline:
    timeline = new_timeline(language="pl")
    cursor = 0.0
    for i, (clip, length) in enumerate([("c001", 25.0), ("c002", 20.0), ("c001", 15.0)]):
        timeline.tracks.video.append(
            VideoSegment(id=f"s{i + 1:03d}", clip=clip, **{"in": cursor}, out=cursor + length,
                         role="a-roll")
        )
        cursor += length
    timeline.markers = [Marker(at=0.0, label="hook"), Marker(at=7.0, label="promise")]
    timeline.chapters = [
        Chapter(at=0.0, title="Przyjazd do Lizbony"),
        Chapter(at=25.0, title="Tramwaj 28"),
        Chapter(at=45.0, title="Belém"),
    ]
    return timeline


@pytest.fixture()
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    """A project with a plan, a timeline and two real source clips."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("FAL_KEY", raising=False)
    project = Project.create("t-pub", language="pl", title="Lizbona", root=tmp_path / "projects")
    with project.edit_state() as state:
        state["clips"] = json.loads(json.dumps(CLIPS))
        state["budget_usd"] = 20.0
    (project.analysis_dir / "footage_log.json").write_text(
        json.dumps(FOOTAGE_LOG, ensure_ascii=False), encoding="utf-8"
    )

    timeline = _timeline()
    timeline.save(project.timeline_file)
    project.edit_plan_file.write_text(
        json.dumps(
            {
                "project": project.slug,
                "plan": {
                    "story": {
                        "title_working": "Lizbona w jeden dzień",
                        "beats": [{"label": "hook", "at_s_target": 0.0, "clips": ["c001"],
                                   "description": "tramwaj"}],
                    },
                    "thumbnail_concepts": [
                        {"concept": "twarz w tramwaju", "frame_clip": "c001", "frame_t": 1.0,
                         "text": "3 EURO ZA DZIEŃ", "colors": ["żółty"]}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _make_clip(project.source_path("c001"))
    _make_clip(project.source_path("c002"))

    FakeOpenRouter.calls = []
    FakeOpenRouter.answer = LLM_PUBLISH
    FakeFal.calls = []
    monkeypatch.setattr("ytedit.ai.publish.OpenRouter", FakeOpenRouter)
    monkeypatch.setattr("ytedit.ai.publish.Fal", FakeFal)
    return project


# ----------------------------------------------------------------------
# title validator (research rule 27)
# ----------------------------------------------------------------------
def test_validate_title_accepts_a_good_polish_title() -> None:
    title = "Lizbona za 30 euro dziennie — czego nikt ci nie mówi o tramwaju"
    assert 55 <= len(title) <= 70
    assert validate_title(title, "pl") == []


def test_validate_title_flags_length() -> None:
    assert any("under" in i for i in validate_title("Za krótki tytuł 3", "pl"))
    long_title = "Lizbona za 30 euro dziennie i wszystko, czego nikt ci nie mówi o tym mieście"
    issues = validate_title(long_title, "pl")
    assert any("over the 70 target" in i for i in issues)
    assert any("pl runs long" in i for i in issues)
    assert any("hard maximum" in i for i in validate_title("Lizbona 3 " + "x" * 120, "pl"))


def test_validate_title_requires_a_formula() -> None:
    plain = "Spacer po mieście oraz kilka spokojnych ujęć z ulicy portowej"
    assert any("number, curiosity gap" in i for i in validate_title(plain, "pl"))


def test_validate_title_flags_repeated_keywords() -> None:
    repeated = "Lizbona: dlaczego Lizbona jest tańsza, niż myślisz o wyjazdach"
    assert any("repeated keyword" in i for i in validate_title(repeated, "pl"))


def test_title_formulas_detects_polish_variants() -> None:
    assert "number" in title_formulas("3 dni w Lizbonie")
    assert "curiosity" in title_formulas("Dlaczego nikt tam nie jeździ")
    assert "superlative" in title_formulas("Najtańszy dzień w Portugalii")
    assert "transformation" in title_formulas("Z Warszawy do Lizbony")


# ----------------------------------------------------------------------
# chapters (research rule 25)
# ----------------------------------------------------------------------
def test_validate_chapters_accepts_a_valid_list() -> None:
    chapters = [{"at": 0, "title": "Przyjazd"}, {"at": 30, "title": "Alfama"},
                {"at": 90, "title": "Belém"}]
    assert validate_chapters(chapters, duration=200.0) == []


def test_validate_chapters_flags_every_rule() -> None:
    issues = validate_chapters([{"at": 5, "title": "Start"}, {"at": 9, "title": "Część 2"}],
                               duration=600.0)
    assert any("must be 0:00" in i for i in issues)
    assert any("at least 3" in i for i in issues)
    assert any("minimum 10s" in i for i in issues)
    assert any("generic label" in i for i in issues)


def test_validate_chapters_flags_descending_and_past_the_end() -> None:
    issues = validate_chapters(
        [{"at": 0, "title": "A"}, {"at": 60, "title": "B"}, {"at": 30, "title": "C"}],
        duration=45.0,
    )
    assert any("not ascending" in i for i in issues)
    assert any("past the end" in i for i in issues)


def test_format_chapters_and_timecode() -> None:
    block = format_chapters([{"at": 0, "title": "Przyjazd"}, {"at": 95, "title": "Alfama"},
                             {"at": 3725, "title": "Powrót"}])
    assert block.splitlines() == ["0:00 Przyjazd", "1:35 Alfama", "1:02:05 Powrót"]
    assert format_timecode(0) == "0:00"
    assert format_timecode(61.4) == "1:01"


# ----------------------------------------------------------------------
# thumbnail text (research rule 28)
# ----------------------------------------------------------------------
def test_validate_thumbnail_text() -> None:
    assert validate_thumbnail_text("3 EURO ZA DZIEŃ") == []
    assert any("maximum 5" in i for i in validate_thumbnail_text("raz dwa trzy cztery pięć sześć"))
    assert any("may not render" in i for i in validate_thumbnail_text("LIZBONA 🔥"))
    assert validate_thumbnail_text("  ") == ["empty thumbnail text"]


# ----------------------------------------------------------------------
# thumbnails
# ----------------------------------------------------------------------
def test_extract_frame_and_local_thumbnail_with_polish_text(tmp_path: Path) -> None:
    source = _make_clip(tmp_path / "src.mp4")
    base = extract_frame(source, 1.0, tmp_path / "base.jpg")
    assert base.exists()
    with Image.open(base) as img:
        assert img.size == (1280, 720)

    local = render_local_thumbnail(base, "ŁÓDŹ ZA 3 EURO", tmp_path / "local.jpg")
    with Image.open(local) as img:
        assert img.size == (1280, 720)
    assert local.read_bytes() != base.read_bytes(), "the text overlay changed nothing"

    strip = preview_strip([base, local], tmp_path / "preview_120px.jpg")
    with Image.open(strip) as img:
        assert img.height == 120 and img.width > 120


def test_extract_frame_from_a_vertical_clip_is_still_16x9(tmp_path: Path) -> None:
    source = _make_clip(tmp_path / "vert.mp4", size="1080x1920")
    base = extract_frame(source, 0.5, tmp_path / "vbase.jpg")
    with Image.open(base) as img:
        assert img.size == (1280, 720)


def test_preview_strip_needs_at_least_one_image(tmp_path: Path) -> None:
    with pytest.raises(PublishError, match="no thumbnails"):
        preview_strip([tmp_path / "nope.jpg"], tmp_path / "strip.jpg")


# ----------------------------------------------------------------------
# LLM output normalization
# ----------------------------------------------------------------------
def test_publish_pack_accepts_the_promptsmd_shape() -> None:
    pack = PublishPack.from_llm(
        {
            "title_candidates": ["Tytuł jeden", {"title": "Tytuł dwa", "rationale": "ciekawość"}],
            "chapters": [{"at_seconds": 0, "title": "Start"}],
            "thumbnail_prompts": [
                {"on_image_text": "3 EURO", "description": "twarz", "colors": ["żółty"],
                 "source_frame_hint": "c003@12.4"}
            ],
            "tags": [f"tag{i}" for i in range(30)],
            "test_and_compare_note": "porównaj",
        }
    )
    assert [t.title for t in pack.titles] == ["Tytuł jeden", "Tytuł dwa"]
    assert pack.titles[0].chars == len("Tytuł jeden")
    assert pack.chapters[0].at == 0.0 and pack.chapters[0].title == "Start"
    assert pack.thumbnail_variants[0].frame_clip == "c003"
    assert pack.thumbnail_variants[0].frame_t == 12.4
    assert len(pack.tags) == 15
    assert pack.test_and_compare == "porównaj"


def test_publish_pack_rejects_non_objects() -> None:
    with pytest.raises(PublishError, match="expected a JSON object"):
        PublishPack.from_llm("nope")
    with pytest.raises(PublishError, match="cut off"):
        PublishPack.from_llm([1, 2, 3])


def test_publish_pack_salvages_a_bare_title_array() -> None:
    """A reply cut mid-JSON often parses back as just the title array."""
    pack = PublishPack.from_llm([{"title": "Lizbona za 30 euro"}, "Drugi tytuł"])
    assert [t.title for t in pack.titles] == ["Lizbona za 30 euro", "Drugi tytuł"]


def test_candidate_frames_puts_thumbnail_candidates_first() -> None:
    frames = candidate_frames(FOOTAGE_LOG)
    assert [f["clip"] for f in frames] == ["c001", "c002"]
    assert frames[0]["thumbnail_candidate"] is True


# ----------------------------------------------------------------------
# stage behaviour
# ----------------------------------------------------------------------
def test_publish_writes_the_pack_and_local_thumbnails(published: Project) -> None:
    result = publish(published, thumbnails=False)

    pack = json.loads((published.exports_dir / "publish.json").read_text(encoding="utf-8"))
    assert pack["language"] == "pl"
    assert pack["chapters_block"].startswith("0:00 ")
    assert pack["tags"] == ["lizbona", "portugalia", "tramwaj 28"]

    # Every title carries its validator verdict; the short one is flagged.
    assert pack["title_issues"]["Za krótki"], "the short title should be flagged"
    good = "Lizbona za 30 euro dziennie — czego nikt ci nie mówi o tramwaju"
    assert pack["title_issues"][good] == []

    md = (published.exports_dir / "publish.md").read_text(encoding="utf-8")
    assert "## Titles" in md and "## Chapters" in md and "## Test & Compare" in md
    assert good in md

    # Local thumbnails are always rendered, even with fal turned off.
    thumbs = published.exports_dir / "thumbnails"
    assert (thumbs / "base_1.jpg").exists() and (thumbs / "local_1.jpg").exists()
    assert not list(thumbs.glob("fal_*.jpg"))
    assert (thumbs / "preview_120px.jpg").exists()
    with Image.open(thumbs / "local_1.jpg") as img:
        assert img.size == (1280, 720)

    assert result["cost_usd"] == pytest.approx(0.02)
    assert published.stage_status("publish") == "done"
    costs = published.load_state()["costs"]
    assert costs[-1]["service"] == "openrouter" and costs[-1]["units"] == "in=900, out=400"


def test_publish_runs_fal_when_asked_and_keyed(
    published: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAL_KEY", "test-fal-key")
    result = publish(published, thumbnails=True, n_thumbs=2)

    thumbs = published.exports_dir / "thumbnails"
    assert (thumbs / "fal_1.jpg").exists() and (thumbs / "fal_2.jpg").exists()
    assert len(FakeFal.calls) == 2
    # The image prompt is rendered from prompts.md with the clip's real location.
    assert "Alfama" in FakeFal.calls[0]["prompt"]
    assert FakeFal.calls[0]["aspect_ratio"] == "16:9"
    assert result["cost_usd"] == pytest.approx(0.02 + 2 * 0.15)


def test_publish_falls_back_to_the_plan_and_timeline(published: Project) -> None:
    FakeOpenRouter.answer = {
        "titles": [{"title": "Lizbona za 30 euro dziennie — czego nikt ci nie mówi o tramwaju"}],
        "description": "opis",
    }
    publish(published, thumbnails=False)
    pack = json.loads((published.exports_dir / "publish.json").read_text(encoding="utf-8"))

    # Chapters come from the timeline, tags from the footage log, thumbnails
    # from the plan's concepts.
    assert [c["title"] for c in pack["chapters"]] == [
        "Przyjazd do Lizbony", "Tramwaj 28", "Belém"
    ]
    assert "Alfama" in pack["tags"]
    assert pack["thumbnail_variants"][0]["text"] == "3 EURO ZA DZIEŃ"
    assert pack["test_and_compare"].startswith("Upload 3 titles")


def test_publish_needs_a_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    project = Project.create("bare", language="pl", root=tmp_path / "projects")
    with pytest.raises(PublishError, match="no edit plan"):
        publish(project)


def test_publish_records_an_error_status(published: Project) -> None:
    FakeOpenRouter.answer = {"description": "brak tytułów"}
    with pytest.raises(PublishError, match="no titles"):
        publish(published)
    assert published.stage_status("publish") == "error"


# ----------------------------------------------------------------------
# prompt / schema economy
# ----------------------------------------------------------------------
def test_schema_hint_points_at_the_system_prompt_instead_of_repeating_it() -> None:
    assert len(SCHEMA_HINT) < 200
    assert '"titles"' not in SCHEMA_HINT


def test_publish_system_prompt_carries_exactly_one_schema_matching_the_parser() -> None:
    text = load_prompt("publish.system")
    # The old prompt asked for title_candidates/thumbnail_prompts while the code
    # parsed titles/thumbnail_variants; there is now one schema, and it is the
    # one PublishPack.from_llm is built around.
    assert '"title_candidates"' not in text
    assert text.count('"titles"') == 1
    assert text.count('"thumbnail_variants"') == 1
    assert '"tags"' in text and '"test_and_compare"' in text
