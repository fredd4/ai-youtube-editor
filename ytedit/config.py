"""Settings: ``.env`` + ``config/defaults.yaml`` + optional ``project.yaml``.

``Settings`` is a plain read-only view over a deep-merged dict. Access nested
keys with :meth:`Settings.get` (``"encoding.master.crf"``) or the typed
convenience properties.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from dotenv import load_dotenv

from .log import get_logger

log = get_logger(__name__)

#: Repository root (the directory that contains ``config/`` and ``projects/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = PROJECT_ROOT / "config"
PROJECTS_DIR: Path = PROJECT_ROOT / "projects"
DEFAULTS_FILE: Path = CONFIG_DIR / "defaults.yaml"
CAPTION_STYLES_FILE: Path = CONFIG_DIR / "caption_styles.yaml"
MUSIC_STYLES_FILE: Path = CONFIG_DIR / "music_styles.yaml"

_MISSING = object()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base`` (dicts only, lists replace).

    Args:
        base: Lower-priority mapping.
        override: Higher-priority mapping.

    Returns:
        A new merged dict; inputs are not mutated.
    """
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping, returning ``{}`` for a missing or empty file."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class Settings:
    """Merged configuration for a run.

    Attributes:
        data: The merged ``defaults.yaml`` (+ ``project.yaml``) tree.
        project_root: Repository root directory.
        project_dir: Project directory when the settings were loaded for one.
    """

    data: dict[str, Any] = field(default_factory=dict)
    project_root: Path = PROJECT_ROOT
    project_dir: Path | None = None

    # -- generic access -------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Read a nested key by dotted path.

        Args:
            dotted: e.g. ``"encoding.master.crf"``.
            default: Returned when the path is absent; omitting it makes a
                missing key raise ``KeyError``.
        """
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level (or dotted) section as a dict, ``{}`` if absent."""
        value = self.get(name, {})
        return dict(value) if isinstance(value, Mapping) else {}

    # -- secrets --------------------------------------------------------
    @property
    def keys(self) -> dict[str, str | None]:
        """API keys read from the environment / ``.env``."""
        return {
            "fal": os.environ.get("FAL_KEY"),
            "openrouter": os.environ.get("OPENROUTER_API_KEY"),
            "elevenlabs": os.environ.get("ELEVENLABS_API_KEY"),
        }

    def require_key(self, name: str) -> str:
        """Return an API key or raise a helpful error.

        Args:
            name: One of ``fal``, ``openrouter``, ``elevenlabs``.
        """
        value = self.keys.get(name)
        if not value:
            env = {"fal": "FAL_KEY", "openrouter": "OPENROUTER_API_KEY",
                   "elevenlabs": "ELEVENLABS_API_KEY"}.get(name, name.upper())
            raise RuntimeError(f"missing API key: set {env} in {self.project_root / '.env'}")
        return value

    # -- typed conveniences --------------------------------------------
    @property
    def language(self) -> str:
        """Project content language (ISO-639-1), e.g. ``pl``."""
        return str(self.get("language", "pl"))

    @property
    def budget_usd(self) -> float:
        """Spending cap for the project in USD."""
        return float(self.get("budget_usd", 20.0))

    @property
    def canvas(self) -> tuple[int, int, int]:
        """``(width, height, fps)`` of the output canvas."""
        return (
            int(self.get("canvas.width", 1920)),
            int(self.get("canvas.height", 1080)),
            int(self.get("canvas.fps", 30)),
        )

    @property
    def models(self) -> dict[str, str]:
        """Model ids by role (``planner``, ``analyst``, ``vision``, ...)."""
        return {k: str(v) for k, v in self.section("models").items()}

    @property
    def prices(self) -> dict[str, Any]:
        """Price table used by :mod:`ytedit.costs`."""
        return self.section("prices")

    def model(self, role: str) -> str:
        """Return the model id configured for ``role``."""
        models = self.models
        if role not in models:
            raise KeyError(f"unknown model role {role!r}; known: {sorted(models)}")
        return models[role]

    def encoding(self, preset: str) -> dict[str, Any]:
        """Return an encoding preset (``mezzanine``/``segment``/``master``/``preview``)."""
        presets = self.section("encoding")
        if preset not in presets:
            raise KeyError(f"unknown encoding preset {preset!r}; known: {sorted(presets)}")
        return dict(presets[preset])

    def grade_preset(self, name: str) -> list[str]:
        """Return the ordered filter list for a grade preset."""
        presets = self.get("grade.presets", {})
        if name not in presets:
            raise KeyError(f"unknown grade preset {name!r}; known: {sorted(presets)}")
        return list(presets[name] or [])

    # -- side files ------------------------------------------------------
    @property
    def caption_styles(self) -> dict[str, Any]:
        """Parsed ``config/caption_styles.yaml``."""
        return load_yaml(CAPTION_STYLES_FILE)

    @property
    def music_styles(self) -> dict[str, Any]:
        """Parsed ``config/music_styles.yaml``."""
        return load_yaml(MUSIC_STYLES_FILE)

    def caption_font(self) -> tuple[str, Path]:
        """Return ``(font_name, font_file)`` for burned-in captions."""
        font = self.caption_styles.get("font", {})
        return str(font.get("name", "Arial")), Path(
            font.get("file", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        )


def load_settings(
    project_dir: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Build :class:`Settings` from ``.env`` + defaults + optional project.yaml.

    Args:
        project_dir: A ``projects/<slug>`` directory; its ``project.yaml`` is
            deep-merged on top of the defaults.
        overrides: Extra mapping merged last (used by tests).

    Returns:
        A frozen :class:`Settings`.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    data = load_yaml(DEFAULTS_FILE)
    pdir = Path(project_dir) if project_dir else None
    if pdir is not None:
        data = deep_merge(data, load_yaml(pdir / "project.yaml"))
    if overrides:
        data = deep_merge(data, overrides)
    return Settings(data=data, project_root=PROJECT_ROOT, project_dir=pdir)


@lru_cache(maxsize=1)
def global_settings() -> Settings:
    """Cached settings with no project overlay."""
    return load_settings()


def known_config_files() -> Iterable[Path]:
    """Yield the config files that participate in a load (for diagnostics)."""
    yield DEFAULTS_FILE
    yield CAPTION_STYLES_FILE
    yield MUSIC_STYLES_FILE
