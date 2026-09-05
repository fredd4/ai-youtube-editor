"""Loader for the canonical prompt templates in ``docs/playbook/prompts.md``.

Prompts live in Markdown so they can be edited without touching code. Each
prompt is a fenced code block under a ``## stage.role`` heading, e.g.::

    ## analyze.system

    ```
    You are ...
    ```

Placeholders use ``{snake_case}`` and are filled with :func:`render`, which
uses ``str.format_map`` with a tolerant mapping so a literal ``{`` in JSON
examples does not explode.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from ytedit.config import PROJECT_ROOT

PROMPTS_FILE = PROJECT_ROOT / "docs" / "playbook" / "prompts.md"

_HEADING = re.compile(r"^##\s+([A-Za-z0-9_.-]+)", re.MULTILINE)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


class PromptNotFound(KeyError):
    """Raised when a ``stage.role`` block is missing from prompts.md."""


@lru_cache(maxsize=1)
def _load_all(path: Path = PROMPTS_FILE) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    headings = list(_HEADING.finditer(text))
    for i, m in enumerate(headings):
        name = m.group(1)
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        fence = _FENCE.search(text, start, end)
        if fence:
            blocks[name] = fence.group(1).strip("\n")
    return blocks


def names() -> list[str]:
    """All available prompt names (``stage.role``)."""
    return sorted(_load_all().keys())


def load(name: str) -> str:
    """Return the raw template text for ``name`` (e.g. ``"analyze.system"``)."""
    blocks = _load_all()
    if name not in blocks:
        raise PromptNotFound(f"{name!r} not found in {PROMPTS_FILE} (have: {', '.join(sorted(blocks))})")
    return blocks[name]


class _Tolerant(dict):
    """format_map helper: unknown placeholders are left untouched."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
        return "{" + key + "}"


def render(name: str, **values: object) -> str:
    """Load ``name`` and substitute ``{placeholders}``.

    Braces that are not placeholders (JSON examples inside the prompt) are
    preserved: we only replace ``{identifier}`` tokens that appear in
    ``values``; everything else is left as-is.
    """
    template = load(name)

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _sub, template)


def placeholders(name: str) -> list[str]:
    """List the ``{placeholders}`` a template expects."""
    return sorted(set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", load(name))))


def reload() -> None:
    """Drop the cache (useful after editing prompts.md in a long session)."""
    _load_all.cache_clear()


def describe(names_: Mapping[str, str] | None = None) -> str:
    """Human-readable summary of available prompts and their placeholders."""
    lines = []
    for n in names():
        lines.append(f"{n}: {', '.join(placeholders(n)) or '-'}")
    return "\n".join(lines)
