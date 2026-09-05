"""Shared pytest fixtures: generated media and throwaway projects."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import FIXTURE_DIR, build_all  # noqa: E402

from ytedit.project import Project  # noqa: E402


@pytest.fixture(scope="session")
def media() -> dict[str, Path]:
    """Generate (once) and return the synthetic ffmpeg fixtures."""
    return build_all()


@pytest.fixture()
def project(tmp_path: Path) -> Project:
    """An empty project in a temporary projects root."""
    return Project.create("t-proj", language="pl", title="Temp", root=tmp_path / "projects")
