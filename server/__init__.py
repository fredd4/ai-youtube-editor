"""The local web editor (FastAPI + vanilla JS).

``server.app`` exposes the FastAPI application and :func:`server.app.serve`,
which the ``ytedit serve`` CLI command calls. ``server.jobs`` runs pipeline
stages as subprocesses on behalf of the browser.
"""

__all__ = ["app", "jobs"]
