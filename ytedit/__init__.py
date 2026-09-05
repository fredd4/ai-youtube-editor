"""ytedit — semi-automated AI video editor for travel vlogs.

Everything is a file on disk: one directory per project, JSON/YAML state, no
database. Deterministic media work (ffmpeg) is driven from an explicit timeline
(EDL); LLMs only produce or modify JSON.
"""

__version__ = "0.1.0"
