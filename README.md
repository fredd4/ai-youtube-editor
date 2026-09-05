# AI YouTube Editor

Semi-automated pipeline plus a local web editor that turns raw phone footage
(travel vlogs, Polish narration by default) into a finished 16:9 YouTube master.
Claude acts as editor-in-chief on top of deterministic ffmpeg tooling: the AI
only produces and edits JSON, ffmpeg does all the media work.

Design principles (full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

1. **Everything is a file on disk.** One directory per project, JSON/YAML state,
   no database. Every stage is re-runnable and idempotent.
2. **Deterministic tooling, AI decisions.** ffmpeg renders from an explicit
   timeline (EDL); LLMs only produce/modify JSON.
3. **Language is per project** (`project.yaml: language: pl`).
4. **Code, comments, docs and UI strings are English**; narration, captions and
   titles are in the project language.
5. **Cost-aware.** Every API call logs an estimate to `state.json` and a budget
   cap refuses to run when exceeded.

## Requirements

- macOS or Linux, Python 3.12
- `ffmpeg` 7.x with `libx264`, `libass`, `libzimg` (and `videotoolbox` on macOS)
- [`uv`](https://docs.astral.sh/uv/)
- API keys in `.env`: `FAL_KEY`, `OPENROUTER_API_KEY`, `ELEVENLABS_API_KEY`

## Setup

```bash
make setup                 # uv venv .venv && uv pip install -e ".[dev]"
source .venv/bin/activate  # or call .venv/bin/ytedit directly
ytedit --help
```

## Quickstart

```bash
make new NAME=lisbon-day-1 LANG=pl      # create projects/lisbon-day-1/
cp ~/Movies/IMG_*.MOV projects/lisbon-day-1/input/
make ingest NAME=lisbon-day-1           # normalize, proxies, audio, peaks, thumbs
make status NAME=lisbon-day-1           # clip registry + stage status + spend

ytedit run        lisbon-day-1          # transcribe -> analyze -> plan (skips what is done)
ytedit denoise    lisbon-day-1 --clip c004 --preview   # optional: voice isolation for a windy clip
ytedit music      lisbon-day-1          # -> music/*.mp3 (ElevenLabs, ~$0.15/min)
ytedit render     lisbon-day-1 --preview
ytedit qc         lisbon-day-1
ytedit render     lisbon-day-1 --master
ytedit publish    lisbon-day-1          # titles, description, chapters, thumbnails

make serve                              # web editor on http://localhost:8765
```

Review and adjust the cut in the web editor's **Program** view (final-cut strip,
segment inspector, "Save & render preview"); `ytedit tidy <slug>` re-applies the
air-around-speech rule after manual edits. Every stage also has a `make` target
(`make help`).

## Repository layout

```
config/          defaults.yaml, caption_styles.yaml, music_styles.yaml
docs/            ARCHITECTURE.md, research/, playbook/
ytedit/          the package
  config.py      .env + defaults.yaml + project.yaml -> Settings
  project.py     project paths, state.json (atomic + locked), clip registry
  timeline.py    the EDL model (pydantic) + validation + speech ranges
  costs.py       price table, estimates, ledger, budget cap
  media/         ffmpeg runner, probe, colour/fit filter builders, ingest, frames
  ai/            OpenRouter / ElevenLabs / fal clients and the LLM stages
server/          FastAPI web editor
tests/           pytest; fixtures generated with ffmpeg lavfi
projects/<slug>/ input/ media/ transcripts/ analysis/ plan/ music/ voice/
                 renders/ exports/ jobs/ state.json project.yaml
```

## Project directory

`input/` is the raw drop zone — copy phone clips there in any orientation.
Ingest assigns clip ids `c001…` in recording order (from `creation_time`,
falling back to file mtime, then name) and writes:

| Path | Content |
|---|---|
| `media/sources/<id>.mp4` | normalized mezzanine: CFR, rotation baked, HDR→SDR, always has audio |
| `media/proxies/<id>.mp4` | 720p H.264 for the browser and vision passes |
| `media/audio/<id>.wav` | mono 48 kHz PCM for STT and analysis |
| `media/peaks/<id>.json` | pre-computed waveform peaks for wavesurfer |
| `media/thumbs/<id>.jpg` | poster frame |
| `media/thumbs/frames/<id>/NNN.jpg` | frames sampled every 3 s for the vision model |

`plan/timeline.json` is the single source of truth for rendering.

## Tests

```bash
make test    # generates synthetic fixtures with ffmpeg lavfi, then runs pytest
```

No API keys are needed for the test suite.
