# AI YouTube Editor — Architecture

Semi-automated pipeline + local web editor that turns raw phone footage (travel vlogs, Polish narration by default) into a finished 16:9 YouTube master, with an AI agent (Claude) acting as the editor-in-chief on top of deterministic tooling.

Design principles:
1. **Everything is a file on disk.** One directory per project, JSON/YAML state, no database. Every stage writes artifacts that the next stage reads; every stage is re-runnable and idempotent.
2. **Deterministic tooling, AI decisions.** ffmpeg/ffprobe do all media work from an explicit **timeline JSON** (EDL). LLMs only produce/modify JSON (analysis, plan, captions, titles). The human and the agent review in the web editor.
3. **Language is per project.** `project.yaml: language: pl`. Every AI prompt receives the project language; transcripts record detected language and flag mismatches.
4. **Code, comments, docs, UI strings: English.** Content (narration, captions, titles) in the project language.
5. **Cost-aware.** Every API call logs estimated/actual cost to `state.json`; a budget cap per project refuses to run when exceeded.

---

## Repository layout

```
ai-youtube-editor/
├── .env                      # FAL_KEY, OPENROUTER_API_KEY, ELEVENLABS_API_KEY
├── pyproject.toml            # package "ytedit", managed with uv
├── Makefile                  # make serve / make new NAME=x / make run PROJECT=x STAGE=y
├── CLAUDE.md                 # agent entry point: how Claude operates a project
├── README.md
├── config/
│   ├── defaults.yaml         # global defaults (models, loudness, grade, encoding, budgets, prices)
│   ├── music_styles.yaml     # named music style presets (prompt formula)
│   └── caption_styles.yaml   # ASS styles (font, size, safe area)
├── docs/
│   ├── ARCHITECTURE.md       # this file
│   ├── research/             # research reports (playbook, technical stack, old-project analysis)
│   └── playbook/             # agent editing playbook (rules + checklists, derived from research)
├── ytedit/                   # Python package
│   ├── config.py             # load .env + defaults.yaml + project.yaml → Settings
│   ├── project.py            # Project: paths, state.json (atomic writes, file lock), clip registry
│   ├── timeline.py           # Timeline / EDL data model (pydantic) + validation
│   ├── costs.py              # cost ledger
│   ├── log.py                # logging setup (rich)
│   ├── media/
│   │   ├── ffmpeg.py         # ff()/ffprobe() runners, progress parsing (-progress pipe:1)
│   │   ├── probe.py          # MediaInfo: duration, w/h, fps, vfr, rotation, hdr (color_transfer), audio channels
│   │   ├── ingest.py         # register sources, normalize (CFR, rotation, HDR→SDR), proxies, audio wav, peaks, thumbs
│   │   ├── color.py          # filter builders: tonemap chain, grade chain, vertical-in-16:9
│   │   ├── audio.py          # loudnorm 2-pass, silencedetect, duck automation (sendcmd), voice cleanup, demucs wrapper
│   │   ├── frames.py         # frame sampling + annotation for vision models
│   │   ├── captions.py       # ASS/SRT writers (location cards, hook text, subtitles)
│   │   └── render.py         # Timeline → ffmpeg: segments, xfade/concat, music+ducking, captions, master export
│   ├── ai/
│   │   ├── openrouter.py     # chat client (text + vision), JSON extraction, retries, cost logging
│   │   ├── elevenlabs.py     # STT (scribe_v2), music, TTS, voice clone, audio isolation, SFX
│   │   ├── fal.py            # upload, submit/status/result, seedance i2v, nano-banana thumbnails, topaz upscale
│   │   ├── transcribe.py     # clip → transcripts/<clip>.json (+ .srt); fallback mlx-whisper
│   │   ├── analyze.py        # transcript+frames → analysis/<clip>.json (takes, instructions, topics, locations, quality)
│   │   ├── plan.py           # all analyses → plan/edit_plan.json (timeline draft, captions, music cues, narration asks, titles)
│   │   ├── music.py          # style selection + generation → music/*.mp3 with sidecar json
│   │   └── publish.py        # titles, description, chapters, thumbnail prompts/generation
│   ├── qc.py                 # playbook rule checker over a Timeline (+ rendered file loudness)
│   └── cli.py                # typer CLI
├── server/
│   ├── app.py                # FastAPI: project API, media (Range), jobs, timeline edits, render triggers
│   ├── jobs.py               # background job runner (subprocess/thread) with progress files
│   └── web/                  # vanilla JS editor: index.html, app.js, style.css (wavesurfer from cdnjs)
├── tests/                    # pytest; fixtures generated with ffmpeg lavfi (+ optional TTS narration)
└── projects/
    └── <slug>/               # one project = one video
        ├── project.yaml      # language, title, style, music mood, budget, output preset
        ├── input/            # RAW DROP ZONE — user copies phone clips here (mp4/mov, any orientation)
        ├── media/
        │   ├── sources/      # normalized mezzanine (CFR, rotation baked, SDR, ProRes or high-CRF H.264)
        │   ├── proxies/      # 720p H.264 for browser + analysis
        │   ├── audio/        # <clip>.wav 48k mono for STT/analysis; cleaned/isolated variants
        │   ├── peaks/        # <clip>.peaks.json for wavesurfer
        │   ├── thumbs/       # <clip>.jpg poster + frames/<clip>/NNN.jpg samples
        ├── transcripts/      # <clip>.json (words[], events[], language), <clip>.srt
        ├── analysis/         # <clip>.json per clip + footage_log.json (merged, chronological)
        ├── plan/             # edit_plan.json (LLM draft), timeline.json (edited, source of truth for render)
        ├── music/            # generated tracks + sidecar json
        ├── voice/            # narration pickups (recorded or TTS)
        ├── renders/          # preview.mp4 (fast), segments cache
        ├── exports/          # master_1080p.mp4 / master_2160p.mp4, captions.srt, thumbnails/, publish.json
        ├── jobs/             # job progress files
        └── state.json        # clip registry + stage status + cost ledger
```

---

## Pipeline stages

| # | Stage | Command | Input → Output | Tool |
|---|---|---|---|---|
| 0 | new | `ytedit new <slug> --language pl` | creates project dirs + project.yaml | — |
| 1 | ingest | `ytedit ingest <slug>` | input/* → media/sources, proxies, audio, peaks, thumbs; state.json clips[] | ffprobe/ffmpeg |
| 2 | transcribe | `ytedit transcribe <slug>` | audio/*.wav → transcripts/*.json + .srt | ElevenLabs scribe_v2 (fallback whisper) |
| 3 | analyze | `ytedit analyze <slug>` | transcripts + frames → analysis/*.json + footage_log.json | OpenRouter (Claude/Gemini) |
| 4 | plan | `ytedit plan <slug>` | footage_log → plan/edit_plan.json + plan/timeline.json (draft) + narration_requests.md | OpenRouter (Opus) |
| 4b | tidy | `ytedit tidy <slug>` | timeline.json → padded cuts (~0.3 s before / 0.45 s after speech); auto at the end of plan | deterministic (`ytedit/ai/tidy.py`) |
| 1b | denoise | `ytedit denoise <slug> --clip cNNN` | media/audio/<clip>.wav → <clip>.denoised.wav; render uses it when `clips[id].use_denoised` | ElevenLabs Audio Isolation or local afftdn |
| 5 | music | `ytedit music <slug>` | plan music cues → music/*.mp3 | ElevenLabs Music |
| 6 | render | `ytedit render <slug> --preview` / `--master` | timeline.json → renders/preview.mp4 / exports/master.mp4 | ffmpeg |
| 7 | qc | `ytedit qc <slug>` | timeline + master → qc_report.json/md | rules + loudnorm measure |
| 8 | publish | `ytedit publish <slug>` | → exports/publish.json (titles, description, chapters), thumbnails/ | OpenRouter + fal |
| — | serve | `ytedit serve` | web editor at http://localhost:8765 | FastAPI |
| — | run | `ytedit run <slug>` | stages 1–5 in order, skipping done | — |

Each stage records `state.json.stages[<name>] = {status, started, finished, cost_usd, error}`; per-clip stage status lives in `state.json.clips[<id>]`.

---

## Data models (pydantic, `ytedit/timeline.py` and `ytedit/project.py`)

### state.json
```json
{
  "project": "my-video",
  "clips": {
    "c001": {
      "id": "c001", "source_file": "input/a clip.MOV", "recorded_at": "2026-08-12T10:31:05",
      "order": 1, "duration": 42.3, "width": 3840, "height": 2160, "fps": 29.97, "vfr": true,
      "rotation": 90, "orientation": "vertical", "hdr": "hlg", "has_audio": true,
      "normalized": "media/sources/c001.mp4", "proxy": "media/proxies/c001.mp4",
      "audio": "media/audio/c001.wav", "peaks": "media/peaks/c001.json", "poster": "media/thumbs/c001.jpg",
      "stages": {"ingest": "done", "transcribe": "done", "analyze": "pending"}
    }
  },
  "stages": {"ingest": {"status": "done", "finished": "...", "cost_usd": 0}},
  "costs": [{"ts": "...", "service": "elevenlabs", "op": "stt", "units": "42.3s", "usd": 0.0026}],
  "budget_usd": 20.0
}
```
Clip ids are `c` + zero-padded index in **recording order** (from `creation_time` metadata, fallback file mtime, fallback name). Order is the travel chronology and is the default narrative order.

### transcripts/<clip>.json
```json
{"clip": "c001", "language": "pl", "language_probability": 0.98, "project_language": "pl", "language_mismatch": false,
 "text": "...", "words": [{"t": "Dzień", "s": 0.32, "e": 0.72, "p": -0.04}],
 "events": [{"type": "music", "s": 10.0, "e": 25.0}], "speakers": ["speaker_0"], "engine": "elevenlabs/scribe_v2"}
```

### analysis/<clip>.json (LLM output, schema-validated)
```json
{"clip": "c001", "summary": "Arrival at Alfama viewpoint, narrator explains the tram 28 route",
 "location": {"name": "Miradouro de Santa Luzia", "city": "Lisbon", "country": "Portugal", "confidence": 0.8},
 "kind": "a-roll|b-roll|silent-broll|audio-only|instruction",
 "instructions": [{"s": 0.0, "e": 4.1, "text": "put this at the very end as the summary", "action": "move_to_end"}],
 "takes": [{"topic": "tram 28 explanation", "attempts": [{"s": 5.0, "e": 20.0}, {"s": 21.0, "e": 34.0}], "keep": 1, "reason": "last take, cleaner"}],
 "segments": [{"s": 21.0, "e": 34.0, "role": "narration", "keep": true, "text": "...", "quality": 0.9}],
 "background_music": [{"s": 10.0, "e": 25.0, "confidence": 0.7, "suggest": "mute|duck|keep"}],
 "visual": {"quality": 0.8, "issues": ["slightly underexposed"], "best_frames": [12.4, 30.1], "thumbnail_candidate": true},
 "hooks": ["the tram was so full we walked", ...], "numbers": ["3 EUR ticket"], "topics": ["tram 28", "Alfama"]}
```

### plan/timeline.json — the EDL (single source of truth for render)
```json
{
  "version": 1, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
  "tracks": {
    "video": [
      {"id": "s001", "clip": "c003", "in": 12.0, "out": 16.5, "role": "cold-open",
       "transform": {"fit": "cover|contain|blur-fill|crop-pan", "zoom": 1.0}, "grade": "default",
       "transition_in": {"type": "cut|fade|xfade", "duration": 0.0}, "speed": 1.0, "mute_source": false,
       "source_audio_gain_db": 0, "notes": "drone reveal"}
    ],
    "voice": [
      {"id": "v001", "file": "voice/intro_pickup.wav", "at": 0.5, "gain_db": 0}
    ],
    "music": [
      {"id": "m001", "file": "music/arrival_warm.mp3", "at": 0.0, "end": 185.0, "gain_db": -18, "fade_in": 2.0, "fade_out": 3.0,
       "duck": {"mode": "auto", "amount_db": -12, "attack": 0.15, "release": 0.6}}
    ],
    "captions": [
      {"id": "t001", "at": 0.5, "end": 3.0, "text": "LIZBONA, PORTUGALIA", "style": "location", "position": "lower-left"}
    ],
    "sfx": []
  },
  "mute_ranges": [{"clip": "c005", "s": 3.0, "e": 20.0, "gain_db": -60, "reason": "copyrighted bar music"}],
  "markers": [{"at": 0.0, "label": "hook"}, {"at": 7.0, "label": "promise"}, {"at": 180.0, "label": "re-engagement-1"}],
  "chapters": [{"at": 0, "title": "Przyjazd do Lizbony"}],
  "meta": {"title_candidates": [], "generated_by": "plan@2026-09-04", "edited_by_human": false}
}
```
Rules: times in seconds (float). Timeline time for a video segment = cumulative position; `render.py` computes absolute placement. Captions/music/voice use absolute timeline time. `mute_ranges` are in **clip** time and apply to source audio wherever that clip range is used.

`plan.py`'s planner-facing segment schema (before deterministic post-processing) also accepts an optional `voice_over: {"picture": [{"clip", "in", "out"}]}` on a segment: `build_timeline` extracts that segment's own clip audio as a `tracks.voice` item (a WAV cut from `media/audio/<clip>.wav`, or the denoised variant when active, written to `voice/vo_<clip>_<in>_<out>.wav`) and replaces the segment with the listed picture cuts (`mute_source: true`, `role: "b-roll"`) so the narrator is heard but not seen, except where those cuts fall short of the narration's length — then the clip's own picture fills the gap.

### plan/edit_plan.json (LLM reasoning, human-readable)
Story outline (beats with target timecodes per playbook: 0:00 hook, 0:07 promise, 0:30 interrupt, 3:00/6:00 re-engagements, end payoff), which clips serve which beat, cold-open montage picks, subscribe-CTA placement, music cue sheet with mood per section, list of **narration requests** (what the user should record for intro/outro/bridges, with suggested scripts in the project language), risk flags (copyright music, missing footage), title candidates, thumbnail concepts. Written also as `plan/edit_plan.md`.

---

## Rendering model (`media/render.py`)

1. **Segment pass** (cached by hash of segment spec, including its frame count): for each video segment cut from the normalized source: `-ss in`, then exactly `round((out−in)/speed × fps)` frames (`setpts=PTS-STARTPTS` + `-frames:v`, audio `atrim`/`apad` to the same number of samples), apply transform (fit for vertical, crop-pan), grade chain, scale/pad to canvas, `fps=30`, set source-audio gain / mute ranges. `Timeline.segment_positions()` does the same whole-frame arithmetic, so absolute placements (voice, captions, music, markers, chapters) match the rendered file to the frame regardless of how many fractional cuts precede them. Intermediate: `libx264 -crf 16 -preset fast` (or `h264_videotoolbox` for previews) + `pcm_s16le` audio, 48 kHz stereo.
2. **Join pass**: concat demuxer for cuts (each entry carries a `duration` directive pinned to its frame count); `xfade`/`acrossfade` chained for transitions with frame-rounded offsets/overlaps (only when requested; default hard cuts).
3. **Audio bus**: `[program_audio]` = joined source audio (+ voice track overlays). Speech ranges for ducking come from transcripts mapped to timeline time (`duck.mode=auto`) or from an explicit list. Music gain automation via `sendcmd` file with ramps (attack/release) → mixed with `amix=normalize=0`. Then voice cleanup (optional), then **two-pass loudnorm to −14 LUFS / −1 dBTP** on the final mix.
4. **Captions**: generated ASS (libass) with fonts checked for Polish glyphs, burned in the final pass via `ass=` filter. Separate `captions.srt` (subtitles from transcript) written for upload, not burned.
5. **Master encode**: `libx264 -profile:v high -level 4.2 -crf 18 -preset slow -pix_fmt yuv420p -g 15 -bf 2 -movflags +faststart -color_primaries bt709 -color_trc bt709 -colorspace bt709`, `aac 384k 48 kHz stereo`. Preview: 720p, videotoolbox, single-pass loudnorm.
6. Progress via `-progress pipe:1` parsed into `jobs/<job>.json` (percent, eta, current step).

---

## AI layer conventions

- `ai/openrouter.py`: `chat(model, messages, json_schema=None, images=[], temperature=0.2, max_tokens=...)` → parsed JSON (`response_format` json_object + robust extraction fallback from amazonia-studio), retries with backoff on 429/5xx, logs cost from `usage` to the project ledger. Model roles in `defaults.yaml`: `models.planner: anthropic/claude-opus-5`, `models.analyst: anthropic/claude-sonnet-5`, `models.vision: google/gemini-3.8-flash`, `models.writer: anthropic/claude-sonnet-5`.
- Every prompt states: project language, that instructions spoken to the editor at clip start must be extracted, that the LAST take usually wins, and returns strict JSON validated with pydantic; on validation failure retry once with the error appended.
- `ai/elevenlabs.py`: thin httpx client (no SDK dependency required, but SDK allowed): `transcribe(wav) -> dict`, `compose_music(prompt, ms, mode) -> bytes`, `tts(text, voice_id, model) -> bytes`, `clone_voice(name, files) -> voice_id`, `isolate(wav) -> bytes`, `sfx(text, seconds) -> bytes`.
- `ai/fal.py`: `upload(path)`, `run(app, args)` (subscribe with logs), `submit/status/result`, `thumbnail_edit(prompt, image_paths, n)`, `seedance_i2v(image, prompt, seconds)`, `topaz_upscale(video)`.

---

## Web editor (`server/`)

FastAPI on `http://localhost:8765`. Pages: project list → project page (clip grid with status, footage log, plan) → timeline editor.

Views: **Clips** (source review: transcript, waveform, mute tool, denoise buttons), **Program** (default when a timeline exists: final-cut strip with frames, lanes for captions/music/mutes/markers, segment inspector with in/out nudges + looping source player + transcript-word snapping, split/reorder/delete, Save, Save & render preview, Tidy cuts, Accept draft), **Advanced** (raw lists), Plan / Footage log / QC / Publish tabs.

Original v1 feature list (still valid, lives under Clips/Advanced):
- Video player (proxy) synced with wavesurfer waveform + regions: transcript words rendered under the waveform, takes highlighted, detected music regions shaded.
- Clip list in order with kind/role badges; drag to reorder segments; set in/out; toggle include; choose transform for vertical clips.
- **Mute/duck tool**: select a range on the waveform → set gain (mute / −12 dB / custom) → saved to `mute_ranges`.
- Captions list (edit text/time/style); music cues list (choose track, gain, duck amount); markers.
- Buttons: Transcribe / Analyze / Plan / Generate music / Render preview / Render master / QC / Publish pack; job progress bar; "what to do next" hint (state machine like amazonia-studio's `renderActions`).
- Save writes `plan/timeline.json` (with `edited_by_human: true`) and never overwrites human edits with a new plan unless asked (`plan --force` writes `timeline.draft.json`).

---

## Testing

- `tests/fixtures/make_fixtures.py`: generates synthetic clips with ffmpeg lavfi (testsrc2 + sine, vertical and horizontal, HLG-tagged variant, VFR variant) and, when keys are present, a Polish narration WAV via ElevenLabs TTS muxed into a clip so transcribe/analyze can be tested end-to-end for cents.
- Unit tests for: probe parsing, timeline validation, duck automation file generation, ASS generation (Polish glyph check), loudnorm parsing, render of a 3-segment timeline (no API).
