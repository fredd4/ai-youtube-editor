# AI YouTube Editor — Architecture

Semi-automated pipeline + local web editor that turns raw phone footage (travel vlogs, Polish narration by default) into a finished 16:9 YouTube master, with an AI agent (Claude) acting as the editor-in-chief on top of deterministic tooling.

Design principles:
1. **Everything is a file on disk.** One directory per project, JSON/YAML state, no database. Every stage writes artifacts that the next stage reads; every stage is re-runnable and idempotent.
2. **Deterministic tooling, AI decisions.** The edit lives in an explicit **cut** (`plan/cut.json`, speech addressed by sentence ids) which one deterministic resolver turns into a **timeline JSON** (EDL) for ffmpeg/ffprobe to render. LLMs only produce/modify JSON (analysis, plan, captions, titles). The human and the agent review in the web editor.
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
│   ├── words.py              # transcript word spans in clip time — the bottom of the stack
│   ├── cut.py                # cut v2: plan/cut.json models, validate(), resolve() → Timeline
│   ├── timeline.py           # Timeline / EDL data model (pydantic) + structural checks (derived; no edit API)
│   ├── migrate.py            # one-shot v1 timeline.json → v2 cut.json (`ytedit migrate`)
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
│   │   ├── sentences.py      # transcripts+analysis → analysis/sentences.json (numbered sentence catalogue; script-first planning)
│   │   ├── plan.py           # all analyses → plan/edit_plan.json + plan/cut.json (beats, captions, music cues, narration asks, titles)
│   │   ├── voice.py          # voice/incoming/*.wav (manifest-mapped) → cleaned voice/*.wav + a voice beat in cut.json
│   │   ├── locations.py      # footage_log locations → analysis/places.json (canonical) → location captions on beats
│   │   ├── music.py          # style selection + generation → music/*.mp3 with sidecar json
│   │   └── publish.py        # titles, description, chapters, thumbnail prompts/generation
│   ├── inspect.py            # timecode inspector (`ytedit at`): rendered timecode → segment/sentence/caption
│   ├── verify.py             # render verification (`ytedit check-render`): transcribe the finished file
│   ├── qc.py                 # cut validator + playbook rule checker (+ rendered file loudness)
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
        ├── analysis/         # <clip>.json per clip + footage_log.json (merged, chronological) + sentences.json/.md + places.json + captions_report.md
        ├── plan/             # cut.json (the edit, source of truth), timeline.json (derived), edit_plan.json, history/
        ├── music/            # generated tracks + sidecar json
        ├── voice/            # narration pickups (recorded or TTS): <label>.wav + <label>.json sidecar (source, kept ranges, text, duration)
        │   └── incoming/     # RAW DROP ZONE for `ytedit voice` — manifest.yaml, raw WAVs, <file>.transcript.json, state.json (hash → output), report.md
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
| 3b | sentences | `ytedit sentences <slug>` | transcripts + analysis → analysis/sentences.json + .md (numbered sentence catalogue); auto-runs at the start of `plan` | deterministic (`ytedit/ai/sentences.py`) |
| 4 | plan | `ytedit plan <slug>` | footage_log (compacted with the sentence catalogue) → plan/edit_plan.json + **plan/cut.json** (+ the resolved plan/timeline.json) + narration_requests.md | OpenRouter (Opus) |
| 4a | resolve | `ytedit resolve <slug>` | plan/cut.json → plan/timeline.json (overwritten every time); runs implicitly whenever the timeline is older than the cut | deterministic (`ytedit/cut.py`) |
| 4b | validate | `ytedit validate <slug>` | plan/cut.json → the validator's issues, no writes | deterministic (`ytedit/cut.py`) |
| 1b | denoise | `ytedit denoise <slug> --clip cNNN` | media/audio/<clip>.wav → <clip>.denoised.wav; render uses it when `clips[id].use_denoised` | ElevenLabs Audio Isolation or local afftdn |
| 4c | voice | `ytedit voice <slug>` | voice/incoming/*.wav (manifest-mapped) → transcribe, cut retakes/instructions/stutters/pauses, insert a `voice` beat into plan/cut.json | ElevenLabs Scribe + deterministic (`ytedit/ai/voice.py`) |
| 4d | captions | `ytedit captions <slug>` | footage_log locations → analysis/places.json (canonical, one writer call, cached) → location captions written onto beats in plan/cut.json + analysis/captions_report.md | OpenRouter (writer) + deterministic (`ytedit/ai/locations.py`) |
| 5 | music | `ytedit music <slug>` | plan music cues → music/*.mp3 + cue beat ranges in plan/cut.json | ElevenLabs Music |
| 6 | render | `ytedit render <slug> --draft` / `--preview` / `--master` | timeline.json → renders/draft.mp4 (from proxy, ~1 MB/10s) / renders/preview.mp4 / exports/master.mp4 (hardware tier by default, `--x264` for the slow tier) | ffmpeg |
| 7 | qc | `ytedit qc <slug>` | cut (re-resolved first, validator issues prefixed `cut:`) + timeline + master → qc_report.json/md | `ytedit.cut.validate` + rules + loudnorm measure |
| 8 | publish | `ytedit publish <slug>` | → exports/publish.json (titles, description, chapters), thumbnails/ | OpenRouter + fal |
| — | at | `ytedit at <slug> <mm:ss>` | cut.json + timeline.json → the **beat** (id, kind, clip, sentences, which shot is on screen) plus the segment/audio/caption/chapter at that rendered timecode | deterministic (`ytedit/inspect.py`) |
| — | check-render | `ytedit check-render <slug> [--render draft\|preview\|master]` | rendered file + timeline → `renders/check/<name>.report.md` (air, chopped words, replayed audio at every audio boundary) | ElevenLabs Scribe (`ytedit/verify.py`) |
| — | migrate | `ytedit migrate <slug> [--dry-run]` | a v1 plan/timeline.json → plan/cut.json + plan/migrate_report.md; the v1 file moves to plan/history/timeline.v1.json | deterministic (`ytedit/migrate.py`) |
| — | serve | `ytedit serve` | web editor at http://localhost:8765 | FastAPI |
| — | run | `ytedit run <slug> [--until …]` | ingest → transcribe → analyze → sentences → plan, skipping stages already done, then re-resolves the timeline. Music is deliberately **not** in the chain: a bed is only worth generating once the cut is stable and it costs real money per track. | — |

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

### analysis/sentences.json (deterministic, `ytedit/ai/sentences.py`) — script-first planning
A pre-pass over every clip's transcript, run automatically at the start of `plan` (also `ytedit sentences <slug>` on its own): splits words into sentences — ending at a word whose text ends in `.?!…`, at a pause longer than 1.2 s, or at the clip's last word — and numbers them `<clip>#<n>` (1-based) so the planner can reference dialogue by id instead of picking raw seconds.
```json
{
  "project": "the reference project", "language": "pl", "generated": "2026-09-06T12:00:00+00:00",
  "clips_count": 218, "sentences_count": 803,
  "clips": [
    {"id": "c001", "sentences": [
      {"id": "c001#1", "clip": "c001", "n": 1, "s": 5.0, "e": 8.4, "text": "To jest tramwaj numer 28.",
       "words": 6, "lang": "pl",
       "instruction": false,       // overlaps analysis.instructions[] — never usable
       "retake_of": null,          // "<clip>#<m>" when this is a rejected take attempt; points at the kept one
       "duplicate_of": null,       // "<clip>#<m>" when a LATER sentence (anywhere in the project) says
                                   // near-the-same thing (Jaccard word-overlap >= 0.7) — last-take rule,
                                   // generalized across clips; never set when that later match is itself
                                   // an instruction
       "keep_default": true}       // convenience only: true iff none of the three flags above are set
    ]}
  ]
}
```
This catalogue is what a `speech` beat's `sentences` ids point into, and `ytedit.cut` validates every one of them against it (unknown, belonging to another clip, already used by another beat, a spoken instruction, or non-contiguous → an error; a `retake_of`/`duplicate_of` flag → a warning) before deriving any seconds from it. A clip with no transcribed speech gets `{"id": "<clip>", "sentences": []}`. Before it reaches the planner prompt, `ytedit.ai.sentences.compact_footage_log_for_planner` replaces each clip's footage-log `segments[]`/`takes[]` with this sentence list (instruction sentences omitted outright; retake/duplicate ones kept but marked `"skip": "retake, use ..."` / `"skip": "duplicate, use ..."`) — measured on the 218-clip `projects/the reference project` footage log, this shrinks the compact-JSON prompt payload by about 5% even though it adds structured per-sentence metadata, because it replaces the old free-text segment/take detail for every clip that has transcribed speech.

### analysis/places.json (deterministic + one writer call, `ytedit/ai/locations.py`)
Canonicalizes every clip's free-text `analysis/<clip>.json: location` into a stable place, so `ytedit captions` can tell "new place" from "same place, different wording" without asking a model per clip. Built by `ytedit captions <slug>` (cached; `--force` re-asks the writer):
```json
{
  "project": "the reference project", "language": "pl", "generated": "2026-09-07T12:00:00+00:00",
  "model": "anthropic/claude-sonnet-5", "cost_usd": 0.004,
  "clips": [
    {"clip": "c001", "place_id": "alfama", "label": "Alfama · Lisboa", "region": "Lisboa",
     "source_name": "Alfama", "confidence": 0.85, "inherited": false}
  ]
}
```
Distinct raw `(name, city, country)` strings are grouped once (exact match only) and sent to the writer model (`captions.places.system`/`.user` in `docs/playbook/prompts.md`) in a single call per project; the model merges near-duplicates ("Belém (prawdopodobnie)" == "Belém") into one `place_id`/`label`/`region`. A clip with no location name and confidence below 0.5 has nothing to normalize, so it instead inherits the place of its nearest chronological neighbour (by clip order, `inherited: true`) that has one; a clip with no place at all (nothing named anywhere nearby) gets `place_id: ""` and is skipped when placing cards.

### plan/cut.json — the edit (source of truth)
The cut is a list of **beats** in screen order. Speech is addressed by sentence (or word) ids, never by seconds; seconds are produced in exactly one place, the resolver. `plan/cut.json` is what `plan` writes, what `voice` / `captions` / `music` update, what the web editor edits, and what `ytedit migrate` produces from a v1 timeline.
```json
{
  "version": 2, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
  "beats": [
    {"id": "b001", "uid": "3e4669c0", "kind": "broll", "clip": "c171", "in": 0.0, "out": 3.4,
     "audio": "ambient", "role": "cold-open", "transform": {"fit": "cover"}, "grade": "default",
     "transition_in": {"type": "cut", "duration": 0.0}, "notes": "opening reveal"},

    {"id": "b002", "uid": "9a11ff02", "kind": "speech", "clip": "c048",
     "sentences": ["c048#3", "c048#4", "c048#5"],
     "on_camera": true,
     "shots": [
       {"clip": "c013", "in": 0.0, "out": 3.5, "after": "c048#3", "transform": {"fit": "cover"}, "notes": "stalls"}
     ],
     "transform": {"fit": "cover"}, "grade": "default", "transition_in": {"type": "cut"},
     "gain_db": 0.0, "role": "a-roll", "notes": "one ticket, 16 sites"},

    {"id": "b003", "uid": "c0ffee01", "kind": "speech", "clip": "c211",
     "sentences": ["c211#2", "c211#3"], "on_camera": false,
     "shots": [{"clip": "c020", "in": 4.0, "out": 9.0}, {"clip": "c031", "in": 10.0, "out": 16.0}],
     "role": "a-roll", "notes": "post-trip narration as voice-over"},

    {"id": "b004", "uid": "d00d0004", "kind": "voice", "file": "voice/n004.wav",
     "shots": [{"clip": "c150", "in": 0.0, "out": 5.0}, {"clip": "c152", "in": 2.0, "out": 8.0}],
     "gain_db": 0.0, "role": "b-roll", "notes": "pickup: next day"}
  ],
  "music": [
    {"id": "m001", "file": "music/m001.mp3", "from": "b001", "to": "b007",
     "gain_db": -16.0, "fade_in": 2.0, "fade_out": 3.0,
     "duck": {"mode": "auto", "amount_db": -12.0, "attack": 0.15, "release": 0.6}}
  ],
  "captions": [
    {"id": "t001", "beat": "b002", "offset": 0.3, "duration": 3.0,
     "text": "LISBOA, PORTUGAL", "style": "location", "position": "lower-left"}
  ],
  "chapters": [{"beat": "b001", "title": "Portugal, którego nie planowałem"}],
  "markers": [{"beat": "b001", "label": "hook"}],
  "mute_ranges": [{"clip": "c027", "s": 0.1, "e": 3.0, "gain_db": -60.0, "reason": "car radio"}],
  "meta": {"title_candidates": [], "generated_by": "plan@…", "edited_by_human": false, "notes": ""}
}
```

**Beat kinds.** Common fields: `id`, `uid`, `kind`, `role` (`cold-open`, `a-roll`, `b-roll`, `outro`, free text), `transform`, `grade`, `transition_in` (`cut`|`fade`|`xfade` + duration — into this beat), `notes`.

- **`speech`** — the narrator talking, from a source clip. `clip`, plus exactly one of `sentences` (contiguous ids of that clip's catalogue, in order — a gap in `n` is a validation error, split it into two beats) or `words` (`[first_index, last_index]` inclusive into the clip's transcript `words[]`, the escape hatch for cutting inside a sentence). `on_camera` (default `true`): `true` → the narrator's own picture is the base and `shots` are inserts; `false` → he is heard, not seen, `shots` cover the whole beat and his own picture only fills a tail no shot covers. `shots[]` are picture-only inserts `{clip, in, out, after?, transform?, grade?, notes?}`; `after` is a sentence id of this beat (or a word index when the beat uses `words`) and the shot starts at that sentence's end, an omitted `after` means the start of the beat, and shots after the same point chain in list order. **A shot never carries audio of its own** — the beat's audio keeps playing underneath. `gain_db` is a manual trim on top of automatic speech levelling.
- **`broll`** — picture with its own ambient sound, no narration: `clip`, `in`, `out` in seconds (these *are* seconds — there is no speech to protect), `audio: ambient` (default: source audio, mute ranges apply) or `mute`.
- **`voice`** — a recorded narration pickup (`voice/*.wav`, already trimmed by `ytedit voice`): `file`, `shots[]` (no `after` — they chain from the beat start), `gain_db`. The beat's length is the WAV's length (ffprobe, cached in `voice/<file>.wav.probe.json` keyed by mtime). Shots chain from 0, the last one is extended to the WAV end (clamped by its clip's duration), shots longer than the WAV are trimmed, and picture that still falls short is the error `voice_picture_short`. Its shots are silent by default; `audio: ambient` deliberately keeps the location's own sound under the pickup.

**Music, captions, chapters, markers are positioned by beat, never by second:** `music.from`/`to` is an inclusive beat range (→ `at` = start of `from`, `end` = end of `to`), `captions.beat` + `offset` + `duration`, `chapters.beat`, `markers.beat`. Deleting a beat something references drops that caption/chapter/marker with a warning on resolve; a music cue that lost one end snaps to the first or last beat (`music_beat_snapped`), losing both ends is an error. `mute_ranges` stay in **clip** time — they describe the source, not the cut. `meta.edited_by_human` lives here, in `cut.json.meta`.

**Display id vs. identity (`id` vs. `uid`).** `id` (`b001`…) is a readable display label, renumbered in list order on every save; `uid` (8 hex chars, `secrets.token_hex(4)`) is assigned once and never changed, and it is what every reference resolves through. `load_cut` accepts either spelling for `music.from`/`to` and `captions`/`chapters`/`markers` `beat` and rewrites it to the uid on the way in; `save_cut` renumbers the display ids and then writes **uids** for every reference, because writing ids back would silently re-point a caption at a different beat the moment someone deleted a beat by hand. `save_cut` writes with `exclude_none` in the models' declaration order, so saving an unchanged cut is byte-stable.

### plan/timeline.json — the resolved render artifact
A **derived** file: `version: 2`, written only by `ytedit.cut.resolve`, never edited by hand or by any pass, safe to delete and regenerate. Nothing in the codebase edits a timeline any more — `ytedit/timeline.py` is a data model plus its structural checks, with no edit API and no anchors.
```json
{
  "version": 2, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
  "tracks": {
    "video": [
      {"id": "s001", "uid": "3e4669c0", "clip": "c003", "in": 12.0, "out": 16.5, "role": "cold-open",
       "beat": "9a11ff02", "transform": {"fit": "cover|contain|blur-fill|crop-pan", "zoom": 1.0},
       "grade": "default", "transition_in": {"type": "cut|fade|xfade", "duration": 0.0}, "speed": 1.0,
       "mute_source": false, "source_audio_gain_db": 0, "notes": "drone reveal",
       "audio_from": {"clip": "c048", "in": 30.2, "out": 33.7},
       "audio_window": {"in": 12.05, "out": 16.4}}
    ],
    "voice": [{"id": "v001", "file": "voice/n004.wav", "at": 0.5, "end": 6.1, "gain_db": 0}],
    "music": [
      {"id": "m001", "file": "music/arrival_warm.mp3", "at": 0.0, "end": 185.0, "gain_db": -18,
       "fade_in": 2.0, "fade_out": 3.0,
       "duck": {"mode": "auto", "amount_db": -12, "attack": 0.15, "release": 0.6}}
    ],
    "captions": [
      {"id": "t001", "at": 0.5, "end": 3.0, "text": "LIZBONA, PORTUGALIA", "style": "location",
       "position": "lower-left"}
    ],
    "sfx": []
  },
  "mute_ranges": [{"clip": "c005", "s": 3.0, "e": 20.0, "gain_db": -60, "reason": "copyrighted bar music"}],
  "markers": [{"at": 0.0, "label": "hook"}],
  "chapters": [{"at": 0, "title": "Przyjazd do Lizbony"}],
  "meta": {"title_candidates": [], "generated_by": "plan@2026-09-04", "edited_by_human": false,
           "source": "cut.json@2026-09-09T18:22:00+00:00"}
}
```
Times are seconds (float). A video segment's timeline position is cumulative (an `xfade` overlaps the previous segment by its duration); voice, music, captions, markers and chapters are absolute timeline time; `mute_ranges` are **clip** time and apply wherever that clip range is used.

Compared with v1 the segment shape gains two fields and loses several:
- `beat` — the `uid` of the `cut.json` beat this segment came from. Every segment of one beat carries the same value and they are audio-contiguous by construction (`prev.audio_out == next.audio_in` exactly).
- `audio_window: {in, out} | null` — the part of the segment's audio that is actually **heard**, in the time base of its audio source (the `audio_from` clip when one is set, otherwise the segment's own clip, exactly like `mute_ranges`). Everything inside the segment but outside the window renders as silence. `null` means the whole segment is heard.
- `audio_from: {clip, in, out}` keeps its v1 meaning: the picture stays `clip[in, out]` but the sound is read from the named range instead, trimmed/padded to the segment's own frame count, with that segment's `speed`/`source_audio_gain_db` and the audio clip's mute ranges and denoised WAV. `mute_source: true` still wins and renders silence. The resolver sets it on every shot, so a picture insert never interrupts the narration underneath.
- Gone: `sentence_ids` on a segment, and every `anchor` — on voice items, on captions, and `meta.anchor_issues`. Nothing drifts, because nothing moves: a re-resolve rebuilds the whole file from the cut.
- `meta.source` records `cut.json@<mtime iso>` — which state of the cut this timeline was resolved from.

**Deterministic segment identity.** `VideoSegment.id` is the display label (`s001`…, reassigned on every resolve); `VideoSegment.uid` is `sha1(beat uid + index)[:8]` — derived, not random. The uid is part of the render's per-segment cache key, so re-resolving an unchanged cut must not invalidate every cached segment; a random uid per resolve would have thrown the entire segment cache away on each run.

### The resolver (`ytedit/cut.py`)
`resolve(project, cut) -> Timeline` is the **only** place in the system where speech becomes seconds. It is deterministic and side-effect free apart from the cached WAV probes: the same cut and the same project always produce the same timeline, down to the segment uids. It reads the transcripts, the sentence catalogue, the per-clip analyses and the clip registry — no LLM, no ffmpeg beyond a cached `ffprobe` of a narration WAV.

Settings (`config/defaults.yaml.pacing`): `speech_pad_before` (0.30), `speech_pad_after` (0.45), `word_guard` (0.05, the minimum clearance from a neighbouring word), `min_shot_seconds` (0.8), `air_warn_below` (0.35).

**Why the previous design failed.** v1 stored the edit as an EDL in seconds and made that file the source of truth. Every producer and every fixer — the planner, the speech pad, the sentence snap, overlay cutaways, the audio ledger, the anchors, the web editor — had to re-derive from the transcript where a word starts and ends in order not to chop it, and they moved each other's boundaries. Roughly 4 000 lines existed only to police cuts given in seconds (2 325 of `ai/tidy.py` + `ai/overlay.py` + `ai/ledger.py`, plus their 1 686 lines of tests, plus the timeline edit API and the anchors), and the audible defects in the the reference project draft (a cutaway hand-off replaying 0.05–0.15 s of a word, a cut landing on the last syllable, a voice-over ending inside a word) all came from those passes disagreeing. The v2 rule is the fix: **speech is addressed by sentence (or word) ids, never by seconds, and seconds are produced in exactly one place.**

**Speech beat → segments.**
1. Resolve the word range: `sentences` → `[first.s, last.e]` from the catalogue; `words` → the transcript word indices. An unknown id, an id belonging to another clip, an id already used by another beat, a spoken editor instruction, or a non-contiguous run is an **error**. A `retake_of`/`duplicate_of` flag only warns — the catalogue's detection is heuristic (a short "Zobaczcie." or a song chorus trips it), so using such a sentence is the editor's call.
2. Audio window, in clip time: `a0 = first.s − pad_before`, `a1 = last.e + pad_after`, clamped so that `a0 ≥ prev_word.e + word_guard` and `a1 ≤ next_word.s − word_guard`, kept inside the clip, and never reaching into a range the analysis marked as a spoken instruction or a rejected take.
3. Air deficit: whatever step 2 took away, the **picture** keeps. `seg.in`/`seg.out` get the full `speech_pad_before`/`speech_pad_after` (clamped only by the clip's own edges) and the segment carries `audio_window: {in: a0, out: a1}`, so the neighbouring word is on screen for a fraction of a second but is never heard. That is how "there is a pause here" becomes a guarantee rather than a hope. When the clip boundary also blocks the picture the air is genuinely shorter, and the validator reports `air_short` when the total air at a real cut (this beat's tail plus the next beat's head, and only where two speech beats are adjacent — anything else between two takes *is* the air) falls below `pacing.air_warn_below`.
4. Shots split the run. For a shot with `after = X` the own-picture piece before it ends exactly at `X.e` (no air — the narration continues), the shot segment is `{clip, in, out, audio_from: {clip: beat.clip, in: X.e, out: X.e + shot_len}}`, and the next piece resumes where that borrowed audio ended. Chained shots continue from the previous one's end; borrowed audio is clamped to `seg.out` and a shot that would run past it is trimmed (`shot_trimmed`). If the resumed own-picture piece would be shorter than `min_shot_seconds`, the last shot is held to `seg.out` instead of flashing back to the narrator for a quarter of a second (clip permitting; otherwise `short_piece`). With `on_camera: false` the shots chain from `seg.in`, an `after` is meaningless and is ignored with `shot_after_ignored`, and a tail no shot covers is filled with the narrator's own picture and warned as `narrator_visible`.
5. Every produced segment carries `beat`, `role`, `transform` (the shot's own or the beat's), `grade`, `gain_db`, and `transition_in` only on the beat's first segment.

A **broll** beat resolves to one segment (`mute_source` when `audio: mute`); an ambient range that contains transcript words warns `speech_in_broll` — make it a speech beat or mute it. A **voice** beat resolves to its shot segments plus one `tracks.voice` item `{id, file, at, end, gain_db}`.

Picture ranges are clamped to the clip's **last whole frame** (registry duration minus one frame), not to the raw container duration: the reference project's `c077` registers 15.806 s but holds 474 frames = 15.800 s, and a cut reaching past that renders one frame short.

After the segments are placed (`Timeline.segment_positions()`, frame-exact), music, captions, chapters and markers are converted from beat references to absolute seconds. A caption's `offset` is relative to its beat's start and its end is clamped to the beat end + 0.5 s. Music beds never overlap: when two cues meet inside one beat the later cue's start wins and the earlier one ends there.

**Validation *is* resolution.** Rules such as `voice_picture_short`, `air_short`, `short_piece` and `shot_trimmed` only exist once a beat has been laid out, so `validate(project, cut) -> list[Issue]` runs the resolver and keeps its issues instead of its timeline, while `resolve()` runs the same pass and raises `CutError` if any issue is an `error`. Nothing is computed twice and the two can never disagree. `resolve_verbose()` hands back both halves for a producer that needs the geometry *and* the findings; `beat_spans()`/`beat_at()` give a producer holding a time (a planner caption at 41.2 s, a music cue ending at 3:20) the beat to attach it to; `ensure_resolved(project)` re-resolves whenever `plan/timeline.json` is older than `plan/cut.json` and is called at the start of `render`, `qc`, `at` and `check-render`.

Issue codes — errors: `unknown_clip`, `bad_range`, `beat_range`, `unknown_sentence`, `instruction_sentence`, `sentence_reused`, `sentences_not_contiguous`, `unknown_word_index`, `words_in_instruction`, `words_reused`, `unknown_shot_after`, `voice_file_missing`, `voice_picture_short`, `duplicate_uid`, `music_beat_missing`. Warnings: `retake_sentence`, `duplicate_sentence`, `air_short`, `speech_in_broll`, `narrator_visible`, `short_piece`, `shot_trimmed`, `shot_after_ignored`, `music_beat_snapped`, `music_overlap`, `caption_dropped`, `chapter_dropped`, `marker_dropped`.

Because each sentence/word range is claimed at most once (a `sentences` beat also claims the transcript word indices its range covers, so a `words` beat overlapping it is caught by the same `words_reused` rule) and shots never carry their own audio, **no audio can play twice** — structurally, not by a checking pass. The v1 audio ledger and QC rules 31–36 are therefore deleted rather than ported.

### Migration (`ytedit migrate`)
`ytedit migrate <slug> [--dry-run]` (`ytedit/migrate.py`) converts a v1 `plan/timeline.json` into a v2 `plan/cut.json`. There is no backward compatibility: a project is migrated once. Input is the v1 timeline plus `analysis/sentences.json` (built first if missing), the transcripts and the voice sidecars; output is `plan/cut.json`, `plan/migrate_report.md` (every decision, every ambiguity), the v1 file moved to `plan/history/timeline.v1.json`, and a freshly resolved `plan/timeline.json`. `migrate_project(project, dry_run=False, out_dir=None) -> MigrationReport` carries `cut`, `issues`, `notes`, `validation` (`str(Issue)` lines), `duration_v1`/`duration_v2`, `beat_lines`, `duration_drift` and `markdown()`. A dry run touches nothing inside the project: it writes into `out_dir` when one is given and otherwise only returns the report.

Per video segment, in order: muted or wordless segments become `broll` beats (`audio: mute` when muted, else `ambient`), and a run of consecutive broll segments under one voice item becomes that voice beat's shots. A segment with words becomes a `speech` beat whose sentences are those of its audio clip overlapping the audio range by at least 50 % of the sentence; consecutive segments of the same clip whose audio ranges abut merge into one beat, and a segment with an `audio_from` of that clip in between becomes a `shot` with `after` = the sentence ending at `audio_from.in`. A `voice/vo_<clip>_<in>_<out>.wav` item (v1 cut the narrator's own audio out to a WAV) becomes an off-camera `speech` beat of that clip; any other voice item becomes a `voice` beat with those shots. Captions/music/chapters/markers move onto the beat under their absolute time; `mute_ranges` are copied; `meta` is copied with `generated_by += " + migrate@<date>"`. Then `validate` + `resolve`, and the report ends with the validator's output and a duration comparison.

**Decisions taken while implementing this** — the rules above leave real choices open; each of these carries the issue code the report uses, so the user can find them in `migrate_report.md`.

1. **Abutting** (`MAX_HANDOFF_OVERLAP = 0.25`): two audio ranges of the same clip continue one take when the second starts no later than one frame after the first ends *and* overlaps it by less than 0.25 s. v1's cutaway hand-offs habitually replayed 0.05–0.15 s (`s124` audio `c141 7.00–11.00` → `s125` `c141 10.91–12.00`); that is drift, not a second reading.
2. **Chained cutaways share one `after`.** A run of back-to-back cutaways over one continuous take (`s019 s020 s021` over `c075`) is exactly what v2 chaining produces, so only the first names a sentence; the rest carry the same `after` and chain in list order. Measuring each one's own start against the nearest sentence end would have reported three false `shot_after_approx` for one correct hand-off.
3. **`after = None`** when the cutaway's borrowed audio starts at or before the first kept sentence's start. If the beat also has its own picture, the picture order changes (v2 plays the beat's own picture first) — reported `shot_leads_beat`, 3 cases in the reference project.
4. **A run with no sentence at ≥ 50 %** is not a speech beat: each of its segments becomes an ambient `broll` beat, and the v2 validator's `speech_in_broll` warning is what flags the leftover words. This is what keeps a two-word tail such as `s045` from becoming a speech beat with an empty `sentences` list.
5. **Roles.** A beat takes the role of its first own-picture segment (or `a-roll`/`b-roll` when every segment was a `cutaway`, since that role describes a segment that no longer exists in v2). A segment v1 called `b-roll`/`cold-open`/`outro` whose own sound carries whole sentences still becomes a speech beat (the rule is about words, not roles) but is reported `broll_became_speech` — 29 in the reference project — because v2 will snap and pad that range to the sentence, and background chatter should be muted instead.
6. **Voice items are trimmed at both edges**, not only at the start: the picture after the pickup ends stays `broll` too. Without the trailing split the resolver would trim the last shot to the WAV length and the programme would lose that picture. A leftover under 0.5 s is reported `short_remainder` (3 in the reference project) so it can be deleted rather than shipped.
7. **Ambient under a pickup is lost.** A segment that played its own sound under a v1 pickup is reported `voice_ambient_lost` (or `voice_over_speech` when the lost audio contained whole sentences).
8. **Music range edges snap** (`MIN_CUE_OVERLAP = 0.5`): a beat a v1 cue covers for less than 0.5 s is dropped from the cue's beat range in favour of its neighbour, reported `music_range_approx`. v1 cues ended on chapter times that land a few tenths inside the beat opening the next chapter; taking `end − ε` literally handed a whole 26 s beat to the outgoing cue.
9. **Flagged sentences are kept, not dropped.** A sentence the catalogue marks `instruction`/`retake_of`/`duplicate_of` that the v1 cut actually played stays in the beat and is reported `flagged_sentence` (7 in the reference project). The v2 validator errors on an instruction, so migration can legitimately produce a cut that does not resolve yet — the alternative, silently deleting narration the user approved in the v1 draft, is worse. A failing `resolve` is recorded in the report's Notes and leaves `duration_v2` unset.
10. **Sentences that reach outside the v1 cut** by more than 0.5 s in total are reported `sentence_extends`: a sentence kept at 50–99 % coverage brings its missing head or tail back, so v2 plays words the v1 draft cut off.
11. **A dry run never writes into the project.** When `analysis/sentences.json` is missing it is built in memory, and `validate`/`resolve` are handed a project view whose `analysis/` is a temporary directory of symlinks plus that catalogue — otherwise every beat would come back `unknown_sentence` and the validator section of the report would be pure noise.
12. **`vo_<clip>_<in>_<out>.wav` files are dead after migration**: the beat reads the clip's own audio through the resolver instead. They are left on disk (nothing under a project is ever deleted) and each is reported `vo_extract`.

The report also uses `edge_trimmed` (a partial sentence dropped at a run's edge), `shot_after_approx`, `cutaway_orphan`, `vo_no_sentences`, `voice_pickup`, `voice_overlap`, `voice_no_end`, `voice_no_picture`, `voice_unplaced`, and `caption_dropped`/`music_dropped`/`chapter_dropped`/`marker_dropped`.

**Measured on the reference project** (dry run, 2026-09-09): 186 v1 segments → **121 beats** (73 speech, 40 broll, 8 voice) with 73 shots, 19 captions, 10 music cues, 10 chapters, 7 markers; 85 issues and 20 validator findings (6 errors, all `flagged_sentence` fallout). Resolving with the errors bypassed gives 1017.4 s against v1's 1114.2 s — and the whole 96.8 s of that gap is the six erroring beats producing no picture at all. Only 7 of the other 115 beats drift by more than 0.5 s, the largest by 2.5 s. The reconstruction is faithful; the six flagged sentences are the whole of the manual work.

### plan/edit_plan.json (LLM reasoning, human-readable)
Story outline (beats with target timecodes per playbook: 0:00 hook, 0:07 promise, 0:30 interrupt, 3:00/6:00 re-engagements, end payoff), which clips serve which beat, cold-open montage picks, subscribe-CTA placement, music cue sheet with mood per section, list of **narration requests** (what the user should record for intro/outro/bridges, with suggested scripts in the project language), risk flags (copyright music, missing footage), title candidates, thumbnail concepts. Written also as `plan/edit_plan.md`.

---

## Rendering model (`media/render.py`)

0. **Pre-flight** (`preflight()`, before any ffmpeg work): runs `Timeline.validate(project, skip_music=, skip_voice=)` — structural checks plus, for every video segment and every `audio_from` override, its range checked against that clip's known duration from the registry (`0 <= in < out <= duration + 0.05`) — and additionally confirms every referenced clip actually has a normalized source on disk (the one thing the JSON-only validator cannot see). This is a structural backstop on the derived file: the editorial gate is `ytedit validate`, which runs the cut validator over `plan/cut.json` (and `ytedit.cut.ensure_resolved` re-resolves the timeline before the render reads it, so pre-flight never sees a stale file). All problems are collected and reported at once as a numbered list; `render` refuses before touching ffmpeg rather than failing 20 minutes in on segment 179 of 190. `--no-music`/`--no-voice` skip the corresponding track's checks (a missing music file is not a reason to refuse a render that ignores music) and, symmetrically, drop that track from the mix without modifying the timeline file. A disk-space guard (`render.mb_per_second` × programme seconds × `render.disk_headroom_factor`, defaults 20 MB/s × 3x; `render.draft_mb_per_second`, default 1, for `--draft`) logs free space and refuses early when there isn't enough headroom for the segment cache + intermediates + export to coexist.
1. **Segment pass** (cached by hash of segment spec, including its frame count, canvas and mode): for each video segment cut from the normalized source: `-ss in`, then exactly `round((out−in)/speed × fps)` frames (`setpts=PTS-STARTPTS` + `-frames:v`, audio `atrim`/`apad` to the same number of samples), apply transform (fit for vertical, crop-pan), grade chain, scale/pad to canvas, `fps=30`, set source-audio gain / mute ranges. A segment carrying an `audio_window` also gets a `volume=enable='not(between(t,w0,w1))':volume=0` gate in its audio chain — applied in the same post-`atempo`, post-`-ss` time base as the mute ranges and *before* the speech measurement below, so audio outside the window renders silent inside that segment: the picture can hold the full `speech_pad_before`/`speech_pad_after` while a neighbouring word that happens to be on screen is never heard, and a silenced word never pulls the cut's measured level. `--draft` cuts *picture* from the 720p ingest proxy (`media/proxies/<clip>.mp4`) instead of the mezzanine — using the proxy's own probed size, not the mezzanine's `state.json` width/height, since a vertical clip's proxy is a different shape — falling back to the mezzanine with a warning when a clip has no proxy yet; its *audio* always still comes from the mezzanine/denoised WAV, never the proxy's lossy 128k AAC, so a draft sounds exactly like the master. Immediately after that (still inside the segment's own audio chain, before `atrim`/`apad`), any cut whose audio carries transcript words is **speech-levelled**: `measure_speech_gain` (`ytedit/media/audio.py`) runs a `loudnorm print_format=json` analysis pass over the cut's own `-ss/-t` range — gated to just the word ranges via a `volume=enable=...` mute of everything else when they cover under 60% of the cut, measured whole otherwise — and the resulting gain (target `audio.speech_target_lufs`, default −16 LUFS, clamped to ±`audio.speech_gain_max_db`, default 10 dB) is folded into the same `volume=` filter used for the manual `source_audio_gain_db`. Ambient/B-roll cuts (no transcript) and muted cuts are left at their recorded level. The gain actually applied is written next to the cached segment as `<hash>.gain.json` (read back for the render log's speech-leveling table and the job's `speech_gains` list, on a cache hit too) and takes part in the segment's cache key (alongside `audio.speech_target_lufs`/`speech_gain_max_db` and the source clip's transcript mtime) so a re-transcribe or a changed target invalidates it. `Timeline.segment_positions()` does the same whole-frame arithmetic, so absolute placements (voice, captions, music, markers, chapters) match the rendered file to the frame regardless of how many fractional cuts precede them. Intermediate: `libx264 -crf 16 -preset fast` (`h264_videotoolbox` for previews, `-q:v 45`/libx264 ultrafast crf 26 for drafts) + `pcm_s16le` audio, 48 kHz stereo, cached in `renders/segments/` (`renders/segments_draft/` for `--draft` — its own namespace, so a draft render never invalidates or is invalidated by the preview/master cache). `render_segments()` runs this pass across a `ThreadPoolExecutor` sized by `render.workers` (default 3 — each ffmpeg process is already internally multi-threaded, so more than 3-4 rarely helps on an 8-core machine); a cache hit costs a stat + probe, never spawns ffmpeg (and never re-measures speech gain — the sidecar from the original render is reused), so extra workers are free on an already-rendered timeline. Segment order in the output list always matches the timeline regardless of completion order; a failure names its segment id precisely.
2. **Join pass**: concat demuxer for cuts (each entry carries a `duration` directive pinned to its frame count); `xfade`/`acrossfade` chained for transitions with frame-rounded offsets/overlaps (only when requested; default hard cuts).
3. **Audio bus**: `[program_audio]` = joined source audio (already speech-levelled per cut, see above) + voice track overlays, unless `--no-voice`. Each voice pickup is levelled to the same `audio.speech_target_lufs` target (measured once and cached alongside the file as `<file>.loudness.json`, keyed on its mtime and the target/clamp) before its own `gain_db` is added on top. Speech ranges for ducking come from transcripts mapped to timeline time (`duck.mode=auto`) or from an explicit list — for an `audio_from` overlay these already come from the borrowed clip, not the picture clip, and both the ducking ranges (`speech_ranges_from_transcripts`) and the per-cut speech levelling (`segment_speech_ranges`) are clipped to the segment's `audio_window`: a word that is never heard must not duck the music either. Music gain automation via `sendcmd` file with ramps (attack/release), ducking `ducking.amount_db` (default −15 dB, chosen against a −16 LUFS levelled voice and a −18…−21 dB cue gain) → mixed with `amix=normalize=0` (skipped entirely with `--no-music`). Then voice cleanup (optional: `none`/`light`/`full`, see `audio.voice_cleanup`) — a gentle `acompressor` that runs *after* the leveling above, so it is consistency glue rather than gain-riding — then **loudnorm to −14 LUFS / −1 dBTP** on the final mix (two-pass for `--master`, single-pass for `--preview`/`--draft`). This whole pass is identical across all three tiers, so a `--draft` already sounds like the eventual master.
4. **Captions**: generated ASS (libass) with fonts checked for Polish glyphs, burned in the final pass via `ass=` filter. Separate `captions.srt` (subtitles from transcript) written for upload, not burned.
5. **Final encode**: `--master` DEFAULT tier `encoding.master_fast` (`h264_videotoolbox -b:v 24M -maxrate 30M -bufsize 60M -profile:v high -pix_fmt yuv420p -g 15 -bf 2`) — several times faster than x264 at some quality cost YouTube's own re-encode-on-ingest mostly absorbs, falling back to the x264 tier when the hardware encoder isn't available. `ytedit render --master --x264` instead uses `encoding.master`: `libx264 -profile:v high -level 4.2 -crf 18 -preset medium -pix_fmt yuv420p -g 15 -bf 2 -movflags +faststart -color_primaries bt709 -color_trc bt709 -colorspace bt709` (`preset medium`, not `slow` — the extra encode time bought little visible quality even before the hardware tier existed); reach for it only for a final upload where the extra quality margin matters. `aac 384k 48 kHz stereo` either way. `--preview`: 720p, videotoolbox, single-pass loudnorm, full-mezzanine picture. `--draft`: same 720p canvas as preview but proxy-sourced picture, targeting ~1 MB/10s (`encoding.draft`: hardware `-b:v 900k`, falling back to libx264 crf 30 ultrafast; audio drops to `audio.draft_abitrate`, default 128k) — `renders/draft.mp4`, meant for a first full-timeline review, not for watching quality.
6. Progress via `-progress pipe:1` parsed into `jobs/<job>.json` (percent, eta, current step); the segment pass logs `segment i/N done` as each one finishes, and every pass logs its own elapsed time (handy for comparing a draft's speed against a preview's).
7. **Cache hygiene** (`ytedit clean <slug>`, `media/render.clean()`): with no flags removes both the always-regenerated `renders/` intermediates (`program_video.mp4`, `program_audio*.wav`, `mix.wav`, `final_audio*.wav`, `concat.txt`, `duck.cmd`) and any `renders/segments/*.mp4` / `renders/segments_draft/*.mp4` not among the cache keys the current `plan/timeline.json` would produce at its preview/master/draft canvas (`--segments`/`--intermediates` restrict the sweep to one or the other); never touches `media/`, `input/` or `exports/`. Reports bytes freed.
8. **Timecode inspector** (`ytedit at <slug> <mm:ss> [--around N] [--json]`, `ytedit/inspect.py`): given one or more render-time timecodes (the position in the actual rendered file — `Timeline.segment_positions(fade_overlaps=True)`, what `render_positions()` uses), resolves the video segment on screen (id/uid/clip/in-out/role/mute/`audio_from`), maps that instant back into the audio clip's own time base to report the nearby transcript words / sentence catalogue ids (or the voice pickup file name if one is playing there, or "music only"/"silence" otherwise), and the active captions/music cue/chapter (captions/voice/music/chapters are stored in *timeline* time and mapped forward through `build_time_map()` for the comparison). `--around N` additionally lists every segment within `±N` seconds. Read-only — no ffmpeg encoding, at most a cheap `ffprobe` for an un-ended voice pickup's length. This is how a timecoded note from the user ("at 4:27 the sentence is cut") becomes a precise segment/sentence reference instead of a guess.
9. **Render verification** (`ytedit check-render <slug> [--render draft|preview|master|<path>] [--force] [--json]`, `ytedit/verify.py`): transcribes the *finished file's* audio with ElevenLabs Scribe (mono 16 kHz extracted into `renders/check/<name>.wav`; the transcript is cached as `renders/check/<name>.transcript.json` against the render's mtime+size, so re-running is free) and lines it up against the timeline. A boundary is any pair of consecutive video segments whose `audio_source` is not the same clip continuing within one frame (a continuous hand-off — the pieces of one beat around a shot, carrying the narration under the insert — is not a cut and is skipped), plus every `tracks.voice` pickup's start and end. Every boundary is labelled with the cut beats on both sides (`b012 (own picture) -> b013 (shot 1)`), because a correction is applied to `plan/cut.json` and the timeline's own segment ids are renumbered on every resolve. For each it reports the source-side margins (`out - last word end`, `first word start - in`, from the clip transcripts), whether `in`/`out` land inside a word, whether both sides replay the same stretch of the same clip, and — on the render side — the gap between the last transcribed word before the boundary and the first after it plus any word the cut chops in half. Flags: `no air` (< 0.35 s), `chopped word`, `out/in inside word`, `replayed`. Writes `renders/check/<name>.report.md` (+ `.json` with `--json`) and always exits 0 — it is a report, not a gate (`ytedit qc` is the gate). This is what catches a plan that is internally consistent but sounds wrong, e.g. a narration pickup ending inside a word.

---

## AI layer conventions

- `ai/openrouter.py`: `chat(model, messages, json_schema=None, images=[], temperature=0.2, max_tokens=...)` → parsed JSON (`response_format` json_object + robust extraction fallback from amazonia-studio), retries with backoff on 429/5xx, logs cost from `usage` to the project ledger. Model roles in `defaults.yaml`: `models.planner: anthropic/claude-opus-5`, `models.analyst: anthropic/claude-sonnet-5`, `models.vision: google/gemini-3.8-flash`, `models.writer: anthropic/claude-sonnet-5`.
- Every prompt states: project language, that instructions spoken to the editor at clip start must be extracted, that the LAST take usually wins, and returns strict JSON validated with pydantic; on validation failure retry once with the error appended.
- `ai/elevenlabs.py`: thin httpx client (no SDK dependency required, but SDK allowed): `transcribe(wav) -> dict`, `compose_music(prompt, ms, mode) -> bytes`, `tts(text, voice_id, model) -> bytes`, `clone_voice(name, files) -> voice_id`, `isolate(wav) -> bytes`, `sfx(text, seconds) -> bytes`.
- `ai/fal.py`: `upload(path)`, `run(app, args)` (subscribe with logs), `submit/status/result`, `thumbnail_edit(prompt, image_paths, n)`, `seedance_i2v(image, prompt, seconds)`, `topaz_upscale(video)`.

---

## Web editor (`server/`)

FastAPI on `http://localhost:8765`. Pages: project list → project page (clip grid with status, footage log, plan) → timeline editor.

> **Phase 4 — not yet implemented, and `ytedit serve` does not start.** `server/app.py` still imports the v1 timeline edit API (`ensure_segment_uids`, the anchor helpers) that cut v2 deleted, so the web editor fails at import until it is ported. The beat-based UI described below is the target, not the current state; until then the cut is edited through the CLI stages or by hand in `plan/cut.json` (`ytedit validate` then `ytedit resolve` after every hand edit).

Views: **Clips** (source review: transcript, waveform, mute tool, denoise buttons), **Program** (default when a cut exists), **Advanced** (raw lists), Plan / Footage log / QC / Publish tabs.

**The Program view edits beats, not seconds.** A beat strip with frames; a speech beat shows its sentences (drop the first or last sentence, split the beat at a sentence, switch to `words` and drag a word boundary), its shots (add from the clip grid, trim in seconds, move `after`) and `on_camera`; a broll beat has in/out nudges as today; a voice beat has its shots and the WAV; beats are dragged to reorder; music cues, captions, chapters and markers pick a beat rather than a time. Save → `validate` → `resolve` → `plan/timeline.json` → the existing player/preview. "Tidy cuts", "Accept draft", the anchor retrofit and split-by-frame on a speech segment are gone — there is nothing left for them to do.

Original v1 feature list (still valid, lives under Clips/Advanced):
- Video player (proxy) synced with wavesurfer waveform + regions: transcript words rendered under the waveform, takes highlighted, detected music regions shaded.
- Clip list in order with kind/role badges; drag to reorder segments; set in/out; toggle include; choose transform for vertical clips.
- **Mute/duck tool**: select a range on the waveform → set gain (mute / −12 dB / custom) → saved to `mute_ranges`.
- Captions list (edit text/time/style); music cues list (choose track, gain, duck amount); markers.
- Buttons: Transcribe / Analyze / Plan / Generate music / Render preview / Render master / QC / Publish pack; job progress bar; "what to do next" hint (state machine like amazonia-studio's `renderActions`).
- Save writes `plan/cut.json` (with `meta.edited_by_human: true`), backs the previous version up into `plan/history/`, and resolves the derived `plan/timeline.json`. A human-edited cut is never overwritten by a new plan unless asked: `ytedit plan` then writes `plan/cut.draft.json` for diffing (`--force` to overwrite).

---

## Testing

- `tests/fixtures/make_fixtures.py`: generates synthetic clips with ffmpeg lavfi (testsrc2 + sine, vertical and horizontal, HLG-tagged variant, VFR variant) and, when keys are present, a Polish narration WAV via ElevenLabs TTS muxed into a clip so transcribe/analyze can be tested end-to-end for cents.
- Unit tests for: probe parsing, timeline validation, duck automation file generation, ASS generation (Polish glyph check), loudnorm parsing, render of a 3-segment timeline (no API).

---

## Decisions

Where the cut v2 contract was silent or self-contradictory, these are the calls that were made. They are recorded because the code follows them and re-deriving them from first principles is how the same argument gets had twice.

### Phase 1 (models, validator, resolver, timeline v2)

* **Air deficit sign.** The original formula for the deficit was inverted. Net effect of the correction: the picture always gets the full `speech_pad_before`/`speech_pad_after` (clip-clamped) and only the audio window is pulled off a neighbouring word.
* **Validation *is* resolution.** `voice_picture_short`, `air_short`, `short_piece` and `shot_trimmed` only exist once a beat has been laid out, so `validate()` runs the resolver and keeps its issues; `resolve()` runs the same pass and raises `CutError` if any issue is an error. Nothing is computed twice and the two can never disagree.
* **Word claims cover sentence beats too.** A `sentences` beat also claims the transcript word indices its range covers, so a `words` beat overlapping a `sentences` beat is caught by the same `words_reused` rule (the contract only promised "another beat's words").
* **`air_short` is checked at speech→speech adjacency only.** Anything else between two takes (B-roll, a pickup) *is* the air. The measured air is the picture air (`seg.out − last.e` plus the next beat's `first.s − seg.in`), since what the picture extension adds renders as silence.
* **The min-shot rule also applies off camera.** If the uncovered tail of an `on_camera: false` beat is shorter than `min_shot_seconds`, the last shot is extended over it (clip permitting) instead of flashing the narrator for a quarter of a second; only a tail that survives that warns `narrator_visible`. An own-picture piece that stays shorter than `min_shot_seconds` warns `short_piece`.
* **`after` on an off-camera shot** is meaningless (shots chain from the beat start there) and is ignored with a `shot_after_ignored` warning.
* **References on disk.** `load_cut` accepts a display id *or* a uid for `music.from`/`to` and `captions`/`chapters`/`markers` `beat`, and rewrites it to the uid; `save_cut` renumbers the display ids and then writes **uids** for every reference. Writing ids back would silently re-point a caption at a different beat as soon as anyone deleted a beat by hand. `save_cut` writes with `exclude_none` and the models' declaration order, so saving an unchanged cut is byte-stable.
* **A music cue that lost one end** snaps to the first beat (lost `from`) or the last beat (lost `to`) — the cut no longer holds any position information about the beat that vanished. `music_beat_snapped` warns; only losing *both* ends is an error.
* **Segment uids are deterministic** (`sha1(beat uid + index)`), not random: the uid is part of the render's segment cache key, so a re-resolve must not invalidate every cached segment.
* **`speech_ranges_from_transcripts`** (the music duck) honours `audio_window`, as does `segment_speech_ranges` (the per-cut speech levelling) — a word that is never heard must not duck the music either.
* **The voice probe sidecar** is `voice/<file>.wav.probe.json`, matching the existing `<file>.loudness.json` convention, and is keyed by mtime. `ytedit.cut.probe_voice_duration` is the seam tests monkeypatch.
* **`meta.source`** falls back to the current time when `plan/cut.json` does not exist on disk (a cut resolved straight from memory).

### Phase 2 (migration)

The twelve decisions taken while implementing `ytedit/migrate.py`, each with the issue code its report uses, are listed in full under [Migration (`ytedit migrate`)](#migration-ytedit-migrate) above rather than repeated here.

### Phase 3 (producers, QC, tooling)

* **Speech in seconds is a planner error, not a fallback.** A planner segment that carries no `sentences`, is not `mute_source`, and whose `in`/`out` range covers transcript words cannot be made into a beat without guessing where a word starts — which is the whole thing v2 exists to stop. `build_cut` raises `SpeechSecondsError`, and `plan` shows the model exactly which segments broke the rule and asks once more (the AI-layer retry convention); a second failure is a prompt problem and surfaces as a `PlanError`. There is no raw-seconds speech path left.
* **A non-contiguous sentence run is split, not rejected.** The resolver errors on a beat whose ids skip an `n`, so `build_cut` splits one planner segment into one beat per contiguous run instead of handing the user a plan that does not resolve. `transition_in` rides on the first beat of the group only: a segment split by a gap (or by an excised instruction range) is still one editorial move on screen.
* **An `instruction` sentence reference is dropped; a `retake_of`/`duplicate_of` one is kept.** A spoken "wytnij to" must never reach the cut, so it is dropped outright. The other two flags are heuristic — a short "Zobaczcie." or a chorus trips them — so overriding the catalogue is the editor's call; the id stays in the beat and is listed in `edit_plan.md`'s script check as `flagged_sentence_refs`. This matches migration decision 9.
* **`voice_over` flips the whole segment off camera and its `picture[]` become the *first* beat's shots.** Picture cuts carry no sentence to hang the rest off, so a voice-over segment that splits into several runs puts all its picture on the opening beat and reports `voice_over_split`; the planner should not be producing that shape in the first place.
* **The planner's absolute seconds are only ever *read* as positions.** Captions, chapters, markers and music cues still arrive in seconds because that is what the model can see; `build_cut` resolves the cut once (`beat_spans`) and attaches each of them to the beat playing at that second. Nothing keeps the second itself — a beat is the only position that survives a re-cut.
* **A pickup absorbs whole broll beats, never half of one.** `ytedit voice` moves the `broll` beats following the target beat into the new `voice` beat's `shots` until they cover the WAV; the resolver trims the last shot to the pickup's length (and stretches it when the clip has room), so nothing here cuts a beat in half. Moving rather than copying is what keeps the programme's length honest — the same footage plays once, now under narration. With no broll to take, the beat gets `shots: []`, which is the `voice_picture_short` error, and the report tells the user to pick the picture in the editor rather than the stage inventing one.
* **A pickup's `audio` mirrors the picture it took over.** The voice beat is set to `ambient` only when *every* absorbed beat was heard; otherwise the field is left unset (silent by default). A pickup recorded at home has no business resurrecting a muted car radio, and no business silencing a street party the cut deliberately kept.
* **Re-running `voice` is idempotent and never drags a pickup across the cut.** A `voice` beat that already carries the same `file` is refreshed in place — same position, same shots, hand-set `gain_db` intact — instead of a second one being inserted and more B-roll swallowed. When that position is not after the manifest's target any more, the report says so and asks the user to delete the beat in the editor: silently moving it would strand the picture it already owns.
* **QC's rules 31–36 became the validator's output, prefixed `cut:`.** `ytedit qc` calls `ensure_resolved` first (so it never judges a stale timeline), then runs `ytedit.cut.validate` and maps its issues one-to-one — errors to QC errors, warnings to QC warnings. A project with no `cut.json` is a QC error in itself: the cut is the source of truth since v2. Everything downstream of the resolve (loudness, captions, pacing, music separation, the rendered file) is unchanged.
* **A cut that does not resolve stops QC at the cut.** `ensure_resolved` deliberately leaves `plan/timeline.json` stale when the cut has errors, so the timeline and render rules would be judging a file the cut no longer describes — worse than not judging it. QC reports the cut errors, warns that the rest was skipped, and still writes both report files; it only *raises* when the project has neither a cut nor a timeline. This is why `check_cut` returns whether the cut resolved rather than just accumulating findings.
* **`ytedit at` and `check-render` answer in beats.** `at` prints the beat (id, kind, clip/file, its sentences, and which of its shots is on screen) above the segment block, and its window search for what is *heard* around the instant is now called `sentences_heard` so the two cannot be confused. `check-render` labels every boundary with the beats on both sides (`b012 (own picture) -> b013 (shot 1)`), because the timeline's segment ids are renumbered on every resolve and a correction is applied to the cut.
* **`ytedit music` sizes a bed from its cue's beat range** (`beat_spans` start of `from` → end of `to`). A cue whose beats no longer resolve is skipped with a warning rather than generated at a guessed length: an ElevenLabs generation is paid for by the minute and a bed of the wrong length is worse than no bed. The musical direction (style/mood/section/prompt) rides on the cut's cue as extra fields, falling back to the same-id cue in `plan/edit_plan.json` for a cut produced by `ytedit migrate`, which keeps only file/gain/fades.
* **Tests state timelines by hand where the renderer is what is under test.** `tests/test_render.py` writes `plan/timeline.json` directly — fractional in/out points, an `xfade` overlap and an `audio_window` are far more directly stated as a timeline than derived from a cut — and writes a stand-in `plan/cut.json` (one `broll` beat per segment) *before* it, so the cut-level checks have something to run on and `ensure_resolved` never replaces the hand-written file. The cut→timeline direction is covered where it belongs, in `tests/test_cut.py` and the producers' own tests. Such a fixture must spell out each segment's `uid`: `VideoSegment.uid` defaults to a *random* value, which is invisible in a resolved timeline (the resolver derives it) but means a hand-written file keys differently on every `Timeline.load` — and the uid is part of the render's per-segment cache key.
* **`duplicate_of` needs at least four distinct normalized words on both sides.** The flag is a global, cross-clip last-take rule, and short phrases ("No dobra.", "Zobaczcie to.") collide constantly; below that length a Jaccard match says nothing about intent. This is why the flag warns rather than errors in the resolver.
