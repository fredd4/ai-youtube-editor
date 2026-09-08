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
│   │   ├── sentences.py      # transcripts+analysis → analysis/sentences.json (numbered sentence catalogue; script-first planning)
│   │   ├── plan.py           # all analyses → plan/edit_plan.json (timeline draft, captions, music cues, narration asks, titles)
│   │   ├── voice.py          # voice/incoming/*.wav (manifest-mapped) → cleaned voice/*.wav + placed on timeline.json
│   │   ├── locations.py      # footage_log locations → analysis/places.json (canonical) → anchored location captions
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
        ├── analysis/         # <clip>.json per clip + footage_log.json (merged, chronological) + sentences.json/.md + places.json + captions_report.md
        ├── plan/             # edit_plan.json (LLM draft), timeline.json (edited, source of truth for render)
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
| 4 | plan | `ytedit plan <slug>` | footage_log (compacted with the sentence catalogue) → plan/edit_plan.json + plan/timeline.json (draft) + narration_requests.md | OpenRouter (Opus) |
| 4b | tidy | `ytedit tidy <slug>` | timeline.json → padded cuts (~0.3 s before / 0.45 s after speech); auto at the end of plan | deterministic (`ytedit/ai/tidy.py`) |
| 1b | denoise | `ytedit denoise <slug> --clip cNNN` | media/audio/<clip>.wav → <clip>.denoised.wav; render uses it when `clips[id].use_denoised` | ElevenLabs Audio Isolation or local afftdn |
| 4c | voice | `ytedit voice <slug>` | voice/incoming/*.wav (manifest-mapped) → transcribe, cut retakes/instructions/stutters/pauses, place on timeline.json | ElevenLabs Scribe + deterministic (`ytedit/ai/voice.py`) |
| 4d | captions | `ytedit captions <slug>` | footage_log locations → analysis/places.json (canonical, one writer call, cached) → anchored location captions on timeline.json + analysis/captions_report.md | OpenRouter (writer) + deterministic (`ytedit/ai/locations.py`) |
| 5 | music | `ytedit music <slug>` | plan music cues → music/*.mp3 | ElevenLabs Music |
| 6 | render | `ytedit render <slug> --draft` / `--preview` / `--master` | timeline.json → renders/draft.mp4 (from proxy, ~1 MB/10s) / renders/preview.mp4 / exports/master.mp4 (hardware tier by default, `--x264` for the slow tier) | ffmpeg |
| 7 | qc | `ytedit qc <slug>` | timeline + master → qc_report.json/md | rules + loudnorm measure |
| 8 | publish | `ytedit publish <slug>` | → exports/publish.json (titles, description, chapters), thumbnails/ | OpenRouter + fal |
| — | at | `ytedit at <slug> <mm:ss>` | timeline.json → segment/audio/caption/chapter at that rendered timecode | deterministic (`ytedit/inspect.py`) |
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
`build_timeline` validates every `segments[].sentences` id against this catalogue (unknown/already-used/instruction/retake/duplicate ids are dropped with a stat, never silently kept) and derives the segment's `in`/`out` from the sentence boundaries plus `pacing.speech_pad_*`. A clip with no transcribed speech gets `{"id": "<clip>", "sentences": []}`. Before it reaches the planner prompt, `ytedit.ai.sentences.compact_footage_log_for_planner` replaces each clip's footage-log `segments[]`/`takes[]` with this sentence list (instruction sentences omitted outright; retake/duplicate ones kept but marked `"skip": "retake, use ..."` / `"skip": "duplicate, use ..."`) — measured on the 218-clip `projects/the reference project` footage log, this shrinks the compact-JSON prompt payload by about 5% even though it adds structured per-sentence metadata, because it replaces the old free-text segment/take detail for every clip that has transcribed speech.

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

### plan/timeline.json — the EDL (single source of truth for render)
```json
{
  "version": 1, "fps": 30, "width": 1920, "height": 1080, "language": "pl",
  "tracks": {
    "video": [
      {"id": "s001", "uid": "3e4669c0", "clip": "c003", "in": 12.0, "out": 16.5, "role": "cold-open",
       "transform": {"fit": "cover|contain|blur-fill|crop-pan", "zoom": 1.0}, "grade": "default",
       "transition_in": {"type": "cut|fade|xfade", "duration": 0.0}, "speed": 1.0, "mute_source": false,
       "source_audio_gain_db": 0, "notes": "drone reveal"}
    ],
    "voice": [
      {"id": "v001", "file": "voice/intro_pickup.wav", "at": 0.5, "gain_db": 0,
       "anchor": {"segment": "s003", "offset": 0.0, "uid": "004a3254", "signature": {"clip": "c003", "in": 12.0}}}
    ],
    "music": [
      {"id": "m001", "file": "music/arrival_warm.mp3", "at": 0.0, "end": 185.0, "gain_db": -18, "fade_in": 2.0, "fade_out": 3.0,
       "duck": {"mode": "auto", "amount_db": -12, "attack": 0.15, "release": 0.6}}
    ],
    "captions": [
      {"id": "t001", "at": 0.5, "end": 3.0, "text": "LIZBONA, PORTUGALIA", "style": "location", "position": "lower-left",
       "anchor": {"segment": "s001", "offset": 0.3, "uid": "3e4669c0", "signature": {"clip": "c003", "in": 12.0}}}
    ],
    "sfx": []
  },
  "mute_ranges": [{"clip": "c005", "s": 3.0, "e": 20.0, "gain_db": -60, "reason": "copyrighted bar music"}],
  "markers": [{"at": 0.0, "label": "hook"}, {"at": 7.0, "label": "promise"}, {"at": 180.0, "label": "re-engagement-1"}],
  "chapters": [{"at": 0, "title": "Przyjazd do Lizbony"}],
  "meta": {"title_candidates": [], "generated_by": "plan@2026-09-04", "edited_by_human": false, "anchor_issues": []}
}
```
Rules: times in seconds (float). Timeline time for a video segment = cumulative position; `render.py` computes absolute placement. Captions/music/voice use absolute timeline time. `mute_ranges` are in **clip** time and apply to source audio wherever that clip range is used.

**Stable segment identity (`VideoSegment.uid`):** every video segment carries `id` (the display label, `s001`/`s002`/... — renumbered by `plan.py` and `ytedit tidy` every time a segment is inserted, dropped or merged) and `uid` (8 hex chars from `secrets.token_hex(4)`, assigned once on creation and **never** reassigned or renumbered by anything). A file saved before `uid` existed gets one filled in deterministically from `(clip, in, out, index)` the first time it is loaded (`ensure_segment_uids`, called by `Timeline.load()` and `Timeline.model_validate_migrated()`) rather than a fresh random one on every read, so two loads of the same not-yet-migrated file agree on identity. This exists because an anchor that keys on the display id silently resolves to the wrong picture once ids shift under it — three segments inserted upstream used to mean every anchor after that point pointed three segments early, with nothing detecting it (see the process retro for the incident this fixed).

**Anchors key on `uid`, not `id`:** `VoiceAnchor`/`CaptionAnchor` carry `segment` (the display id, kept only for readability — logs, the web editor), `offset`, `uid` (the segment's stable identity at anchoring time) and `signature: {clip, in}` (a fingerprint of that segment, for repair). `Timeline.resolve_anchors()` looks the segment up by `uid` first; if the uid is missing (a legacy anchor) it falls back once to the display id, and if the uid doesn't resolve (or resolves to a segment whose `clip`/`in` no longer matches `signature` within 0.5 s — should not normally happen, but is the safety net for it) it searches every segment for a unique `signature` match and repairs the anchor, logging a WARNING that names the item. An anchor that cannot be resolved at all keeps its last absolute position and is recorded in `meta.anchor_issues` (cleared and rebuilt on every `resolve_anchors()` call) — see `ytedit qc` rule 34. Every anchor-creating call site (`plan.py`'s `_build_voice_items`, `voice.py`'s pickup placement, `locations.py`'s location/hook cards, the `ytedit voice-anchor` CLI command, and the web editor's own anchor retrofit in `PUT /timeline`) fills `uid` + `signature`, not just `segment`.

**The video-track edit API (`Timeline.insert_segments`/`remove_segments`/`replace_segment`):** the one supported way to grow, drop or split video segments outside of the initial `build_timeline` construction. Each assigns a `uid` to any new segment that doesn't have one, re-times everything downstream (unanchored voice items, captions, music cues, markers, chapters) by the frame-exact duration the edit adds or removes at that point — the same shift-map machinery `ytedit.ai.tidy.pad_segments_to_speech` uses for speech padding, now living in `ytedit/timeline.py` so both the tidy pass and the edit API share one implementation — renumbers every segment's display `id` (unless called with `renumber=False`, for a pass like `ytedit.ai.voice` that must keep its own numbering across the edit because something upstream already anchored against it), and re-resolves every anchor. `ytedit/ai/overlay.py` and `ytedit/ai/ledger.py` route their segment drops through `remove_segments`; `ytedit/ai/voice.py`'s pickup-growth pass routes its B-roll inserts through `insert_segments`. The web editor's split action creates the new piece without a `uid` so the server assigns a fresh one on save.

A video segment also accepts an optional `audio_from: {"clip", "in", "out"}` (an *overlay cutaway*): the picture stays `clip[in, out]` but the rendered audio is read from `audio_from.clip[audio_from.in, audio_from.out]` instead, trimmed or padded to the segment's own picture frame count, with the segment's own `speed`/`source_audio_gain_db` and the audio clip's mute ranges and denoised WAV. `mute_source: true` still wins and renders silence. `ytedit/ai/overlay.py` sets it so a cutaway dropped between two contiguous pieces of one take does not interrupt the narration underneath.

`plan.py`'s planner-facing segment schema (before deterministic post-processing) also accepts an optional `voice_over: {"picture": [{"clip", "in", "out"}]}` on a segment: `build_timeline` extracts that segment's own clip audio as a `tracks.voice` item (a WAV cut from `media/audio/<clip>.wav`, or the denoised variant when active, written to `voice/vo_<clip>_<in>_<out>.wav`) and replaces the segment with the listed picture cuts (`mute_source: true`, `role: "b-roll"`) so the narrator is heard but not seen, except where those cuts fall short of the narration's length — then the clip's own picture fills the gap.

A `tracks.voice` item also accepts an optional `anchor: {"segment", "offset"}`: the pickup is pinned `offset` seconds after the start of that video segment id instead of an absolute time, and `Timeline.resolve_anchors()` recomputes `at`/`end` from the segment's current position (keeping the pickup's length) every time padding, sentence snapping, overlay cutaways, audio dedupe or a hand edit in the web editor moves segments around — `plan.py` sets it on every voice-over pickup it creates (anchored to the first picture segment, offset 0) and `ytedit tidy`/`PUT /timeline`/`accept_draft` all re-resolve it before saving; `ytedit voice-anchor <slug> <voice_id> <segment_id>` sets it by hand. A dangling anchor (its `uid` no longer resolves, and no unique `signature` match can repair it) keeps the pickup's last absolute time, logs a warning, and is recorded in `meta.anchor_issues` instead of failing outright. (`Timeline.resolve_voice_anchors()` still exists as an alias, kept for callers written before captions could anchor too.)

A `tracks.captions` cue accepts the same `anchor: {"segment", "offset"}` (duration `end - at` kept fixed instead of a voice item's file length), resolved by the same `Timeline.resolve_anchors()` call. This is how `ytedit captions <slug>` (`ytedit/ai/locations.py`) fixes location cards drifting under the wrong picture: instead of a card pinned at an absolute second that a later pad/snap/overlay/dedupe pass silently invalidates, it is pinned to the video segment showing that place, offset `captions.card_offset_s` (default 0.3 s) into it, and always renders under the right shot regardless of how much the timeline has shifted since. The planner's own `hook` captions get the same anchor treatment (retrofitted to whichever segment sits under their current `at`) the first time `ytedit captions` runs; its `location` captions are dropped and replaced by the generated ones unless `--keep-existing`.

**Script-first speech (`ytedit/ai/sentences.py` + `plan.py`):** a speech segment sets `sentences: ["c030#1", "c030#2"]` — contiguous ids of one clip's sentence catalogue, in transcript order — instead of `in`/`out`; `build_timeline` derives `in = first_sentence.s - pacing.speech_pad_before` and `out = last_sentence.e + pacing.speech_pad_after`. Optional `cutaways: [{"clip", "in", "out", "after_sentence": "c030#1"}]` split the run at that sentence: the piece before it ends exactly at the sentence boundary (no trailing air — the narration keeps going), the cutaway gets `audio_from` over the stretch of the a-roll clip it would otherwise skip, and the next piece resumes exactly where that borrowed audio ends — the same on-disk shape `ytedit/ai/overlay.py` produces for the legacy (raw-seconds) pattern, just constructed directly instead of pattern-matched after the fact. Every sentence id may appear at most once in the whole plan; an id `build_timeline` cannot resolve (unknown, already used, an instruction, or a `retake_of`/`duplicate_of`) is dropped with a stat rather than silently rendered, and a non-contiguous id list is split into separate segments. A plan with no `sentences` on any segment (a legacy `planner_response.json`, or `plan --from-response` on one) behaves exactly as before — this is additive, not a breaking schema change.

### plan/edit_plan.json (LLM reasoning, human-readable)
Story outline (beats with target timecodes per playbook: 0:00 hook, 0:07 promise, 0:30 interrupt, 3:00/6:00 re-engagements, end payoff), which clips serve which beat, cold-open montage picks, subscribe-CTA placement, music cue sheet with mood per section, list of **narration requests** (what the user should record for intro/outro/bridges, with suggested scripts in the project language), risk flags (copyright music, missing footage), title candidates, thumbnail concepts. Written also as `plan/edit_plan.md`.

---

## Rendering model (`media/render.py`)

0. **Pre-flight** (`preflight()`, before any ffmpeg work): runs `Timeline.validate(project, skip_music=, skip_voice=)` — structural checks plus, for every video segment and every `audio_from` override, its range checked against that clip's known duration from the registry (`0 <= in < out <= duration + 0.05`) — and additionally confirms every referenced clip actually has a normalized source on disk (the one thing the JSON-only validator cannot see). `ytedit validate` runs the same `Timeline.validate` check, so a bad range is reported identically outside of a render. All problems are collected and reported at once as a numbered list; `render` refuses before touching ffmpeg rather than failing 20 minutes in on segment 179 of 190. `--no-music`/`--no-voice` skip the corresponding track's checks (a missing music file is not a reason to refuse a render that ignores music) and, symmetrically, drop that track from the mix without modifying the timeline file. A disk-space guard (`render.mb_per_second` × programme seconds × `render.disk_headroom_factor`, defaults 20 MB/s × 3x; `render.draft_mb_per_second`, default 1, for `--draft`) logs free space and refuses early when there isn't enough headroom for the segment cache + intermediates + export to coexist.
1. **Segment pass** (cached by hash of segment spec, including its frame count, canvas and mode): for each video segment cut from the normalized source: `-ss in`, then exactly `round((out−in)/speed × fps)` frames (`setpts=PTS-STARTPTS` + `-frames:v`, audio `atrim`/`apad` to the same number of samples), apply transform (fit for vertical, crop-pan), grade chain, scale/pad to canvas, `fps=30`, set source-audio gain / mute ranges. `--draft` cuts *picture* from the 720p ingest proxy (`media/proxies/<clip>.mp4`) instead of the mezzanine — using the proxy's own probed size, not the mezzanine's `state.json` width/height, since a vertical clip's proxy is a different shape — falling back to the mezzanine with a warning when a clip has no proxy yet; its *audio* always still comes from the mezzanine/denoised WAV, never the proxy's lossy 128k AAC, so a draft sounds exactly like the master. Immediately after that (still inside the segment's own audio chain, before `atrim`/`apad`), any cut whose audio carries transcript words is **speech-levelled**: `measure_speech_gain` (`ytedit/media/audio.py`) runs a `loudnorm print_format=json` analysis pass over the cut's own `-ss/-t` range — gated to just the word ranges via a `volume=enable=...` mute of everything else when they cover under 60% of the cut, measured whole otherwise — and the resulting gain (target `audio.speech_target_lufs`, default −16 LUFS, clamped to ±`audio.speech_gain_max_db`, default 10 dB) is folded into the same `volume=` filter used for the manual `source_audio_gain_db`. Ambient/B-roll cuts (no transcript) and muted cuts are left at their recorded level. The gain actually applied is written next to the cached segment as `<hash>.gain.json` (read back for the render log's speech-leveling table and the job's `speech_gains` list, on a cache hit too) and takes part in the segment's cache key (alongside `audio.speech_target_lufs`/`speech_gain_max_db` and the source clip's transcript mtime) so a re-transcribe or a changed target invalidates it. `Timeline.segment_positions()` does the same whole-frame arithmetic, so absolute placements (voice, captions, music, markers, chapters) match the rendered file to the frame regardless of how many fractional cuts precede them. Intermediate: `libx264 -crf 16 -preset fast` (`h264_videotoolbox` for previews, `-q:v 45`/libx264 ultrafast crf 26 for drafts) + `pcm_s16le` audio, 48 kHz stereo, cached in `renders/segments/` (`renders/segments_draft/` for `--draft` — its own namespace, so a draft render never invalidates or is invalidated by the preview/master cache). `render_segments()` runs this pass across a `ThreadPoolExecutor` sized by `render.workers` (default 3 — each ffmpeg process is already internally multi-threaded, so more than 3-4 rarely helps on an 8-core machine); a cache hit costs a stat + probe, never spawns ffmpeg (and never re-measures speech gain — the sidecar from the original render is reused), so extra workers are free on an already-rendered timeline. Segment order in the output list always matches the timeline regardless of completion order; a failure names its segment id precisely.
2. **Join pass**: concat demuxer for cuts (each entry carries a `duration` directive pinned to its frame count); `xfade`/`acrossfade` chained for transitions with frame-rounded offsets/overlaps (only when requested; default hard cuts).
3. **Audio bus**: `[program_audio]` = joined source audio (already speech-levelled per cut, see above) + voice track overlays, unless `--no-voice`. Each voice pickup is levelled to the same `audio.speech_target_lufs` target (measured once and cached alongside the file as `<file>.loudness.json`, keyed on its mtime and the target/clamp) before its own `gain_db` is added on top. Speech ranges for ducking come from transcripts mapped to timeline time (`duck.mode=auto`) or from an explicit list — for an `audio_from` overlay these already come from the borrowed clip, not the picture clip. Music gain automation via `sendcmd` file with ramps (attack/release), ducking `ducking.amount_db` (default −15 dB, chosen against a −16 LUFS levelled voice and a −18…−21 dB cue gain) → mixed with `amix=normalize=0` (skipped entirely with `--no-music`). Then voice cleanup (optional: `none`/`light`/`full`, see `audio.voice_cleanup`) — a gentle `acompressor` that runs *after* the leveling above, so it is consistency glue rather than gain-riding — then **loudnorm to −14 LUFS / −1 dBTP** on the final mix (two-pass for `--master`, single-pass for `--preview`/`--draft`). This whole pass is identical across all three tiers, so a `--draft` already sounds like the eventual master.
4. **Captions**: generated ASS (libass) with fonts checked for Polish glyphs, burned in the final pass via `ass=` filter. Separate `captions.srt` (subtitles from transcript) written for upload, not burned.
5. **Final encode**: `--master` DEFAULT tier `encoding.master_fast` (`h264_videotoolbox -b:v 24M -maxrate 30M -bufsize 60M -profile:v high -pix_fmt yuv420p -g 15 -bf 2`) — several times faster than x264 at some quality cost YouTube's own re-encode-on-ingest mostly absorbs, falling back to the x264 tier when the hardware encoder isn't available. `ytedit render --master --x264` instead uses `encoding.master`: `libx264 -profile:v high -level 4.2 -crf 18 -preset medium -pix_fmt yuv420p -g 15 -bf 2 -movflags +faststart -color_primaries bt709 -color_trc bt709 -colorspace bt709` (`preset medium`, not `slow` — the extra encode time bought little visible quality even before the hardware tier existed); reach for it only for a final upload where the extra quality margin matters. `aac 384k 48 kHz stereo` either way. `--preview`: 720p, videotoolbox, single-pass loudnorm, full-mezzanine picture. `--draft`: same 720p canvas as preview but proxy-sourced picture, targeting ~1 MB/10s (`encoding.draft`: hardware `-b:v 900k`, falling back to libx264 crf 30 ultrafast; audio drops to `audio.draft_abitrate`, default 128k) — `renders/draft.mp4`, meant for a first full-timeline review, not for watching quality.
6. Progress via `-progress pipe:1` parsed into `jobs/<job>.json` (percent, eta, current step); the segment pass logs `segment i/N done` as each one finishes, and every pass logs its own elapsed time (handy for comparing a draft's speed against a preview's).
7. **Cache hygiene** (`ytedit clean <slug>`, `media/render.clean()`): with no flags removes both the always-regenerated `renders/` intermediates (`program_video.mp4`, `program_audio*.wav`, `mix.wav`, `final_audio*.wav`, `concat.txt`, `duck.cmd`) and any `renders/segments/*.mp4` / `renders/segments_draft/*.mp4` not among the cache keys the current `plan/timeline.json` would produce at its preview/master/draft canvas (`--segments`/`--intermediates` restrict the sweep to one or the other); never touches `media/`, `input/` or `exports/`. Reports bytes freed.
8. **Timecode inspector** (`ytedit at <slug> <mm:ss> [--around N] [--json]`, `ytedit/inspect.py`): given one or more render-time timecodes (the position in the actual rendered file — `Timeline.segment_positions(fade_overlaps=True)`, what `render_positions()` uses), resolves the video segment on screen (id/uid/clip/in-out/role/mute/`audio_from`), maps that instant back into the audio clip's own time base to report the nearby transcript words / sentence catalogue ids (or the voice pickup file name if one is playing there, or "music only"/"silence" otherwise), and the active captions/music cue/chapter (captions/voice/music/chapters are stored in *timeline* time and mapped forward through `build_time_map()` for the comparison). `--around N` additionally lists every segment within `±N` seconds. Read-only — no ffmpeg encoding, at most a cheap `ffprobe` for an un-ended voice pickup's length. This is how a timecoded note from the user ("at 4:27 the sentence is cut") becomes a precise segment/sentence reference instead of a guess.

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
