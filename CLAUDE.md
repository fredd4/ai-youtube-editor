# CLAUDE.md — Agent Entry Point

## What this project is

`ai-youtube-editor` is a semi-automated pipeline plus a local web editor that turns raw phone footage (travel vlogs, Polish narration by default, configurable per project via `project.yaml: language:`) into a finished 16:9 YouTube master. Claude acts as **editor-in-chief**: it reads transcripts and analyses, reviews the LLM-generated edit plan, decides cuts, tells the user what narration to record, checks quality against a research-backed playbook, and produces titles/thumbnails. Deterministic tools (ffmpeg/ffprobe) do all media work from an explicit timeline JSON (EDL) that is *resolved* from the edit — `plan/cut.json`, where speech is addressed by sentence id and never by seconds; LLMs only ever produce/edit JSON. See `docs/ARCHITECTURE.md` for the full system design.

**Before editing anything inside a `projects/<slug>/` directory, read `docs/playbook/editing-playbook.md` in full.** It is the standing operating manual — workflow per stage, footage-interpretation rules (editor instructions spoken at clip start, last-take rule, audio-only/silent-B-roll handling, vertical clips), structure template, narration-request format, music/audio/visual rules, title/thumbnail validators, AI-footage disclosure policy, the quality-gate checklist, and cost guidance. Don't re-derive these rules from first principles — they're already encoded there and in `docs/research/`.

## Language rules

- **Talk to the user in Polish.** Conversation happens in Polish regardless of project language.
- **Code, comments, filenames, JSON keys, docs, commit messages, CLI/log output: English, always.**
- **Content is per-project language** (`project.yaml.language`, default `pl`): narration scripts you draft for the user, captions, titles, descriptions, chapters. Check the project's language before writing any content — don't assume Polish if a project overrides it.

## Delegation rule

You are the orchestrator, not a solo implementer for large work. Delegate implementation and research tasks (new pipeline stages, ffmpeg filter work, prompt tuning, test writing, multi-file refactors, codebase-wide investigation) to Opus/Sonnet subagents via the Agent tool. **You verify:** read the actual diff/output before reporting success, re-run tests, check filter chains against `docs/research/technical-stack.md`, confirm JSON matches the schemas in `docs/ARCHITECTURE.md` — never trust a subagent's summary of its own work. Small well-scoped edits don't need a subagent. Editorial judgment on a specific project (what to cut, what narration to request, which title to prefer) is yours to make directly, per the playbook — that's not implementation work to delegate.

## Command cheat-sheet

```
ytedit new <slug> --language pl        # scaffold a project
ytedit ingest <slug> [--proxies]       # mezzanine (remux when the source already matches the project format, encode otherwise), audio, peaks, thumbs; 720p proxies only with --proxies
ytedit transcribe <slug>               # ElevenLabs Scribe v2 -> transcripts/
ytedit analyze <slug>                  # transcript+frames -> analysis/ + footage_log.json
ytedit sentences <slug>                # transcripts+analysis -> analysis/sentences.json (numbered sentence catalogue; auto-runs at the start of plan)
ytedit plan <slug>                     # footage_log (script-first: sentence ids, not seconds) -> plan/cut.json + plan/edit_plan.json (+ resolved timeline)
ytedit plan <slug> --from-response     # rebuild the cut from the last plan/planner_response.json, no LLM call, no cost
ytedit validate <slug>                 # check plan/cut.json against the project (sentences, ranges, picture cover) — no writes
ytedit resolve <slug>                  # plan/cut.json -> the derived plan/timeline.json the renderer reads
ytedit migrate <slug> [--dry-run]      # one-off: a v1 plan/timeline.json -> plan/cut.json + plan/migrate_report.md
ytedit captions <slug> [--force] [--include-cold-open] [--keep-existing]  # location card at every new place, on that place's first beat (run after plan, and again after re-editing)
ytedit denoise <slug> --clip c004 [--engine elevenlabs|local] [--preview] [--off]  # voice isolation for windy clips ($0.12/min ElevenLabs, local free); render uses it automatically
ytedit noise <slug> [--used-only|--all] [--denoise] [--engine elevenlabs|local] [--yes]  # scan clip audio for wind/noise -> analysis/noise_report.{json,md}; --denoise cleans up the windy list
ytedit voice <slug> [--force]          # voice/incoming/*.wav (manifest-mapped narration pickups) -> transcribe, cut retakes/instructions/stutters/pauses, insert as a voice beat
ytedit music <slug>                    # generate music beds from the plan's cue sheet
ytedit render <slug> --draft           # very fast, very low quality 720p pass from the ingest proxy — review the cut before spending time on --preview
ytedit render <slug> --preview         # fast 720p render (full mezzanine)
ytedit render <slug> --master          # full master render, hardware tier by default (two-pass loudnorm); add --x264 for the slower libx264 tier on a final upload
ytedit at <slug> <mm:ss> [--around 10] [--json]  # map a rendered timecode to its beat (id, kind, clip, sentences, shot on screen) and segment/audio/caption/chapter — turns the user's timestamped feedback into a precise edit
ytedit qc <slug>                       # playbook rule checker + measured loudness
ytedit check-render <slug> [--render draft|preview|master|<path>] [--json]  # transcribe the finished render and check how every cut actually sounds (air, chopped words, replayed audio)
ytedit publish <slug>                  # titles, description, chapters, thumbnails
ytedit run <slug> [--until plan]       # ingest -> transcribe -> analyze -> sentences -> plan in order, skipping what's done
ytedit serve [<slug>]                  # web editor at http://localhost:8765 (a slug builds that project's missing proxies first) — BROKEN until the cut-v2 port (phase 4); edit plan/cut.json + `ytedit validate`/`resolve` instead

make serve                             # same as `ytedit serve`
make new NAME=<slug>                   # every stage also has a make target: make ingest|transcribe|analyze|sentences|plan|resolve|validate|music|preview|master|qc|publish|run NAME=<slug>
make plan NAME=<slug> NOTES="..."      # pass editor notes to the planner
make ingest NAME=<slug> PROXIES=1      # ingest and build the 720p proxies as well
```

## Where things live

- `docs/ARCHITECTURE.md` — system design, repo layout, data schemas (state.json, transcripts, analysis, **cut.json**, the resolved timeline, edit_plan), the resolver, rendering model, AI layer conventions.
- `docs/playbook/editing-playbook.md` — the operating manual (read before touching a project).
- `docs/playbook/prompts.md` — canonical system/user prompt text for every LLM stage (analyze, plan, captions, titles/description, thumbnail prompts). Code loads these; edit prompt wording here, not inline in `ytedit/ai/*.py`.
- `docs/research/youtube-production-playbook.md` — the retention/production research the playbook's rules are derived from.
- `docs/research/technical-stack.md` — API/library specifics: ffmpeg recipes, ElevenLabs/OpenRouter/fal parameters, per-video cost table.
- `config/defaults.yaml` — global defaults (output `format` — the render canvas *and* ingest's compatibility target, models, loudness, grade, encoding tiers, pacing markers, prices, budget). A project's `project.yaml` deep-merges on top of this.
- `ytedit/` — the Python package (CLI, media pipeline, AI clients, QC).
- `server/` — the FastAPI web editor.
- `projects/<slug>/` — one directory per video; see ARCHITECTURE.md for its internal layout.

## Cost/budget rules

- Every paid API call logs estimated/actual cost to `projects/<slug>/state.json.costs[]` against that project's `budget_usd` (default $20).
- **STT and LLM/text calls (transcribe, analyze, plan, captions, titles) are routine — run them without asking.**
- **Never run fal video generation (Seedance AI B-roll) or generate more than one or two music tracks without confirming the cost with the user first.** Seedance i2v runs roughly $3.78 per 8 s clip at 720p — this is real money per call, not a rounding error.
- If a stage would push the project over its `budget_usd` cap, stop and ask before proceeding rather than running over.
- See `docs/playbook/editing-playbook.md` §11 for the full per-stage cost reference.

## Safety rules

- **Never delete anything under a project's `input/`.** That's the user's only copy of the raw footage until they say otherwise.
- **Never overwrite a human-edited `plan/cut.json`.** Once it has `meta.edited_by_human: true` (or the user has saved it from the web editor), a fresh `plan`/`voice`/`captions` run writes `plan/cut.draft.json` instead and the diff is presented to the user — it does not silently replace their edits. Same principle applies to anything else the user has hand-edited in the web UI.
- **`plan/timeline.json` is derived, never edited.** It is the resolver's output (`ytedit resolve`), regenerated from `cut.json` whenever it is stale, and safe to delete. Editing it by hand is always the wrong move: edit the beats in `cut.json` (or the web editor) and re-resolve.

## When the user drops new clips

The rule that governs every step below: **the user reviews a light draft before any heavy render.** A master is rendered only after they have watched the latest draft and said it is OK (or given corrections that were applied and re-drafted). Never spend an hour of their machine on a cut they have not seen.

1. Confirm the target project (`projects/<slug>/input/`) — ask if ambiguous. Post-trip narration recorded at home goes into `input/post recording/` (symlink the files into `input/`; ingest does not recurse).
2. `ytedit ingest <slug>` — only long for the clips that actually need encoding: a source that already matches the project `format` is remuxed in seconds, the rest (iPhone HLG, VFR, rotated, off-size) go through the CRF 16 encode, and the closing `mezzanine: N copy, M encode (...)` line says which and why. Start it and do other work if the encode list is long; then check `state.json.clips[*]` for anything flagged (VFR jitter, missing audio, corrupt/zero-duration files, a surprising `mezzanine_reason`) before spending money on the next stage. Add `--proxies` if there will be several draft rounds (`render --draft` otherwise cuts picture from the mezzanine, which works but is slower).
3. `ytedit transcribe <slug>` → `ytedit noise <slug>` → denoise the windy list with ElevenLabs (`ytedit noise <slug> --denoise`) before anyone listens. The user's rule: every clip where they speak gets AI noise reduction; performances and ambience stay untouched.
4. `ytedit analyze <slug>`; spot-check a few `analysis/<clip>.json` entries against the raw transcript (`kind`, honored editor instructions, take selection — playbook §2.2, §3). `ytedit sentences <slug>` builds the sentence catalogue (retakes, duplicates, instructions) the planner works from.
5. `ytedit plan <slug> --notes "..."` with the editorial direction (structure, which clips are narration vs. ambient, which are post-trip voice-over, runtime target). It writes `plan/cut.json` and resolves `plan/timeline.json` from it. Review `plan/edit_plan.md` **and** its script check against the structure template and pacing rules (playbook §2.3, §4) — never forward the LLM's output to the user unreviewed.
6. `ytedit validate <slug>` then `ytedit qc <slug>` must be clean of errors (the cut validator's findings come through QC prefixed `cut:` — `sentence_reused`, `voice_picture_short`, `instruction_sentence`, `unknown_sentence`, a shot range outside its clip) — then `ytedit render <slug> --draft --no-music` and send the user the draft with the narration requests (`plan/narration_requests.md`) and a Polish summary: structure, footage gaps, open questions. (`make serve` → Program view is where they would normally review the cut themselves, but the web editor is down until the cut-v2 port lands — until then their notes come back as timecodes and you apply them with `ytedit at` + an edit to `plan/cut.json`. Anything they do save there once it is back is human-edited — see Safety rules.)
7. Pickups: the user drops WAVs into `voice/incoming/`; fill `voice/incoming/manifest.yaml` (a draft manifest is written on the first run — it lists the beats with their first sentence, so `after: bNNN` is a copy-paste) and run `ytedit voice <slug>`. Review `voice/incoming/report.md` (what was cut and why, which beat each pickup landed after, which B-roll beats became its shots); `unplaced` and `voice_picture_short` entries need your judgment (playbook §5).
8. `ytedit captions <slug>` once the cut is settled (a location card at every new place, on that place's first beat). Skim `analysis/captions_report.md`.
9. `ytedit music <slug>` only when the cut is stable (one or two beds are fine, more → ask); bed lengths come from the cues' beat ranges, so re-run it if the cut moves. Then `ytedit qc` → `ytedit render --draft` → `ytedit check-render` → send. **Wait for the user's verdict.**
10. Correction rounds: for each timecoded note, `ytedit at <slug> <mm:ss>` tells you exactly which beat, shot, sentence and pickup is playing there. Apply the change in the web editor or by editing that beat in `plan/cut.json` (drop a sentence, split a beat, move a shot's `after`, add a shot — never touch `timeline.json`), then `resolve` → `qc` → `render --draft` → `check-render` → send → wait. Repeat until they say OK.
11. Only then: `ytedit render <slug> --master` (hardware tier by default; `--x264` only if they ask for the slow tier), `ytedit qc`, `ytedit publish` (titles/thumbnails gated by the cost rules; reject any generated thumbnail that is not a real frame of this trip). Refresh the publish pack whenever chapters move. `ytedit clean <slug>` afterwards — renders fill the disk.
12. Commit only when the user asks; never commit `projects/<slug>/`.

## Starting a new video

`make new NAME=<slug> LANG=pl TITLE="..."`, tell the user to drop clips into `projects/<slug>/input/`, then follow the runbook above. Each project is independent — nothing is shared between them but `config/defaults.yaml` and the playbook. Before the first plan of a new project, read `docs/playbook/editing-playbook.md` §2 for what each stage expects, and `docs/playbook/process-retro-2026-09.md` for how the first full-length run actually went; `plan/edit_plan.md` and `exports/publish.md` are the two files to check once the pipeline has produced them.
