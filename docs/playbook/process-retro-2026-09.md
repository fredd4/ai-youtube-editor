# Process retrospective — first real project (the reference project, September 2026)

The first end-to-end run on real footage (218 clips, 76 min, Polish narration, 8 post-trip
narration clips, 8 recorded pickups) produced a publishable 17-minute master, but only after
three master renders and two rounds of corrections from the user. This note records what went
wrong, why, what was changed in the pipeline as a result, and what is still worth changing.
It complements `editing-playbook.md` (the operating manual) — read it when planning the next
project, not during one.

## 1. What went wrong, root cause, fix

| Symptom the user saw | Root cause | Fix (commit) |
|---|---|---|
| Narration cut mid-sentence ("…obecnie na wysokości" [cut]) | The planner picked `in`/`out` seconds; tidy only snapped to **word** boundaries | Sentence catalogue + sentence ids in the plan (`ytedit sentences`, `b2b359d`); tidy snaps true breaks to sentence boundaries, retracts when it cannot extend (`66ad648`, `b2b359d`) |
| A cutaway paused the narration for its own length | A cutaway replaced picture **and** audio | Overlay cutaways: `audio_from` keeps the take's audio under B-roll picture (`66ad648`) |
| The same narration heard twice under different pictures | (a) planner emitted overlapping pieces of one take and the overlay pass did not advance later pieces; (b) the user's own retake inside a home recording was kept whole | (a) audio ledger: no clip audio range plays twice, QC rule 31 (`b2b359d`); (b) retake/duplicate detection in the sentence catalogue; `ytedit voice` cleans pickups (in progress) |
| Post-trip narration only usable on camera | No way to play one clip's audio under other clips | Voice-over segments → `tracks.voice` + muted picture cuts (`f2cc46a`) |
| Wind on the ticket clip and others | Nobody listened; the footage log's `visual.issues` does not hear audio | `ytedit noise` ranks clips by gap noise floor / SNR / low-band ratio; ElevenLabs isolation on the windy list (`66ad648`) |
| Master 1 LU too quiet | loudnorm linear mode capped by true-peak; AAC adds ~0.4 dB overshoot | Pre-gain + true-peak limiter ahead of two-pass loudnorm; verify after encode (`34278e6`, `5d5eebb`) |
| Captions/voice/chapters drifting after edits | Absolute-time tracks re-timed only partially; voice items stretched; chapters never moved | Uniform re-timing incl. markers/chapters (`54d2286`), voice keeps its length, **voice anchors** to segments (`71478a8`) |
| Joined programme 0.5 s longer than the timeline | Segments cut on whole frames, positions computed in seconds | Frame-exact segment cuts and positions (`f2cc46a`) |
| Render failed at segment 179 after 20 min | Picture range beyond the clip's duration; validation happened per segment | Pre-flight validation of every range before any ffmpeg call (in progress) |
| Master lost, re-rendered from scratch | A remux script deleted its backup before verifying the replacement | Rule: replace only after the new file is verified; `exports/` is never touched by scratch scripts |
| Narration pickups 10–20 s early after a manual insert (street party before the pole, CTA over the market take) | Anchors referenced display ids (`s073`), which every insert/drop renumbers; resolution only checked that an id existed | Stable `VideoSegment.uid`, anchors carry uid + signature and self-repair, `Timeline.insert_segments/remove_segments` API, QC rules 33–35 and render pre-flight refuse a pickup over on-camera speech |
| Disk full twice | 4.7 GB segment cache + 3 GB intermediates per render; a worker copied the project | `ytedit clean`, free-space check before render (in progress); never copy project media |

Cost of the project: $12.32 of the $20 budget (STT $0.28, analysis $4.00, plan $3.72,
music $2.30, voice isolation $1.43, thumbnails $0.65). Wall-clock was dominated by media:
ingest ~2 h (HLG tonemap), each full render 25–45 min.

## 2. The process as it should run next time

1. `new` → drop clips (+ `input/post recording/` for home narration, symlinked into `input/`).
2. `ingest` (runs unattended; HLG tonemap is CPU-bound, ~1.5× realtime on an M1 — start it and
   do something else). Check `state.json` flags.
3. `transcribe` → `noise` → denoise the windy list with ElevenLabs **before** anyone listens.
4. `analyze` → spot-check 3 clips → `sentences` (catalogue: retakes, duplicates, instructions).
5. `plan --notes "..."` with the editorial direction (structure, what is narration, what is
   ambient, which clips are post-trip voice-over, runtime target). The planner references
   sentence ids; the build validates uniqueness/contiguity, applies sentence snapping,
   overlays, the audio ledger and voice anchors. Review `edit_plan.md` **and** the script check.
6. `render --preview --no-music` → send to the user with the narration requests. Never hand over a
   preview that has not passed `qc` (rules 31/32: no duplicate audio, no mid-sentence break).
7. the user records pickups → `voice/incoming/manifest.yaml` → `ytedit voice` (transcribe, clean
   retakes/stutters, trim, anchor, extend picture) → `tidy` → `render --preview` → `qc`.
8. `music` (after the cut is stable; regenerate only cues whose length changed) → `render
   --master` (hardware tier for drafts, x264 for the upload) → `qc` → `publish`.
9. Every correction round: edit the timeline (editor or scripts) → `tidy` → `qc` → preview.
   Masters only when the preview has been accepted.

## 3. Still worth doing (ordered by payoff)

1. **Two-stage planner**: stage 1 writes only the narration script (ordered sentence ids,
   voice-over/pickup requests, CTA), stage 2 assigns picture per script line. Removes the last
   class of story errors (redundant statements said twice in different clips, e.g. the 66 %
   air-pressure line on a location and again at home) because the script is reviewed as text
   before any picture exists. Add a semantic duplicate check across clips (embedding or LLM
   pass over the catalogue) to the script check.
2. **Preview from proxies**: the 720p preview can be cut from `media/proxies/` instead of the
   mezzanine sources (same frame counts; proxies are already CFR and SDR). Cuts preview time
   by ~3× and makes `ingest` cheaper if the mezzanine becomes lazy (only clips used in the
   timeline get normalized at render time).
3. **Lazy mezzanine at ingest**: normalize proxies + audio + thumbs at ingest (hardware
   encode), produce the full-quality source only for clips referenced by the timeline. On this
   project 59 of 76 minutes were never used but cost 1.5 h of tonemapping.
4. **Editor feedback loop**: the user's notes arrive as text ("at 4:27 the sentence is cut").
   A `ytedit notes` command that maps timestamps to segment ids and sentence ids would turn
   his review into a checklist the agent can act on without guessing.
5. **Music regeneration by cue**: when a section's length changes, regenerate only that cue
   (or loop/trim the existing file) instead of all ten.
6. **Thumbnail policy**: `publish` should generate from real frames by default (local
   composites) and use fal only for text/colour treatment of a real frame; a generated face
   that is not the user's, or a generated landscape, must never reach the candidates.
7. **Hold-out check before the master**: a scripted listen-through — transcribe the rendered
   preview's audio and diff it against the planned script (sentence ids). Any sentence heard
   twice or cut short is caught by machine before the user watches.

## 4. Lessons (added after the third correction round, 2026-09-08)

1. **Draft first, always.** Every review round is a light draft from proxies (`render --draft`; measured on the 18:27 The reference project cut: 10.6 min cold with a test suite running alongside, of which 8.3 min was the segment pass — subsequent drafts reuse the draft cache and take ~2–3 min), never a master. The master is rendered once, after the user's OK, on the hardware tier. Rationale: three masters were rendered on this project, each an hour of machine time, and every one was superseded by a note that a draft would have surfaced.
2. **Identity, not position.** Anything that must stay attached to a picture (pickups, captions, chapters) references a stable segment uid, never an absolute time or an ordinal. Two of the three correction rounds were drift bugs of this kind.
3. **Machine-checkable gates before every hand-over.** QC 31–35 (no audio twice, sentence boundaries, pickup over speech, anchors) plus the script check and the reports of `voice`/`captions`. If a class of error reaches the user, the fix is a rule, not a manual check.
4. **No scratch scripts on the timeline.** Every splice goes through `Timeline.insert_segments/remove_segments/replace_segment` or a CLI stage. The scratch script that renumbered segments is exactly how round three happened.
5. **Ask for the timecode, then look it up.** `ytedit at` turns "at 4:27" into segment and sentence ids; guessing from memory of the cut is how narration was moved to the wrong place.
6. **Cost and disk are budgets, not surprises.** Denoise everything the user speaks in (about $0.12 per clip minute), generate music once the cut is stable, `clean` after every master, keep 20 GB free.
7. **The story is a script.** Sentences are the unit; retakes and semantic repeats are removed before picture exists; a summary-level callback is fine, a restated fact is not.
