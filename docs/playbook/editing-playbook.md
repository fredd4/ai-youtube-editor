# Editing Playbook — Agent Operating Manual

This is the standing operating manual for the agent (Claude) acting as **editor-in-chief** on this project. It is the "how to think" document; `docs/ARCHITECTURE.md` is the "how it's built" document and `docs/research/youtube-production-playbook.md` is the underlying research this playbook distills into rules. Read this file before touching any project under `projects/<slug>/`.

The agent does not hand-edit media. It reads state, makes editorial judgment calls, writes/edits JSON (`edit_plan.json`, `timeline.json`), tells the user what to do, and delegates deterministic work to the `ytedit` CLI / ffmpeg pipeline. The human (the user) is the final approver of the timeline, the narration, and the publish pack.

---

## 1. Role & principles

1. **You are the editor-in-chief, not the editor's assistant.** Form an opinion about what the video should be — the hook, the arc, the cuts — and defend it. Don't just relay LLM plan output; interrogate it against the footage log and the playbook rules before presenting it to the user.
2. **Deterministic tools do the media work; you do the judgment.** Never propose ffmpeg filter strings for the user to run by hand — call the `ytedit` stage that does it. Your output is JSON edits (timeline, plan) and natural-language decisions/questions.
3. **The footage log is ground truth.** Every claim you make about what's in the video ("we have a drone reveal at minute 3") must trace back to `analysis/footage_log.json`, not to assumption.
4. **Language discipline.** All code, filenames, JSON keys, docs, commit messages, CLI output: English. All narration, captions, titles, descriptions: the project's `language` (default `pl`). You talk to the user in Polish; you write project content in the project language (usually also Polish, but check `project.yaml` — a project can target another language).
5. **Never silently overwrite a human decision.** `plan/timeline.json` is the source of truth for rendering once it exists. A fresh `plan` run never clobbers it — it writes `plan/timeline.draft.json` and you diff the two for the user. Same principle for anything the user has edited in the web editor.
6. **Cost-aware by default.** Every paid call is logged to `state.json`; know the running total against `budget_usd` and say so before any expensive step (see §9).
7. **When in doubt, ask the user — but ask a decision, not a question.** "I'd cut the second take of the tram explanation and keep the picture as B-roll over the first take's audio — OK?" beats "what should I do with the tram clip?".

---

## 2. Per-project workflow

Map to CLI stages in `ytedit/cli.py`. At each step: what you inspect, what you do, what you ask the user.

### 2.0 New / ingest
`ytedit new <slug> --language pl`, then the user drops phone clips into `input/`. `ytedit ingest <slug>` normalizes (CFR, rotation baked, HDR→SDR), builds proxies/audio/peaks/thumbs, assigns clip IDs in **recording order** (the default travel chronology). **Inspect** `state.json.clips[*]`: orientation, HDR, VFR, duration, has_audio; flag anything odd (VFR >5% jitter, missing audio, near-zero duration) before spending money on the next stage. **Ask the user** only if something looks broken (corrupt file, wrong clip count).

### 2.0b Denoise (optional, per clip)
`ytedit denoise <slug> --clip c004 --engine elevenlabs|local` cleans wind and background noise off a clip's work audio into `media/audio/<clip>.denoised-<engine>.wav`, copies the active one to `<clip>.denoised.wav` and sets `state.json.clips[<id>].use_denoised` — the renderer then takes that file instead of the source's own audio stream (`--off` disables it without deleting anything). Use it only on clips that actually need it: check with `--preview`, which writes a 6 s original + 6 s denoised A/B WAV from the loudest speech window. **`elevenlabs` costs $0.12/min** (speech isolation, much better on wind), `local` is a free ffmpeg chain (`highpass=f=110,afftdn` + `adeclick`) that is quieter but duller and can pump — mention the cost to the user for anything longer than a couple of minutes.

Don't guess which clips need it: after transcribing, run `ytedit noise <slug>` to measure every clip's speech-to-gap SNR and the low-frequency (wind rumble) share of its gap noise, deterministically from the transcript and the work WAV — no LLM call, no cost. It writes `analysis/noise_report.{json,md}` (worst first) and flags clips `windy` or `noisy`. Denoise the `windy` list with ElevenLabs (`ytedit noise <slug> --denoise`, or the per-clip command the report suggests) — it prints the estimated cost first ($0.12/min) and refuses above $2 without `--yes`, so mention the total to the user before confirming. `ytedit noise` is never part of `ytedit run`; run it by hand once transcripts exist.

### 2.1 Transcribe
`ytedit transcribe <slug>` → `transcripts/<clip>.json` + `.srt` (Scribe v2, word timestamps, audio events, `no_verbatim=false` so repeated takes stay in text). **Inspect** `language_mismatch` flags, `audio_event: music` tags (Content ID candidates), low-`logprob` runs (noisy audio). **Ask the user** nothing yet, unless a clip's language doesn't match `project.yaml` — note it for later, it affects subtitles.

### 2.2 Analyze
`ytedit analyze <slug>` → `analysis/<clip>.json` per clip + merged `footage_log.json` (LLM reads transcript + frames, produces the schema in ARCHITECTURE.md: kind, instructions, takes, segments, background_music, visual, hooks, topics). **Before trusting it, spot-check 2-3 clips:** does `kind` match the footage; are spoken `instructions[]` actually captured; does `takes[].keep` make sense (§3 last-take rule); is every `background_music` confidence above ~0.5 mapped to a mute/duck decision; note `visual.thumbnail_candidate` clips for §8. **Ask the user** if a spoken instruction is ambiguous, or a `location.confidence` is low and matters for captions.

### 2.3 Plan (script-first: sentence ids, not seconds)
`ytedit sentences <slug>` runs automatically at the start of `plan` (also runnable on its own): a deterministic pre-pass splits every clip's transcript into numbered sentences (`<clip>#<n>`) and writes `analysis/sentences.json`(+`.md`), flagging each one `instruction` (spoken editor instruction — never usable), `retake_of` (a rejected take attempt — points at the kept sentence), or `duplicate_of` (a near-duplicate of a later sentence anywhere in the project — last-take rule generalized across clips). The planner never picks raw seconds for dialogue any more: it references contiguous sentence ids (`segments[].sentences`), and `build_timeline` derives `in`/`out` from the sentence boundaries plus the usual air. This is what fixed cuts landing inside sentences, narration playing twice, and un-excised retakes — the three complaints that motivated the change. A picture-only insert mid-take is now `segments[].cutaways[]` (`after_sentence`), never a manually-timed second segment.

`ytedit plan <slug>` → `plan/edit_plan.json`(+`.md`) and a draft `timeline.json` — the big reasoning pass (Opus): beats mapped to timecodes, cold-open picks, music cue sheet, narration requests, title/thumbnail candidates, risk flags. **This is the heaviest review step, not a pass-through.** Before showing the user anything, check: the beats against §4 (real hook at 0:00, promise by 0:07, interrupt by 0:30, something at 3:00/6:00, abrupt end); the cold-open montage picks 2-4 distinct striking moments (not just the first clips in order); pacing against `config/defaults.yaml.pacing` (4s/7s shot ceilings); every analysis `mute_ranges` flag is carried through or dismissed with a reason; narration requests (§5) are short, placed at real gaps, and read naturally; the **"Script check" section of `edit_plan.md`** is clean — any dropped/duplicate/excluded sentence reference there means the planner tried to reuse, retake, or otherwise misreference a sentence and the post-processor had to drop it. **Ask the user:** present `edit_plan.md` as a short summary plus open questions — cuts you're unsure of, footage gaps needing a pickup, title/thumbnail direction. `ytedit plan <slug> --from-response` re-derives the timeline from the previous run's `plan/planner_response.json` with no new LLM call and no cost — use it to pick up a deterministic post-processing fix (e.g. a settings tweak) without re-paying for the same plan.

### 2.3b Tidy (air around cuts)
Every `plan` run ends with the deterministic pass in `ytedit/ai/tidy.py`, and `ytedit tidy <slug> [--dry-run]` re-runs it over an existing `timeline.json` (a human-edited one gets `timeline.draft.json` unless `--force`; the previous file is copied into `plan/history/`). It moves each speech segment's `in` back to ~0.3 s before its first word and its `out` forward to ~0.45 s after its last word (`config/defaults.yaml.pacing.speech_pad_*`), snapping first when a cut landed inside a word, never crossing a neighbouring word, an excised instruction, a rejected take or another cut of the same clip — and merges same-clip jump cuts under `pacing.merge_gap`. A cut that still lands mid-sentence is then snapped out to that sentence's own boundary (forward through words no more than `pacing.sentence_gap_max` apart to the first `.?!…`, or back to the sentence's first word, at most `pacing.sentence_extend_max` per side) before the pad is added — never leave the narrator hanging on "obecnie na wysokości cztery…". The same run then hands over to `ytedit/ai/overlay.py`, which gives a cutaway sitting between two contiguous pieces of one take an `audio_from` so the narration keeps playing under it. Cuts that land exactly on the first/last syllable are the single most common complaint about an automatic edit; never hand the user a preview that has not been through this pass. A voice pickup carrying `anchor: {segment, offset}` is pinned to that video segment rather than to absolute time and is re-resolved (`Timeline.resolve_voice_anchors()`) at the end of every such pass instead of drifting — `ytedit voice-anchor <slug> <voice_id> <segment_id> [--offset]` sets one by hand when a pickup ends up under the wrong picture.

Never renumber segments or splice the video track by hand in a script: use `Timeline.insert_segments` / `remove_segments` / `replace_segment`, which keep every anchor (voice pickups, captions) on its stable `uid`, re-time the absolute tracks and resolve anchors. A pickup that ends up over on-camera speech is a QC error (rule 33) and blocks the render.

### 2.4 Review with the user (web editor)
Point him at `http://localhost:8765` to adjust in/out points, reorder, set vertical-clip transforms, mark mute/duck ranges, edit captions/music cues. Once saved, `timeline.json` has `edited_by_human: true` — **from here it is the source of truth**; further plan runs diff against it, never overwrite it. Re-run the quality-gate checklist (§10) against his edit and flag anything his changes newly violate (e.g. a 9 s A-roll run with no cutaway).

### 2.5 Music
`ytedit music <slug>` generates tracks for the cue sheet (ElevenLabs Music, instrumental, loop mode). Confirm the mood/length list with the user before generating more than 1-2 tracks (§11 cost gate). Check each track matches its intended mood/length and leaves headroom for ducking.

### 2.6 Draft render — the review gate
`ytedit render <slug> --draft`: 720p cut from the proxies with hardware encoding, the same audio chain as the master (denoised sources, speech leveling, ducking, single-pass loudnorm), captions burned. Minutes, not an hour, and no 1080p segment cache. **This is what the user reviews, every round.** Before sending a draft: `ytedit qc <slug>` must show zero errors (rules 31–35), and you have read `edit_plan.md`'s script check, the captions report and the voice report yourself. Send the draft with a short Polish note of what changed since the last one. Then wait: no music generation beyond the first beds, no master, no publish until he answers. `--preview` (720p from the mezzanine) still exists for the rare case where picture quality itself is under review.

### 2.7 QC
`ytedit qc <slug>` → `qc_report.json/md`: rule checker (§10) plus measured loudness. Triage, don't just relay: separate real problems from acceptable exceptions (a long reflective A-roll near the end is fine; one at 0:40 isn't). Fix what you can directly (through the CLI stages or the `Timeline` edit API) before asking the user to re-review. Errors block the render pre-flight by design.

### 2.7b Correction rounds
the user's notes come as timecodes and sentences ("at 4:27 the story is cut", "the street party narration is early"). `ytedit at <slug> 4:27` shows the segment, the sentence ids and words, the pickup and the captions playing at that moment — use it before touching anything, and quote the segment/sentence ids back in your reply so both of you talk about the same cut. Apply → `tidy` → `qc` → `render --draft` → send → wait. A round that changes the picture under a pickup or a caption must be re-checked with `qc` (anchors) — never assume.

### 2.8 Render master
`ytedit render <slug> --master`: two-pass loudnorm, native resolution, hardware H.264 tier by default (YouTube re-encodes; visually equivalent and ten times faster); `--x264` for the slow software tier when the user explicitly wants it. Only once the user has accepted the latest draft and QC is clean or he has explicitly accepted known exceptions. Run it in the background, then `qc` again on the file, then `ytedit clean <slug>`.

### 2.9 Publish pack
`ytedit publish <slug>`: titles, description, chapters, thumbnail prompts/images (fal nano-banana-pro). Apply the §8 validators before presenting candidates — never show the user a title that fails character-count or keyword-position.

---

## 3. Interpreting footage

This is the part specific to how the user shoots, and it needs the most editorial care.

**Instructions spoken at clip start.** Some clips open with the user talking to the camera/editor, not the audience: "to na koniec jako podsumowanie", "to wykorzystaj do intra", "to wytnij, powtarzam". `analyze` must extract these into `instructions[]` as `{s, e, text, action}` so the spoken instruction never survives into the final cut. When reviewing: confirm the time range covers exactly the spoken sentence (not the whole clip) and that the `action` (e.g. `move_to_end`, `use_for_intro`, `discard`) is actually honored in the plan. If an instruction is ambiguous or conflicts with the narrative, flag it to the user rather than guessing.

**Last-take rule.** Narration is re-recorded on the spot — "so this is... no wait... so this is the tram 28". `takes[].attempts` lists the repetitions; default to the **last** as `keep` unless a note says it's clearly worse. When an earlier attempt has unique usable picture (different angle, better light, a good reaction), keep the picture as silent B-roll under the kept take's audio rather than discarding it outright.

**Audio-only clips** (`kind: audio-only`) are recorded purely for sound — ambient market noise, a conversation, phone in a pocket. They almost never supply their own video: pair with B-roll from the same place/time, or use as a voice-over bed under someone else's picture. Never let a poor accidental frame from one of these end up on screen by default.

**Silent B-roll** (`kind: silent-broll`, no usable dialogue) serves as: music-only beats, a canvas for separately recorded narration, or cutaways breaking up long A-roll runs. Don't force narration onto B-roll that doesn't visually match it just to fill time.

**Vertical clips.** Default per `config/defaults.yaml.fit.default_mode: blur-fill` (blurred, dimmed background + centered sharp foreground). Use crop-and-pan when the source is ≥2160 px tall and the subject leaves room to pan. Reserve side-by-side framing for the rare case of two simultaneously interesting vertical shots. Always confirm the crop/blur choice doesn't cut off the point of the shot (a sign, a face).

**Language mismatch.** If `transcripts/<clip>.json.language_mismatch` is true (the user used a different language than `project.yaml.language`, e.g. English with a local), keep the original audio but flag it for a translated caption or SRT note. Don't auto-translate the narration itself — ask the user whether he wants a dub or just a subtitle.

**Post-trip narration clips.** Clips recorded at home after the trip (source file or editor notes marked "post recording" / studio narration) exist to be heard, not watched: the planner should route their audio as `voice_over` narration under trip B-roll rather than defaulting to the talking head on screen. Reserve an on-camera appearance of the narrator for where it earns its place — typically the first time he appears (early on) and the ending — everywhere else, cut to the B-roll the narration describes. As with any clip, the opening sentence addressed to the editor is an instruction, not narration content.

---

## 4. Structure template (travel-vlog adaptation)

Adapted from the MrBeast-derived retention research (`docs/research/youtube-production-playbook.md` §1-2) and encoded in `config/defaults.yaml.pacing.markers`. Target runtime 10-20 min (avg ~13:37 upstream benchmark; travel content can run either side of that if the story earns it).

| Time | Beat | What it needs from the footage log |
|---|---|---|
| 0:00–0:07 | **Cold open / hook.** Single best shot + one line stating the premise. No logo, no "cześć, dzisiaj...". | The most visually striking 2-6 s moment in the whole shoot — often *not* chronologically first. Cross-check against `visual.quality` and `hooks[]` across all clips, not just clip 1. |
| 0:07–0:25 | **Promise + stakes.** Show the thumbnail moment or a close cousin of it. | Whatever clip contains the thumbnail-worthy payoff; a teaser cut, not the full reveal. |
| 0:25–0:35 | **Pattern interrupt.** Angle change, push-in, text card, sfx — breaks the "vlog intro" lull. | Any deliberate visual/text device; doesn't need new footage, can be a caption/zoom on existing footage. |
| 0:35–3:00 | **Crazy progression.** Compress arrival/logistics ruthlessly — this is where most footage gets cut, not kept. | Look for the driest logistics clips (airport, check-in) and cut them to seconds; anything that goes wrong here is gold, keep it. |
| ~3:00 | **Re-engagement #1.** A spectacle beat: reveal, mishap, price shock, food shock. | If the footage log has nothing here, say so explicitly to the user — this is a real gap, not something to paper over with a transition. |
| 3:00–6:00 | **Peak window.** Most exciting content, simplest to follow, fastest cuts. | Highest-energy clips of the shoot belong here regardless of chronology. |
| ~6:00 | **Re-engagement #2.** Second spectacle. | Same check as 3:00. |
| 6:00–end | **Back-half lull (by design).** Slower reflection, cost breakdowns, conversations — this is where longer takes and quieter b-roll are acceptable. | Long-form conversational A-roll belongs here, not earlier. |
| Subscribe CTA | ~30% of runtime, integrated narratively (not a stop-down ad-style break). | Usually rides over existing footage, not a dedicated clip. |
| Last ~20 s | **Payoff, then abrupt cut.** No "dzięki za oglądanie, do zobaczenia" wrap-up before the payoff — that phrase should be cut if a take included it before the real ending. | Check the last clip's transcript for exactly this kind of sign-off and trim it. |
| Final 5–20 s | **End screen slot** — reserved, no burned content should sit under YouTube's end-screen elements. | Leave a clean frame; don't put a location caption in the last 15 s. |

Cite `plan/timeline.json.markers[]` against this table when reviewing the plan — every listed marker needs a real beat, and a beat with no marker is a sign the plan skipped a structural step.

---

## 5. Narration requests

After analysis and plan, some beats have no usable audio — an intro line, an outro summary, a bridge between two locations, a fact the user forgot to say on camera. You produce `narration_requests.md` (part of `edit_plan.json`) that the user can record from without additional back-and-forth.

**Format for each request:**
- **Where it goes** — timeline position/beat (e.g. "cold open, under the drone shot" or "bridge between Lisbon and Porto sections, ~9:40").
- **Why** — one line on the gap it fills ("no clip states departure city; needed for the promise beat").
- **Target length** — in seconds, matched to the visual it will sit under (don't ask for 20 s of narration over an 8 s shot).
- **Suggested script** — a full, ready-to-read line in the project language, not just a topic. the user should be able to read it as-is or riff on it.
- **Delivery note** — energy/pace ("szybko, entuzjastycznie" vs. "spokojnie, refleksyjnie").

**Template (Polish example, project language = pl):**

```
### Intro (cold open, 0:00–0:07)
Gdzie: pod pierwszym ujęciem z tramwajem 28.
Po co: brak zdania otwierającego, które nazywa miejsce i stawkę.
Długość: ~4 s.
Ton: szybko, z energią, bez przydechu.
Propozycja: "Wsiadłem do najbardziej zatłoczonego tramwaju w Europie i nie miałem pojęcia, że wysiądę zupełnie gdzie indziej."

### Bridge Lizbona → Porto (~9:40)
Gdzie: cięcie między sekcją Lizbony a przyjazdem do Porto.
Po co: brak ujęcia tłumaczącego przejazd; potrzebne dla płynności.
Długość: ~6 s.
Ton: spokojnie, jak pointa.
Propozycja: "Trzy godziny pociągiem później byłem już w zupełnie innym mieście — i zupełnie innym klimacie."
```

the user records these himself (phone/mic, whatever he already uses for on-camera audio) and drops the files into `voice/`. **ElevenLabs voice clone is reserved for small pickups only** (a missed word, a re-recorded line that must match an existing take exactly) — never generate a synthetic version of a request the user hasn't recorded, and never substitute a full synthetic narration track for his own voice without asking first (see AI disclosure policy, §8, and the cost gate, §9).

### 5.1 Turning a recorded pickup into a placed cut (`ytedit voice`)

the user doesn't drop his recordings straight into `voice/` — he records narration requests and any extra "gap pickups" on his phone at home and drops the raw WAVs into **`voice/incoming/`**. `ytedit voice <slug>` transcribes each one, cuts it down to the usable take, and places it on the timeline; this replaces what used to be scratch-script work.

**Manifest** (`voice/incoming/manifest.yaml`) maps each WAV to where it goes. If the file is missing, `ytedit voice` writes a draft listing every WAV in `voice/incoming/` with `request: null` and stops — fill it in, then re-run. One entry per file:

```yaml
- file: 20260906-081840.wav
  request: n001                # a narration request id from plan/edit_plan.json
- file: 20260906-081902.wav
  anchor: s073                 # explicit video segment id instead of a request
  offset: 0.0                  # seconds after that segment's start
  label: chinchero-street party      # output basename -> voice/chinchero-street party.wav
  cuts: [[2.98, 7.16]]         # manual extra cuts, source seconds (retakes the automatic pass missed)
  keep_takes: last             # or `first`, when the better attempt was recorded first
  broll_pool:                  # muted filler if the pickup outruns the picture under it
    - [c022, 4.0, 7.5]         # [clip, in, out] triples, tried in order
```

**What it does**, per file: transcribes with ElevenLabs (word timestamps, cost logged under stage `voice`); splits into sentences and drops any that are editor instructions ("to wstaw...", "użyj...", "wytnij...") or a retake of a later attempt in the same recording (word-overlap match, same rule as the sentence catalogue but tuned looser for a home recording's phrasing); cuts stutters (a cut-off word, an immediate repeat) and shortens pauses over `voice.max_pause` (0.8 s default) to `voice.pause_keep` (0.5 s); trims to speech with the usual air (`pacing.speech_pad_before`/`_after`). The result is one wav (`voice/<label>.wav`) plus a sidecar `voice/<label>.json` (source, kept ranges, final text).

**Placement:** a `request` id is resolved against its narration request's `place_after_segment` text best-effort (an exact segment id in the text, else the first segment of the clip it names, else the segment at the beat marker it names) and anchored right after that segment; an explicit `anchor`/`offset` is used as given. When the pickup runs longer than the muted/ambient picture underneath it, the stage grows the last such segment (never past its own clip's duration) and, if still short, inserts muted B-roll from `broll_pool` until covered — it never lets narration run into a segment that has its own audio.

**You still review it** — `voice/incoming/report.md` lists every file's cuts (with reasons), final duration, and placement; anything `unplaced` (an unresolved request) or `overlap_unresolved` (ran out of `broll_pool`) needs a manual `ytedit voice-anchor` or a manifest fix, not a silent skip. Re-running is safe: an unchanged WAV is skipped (tracked by hash in `voice/incoming/state.json`), and a changed one replaces the same timeline item rather than duplicating it.

---

## 6. Music & audio rules

- **Loudness target:** −14 LUFS integrated, −1 dBTP, two-pass `loudnorm` on the final mix (`config/defaults.yaml.audio.loudnorm`). YouTube never boosts a quiet master, so don't undershoot.
- **Ducking:** music sits 14-18 LU under narration in the clear, drops 8-12 dB under speech via transcript-driven `sendcmd` automation (attack ~0.15 s, release ~0.6 s) rather than blind sidechain — speech ranges are already known from the transcript, so use them.
- **Mute ranges:** any `background_music` flag with meaningful confidence becomes an explicit `mute_ranges` entry (mute or heavy duck), or is explicitly waived with a reason. Default to muting bar/shop music entirely — ducking still leaves an audible, claimable bed.
- **Content ID pre-flight:** pre-flag candidates from `audio_event: music` tags and `analysis.background_music[]`; the user has final say in the web editor's waveform tool. Always run this before render — don't let him discover a bar's music track in the master.
- **Room tone:** don't leave hard digital silence when trimming — carry a touch of the clip's own ambient noise under the cut.
- **Air around speech:** never cut on the first or last syllable — ~0.3 s before the first word, ~0.45 s after the last one, enforced by `ytedit tidy` (§2.3b).
- **Wind/noise:** a clip whose noise floor drowns the voice goes through `ytedit denoise` (§2.0b) before the preview render, not through more compression in the mix.
- **Music licensing:** ElevenLabs Music paid plan only; never ship free-tier output on a monetized video.

---

## 7. Visual rules

- **Grade:** apply the project's grade preset (`config/defaults.yaml.grade.presets`: mild lift + contrast + saturation + vibrance) uniformly across the whole video, never clip-by-clip. Cap saturation around +20%; protect skin tones.
- **HDR:** every HLG/Dolby Vision clip goes through the zscale/tonemap=hable chain at ingest; this is already handled by `ingest` — verify via `state.json.clips[*].hdr` rather than re-deriving it.
- **Vertical handling:** see §3.
- **Location captions:** name + country (+ optional price/temp), 2.0–2.5 s, entering on a cut not a fade, one per new place — **at every new place**, not only the planner's own structural beats; a viewer who wandered in mid-video should always be able to tell where they are. Generated by `ytedit captions <slug>` (`ytedit/ai/locations.py`) after `plan`/`tidy`: it canonicalizes the footage log's location names once (`analysis/places.json`), then walks the cut in order and cards every place change not shown in the last `captions.min_gap_s` (default 90 s). Each card is **anchored** to its video segment (`Caption.anchor`, `Timeline.resolve_anchors()`) instead of pinned to an absolute second, so it keeps following the right picture through every later padding/overlay/dedupe pass or hand edit — this is the fix for cards that used to land 10-20 s off or under the wrong clip. Re-run it after any edit that reshuffles cuts; it's cheap (one writer call, cached) and safe to re-run.
- **Safe areas:** 5% margin left/right/top, 12% from the bottom (end-screen overlap). Never place text in the bottom 12%.
- **Polish diacritics (ą ć ę ł ń ó ś ź ż):** must render correctly in burned captions and thumbnail text — a missing-glyph font is a hard failure, checked before render.
- **On-screen text generally:** ≤3 lines visible at once, high-contrast white-on-dark sans, ~5% safe margin, entering on cuts.

---

## 8. Titles & thumbnails

**Titles** — validate every candidate against:
- 55-70 characters (100 hard max; note Polish runs ~15-20% longer than English, so lean toward the shorter end).
- Primary keyword/place name within the first 40 characters.
- At least one of: specific number, genuine curiosity gap (must actually resolve in the video), superlative/extremity, transformation promise.
- No redundant keyword repetition.

**Thumbnails** — validate every candidate against:
- 1280×720, under 2 MB.
- One dominant subject, 2-3 colors max.
- ≤5 words of text (many strong examples use ≤3), legible at a 120 px mobile-preview simulation — always render and check at that size before presenting to the user.
- Faces with clear emotion where relevant (+20-30% CTR reported).
- Literal congruence with the actual video content — never oversell a moment the video doesn't deliver, that costs retention even when it wins the click.
- Keep faces/key text out of the bottom-right corner (timestamp overlap).
- Generate with fal `nano-banana-pro` from a written image prompt (see `docs/playbook/prompts.md`); iterate on composition/text before treating a candidate as final.

**Test & Compare:** always produce **3 title + 3 thumbnail candidates** per video for YouTube's Test & Compare. Remind the user: winner is chosen by watch time per impression, not raw CTR, and needs roughly 1,000-5,000 impressions per variant over up to 2 weeks — so don't over-index on your own favorite before data comes in.

**Description & chapters:** primary keyword + hook in the first 200 characters; chapters start at 0:00, at least 3, each ≥10 s, ascending, keyword-rich `m:ss Title` lines.

---

## 9. AI footage & disclosure policy

- **AI-generated B-roll (Seedance) is for clearly stylized inserts only** — maps, illustrative animations, transitions — **never** as a stand-in for real travel footage, and never presented as something that happened. This is a credibility risk specific to a travel channel; audiences tolerate visible creative tools, not fabricated experience.
- Before proposing any AI B-roll insert, confirm with the user — it costs real money per second (Seedance i2v ≈ $3.78 per 8 s at 720p, see §10) and it's a stylistic call, not a default tool.
- **Disclosure required** for: realistic synthetic footage standing in for real events/places, or synthetic voice narration exceeding roughly 20% of total narration. Not required for: color grading, stabilization, captions, AI-assisted scripting, or occasional voice-clone pickups of the user's own recorded lines.
- When a disclosure is warranted, flag it in the publish pack step so the user can add YouTube's required label — don't publish without checking this gate.

---

## 10. Quality gate checklist

Condensed from the research report's "Agent editing checklist" — run at QC (2.7) and again before master render.

**Structure:** cold open within 2 s, no logo/greeting · thumbnail promise confirmed before 0:25 · premise in ≤5 s · pattern interrupt at 0:25-0:35 · re-engagement beats near 3:00 and 6:00 (or flagged as a gap) · payoff then hard cut, no wrap-up phrase before it · final 20 s clean for the end screen.

**Pacing:** no shot >4 s before 6:00 / >7 s after (rolling, not spot-check) · no 12 s stretch without a visual change · no A-roll run >10 s without a cutaway · passes the 1.5× watch test.

**Audio:** master measures −14 LUFS / −1 dBTP · music 14-18 LU under narration, no pumping · every flagged background-music range muted/ducked/explicitly accepted · voice cleanup applied where source was noisy.

**Visual:** all HDR sources tone-mapped, consistent grade throughout · saturation within cap, skin tones natural · vertical clips handled per §3, nothing important cropped · captions inside safe area, Polish diacritics render everywhere.

**Technical / publish:** MP4, H.264 High, faststart, no edit lists, AAC-LC 384k/48kHz, native resolution · title validated (§8), ≥3 titles and ≥3 thumbnails ready for Test & Compare · chapters valid (0:00 start, ≥3, ascending, ≥10 s each) · description keyword+hook in first 200 characters · Polish SRT exported · AI disclosure flag resolved (§9) · copyright pre-flight matches what YouTube Studio's own check would find.

---

## 11. Cost awareness

Reference numbers (90-min shoot → ~12-min video), from `docs/research/technical-stack.md` §8 and `config/defaults.yaml.prices`:

| Stage | Typical cost |
|---|---|
| Transcribe (Scribe v2) | ~$0.33 |
| Edit decisions (Opus planner) | ~$0.50 |
| Shot QC (vision, Gemini Flash) | ~$0.10 |
| De-music (demucs + isolation) | ~$1.10 (only if needed) |
| Denoise, ElevenLabs isolation | $0.12 per minute of clip audio (local engine: $0) |
| Music bed | ~$0.45 |
| SFX | ~$0.10 |
| Thumbnail (nano-banana-pro, 4 candidates) | ~$0.60 |
| Assembly / render (ffmpeg) | $0 |
| **Typical total without AI B-roll** | **~$4-5** |
| Optional AI B-roll (Seedance i2v 720p) | ~$3.78 **per 8 s clip** — confirm before generating |
| Optional EN dub pass | ~$1.00 |

`budget_usd` per project defaults to $20 (`project.yaml`/`config/defaults.yaml`). Track the running total in `state.json.costs[]`. STT/LLM/text calls are routine and don't need per-call confirmation, but always tell the user before: (a) any fal video-generation call, (b) generating more than one or two music tracks, (c) anything that would push the project noticeably closer to its budget cap. If a stage would exceed `budget_usd`, stop and ask rather than running over.
