# Technical Stack Report (Sept 2026)

Scope: Polish-narrated iPhone travel footage. Keys: ElevenLabs, OpenRouter, fal.ai. Local: Python 3.12, ffmpeg 7.1.1 (Homebrew; has libx264, libx265, libvidstab, libass, libzimg, videotoolbox, audiotoolbox; **no libplacebo**), Apple Silicon.
Items marked **[UNVERIFIED]** were not confirmed against a primary source.

---

## 1. Speech-to-text with word timestamps (Polish)

### 1.1 ElevenLabs Scribe v2 — PRIMARY

`scribe_v1` deprecated; use `scribe_v2`. Docs: https://elevenlabs.io/docs/capabilities/speech-to-text, OpenAPI: https://api.elevenlabs.io/openapi.json

`POST https://api.elevenlabs.io/v1/speech-to-text` (multipart/form-data)

| Param | Default | Notes |
|---|---|---|
| `model_id` | — | `scribe_v2` |
| `file` | — | max 3–5 GB, up to 10 h |
| `source_url` | — | hosted audio/video URL (max 2 GB) |
| `language_code` | auto | `pol` / `pl` |
| `timestamps_granularity` | `word` | `none|word|character` |
| `tag_audio_events` | `true` | emits `(laughter)`, `(music)` etc. as `audio_event` items |
| `diarize` | `false` | speaker labels; `num_speakers` max 32 |
| `no_verbatim` | `false` | **keep false** — we want false starts/repeated takes in the transcript |
| `keyterms` | `[]` | max 1000 terms — bias toward place names |
| `webhook` | `false` | 202 immediately |
| `additional_formats` | — | SRT/VTT export |

Response:
```json
{"language_code":"pol","language_probability":0.98,"text":"...",
 "words":[{"text":"Dzień","start":0.319,"end":0.719,"type":"word","speaker_id":"speaker_0","logprob":-0.041},
          {"text":" ","start":0.719,"end":0.739,"type":"spacing"},
          {"text":"(laughter)","start":5.1,"end":6.0,"type":"audio_event"}],
 "audio_duration_secs":743.2,"transcription_id":"..."}
```
`type ∈ word | spacing | audio_event`. `logprob` per word = confidence (flag mumbled takes).

Price **$0.22/hour**. Polish is in ElevenLabs' "Excellent" tier (≤5% WER); arXiv 2603.02246 finds Scribe best for Polish.

```python
from elevenlabs.client import ElevenLabs
el = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
with open("clip.wav","rb") as f:
    tr = el.speech_to_text.convert(file=f, model_id="scribe_v2", language_code="pol",
        timestamps_granularity="word", tag_audio_events=True, diarize=True,
        keyterms=["Lizbona","Alfama"])
words = [w for w in tr.words if w.type == "word"]
events = [w for w in tr.words if w.type == "audio_event"]
```

### 1.2 OpenRouter fallback
`POST https://openrouter.ai/api/v1/audio/transcriptions` with `{"model":"openai/whisper-large-v3","input_audio":{"data":b64,"format":"mp3"},"language":"pl","response_format":"verbose_json","timestamp_granularities":["word"]}` — word timestamps only for OpenAI-compatible providers. Models: `openai/whisper-large-v3` (~$0.027/h via DeepInfra), `openai/whisper-large-v3-turbo`, `openai/gpt-4o-transcribe`, `deepgram/nova-3`, `google/chirp-3`. Billing units vary by provider — read `usage.cost`.

Audio as chat content part (semantic questions): `{"type":"input_audio","input_audio":{"data":b64,"format":"wav"}}` on Gemini models. Base64 only.

### 1.3 fal.ai STT
`fal-ai/elevenlabs/speech-to-text` ($0.03/min — 8× direct price, avoid). `fal-ai/whisper` (`audio_url`, `language`, `chunk_level=word`, `diarize`) price [UNVERIFIED].

### 1.4 Local (Apple Silicon)
`mlx-whisper` 0.4.3 with `mlx-community/whisper-large-v3-turbo` ≈ 14× realtime on M4 Pro, `word_timestamps=True`. Less precise word timings than Scribe. Offline fallback.

### 1.5 Recommendation
Primary Scribe v2; fallback OpenRouter whisper-large-v3; offline mlx-whisper. **Take/repetition detection** is downstream: normalise words, sliding n-gram (n≈5) within 60 s window, hand candidates + logprob to the LLM to choose the keeper (usually the last take).

---

## 2. ElevenLabs Music API

| Path | Purpose |
|---|---|
| `POST /v1/music` | compose → raw audio bytes |
| `POST /v1/music/detailed` | multipart/mixed: JSON (composition_plan + metadata) + audio; `song-id` header |
| `POST /v1/music/plan` | generate a composition plan from a prompt |

Body for `/v1/music`:
```json
{"prompt":"warm nostalgic acoustic travel bed, fingerpicked guitar, light percussion",
 "music_length_ms":90000,"model_id":"music_v2","force_instrumental":true,
 "generation_mode":"track","seed":42,"store_for_inpainting":true}
```
- `generation_mode ∈ track | loop | ambience | video_to_music` — `loop` for seamless beds. [UNVERIFIED how video_to_music takes video]
- `model_id`: `music_v1` deprecated; **always pass `music_v2`**.
- `prompt` and `composition_plan` mutually exclusive. Length 3 000–600 000 ms (docs say 5 min max) [UNVERIFIED].
- Query `output_format`: `mp3_44100_192`, `mp3_48000_320`, etc.
- Composition plan: `positive_global_styles[]`, `negative_global_styles[]`, `sections[{section_name, positive_local_styles, negative_local_styles, duration_ms, lines}]`.
- **Price $0.15/min.** Concurrency 2 on Starter/Creator/Pro.
- Licensing: cleared for nearly all commercial uses incl. social video on paid plans; free tier not for monetized use; prompts naming artists rejected (`bad_prompt`). [UNVERIFIED per-tier terms — confirm before monetising]

```python
audio = el.music.compose(prompt="...", music_length_ms=90_000, model_id="music_v2",
                         force_instrumental=True, output_format="mp3_44100_192")
open("bed.mp3","wb").write(b"".join(audio))
```
Raw httpx variant that worked in amazonia-studio: POST `/v1/music` with `Accept: audio/mpeg`, bytes in `r.content`, `song-id` header.

---

## 3. ElevenLabs voice / isolation / SFX / dubbing

**Instant Voice Clone:** `POST /v1/voices/add` multipart: `name`, `files[]`, `remove_background_noise`, `description`, `labels`. Response `{voice_id, requires_verification}`. 1–2 min clean single-speaker audio; >3 min can hurt. Professional clone: 30–180 min audio, 3–6 h training.

**TTS:** `POST /v1/text-to-speech/{voice_id}`. Models: `eleven_v3` (5k chars, 70+ langs, $0.10/1k chars), `eleven_multilingual_v2` (10k, $0.10/1k), `eleven_flash_v2_5` (40k, $0.05/1k). Body: `text`, `model_id`, `language_code="pl"`, `voice_settings{stability,similarity_boost,style,speed,use_speaker_boost}`, `seed`, `previous_text`/`next_text` (preserve prosody when stitching chunks). `output_format=mp3_44100_192` or `pcm_48000`.

**Audio Isolation:** `POST /v1/audio-isolation` multipart `audio`. $0.12/min. Speech-from-noise isolator, **not** a music stem splitter (use Demucs for music beds, then isolation for cleanup).

**Sound Effects:** `POST /v1/sound-generation`: `text`, `duration_seconds` (0.5–30), `prompt_influence` (0.3), `loop`, `model_id=eleven_text_to_sound_v2`. $0.12/min.

**Dubbing:** `POST /v1/dubbing` (`file`/`source_url`, `target_lang`, `source_lang`, `drop_background_audio`, ...) → `{dubbing_id}`; `GET /v1/dubbing/{id}`, `/audio/{lang}`, `/transcript/{lang}`. v1 $0.33–0.50/min, v2 $2.20/min. Cheaper: translate transcript with LLM + IVC voice + TTS (~$1 per 12-min video).

---

## 4. fal.ai (Sept 2026)

```python
import fal_client            # fal-client 1.0.1
url = fal_client.upload_file("frame.jpg")
result = fal_client.subscribe(APP, arguments={...}, with_logs=True, on_queue_update=cb)
handle = fal_client.submit(APP, arguments={...}); handle.request_id
st = fal_client.status(APP, request_id); res = fal_client.result(APP, request_id)
# *_async twins exist
```
Any model schema: `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<id>`

**Image-to-video:**
- `bytedance/seedance-2.5/image-to-video`: `image_url` (req), `prompt` (req), `end_image_url`, `resolution 480p|720p|1080p` (720p), `duration "auto"|"4".."30"`, `aspect_ratio`, `generate_audio` (true), `seed`. Output `{video:{url,...}, seed}`. ~$0.22/s 480p, ~$0.47/s 720p. No `camera_fixed` param (prompt-driven).
- `fal-ai/kling-video/v3/pro/image-to-video`: `start_image_url`, `prompt`/`multi_prompt[]`, `duration "3".."15"`, `end_image_url`, `elements[]`, `negative_prompt`, `cfg_scale`. $0.112/s (no audio), $0.168/s (audio).
- `fal-ai/veo3.1/image-to-video`: `duration 4s|6s|8s`, `resolution 720p|1080p|4k`, `generate_audio`. Price [UNVERIFIED].

**Video-to-video / alternate angle:** weak for real footage. `bytedance/seedance-2.5/reference-to-video` (`image_urls[]`, `video_urls[]`, `audio_urls[]`, referenced as `[Video1]`; editing/extension; fal says NOT for different camera angles; ~$0.28/s 720p). `fal-ai/kling-video/o3/4k/video-to-video/reference|edit` ($0.42/s). Runway Aleph does new angles but on Runway's own API [UNVERIFIED on fal]. **Guidance:** don't fake alternate angles of real footage; use image-to-video on a still frame you actually shot for a genuinely different B-roll beat (~$3.78 per 8 s at 720p), clearly stylized inserts only.

**Upscaling:** `fal-ai/topaz/upscale/video` (`video_url`, `model` Proteus/Artemis/..., `upscale_factor`, `target_fps`; $0.01/s ≤720p, $0.02/s ≤1080p, $0.08/s >1080p) — best for live action. `fal-ai/seedvr/upscale/video` ($0.001/megapixel) — for AI video.

**Thumbnails:** `fal-ai/nano-banana-pro/edit` (`prompt`, `image_urls[]` up to 14, `num_images` 1–4, `aspect_ratio "16:9"`, `resolution 1K|2K|4K`, `output_format`) $0.15/image (4K 2×) — face + scene + logo compositing, strong typography. `bytedance/seedream/v5/pro/edit` $0.0675–0.135/image. `fal-ai/nano-banana-pro` text-to-image.

---

## 5. OpenRouter — the brain (live prices 2026-09-04, $/M tokens in/out)

| Model | Context | In | Out | Modalities |
|---|---|---|---|---|
| `anthropic/claude-fable-5.1` | 1M | 10 | 50 | text, image, file |
| `anthropic/claude-opus-5` | 1M | 5 | 25 | text, image, file |
| `anthropic/claude-sonnet-5` | 1M | 2 | 10 | text, image, file |
| `anthropic/claude-haiku-4.5` | 200k | 1 | 5 | text, image, file |
| `google/gemini-3.1-pro-preview` | 1M | 2 | 12 | text, image, **video**, audio |
| `google/gemini-3.8-flash` | 1M | 0.75 | 3.75 | text, image, **video**, audio |
| `google/gemini-3.5-flash-lite` | 1M | 0.30 | 2.50 | text, image, video, audio |
| `openai/gpt-5.6-terra` | 1M | 2 | 12 | text, image |
| `openai/gpt-5.6-luna` | 1M | 0.2 | 1.2 | text, image |

`:batch` variants at 50% off. Recommendation: `anthropic/claude-opus-5` for edit-decision pass; `anthropic/claude-sonnet-5` or `google/gemini-3.8-flash` for cheap passes/vision frame QC.

Vision request:
```python
payload = {"model":"anthropic/claude-opus-5",
 "messages":[{"role":"user","content":[
   {"type":"text","text":"Rate each frame... Return JSON"},
   {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,...","detail":"low"}}]}],
 "response_format":{"type":"json_object"}}
headers={"Authorization":f"Bearer {KEY}","HTTP-Referer":"http://localhost:8000","X-Title":"yt-editor"}
```
Extract frames via `-vf "select='gt(scene,0.3)',scale=512:-2" -vsync vfr`, send 8–16 per call.

Native video input exists on Gemini models but the OpenRouter content-part shape is [UNVERIFIED]; pragmatic: 1 fps frame grid → images to `google/gemini-3.8-flash`.

---

## 6. ffmpeg 7.1 recipes (all filters verified in local build)

### 6.1 Two-pass loudnorm → −14 LUFS / −1 dBTP
```python
def measure(path):
    p = subprocess.run(["ffmpeg","-hide_banner","-i",path,"-af",
        "loudnorm=I=-14:TP=-1:LRA=11:print_format=json","-f","null","-"],capture_output=True,text=True)
    return json.loads(re.findall(r"\{[^{}]*\"input_i\"[\s\S]*?\}", p.stderr)[-1])
s = measure("mix.wav")
af = (f"loudnorm=I=-14:TP=-1:LRA=11:measured_I={s['input_i']}:measured_LRA={s['input_lra']}:"
      f"measured_TP={s['input_tp']}:measured_thresh={s['input_thresh']}:offset={s['target_offset']}:linear=true")
```
Alt: `ffmpeg-normalize input.mp4 -nt ebu -t -14 -tp -1.0 -lra 11 -c:a aac -b:a 384k -ar 48000 -o out.mp4`

### 6.2 Ducking
Sidechain:
```
[1:a]asplit=2[vc][sc];
[0:a][sc]sidechaincompress=threshold=0.03:ratio=12:attack=25:release=450:makeup=1:knee=2.8:link=maximum:detection=rms:level_sc=1.5[duck];
[duck][vc]amix=inputs=2:duration=longest:normalize=0[mix]
```
Both inputs need same rate/layout: `aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo`.

**Preferred (deterministic, transcript-driven):** `sendcmd` file
```
1.20 volume volume 0.25;
5.80 volume volume 1.00;
```
`-af "sendcmd=f=ducks.cmd,volume=1:eval=frame"` or inline `volume='if(between(t,1.2,5.8),0.25,1)':eval=frame`. Shape ramps explicitly.

### 6.3 Silence detection
`-af silencedetect=noise=-35dB:d=0.45 -f null -` → parse `silence_start`/`silence_end`, subtract ~120 ms headroom.

### 6.4 Scene detection
`-vf "scdet=threshold=12,metadata=print:file=scenes.txt" -f null -` (writes `lavfi.scd.score/time`), or `select='gt(scene,0.3)',showinfo`.

### 6.5 Colour grade
```
eq=brightness=0.02:contrast=1.08:saturation=1.06:gamma=1.02,
colorbalance=rs=-0.02:bs=0.03:rm=0.01:bm=-0.01,
curves=preset=lighter,
vibrance=intensity=0.25,
unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.6
```
`vibrance` protects skin better than `eq=saturation`. unsharp ≤0.8; skip for iPhone.

### 6.6 HDR → SDR
Detect: `ffprobe -v error -select_streams v:0 -show_entries stream=color_transfer,color_primaries,color_space,pix_fmt -of json in.mov` → `arib-std-b67` = HLG, `smpte2084` = PQ; key absent = SDR (handle KeyError).
```
zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p
```
PQ: `npl=1000`. `desat=0` punchy; raise to 1–2 if highlights go neon.

### 6.7 Vertical in 16:9 with blurred fill
```
[0:v]split=2[bg][fg];
[bg]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=luma_radius=45:luma_power=2:chroma_radius=25:chroma_power=1,eq=brightness=-0.06[bgb];
[fg]scale=-2:1080:flags=lanczos[fgs];
[bgb][fgs]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p[v]
```

### 6.8 Crossfades, concat, titles
`[0:v][1:v]xfade=transition=fade:duration=0.75:offset=9.25[v];[0:a][1:a]acrossfade=d=0.75:c1=tri:c2=tri[a]` — offset = dur(clip1) − transition; xfade requires identical resolution/pix_fmt/fps (normalise first); 58 transitions. N clips: chain pairwise with accumulating offsets, or concat demuxer with hard cuts.

ASS lower-third (libass): `Dialogue: 0,0:00:03.00,0:00:08.00,Loc,,0,0,0,,{\fad(400,400)\pos(120,900)}LIZBONA, PORTUGALIA` → `-vf "ass=lower3rd.ass"`.

drawtext with fades: `drawtext=fontfile=...:text='Lizbona':fontcolor=white:fontsize=54:x=120:y=h-220:box=1:boxcolor=black@0.45:boxborderw=18:alpha='if(lt(t,3),0,if(lt(t,3.4),(t-3)/0.4,if(lt(t,7.6),1,if(lt(t,8),(8-t)/0.4,0))))'`

PNG overlay: `[0:v][1:v]overlay=80:H-h-80:enable='between(t,3,8)'`.

### 6.9 Stabilise / denoise / VFR→CFR
- `vidstabdetect=shakiness=6:accuracy=15:result=tf.trf` then `vidstabtransform=input=tf.trf:smoothing=24:zoom=0:optzoom=1:interpol=bicubic`
- `hqdn3d=luma_spatial=3:chroma_spatial=2:luma_tmp=6:chroma_tmp=4` (fast) / `nlmeans=s=1.5:p=7:r=15` (slow)
- VFR→CFR (ffmpeg 7 syntax): `-fps_mode cfr -r 30 -vf fps=30`. **Normalise before xfade/concat.**
- 10-bit HEVC/ProRes decode natively; intermediate `-c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le` or `prores_videotoolbox`.

### 6.10 Encoding
- Previews/proxies: `-c:v h264_videotoolbox -q:v 65 -profile:v high -level 4.2 -pix_fmt yuv420p -movflags +faststart -c:a aac_at -b:a 256k` (no CRF; `-q:v` 1–100). ~5× realtime at 720p. Visibly worse than libx264 at equal bitrate.
- Master: `libx264 -crf 18 -preset slow`.

### 6.11 YouTube master
```
ffmpeg -i edit.mov -c:v libx264 -profile:v high -level 4.2 -crf 18 -preset slow -pix_fmt yuv420p \
  -g 30 -keyint_min 30 -sc_threshold 0 -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -movflags +faststart -c:a aac -b:a 384k -ar 48000 -ac 2 master.mp4
```
(GOP half the frame rate per YouTube: 30 fps → `-g 15`; 1 s GOP also acceptable.)

### 6.12 Removing copyrighted music under speech
- **Demucs** `htdemucs_ft` (`demucs==4.1.0`): `demucs --two-stems=vocals -n htdemucs_ft -d mps --shifts 2 -o out/ in.wav` → `vocals.wav`, `no_vocals.wav`. MPS ~28× realtime; `demucs-mlx` ~73×. Trained on sung vocals; spoken narration separates well with some thinning.
- Then ElevenLabs audio-isolation for cleanup, then re-loudnorm.

---

## 7. Python libs & editor UI

| Package | Version | Role |
|---|---|---|
| `elevenlabs` | 2.66.0 | STT, TTS, Music, IVC, isolation, dubbing |
| `fal-client` | 1.0.1 | video/image models |
| `openai` | 3.8.0 | whisper fallback (also OpenRouter base_url) |
| `ffmpeg-normalize` | 1.42.0 | EBU R128 wrapper |
| `pyloudnorm` | 0.2.0 | in-process LUFS |
| `librosa` | 1.0.0 | onset/beat detection for music-synced cuts |
| `demucs` | 4.1.0 | source separation |
| `mlx-whisper` | 0.4.3 | offline STT |
| `ffmpeg-python`, `pydub` | stale | **avoid** — use subprocess with explicit arg lists |

```python
def ff(*args):
    cmd = ["ffmpeg","-hide_banner","-nostdin","-y",*map(str,args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(p.stderr[-4000:])
    return p.stderr   # loudnorm/silencedetect/showinfo write here
```

**Editor UI: FastAPI + uvicorn + vanilla JS.** Native async for API jobs, Starlette `StaticFiles` implements HTTP Range (needed for `<video>` seeking). Serve H.264/AAC MP4 proxies (videotoolbox), never ProRes/10-bit HEVC to the browser.

**wavesurfer.js 7.12.11**: `https://cdnjs.cloudflare.com/ajax/libs/wavesurfer.js/7.12.11/wavesurfer.min.js`; ESM + regions plugin from `https://cdn.jsdelivr.net/npm/wavesurfer.js@7.12.11/dist/wavesurfer.esm.js` and `.../dist/plugins/regions.esm.js`. Use `media: videoEl` to sync with the video element; precompute peaks server-side (`ffmpeg -f f32le` → numpy downsample) and pass `peaks:`.

---

## 8. Stack summary & per-video cost (90-min shoot → 12-min video)

| Stage | Choice | Cost |
|---|---|---|
| Transcribe | Scribe v2, word timestamps, audio events | $0.33 |
| Take detection | n-gram dedupe + logprob + LLM | — |
| Edit decisions | OpenRouter `anthropic/claude-opus-5` | ~$0.50 |
| Shot QC | frames → `google/gemini-3.8-flash` | ~$0.10 |
| De-music | demucs htdemucs_ft → ElevenLabs isolation | ~$1.10 |
| Music bed | `music_v2`, instrumental, loop mode | $0.45 |
| SFX | `/v1/sound-generation` | ~$0.10 |
| Assembly | ffmpeg: loudnorm, sendcmd ducking, xfade, ASS, HLG→SDR | $0 |
| Optional AI B-roll | seedance-2.5 i2v 720p | $3.78 / 8 s |
| Thumbnail | nano-banana-pro/edit, 4 candidates @2K | $0.60 |
| EN version | LLM translate + IVC + eleven_v3 | ~$1.00 |
| Master | libx264 crf 18 slow, faststart, AAC 384k | $0 |

**Total ≈ $4–5 per video without generated B-roll.**

## Not verified
1. Polish WER numbers per model. 2. Music max length 5 vs 10 min. 3. Music per-tier YouTube monetisation rights. 4. `video_to_music` input. 5. `fal-ai/whisper` price. 6. OpenRouter transcription billing units. 7. OpenRouter video content-part shape. 8. `timestamp_granularities` per provider. 9. Veo 3.1 / Flux prices on fal. 10. Runway Aleph / Luma Modify on fal — no fal model verified does true novel-angle synthesis. 11. Audio-isolation limits. 12. libplacebo absent locally (verified).
