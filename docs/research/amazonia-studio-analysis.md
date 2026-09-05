# Amazonia Studio — Technical Report for Reuse

Source project: `a separate local codebase` (symlinked at `other_projects/amazonia-studio`).
Stack: Python 3.10+, Flask editor (all HTML inline), httpx for all HTTP, `fal_client`, asyncio + `asyncio.to_thread`, PyYAML, Pillow, ffmpeg/ffprobe via subprocess. No `elevenlabs`/`openai`/whisper deps — all raw REST.

| File | Lines | Role |
|---|---|---|
| `film_editor.py` | 2453 | Flask web editor + studio dashboard + reel editor (HTML inline) |
| `gen_film.py` | 961 | CLI pipeline: stills (OpenRouter/Gemini) → video (fal/Kling) → ffmpeg export |
| `gen_music.py` | 386 | ElevenLabs Music API generator |
| `gen_reel.py` | 361 | 9:16 vertical reel exporter (ffmpeg crop/concat/music mix) |
| `analyze_reel.py` | 371 | ffmpeg frame sampling → Gemini Vision via OpenRouter → shot suggestions |
| `gen_companion.py` | 225 | Character image gen via OpenRouter image modality |
| `archive/studio/*.py` | 1216 | Older library: providers, sqlite db, config, FastAPI review app |

## 1. ElevenLabs Music API (`gen_music.py:149-214`)

```python
EL_URL = "https://api.elevenlabs.io/v1/music"
payload = {
    "prompt": style["prompt"],
    "music_length_ms": duration_s * 1000,
    "model_id": model,                     # "music_v2" default, "music_v1" also accepted
    "force_instrumental": style.get("force_instrumental", True),
}
async with httpx.AsyncClient(timeout=300) as cl:
    r = await cl.post(EL_URL, json=payload, headers={
        "xi-api-key": EL_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    audio_data = r.content                 # raw MP3 bytes, synchronous — no polling
    song_id = r.headers.get("song-id", "unknown")
out_path.write_bytes(audio_data)
# sidecar {style_id}.json with prompt, song_id, duration, timings; manifest.json per run
```

- Observed: 180 s of audio generated in ~22 s. Cost estimate used: `$0.12/min`.
- Concurrency: `asyncio.Semaphore(2)`.
- Prompt formula that worked: *genre/fusion sentence → named instruments → atmosphere layer → arc → reference artists/films → explicit BPM → "Instrumental."*, with explicit negatives ("No drums, no percussion...").
- Example prompt: "Ethereal space ambient music featuring solo pan flute as the lead instrument. Only wind instruments and soft breath sounds... Absolutely no drums, no percussion, no bass, no beats... 50 BPM. Instrumental."
- Output layout: `data/music/run-YYYYmmdd-HHMMSS/{style_id}.mp3 + .json + manifest.json`.
- No other ElevenLabs usage in the project (no TTS, STT, voice clone).

## 2. fal.ai usage

Model IDs seen: `fal-ai/kling-video/v3/pro/image-to-video`, `fal-ai/kling-video/v3/4k/image-to-video`, `fal-ai/kling-video/o1/reference-to-video`, `fal-ai/bytedance/seedance/v1.5/pro/image-to-video`, `fal-ai/wan-2.6/image-to-video`, `fal-ai/flux/schnell` ($0.003), `fal-ai/flux-2-pro` ($0.018), `fal-ai/bytedance/seedream/v4/text-to-image` ($0.03).

Queue pattern (`gen_film.py:636-736`) — the reusable one:

```python
import fal_client
still_url = await asyncio.to_thread(fal_client.upload_file, str(still_path))
request_id = await asyncio.to_thread(lambda: fal_client.submit(MODEL, arguments=args).request_id)
# persist request_id in state JSON; later:
status = await asyncio.to_thread(lambda: fal_client.status(MODEL, rid, with_logs=True))
s = type(status).__name__            # "Completed" / "Failed" / "InProgress" / "Queued"
result = await asyncio.to_thread(lambda: fal_client.result(MODEL, rid))
video_url = result.get("video", {}).get("url") or result.get("video_url", "")
# download with httpx
```

Blocking pattern: `fal_client.subscribe(model_id, arguments=args)` → `result["images"][0]["url"]`.

Schema-negotiation trick (`archive/studio/video_gen.py:71-100`): try arg dicts from richest to barest, only swallow 422/validation errors.

`fal_client` imported lazily inside functions so the app starts without the SDK/key.

## 3. OpenRouter usage

Endpoint `https://openrouter.ai/api/v1/chat/completions`, header `Authorization: Bearer {OR_KEY}`.

Vision analysis of frames (`analyze_reel.py:192-254`):

```python
content = [{"type": "text", "text": prompt_text}]
for i, (t, jpeg) in enumerate(frames):
    b64 = base64.b64encode(annotate_frame(jpeg, i, t)).decode()
    content.append({"type": "text", "text": f"━ Frame {i} (t={t:.1f}s):"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
resp = httpx.post(URL, headers=..., json={"model": "google/gemini-2.5-pro",
        "messages": [{"role": "user", "content": content}], "temperature": 0.2}, timeout=120)
msg = resp.json()["choices"][0]["message"]
raw_content = msg.get("content")
if raw_content is None:                      # Gemini thinking-mode quirk
    raw_content = msg.get("reasoning") or ""
    for part in (msg.get("parts") or []):
        if isinstance(part, dict) and part.get("type") == "text":
            raw_content = part.get("text", ""); break
if isinstance(raw_content, list):
    text = " ".join(p.get("text", "") for p in raw_content if isinstance(p, dict))
m = re.search(r'\[.*?\]', text, re.DOTALL)   # extract JSON array
```

Image generation via chat/completions: `"modalities": ["image","text"], "image_config": {"aspect_ratio": "16:9"}`; on HTTP 400 drop `image_config` and retry; result at `choices[0].message.images[0].image_url.url` (data URL, split on "," and b64-decode). Reference images sent as data URLs.

Frame extraction (`analyze_reel.py:62-77`): one `ffmpeg -ss t -i video -vframes 1 -vf scale=960:-1 -f image2pipe -vcodec mjpeg pipe:1` per sample (interval 2 s, max 8). Annotation with PIL: vertical percentage grid lines (10%..90%) + frame index label so the LLM can answer `subject_x_pct` reliably; clamp 5–95 and convert to crop_x.

## 4. `film_editor.py` web editor

- Flask, single file, HTML/CSS/JS as Python string constants via `render_template_string`; vanilla JS + `fetch`.
- State: plain JSON `projects/{pid}/stories/{sid}/story.json`, read-modify-write on every mutation, no locking, no DB. Paths inside are repo-root-relative strings.
- Projects discovered by filesystem convention (`projects/*/meta.json`, `stories/*/story.json`).
- Routes: `/`, `/project/<pid>`, `/project/<pid>/story/<sid>`, `/api/state` (enriched with `*_exists`, `*_mtime`), `/api/jobs`, `/api/approve|reject/...`, `/api/scene/mute/<sid>` (per-clip `mute_music` flag), `/api/trim/<sid>` (trim_in/out), `/api/notes/<sid>`, `/api/export`, `/asset/video/<sid>` (`send_file(conditional=True)` + `Cache-Control: no-cache`).
- Background jobs: `subprocess.Popen([sys.executable, "gen_film.py", ...])`, child writes status into story.json; frontend polls `/api/jobs` every 5 s; completion detected via file mtime change; `?v=mtime` cache-busts `<video src>`.
- Best UX ideas: context-sensitive `renderActions()` state machine + `nextStepHint()` banner; timeline cards updated in place (no flicker); two-handle trim slider; toast(); STUDIO_CSS design tokens; reel editor with live 9:16 crop overlay.
- Rough edges: `/api/export` blocks Flask (`communicate(timeout=120)`); hardcoded `VIDEO_DURATION = 15`; no auth, `host=0.0.0.0`; path traversal in asset route.

## 5. Export code (`gen_film.py:741-867`)

1. Trim each clip and normalize to canvas: `-ss t_in -i src -t dur -c:v libx264 -c:a aac -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"`; accumulate `(start, end, mute)` in timeline coordinates.
2. Concat demuxer `-f concat -safe 0 -i list.txt -c copy`.
3. Music mix: `[0:a]volume=0.30[va];[1:a]volume=enable='between(t,a,b)+between(t,c,d)':volume=0,volume=1.0[ma];[va][ma]amix=inputs=2:duration=first[aout]`, `-c:v copy -c:a aac -b:a 192k -shortest`. If music longer than video: `-stream_loop -1` + `-t music_dur`.

Absent: xfade, fades, drawtext/subtitles, loudnorm, sidechaincompress, faststart, CRF control on final encode.

Reel exporter (`gen_reel.py`): `crop=W:H:X:0,scale=1080:1920:flags=lanczos,setsar=1`, `-ss` before `-i`, `-an`, ultrafast/crf 22 intermediates; `ThreadPoolExecutor(max_workers=6)` with ordered results; `& ~1` even-pixel clamp; saliency-based smart crop (saturation + edges, sliding window argmax).

## 6. Recommendations for the new project

**Copy nearly verbatim:** ElevenLabs music call + sidecar/manifest convention; JSON-state-per-story model with derived `*_exists/*_mtime`; Flask skeleton (asset route with conditional=True, jobs polling, toast, design tokens); `renderActions/nextStepHint` pattern; mtime completion detection; ffprobe/concat/extract helpers; frame extraction + annotation + OpenRouter vision request/parse pair; run-id + manifest convention.

**Adapt:** scenario config → YAML; export → add ducking over speech ranges (transcript-derived), captions (ASS/drawtext with `between(t,..)`), real encode params (`libx264 -crf 18 -preset medium -pix_fmt yuv420p -profile:v high -movflags +faststart -g 60`, aac 192k+ 48 kHz), `loudnorm=I=-14:TP=-1.5:LRA=11`, xfade; non-blocking export with job files and ffmpeg `-progress`; trim UI driven by real duration; cost tracking with actual spend per run.

**Drop:** Kling/character machinery, OpenRouter image gen (except thumbnails), SQLite/FastAPI leftovers, hardcoded scene lists, Polish UI strings/print logging, media files in repo root.

**Must build from scratch:** speech transcription with word timestamps, SRT/ASS generation, scene/shot detection, silence/filler trimming, ducking, loudness normalization, burned-in text, crossfades, color matching, iPhone HEVC/rotation metadata handling, YouTube metadata.
