"""ElevenLabs REST client: STT (scribe_v2), music, TTS, IVC, isolation, SFX.

Raw ``httpx`` against ``https://api.elevenlabs.io`` with the ``xi-api-key``
header - no SDK dependency.  Self-contained: the API key and an optional
``cost_callback`` are injected; nothing from ytedit.config / ytedit.project is
imported.

Every parameter name below was checked against the live OpenAPI document
(``https://api.elevenlabs.io/openapi.json``, fetched 2026-09-04).  Points where
the spec differs from ``docs/research/technical-stack.md``:

* ``POST /v1/music`` defaults to ``model_id="music_v1"``; ``music_v2`` must be
  sent explicitly on every call (we always do).
* ``force_instrumental`` defaults to ``false`` server-side, not ``true``.
* ``/v1/audio-isolation`` takes the file in a form field named ``audio``.
* ``/v1/sound-generation`` defaults ``output_format=mp3_44100_128``.
* A composition plan is the ``MusicPrompt`` shape
  (``positive_global_styles`` / ``negative_global_styles`` / ``sections``);
  ``CompositionPlan`` proper is a different, chunk-based schema.
* STT granularity is ``timestamps_granularity`` (plural "timestamps").
* STT ``keyterms`` must be sent as repeated multipart fields; a JSON array
  string is rejected with 400 ``invalid_keyword``.
* TTS output formats above 128 kbps need the Creator tier (see
  ``FALLBACK_OUTPUT_FORMAT``); ``/v1/music`` accepts them on payg.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx

logger = logging.getLogger("ytedit.ai.elevenlabs")

BASE_URL = "https://api.elevenlabs.io"

STT_USD_PER_HOUR = 0.22
MUSIC_USD_PER_MINUTE = 0.15
ISOLATION_USD_PER_MINUTE = 0.12
SFX_USD_PER_MINUTE = 0.12

# $ per 1000 characters, per TTS model.
TTS_USD_PER_1K_CHARS: dict[str, float] = {
    "eleven_v3": 0.10,
    "eleven_multilingual_v2": 0.10,
    "eleven_flash_v2_5": 0.05,
    "eleven_turbo_v2_5": 0.05,
}

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524}

# High-bitrate output is gated behind the Creator tier and above.  Verified on a
# "payg" account 2026-09-04: POST /v1/text-to-speech/{id}?output_format=
# mp3_44100_192 answers 403 with
# {"detail": {"status": "output_format_not_allowed", ...}} while POST /v1/music
# accepts the same format on the same account.  We degrade instead of failing.
FALLBACK_OUTPUT_FORMAT = "mp3_44100_128"

CostCallback = Callable[..., None]

# Audio-event labels are wrapped in brackets and are localised: English gives
# "(laughter)", Polish gives "[dzwiek]" (square brackets).  Strip either.
_EVENT_RE = re.compile(r"^[\(\[\{]?\s*(.*?)\s*[\)\]\}]?$")


class ElevenLabsError(RuntimeError):
    """Non-2xx response from the ElevenLabs API."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"ElevenLabs HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body[:500]


def parse_event_type(text: str) -> str:
    """``"(laughter)"`` -> ``"laughter"``; ``"[dzwiek]"`` -> ``"dzwiek"``.

    Spaces become underscores so the type is a usable key.
    """
    inner = _EVENT_RE.match((text or "").strip())
    label = (inner.group(1) if inner else text or "").strip().lower()
    return re.sub(r"\s+", "_", label)


@dataclass(slots=True)
class Transcript:
    """Normalised to the ARCHITECTURE ``transcripts/<clip>.json`` schema."""

    language: str | None
    language_probability: float | None
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    engine: str = "elevenlabs/scribe_v2"
    duration_s: float = 0.0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Exactly the fields the project schema stores (no raw payload)."""
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "text": self.text,
            "words": self.words,
            "events": self.events,
            "speakers": self.speakers,
            "engine": self.engine,
        }


@dataclass(slots=True)
class MusicResult:
    audio: bytes
    song_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.audio)
        return out


def normalize_transcript(
    raw: dict[str, Any], *, model_id: str = "scribe_v2"
) -> Transcript:
    """Turn a raw ``/v1/speech-to-text`` payload into a :class:`Transcript`.

    Scribe returns one flat ``words`` list whose items are typed
    ``word | spacing | audio_event``.  Spacing items are dropped, audio events
    move to ``events`` with the parenthesised label parsed into a bare type.
    """
    words: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    speakers: list[str] = []

    for item in raw.get("words") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or "word"
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or 0.0)
        if kind == "spacing":
            continue
        if kind == "audio_event":
            events.append({"type": parse_event_type(item.get("text", "")), "s": start, "e": end})
            continue
        speaker = item.get("speaker_id")
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        words.append(
            {
                "t": item.get("text", ""),
                "s": start,
                "e": end,
                "p": item.get("logprob"),
                "speaker": speaker,
            }
        )

    duration = float(raw.get("audio_duration_secs") or 0.0)
    if not duration:
        duration = max((w["e"] for w in words), default=0.0)

    return Transcript(
        language=raw.get("language_code"),
        language_probability=raw.get("language_probability"),
        text=raw.get("text", ""),
        words=words,
        events=events,
        speakers=speakers,
        engine=f"elevenlabs/{model_id}",
        duration_s=duration,
        cost_usd=duration / 3600.0 * STT_USD_PER_HOUR,
        raw=raw,
    )


def parse_multipart_mixed(content_type: str, body: bytes) -> list[tuple[dict[str, str], bytes]]:
    """Split a ``multipart/mixed`` body into ``(headers, payload)`` parts.

    Used by :meth:`ElevenLabs.compose_music_detailed`, whose response bundles a
    JSON metadata part and the audio part.
    """
    match = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not match:
        raise ElevenLabsError(200, f"no boundary in Content-Type: {content_type!r}")
    delimiter = b"--" + match.group(1).encode()

    parts: list[tuple[dict[str, str], bytes]] = []
    for chunk in body.split(delimiter):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        if not _:
            head, _, payload = chunk.partition(b"\n\n")
        headers: dict[str, str] = {}
        for line in head.decode("utf-8", "replace").splitlines():
            key, sep, value = line.partition(":")
            if sep:
                headers[key.strip().lower()] = value.strip()
        parts.append((headers, payload))
    return parts


class ElevenLabs:
    """Thin ElevenLabs REST client with retries and cost reporting."""

    def __init__(
        self,
        api_key: str,
        cost_callback: CostCallback | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabs api_key is required")
        self.api_key = api_key
        self.cost_callback = cost_callback
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ElevenLabs":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------- #

    def _report_cost(self, op: str, model: str | None, units: str, usd: float | None) -> None:
        if self.cost_callback is None or usd is None:
            return
        try:
            self.cost_callback(
                service="elevenlabs", op=op, model=model, units=units, usd=usd
            )
        except Exception:
            logger.exception("cost_callback failed for elevenlabs/%s", op)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: list[tuple[str, Any]] | None = None,
        data: dict[str, Any] | None = None,
        accept: str = "application/json",
        timeout: float | None = None,
        retries: int = 3,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = {"xi-api-key": self.api_key, "Accept": accept}
        delay = 1.0

        for attempt in range(retries + 1):
            # Files are consumed by the transport, so rebuild handles per attempt.
            try:
                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    params=params,
                    files=files,
                    data=data,
                    timeout=timeout or self.timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= retries:
                    raise ElevenLabsError(0, f"{type(exc).__name__}: {exc}") from exc
                self._sleep(delay, None, attempt, type(exc).__name__)
                delay *= 2
                continue

            if response.status_code in RETRY_STATUS and attempt < retries:
                self._sleep(
                    delay, response.headers.get("Retry-After"), attempt, f"HTTP {response.status_code}"
                )
                delay *= 2
                continue

            if response.status_code >= 300:
                raise ElevenLabsError(response.status_code, response.text)
            return response

        raise ElevenLabsError(0, f"request to {path} failed after {retries} retries")

    @staticmethod
    def _sleep(delay: float, retry_after: str | None, attempt: int, reason: str) -> None:
        wait = delay + random.uniform(0, 0.3)
        if retry_after:
            try:
                wait = max(wait, float(retry_after))
            except ValueError:
                pass
        logger.warning("elevenlabs retry %d after %s, sleeping %.1fs", attempt + 1, reason, wait)
        time.sleep(wait)

    @staticmethod
    def _file_tuple(field_name: str, path: Path | str) -> tuple[str, tuple[str, bytes, str]]:
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return (field_name, (p.name, p.read_bytes(), mime))

    def _request_audio_with_format_fallback(
        self,
        path: str,
        payload: dict[str, Any],
        output_format: str,
        *,
        timeout: float,
    ) -> httpx.Response:
        """POST expecting audio back, degrading the bitrate if the plan bars it.

        Free/payg plans reject ``mp3_44100_192`` on the speech endpoints with a
        403 ``output_format_not_allowed``; retry once at
        ``FALLBACK_OUTPUT_FORMAT`` rather than failing a whole render.
        """
        try:
            return self._request(
                "POST",
                path,
                json_body=payload,
                params={"output_format": output_format},
                accept="audio/mpeg",
                timeout=timeout,
            )
        except ElevenLabsError as exc:
            if exc.status != 403 or "output_format" not in exc.body:
                raise
            if output_format == FALLBACK_OUTPUT_FORMAT:
                raise
            logger.warning(
                "elevenlabs: output_format=%s not allowed on this plan, "
                "falling back to %s",
                output_format,
                FALLBACK_OUTPUT_FORMAT,
            )
            return self._request(
                "POST",
                path,
                json_body=payload,
                params={"output_format": FALLBACK_OUTPUT_FORMAT},
                accept="audio/mpeg",
                timeout=timeout,
            )

    # -- speech to text ----------------------------------------------------- #

    def transcribe(
        self,
        audio_path: Path | str,
        language_code: str | None = "pol",
        diarize: bool = True,
        tag_audio_events: bool = True,
        keyterms: Sequence[str] = (),
        no_verbatim: bool = False,
        model_id: str = "scribe_v2",
        *,
        num_speakers: int | None = None,
        additional_formats: list[dict[str, Any]] | None = None,
        timeout: float = 600.0,
    ) -> Transcript:
        """``POST /v1/speech-to-text`` (multipart) -> normalised transcript.

        Keep ``no_verbatim=False``: we want false starts and repeated takes in
        the transcript so take detection can pick the keeper.

        ``keyterms`` (max 1000) biases recognition toward place names.

        Files longer than ~25 minutes should ideally use ``webhook=true`` plus a
        ``webhook_id`` and be collected asynchronously; the pipeline transcribes
        per clip, so the synchronous path with a 600 s timeout is enough today.
        Very large media can also be passed as ``source_url``/``cloud_storage_url``
        instead of an upload.
        """
        path = Path(audio_path)
        form: dict[str, Any] = {
            "model_id": model_id,
            "timestamps_granularity": "word",
            "tag_audio_events": str(bool(tag_audio_events)).lower(),
            "diarize": str(bool(diarize)).lower(),
            "no_verbatim": str(bool(no_verbatim)).lower(),
        }
        if language_code:
            form["language_code"] = language_code
        if num_speakers:
            form["num_speakers"] = str(int(num_speakers))
        if keyterms:
            # Repeated multipart fields, NOT a JSON array: a JSON-encoded string
            # is rejected with 400 invalid_keyword ("Some keyword contains
            # invalid characters") - verified live 2026-09-04.
            form["keyterms"] = list(keyterms)
        if additional_formats:
            form["additional_formats"] = json.dumps(additional_formats)

        started = time.monotonic()
        response = self._request(
            "POST",
            "/v1/speech-to-text",
            data=form,
            files=[self._file_tuple("file", path)],
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        transcript = normalize_transcript(response.json(), model_id=model_id)

        logger.info(
            "elevenlabs stt model=%s audio=%.1fs words=%d events=%d cost=$%.4f duration=%.2fs",
            model_id,
            transcript.duration_s,
            len(transcript.words),
            len(transcript.events),
            transcript.cost_usd,
            elapsed,
        )
        self._report_cost("stt", model_id, f"{transcript.duration_s:.1f}s", transcript.cost_usd)
        return transcript

    # -- music -------------------------------------------------------------- #

    def compose_music(
        self,
        prompt: str,
        length_ms: int,
        *,
        model_id: str = "music_v2",
        force_instrumental: bool = True,
        generation_mode: str = "loop",
        seed: int | None = None,
        output_format: str = "mp3_44100_192",
        store_for_inpainting: bool = False,
        timeout: float = 600.0,
    ) -> MusicResult:
        """``POST /v1/music`` -> raw audio bytes (synchronous, no polling).

        ``length_ms`` must be 3000..600000.  ``generation_mode`` is one of
        ``track | loop | ambience | video_to_music``; ``loop`` gives seamless
        beds.  Cost is $0.15 per minute of generated audio.
        """
        if not 3000 <= int(length_ms) <= 600_000:
            raise ValueError("length_ms must be between 3000 and 600000")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "music_length_ms": int(length_ms),
            "model_id": model_id,
            "force_instrumental": bool(force_instrumental),
            "generation_mode": generation_mode,
            "store_for_inpainting": bool(store_for_inpainting),
        }
        if seed is not None:
            payload["seed"] = int(seed)

        started = time.monotonic()
        response = self._request(
            "POST",
            "/v1/music",
            json_body=payload,
            params={"output_format": output_format},
            accept="audio/mpeg",
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        cost = int(length_ms) / 60_000.0 * MUSIC_USD_PER_MINUTE

        logger.info(
            "elevenlabs music model=%s length=%dms mode=%s bytes=%d cost=$%.4f duration=%.2fs",
            model_id,
            length_ms,
            generation_mode,
            len(response.content),
            cost,
            elapsed,
        )
        self._report_cost("music", model_id, f"{length_ms / 1000:.1f}s", cost)
        return MusicResult(
            audio=response.content,
            song_id=response.headers.get("song-id"),
            meta={
                "prompt": prompt,
                "music_length_ms": int(length_ms),
                "model_id": model_id,
                "generation_mode": generation_mode,
                "force_instrumental": bool(force_instrumental),
                "seed": seed,
                "output_format": output_format,
                "cost_usd": cost,
            },
        )

    def music_plan(
        self, prompt: str, length_ms: int, *, model_id: str = "music_v2"
    ) -> dict[str, Any]:
        """``POST /v1/music/plan`` -> composition plan dict.

        The plan is the ``MusicPrompt`` shape: ``positive_global_styles``,
        ``negative_global_styles`` and ``sections[{section_name,
        positive_local_styles, negative_local_styles, duration_ms, lines}]``.
        Planning itself is not billed as generated audio.
        """
        response = self._request(
            "POST",
            "/v1/music/plan",
            json_body={
                "prompt": prompt,
                "music_length_ms": int(length_ms),
                "model_id": model_id,
            },
        )
        return response.json()

    def compose_music_detailed(
        self,
        plan: dict[str, Any],
        *,
        model_id: str = "music_v2",
        output_format: str = "mp3_44100_192",
        force_instrumental: bool = True,
        with_timestamps: bool = False,
        timeout: float = 600.0,
    ) -> MusicResult:
        """``POST /v1/music/detailed`` with a composition plan.

        The response is ``multipart/mixed``: a JSON part (composition plan +
        metadata, including ``song_id``) and an audio part.  ``prompt`` and
        ``composition_plan`` are mutually exclusive, so this method only ever
        sends the plan.
        """
        sections = plan.get("sections") or []
        length_ms = sum(int(s.get("duration_ms") or 0) for s in sections if isinstance(s, dict))
        payload: dict[str, Any] = {
            "composition_plan": plan,
            "model_id": model_id,
            "force_instrumental": bool(force_instrumental),
            "with_timestamps": bool(with_timestamps),
        }
        response = self._request(
            "POST",
            "/v1/music/detailed",
            json_body=payload,
            params={"output_format": output_format},
            accept="multipart/mixed",
            timeout=timeout,
        )

        content_type = response.headers.get("content-type", "")
        audio = b""
        meta: dict[str, Any] = {}
        if "multipart" in content_type:
            for headers, body in parse_multipart_mixed(content_type, response.content):
                part_type = headers.get("content-type", "")
                if "json" in part_type:
                    try:
                        meta = json.loads(body.decode("utf-8", "replace"))
                    except ValueError:
                        meta = {"raw": body.decode("utf-8", "replace")[:2000]}
                elif "audio" in part_type or not audio:
                    audio = body
        else:  # some deployments answer with plain audio
            audio = response.content

        if not length_ms:
            length_ms = int(meta.get("music_length_ms") or 0)
        cost = length_ms / 60_000.0 * MUSIC_USD_PER_MINUTE if length_ms else 0.0
        logger.info(
            "elevenlabs music/detailed model=%s length=%dms bytes=%d cost=$%.4f",
            model_id,
            length_ms,
            len(audio),
            cost,
        )
        self._report_cost("music", model_id, f"{length_ms / 1000:.1f}s", cost or None)
        return MusicResult(
            audio=audio,
            song_id=response.headers.get("song-id") or meta.get("song_id"),
            meta={**meta, "cost_usd": cost, "model_id": model_id},
        )

    # -- text to speech ----------------------------------------------------- #

    def tts(
        self,
        text: str,
        voice_id: str,
        *,
        model_id: str = "eleven_v3",
        language_code: str | None = "pl",
        output_format: str = "mp3_44100_192",
        voice_settings: dict[str, Any] | None = None,
        previous_text: str | None = None,
        next_text: str | None = None,
        seed: int | None = None,
        timeout: float = 300.0,
    ) -> bytes:
        """``POST /v1/text-to-speech/{voice_id}`` -> audio bytes.

        ``previous_text`` / ``next_text`` preserve prosody when a long narration
        is stitched from chunks.  ``voice_settings`` accepts
        ``{stability, similarity_boost, style, speed, use_speaker_boost}``.
        """
        payload: dict[str, Any] = {"text": text, "model_id": model_id}
        if language_code:
            payload["language_code"] = language_code
        if voice_settings:
            payload["voice_settings"] = voice_settings
        if previous_text:
            payload["previous_text"] = previous_text
        if next_text:
            payload["next_text"] = next_text
        if seed is not None:
            payload["seed"] = int(seed)

        started = time.monotonic()
        response = self._request_audio_with_format_fallback(
            f"/v1/text-to-speech/{voice_id}",
            payload,
            output_format,
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        cost = len(text) / 1000.0 * TTS_USD_PER_1K_CHARS.get(model_id, 0.10)

        logger.info(
            "elevenlabs tts model=%s voice=%s chars=%d bytes=%d cost=$%.4f duration=%.2fs",
            model_id,
            voice_id,
            len(text),
            len(response.content),
            cost,
            elapsed,
        )
        self._report_cost("tts", model_id, f"{len(text)} chars", cost)
        return response.content

    def list_voices(self) -> list[dict[str, Any]]:
        """``GET /v1/voices`` -> voice dicts (``voice_id``, ``name``, ...)."""
        return list(self._request("GET", "/v1/voices").json().get("voices") or [])

    def clone_voice(
        self,
        name: str,
        files: Sequence[Path | str],
        description: str = "",
        remove_background_noise: bool = True,
        *,
        labels: dict[str, str] | None = None,
        timeout: float = 600.0,
    ) -> str:
        """``POST /v1/voices/add`` (Instant Voice Clone) -> ``voice_id``.

        1-2 minutes of clean single-speaker audio is the sweet spot; more than
        ~3 minutes can hurt.  Requires a plan with IVC enabled - check
        ``subscription()["can_use_instant_voice_cloning"]`` first.
        """
        form: dict[str, Any] = {
            "name": name,
            "remove_background_noise": str(bool(remove_background_noise)).lower(),
        }
        if description:
            form["description"] = description
        if labels:
            form["labels"] = json.dumps(labels, ensure_ascii=False)

        response = self._request(
            "POST",
            "/v1/voices/add",
            data=form,
            files=[self._file_tuple("files", f) for f in files],
            timeout=timeout,
        )
        body = response.json()
        voice_id = body.get("voice_id") or ""
        logger.info("elevenlabs clone_voice name=%s voice_id=%s", name, voice_id)
        return voice_id

    # -- audio tools -------------------------------------------------------- #

    def isolate_audio(
        self,
        path: Path | str,
        *,
        duration_s: float | None = None,
        timeout: float = 600.0,
    ) -> bytes:
        """``POST /v1/audio-isolation`` -> cleaned speech audio bytes.

        This is a speech-from-noise isolator, not a music stem splitter: run
        Demucs first for music beds, then this for cleanup.  $0.12/min - pass
        ``duration_s`` (from ffprobe) to get the cost into the ledger.
        """
        started = time.monotonic()
        response = self._request(
            "POST",
            "/v1/audio-isolation",
            files=[self._file_tuple("audio", path)],
            accept="audio/mpeg",
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        cost = None if duration_s is None else duration_s / 60.0 * ISOLATION_USD_PER_MINUTE
        logger.info(
            "elevenlabs isolate bytes=%d cost=%s duration=%.2fs",
            len(response.content),
            f"${cost:.4f}" if cost is not None else "unknown",
            elapsed,
        )
        self._report_cost(
            "isolate", None, f"{duration_s:.1f}s" if duration_s else "unknown", cost
        )
        return response.content

    def sound_effect(
        self,
        text: str,
        duration_seconds: float | None = None,
        prompt_influence: float = 0.3,
        *,
        loop: bool = False,
        model_id: str = "eleven_text_to_sound_v2",
        output_format: str = "mp3_44100_192",
        timeout: float = 300.0,
    ) -> bytes:
        """``POST /v1/sound-generation`` -> SFX audio bytes.

        ``duration_seconds`` is 0.5..30 (omit to let the model choose).
        """
        payload: dict[str, Any] = {
            "text": text,
            "prompt_influence": prompt_influence,
            "loop": bool(loop),
            "model_id": model_id,
        }
        if duration_seconds is not None:
            payload["duration_seconds"] = float(duration_seconds)

        response = self._request_audio_with_format_fallback(
            "/v1/sound-generation", payload, output_format, timeout=timeout
        )
        cost = (
            None
            if duration_seconds is None
            else float(duration_seconds) / 60.0 * SFX_USD_PER_MINUTE
        )
        logger.info(
            "elevenlabs sfx model=%s seconds=%s bytes=%d cost=%s",
            model_id,
            duration_seconds,
            len(response.content),
            f"${cost:.4f}" if cost is not None else "unknown",
        )
        self._report_cost(
            "sfx", model_id, f"{duration_seconds}s" if duration_seconds else "auto", cost
        )
        return response.content

    def subscription(self) -> dict[str, Any]:
        """``GET /v1/user/subscription`` -> plan, character quota, voice limits."""
        body = self._request("GET", "/v1/user/subscription").json()
        logger.info(
            "elevenlabs subscription tier=%s characters=%s/%s ivc=%s",
            body.get("tier"),
            body.get("character_count"),
            body.get("character_limit"),
            body.get("can_use_instant_voice_cloning"),
        )
        return body


__all__ = [
    "ElevenLabs",
    "ElevenLabsError",
    "Transcript",
    "MusicResult",
    "normalize_transcript",
    "parse_event_type",
    "parse_multipart_mixed",
    "STT_USD_PER_HOUR",
    "MUSIC_USD_PER_MINUTE",
    "TTS_USD_PER_1K_CHARS",
    "FALLBACK_OUTPUT_FORMAT",
]
