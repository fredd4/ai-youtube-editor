"""OpenRouter chat client (text + vision) with robust JSON extraction.

Self-contained: this module never imports ytedit.config / ytedit.project.  The
API key and an optional ``cost_callback`` are passed in explicitly so the client
can be reused from tests, scripts and the pipeline alike.

Verified live against OpenRouter on 2026-09-04:

* ``POST /api/v1/chat/completions`` returns ``usage.cost`` (actual USD spend)
  only when the request body carries ``{"usage": {"include": true}}``.  We always
  send it, so :attr:`ChatResult.cost_usd` is real money, not an estimate,
  whenever the provider reports it.
* ``response_format={"type": "json_object"}`` is *not* honoured by every
  provider: Claude Haiku 4.5 via Amazon Bedrock still wrapped its answer in a
  ```json fence.  Fence stripping / balanced-block extraction is therefore
  mandatory, not a nicety.
* ``POST /api/v1/audio/transcriptions`` accepts ``openai/whisper-large-v3``
  even though that id is absent from ``GET /api/v1/models`` (which only lists
  chat models).  Response shape is ``{text, task, language, duration,
  words: [{word, start, end}], usage: {seconds, cost}}``.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx

logger = logging.getLogger("ytedit.ai.openrouter")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# $ per 1M tokens (prompt, completion).  Live OpenRouter prices confirmed
# 2026-09-04 against GET /api/v1/models; used only as a fallback when the
# response does not carry usage.cost.  ":batch" variants are 50% off and are
# handled by price_for_model(); ":free" is zero.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-fable-5.1": (10.0, 50.0),
    "anthropic/claude-opus-5": (5.0, 25.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "google/gemini-3.1-pro-preview": (2.0, 12.0),
    "google/gemini-3.8-flash": (0.75, 3.75),
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "openai/gpt-5.6-terra": (2.0, 12.0),
    "openai/gpt-5.6-luna": (0.20, 1.20),
}

# $ per hour of audio, fallback only (the endpoint reports usage.cost).
AUDIO_PRICES_PER_HOUR: dict[str, float] = {
    "openai/whisper-large-v3": 0.027,
    "openai/whisper-large-v3-turbo": 0.027,
}

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}

CostCallback = Callable[..., None]


class OpenRouterError(RuntimeError):
    """Any non-recoverable OpenRouter failure (HTTP, transport or parsing)."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body[:2000]


# --------------------------------------------------------------------------- #
# JSON extraction helpers (module level so they can be unit-tested directly)
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)(?:```|\Z)", re.DOTALL)


def message_text(message: dict[str, Any]) -> str:
    """Pull plain text out of an OpenRouter ``choices[i].message``.

    Handles three shapes seen in the wild:

    * ``content`` is a string (the normal case);
    * ``content`` is ``None`` because a thinking model put its answer in
      ``reasoning`` or in a ``parts`` list (Gemini thinking mode);
    * ``content`` is a list of content parts (``{"type": "text", "text": ...}``).
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        joined = " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        ).strip()
        if joined:
            return joined
    if content:  # non-empty, non-str, non-list -> stringify
        return str(content)

    # content missing/empty: try parts, then reasoning.
    parts = message.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                return str(part["text"])
        joined = " ".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        ).strip()
        if joined:
            return joined
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, dict):
        return str(reasoning.get("text") or "")
    return ""


def strip_code_fences(text: str) -> str:
    """Return the body of the first ``` fenced block, or the text unchanged."""
    match = _FENCE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def find_json_block(text: str) -> str | None:
    """Return the first balanced ``{...}`` / ``[...]`` substring, or None.

    String literals and escapes are respected so that braces inside strings do
    not unbalance the scan.  Trailing prose after the block is ignored.
    """
    for start, opener in enumerate(text):
        if opener not in "{[":
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # Unbalanced from here; fall through and try the next opener.
    return None


def extract_json(text: str) -> Any:
    """Parse JSON out of a model answer that may be fenced or padded with prose.

    Raises ``json.JSONDecodeError`` if nothing parses.
    """
    if not text or not text.strip():
        raise json.JSONDecodeError("empty model output", text or "", 0)

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    unfenced = strip_code_fences(stripped)
    if unfenced != stripped:
        candidates.append(unfenced)
    block = find_json_block(unfenced)
    if block:
        candidates.append(block)
    if unfenced != stripped:
        block2 = find_json_block(stripped)
        if block2 and block2 not in candidates:
            candidates.append(block2)

    error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:  # keep the last failure for context
            error = exc
    raise error or json.JSONDecodeError("no JSON found", text, 0)


def price_for_model(model: str) -> tuple[float, float] | None:
    """($/M prompt, $/M completion) for a model id, honouring :batch and :free."""
    base, _, suffix = model.partition(":")
    price = PRICES.get(base)
    if price is None:
        return None
    if suffix == "free":
        return (0.0, 0.0)
    if suffix == "batch":
        return (price[0] / 2, price[1] / 2)
    return price


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = price_for_model(model)
    if price is None:
        return 0.0
    return prompt_tokens / 1e6 * price[0] + completion_tokens / 1e6 * price[1]


def data_url(image: Path | str | bytes) -> str:
    """Build a ``data:`` URL for a JPEG/PNG frame given a path or raw bytes."""
    if isinstance(image, (str, Path)):
        raw = Path(image).read_bytes()
    else:
        raw = bytes(image)
    mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def system(text: str) -> dict[str, Any]:
    """Build a system message."""
    return {"role": "system", "content": text}


def user_message(
    text: str,
    images: Sequence[Path | str | bytes] = (),
    detail: str = "low",
    labels: Sequence[str | float] | None = None,
) -> dict[str, Any]:
    """Build a (multimodal) user message.

    ``labels`` interleaves a short text part before each image.  A float label
    is rendered as ``"Frame i (t=12.4s):"``; a string label is used verbatim.
    This mirrors the amazonia-studio pattern that made frame-indexed answers
    reliable.
    """
    if not images:
        return {"role": "user", "content": text}

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for i, image in enumerate(images):
        if labels is not None and i < len(labels):
            label = labels[i]
            label_text = label if isinstance(label, str) else f"Frame {i} (t={float(label):.1f}s):"
            content.append({"type": "text", "text": label_text})
        content.append(
            {"type": "image_url", "image_url": {"url": data_url(image), "detail": detail}}
        )
    return {"role": "user", "content": content}


@dataclass(slots=True)
class ChatResult:
    text: str
    json: Any | None
    usage: dict[str, Any]
    cost_usd: float
    model: str
    raw: dict[str, Any]
    duration_s: float = 0.0
    cost_estimated: bool = False

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens") or 0)

    @property
    def finish_reason(self) -> str:
        """Why the model stopped: ``"stop"``, ``"length"`` (hit max_tokens), ...

        OpenRouter normalizes the provider's own value into ``finish_reason`` and
        keeps the original in ``native_finish_reason``; either may be missing.
        """
        choices = self.raw.get("choices") or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        return str(first.get("finish_reason") or first.get("native_finish_reason") or "")


class OpenRouter:
    """Minimal, dependency-light OpenRouter client."""

    def __init__(
        self,
        api_key: str,
        cost_callback: CostCallback | None = None,
        referer: str = "http://localhost:8765",
        title: str = "ytedit",
        timeout: float = 180.0,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter api_key is required")
        self.api_key = api_key
        self.cost_callback = cost_callback
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": referer,
            "X-Title": title,
            "Content-Type": "application/json",
        }
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._models_cache: list[dict[str, Any]] | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenRouter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------- #

    def _report_cost(self, op: str, model: str, units: str, usd: float) -> None:
        if self.cost_callback is None:
            return
        try:
            self.cost_callback(
                service="openrouter", op=op, model=model, units=units, usd=usd
            )
        except Exception:  # a broken ledger must never kill a pipeline stage
            logger.exception("cost_callback failed for openrouter/%s", op)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                response = self._client.request(
                    method, url, headers=self._headers, json=payload, timeout=self.timeout
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= retries:
                    raise OpenRouterError(f"{type(exc).__name__}: {exc}") from exc
                self._sleep(delay, None, attempt, f"{type(exc).__name__}")
                delay *= 2
                continue

            if response.status_code in RETRY_STATUS and attempt < retries:
                self._sleep(delay, response.headers.get("Retry-After"), attempt, f"HTTP {response.status_code}")
                delay *= 2
                continue

            if response.status_code >= 400:
                raise OpenRouterError(
                    f"OpenRouter HTTP {response.status_code} for {path}",
                    status=response.status_code,
                    body=response.text,
                )

            try:
                body = response.json()
            except ValueError as exc:
                raise OpenRouterError(
                    "OpenRouter returned non-JSON body", status=response.status_code, body=response.text
                ) from exc

            # OpenRouter can return HTTP 200 with an {"error": {...}} envelope.
            if isinstance(body, dict) and body.get("error") and not body.get("choices"):
                err = body["error"]
                code = err.get("code") if isinstance(err, dict) else None
                message = err.get("message") if isinstance(err, dict) else str(err)
                raise OpenRouterError(
                    f"OpenRouter error: {message}",
                    status=int(code) if isinstance(code, int) else None,
                    body=json.dumps(body),
                )
            return body

        raise OpenRouterError(f"OpenRouter request failed after {retries} retries: {last_error}")

    @staticmethod
    def _sleep(delay: float, retry_after: str | None, attempt: int, reason: str) -> None:
        wait = delay + random.uniform(0, 0.3)
        if retry_after:
            try:
                wait = max(wait, float(retry_after))
            except ValueError:  # HTTP-date form; fall back to backoff
                pass
        logger.warning("openrouter retry %d after %s, sleeping %.1fs", attempt + 1, reason, wait)
        time.sleep(wait)

    # -- chat --------------------------------------------------------------- #

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        """One chat completion.

        ``json_schema`` takes precedence over ``json_mode``; it is sent as
        ``response_format={"type": "json_schema", "json_schema": {...}}``.  Pass
        either a full ``{"name": ..., "schema": ...}`` wrapper or a bare JSON
        Schema (it is wrapped for you).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Required for usage.cost to appear in the response (verified live).
            "usage": {"include": True},
        }
        if json_schema is not None:
            wrapper = (
                json_schema
                if "schema" in json_schema
                else {"name": "response", "strict": True, "schema": json_schema}
            )
            payload["response_format"] = {"type": "json_schema", "json_schema": wrapper}
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        if extra_body:
            payload.update(extra_body)

        started = time.monotonic()
        body = self._request("POST", "/chat/completions", payload=payload, retries=retries)
        duration = time.monotonic() - started

        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenRouter returned no choices", body=json.dumps(body)[:2000])
        text = message_text(choices[0].get("message") or {})

        usage = body.get("usage") or {}
        cost = usage.get("cost")
        estimated = cost is None
        if estimated:
            cost = estimate_cost(
                model,
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
        cost = float(cost or 0.0)

        parsed: Any | None = None
        if json_mode or json_schema is not None:
            try:
                parsed = extract_json(text)
            except json.JSONDecodeError:
                parsed = None

        logger.info(
            "openrouter chat model=%s tokens=%s/%s cost=$%.6f%s duration=%.2fs",
            body.get("model", model),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            cost,
            " (est)" if estimated else "",
            duration,
        )
        self._report_cost(
            "chat",
            model,
            f"{usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} tok",
            cost,
        )
        return ChatResult(
            text=text,
            json=parsed,
            usage=usage,
            cost_usd=cost,
            model=str(body.get("model") or model),
            raw=body,
            duration_s=duration,
            cost_estimated=estimated,
        )

    def ask_json(
        self,
        model: str,
        system_prompt: str,
        user: str,
        schema_hint: str,
        images: Sequence[Path | str | bytes] = (),
        *,
        labels: Sequence[str | float] | None = None,
        detail: str = "low",
        temperature: float = 0.2,
        max_tokens: int = 8000,
        retries: int = 3,
    ) -> Any:
        """Ask for JSON and return the parsed dict/list.

        Strategy: ``response_format=json_object`` first; if the provider rejects
        it, retry once without.  The answer is then run through
        :func:`extract_json` (fences, prose, thinking-mode ``reasoning``).  On a
        parse failure the whole exchange is retried once with an explicit
        "Return ONLY valid JSON" nudge appended.  Raises :class:`OpenRouterError`
        after that.
        """
        full_system = (
            f"{system_prompt}\n\nReturn ONLY valid JSON matching this shape "
            f"(no prose, no markdown fences):\n{schema_hint}"
        )
        base_messages = [
            system(full_system),
            user_message(user, images=images, detail=detail, labels=labels),
        ]

        use_json_mode = True
        last_text = ""
        for attempt in range(2):
            messages = list(base_messages)
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your previous answer was not valid JSON. "
                        "Return ONLY valid JSON, nothing else.",
                    }
                )
            try:
                result = self.chat(
                    model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=use_json_mode,
                    retries=retries,
                )
            except OpenRouterError as exc:
                # Some providers 400/404/422 on response_format; drop it once.
                if use_json_mode and exc.status in (400, 404, 422):
                    logger.warning(
                        "openrouter: %s rejected response_format, retrying without it", model
                    )
                    use_json_mode = False
                    result = self.chat(
                        model,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=False,
                        retries=retries,
                    )
                else:
                    raise

            last_text = result.text
            if result.json is not None:
                return result.json
            try:
                return extract_json(result.text)
            except json.JSONDecodeError as exc:
                logger.warning("openrouter: JSON extraction failed (%s), retrying", exc)

        raise OpenRouterError(
            f"model {model} did not return parseable JSON after 2 attempts",
            body=last_text,
        )

    # -- misc --------------------------------------------------------------- #

    def list_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """GET /api/v1/models, cached for the lifetime of the client.

        Note: this lists *chat* models only.  Transcription-only ids such as
        ``openai/whisper-large-v3`` are absent but still work on
        ``/audio/transcriptions`` (verified 2026-09-04).
        """
        if self._models_cache is None or refresh:
            body = self._request("GET", "/models", retries=2)
            self._models_cache = list(body.get("data") or [])
        return self._models_cache

    def model_ids(self) -> set[str]:
        return {m.get("id", "") for m in self.list_models()}

    def transcribe_audio(
        self,
        path: Path | str,
        model: str = "openai/whisper-large-v3",
        language: str | None = None,
        *,
        retries: int = 3,
    ) -> dict[str, Any]:
        """Best-effort STT fallback via ``POST /api/v1/audio/transcriptions``.

        ElevenLabs scribe_v2 is the primary engine (see ``elevenlabs.py``); this
        exists so a run can continue when ElevenLabs is down or out of credit.
        Word timestamps only come back from OpenAI-compatible providers, and the
        normalised words carry no confidence (``p``) or speaker, unlike Scribe.

        Returns ``{"text", "words": [{"t", "s", "e"}], "language", "duration",
        "cost_usd", "engine", "raw"}``.
        """
        audio_path = Path(path)
        raw_bytes = audio_path.read_bytes()
        fmt = audio_path.suffix.lstrip(".").lower() or "mp3"
        payload: dict[str, Any] = {
            "model": model,
            "input_audio": {"data": base64.b64encode(raw_bytes).decode(), "format": fmt},
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
        if language:
            payload["language"] = language

        started = time.monotonic()
        body = self._request("POST", "/audio/transcriptions", payload=payload, retries=retries)
        duration_call = time.monotonic() - started

        words = [
            {
                "t": w.get("word") or w.get("text") or "",
                "s": float(w.get("start", 0.0)),
                "e": float(w.get("end", 0.0)),
            }
            for w in (body.get("words") or [])
            if isinstance(w, dict)
        ]
        usage = body.get("usage") or {}
        audio_seconds = float(usage.get("seconds") or body.get("duration") or 0.0)
        cost = usage.get("cost")
        if cost is None:
            per_hour = AUDIO_PRICES_PER_HOUR.get(model.partition(":")[0], 0.0)
            cost = audio_seconds / 3600.0 * per_hour
        cost = float(cost or 0.0)

        logger.info(
            "openrouter transcribe model=%s audio=%.1fs cost=$%.6f duration=%.2fs",
            model,
            audio_seconds,
            cost,
            duration_call,
        )
        self._report_cost("transcribe", model, f"{audio_seconds:.1f}s", cost)
        return {
            "text": body.get("text", ""),
            "words": words,
            "language": body.get("language") or language,
            "duration": float(body.get("duration") or audio_seconds),
            "cost_usd": cost,
            "engine": f"openrouter/{model}",
            "raw": body,
        }


__all__ = [
    "OpenRouter",
    "OpenRouterError",
    "ChatResult",
    "PRICES",
    "AUDIO_PRICES_PER_HOUR",
    "system",
    "user_message",
    "data_url",
    "extract_json",
    "find_json_block",
    "strip_code_fences",
    "message_text",
    "price_for_model",
    "estimate_cost",
]
