"""fal.ai client: uploads, queue jobs, thumbnails, image-to-video, upscaling.

Self-contained: the key is injected and pushed into ``os.environ["FAL_KEY"]``
right before ``fal_client`` is imported (the SDK reads it at import/call time),
so nothing from ytedit.config is needed.  ``fal_client`` is imported lazily so
the package still imports when the SDK or the key is missing.

Argument names for every high-level method were confirmed against the live
OpenAPI schema at
``https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<id>``
(fetched 2026-09-04) - see each method's docstring for the exact field list.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

logger = logging.getLogger("ytedit.ai.fal")

# --- app ids ---------------------------------------------------------------- #
APP_NANO_BANANA_PRO = "fal-ai/nano-banana-pro"
APP_NANO_BANANA_PRO_EDIT = "fal-ai/nano-banana-pro/edit"
APP_SEEDANCE_I2V = "bytedance/seedance-2.5/image-to-video"
APP_TOPAZ_UPSCALE = "fal-ai/topaz/upscale/video"

# --- prices ----------------------------------------------------------------- #
NANO_BANANA_USD_PER_IMAGE = 0.15  # 2x at 4K
# $ per generated second, by resolution.  480p/720p from the fal pricing page;
# 1080p is an extrapolation and is flagged as an estimate in the log line.
SEEDANCE_USD_PER_SECOND = {"480p": 0.22, "720p": 0.47, "1080p": 0.94}
# $ per second of *output* video, by output height bucket.
TOPAZ_USD_PER_SECOND = {720: 0.01, 1080: 0.02, 99999: 0.08}

_STATUS_MAP = {
    "Queued": "queued",
    "InProgress": "in_progress",
    "InQueue": "queued",
    "Completed": "completed",
    "Failed": "failed",
}

CostCallback = Callable[..., None]


class FalError(RuntimeError):
    """Any fal.ai failure surfaced by this client."""


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def map_status(status: Any) -> str:
    """Map a ``fal_client`` status object (or its class name) to a plain string.

    The SDK returns instances of ``Queued`` / ``InProgress`` / ``Completed``;
    there is no shared enum, so class-name detection is the documented approach.
    """
    name = status if isinstance(status, str) else type(status).__name__
    return _STATUS_MAP.get(name, _camel_to_snake(name))


class Fal:
    """fal.ai queue client with cost estimates and logging."""

    def __init__(
        self,
        api_key: str,
        cost_callback: CostCallback | None = None,
        *,
        download_timeout: float = 600.0,
    ) -> None:
        if not api_key:
            raise ValueError("fal api_key is required")
        self.api_key = api_key
        self.cost_callback = cost_callback
        self.download_timeout = download_timeout
        self._sdk: Any | None = None

    # -- low level ---------------------------------------------------------- #

    @property
    def sdk(self) -> Any:
        """Lazily import ``fal_client`` with FAL_KEY set in the environment."""
        if self._sdk is None:
            os.environ["FAL_KEY"] = self.api_key
            try:
                import fal_client  # noqa: PLC0415 - deliberate lazy import
            except ImportError as exc:  # pragma: no cover - env-specific
                raise FalError("fal-client is not installed (pip install fal-client)") from exc
            self._sdk = fal_client
        return self._sdk

    def _report_cost(self, op: str, model: str, units: str, usd: float) -> None:
        if self.cost_callback is None:
            return
        try:
            self.cost_callback(service="fal", op=op, model=model, units=units, usd=usd)
        except Exception:
            logger.exception("cost_callback failed for fal/%s", op)

    # -- primitives --------------------------------------------------------- #

    def upload(self, path: Path | str) -> str:
        """Upload a local file to fal storage and return its URL.  Free."""
        p = Path(path)
        started = time.monotonic()
        url = self.sdk.upload_file(str(p))
        logger.info(
            "fal upload %s (%d bytes) -> %s in %.2fs",
            p.name,
            p.stat().st_size,
            url,
            time.monotonic() - started,
        )
        return url

    def run(self, app: str, arguments: dict[str, Any], with_logs: bool = True) -> dict[str, Any]:
        """Blocking ``subscribe`` call; queue logs are forwarded to the logger."""

        def on_queue_update(update: Any) -> None:
            for entry in getattr(update, "logs", None) or []:
                message = entry.get("message") if isinstance(entry, dict) else str(entry)
                if message:
                    logger.info("[fal %s] %s", app, message)

        started = time.monotonic()
        result = self.sdk.subscribe(
            app,
            arguments=arguments,
            with_logs=with_logs,
            on_queue_update=on_queue_update if with_logs else None,
        )
        logger.info("fal run %s finished in %.1fs", app, time.monotonic() - started)
        return dict(result or {})

    def submit(self, app: str, arguments: dict[str, Any]) -> str:
        """Enqueue a job and return its ``request_id`` (persist this)."""
        handle = self.sdk.submit(app, arguments=arguments)
        request_id = handle.request_id
        logger.info("fal submit %s -> request_id=%s", app, request_id)
        return request_id

    def status(self, app: str, request_id: str, *, with_logs: bool = False) -> str:
        """``"queued" | "in_progress" | "completed" | "failed"``."""
        raw = self.sdk.status(app, request_id, with_logs=with_logs)
        state = map_status(raw)
        logger.debug("fal status %s %s -> %s", app, request_id, state)
        return state

    def result(self, app: str, request_id: str) -> dict[str, Any]:
        """Fetch the finished result payload for a submitted request."""
        return dict(self.sdk.result(app, request_id) or {})

    def wait(
        self,
        app: str,
        request_id: str,
        *,
        poll_s: float = 5.0,
        timeout_s: float = 1800.0,
    ) -> dict[str, Any]:
        """Poll ``status`` until completion, then return ``result``."""
        deadline = time.monotonic() + timeout_s
        while True:
            state = self.status(app, request_id)
            if state == "completed":
                return self.result(app, request_id)
            if state == "failed":
                raise FalError(f"fal job {app}/{request_id} failed")
            if time.monotonic() > deadline:
                raise FalError(f"fal job {app}/{request_id} timed out after {timeout_s}s")
            time.sleep(poll_s)

    def download(self, url: str, dest: Path | str) -> Path:
        """Stream a result URL to disk."""
        out = Path(dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=self.download_timeout, follow_redirects=True) as r:
            r.raise_for_status()
            with out.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 16):
                    fh.write(chunk)
        logger.info("fal download %s -> %s (%d bytes)", url, out, out.stat().st_size)
        return out

    # -- high level: images ------------------------------------------------- #

    def thumbnail_edit(
        self,
        prompt: str,
        image_paths: Sequence[Path | str],
        n: int = 4,
        aspect_ratio: str = "16:9",
        resolution: str = "2K",
        *,
        out_dir: Path | str | None = None,
        output_format: str = "jpeg",
        seed: int | None = None,
    ) -> list[Path]:
        """``fal-ai/nano-banana-pro/edit`` -> generated thumbnail candidates.

        Schema (confirmed 2026-09-04): required ``prompt``, ``image_urls``
        (array, up to 14); optional ``num_images`` (default 1), ``aspect_ratio``
        (``auto|21:9|16:9|3:2|4:3|5:4|1:1|4:5|3:4|2:3|9:16``, default ``auto``),
        ``resolution`` (``1K|2K|4K``, default ``1K``), ``output_format``
        (``jpeg|png|webp``, default ``png``), ``seed``, ``system_prompt``,
        ``enable_web_search``, ``safety_tolerance``, ``sync_mode``,
        ``limit_generations``.  Output: ``{"images": [{"url", ...}],
        "description": str}``.

        $0.15 per image, doubled at 4K.
        """
        if not image_paths:
            raise ValueError("thumbnail_edit needs at least one reference image")
        image_urls = [self.upload(p) for p in image_paths]
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": int(n),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
        }
        if seed is not None:
            arguments["seed"] = int(seed)

        cost = int(n) * NANO_BANANA_USD_PER_IMAGE * (2 if resolution.upper() == "4K" else 1)
        logger.info(
            "fal thumbnail_edit n=%d resolution=%s estimated cost=$%.2f",
            n,
            resolution,
            cost,
        )
        result = self.run(APP_NANO_BANANA_PRO_EDIT, arguments)
        self._report_cost("thumbnail", APP_NANO_BANANA_PRO_EDIT, f"{n} images", cost)
        return self._save_images(result, out_dir, "thumb", output_format)

    def text_to_image(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        *,
        n: int = 1,
        resolution: str = "2K",
        out_dir: Path | str | None = None,
        output_format: str = "jpeg",
        seed: int | None = None,
    ) -> list[Path]:
        """``fal-ai/nano-banana-pro`` text-to-image.

        Schema (confirmed 2026-09-04): required ``prompt``; optional
        ``num_images`` (default 1), ``aspect_ratio`` (default ``1:1``),
        ``resolution`` (``1K|2K|4K``, default ``1K``), ``output_format``
        (default ``png``), ``seed``, ``system_prompt``, ``enable_web_search``,
        ``safety_tolerance``, ``sync_mode``, ``limit_generations``.
        Same $0.15/image pricing as the edit endpoint.
        """
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "num_images": int(n),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
        }
        if seed is not None:
            arguments["seed"] = int(seed)

        cost = int(n) * NANO_BANANA_USD_PER_IMAGE * (2 if resolution.upper() == "4K" else 1)
        logger.info("fal text_to_image n=%d estimated cost=$%.2f", n, cost)
        result = self.run(APP_NANO_BANANA_PRO, arguments)
        self._report_cost("text_to_image", APP_NANO_BANANA_PRO, f"{n} images", cost)
        return self._save_images(result, out_dir, "image", output_format)

    def _save_images(
        self,
        result: dict[str, Any],
        out_dir: Path | str | None,
        stem: str,
        output_format: str,
    ) -> list[Path]:
        directory = Path(out_dir) if out_dir else Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "jpg" if output_format == "jpeg" else output_format
        paths: list[Path] = []
        for i, image in enumerate(result.get("images") or []):
            url = image.get("url") if isinstance(image, dict) else str(image)
            if not url:
                continue
            paths.append(self.download(url, directory / f"{stem}_{i:02d}.{suffix}"))
        if not paths:
            raise FalError(f"fal returned no images: {str(result)[:500]}")
        return paths

    # -- high level: video -------------------------------------------------- #

    def seedance_i2v(
        self,
        image_path: Path | str,
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        generate_audio: bool = False,
        *,
        confirm: bool = False,
        out_path: Path | str | None = None,
        aspect_ratio: str = "auto",
        end_image_path: Path | str | None = None,
        seed: int | None = None,
        poll_s: float = 5.0,
        timeout_s: float = 1800.0,
    ) -> Path:
        """``bytedance/seedance-2.5/image-to-video`` (queue submit + poll).

        Schema (confirmed 2026-09-04): required ``prompt`` and ``image_url``;
        optional ``end_image_url``, ``duration`` (string ``"auto"`` or
        ``"4".."30"``, default ``"auto"``), ``resolution``
        (``480p|720p|1080p``, default ``720p``), ``aspect_ratio`` (default
        ``auto``), ``generate_audio`` (default **true** - we default it to
        false), ``bitrate_mode`` (``standard|high``), ``seed``, ``end_user_id``.
        Output: ``{"video": {"url", ...}, "seed": int}``.  Note ``duration`` is
        a *string* in the schema; an int is coerced here.  There is no
        ``camera_fixed`` parameter - camera behaviour is prompt-driven.

        This is the most expensive call in the pipeline (~$0.47 per generated
        second at 720p, so ~$2.35 for 5 s), therefore ``confirm=True`` is
        mandatory and the estimate is logged before anything is submitted.
        """
        per_second = SEEDANCE_USD_PER_SECOND.get(resolution)
        seconds = 0 if str(duration) == "auto" else int(duration)
        cost = (per_second or 0.0) * seconds
        logger.warning(
            "fal seedance_i2v: %ss at %s -> estimated cost $%.2f%s",
            duration,
            resolution,
            cost,
            " (per-second price unverified)" if per_second is None else "",
        )
        if not confirm:
            raise FalError(
                f"seedance_i2v would cost about ${cost:.2f} "
                f"({duration}s at {resolution}); call again with confirm=True"
            )

        arguments: dict[str, Any] = {
            "prompt": prompt,
            "image_url": self.upload(image_path),
            "duration": str(duration),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": bool(generate_audio),
        }
        if end_image_path is not None:
            arguments["end_image_url"] = self.upload(end_image_path)
        if seed is not None:
            arguments["seed"] = int(seed)

        request_id = self.submit(APP_SEEDANCE_I2V, arguments)
        result = self.wait(APP_SEEDANCE_I2V, request_id, poll_s=poll_s, timeout_s=timeout_s)
        url = (result.get("video") or {}).get("url")
        if not url:
            raise FalError(f"seedance returned no video url: {str(result)[:500]}")

        self._report_cost("i2v", APP_SEEDANCE_I2V, f"{seconds}s {resolution}", cost)
        dest = Path(out_path) if out_path else Path.cwd() / f"seedance_{request_id}.mp4"
        return self.download(url, dest)

    def upscale_video(
        self,
        path: Path | str,
        factor: int = 2,
        *,
        model: str = "Proteus",
        target_fps: int | None = None,
        out_path: Path | str | None = None,
        duration_s: float | None = None,
        output_height: int | None = None,
        h264_output: bool = True,
        poll_s: float = 10.0,
        timeout_s: float = 3600.0,
    ) -> Path:
        """``fal-ai/topaz/upscale/video`` -> upscaled video on disk.

        Schema (confirmed 2026-09-04): required ``video_url``; optional
        ``model`` (``Proteus`` default, plus Artemis/Gaia/Nyx/Starlight
        variants), ``upscale_factor`` (number, default 2), ``target_fps``
        (16..60), ``H264_output`` (note the capital H, default false),
        ``noise`` / ``halo`` / ``grain`` / ``recover_detail`` / ``compression``
        (0..1 fine-tuning knobs).  Output: ``{"video": {"url", ...}}``.

        Billing is per second of output: $0.01 (<=720p), $0.02 (<=1080p),
        $0.08 (>1080p).  Pass ``duration_s`` and ``output_height`` to get a real
        estimate into the ledger.
        """
        arguments: dict[str, Any] = {
            "video_url": self.upload(path),
            "upscale_factor": factor,
            "model": model,
            "H264_output": bool(h264_output),
        }
        if target_fps is not None:
            arguments["target_fps"] = int(target_fps)

        cost = 0.0
        if duration_s is not None:
            height = output_height or 1080
            rate = next(v for k, v in sorted(TOPAZ_USD_PER_SECOND.items()) if height <= k)
            cost = duration_s * rate
        logger.info(
            "fal upscale_video factor=%s model=%s estimated cost=%s",
            factor,
            model,
            f"${cost:.2f}" if cost else "unknown (pass duration_s)",
        )

        request_id = self.submit(APP_TOPAZ_UPSCALE, arguments)
        result = self.wait(APP_TOPAZ_UPSCALE, request_id, poll_s=poll_s, timeout_s=timeout_s)
        url = (result.get("video") or {}).get("url")
        if not url:
            raise FalError(f"topaz returned no video url: {str(result)[:500]}")

        if cost:
            self._report_cost("upscale", APP_TOPAZ_UPSCALE, f"{duration_s:.1f}s", cost)
        dest = Path(out_path) if out_path else Path(path).with_name(f"{Path(path).stem}_up{factor}x.mp4")
        return self.download(url, dest)


__all__ = [
    "Fal",
    "FalError",
    "map_status",
    "APP_NANO_BANANA_PRO",
    "APP_NANO_BANANA_PRO_EDIT",
    "APP_SEEDANCE_I2V",
    "APP_TOPAZ_UPSCALE",
    "SEEDANCE_USD_PER_SECOND",
    "NANO_BANANA_USD_PER_IMAGE",
    "TOPAZ_USD_PER_SECOND",
]
