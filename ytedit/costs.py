"""Cost estimation and the per-project spend ledger.

Every paid API call estimates its cost from ``config/defaults.yaml: prices``,
checks it against the project budget, then appends an entry to
``state.json: costs``. A stage that would push the project over
``budget_usd`` raises :class:`BudgetExceeded` instead of spending money.
"""

from __future__ import annotations

from typing import Any, Mapping

from .config import Settings, global_settings
from .log import get_logger
from .project import Project, utcnow

log = get_logger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when an operation would exceed the project's ``budget_usd``."""


#: How each (service, op) pair interprets ``units``.
UNIT_KINDS: dict[tuple[str, str], str] = {
    ("elevenlabs", "stt"): "seconds",
    ("elevenlabs", "music"): "seconds",
    ("elevenlabs", "tts"): "chars",
    ("elevenlabs", "isolation"): "seconds",
    ("elevenlabs", "sfx"): "seconds",
    ("openrouter", "chat"): "tokens",
    ("fal", "image"): "images",
    ("fal", "video"): "seconds",
}


def unit_label(service: str, op: str, units: Any) -> str:
    """Render ``units`` as a short human string for the ledger."""
    kind = UNIT_KINDS.get((service, op), "units")
    if isinstance(units, Mapping):
        return ", ".join(f"{k}={v}" for k, v in units.items())
    if kind == "seconds":
        return f"{float(units):.1f}s"
    if kind == "chars":
        return f"{int(units)} chars"
    if kind == "tokens":
        return f"{int(units)} tok"
    if kind == "images":
        return f"{int(units)} img"
    return str(units)


def estimate(
    service: str,
    op: str,
    units: float | Mapping[str, Any],
    settings: Settings | None = None,
    model: str | None = None,
) -> float:
    """Estimate the USD cost of one API call.

    Args:
        service: ``elevenlabs`` | ``openrouter`` | ``fal``.
        op: For ElevenLabs one of ``stt``/``music``/``tts``/``isolation``/``sfx``
            (units in seconds, except ``tts`` in characters). For OpenRouter use
            ``chat`` with ``units={"in": n_prompt_tokens, "out": n_completion_tokens}``
            and ``model=<openrouter model id>``. For fal use ``image`` (units =
            number of images) or ``video`` (units = seconds) with ``model=<app id>``.
        units: Amount of work, interpreted per the table above.
        settings: Price source; defaults to the global settings.
        model: Model / app id, required for ``openrouter`` and ``fal``.

    Returns:
        Estimated cost in USD (``0.0`` when no price is configured).
    """
    cfg = settings or global_settings()
    prices = cfg.prices

    if service == "elevenlabs":
        table = prices.get("elevenlabs", {})
        amount = float(units) if not isinstance(units, Mapping) else 0.0
        if op == "stt":
            return amount / 3600.0 * float(table.get("stt_per_hour", 0.0))
        if op == "music":
            return amount / 60.0 * float(table.get("music_per_minute", 0.0))
        if op == "tts":
            return amount / 1000.0 * float(table.get("tts_per_1k_chars", 0.0))
        if op == "isolation":
            return amount / 60.0 * float(table.get("isolation_per_minute", 0.0))
        if op == "sfx":
            return amount / 60.0 * float(table.get("sfx_per_minute", 0.0))
        log.warning("no price for elevenlabs op %r", op)
        return 0.0

    if service == "openrouter":
        table = prices.get("openrouter", {})
        entry = table.get(model or "", {})
        if not entry:
            log.warning("no price for openrouter model %r", model)
            return 0.0
        tokens = units if isinstance(units, Mapping) else {"in": float(units), "out": 0.0}
        tin = float(tokens.get("in", 0.0))
        tout = float(tokens.get("out", 0.0))
        return (tin * float(entry.get("in", 0.0)) + tout * float(entry.get("out", 0.0))) / 1e6

    if service == "fal":
        table = prices.get("fal", {})
        entry = table.get(model or "", {})
        if not entry:
            log.warning("no price for fal app %r", model)
            return 0.0
        amount = float(units) if not isinstance(units, Mapping) else 0.0
        if op == "image":
            return amount * float(entry.get("per_image", 0.0))
        if op == "video":
            key = next((k for k in entry if k.startswith("per_second")), None)
            return amount * float(entry.get(key, 0.0)) if key else 0.0
        return 0.0

    log.warning("unknown cost service %r", service)
    return 0.0


def spent(project: Project) -> float:
    """Total USD already recorded in the project ledger."""
    return round(sum(float(e.get("usd", 0.0)) for e in project.load_state().get("costs", [])), 6)


def budget(project: Project) -> float:
    """The project's spending cap in USD."""
    state = project.load_state()
    return float(state.get("budget_usd", project.settings.budget_usd))


def remaining(project: Project) -> float:
    """USD left before the cap is hit."""
    return round(budget(project) - spent(project), 6)


def check_budget(project: Project, additional_usd: float = 0.0) -> None:
    """Raise :class:`BudgetExceeded` if ``additional_usd`` would break the cap.

    Args:
        project: Target project.
        additional_usd: Cost of the operation about to run.
    """
    cap = budget(project)
    used = spent(project)
    if used + additional_usd > cap + 1e-9:
        raise BudgetExceeded(
            f"{project.slug}: spent ${used:.4f} + ${additional_usd:.4f} exceeds "
            f"budget ${cap:.2f}; raise budget_usd in project.yaml or state.json"
        )


def record(
    project: Project,
    service: str,
    op: str,
    units: float | Mapping[str, Any],
    usd: float,
    **extra: Any,
) -> dict[str, Any]:
    """Append one entry to the project cost ledger.

    Args:
        project: Target project.
        service: Service name.
        op: Operation name.
        units: Work amount (stored raw and as a label).
        usd: Cost in USD.
        **extra: Extra fields (``model``, ``clip``, ``stage``, ``actual``...).

    Returns:
        The ledger entry that was written.
    """
    entry: dict[str, Any] = {
        "ts": utcnow(),
        "service": service,
        "op": op,
        "units": unit_label(service, op, units),
        "usd": round(float(usd), 6),
    }
    entry.update(extra)
    with project.edit_state() as state:
        state.setdefault("costs", []).append(entry)
    log.info("[cost]$%.4f[/] %s/%s (%s)", entry["usd"], service, op, entry["units"])
    return entry


def charge(
    project: Project,
    service: str,
    op: str,
    units: float | Mapping[str, Any],
    usd: float | None = None,
    model: str | None = None,
    **extra: Any,
) -> float:
    """Estimate (or accept) a cost, enforce the budget, and record it.

    Args:
        project: Target project.
        service: Service name.
        op: Operation name.
        units: Work amount.
        usd: Actual cost when the provider reported one; estimated otherwise.
        model: Model / app id for per-model pricing.
        **extra: Extra ledger fields.

    Returns:
        The charged amount in USD.

    Raises:
        BudgetExceeded: When the charge would exceed ``budget_usd``.
    """
    amount = (
        float(usd)
        if usd is not None
        else estimate(service, op, units, settings=project.settings, model=model)
    )
    check_budget(project, amount)
    if model:
        extra.setdefault("model", model)
    if usd is not None:
        extra.setdefault("actual", True)
    record(project, service, op, units, amount, **extra)
    return amount


def summary(project: Project) -> dict[str, Any]:
    """Return ``{"spent", "budget", "remaining", "by_service"}`` for reporting."""
    state = project.load_state()
    by_service: dict[str, float] = {}
    for entry in state.get("costs", []):
        key = f"{entry.get('service')}/{entry.get('op')}"
        by_service[key] = round(by_service.get(key, 0.0) + float(entry.get("usd", 0.0)), 6)
    used = round(sum(by_service.values()), 6)
    cap = float(state.get("budget_usd", project.settings.budget_usd))
    return {
        "spent": used,
        "budget": cap,
        "remaining": round(cap - used, 6),
        "by_service": by_service,
    }
