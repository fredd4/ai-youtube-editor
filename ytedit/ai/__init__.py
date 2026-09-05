"""AI service clients.

Each client is self-contained: it takes an API key and an optional
``cost_callback(service, op, model, units, usd)`` and imports nothing from the
rest of the package, so it can be used from tests and one-off scripts.
"""

from ytedit.ai.elevenlabs import ElevenLabs, ElevenLabsError, MusicResult, Transcript
from ytedit.ai.fal import Fal, FalError
from ytedit.ai.openrouter import ChatResult, OpenRouter, OpenRouterError

__all__ = [
    "OpenRouter",
    "OpenRouterError",
    "ChatResult",
    "ElevenLabs",
    "ElevenLabsError",
    "Transcript",
    "MusicResult",
    "Fal",
    "FalError",
]
