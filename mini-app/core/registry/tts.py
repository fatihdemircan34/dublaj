from __future__ import annotations

from pathlib import Path
from typing import Dict, Protocol

from .base import ModelRegistry, register


class BaseTTS(Protocol):
    def synthesize(self, text: str, output_path: Path, voice_id: str | None = None, **kwargs) -> Path:
        """Synthesize ``text`` into ``output_path`` and return the path."""
        ...

    def list_voices(self) -> Dict[str, str]:  # pragma: no cover - optional
        ...

    def is_available(self) -> bool:  # pragma: no cover - optional
        ...


TTS_REGISTRY: ModelRegistry[BaseTTS] = ModelRegistry()


def register_tts(name: str):
    return register(TTS_REGISTRY, name)
