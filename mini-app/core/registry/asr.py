from __future__ import annotations

from typing import Protocol

from .base import ModelRegistry, register


class BaseASR(Protocol):
    """ASR model interface."""

    def transcribe(self, wav_path: str) -> list[dict]:
        """Return list of segment dicts with start/end/text."""
        ...


ASR_REGISTRY: ModelRegistry[BaseASR] = ModelRegistry()


def register_asr(name: str):
    return register(ASR_REGISTRY, name)
