from __future__ import annotations

from typing import Protocol

from .base import ModelRegistry, register


class BaseLipSync(Protocol):
    def sync(self, video_path: str, audio_path: str) -> str:
        """Return path to processed video."""
        ...


LIPSYNC_REGISTRY: ModelRegistry[BaseLipSync] = ModelRegistry()


def register_lipsync(name: str):
    return register(LIPSYNC_REGISTRY, name)
