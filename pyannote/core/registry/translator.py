from __future__ import annotations

from typing import Protocol

from .base import ModelRegistry, register


class BaseTranslator(Protocol):
    """Translator model interface."""

    def translate(self, text: str) -> str:
        ...


TRANSLATOR_REGISTRY: ModelRegistry[BaseTranslator] = ModelRegistry()


def register_translator(name: str):
    return register(TRANSLATOR_REGISTRY, name)
