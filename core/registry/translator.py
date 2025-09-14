from __future__ import annotations

from typing import Optional, Protocol

from .base import ModelRegistry, register


class TranslationError(Exception):
    """Raised when a translation attempt fails."""


class BaseTranslator(Protocol):
    """Translator model interface."""

    def translate(self, text: str) -> Optional[str]:
        """Return translated text or ``None`` on failure.

        Implementations may raise :class:`TranslationError` for
        exceptional cases.
        """
        ...


TRANSLATOR_REGISTRY: ModelRegistry[BaseTranslator] = ModelRegistry()


def register_translator(name: str):
    return register(TRANSLATOR_REGISTRY, name)
