from __future__ import annotations

import logging
from typing import List, Optional

from google.cloud import translate_v2 as translate

from core.registry.translator import TranslationError, register_translator

logger = logging.getLogger(__name__)


@register_translator("google_cloud")
class GoogleCloudTranslator:
    """Translator backed by Google Cloud Translation API."""

    def __init__(self, src_lang: str = "en", target_lang: str = "tr", **client_kw) -> None:
        self.src_lang = src_lang
        self.target_lang = target_lang
        try:
            self.client = translate.Client(**client_kw)
        except Exception as exc:  # pragma: no cover - network/credential errors
            logger.error("Failed to initialize Google Cloud Translation client: %s", exc)
            raise TranslationError("Google Cloud client init failed") from exc

    # single -------------------------------------------------------------
    def translate(self, text: str) -> Optional[str]:
        if not text:
            return text
        try:
            result = self.client.translate(
                text,
                source_language=self.src_lang,
                target_language=self.target_lang,
                format_="text",
            )
            return result.get("translatedText")
        except Exception as exc:
            logger.error("Google Cloud translation error: %s", exc)
            raise TranslationError("Google Cloud translation failed") from exc

    # batch --------------------------------------------------------------
    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        try:
            results = self.client.translate(
                texts,
                source_language=self.src_lang,
                target_language=self.target_lang,
                format_="text",
            )
            return [r.get("translatedText", t) for r, t in zip(results, texts)]
        except Exception as exc:
            logger.error("Google Cloud batch translation error: %s", exc)
            raise TranslationError("Google Cloud batch translation failed") from exc
