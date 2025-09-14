from __future__ import annotations

import logging

from core.pipeline.base import Context
from core.registry.translator import TRANSLATOR_REGISTRY, TranslationError


logger = logging.getLogger(__name__)


class TranslationStep:
    name = "Translate"

    def __init__(
        self,
        model_name: str = "nllb",
        src_lang: str = "en",
        target_lang: str = "tr",
    ) -> None:
        self.model_name = model_name
        self.src_lang = src_lang
        self.target_lang = target_lang

    def run(self, ctx: Context) -> None:
        segments = ctx["artifacts"].get("segments", [])
        logger.debug("Starting translation step with %d segments", len(segments))
        if not segments:
            logger.warning("No segments available for translation; skipping step")
            return

        translator = (
            ctx.get("config", {}).get("translator")
            or getattr(self, "model_name", None)
            or "nllb"
        )
        try:
            model = TRANSLATOR_REGISTRY.create(
                translator, src_lang=self.src_lang, target_lang=self.target_lang
            )
        except KeyError:
            if translator in {"opus_mt", "opus-mt", "marian"}:
                translator = "nllb"
                model = TRANSLATOR_REGISTRY.create(
                    translator, src_lang=self.src_lang, target_lang=self.target_lang
                )
            else:
                raise
        for idx, seg in enumerate(segments):
            text = seg.get("text", "")
            logger.debug("Translating segment %d: %s", idx, text)
            try:
                translation = model.translate(text)
                if translation is None:
                    raise TranslationError("Çeviri boş döndü")
                seg["translation"] = translation
                logger.debug("Translation result for segment %d: %s", idx, translation)
            except Exception as exc:
                logger.error("Çeviri hatası segment %d: %s", idx, exc)
                seg["translation_failed"] = True
        logger.debug("Translation step completed")
