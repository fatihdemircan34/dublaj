from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests
from deep_translator import GoogleTranslator

from core.registry.translator import TranslationError, register_translator

logger = logging.getLogger(__name__)


class LibreTranslateClient:
    """Minimal LibreTranslate client used as an offline backend."""

    def __init__(self, host: str = "http://localhost:5000", api_key: str | None = None):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            resp = self.session.get(f"{self.host}/languages", timeout=5)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def get_languages(self) -> List[Dict[str, str]]:
        if not self.is_available():
            return []
        try:
            resp = self.session.get(f"{self.host}/languages", timeout=5)
            return resp.json()
        except Exception:
            return []

    def translate(self, text: str, source: str, target: str) -> Optional[str]:
        if not self.is_available():
            return None
        data = {"q": text, "source": source, "target": target, "format": "text"}
        if self.api_key:
            data["api_key"] = self.api_key
        try:
            resp = self.session.post(f"{self.host}/translate", data=data, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("translatedText", "")
        except Exception:
            pass
        return None


@register_translator("advanced")
class AdvancedTranslator:
    """Translator with LibreTranslate and Google backends."""

    def __init__(self, src_lang: str = "en", target_lang: str = "tr") -> None:
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.libretranslate = LibreTranslateClient()
        self._google_translator: GoogleTranslator | None = None
        self.lang_mapping = {
            "en": "en",
            "tr": "tr",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "ru": "ru",
            "ja": "ja",
            "ko": "ko",
            "zh": "zh",
        }
        self.primary_backend = self._select_primary_backend()
        logger.info("Çeviri altyapısı: %s", self.primary_backend)

    # backend selection -------------------------------------------------
    def _select_primary_backend(self) -> str:
        if self.libretranslate.is_available():
            languages = self.libretranslate.get_languages()
            codes = [lang["code"] for lang in languages]
            src = self.lang_mapping.get(self.src_lang, self.src_lang)
            tgt = self.lang_mapping.get(self.target_lang, self.target_lang)
            if src in codes and tgt in codes:
                return "libretranslate"
        return "google"

    def _get_google_translator(self) -> GoogleTranslator:
        if self._google_translator is None:
            self._google_translator = GoogleTranslator(source=self.src_lang, target=self.target_lang)
        return self._google_translator

    # translation -------------------------------------------------------
    def translate(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return text
        try:
            if self.primary_backend == "libretranslate":
                result = self._translate_with_libretranslate(text)
                if result and result.strip() != text.strip():
                    return result
                logger.warning("LibreTranslate başarısız, Google çevirisine geçiliyor")
            result = self._translate_with_google(text)
            if result and result.strip() != text.strip():
                return result
            logger.error("Çeviri başarısız: sonuç orijinal metinle aynı")
            return None
        except Exception as exc:
            logger.error("Çeviri sırasında hata: %s", exc)
            raise TranslationError("Çeviri sırasında hata") from exc

    def _translate_with_libretranslate(self, text: str) -> Optional[str]:
        try:
            src = self.lang_mapping.get(self.src_lang, self.src_lang)
            tgt = self.lang_mapping.get(self.target_lang, self.target_lang)
            return self.libretranslate.translate(text, src, tgt)
        except Exception as exc:
            logger.error("LibreTranslate hatası: %s", exc)
            return None

    def _translate_with_google(self, text: str) -> Optional[str]:
        try:
            translator = self._get_google_translator()
            return translator.translate(text)
        except Exception as exc:
            logger.error("Google Çeviri hatası: %s", exc)
            return None

    # batch -------------------------------------------------------------
    def translate_batch(self, texts: List[str]) -> List[str]:
        results: List[str] = []
        for txt in texts:
            try:
                translated = self.translate(txt)
            except TranslationError:
                translated = None
            results.append(translated or txt)
        return results
