from __future__ import annotations

import logging
import os
from typing import Optional, List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core.registry.translator import (
    TRANSLATOR_REGISTRY,
    TranslationError,
    register_translator,
)

logger = logging.getLogger(__name__)


@register_translator("nllb")
class NLLBTranslator:
    """Translator backed by Meta AI's NLLB models."""

    def __init__(
            self,
            src_lang: str = "en",
            target_lang: str = "tr",
            model_name: str = "facebook/nllb-200-distilled-600M",
            token: Optional[str] = None,
            local_files_only: bool = False,
            revision: Optional[str] = None,
            cache_dir: Optional[str] = None,
    ) -> None:
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.fallback = None

        self.lang_mapping = {
            "en": "eng_Latn",
            "tr": "tur_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "zh": "zho_Hans",
            "ar": "arb_Arab",
        }

        # 1) Token’ı ENV > parametre önceliğiyle al
        # (Parametre verilmişse o kullanılsın istiyorsan sırayı değiştir.)
        self.hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or token

        # 2) Ortak kwargs (token + offline + cache + revision)
        common_kwargs = {
            "local_files_only": local_files_only,
            "cache_dir": cache_dir,
            "revision": revision,
        }
        # Token parametresi transformers sürümüne göre 'token' ya da 'use_auth_token' olabilir.
        if self.hf_token:
            # Önce yeni API
            common_kwargs["token"] = self.hf_token
            # Eski API’yi de fallback olarak ekleyelim (zararsızsa yok sayılır)
            # common_kwargs["use_auth_token"] = self.hf_token

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **common_kwargs)
        except Exception as exc:  # pragma: no cover - network/IO errors
            logger.warning("NLLB model yüklenemedi (%s), advanced'e düşülüyor", exc)
            self.tokenizer = None
            self.model = None
            try:
                self.fallback = TRANSLATOR_REGISTRY.create(
                    "advanced", src_lang=src_lang, target_lang=target_lang
                )
            except Exception as inner:
                raise TranslationError("Çeviri modeli yüklenemedi") from inner

    def _translate_single(self, text: str) -> str:
        src = self.lang_mapping.get(self.src_lang, self.src_lang)
        tgt = self.lang_mapping.get(self.target_lang, self.target_lang)
        self.tokenizer.src_lang = src
        inputs = self.tokenizer(text, return_tensors="pt")

        lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
        if lang_code_to_id and tgt in lang_code_to_id:
            tgt_lang_id = lang_code_to_id[tgt]
        else:
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt)

        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
        )
        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def translate(self, text: str) -> str | None:
        if not text:
            return text
        if self.fallback:
            return self.fallback.translate(text)
        try:
            return self._translate_single(text)
        except Exception as exc:
            logger.warning("NLLB çeviri hatası (%s), fallback kullanılıyor", exc)
            if not self.fallback:
                self.fallback = TRANSLATOR_REGISTRY.create(
                    "advanced", src_lang=self.src_lang, target_lang=self.target_lang
                )
            return self.fallback.translate(text)

    def translate_batch(self, texts: List[str]) -> List[str]:
        if self.fallback:
            return [self.fallback.translate(t) for t in texts]
        results = []
        for t in texts:
            if not t:
                results.append(t)
                continue
            try:
                results.append(self._translate_single(t))
            except Exception as exc:
                logger.warning(
                    "NLLB toplu çeviri hatası (%s), fallback kullanılacak", exc
                )
                if not self.fallback:
                    self.fallback = TRANSLATOR_REGISTRY.create(
                        "advanced", src_lang=self.src_lang, target_lang=self.target_lang
                    )
                results.append(self.fallback.translate(t))
        return results
