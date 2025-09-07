import logging
from typing import Any, Dict, List, Optional
logger = logging.getLogger("app-pipeline")

class SentenceTranslationStep:
    """Cümleleri hedef dile çevirir (engine='none' varsayılan)."""
    name = "SentenceTranslation"

    def __init__(self, engine: str = "none", source_lang: str = "en", target_lang: str = "tr"):
        self.engine = engine
        self.src = source_lang
        self.tgt = target_lang

    def run(self, ctx: Dict[str, Any]) -> None:
        sents: List[Dict[str, Any]] = ctx["artifacts"].get("sentences") or []
        if not sents:
            logger.info("SentenceTranslation: cümle yok; atlanıyor.")
            return

        if self.engine == "none":
            ctx["artifacts"]["sentences_tr"] = sents
            return

        if self.engine == "googletrans":
            try:
                from googletrans import Translator  # type: ignore
                tr = Translator()
                texts = [s["text"] for s in sents]
                out = []
                chunk = 100
                for i in range(0, len(texts), chunk):
                    part = texts[i:i+chunk]
                    res = tr.translate(part, src=self.src, dest=self.tgt)
                    for s, rr in zip(sents[i:i+chunk], res):
                        s2 = dict(s)
                        s2["text_tr"] = rr.text
                        out.append(s2)
                ctx["artifacts"]["sentences_tr"] = out
                return
            except Exception as e:
                logger.warning("googletrans başarısız (%s); pas geçiliyor.", e)

        ctx["artifacts"]["sentences_tr"] = sents


# -------------------- TTS (çoklu backend fallback) -------------------- #
