from __future__ import annotations

from core.registry.translator import register_translator
from .nllb import NLLBTranslator


@register_translator("opus_mt")
class OpusMTTranslator(NLLBTranslator):
    """Backward-compatible alias for the removed Opus-MT translator.

    Delegates all behaviour to :class:`NLLBTranslator` while keeping the
    previous registry name so legacy configurations referencing
    ``"opus_mt"`` continue to work.
    """

    pass
