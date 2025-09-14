from .advanced import AdvancedTranslator  # registers "advanced"
from .google_cloud import GoogleCloudTranslator  # registers "google_cloud"
try:
    from .opus_mt import OpusMTTranslator  # registers "opus_mt"
except Exception:  # transformers missing
    OpusMTTranslator = None
try:
    from .nllb import NLLBTranslator  # registers "nllb"
except Exception:  # transformers or sentencepiece missing
    NLLBTranslator = None
