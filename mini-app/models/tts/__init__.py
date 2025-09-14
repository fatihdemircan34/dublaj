"""TTS engines package.

This module imports available TTS engine implementations so that any
``@register_tts`` decorators run on import.  Some engines depend on optional
third-party libraries; if those dependencies are missing we still want the
package to be importable so that the available engines can be used.
"""

from . import gtts, xtts  # always available engines

# Optional engines ---------------------------------------------------------
# CosyVoice3 requires an external ``tts`` package.  It may not be installed in
# all environments (e.g. lightweight inference setups), so we attempt the
# import but silently ignore a ``ModuleNotFoundError``.  This mirrors the
# behaviour of other optional dependencies and keeps ``models.tts`` importable
# even when CosyVoice3 isn't available.
try:  # pragma: no cover - best effort import of optional dependency
    from . import cosyvoice3  # triggers @register_tts decorators
except ModuleNotFoundError:
    pass


