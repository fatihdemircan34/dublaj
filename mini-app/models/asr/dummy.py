from __future__ import annotations

from core.registry.asr import register_asr


@register_asr("dummy")
class DummyASR:
    """Trivial ASR model used for tests."""

    def __init__(self, **_: object) -> None:
        pass

    def transcribe(self, wav_path: str) -> list[dict]:  # pragma: no cover - simple
        return [{"start": 0.0, "end": 2.0, "text": "merhaba dunya"}]
