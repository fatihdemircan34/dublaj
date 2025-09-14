from __future__ import annotations

from core.io import demucs
from core.pipeline.base import Context


class VoiceSeparationStep:
    """Produce music-only audio using Demucs."""

    name = "VoiceSeparation"

    def __init__(self, model: str = "htdemucs") -> None:
        self.model = model

    def run(self, ctx: Context) -> None:
        audio_path = ctx["artifacts"].get("original_audio")
        if not audio_path:
            return
        out = demucs.separate_music(audio_path, model=self.model, temp_dir=ctx["temp_dir"])
        ctx["artifacts"]["music_audio"] = out
