from __future__ import annotations

import shutil
from pathlib import Path

from core.io import demucs
from core.pipeline.base import Context


class MixAudioStep:
    """Mix music-only audio with synthesized speech."""

    name = "MixAudio"

    def __init__(self, music_volume: float = 0.8) -> None:
        self.music_volume = music_volume

    def run(self, ctx: Context) -> None:
        music = ctx["artifacts"].get("music_audio")
        voice = ctx["artifacts"].get("synth_audio")
        if not music or not voice:
            return
        out = Path(ctx["temp_dir"]) / "final_audio.wav"
        try:
            mixed = demucs.mix_with_voice(music, voice, str(out), music_volume=self.music_volume)
        except Exception:
            shutil.copy(voice, out)
            mixed = str(out)
        ctx["artifacts"]["final_audio"] = mixed
