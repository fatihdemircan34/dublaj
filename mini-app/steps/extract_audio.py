from __future__ import annotations

from pathlib import Path

from core.io import audio
from core.pipeline.base import Context


class ExtractAudioStep:
    name = "ExtractAudio"

    def run(self, ctx: Context) -> None:
        video_path = ctx["artifacts"]["video"]
        out = Path(ctx["temp_dir"]) / "original_audio.wav"
        path = audio.extract_audio(video_path, str(out))
        ctx["artifacts"]["original_audio"] = path
