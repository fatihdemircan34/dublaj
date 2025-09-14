from __future__ import annotations

from pathlib import Path
import subprocess

from core.registry.tts import register_tts
from .gtts import BaseTTSEngine
from tts.cosyvoice3_adapter import CosyVoice3TTSAdapter


@register_tts("cosyvoice3")
class CosyVoice3Engine(BaseTTSEngine):
    """TTS engine built around :class:`CosyVoice3TTSAdapter`.

    The adapter always returns WAV bytes.  This engine writes those bytes to
    ``output_path`` and converts them to MP3 if requested by the file suffix.
    """

    def __init__(self, model_dir: str, language: str = "EN", speed: float = 1.0) -> None:
        self.model_dir = model_dir
        self.language = language
        self.speed = speed
        self.adapter = CosyVoice3TTSAdapter()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_id: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
    ) -> Path:
        """Synthesize ``text`` into ``output_path`` and return the path."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lang = lang or self.language
        speed = speed or self.speed

        wav_bytes = self.adapter.synthesize(text, speaker_profile=voice_id, lang=lang, speed=speed)

        if output_path.suffix.lower() == ".mp3":
            tmp_wav = output_path.with_suffix(".wav")
            with open(tmp_wav, "wb") as f:
                f.write(wav_bytes)
            cmd = [
                "ffmpeg",
                "-i",
                str(tmp_wav),
                "-q:a",
                "4",
                "-y",
                str(output_path),
                "-loglevel",
                "error",
            ]
            subprocess.run(cmd, check=True)
            try:
                tmp_wav.unlink()
            except FileNotFoundError:
                pass
        else:
            with open(output_path, "wb") as f:
                f.write(wav_bytes)

        return output_path
