from __future__ import annotations

from pathlib import Path
from typing import Dict
import subprocess
import wave
import logging

from core.registry.tts import register_tts


class BaseTTSEngine:
    """Minimal abstract base for TTS engines."""

    def synthesize(self, text: str, output_path: Path, voice_id: str | None = None, **kwargs) -> Path:
        raise NotImplementedError

    def list_voices(self) -> Dict[str, str]:  # pragma: no cover - simple
        return {}

    def is_available(self) -> bool:  # pragma: no cover - simple
        return True


@register_tts("gtts")
class GTTSEngine(BaseTTSEngine):
    """TTS engine using Google Text-to-Speech.

    Requires network connectivity; requests may fail without internet access.
    An optional ``silent_fallback`` flag can create placeholder audio files
    when synthesis fails.
    """

    def __init__(
        self, lang: str = "en", slow: bool = False, silent_fallback: bool = True
    ) -> None:
        self.lang = lang
        self.slow = slow
        self.silent_fallback = silent_fallback

    def synthesize(self, text: str, output_path: Path, voice_id: str | None = None, **kwargs) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(str(output_path))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("gTTS synthesis failed: %s", e)
            if self.silent_fallback:
                # Fallback: generate a short silent audio file so downstream
                # steps receive a valid input instead of an empty file. Try
                # ffmpeg first to create an MP3, and if that fails, write a
                # silent WAV directly.
                cmd = [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=16000:cl=mono",
                    "-t",
                    "1",
                    "-q:a",
                    "9",
                    "-y",
                    str(output_path),
                    "-nostdin",
                    "-loglevel",
                    "error",
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except Exception:
                    with wave.open(str(output_path), "w") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(b"\x00\x00" * 16000)
            raise RuntimeError("gTTS synthesis failed") from e
        return output_path
