from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List
import logging

from .base import BaseLipSyncModel
from core.registry.lipsync import register_lipsync

logger = logging.getLogger(__name__)


@register_lipsync("simple")
class SimpleMuxModel(BaseLipSyncModel):
    """Simple audio-video multiplexing (no actual lip sync)."""

    def __init__(
        self,
        *,
        copy: bool = False,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
    ) -> None:
        """Create a SimpleMuxModel.

        Parameters
        ----------
        copy:
            If ``True`` the video stream will be copied rather than re-encoded.
            Defaults to ``False`` which encodes the video with H.264 for better
            compatibility.
        video_codec:
            Video codec to use when ``copy`` is ``False``. Defaults to
            ``"libx264"``.
        audio_codec:
            Audio codec to use; defaults to ``"aac"``.
        """

        self.copy = copy
        self.video_codec = video_codec
        self.audio_codec = audio_codec

    def sync(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
        ]

        if self.copy:
            cmd.extend(["-c:v", "copy"])
        else:
            cmd.extend(
                [
                    "-c:v",
                    self.video_codec,
                    "-profile:v",
                    "high",
                    "-level",
                    "4.2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ]
            )

        cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                self.audio_codec,
                "-b:a",
                "192k",
                "-shortest",
                str(output_path),
            ]
        )
        logger.debug("Running ffmpeg: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg mux error: %s", result.stderr)
            raise RuntimeError(f"FFmpeg mux failed: {result.stderr}")
        return output_path

    def is_available(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_requirements(self) -> List[str]:
        return ["ffmpeg"]
