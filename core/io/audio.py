from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract mono 16kHz audio from video using ffmpeg."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Input video file not found: {video}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract audio but was not found in PATH")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to extract audio: {stderr}") from exc
    return str(out)
