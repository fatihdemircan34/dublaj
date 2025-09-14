from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def is_available() -> bool:
    """Return True if the demucs command is available."""
    try:
        result = subprocess.run(
            ["python", "-m", "demucs", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def separate_music(audio_path: str, model: str = "htdemucs", temp_dir: str | None = None) -> str:
    """Separate ``audio_path`` into stems and return a music-only track.

    If Demucs is not available, the original audio is copied.
    """
    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    work_dir = Path(temp_dir) if temp_dir else audio.parent / "demucs_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not is_available():
        out = audio.parent / "music_only.wav"
        shutil.copy(audio, out)
        return str(out)

    cmd = [
        "python",
        "-m",
        "demucs.separate",
        "-n",
        model,
        "-o",
        str(work_dir),
        str(audio),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Demucs separation failed: {result.stderr}")

    separated_dir = work_dir / model / audio.stem
    drums = separated_dir / "drums.wav"
    bass = separated_dir / "bass.wav"
    other = separated_dir / "other.wav"
    stems = [drums, bass, other]
    missing = [p for p in stems if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing separated files: {missing}")

    out = audio.parent / "music_only.wav"
    inputs: list[str] = []
    for p in stems:
        inputs.extend(["-i", str(p)])
    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex",
        f"[0:a][1:a][2:a]amix=inputs={len(stems)}",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(out),
        "-nostdin",
        "-loglevel",
        "error",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Music stem combination failed: {result.stderr}")
    return str(out)


def mix_with_voice(
    music_path: str,
    voice_path: str,
    output_path: str,
    music_volume: float = 0.8,
) -> str:
    """Mix ``music_path`` and ``voice_path`` into ``output_path`` using ffmpeg."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(music_path),
        "-i",
        str(voice_path),
        "-filter_complex",
        f"[0:a]volume={music_volume}[music];[music][1:a]amix=inputs=2:duration=longest",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(out),
        "-nostdin",
        "-loglevel",
        "error",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio mixing failed: {result.stderr}")
    return str(out)
