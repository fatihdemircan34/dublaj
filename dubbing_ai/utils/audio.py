from __future__ import annotations
import subprocess, os, uuid

def to_wav_mono_16k(in_path: str, tmp_dir: str) -> str:
    out = os.path.join(tmp_dir, f"{uuid.uuid4().hex}_16k.wav")
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ac", "1", "-ar", "16000", "-f", "wav", out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out
