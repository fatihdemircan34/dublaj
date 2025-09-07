import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from .utils import ensure_ffmpeg, read_jsonl
logger = logging.getLogger("app-pipeline")

class BuildRefVoicesFromJSONLStep:
    """out.jsonl'den her konuşmacı için ~9 sn referans WAV çıkarır."""
    name = "BuildRefVoices"

    def __init__(self, seconds: float = 5.0, sample_rate: int = 16000) -> None:
        self.seconds = float(seconds)
        self.sample_rate = int(sample_rate)

    def _extract(self, audio_path: Path, start: float, duration: float, out_wav: Path) -> None:
        ensure_ffmpeg()
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(audio_path),
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-ac", "1", "-ar", str(self.sample_rate), "-c:a", "pcm_s16le",
            str(out_wav)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout or "ffmpeg failed")

    def run(self, ctx: Dict[str, Any]) -> None:
        audio_p = ctx["config"].get("audio") or ctx["artifacts"].get("original_audio")
        if not audio_p:
            raise RuntimeError("BuildRefVoicesFromJSONL: audio bulunamadı")
        audio_path = Path(audio_p)

        temp = Path(ctx["temp_dir"])
        out_jsonl = Path(ctx["artifacts"].get("out_jsonl") or (temp / "out.jsonl"))
        if not out_jsonl.exists():
            raise RuntimeError(f"BuildRefVoicesFromJSONL: out.jsonl yok -> {out_jsonl}")

        sentences = read_jsonl(out_jsonl)
        by_spk: Dict[str, List[Dict[str, Any]]] = {}
        for s in sentences:
            spk = str(s.get("speaker") or "SPEAKER_00")
            by_spk.setdefault(spk, []).append(s)

        voices_dir = temp / "voices"
        success = 0
        for spk, segs in by_spk.items():
            segs = sorted(segs, key=lambda x: float(x["start"]))

            acc = 0.0
            start_global: Optional[float] = None
            end_global: Optional[float] = None

            for s in segs:
                s_start = float(s["start"]); s_end = float(s["end"])
                if start_global is None:
                    start_global = s_start; end_global = s_end; acc = s_end - s_start
                else:
                    acc += (s_end - s_start)
                    end_global = s_end
                if acc >= self.seconds:
                    break

            if start_global is None or end_global is None:
                logger.warning("Konuşmacı için uygun cümle bulunamadı: %s", spk)
                continue

            duration = min(self.seconds, max(0.1, (end_global - start_global)))
            out_wav = voices_dir / f"{spk}.wav"
            try:
                self._extract(audio_path, start_global, duration, out_wav)
                success += 1
                logger.info("Ref voice: %s -> %s (%.2fs)", spk, out_wav, duration)
            except Exception as e:
                logger.warning("Ref voice çıkarma başarısız (%s): %s", spk, e)

        if success == 0:
            raise RuntimeError("Ref voice çıkarılamadı; out.jsonl ve diarization'ı kontrol edin.")

        ctx["artifacts"]["ref_voices_dir"] = str(voices_dir)


# -------------------- Cümle çeviri (opsiyonel) -------------------- #
