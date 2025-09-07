import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .utils import read_jsonl
logger = logging.getLogger("app-pipeline")

class AppFullCoreStep:
    """
    app-full.py'yi subprocess ile çağırır ve:
      - step3_diarization.json (temp_dir)
      - kelime-seviye JSONL (temp_dir/out_words.jsonl)
    üretir. words belleğe alınır -> ctx['artifacts']['words'].
    """
    name = "AppFullCore"

    def __init__(self,
                 asr_model: str = "large-v3",
                 asr_device: str = "auto",
                 asr_compute_type: str = "auto",
                 vad_onset: float = 0.5,
                 vad_offset: float = 0.5,
                 vad_min_on: float = 0.0,
                 vad_min_off: float = 0.0,
                 osd_onset: float = 0.7,
                 osd_offset: float = 0.7,
                 osd_min_on: float = 0.10,
                 osd_min_off: float = 0.10,
                 require_vad: bool = False,
                 vad_coverage: float = 0.6,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None) -> None:
        self.asr_model = asr_model
        self.asr_device = asr_device
        self.asr_compute_type = asr_compute_type
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.vad_min_on = vad_min_on
        self.vad_min_off = vad_min_off
        self.osd_onset = osd_onset
        self.osd_offset = osd_offset
        self.osd_min_on = osd_min_on
        self.osd_min_off = osd_min_off
        self.require_vad = require_vad
        self.vad_coverage = vad_coverage
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def run(self, ctx: Dict[str, Any]) -> None:
        temp = Path(ctx["temp_dir"])
        audio_p = ctx["config"].get("audio") or ctx["artifacts"].get("original_audio")
        if not audio_p:
            raise RuntimeError("AppFullCoreStep: audio bulunamadı")

        app_full = Path(__file__).resolve().parent.parent / "app-full.py"
        if not app_full.exists():
            raise FileNotFoundError(f"app-full.py bulunamadı: {app_full}")

        out_words = temp / "out_words.jsonl"
        hf_token = ctx["config"].get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError("HF token gerekli (HF_TOKEN env ya da --hf-token).")

        cmd = [
            sys.executable, str(app_full),
            "--audio", str(audio_p),
            "--out", str(out_words),
            "--hf-token", str(hf_token),
            "--asr-model", self.asr_model,
            "--asr-device", self.asr_device,
            "--asr-compute-type", self.asr_compute_type,
            "--vad-onset", str(self.vad_onset),
            "--vad-offset", str(self.vad_offset),
            "--vad-min-on", str(self.vad_min_on),
            "--vad-min-off", str(self.vad_min_off),
            "--osd-onset", str(self.osd_onset),
            "--osd-offset", str(self.osd_offset),
            "--osd-min-on", str(self.osd_min_on),
            "--osd-min-off", str(self.osd_min_off),
            "--vad-coverage", str(self.vad_coverage),
            "--output-dir", str(temp),
        ]
        if self.require_vad:
            cmd.append("--require-vad")
        if self.min_speakers is not None:
            cmd += ["--min-speakers", str(self.min_speakers)]
        if self.max_speakers is not None:
            cmd += ["--max-speakers", str(self.max_speakers)]

        logger.info("app-full.py çağrılıyor…")
        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.returncode != 0:
            logger.error("app-full.py stderr:\n%s", r.stderr)
            raise RuntimeError("app-full.py çalıştırma hatası")

        if not out_words.exists():
            raise RuntimeError(f"app-full çıktı dosyası yok: {out_words}")

        words = read_jsonl(out_words)
        ctx["artifacts"]["words"] = words
        ctx["artifacts"]["words_jsonl"] = str(out_words)

        diar_json = temp / "step3_diarization.json"
        if diar_json.exists():
            ctx["artifacts"]["step3_diarization_json"] = str(diar_json)

        logger.info("app-full çekirdeği tamam. Kelime sayısı: %d", len(words))


# -------------------- Cümle birleştirme -------------------- #
