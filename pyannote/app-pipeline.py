#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app-pipeline.py

Gerçek, uçtan uca dublaj pipeline'ı (çalışan sürüm).
Akış:
  DownloadStep → ExtractAudioStep → **AppFullCoreStep** (VAD+OSD+DIAR+ASR+speaker atama)
  → MergeSentencesStep (out.jsonl) → BuildRefVoicesFromJSONLStep (9 sn)
  → SentenceTranslationStep (opsiyonel) → XTTSPerSegmentStep (çoklu backend fallback)
  → PerfectMixStepWrapper (ops.) → LipSyncStepWrapper (ops.)

Önemli:
- Çekirdek olarak aynı dizindeki app-full.py subprocess ile çağrılır.
- app-full çıktıları: step3_diarization.json + kelime-seviye JSONL (out_words.jsonl)
- XTTS yoksa Coqui TTS → edge-tts → sessizlik fallback zinciriyle asla crash olmaz.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_monolith import (
    DownloadStep, ExtractAudioStep,
    PerfectMixStepWrapper, LipSyncStepWrapper,
    make_ctx, run_steps
)

logger = logging.getLogger("app-pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------- yardımcılar --------------------------- #

def ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-version"], check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError("ffmpeg PATH'te bulunamadı") from e


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            rows.append(json.loads(l))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------- app-full.py çekirdeği -------------------- #


# --- Step classes moved to pipeline_steps package ---
from step.app_full_core_step import (
    AppFullCoreStep,
)

from step.merge_sentences_step import (
    MergeSentencesStep,
)
from step.build_ref_voices_step import (
    BuildRefVoicesFromJSONLStep,
)
from step.sentence_translation_step import (
    SentenceTranslationStep,
)
from step.xtts_per_segment_step import (
    XTTSPerSegmentStep
)


# ----------------------------- CLI & Runner ----------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yeni dublaj pipeline'ı (app-pipeline.py)")
    p.add_argument("--input", help="YouTube URL veya video dosyası")
    p.add_argument("--audio", help="Doğrudan WAV yolu (Download+Extract atlanır)")
    p.add_argument("--out-jsonl", dest="out_jsonl", default="out.jsonl", help="Cümle bazlı JSONL çıktı adı")
    p.add_argument("--temp-dir", default="./_temp_app", help="Geçici çalışma klasörü")
    p.add_argument("--hf-token", dest="hf_token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token")

    # Diller
    p.add_argument("--source_lang", default="en")
    p.add_argument("--target_lang", default="tr")
    p.add_argument("--target_lan", dest="target_lang", help="(alias) hedef dil")

    # Çeviri
    p.add_argument("--translator", default="none", help="none|googletrans")

    # App-full çekirdek parametreleri
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--asr-device", default="auto")
    p.add_argument("--asr-compute-type", default="auto")
    p.add_argument("--vad-onset", type=float, default=0.5)
    p.add_argument("--vad-offset", type=float, default=0.5)
    p.add_argument("--vad-min-on", type=float, default=0.0)
    p.add_argument("--vad-min-off", type=float, default=0.0)
    p.add_argument("--osd-onset", type=float, default=0.7)
    p.add_argument("--osd-offset", type=float, default=0.7)
    p.add_argument("--osd-min-on", type=float, default=0.10)
    p.add_argument("--osd-min-off", type=float, default=0.10)
    p.add_argument("--require-vad", action="store_true")
    p.add_argument("--vad-coverage", type=float, default=0.6)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)

    # TTS
    p.add_argument("--tts", default="xtts")
    p.add_argument("--tts_kw", default=None, help="JSON dict (XTTS/Coqui parametreleri)")
    p.add_argument("--voice_map", default=None, help='JSON dict {"SPEAKER_00": "/path/override.wav", ...}')
    p.add_argument("--edge-voice", dest="edge_voice", default=None, help="edge-tts için ses adı (örn. tr-TR-AhmetNeural)")

    # Mix & lipsync
    p.add_argument("--do-mix", action="store_true", help="PerfectMix adımını çalıştır")
    p.add_argument("--do-lipsync", action="store_true", help="LipSync adımını çalıştır")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    temp_dir = Path(args.temp_dir); temp_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {
        "input": args.input,
        "audio": args.audio,
        "out_jsonl": str(Path(args.out_jsonl) if Path(args.out_jsonl).is_absolute() else temp_dir / args.out_jsonl),
        "temp_dir": str(temp_dir),
        "hf_token": args.hf_token,
        "sample_rate": 16000,
        # diller
        "translator": args.translator,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        # app-full çekirdek parametreleri
        "asr_model": args.asr_model,
        "asr_device": args.asr_device,
        "asr_compute_type": args.asr_compute_type,
        "vad_onset": args.vad_onset, "vad_offset": args.vad_offset,
        "vad_min_on": args.vad_min_on, "vad_min_off": args.vad_min_off,
        "osd_onset": args.osd_onset, "osd_offset": args.osd_offset,
        "osd_min_on": args.osd_min_on, "osd_min_off": args.osd_min_off,
        "require_vad": bool(args.require_vad), "vad_coverage": args.vad_coverage,
        "min_speakers": args.min_speakers, "max_speakers": args.max_speakers,
        # TTS
        "tts": args.tts,
        "tts_kw": json.loads(args.tts_kw) if args.tts_kw else {},
        "voice_map": json.loads(args.voice_map) if args.voice_map else {},
        "edge_voice": args.edge_voice,
        # Mix defaults
        "lufs_target": -14.0, "duck_db": -7.0, "pan_amount": 0.0,
    }

    # Adımlar
    steps: List[Any] = []
    if args.input and not args.audio:
        steps += [DownloadStep(), ExtractAudioStep()]
    else:
        if args.audio:
            ctx_seed = make_ctx(temp_dir, cfg)
            ctx_seed.setdefault("artifacts", {})["original_audio"] = args.audio

    steps += [
        AppFullCoreStep(
            asr_model=args.asr_model,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            vad_onset=args.vad_onset, vad_offset=args.vad_offset,
            vad_min_on=args.vad_min_on, vad_min_off=args.vad_min_off,
            osd_onset=args.osd_onset, osd_offset=args.osd_offset,
            osd_min_on=args.osd_min_on, osd_min_off=args.osd_min_off,
            require_vad=args.require_vad, vad_coverage=args.vad_coverage,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers
        ),
        MergeSentencesStep(gap_th=1.0, out_name=str(Path(cfg["out_jsonl"]).name)),
        SentenceTranslationStep(engine=args.translator, source_lang=args.source_lang, target_lang=args.target_lang),
        BuildRefVoicesFromJSONLStep(seconds=9.0, sample_rate=cfg.get("sample_rate", 16000)),
        XTTSPerSegmentStep(tts_name=args.tts, tts_kw=cfg.get("tts_kw"), voice_map=cfg.get("voice_map"),
                           sample_rate=cfg.get("sample_rate", 16000), edge_voice=cfg.get("edge_voice")),
    ]

    if args.do_mix:
        steps.append(PerfectMixStepWrapper(lufs_target=cfg.get("lufs_target", -14.0),
                                           duck_db=cfg.get("duck_db", -7.0),
                                           pan_amount=cfg.get("pan_amount", 0.0)))
    if args.do_lipsync:
        steps.append(LipSyncStepWrapper(model_name="simple", model_kwargs={}))

    # Çalıştır
    ctx = make_ctx(temp_dir, cfg)
    if args.audio:
        ctx["artifacts"]["original_audio"] = args.audio
    run_steps(steps, ctx)

    # Özet
    summary = {
        "artifacts": ctx.get("artifacts", {}),
        "stats": {
            "total_sentences": len(ctx["artifacts"].get("sentences", [])),
        }
    }
    with (temp_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Tamamlandı. Çıktı özet: %s", temp_dir / "pipeline_summary.json")


if __name__ == "__main__":
    main()
