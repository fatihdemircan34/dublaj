#!/usr/bin/env python3
"""
WhisperX + Pyannote Diarization
CLI sürümü — kullanıcının verdiği akışın birebir mantığı korunarak `main()` ve parametrelerle genelleştirildi.

Özellikler:
- Zorunlu: --audio (yerel dosya yolu)
- İsteğe bağlı: --audio-url (dosya yoksa indirir)
- HF token: --hf-token veya $HF_TOKEN
- Varsayılanlar kullanıcının paylaştığına uyumlu (whisper model: medium, batch 16, compute-type otomatik)
- GPU varsa otomatik seçilir; compute-type cc<7 ise float32'ye düşer
- Çıktı: Konuşmacı bazlı transcript (stdout) ve istenirse --save-transcript ile .txt

Örnekler:
  python whisperx_diarize_cli.py --audio sample_audio.wav
  python whisperx_diarize_cli.py --audio sample.wav \
    --audio-url https://github.com/pyannote/pyannote-audio/raw/develop/tutorials/assets/sample.wav
  python whisperx_diarize_cli.py --audio sample.wav --save-transcript out.txt
"""

from __future__ import annotations
import argparse
import os
import sys
import requests
import torch
import whisperx
import numpy as np
import random
from pyannote.audio import Pipeline
from typing import Optional, List, Dict, Any


def set_deterministic(seed: int = 42):
    """Deterministik davranış için tüm seed'leri ayarlar"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Deterministik CUDA ayarları
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch deterministik modu
    torch.use_deterministic_algorithms(True)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def download_audio(url: str, out_path: str, timeout: int = 30) -> None:
    if os.path.exists(out_path):
        print(f"Audio file already exists: {out_path}")
        return
    print(f"Attempting to download audio file from: {url}")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Audio file downloaded successfully: {out_path}")
    except requests.RequestException as exc:
        eprint(f"Error downloading audio file: {exc}")
        raise


def pick_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_compute_type(device: str) -> str:
    if device == "cuda":
        try:
            if torch.cuda.get_device_capability(0)[0] >= 7:
                return "float16"
        except Exception:
            pass
    return "float32"


def run_diarization(audio_path: str, model_name: str, hf_token: Optional[str], device: str):
    print("\n--- Running Speaker Diarization ---")
    try:
        print(f"Loading diarization pipeline: {model_name}...")
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        print("Diarization pipeline loaded.")
        print(f"Applying diarization pipeline to {audio_path}...")
        diarization = pipeline(audio_path)
        print("Diarization complete.")
        return diarization
    except Exception as e:
        eprint(f"Error during speaker diarization: {e}")
        eprint("Please ensure:\n1) Valid HF token ve model erişimi,\n2) Model sayfası sözleşmesi kabul edildi: https://huggingface.co/pyannote/speaker-diarization-3.1,\n3) Bağımlılıklar ve internet bağlantısı doğru.")
        raise


def run_asr_and_align(audio_path: str, whisper_model: str, batch_size: int, compute_type: str, device: str):
    print("\n--- Running Speech Recognition & Alignment ---")
    try:
        print(f"Loading WhisperX ASR model: {whisper_model}...")
        asr_model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
        print("WhisperX ASR model loaded.")

        print(f"Loading audio with WhisperX: {audio_path}...")
        audio = whisperx.load_audio(audio_path)
        print("Audio loaded.")

        print("Transcribing audio...")
        result_asr = asr_model.transcribe(audio, batch_size=batch_size)
        lang = result_asr.get("language", "unknown")
        print(f"Transcription complete. Detected language: {lang}")

        print("Aligning transcriptions...")
        model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
        aligned = whisperx.align(result_asr["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        print("Alignment complete.")
        return aligned
    except Exception as e:
        eprint(f"Error during WhisperX processing: {e}")
        eprint("Ensure whisperx, ffmpeg ve torch (CUDA varsa) doğru kurulu.")
        raise


def stitch_transcript(diarization, aligned) -> List[str]:
    print("\n--- Generating Final Transcript ---")
    if diarization is None or aligned is None:
        raise RuntimeError("Missing diarization or alignment results.")

    word_segments: List[Dict[str, Any]] = []
    for seg in aligned.get("segments", []):
        if seg.get("words"):
            word_segments.extend(seg["words"])  # 'word','start','end','score'

    sorted_turns = sorted(diarization.itertracks(yield_label=True), key=lambda x: x[0].start)

    lines: List[str] = []
    for turn, _, speaker in sorted_turns:
        ts, te = turn.start, turn.end
        turn_words: List[str] = []
        for w in word_segments:
            if "start" in w and "end" in w:
                mid = w["start"] + (w["end"] - w["start"]) / 2
                if ts <= mid < te:
                    token = str(w.get("word", "")).strip()
                    if token:
                        turn_words.append(token)
        if turn_words:
            text = " ".join(turn_words).strip()
            lines.append(f"{speaker} ({ts:.2f}s - {te:.2f}s): {text}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR + diarization with WhisperX and pyannote")

    # I/O
    parser.add_argument("--audio", type=str, required=True,
                        help="Yerel ses dosyası yolu (wav/mp3 vs.) — ZORUNLU")
    parser.add_argument("--audio-url", type=str, default=None,
                        help="Opsiyonel: Dosya yoksa indirilecek URL")
    parser.add_argument("--download-timeout", type=int, default=30,
                        help="İndirme zaman aşımı (sn)")
    parser.add_argument("--save-transcript", type=str, default=None,
                        help="Opsiyonel: Transcript'i .txt olarak kaydet")

    # Modeller & token
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"),
                        help="Hugging Face token (varsayılan: $HF_TOKEN)")
    parser.add_argument("--diarization-model", type=str, default="pyannote/speaker-diarization-3.1",
                        help="pyannote diarization pipeline adı")
    parser.add_argument("--whisper-model", type=str, default="medium",
                        help="WhisperX/Whisper model boyutu (tiny/base/small/medium/large-v2/large-v3)")

    # Performans
    parser.add_argument("--batch-size", type=int, default=16,
                        help="ASR batch size (varsayılan: 16)")
    parser.add_argument("--compute-type", type=str, default=None, choices=["float32", "float16", "int8"],
                        help="WhisperX compute type. Varsayılan: cihaza göre otomatik")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                        help="Zorla cihaz seçimi (varsayılan: otomatik)")

    # Deterministik ayarlar
    parser.add_argument("--seed", type=int, default=None,
                        help="Deterministik sonuçlar için seed değeri (varsayılan: rastgele)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Tamamen deterministik modda çalıştır")

    return parser.parse_args()


def main():
    args = parse_args()

    # Deterministik ayarlar
    if args.deterministic or args.seed is not None:
        seed = args.seed if args.seed is not None else 42
        set_deterministic(seed)
        print(f"Deterministic mode enabled with seed: {seed}")

    device = pick_device(args.device)
    print(f"Using device: {device}")

    compute_type = args.compute_type or default_compute_type(device)
    print(f"Compute Type: {compute_type}")

    if not args.hf_token:
        eprint("WARNING: HF token sağlanmadı. Gated modeller için gerekli olabilir.")

    if not os.path.exists(args.audio):
        if args.audio_url:
            download_audio(args.audio_url, args.audio, timeout=args.download_timeout)
        else:
            eprint(f"Audio file not found: {args.audio} and no --audio-url provided.")
            sys.exit(1)

    diarization = run_diarization(
        audio_path=args.audio,
        model_name=args.diarization_model,
        hf_token=args.hf_token,
        device=device,
    )

    aligned = run_asr_and_align(
        audio_path=args.audio,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        compute_type=compute_type,
        device=device,
    )

    lines = stitch_transcript(diarization, aligned)

    print("\nFinal Transcript:")
    for line in lines:
        print(line)

    if args.save_transcript:
        try:
            with open(args.save_transcript, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"\nTranscript saved to: {args.save_transcript}")
        except OSError as exc:
            eprint(f"Failed to write transcript file: {exc}")


if __name__ == "__main__":
    main()
