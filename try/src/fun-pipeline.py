#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WhisperX: Versiyon uyumlu ASR + Diarization
"""

import whisperx
import json
from pathlib import Path
import gc

class IntegratedPipeline:
    def __init__(self, model_size="models--Systran--faster-whisper-large-v3", device="cuda"):
        self.device = device
        # Daha güvenli yükleme
        try:
            self.model = whisperx.load_model(model_size, device, compute_type="float16")
        except TypeError:
            # Eski versiyon için
            self.model =  self.model = whisperx.load_model(
                model_size,
                device,
                compute_type="float16",
                asr_options={
                    "multilingual": True,
                    "max_new_tokens": 448,
                    "clip_timestamps": "0",
                    "hallucination_silence_threshold": 0.0,
                    "hotwords": None
                }
            )
        self.diarize_model = None

    def process(self, audio_path):
        # 1. Audio yükle
        audio = whisperx.load_audio(audio_path)

        # 2. Transcribe
        print("Transcribing...")
        result = self.model.transcribe(audio, batch_size=16)
        print(f"Language detected: {result['language']}")

        # 3. Align
        print("Aligning...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False
        )

        # 4. Diarize
        print("Diarizing...")
        if self.diarize_model is None:
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token="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",
                device=self.device
            )

        diarize_segments = self.diarize_model(audio)

        # 5. Assign speakers
        print("Assigning speakers to words...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return result

    def format_output(self, result):
        """Timeline oluştur"""
        timeline = []
        current_speaker = None
        current_text = []
        current_start = None
        current_end = None

        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")

            if speaker != current_speaker:
                # Önceki segment'i kaydet
                if current_speaker is not None:
                    timeline.append({
                        "speaker": current_speaker,
                        "start": current_start,
                        "end": current_end,
                        "text": " ".join(current_text)
                    })

                # Yeni segment başlat
                current_speaker = speaker
                current_text = [segment["text"]]
                current_start = segment["start"]
                current_end = segment["end"]
            else:
                # Aynı speaker devam ediyor
                current_text.append(segment["text"])
                current_end = segment["end"]

        # Son segment'i kaydet
        if current_speaker:
            timeline.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_text)
            })

        return timeline

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="WhisperX ASR + Diarization")
    parser.add_argument("audio_file", help="Audio file path")
    parser.add_argument("-o", "--output", default="result.json", help="Output JSON file")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")

    args = parser.parse_args()

    # Dosya kontrolü
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    try:
        # Pipeline çalıştır
        pipeline = IntegratedPipeline(model_size=args.model)
        result = pipeline.process(str(audio_path))

        # Timeline oluştur
        timeline = pipeline.format_output(result)

        # Konsola yazdır
        print("\n=== SPEAKER TIMELINE ===")
        for i, turn in enumerate(timeline, 1):
            text_preview = turn['text'][:80] + "..." if len(turn['text']) > 80 else turn['text']
            print(f"{i}. [{turn['speaker']}] {turn['start']:.1f}-{turn['end']:.1f}s: {text_preview}")

        # JSON kaydet
        output_data = {
            "audio_file": str(audio_path),
            "timeline": timeline,
            "segments": result["segments"]
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()