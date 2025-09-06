#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app-pipeline.py

Gerçek pipeline akışı - her adımın çıktısı sonrakine aktarılır.
Akış:
  DownloadStep → ExtractAudioStep → WhisperXDiarizeStep → ASRStep → OverlapDetectionStep
  → MergeSentencesStep (out.jsonl) → BuildRefVoicesFromJSONLStep → XTTSPerSegmentStep
  → PerfectMixStepWrapper → LipSyncStepWrapper
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Pipeline_monolith'teki mevcut adımları kullan
from pipeline_monolith import (
    DownloadStep, 
    ExtractAudioStep, 
    WhisperXDiarizeStep, 
    ASRStep, 
    OverlapDetectionStep,
    PerfectMixStepWrapper, 
    LipSyncStepWrapper, 
    make_ctx, 
    run_steps
)

logger = logging.getLogger("app-pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------- Yardımcılar --------------------------- #

def ensure_ffmpeg() -> None:
    """FFmpeg'in sistemde olduğunu kontrol et"""
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-version"], 
                      check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError("ffmpeg PATH'te bulunamadı") from e


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """JSONL dosyasını oku"""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """JSONL dosyasına yaz"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------- Cümle Birleştirme Adımı -------------------- #

class MergeSentencesStep:
    """
    OverlapDetectionStep'ten gelen kelime düzeyi çıktıyı alır,
    cümle bazlı JSONL üretir (out.jsonl).
    
    Pipeline'dan gelen artifacts['words'] kullanılır.
    """
    name = "MergeSentences"

    def __init__(self, gap_threshold: float = 1.0, out_filename: str = "out.jsonl"):
        self.gap_threshold = gap_threshold
        self.out_filename = out_filename

    def run(self, ctx: Dict[str, Any]) -> None:
        """Pipeline context'inden words alıp cümle birleştir"""
        
        # OverlapDetectionStep'ten gelen kelimeler
        words = ctx["artifacts"].get("words", [])
        if not words:
            raise RuntimeError("MergeSentencesStep: 'words' bulunamadı (OverlapDetectionStep çıktısı bekleniyor)")

        logger.info("Cümle birleştirme başlıyor: %d kelime", len(words))
        
        sentences = []
        current_sentence = None
        
        for word in words:
            speaker = word.get("speaker")
            start = float(word["start"])
            end = float(word["end"])
            text = (word.get("word") or "").strip()
            
            if not text:
                continue
                
            if current_sentence is None:
                # İlk cümle
                current_sentence = {
                    "text": text,
                    "start": start,
                    "end": end,
                    "speaker": speaker
                }
            else:
                # Aynı konuşmacı ve boşluk eşiğinden az ise birleştir
                gap = start - current_sentence["end"]
                if speaker == current_sentence["speaker"] and gap < self.gap_threshold:
                    current_sentence["text"] += " " + text
                    current_sentence["end"] = end
                else:
                    # Yeni cümle başlat
                    sentences.append(current_sentence)
                    current_sentence = {
                        "text": text,
                        "start": start,
                        "end": end,
                        "speaker": speaker
                    }
        
        # Son cümleyi ekle
        if current_sentence and current_sentence.get("text"):
            sentences.append(current_sentence)
        
        # JSONL olarak kaydet
        temp_dir = Path(ctx["temp_dir"])
        out_path = temp_dir / self.out_filename
        write_jsonl(out_path, sentences)
        
        # Artifacts'e ekle - sonraki adımlar için
        ctx["artifacts"]["sentences"] = sentences
        ctx["artifacts"]["out_jsonl"] = str(out_path)
        
        logger.info("Cümle birleştirme tamamlandı: %d cümle -> %s", 
                   len(sentences), out_path)


# -------------------- Referans Ses Çıkarma Adımı -------------------- #

class BuildRefVoicesFromJSONLStep:
    """
    out.jsonl'deki cümlelerden her konuşmacı için ~9 saniyelik
    referans ses örnekleri çıkarır.
    
    Pipeline'dan gelen artifacts['sentences'] ve artifacts['original_audio'] kullanılır.
    """
    name = "BuildRefVoicesFromJSONL"

    def __init__(self, ref_duration: float = 9.0, sample_rate: int = 16000):
        self.ref_duration = ref_duration
        self.sample_rate = sample_rate

    def _extract_audio_segment(self, audio_path: Path, start: float, 
                               duration: float, output_path: Path) -> None:
        """FFmpeg ile ses segmenti çıkar"""
        ensure_ffmpeg()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(audio_path),
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-c:a", "pcm_s16le",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg hatası: {result.stderr}")

    def run(self, ctx: Dict[str, Any]) -> None:
        """Pipeline context'inden sentences alıp referans sesler oluştur"""
        
        # Orijinal ses dosyası
        audio_path = ctx["artifacts"].get("original_audio")
        if not audio_path:
            raise RuntimeError("BuildRefVoicesFromJSONL: Ses dosyası bulunamadı")
        audio_path = Path(audio_path)
        
        # Cümleler
        sentences = ctx["artifacts"].get("sentences", [])
        if not sentences:
            raise RuntimeError("BuildRefVoicesFromJSONL: Cümle bulunamadı")
        
        logger.info("Referans ses çıkarma başlıyor: %d cümle", len(sentences))
        
        # Konuşmacılara göre grupla
        speaker_segments = {}
        for sent in sentences:
            speaker = sent.get("speaker", "SPEAKER_00")
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(sent)
        
        # Her konuşmacı için referans ses oluştur
        temp_dir = Path(ctx["temp_dir"])
        voices_dir = temp_dir / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        
        ref_voices = {}
        
        for speaker, segments in speaker_segments.items():
            # Segmentleri zamana göre sırala
            segments = sorted(segments, key=lambda x: float(x["start"]))
            
            # İlk 9 saniyeyi bulmaya çalış
            total_duration = 0.0
            selected_segments = []
            
            for seg in segments:
                seg_duration = float(seg["end"]) - float(seg["start"])
                selected_segments.append(seg)
                total_duration += seg_duration
                
                if total_duration >= self.ref_duration:
                    break
            
            if not selected_segments:
                logger.warning("Konuşmacı %s için segment bulunamadı", speaker)
                continue
            
            # Seçilen segmentlerin başlangıç ve bitiş zamanları
            ref_start = float(selected_segments[0]["start"])
            ref_end = float(selected_segments[-1]["end"])
            ref_duration = min(self.ref_duration, ref_end - ref_start)
            
            # Referans ses dosyası
            ref_wav_path = voices_dir / f"{speaker}.wav"
            
            try:
                self._extract_audio_segment(audio_path, ref_start, 
                                           ref_duration, ref_wav_path)
                ref_voices[speaker] = str(ref_wav_path)
                logger.info("Referans ses oluşturuldu: %s (%.2f sn)", 
                           speaker, ref_duration)
            except Exception as e:
                logger.error("Referans ses çıkarma hatası (%s): %s", speaker, e)
        
        if not ref_voices:
            raise RuntimeError("Hiçbir referans ses oluşturulamadı")
        
        # Artifacts'e ekle
        ctx["artifacts"]["ref_voices"] = ref_voices
        ctx["artifacts"]["ref_voices_dir"] = str(voices_dir)
        
        logger.info("Referans ses çıkarma tamamlandı: %d konuşmacı", len(ref_voices))


# -------------------- XTTS Sentez Adımı -------------------- #

class XTTSPerSegmentStep:
    """
    Her cümle için XTTS ile sentez yapar ve zamanlamaya göre
    hizalanmış tek bir WAV dosyası oluşturur.
    
    Pipeline'dan gelen artifacts['sentences'] ve artifacts['ref_voices'] kullanılır.
    """
    name = "XTTSPerSegment"

    def __init__(self, tts_engine: str = "xtts", sample_rate: int = 16000,
                 voice_map: Optional[Dict[str, str]] = None):
        self.tts_engine = tts_engine
        self.sample_rate = sample_rate
        self.voice_map = voice_map or {}

    def _load_tts_engine(self):
        """TTS motorunu yükle"""
        try:
            from core.models.tts.xtts import XTTSEngine
            return XTTSEngine()
        except ImportError:
            logger.warning("XTTS motoru bulunamadı, mock TTS kullanılıyor")
            return None

    def _generate_tts_segment(self, text: str, speaker_wav: str, 
                             duration: float) -> 'AudioSegment':
        """Tek segment için TTS üret"""
        from pydub import AudioSegment
        
        if self.tts_engine_instance:
            try:
                # Gerçek XTTS kullan
                audio = self.tts_engine_instance.tts(
                    text=text,
                    speaker_wav=speaker_wav
                )
                return audio
            except Exception as e:
                logger.warning("TTS hatası: %s", e)
        
        # Fallback: sessizlik
        duration_ms = int(duration * 1000)
        return AudioSegment.silent(duration=duration_ms)

    def run(self, ctx: Dict[str, Any]) -> None:
        """Pipeline context'inden sentences ve ref_voices alıp TTS üret"""
        from pydub import AudioSegment
        
        sentences = ctx["artifacts"].get("sentences", [])
        if not sentences:
            raise RuntimeError("XTTSPerSegmentStep: Cümle bulunamadı")
        
        ref_voices = ctx["artifacts"].get("ref_voices", {})
        if not ref_voices:
            logger.warning("Referans ses bulunamadı, varsayılan ses kullanılacak")
        
        logger.info("TTS sentezi başlıyor: %d cümle", len(sentences))
        
        # TTS motorunu yükle
        self.tts_engine_instance = self._load_tts_engine()
        
        # Segment dizini
        temp_dir = Path(ctx["temp_dir"])
        segments_dir = temp_dir / "tts_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Her cümle için TTS üret
        tts_segments = []
        
        for idx, sent in enumerate(sentences):
            text = sent.get("text", "")
            speaker = sent.get("speaker", "SPEAKER_00")
            start_time = float(sent["start"])
            end_time = float(sent["end"])
            duration = end_time - start_time
            
            # Konuşmacı için referans ses
            speaker_wav = self.voice_map.get(speaker) or ref_voices.get(speaker)
            
            if not speaker_wav:
                logger.warning("Konuşmacı %s için referans ses yok", speaker)
                # Varsa ilk konuşmacının sesini kullan
                if ref_voices:
                    speaker_wav = list(ref_voices.values())[0]
            
            # TTS üret
            audio_segment = self._generate_tts_segment(text, speaker_wav, duration)
            
            # Segment dosyasını kaydet
            segment_path = segments_dir / f"seg_{idx:04d}_{speaker}.wav"
            audio_segment.export(segment_path, format="wav")
            
            tts_segments.append({
                "start": start_time,
                "audio": audio_segment,
                "path": str(segment_path),
                "speaker": speaker
            })
            
            logger.debug("Segment %d sentezlendi: %s", idx, speaker)
        
        # Tüm segmentleri zamanlamaya göre birleştir
        if not tts_segments:
            raise RuntimeError("Hiçbir TTS segment üretilemedi")
        
        # Toplam süreyi hesapla
        max_end_time = max(seg["start"] + len(seg["audio"]) / 1000.0 
                          for seg in tts_segments)
        total_duration_ms = int(max_end_time * 1000) + 1000  # 1 sn ekstra
        
        # Boş taban oluştur
        merged_audio = AudioSegment.silent(duration=total_duration_ms)
        
        # Segmentleri yerleştir
        for seg in tts_segments:
            position_ms = int(seg["start"] * 1000)
            merged_audio = merged_audio.overlay(seg["audio"], position=position_ms)
        
        # Birleştirilmiş dosyayı kaydet
        output_path = temp_dir / "tts_merged.wav"
        merged_audio.set_channels(1).set_frame_rate(self.sample_rate)
        merged_audio.export(output_path, format="wav")
        
        # Artifacts'e ekle
        ctx["artifacts"]["synth_audio"] = str(output_path)
        ctx["artifacts"]["tts_segments"] = [seg["path"] for seg in tts_segments]
        
        logger.info("TTS sentezi tamamlandı: %d segment -> %s", 
                   len(tts_segments), output_path)


# -------------------- Çeviri Adımı (Opsiyonel) -------------------- #

class TranslationStep:
    """
    Cümleleri hedef dile çevirir.
    Pipeline'dan gelen artifacts['sentences'] kullanılır.
    """
    name = "Translation"

    def __init__(self, engine: str = "none", source_lang: str = "en", 
                 target_lang: str = "tr"):
        self.engine = engine
        self.source_lang = source_lang
        self.target_lang = target_lang

    def run(self, ctx: Dict[str, Any]) -> None:
        """Pipeline context'inden sentences alıp çevir"""
        
        sentences = ctx["artifacts"].get("sentences", [])
        if not sentences:
            logger.info("Çeviri: Cümle bulunamadı, atlanıyor")
            return
        
        if self.engine == "none":
            logger.info("Çeviri motoru 'none', atlanıyor")
            return
        
        logger.info("Çeviri başlıyor: %d cümle (%s -> %s)", 
                   len(sentences), self.source_lang, self.target_lang)
        
        if self.engine == "googletrans":
            try:
                from googletrans import Translator
                translator = Translator()
                
                # Her cümleyi çevir
                for sent in sentences:
                    original_text = sent.get("text", "")
                    if original_text:
                        try:
                            result = translator.translate(
                                original_text, 
                                src=self.source_lang,
                                dest=self.target_lang
                            )
                            sent["text_original"] = original_text
                            sent["text"] = result.text
                        except Exception as e:
                            logger.warning("Çeviri hatası: %s", e)
                            
            except ImportError:
                logger.warning("googletrans kurulu değil, çeviri atlanıyor")
        
        # Güncellenen cümleleri artifacts'e geri yaz
        ctx["artifacts"]["sentences"] = sentences
        
        # JSONL'i güncelle
        if "out_jsonl" in ctx["artifacts"]:
            out_path = Path(ctx["artifacts"]["out_jsonl"])
            write_jsonl(out_path, sentences)
            logger.info("out.jsonl güncellendi: %s", out_path)


# ----------------------------- Ana Pipeline ----------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Uçtan uca dublaj pipeline'ı"
    )
    
    # Giriş
    parser.add_argument("--input", help="YouTube URL veya video dosyası")
    parser.add_argument("--audio", help="Doğrudan ses dosyası (WAV)")
    
    # Çıktı
    parser.add_argument("--out-jsonl", default="out.jsonl", 
                       help="Cümle bazlı JSONL çıktısı")
    parser.add_argument("--temp-dir", default="./_temp_app",
                       help="Geçici dosyalar klasörü")
    
    # Diller
    parser.add_argument("--source-lang", default="en", help="Kaynak dil")
    parser.add_argument("--target-lang", default="tr", help="Hedef dil")
    
    # Çeviri
    parser.add_argument("--translator", default="none",
                       choices=["none", "googletrans"],
                       help="Çeviri motoru")
    
    # TTS
    parser.add_argument("--tts", default="xtts", help="TTS motoru")
    parser.add_argument("--voice-map", type=json.loads, default=None,
                       help='Konuşmacı ses eşlemesi (JSON)')
    
    # Mix & Lipsync
    parser.add_argument("--do-mix", action="store_true",
                       help="PerfectMix uygula")
    parser.add_argument("--do-lipsync", action="store_true",
                       help="LipSync uygula")
    
    # Diğer
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                       help="Hugging Face token")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Geçici klasör
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Yapılandırma
    config = {
        "input": args.input,
        "audio": args.audio,
        "temp_dir": str(temp_dir),
        "hf_token": args.hf_token,
        "sample_rate": 16000,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
    }
    
    # Pipeline adımları
    steps = []
    
    # 1. İndirme ve ses çıkarma (eğer gerekiyorsa)
    if args.input and not args.audio:
        steps.extend([
            DownloadStep(),
            ExtractAudioStep()
        ])
    
    # 2. VAD + Diarization + ASR + Overlap Detection
    # pipeline_monolith'teki adımları kullan
    steps.extend([
        WhisperXDiarizeStep(
            model_name="large-v2",
            compute_type="int8",
            language=args.source_lang,
            min_chunk=1.5,
            max_chunk=3.0
        ),
        ASRStep(
            model_name="large-v3",
            device="auto",
            compute_type="auto"
        ),
        OverlapDetectionStep(use_gpu=None)
    ])
    
    # 3. Cümle birleştirme (out.jsonl üretimi)
    steps.append(
        MergeSentencesStep(
            gap_threshold=1.0,
            out_filename=args.out_jsonl
        )
    )
    
    # 4. Çeviri (opsiyonel)
    if args.translator != "none":
        steps.append(
            TranslationStep(
                engine=args.translator,
                source_lang=args.source_lang,
                target_lang=args.target_lang
            )
        )
    
    # 5. Referans ses çıkarma
    steps.append(
        BuildRefVoicesFromJSONLStep(
            ref_duration=9.0,
            sample_rate=config["sample_rate"]
        )
    )
    
    # 6. XTTS sentez
    steps.append(
        XTTSPerSegmentStep(
            tts_engine=args.tts,
            sample_rate=config["sample_rate"],
            voice_map=args.voice_map
        )
    )
    
    # 7. Mix & Lipsync (opsiyonel)
    if args.do_mix:
        steps.append(
            PerfectMixStepWrapper(
                lufs_target=-14.0,
                duck_db=-7.0,
                pan_amount=0.0
            )
        )
    
    if args.do_lipsync:
        steps.append(
            LipSyncStepWrapper(
                model_name="simple",
                model_kwargs={}
            )
        )
    
    # Pipeline'ı çalıştır
    logger.info("Pipeline başlatılıyor: %d adım", len(steps))
    ctx = make_ctx(temp_dir, config)
    
    # Eğer doğrudan ses dosyası verilmişse
    if args.audio:
        ctx["artifacts"]["original_audio"] = args.audio
        config["audio"] = args.audio
    
    # Adımları çalıştır
    run_steps(steps, ctx)
    
    # Özet rapor
    summary = {
        "config": config,
        "artifacts": ctx.get("artifacts", {}),
        "steps_completed": [step.name for step in steps],
        "output_files": {
            "out_jsonl": ctx["artifacts"].get("out_jsonl"),
            "synth_audio": ctx["artifacts"].get("synth_audio"),
            "final_video": ctx["artifacts"].get("final_video")
        }
    }
    
    summary_path = temp_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("Pipeline tamamlandı! Özet: %s", summary_path)
    
    # Önemli çıktıları göster
    print("\n" + "="*60)
    print("PIPELINE TAMAMLANDI")
    print("="*60)
    if "out_jsonl" in ctx["artifacts"]:
        print(f"Cümle dosyası: {ctx['artifacts']['out_jsonl']}")
    if "synth_audio" in ctx["artifacts"]:
        print(f"Sentezlenmiş ses: {ctx['artifacts']['synth_audio']}")
    if "final_video" in ctx["artifacts"]:
        print(f"Final video: {ctx['artifacts']['final_video']}")
    print(f"Tüm dosyalar: {temp_dir}")
    print("="*60)


if __name__ == "__main__":
    main()