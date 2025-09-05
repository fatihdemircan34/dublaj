#!/bin/bash

# Enhanced ASR + Diarization Kurulum Script
# Bu script "try" klasöründe Enhanced ASR sistemini kurar

set -e  # Hata durumunda dur

# Renkli çıktı için
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logo
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               Enhanced ASR + Diarization Setup              ║"
echo "║                     WhisperX + Pyannote                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Klasör adı
INSTALL_DIR="try"
CURRENT_DIR=$(pwd)

echo -e "${YELLOW}📁 Kurulum dizini: ${CURRENT_DIR}/${INSTALL_DIR}${NC}"

# Sistem kontrolü
echo -e "${BLUE}🔍 Sistem kontrolleri...${NC}"

# Python kontrolü
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 bulunamadı!${NC}"
    echo "Lütfen Python 3.8+ kurun"
    exit 1
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python ${PYTHON_VERSION} bulundu${NC}"
fi

# pip kontrolü
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 bulunamadı!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ pip3 bulundu${NC}"
fi

# Git kontrolü (opsiyonel)
if command -v git &> /dev/null; then
    echo -e "${GREEN}✅ Git bulundu${NC}"
else
    echo -e "${YELLOW}⚠️ Git bulunamadı (opsiyonel)${NC}"
fi

# FFmpeg kontrolü
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}⚠️ FFmpeg bulunamadı${NC}"
    echo "Audio dönüştürme için FFmpeg gerekebilir:"
    echo "Ubuntu/Debian: sudo apt install ffmpeg"
    echo "macOS: brew install ffmpeg"
    echo "Windows: https://ffmpeg.org/download.html"
else
    echo -e "${GREEN}✅ FFmpeg bulundu${NC}"
fi

# CUDA kontrolü
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU tespit edildi${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    USE_CUDA=true
else
    echo -e "${YELLOW}⚠️ NVIDIA GPU bulunamadı, CPU kullanılacak${NC}"
    USE_CUDA=false
fi

echo ""

# Klasör oluştur
echo -e "${BLUE}📁 Klasör yapısı oluşturuluyor...${NC}"

if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}⚠️ ${INSTALL_DIR} klasörü zaten mevcut${NC}"
    read -p "Üzerine yazmak istiyor musunuz? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$INSTALL_DIR"
    else
        echo -e "${RED}❌ Kurulum iptal edildi${NC}"
        exit 1
    fi
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Klasör yapısını oluştur
mkdir -p {src,models,data,output,logs,configs,scripts}

echo -e "${GREEN}✅ Klasör yapısı oluşturuldu${NC}"

# Virtual environment oluştur
echo -e "${BLUE}🐍 Python virtual environment oluşturuluyor...${NC}"

python3 -m venv venv

# Virtual environment'ı aktif et
source venv/bin/activate

echo -e "${GREEN}✅ Virtual environment oluşturuldu${NC}"

# Gereksinimler dosyası oluştur
echo -e "${BLUE}📋 Gereksinimler dosyası oluşturuluyor...${NC}"

cat > requirements.txt << EOF
# Core dependencies
torch>=2.1.0
torchaudio>=2.1.0
numpy>=1.21.0
pandas>=1.3.0

# Audio processing
librosa>=0.9.0
soundfile>=0.12.1
ffmpeg-python>=0.2.0

# ML/AI
scikit-learn>=1.0.0
scipy>=1.7.0

# WhisperX and dependencies
whisperx>=3.1.1
faster-whisper>=1.0.3

# Pyannote
pyannote.audio>=3.1.0

# Utilities
click>=8.0.0
tqdm>=4.62.0
pydantic>=2.0.0
python-dotenv>=0.19.0

# Development
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
EOF

echo -e "${GREEN}✅ requirements.txt oluşturuldu${NC}"

# PyTorch kurulumu (CUDA desteğine göre)
echo -e "${BLUE}🔥 PyTorch kuruluyor...${NC}"

if [ "$USE_CUDA" = true ]; then
    echo "CUDA destekli PyTorch kuruluyor..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "CPU versiyonu PyTorch kuruluyor..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✅ PyTorch kuruldu${NC}"

# Diğer gereksinimler
echo -e "${BLUE}📦 Diğer paketler kuruluyor...${NC}"

pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✅ Tüm paketler kuruldu${NC}"

# Ana Python dosyalarını oluştur
echo -e "${BLUE}📝 Kaynak dosyalar oluşturuluyor...${NC}"

# enhanced_asr_diarization.py (Standalone versiyon)
cat > src/enhanced_asr_diarization.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced ASR + Diarization System
WhisperX (word-level alignment) + Pyannote (speaker diarization)
Geliştirilmiş speaker assignment ve overlap detection
"""

import os
import json
import torch
import whisperx
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pyannote.audio import Pipeline
import warnings
from pyannote.audio.utils.reproducibility import ReproducibilityWarning
warnings.filterwarnings("ignore", category=ReproducibilityWarning)

@dataclass
class WordSegment:
    word: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None
    speaker_confidence: float = 0.0
    is_overlap: bool = False

@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float
    confidence: Optional[float] = None

@dataclass
class ASRResult:
    words: List[WordSegment]
    segments: List[Dict]
    language: str

class EnhancedASRDiarization:
    def __init__(self, device: str = "auto", model_size: str = "large-v2"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.whisperx_model = None
        self.align_model = None
        self.diarization_pipeline = None

        print(f"🔧 Cihaz: {self.device}")
        print(f"📏 Model boyutu: {model_size}")

    def load_models(self):
        """Modelleri yükle"""
        print("📦 Modeller yükleniyor...")

        # WhisperX ASR model
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.whisperx_model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=compute_type
        )
        print("✓ WhisperX ASR modeli yüklendi")

        # Pyannote diarization pipeline
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            print("✓ Pyannote diarization modeli yüklendi")
        else:
            print("⚠️ HF_TOKEN bulunamadı, diarization atlanacak")

    def transcribe_with_whisperx(self, audio_path: str, language: str = None) -> ASRResult:
        """WhisperX ile word-level transcription"""
        print("🎤 WhisperX transkripsiyon başlıyor...")

        # Initial transcription
        result = self.whisperx_model.transcribe(
            audio_path,
            batch_size=16,
            language=language
        )

        detected_language = result["language"]
        print(f"🗣️ Algılanan dil: {detected_language}")

        # Load alignment model for detected language
        try:
            self.align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )

            # Perform alignment
            result_aligned = whisperx.align(
                result["segments"],
                self.align_model,
                metadata,
                audio_path,
                self.device,
                return_char_alignments=False
            )

            # Extract word-level information
            words = []
            segments = []

            for i, seg in enumerate(result_aligned["segments"]):
                segment_words = []

                for word_info in seg.get("words", []):
                    if word_info.get("start") is not None and word_info.get("end") is not None:
                        word = WordSegment(
                            word=word_info["word"].strip(),
                            start=float(word_info["start"]),
                            end=float(word_info["end"]),
                            confidence=float(word_info.get("probability", 0.9))
                        )
                        words.append(word)
                        segment_words.append(word)

                segments.append({
                    "id": f"seg_{i:04d}",
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": seg.get("text", "").strip(),
                    "words": segment_words
                })

            print(f"✓ {len(words)} kelime, {len(segments)} segment transkribe edildi")

            return ASRResult(
                words=words,
                segments=segments,
                language=detected_language
            )

        except Exception as e:
            print(f"⚠️ Alignment hatası: {e}")
            # Fallback without alignment
            return self._fallback_transcription(result, detected_language)

    def _fallback_transcription(self, result: Dict, language: str) -> ASRResult:
        """Alignment başarısız olursa fallback"""
        words = []
        segments = []

        for i, seg in enumerate(result["segments"]):
            text = seg.get("text", "").strip()
            if not text:
                continue

            # Simple word timing estimation
            word_list = text.split()
            duration = seg["end"] - seg["start"]
            word_duration = duration / len(word_list) if word_list else 0

            segment_words = []
            for j, word_text in enumerate(word_list):
                word = WordSegment(
                    word=word_text,
                    start=seg["start"] + j * word_duration,
                    end=seg["start"] + (j + 1) * word_duration,
                    confidence=0.8  # Lower confidence for estimated timing
                )
                words.append(word)
                segment_words.append(word)

            segments.append({
                "id": f"seg_{i:04d}",
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
                "words": segment_words
            })

        return ASRResult(words=words, segments=segments, language=language)

    def diarize_with_pyannote(self, audio_path: str, min_speakers: int = None, max_speakers: int = None) -> List[SpeakerSegment]:
        """Pyannote ile speaker diarization"""
        if not self.diarization_pipeline:
            print("⚠️ Diarization modeli yüklü değil, varsayılan speaker atanıyor")
            return [SpeakerSegment("SPEAKER_00", 0.0, 300.0)]  # 5 dakikalık varsayılan

        print("👥 Speaker diarization başlıyor...")

        # Diarization parameters
        params = {}
        if min_speakers:
            params["min_speakers"] = min_speakers
        if max_speakers:
            params["max_speakers"] = max_speakers

        # Run diarization
        diarization = self.diarization_pipeline(audio_path, **params)

        # Convert to speaker segments
        speaker_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append(SpeakerSegment(
                speaker=str(speaker),
                start=float(segment.start),
                end=float(segment.end)
            ))

        print(f"✓ {len(speaker_segments)} diarization segmenti bulundu")
        print(f"🎭 Konuşmacılar: {set(seg.speaker for seg in speaker_segments)}")

        return speaker_segments

    def assign_speakers_to_words(self, words: List[WordSegment], speaker_segments: List[SpeakerSegment]) -> List[WordSegment]:
        """Kelimelere speaker atama - geliştirilmiş algoritma"""
        print("🔗 Kelimeler speaker'lara atanıyor...")

        if not speaker_segments:
            return words

        # Create speaker intervals for efficient lookup
        speaker_intervals = []
        for seg in speaker_segments:
            speaker_intervals.append((seg.start, seg.end, seg.speaker))

        # Sort by start time for efficient searching
        speaker_intervals.sort(key=lambda x: x[0])

        assigned_words = []
        overlap_threshold = 0.25  # Minimum overlap ratio for secondary speaker

        for word in words:
            word_mid = (word.start + word.end) / 2

            # Find all overlapping speaker segments
            overlapping_speakers = []

            for start, end, speaker in speaker_intervals:
                if start > word.end:
                    break  # No more possible overlaps

                if end >= word.start:  # There's some overlap
                    # Calculate overlap ratio
                    overlap_start = max(word.start, start)
                    overlap_end = min(word.end, end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    word_duration = word.end - word.start

                    if word_duration > 0:
                        overlap_ratio = overlap_duration / word_duration
                        overlapping_speakers.append((speaker, overlap_ratio, overlap_duration))

            if not overlapping_speakers:
                # No speaker found, assign to nearest
                word.speaker = self._find_nearest_speaker(word, speaker_intervals)
                word.speaker_confidence = 0.3
            else:
                # Sort by overlap ratio
                overlapping_speakers.sort(key=lambda x: x[1], reverse=True)

                # Assign primary speaker
                primary_speaker, primary_ratio, _ = overlapping_speakers[0]
                word.speaker = primary_speaker
                word.speaker_confidence = min(primary_ratio, 1.0)

                # Check for significant overlap with secondary speaker
                if len(overlapping_speakers) > 1:
                    secondary_ratio = overlapping_speakers[1][1]
                    if secondary_ratio >= overlap_threshold:
                        word.is_overlap = True

            assigned_words.append(word)

        # Post-process: smooth speaker assignments
        assigned_words = self._smooth_speaker_assignments(assigned_words)

        return assigned_words

    def _find_nearest_speaker(self, word: WordSegment, speaker_intervals: List[Tuple[float, float, str]]) -> str:
        """En yakın speaker'ı bul"""
        word_mid = (word.start + word.end) / 2
        min_distance = float('inf')
        nearest_speaker = "SPEAKER_00"

        for start, end, speaker in speaker_intervals:
            # Distance to speaker segment
            if word_mid < start:
                distance = start - word_mid
            elif word_mid > end:
                distance = word_mid - end
            else:
                distance = 0  # Word is inside segment

            if distance < min_distance:
                min_distance = distance
                nearest_speaker = speaker

        return nearest_speaker

    def _smooth_speaker_assignments(self, words: List[WordSegment], window_size: int = 3) -> List[WordSegment]:
        """Speaker atamalarını yumuşat (kısa speaker değişimlerini düzelt)"""
        if len(words) < window_size:
            return words

        smoothed_words = words.copy()

        for i in range(len(words)):
            # Get surrounding words
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(words), i + window_size // 2 + 1)
            window_words = words[start_idx:end_idx]

            # Count speakers in window
            speaker_counts = {}
            for w in window_words:
                if w.speaker:
                    speaker_counts[w.speaker] = speaker_counts.get(w.speaker, 0) + 1

            if speaker_counts:
                # Get majority speaker
                majority_speaker = max(speaker_counts, key=speaker_counts.get)
                majority_count = speaker_counts[majority_speaker]

                # If current word is minority and has low confidence, reassign
                current_word = words[i]
                if (current_word.speaker != majority_speaker and
                    current_word.speaker_confidence < 0.7 and
                    majority_count > len(window_words) // 2):

                    smoothed_words[i].speaker = majority_speaker
                    smoothed_words[i].speaker_confidence = 0.6  # Medium confidence for smoothed

        return smoothed_words

    def generate_speaker_timeline(self, words: List[WordSegment]) -> List[Dict]:
        """Speaker timeline oluştur"""
        if not words:
            return []

        timeline = []
        current_speaker = None
        current_start = None
        current_end = None
        current_words = []

        for word in words:
            if word.speaker != current_speaker:
                # Flush current segment
                if current_speaker is not None:
                    timeline.append({
                        "speaker": current_speaker,
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                        "duration": round(current_end - current_start, 3),
                        "text": " ".join(w.word for w in current_words),
                        "word_count": len(current_words),
                        "words": [
                            {
                                "word": w.word,
                                "start": round(w.start, 3),
                                "end": round(w.end, 3),
                                "confidence": round(w.confidence, 3),
                                "speaker_confidence": round(w.speaker_confidence, 3),
                                "is_overlap": w.is_overlap
                            } for w in current_words
                        ]
                    })

                # Start new segment
                current_speaker = word.speaker
                current_start = word.start
                current_end = word.end
                current_words = [word]
            else:
                # Continue current segment
                current_end = word.end
                current_words.append(word)

        # Flush last segment
        if current_speaker is not None:
            timeline.append({
                "speaker": current_speaker,
                "start": round(current_start, 3),
                "end": round(current_end, 3),
                "duration": round(current_end - current_start, 3),
                "text": " ".join(w.word for w in current_words),
                "word_count": len(current_words),
                "words": [
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "confidence": round(w.confidence, 3),
                        "speaker_confidence": round(w.speaker_confidence, 3),
                        "is_overlap": w.is_overlap
                    } for w in current_words
                ]
            })

        return timeline

    def detect_overlaps(self, words: List[WordSegment]) -> List[Dict]:
        """Overlap bölgelerini tespit et"""
        overlaps = []
        overlap_words = [w for w in words if w.is_overlap]

        if not overlap_words:
            return overlaps

        # Group consecutive overlap words
        current_overlap = None

        for word in overlap_words:
            if current_overlap is None:
                current_overlap = {
                    "start": word.start,
                    "end": word.end,
                    "words": [word],
                    "speakers": {word.speaker}
                }
            elif word.start <= current_overlap["end"] + 0.1:  # 100ms tolerance
                # Extend current overlap
                current_overlap["end"] = word.end
                current_overlap["words"].append(word)
                current_overlap["speakers"].add(word.speaker)
            else:
                # Finish current overlap and start new one
                overlaps.append({
                    "start": round(current_overlap["start"], 3),
                    "end": round(current_overlap["end"], 3),
                    "duration": round(current_overlap["end"] - current_overlap["start"], 3),
                    "speakers": list(current_overlap["speakers"]),
                    "text": " ".join(w.word for w in current_overlap["words"]),
                    "word_count": len(current_overlap["words"])
                })

                current_overlap = {
                    "start": word.start,
                    "end": word.end,
                    "words": [word],
                    "speakers": {word.speaker}
                }

        # Add last overlap
        if current_overlap:
            overlaps.append({
                "start": round(current_overlap["start"], 3),
                "end": round(current_overlap["end"], 3),
                "duration": round(current_overlap["end"] - current_overlap["start"], 3),
                "speakers": list(current_overlap["speakers"]),
                "text": " ".join(w.word for w in current_overlap["words"]),
                "word_count": len(current_overlap["words"])
            })

        return overlaps

    def process(self, audio_path: str, output_dir: str = "output",
                language: str = None, min_speakers: int = None, max_speakers: int = None) -> Dict:
        """Ana işlem fonksiyonu"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"🚀 Enhanced ASR+Diarization başlıyor...")
        print(f"📁 Çıktı dizini: {output_dir}")

        # Load models
        self.load_models()

        # Step 1: ASR with WhisperX
        asr_result = self.transcribe_with_whisperx(audio_path, language)

        # Step 2: Speaker diarization with Pyannote
        speaker_segments = self.diarize_with_pyannote(audio_path, min_speakers, max_speakers)

        # Step 3: Assign speakers to words
        words_with_speakers = self.assign_speakers_to_words(asr_result.words, speaker_segments)

        # Step 4: Generate timeline
        timeline = self.generate_speaker_timeline(words_with_speakers)

        # Step 5: Detect overlaps
        overlaps = self.detect_overlaps(words_with_speakers)

        # Generate statistics
        speakers = set(w.speaker for w in words_with_speakers if w.speaker)
        total_words = len(words_with_speakers)
        overlap_words = len([w for w in words_with_speakers if w.is_overlap])
        total_duration = max(w.end for w in words_with_speakers) if words_with_speakers else 0

        stats = {
            "total_duration": round(total_duration, 2),
            "total_words": total_words,
            "total_speakers": len(speakers),
            "speakers": sorted(list(speakers)),
            "total_segments": len(timeline),
            "overlap_segments": len(overlaps),
            "overlap_words": overlap_words,
            "overlap_percentage": round((overlap_words / total_words) * 100, 1) if total_words > 0 else 0,
            "language": asr_result.language,
            "speaker_word_counts": {
                speaker: len([w for w in words_with_speakers if w.speaker == speaker])
                for speaker in speakers
            }
        }

        # Save results
        print("💾 Sonuçlar kaydediliyor...")

        # Timeline (ana çıktı)
        with open(f"{output_dir}/speaker_timeline.jsonl", "w", encoding="utf-8") as f:
            for segment in timeline:
                f.write(json.dumps(segment, ensure_ascii=False) + "\n")

        # Word-level details
        word_details = []
        for word in words_with_speakers:
            word_details.append({
                "word": word.word,
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "confidence": round(word.confidence, 3),
                "speaker": word.speaker,
                "speaker_confidence": round(word.speaker_confidence, 3),
                "is_overlap": word.is_overlap
            })

        with open(f"{output_dir}/word_details.jsonl", "w", encoding="utf-8") as f:
            for word in word_details:
                f.write(json.dumps(word, ensure_ascii=False) + "\n")

        # Overlaps
        with open(f"{output_dir}/overlaps.json", "w", encoding="utf-8") as f:
            json.dump(overlaps, f, indent=2, ensure_ascii=False)

        # Statistics
        with open(f"{output_dir}/statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\n✅ İşlem tamamlandı!")
        print(f"📊 İstatistikler:")
        print(f"  • Süre: {stats['total_duration']:.1f} saniye")
        print(f"  • Kelime sayısı: {stats['total_words']}")
        print(f"  • Konuşmacı sayısı: {stats['total_speakers']}")
        print(f"  • Segment sayısı: {stats['total_segments']}")
        print(f"  • Overlap bölgesi: {stats['overlap_segments']} ({stats['overlap_percentage']}%)")
        print(f"  • Dil: {stats['language']}")
        print(f"📁 Çıktılar: {output_dir}/")

        return {
            "timeline": timeline,
            "overlaps": overlaps,
            "statistics": stats,
            "output_dir": output_dir
        }

def main():
    """Ana fonksiyon - CLI kullanımı"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced ASR + Diarization")
    parser.add_argument("audio_file", help="Ses dosyası yolu")
    parser.add_argument("--output", "-o", default="../output", help="Çıktı dizini")
    parser.add_argument("--language", "-l", help="Dil kodu (tr, en, vs.)")
    parser.add_argument("--model-size", default="large-v2",
                       choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                       help="WhisperX model boyutu")
    parser.add_argument("--min-speakers", type=int, help="Minimum konuşmacı sayısı")
    parser.add_argument("--max-speakers", type=int, help="Maksimum konuşmacı sayısı")
    parser.add_argument("--device", default="auto", help="Cihaz (cuda/cpu/auto)")

    args = parser.parse_args()

    # Check file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Dosya bulunamadı: {args.audio_file}")
        return

    # Create processor
    processor = EnhancedASRDiarization(
        device=args.device,
        model_size=args.model_size
    )

    # Process
    result = processor.process(
        audio_path=args.audio_file,
        output_dir=args.output,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )

if __name__ == "__main__":
    main()
EOF

echo -e "${GREEN}✅ enhanced_asr_diarization.py oluşturuldu${NC}"

# Utility dosyaları
cat > src/utils.py << 'EOF'
"""
Utility functions for Enhanced ASR+Diarization
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, List, Optional

def convert_audio_to_wav(input_path: str, output_path: str = None,
                        sample_rate: int = 16000, channels: int = 1) -> str:
    """Audio dosyasını WAV formatına çevir"""

    if output_path is None:
        name, _ = os.path.splitext(input_path)
        output_path = f"{name}_converted.wav"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-f", "wav",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg hatası: {e}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg bulunamadı, lütfen kurun")

def load_jsonl(file_path: str) -> List[Dict]:
    """JSONL dosyasını yükle"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """JSONL dosyasına kaydet"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def analyze_speaker_distribution(timeline: List[Dict]) -> Dict:
    """Konuşmacı dağılımını analiz et"""
    speaker_stats = {}

    for segment in timeline:
        speaker = segment["speaker"]
        duration = segment["duration"]
        word_count = segment["word_count"]

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                "total_duration": 0,
                "total_words": 0,
                "segment_count": 0,
                "avg_segment_duration": 0
            }

        speaker_stats[speaker]["total_duration"] += duration
        speaker_stats[speaker]["total_words"] += word_count
        speaker_stats[speaker]["segment_count"] += 1

    # Calculate averages
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        stats["avg_segment_duration"] = stats["total_duration"] / stats["segment_count"]
        stats["words_per_minute"] = (stats["total_words"] / stats["total_duration"]) * 60 if stats["total_duration"] > 0 else 0

    return speaker_stats

def export_srt(timeline: List[Dict], output_path: str, speaker_labels: bool = True):
    """SRT altyazı dosyası oluştur"""

    def format_time(seconds: float) -> str:
        """SRT zaman formatına çevir"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(timeline, 1):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])

            text = segment["text"]
            if speaker_labels:
                text = f"[{segment['speaker']}] {text}"

            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
EOF

echo -e "${GREEN}✅ utils.py oluşturuldu${NC}"

# Test dosyası
cat > src/test_enhanced.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced ASR+Diarization Test Script
"""

import os
import sys
from enhanced_asr_diarization import EnhancedASRDiarization
from utils import analyze_speaker_distribution, export_srt

def test_with_sample_audio():
    """Test fonksiyonu"""

    print("🧪 Enhanced ASR+Diarization Test")
    print("=" * 40)

    # Sample audio kontrolü
    sample_audio = "../data/sample_audio.wav"
    if not os.path.exists(sample_audio):
        print(f"⚠️ Sample audio bulunamadı: {sample_audio}")
        print("Kendi audio dosyanızla test edebilirsiniz:")
        print("python test_enhanced.py your_audio.wav")
        return

    # Processor oluştur
    processor = EnhancedASRDiarization(
        device="auto",
        model_size="medium"  # Test için daha hızlı
    )

    # İşle
    result = processor.process(
        audio_path=sample_audio,
        output_dir="../output/test_results",
        language="tr",
        min_speakers=1,
        max_speakers=4
    )

    # Analiz et
    speaker_stats = analyze_speaker_distribution(result["timeline"])

    print("\n📊 Konuşmacı Analizi:")
    for speaker, stats in speaker_stats.items():
        print(f"  {speaker}:")
        print(f"    • Toplam süre: {stats['total_duration']:.1f}s")
        print(f"    • Toplam kelime: {stats['total_words']}")
        print(f"    • Dakikada kelime: {stats['words_per_minute']:.1f}")

    # SRT export
    srt_path = "../output/test_results/subtitles.srt"
    export_srt(result["timeline"], srt_path)
    print(f"\n📝 SRT dosyası oluşturuldu: {srt_path}")

    print(f"\n✅ Test tamamlandı!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom audio file
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            processor = EnhancedASRDiarization()
            processor.process(audio_file, "../output/custom_test")
        else:
            print(f"❌ Dosya bulunamadı: {audio_file}")
    else:
        test_with_sample_audio()
EOF

echo -e "${GREEN}✅ test_enhanced.py oluşturuldu${NC}"

# Konfigürasyon dosyaları
cat > configs/config.json << 'EOF'
{
    "default_settings": {
        "model_size": "large-v2",
        "device": "auto",
        "language": null,
        "min_speakers": null,
        "max_speakers": null,
        "overlap_threshold": 0.25,
        "smoothing_window": 3
    },
    "model_sizes": {
        "tiny": {"memory_usage": "low", "speed": "fastest", "accuracy": "low"},
        "base": {"memory_usage": "low", "speed": "fast", "accuracy": "medium"},
        "small": {"memory_usage": "medium", "speed": "medium", "accuracy": "good"},
        "medium": {"memory_usage": "medium", "speed": "medium", "accuracy": "good"},
        "large-v2": {"memory_usage": "high", "speed": "slow", "accuracy": "excellent"},
        "large-v3": {"memory_usage": "high", "speed": "slow", "accuracy": "excellent"}
    },
    "supported_languages": [
        "tr", "en", "de", "fr", "es", "it", "pt", "ru", "ja", "ko", "zh"
    ]
}
EOF

# Environment örneği
cat > .env.example << 'EOF'
# HuggingFace Token (Pyannote modelleri için gerekli)
HF_TOKEN=your_huggingface_token_here

# Model ayarları
DEFAULT_MODEL_SIZE=large-v2
DEFAULT_DEVICE=auto
DEFAULT_LANGUAGE=tr

# Çıktı dizini
OUTPUT_DIR=output

# Log seviyesi
LOG_LEVEL=INFO
EOF

echo -e "${GREEN}✅ Konfigürasyon dosyaları oluşturuldu${NC}"

# Script dosyaları
cat > scripts/run_enhanced.sh << 'EOF'
#!/bin/bash

# Enhanced ASR+Diarization çalıştırma scripti

# Virtual environment'ı aktif et
source ../venv/bin/activate

# Default values
AUDIO_FILE=""
OUTPUT_DIR="../output"
MODEL_SIZE="large-v2"
LANGUAGE=""
MIN_SPEAKERS=""
MAX_SPEAKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            AUDIO_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -l|--language)
            LANGUAGE="$2"
            shift 2
            ;;
        --min-speakers)
            MIN_SPEAKERS="$2"
            shift 2
            ;;
        --max-speakers)
            MAX_SPEAKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Enhanced ASR+Diarization Script"
            echo "Usage: $0 -i audio_file [options]"
            echo ""
            echo "Options:"
            echo "  -i, --input       Input audio file (required)"
            echo "  -o, --output      Output directory (default: ../output)"
            echo "  -m, --model       Model size (default: large-v2)"
            echo "  -l, --language    Language code (default: auto-detect)"
            echo "      --min-speakers Minimum number of speakers"
            echo "      --max-speakers Maximum number of speakers"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$AUDIO_FILE" ]; then
    echo "Error: Input audio file is required"
    echo "Use -h for help"
    exit 1
fi

# Check file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Build command
CMD="python ../src/enhanced_asr_diarization.py \"$AUDIO_FILE\" --output \"$OUTPUT_DIR\" --model-size \"$MODEL_SIZE\""

if [ -n "$LANGUAGE" ]; then
    CMD="$CMD --language \"$LANGUAGE\""
fi

if [ -n "$MIN_SPEAKERS" ]; then
    CMD="$CMD --min-speakers $MIN_SPEAKERS"
fi

if [ -n "$MAX_SPEAKERS" ]; then
    CMD="$CMD --max-speakers $MAX_SPEAKERS"
fi

echo "Running: $CMD"
eval $CMD
EOF

chmod +x scripts/run_enhanced.sh

echo -e "${GREEN}✅ Script dosyaları oluşturuldu${NC}"

# README dosyası
cat > README.md << 'EOF'
# Enhanced ASR + Diarization System

WhisperX ve Pyannote kombinasyonu ile geliştirilmiş ASR+Diarization sistemi.

## 🚀 Özellikler

- **Word-level alignment** - Her kelimenin hassas zamanlaması
- **Gelişmiş speaker assignment** - IoU algoritması ile doğru konuşmacı atama
- **Overlap detection** - Aynı anda konuşan kişileri tespit
- **Speaker smoothing** - Gürültülü speaker değişimlerini düzelt
- **Detaylı analiz** - Kapsamlı istatistikler ve raporlar

## 📦 Kurulum

```bash
# Bu script ile otomatik kurulum
./setup_enhanced_asr.sh

# Manuel kurulum
cd try
source venv/bin/activate
pip install -r requirements.txt
```

## 🔑 Gereksinimler

- Python 3.8+
- FFmpeg
- HuggingFace Token (Pyannote için)

```bash
export HF_TOKEN=your_huggingface_token
```

## 🎯 Kullanım

### Basit kullanım
```bash
cd scripts
./run_enhanced.sh -i audio.wav
```

### Parametreli kullanım
```bash
./run_enhanced.sh -i audio.wav \
  -o results \
  -m large-v2 \
  -l tr \
  --min-speakers 2 \
  --max-speakers 4
```

### Python API
```python
from src.enhanced_asr_diarization import EnhancedASRDiarization

processor = EnhancedASRDiarization(model_size="large-v2")
result = processor.process("audio.wav", "output")
```

## 📊 Çıktılar

- `speaker_timeline.jsonl` - Konuşmacı bazlı timeline
- `word_details.jsonl` - Kelime düzeyinde detaylar
- `overlaps.json` - Overlap analizi
- `statistics.json` - Detaylı istatistikler
- `subtitles.srt` - SRT altyazı dosyası

## 🔧 Konfigürasyon

`configs/config.json` dosyasını düzenleyerek varsayılan ayarları değiştirebilirsiniz.

## 🧪 Test

```bash
cd src
python test_enhanced.py
```

## 📈 Performans

| Model Size | Hız | Doğruluk | Bellek |
|------------|-----|----------|--------|
| tiny       | En hızlı | Düşük | Az |
| medium     | Orta | İyi | Orta |
| large-v2   | Yavaş | Mükemmel | Yüksek |

## 🛠️ Sorun Giderme

1. **CUDA hatası**: `--device cpu` kullanın
2. **Memory hatası**: Daha küçük model kullanın
3. **HF_TOKEN**: Token'ınızı kontrol edin
4. **FFmpeg**: Kurulu olduğundan emin olun

## 📝 Lisans

MIT License
EOF

echo -e "${GREEN}✅ README.md oluşturuldu${NC}"

# .gitignore
cat > .gitignore << 'EOF'
# Virtual environment
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Models and cache
models/
.cache/
*.pt
*.pth
*.onnx

# Data files
data/*.wav
data/*.mp3
data/*.mp4
data/*.m4a

# Output files
output/
results/
logs/

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Temporary files
*.tmp
*.temp
temp/
tmp/
EOF

echo -e "${GREEN}✅ .gitignore oluşturuldu${NC}"

# Örnek ses dosyası bilgisi
echo -e "${BLUE}📁 Dizin yapısı oluşturuldu:${NC}"
echo "try/"
echo "├── src/                     # Kaynak kodlar"
echo "│   ├── enhanced_asr_diarization.py"
echo "│   ├── utils.py"
echo "│   └── test_enhanced.py"
echo "├── scripts/                 # Çalıştırma scriptleri"
echo "│   └── run_enhanced.sh"
echo "├── configs/                 # Konfigürasyon"
echo "│   └── config.json"
echo "├── data/                    # Ses dosyaları (buraya koyun)"
echo "├── output/                  # Çıktı dosyaları"
echo "├── models/                  # Model cache"
echo "├── logs/                    # Log dosyaları"
echo "├── venv/                    # Virtual environment"
echo "├── requirements.txt"
echo "├── README.md"
echo "├── .env.example"
echo "└── .gitignore"

# Virtual environment'dan çık
deactivate

echo ""
echo -e "${GREEN}✅ Kurulum tamamlandı!${NC}"
echo ""
echo -e "${BLUE}📋 Sonraki adımlar:${NC}"
echo "1. HuggingFace token alın: https://huggingface.co/settings/tokens"
echo "2. Token'ı ayarlayın: export HF_TOKEN=your_token"
echo "3. Ses dosyanızı data/ klasörüne koyun"
echo "4. Test edin:"
echo ""
echo -e "${YELLOW}cd ${INSTALL_DIR}${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo -e "${YELLOW}export HF_TOKEN=your_token${NC}"
echo -e "${YELLOW}cd scripts${NC}"
echo -e "${YELLOW}./run_enhanced.sh -i ../data/your_audio.wav${NC}"
echo ""
echo -e "${GREEN}🎉 Enhanced ASR+Diarization sistemi hazır!${NC}"