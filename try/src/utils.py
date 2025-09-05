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
