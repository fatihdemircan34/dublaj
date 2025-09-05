#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced ASR + Diarization Pipeline with Pyannote-audio v3.1
Production-ready speaker diarization with word-level assignment
Optimized for highest accuracy on GPU platforms

- Tüm metodlar sınıf seviyesinde (iç içe fonksiyon yok)
- Strict diarization ataması (kelime merkezine göre hard-assign)
- Timeline diarization sınırlarına göre kesiliyor
- Full RTTM writer: SPEAKER satırında tüm alanlar dolu (ORT = turn text)
"""
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

# Core imports
import whisperx
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment
import torchaudio  # opsiyonel

# -------------------------
# Configuration
# -------------------------
@dataclass
class PipelineConfig:
    """Pipeline configuration parameters"""
    # Models
    whisper_model: str = "large-v3"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/embedding"

    # Processing
    device: str = "auto"
    batch_size: int = 32
    num_threads: int = 4

    # Diarization parameters
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    min_segment_duration: float = 0.5

    # Speaker assignment
    overlap_threshold: float = 0.5
    confidence_threshold: float = 0.7
    context_window: int = 5

    # Output
    save_timeline: bool = True
    save_rttm: bool = True
    save_segments: bool = True
    merge_gap: float = 0.3


# -------------------------
# Data structures
# -------------------------
@dataclass
class Word:
    """Word-level information"""
    text: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None
    speaker_confidence: float = 0.0
    is_overlap: bool = False
    assignment_method: str = "unknown"
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2


@dataclass
class SpeakerTurn:
    """Speaker turn information"""
    speaker: str
    start: float
    end: float
    confidence: float = 1.0
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class DiarizationResult:
    """Complete diarization result"""
    turns: List[SpeakerTurn]
    words: List[Word]
    overlaps: List[Dict]
    statistics: Dict
    raw_diarization: Optional[Annotation] = None
    raw_transcription: Optional[Dict] = None




# -------------------------
# CLI Interface
# -------------------------
def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced ASR + Diarization with Pyannote-audio v3.1 (strict diarization + full RTTM)"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-l", "--language", help="Language code (e.g., en, tr)")
    parser.add_argument("-n", "--num-speakers", type=int, help="Number of speakers")
    parser.add_argument("--min-speakers", type=int, help="Minimum speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum speakers")
    parser.add_argument("--model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Computation device")
    parser.add_argument("--no-rttm", action="store_true", help="Skip RTTM output")
    parser.add_argument("--no-timeline", action="store_true", help="Skip timeline output")

    args = parser.parse_args()

    # Check audio file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    # Configure pipeline
    config = PipelineConfig(
        whisper_model=args.model,
        device=args.device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        save_rttm=not args.no_rttm,
        save_timeline=not args.no_timeline
    )

    # Process
    # pipeline = PyannoteASRPipeline(config)
    # _ = pipeline.process(
    #     str(audio_path),
    #     args.output,
    #     language=args.language,
    #     num_speakers=args.num_speakers
    # )
    model_dir = "iic/SenseVoiceSmall"


    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
    )

    # en
    res = model.generate(
        input=f"/home/videodubb_voiceprocess_io/PycharmProjects/dublaj/input_6.wav",
        cache={},
        language="en",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )

    text = rich_transcription_postprocess(res[0]["text"])
    print(text)
    print("✅ Processing complete!")


if __name__ == "__main__":
    main()
