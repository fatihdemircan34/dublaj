#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the complete pipeline with all features
"""

import json
from mp4_diarize_pipeline import process_video_wordwise

# Run with sample2.mp4
result = process_video_wordwise(
    video_path="sample2.mp4",
    output_dir="output_final_test",

    # Core features
    stt_model="whisper-1",
    diarize=True,
    use_vad=True,

    # Translation & Dubbing
    do_translate=True,
    translator_model="gpt-4o-mini",
    do_dub=True,
    target_lang="tr",

    # Speaker Analysis & Demucs
    analyze_speakers=True,
    remove_overlaps=True,
    use_demucs=True,
    demucs_model="htdemucs",
    instrumental_volume=0.8,
    dubbing_volume=1.0,

    # Disable lipsync for now
    do_lipsync=False,

    debug=True
)

print(json.dumps(result, ensure_ascii=False, indent=2))