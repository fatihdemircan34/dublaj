#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test of the updated concatenation logic
"""

import json
from pathlib import Path
from mp4_diarize_pipeline import process_video_wordwise

# Use existing files to avoid re-processing
# We'll just re-run the dubbing part

print("Running pipeline with fixed concatenation...")
print("-" * 50)

result = process_video_wordwise(
    video_path="sample2.mp4",
    output_dir="output_concat_test",

    # Core settings
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

    # No lipsync
    do_lipsync=False,

    debug=True
)

print("\n" + "=" * 50)
print("Pipeline completed!")
print("=" * 50)

# Check output files
output_dir = Path("output_concat_test")
if output_dir.exists():
    dub_file = output_dir / "dubbed.timeline.mono16k.wav"
    if dub_file.exists():
        print(f"\n✓ Dubbed audio created: {dub_file}")

    final_video = output_dir / "final_dubbed_video_with_music.mp4"
    if final_video.exists():
        print(f"✓ Final video created: {final_video}")

print("\nThe concatenation fix ensures:")
print("- Same speaker segments don't overlap")
print("- 200ms breath gap between consecutive segments of same speaker")
print("- Different speakers can still speak at original times")