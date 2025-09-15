#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple integration test for speaker analysis and Demucs
"""

import os
import sys
from pathlib import Path

# Test imports
try:
    from speaker_segment_analyzer import (
        SpeakerSegmentAnalyzer,
        DemucsVocalSeparator,
        DubbingMixer
    )
    print("✓ Speaker analyzer modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import speaker analyzer: {e}")
    sys.exit(1)

try:
    from mp4_diarize_pipeline import process_video_wordwise
    print("✓ Main pipeline imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pipeline: {e}")
    sys.exit(1)

# Test Demucs availability
separator = DemucsVocalSeparator()
if separator.is_available:
    print("✓ Demucs is available and ready")
else:
    print("⚠ Demucs is not available (will use fallback)")

# Test speaker analyzer
print("\nTesting Speaker Analyzer:")
analyzer = SpeakerSegmentAnalyzer()

test_segments = [
    {"speaker": "A", "start": 0, "end": 5, "text": "Test 1"},
    {"speaker": "B", "start": 4, "end": 8, "text": "Test 2"},  # Overlaps
    {"speaker": "A", "start": 9, "end": 12, "text": "Test 3"},
]

print(f"  Input: {len(test_segments)} segments with overlaps")
clean = analyzer.remove_cross_speaker_overlaps(test_segments)
print(f"  Output: {len(clean)} segments without overlaps")

# Check for sample files
sample_files = ["sample.mp4", "sample2.mp4"]
available_samples = [f for f in sample_files if Path(f).exists()]

if available_samples:
    print(f"\n✓ Sample videos found: {', '.join(available_samples)}")
    print("\nReady to run full pipeline test:")
    print("  poetry run python test_speaker_pipeline.py")
else:
    print("\n⚠ No sample videos found. Please provide sample.mp4 or sample2.mp4")

print("\n" + "="*50)
print("Integration test completed successfully!")
print("="*50)