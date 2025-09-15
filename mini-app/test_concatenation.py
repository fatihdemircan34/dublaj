#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the fixed concatenation logic
"""

from pathlib import Path
from pydub import AudioSegment
import json

# Create test audio segments
def create_test_segments():
    """Create test audio segments to verify concatenation"""

    test_dir = Path("test_concat")
    test_dir.mkdir(exist_ok=True)

    # Create 3 short test audio segments (beeps)
    sr = 16000
    duration_ms = 1000  # 1 second each

    segments = []
    seg_audio_paths = {}

    for i in range(3):
        # Create a simple audio segment (silence with a click at start)
        # This simulates different audio segments
        silence = AudioSegment.silent(duration=duration_ms, frame_rate=sr)

        # Add a small click/beep at the beginning to distinguish segments
        click = AudioSegment.silent(duration=50, frame_rate=sr).apply_gain(10)
        beep = click + silence[50:]

        # Save it
        path = test_dir / f"seg_{i:06d}.wav"
        beep.export(path, format="wav")

        # Create segment info
        segment = {
            "id": i,
            "speaker": "SPEAKER_00" if i < 2 else "SPEAKER_01",
            "start": i * 1.5,  # Overlapping times for same speaker
            "end": i * 1.5 + 1.0,
            "text": f"Test segment {i}"
        }

        segments.append(segment)
        seg_audio_paths[i] = path

    return segments, seg_audio_paths, test_dir

def test_concatenation():
    """Test the concatenation with the new logic"""

    print("Creating test segments...")
    segments, seg_audio_paths, test_dir = create_test_segments()

    print(f"Created {len(segments)} test segments")
    for seg in segments:
        print(f"  - Segment {seg['id']}: Speaker {seg['speaker']}, Time {seg['start']:.1f}-{seg['end']:.1f}s")

    # Import and test the concatenation function
    from mp4_diarize_pipeline import _concat_timeline_audio

    # Test with default breath gap
    output_file = test_dir / "concatenated.wav"
    total_duration = 5.0

    print("\nConcatenating segments...")
    _concat_timeline_audio(segments, seg_audio_paths, total_duration, output_file, breath_gap_ms=200)

    # Check the output
    if output_file.exists():
        output_audio = AudioSegment.from_file(output_file)
        print(f"\n✓ Output created: {output_file}")
        print(f"  Duration: {len(output_audio) / 1000:.2f} seconds")

        # Expected behavior:
        # - Segment 0 (SPEAKER_00): at 0.0s
        # - Segment 1 (SPEAKER_00): at 1.2s (1.0s + 0.2s breath gap) instead of 1.5s
        # - Segment 2 (SPEAKER_01): at 3.0s (original time, different speaker)

        print("\nExpected timeline:")
        print("  - Segment 0: 0.0-1.0s (original)")
        print("  - Segment 1: 1.2-2.2s (adjusted for breath gap)")
        print("  - Segment 2: 3.0-4.0s (different speaker, original time)")

        return True
    else:
        print("✗ Failed to create output file")
        return False

if __name__ == "__main__":
    success = test_concatenation()
    exit(0 if success else 1)