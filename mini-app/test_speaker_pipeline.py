#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the enhanced pipeline with speaker analysis and Demucs integration
"""

import json
import sys
from pathlib import Path

from mp4_diarize_pipeline import process_video_wordwise

def test_pipeline():
    """Test the complete pipeline with speaker analysis and Demucs"""

    # Check if sample video exists
    sample_files = ["sample.mp4", "sample2.mp4"]
    video_file = None

    for f in sample_files:
        if Path(f).exists():
            video_file = f
            break

    if not video_file:
        print("Error: No sample video found. Please provide sample.mp4 or sample2.mp4")
        return 1

    print(f"Processing video: {video_file}")
    print("-" * 50)

    try:
        result = process_video_wordwise(
            video_path=video_file,
            output_dir="output_test_speaker",

            # STT & Diarization
            stt_model="whisper-1",
            language=None,  # Auto-detect
            diarize=True,
            use_vad=True,

            # Speaker Mapping
            use_optimized_mapping=True,
            min_overlap_ratio=0.2,
            boundary_tolerance=0.1,
            confidence_threshold=0.6,

            # Translation
            do_translate=True,
            translator_model="gpt-4o-mini",

            # Dubbing
            do_dub=True,
            target_lang="tr",
            xtts_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            xtts_speed=None,

            # Lipsync (disable if not available)
            do_lipsync=False,

            # NEW: Speaker Analysis & Demucs
            analyze_speakers=True,
            remove_overlaps=True,
            use_demucs=True,
            demucs_model="htdemucs",
            instrumental_volume=0.8,
            dubbing_volume=1.0,

            # Debug
            debug=True
        )

        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print("=" * 50)

        # Print summary
        print("\nResults Summary:")
        print("-" * 30)

        if "files" in result:
            print("\nGenerated Files:")
            for key, path in result["files"].items():
                print(f"  - {key}: {path}")

        if "speaker_analysis" in result:
            print("\nSpeaker Analysis:")
            print(f"  - Analysis file: {result['speaker_analysis']}")

        if "dub" in result:
            print("\nDubbing Info:")
            print(f"  - Target language: {result['dub']['target_lang']}")
            print(f"  - Audio file: {result['dub']['audio_wav']}")
            if result['dub'].get('video'):
                print(f"  - Video file: {result['dub']['video']}")

        # Save full result
        output_json = Path("output_test_speaker") / "pipeline_result.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nFull result saved to: {output_json}")

        return 0

    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

def test_speaker_analyzer_only():
    """Test just the speaker analyzer component"""
    from speaker_segment_analyzer import SpeakerSegmentAnalyzer

    # Create sample segments
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello world"},
        {"speaker": "SPEAKER_01", "start": 4.5, "end": 8.0, "text": "How are you"},  # Overlaps with SPEAKER_00
        {"speaker": "SPEAKER_00", "start": 8.5, "end": 12.0, "text": "I'm fine"},
        {"speaker": "SPEAKER_01", "start": 11.5, "end": 15.0, "text": "That's good"},  # Overlaps with SPEAKER_00
        {"speaker": "SPEAKER_00", "start": 15.5, "end": 18.0, "text": "Thank you"},
    ]

    print("Testing Speaker Segment Analyzer")
    print("-" * 50)
    print(f"Original segments: {len(segments)}")

    analyzer = SpeakerSegmentAnalyzer()

    # Analyze segments
    speaker_analyses = analyzer.analyze_segments(segments)

    print(f"\nSpeakers found: {list(speaker_analyses.keys())}")

    for speaker_id, analysis in speaker_analyses.items():
        print(f"\n{speaker_id}:")
        print(f"  - Total segments: {len(analysis.segments)}")
        print(f"  - Merged segments: {len(analysis.merged_segments)}")
        print(f"  - Total duration: {analysis.total_duration:.2f}s")

    # Remove overlaps
    clean_segments = analyzer.remove_cross_speaker_overlaps(segments)
    print(f"\nAfter overlap removal: {len(clean_segments)} segments")

    # Check for overlaps
    overlaps_found = False
    for i, seg1 in enumerate(clean_segments):
        for j, seg2 in enumerate(clean_segments[i+1:], i+1):
            if seg1["speaker"] != seg2["speaker"]:
                overlap_start = max(seg1["start"], seg2["start"])
                overlap_end = min(seg1["end"], seg2["end"])
                if overlap_start < overlap_end:
                    print(f"  WARNING: Overlap found between {seg1['speaker']} and {seg2['speaker']}")
                    overlaps_found = True

    if not overlaps_found:
        print("  ✓ No cross-speaker overlaps detected")

    # Export analysis
    output_path = "output_test_speaker/test_speaker_analysis.json"
    Path("output_test_speaker").mkdir(exist_ok=True)
    result = analyzer.export_analysis(output_path)
    print(f"\nAnalysis exported to: {output_path}")

    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--analyzer-only":
        # Test just the analyzer
        exit_code = test_speaker_analyzer_only()
    else:
        # Test full pipeline
        exit_code = test_pipeline()

    sys.exit(exit_code)