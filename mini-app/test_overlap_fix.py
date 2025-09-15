#!/usr/bin/env python3
"""Test overlap handling improvements in the pipeline"""

import json
from pathlib import Path
from speaker_segment_analyzer import SpeakerSegmentAnalyzer

def test_overlap_redistribution():
    """Test the improved overlap redistribution logic"""

    # Create test segments with various overlap scenarios
    test_segments = [
        # Normal conversation - slight overlap
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "text": "Hello there"},
        {"speaker": "SPEAKER_01", "start": 2.8, "end": 5.0, "text": "Hi, how are you"},  # 0.2s overlap

        # Interruption - medium overlap
        {"speaker": "SPEAKER_00", "start": 5.0, "end": 8.0, "text": "I wanted to say"},
        {"speaker": "SPEAKER_01", "start": 6.5, "end": 9.0, "text": "Sorry to interrupt"},  # 1.5s overlap

        # Same speaker segments (should not overlap)
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 12.0, "text": "First part"},
        {"speaker": "SPEAKER_00", "start": 11.5, "end": 13.0, "text": "Second part"},  # Same speaker overlap

        # Multiple speaker rapid exchange
        {"speaker": "SPEAKER_00", "start": 15.0, "end": 16.0, "text": "Quick"},
        {"speaker": "SPEAKER_01", "start": 15.8, "end": 16.8, "text": "Response"},
        {"speaker": "SPEAKER_02", "start": 16.5, "end": 17.5, "text": "Another"},

        # Large problematic overlap
        {"speaker": "SPEAKER_00", "start": 20.0, "end": 25.0, "text": "Long statement"},
        {"speaker": "SPEAKER_01", "start": 21.0, "end": 26.0, "text": "Overlapping long"},  # 4s overlap
    ]

    print("=" * 60)
    print("TESTING OVERLAP REDISTRIBUTION")
    print("=" * 60)

    # Initialize analyzer
    analyzer = SpeakerSegmentAnalyzer(
        min_segment_duration=0.2,
        merge_gap_threshold=0.5,
        overlap_tolerance=0.1
    )

    print(f"\nOriginal segments: {len(test_segments)}")
    print("\nOriginal timeline:")
    for seg in test_segments:
        print(f"  {seg['speaker']:12} [{seg['start']:6.2f} - {seg['end']:6.2f}] : {seg['text']}")

    # Apply overlap redistribution
    redistributed = analyzer.remove_cross_speaker_overlaps(test_segments)

    print(f"\nRedistributed segments: {len(redistributed)}")
    print("\nRedistributed timeline:")
    for seg in sorted(redistributed, key=lambda x: x['start']):
        duration = seg['end'] - seg['start']
        print(f"  {seg['speaker']:12} [{seg['start']:6.2f} - {seg['end']:6.2f}] (dur: {duration:.2f}s) : {seg['text']}")

    # Check for remaining overlaps
    print("\nOverlap check:")
    overlaps_found = []
    for i, seg1 in enumerate(redistributed):
        for j, seg2 in enumerate(redistributed):
            if i >= j:
                continue
            if seg1['speaker'] == seg2['speaker']:
                continue

            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])

            if overlap_start < overlap_end - 0.01:  # Allow 10ms tolerance
                overlap_duration = overlap_end - overlap_start
                overlaps_found.append({
                    'speakers': f"{seg1['speaker']} <-> {seg2['speaker']}",
                    'duration': overlap_duration,
                    'range': f"[{overlap_start:.2f} - {overlap_end:.2f}]"
                })

    if overlaps_found:
        print(f"  Found {len(overlaps_found)} remaining overlaps:")
        for ovl in overlaps_found:
            if ovl['duration'] <= 0.1:
                print(f"    ✓ Natural overlap: {ovl['speakers']} for {ovl['duration']:.3f}s {ovl['range']}")
            else:
                print(f"    ⚠ Large overlap: {ovl['speakers']} for {ovl['duration']:.3f}s {ovl['range']}")
    else:
        print("  ✓ No significant overlaps detected")

    # Test adaptive gaps in _concat_timeline_audio
    print("\n" + "=" * 60)
    print("TESTING ADAPTIVE BREATH GAPS")
    print("=" * 60)

    gap_scenarios = [
        {"prev_end": 1.0, "curr_start": 1.2, "expected": "Short gap (50ms)"},  # 0.2s gap -> rapid
        {"prev_end": 1.0, "curr_start": 1.8, "expected": "Normal gap (100ms)"},  # 0.8s gap -> normal
        {"prev_end": 1.0, "curr_start": 3.0, "expected": "Full gap (200ms)"},  # 2.0s gap -> topic change
    ]

    for scenario in gap_scenarios:
        time_gap = scenario["curr_start"] - scenario["prev_end"]

        # Simulate adaptive gap calculation
        if time_gap < 0.5:
            adaptive_gap = 50
        elif time_gap < 1.5:
            adaptive_gap = 100
        else:
            adaptive_gap = 200

        print(f"  Gap: {time_gap:.1f}s -> {adaptive_gap}ms ({scenario['expected']})")

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

    return redistributed

if __name__ == "__main__":
    result = test_overlap_redistribution()

    # Save results
    output_file = Path("test_overlap_results.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")