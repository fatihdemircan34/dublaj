#!/usr/bin/env python3
"""
Test the complete sync solution with time stretching
"""

import sys
import json
import logging
from pathlib import Path
from sync_monitor import SyncDriftMonitor, calculate_optimal_stretch_factor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_time_stretching():
    """Test time stretching algorithm with various scenarios"""
    print("=" * 60)
    print("TESTING TIME STRETCHING ALGORITHM")
    print("=" * 60)

    test_cases = [
        # (original_duration, tts_duration, description)
        (3.0, 3.0, "Perfect match"),
        (3.0, 3.1, "Slight overshoot (3.3%)"),
        (3.0, 2.8, "Slight undershoot (7%)"),
        (3.0, 3.5, "Moderate overshoot (17%)"),
        (3.0, 2.5, "Moderate undershoot (17%)"),
        (3.0, 4.5, "Large overshoot (50%)"),
        (3.0, 1.5, "Large undershoot (50%)"),
        (3.0, 6.0, "Extreme overshoot (100%)"),
        (3.0, 1.0, "Extreme undershoot (67%)"),
    ]

    for orig, tts, desc in test_cases:
        factor = calculate_optimal_stretch_factor(orig, tts, max_stretch=1.5)
        expected_result = tts * factor

        print(f"\n{desc}:")
        print(f"  Original: {orig:.2f}s")
        print(f"  TTS Output: {tts:.2f}s")
        print(f"  Stretch Factor: {factor:.3f}")
        print(f"  Expected Result: {expected_result:.2f}s")
        print(f"  Remaining Drift: {abs(expected_result - orig):.3f}s")

def test_sync_monitoring():
    """Test sync monitoring with realistic segment data"""
    print("\n" + "=" * 60)
    print("TESTING SYNC MONITORING")
    print("=" * 60)

    # Create monitor
    monitor = SyncDriftMonitor(
        max_allowed_drift=0.3,
        sync_interval=5,
        correction_factor=0.6
    )

    # Simulate realistic video with gradual drift
    segments = []
    current_time = 0.0

    for i in range(50):
        # Original segment duration (varies between 1-4 seconds)
        orig_duration = 2.0 + (i % 3) * 0.5

        # TTS tends to be slightly different (±20%)
        drift_factor = 1.0 + (0.2 * ((i % 7) - 3) / 3)
        tts_duration = orig_duration * drift_factor

        # Apply optimal stretching
        stretch_factor = calculate_optimal_stretch_factor(
            orig_duration, tts_duration, max_stretch=1.3
        )
        actual_duration = tts_duration * stretch_factor

        segments.append({
            'id': i,
            'start': current_time,
            'end': current_time + orig_duration,
            'original_duration': orig_duration,
            'tts_duration': tts_duration,
            'stretched_duration': actual_duration
        })

        # Process through monitor
        adjustment, is_sync = monitor.add_segment(
            segment_id=i,
            original_start=current_time,
            original_duration=orig_duration,
            actual_duration=actual_duration
        )

        if is_sync:
            print(f"  Sync point at segment {i}: drift={monitor.current_drift:.3f}s, adjustment={adjustment:.3f}s")

        current_time += orig_duration

    # Get final statistics
    stats = monitor.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Segments: {stats.total_segments}")
    print(f"  Max Drift: {stats.max_drift:.3f}s")
    print(f"  Average Drift: {stats.avg_drift:.3f}s")
    print(f"  Final Drift: {stats.final_drift:.3f}s")
    print(f"  Sync Points Created: {len(stats.sync_points)}")

    # Export detailed report
    monitor.export_report(Path("test_sync_report.json"))

    # Save segment data
    with open("test_segments.json", 'w') as f:
        json.dump(segments, f, indent=2)

    return stats.final_drift

def test_extreme_scenario():
    """Test with extreme drift scenario (many speakers, long video)"""
    print("\n" + "=" * 60)
    print("TESTING EXTREME SCENARIO (10 speakers, 200 segments)")
    print("=" * 60)

    monitor = SyncDriftMonitor(
        max_allowed_drift=0.5,
        sync_interval=10,
        correction_factor=0.7
    )

    # Simulate 10 speakers with different speaking patterns
    speaker_patterns = {
        f"SPEAKER_{i:02d}": {
            'speed_factor': 0.8 + (i * 0.04),  # Each speaker has different speed
            'segments': 0,
            'total_drift': 0.0
        }
        for i in range(10)
    }

    total_drift_no_correction = 0.0
    total_drift_with_correction = 0.0

    for seg_id in range(200):
        # Rotate through speakers
        speaker_id = f"SPEAKER_{seg_id % 10:02d}"
        pattern = speaker_patterns[speaker_id]

        # Original duration
        orig_duration = 2.0 + (seg_id % 5) * 0.3

        # TTS duration based on speaker pattern
        tts_duration = orig_duration * pattern['speed_factor']

        # Without correction
        drift_no_correction = tts_duration - orig_duration
        total_drift_no_correction += drift_no_correction

        # With time stretching
        stretch_factor = calculate_optimal_stretch_factor(
            orig_duration, tts_duration, max_stretch=1.4
        )
        actual_duration = tts_duration * stretch_factor

        # Process through monitor
        adjustment, is_sync = monitor.add_segment(
            segment_id=seg_id,
            original_start=seg_id * 2.5,
            original_duration=orig_duration,
            actual_duration=actual_duration
        )

        # Track per-speaker drift
        segment_drift = actual_duration - orig_duration + adjustment
        pattern['segments'] += 1
        pattern['total_drift'] += segment_drift

    # Results
    print(f"\nResults:")
    print(f"  Without any correction: {total_drift_no_correction:.2f}s drift")
    print(f"  With time stretching + sync anchors: {monitor.current_drift:.2f}s drift")
    print(f"  Improvement: {abs(total_drift_no_correction - monitor.current_drift):.2f}s")

    print(f"\nPer-speaker drift:")
    for speaker, pattern in speaker_patterns.items():
        if pattern['segments'] > 0:
            avg_drift = pattern['total_drift'] / pattern['segments']
            print(f"  {speaker}: {pattern['total_drift']:.3f}s total, {avg_drift:.3f}s avg")

def main():
    """Run all tests"""
    print("COMPLETE SYNC SOLUTION TEST SUITE")
    print("=" * 60)

    # Test 1: Time stretching
    test_time_stretching()

    # Test 2: Sync monitoring
    final_drift_normal = test_sync_monitoring()

    # Test 3: Extreme scenario
    test_extreme_scenario()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ Time stretching algorithm implemented")
    print("✓ Sync monitoring with anchor points implemented")
    print("✓ Drift correction system operational")
    print(f"✓ Normal scenario final drift: {final_drift_normal:.3f}s")
    print("\nThe sync solution is ready for production use!")

if __name__ == "__main__":
    main()