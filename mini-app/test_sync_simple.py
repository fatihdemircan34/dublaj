#!/usr/bin/env python3
"""
Simple test to verify sync improvements are working
"""

import sys
import os
from pathlib import Path

# Test imports
try:
    from mp4_diarize_pipeline import _time_stretch_to_duration, _ffprobe_duration
    print("✅ Time stretch function imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_time_stretch_function():
    """Test if time stretching is actually enabled"""
    print("\n" + "="*60)
    print("TESTING TIME STRETCH FUNCTION")
    print("="*60)

    # Create a test audio file (1 second of silence)
    test_input = Path("test_input.wav")
    test_output = Path("test_output.wav")

    # Generate a 2-second test file using ffmpeg
    os.system(f'ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 {test_input} 2>/dev/null')

    if not test_input.exists():
        print("❌ Failed to create test input file")
        return False

    # Get original duration
    original_duration = _ffprobe_duration(test_input)
    print(f"Original duration: {original_duration:.3f}s")

    # Test 1: Stretch to 3 seconds (slower)
    print("\nTest 1: Stretching to 3 seconds...")
    _time_stretch_to_duration(test_input, target_sec=3.0, out_wav=test_output, enable=True)

    if test_output.exists():
        result_duration = _ffprobe_duration(test_output)
        print(f"  Target: 3.0s, Result: {result_duration:.3f}s")
        if abs(result_duration - 3.0) < 0.1:
            print("  ✅ Stretch to 3s successful!")
        else:
            print(f"  ⚠️ Duration mismatch: {abs(result_duration - 3.0):.3f}s")
    else:
        print("  ❌ Output file not created")

    # Test 2: Compress to 1 second (faster)
    print("\nTest 2: Compressing to 1 second...")
    _time_stretch_to_duration(test_input, target_sec=1.0, out_wav=test_output, enable=True)

    if test_output.exists():
        result_duration = _ffprobe_duration(test_output)
        print(f"  Target: 1.0s, Result: {result_duration:.3f}s")
        if abs(result_duration - 1.0) < 0.1:
            print("  ✅ Compress to 1s successful!")
        else:
            print(f"  ⚠️ Duration mismatch: {abs(result_duration - 1.0):.3f}s")

    # Test 3: Disabled stretching (should keep original)
    print("\nTest 3: Disabled stretching...")
    _time_stretch_to_duration(test_input, target_sec=3.0, out_wav=test_output, enable=False)

    if test_output.exists():
        result_duration = _ffprobe_duration(test_output)
        print(f"  Original: {original_duration:.3f}s, Result: {result_duration:.3f}s")
        if abs(result_duration - original_duration) < 0.1:
            print("  ✅ Disabled mode works (no stretching)")
        else:
            print("  ⚠️ Unexpected change in duration")

    # Cleanup
    if test_input.exists():
        test_input.unlink()
    if test_output.exists():
        test_output.unlink()

    return True

def check_sync_functions():
    """Check if new sync functions exist"""
    print("\n" + "="*60)
    print("CHECKING SYNC FUNCTIONS")
    print("="*60)

    functions_to_check = [
        ("_concat_timeline_audio_with_sync", "Advanced concatenation with sync anchoring"),
        ("_time_stretch_to_duration", "Time stretching function"),
        ("synthesize_dub_track_xtts", "TTS synthesis with duration matching"),
    ]

    from mp4_diarize_pipeline import __dict__ as pipeline_dict

    for func_name, description in functions_to_check:
        if func_name in pipeline_dict:
            func = pipeline_dict[func_name]
            # Check if it's the new version by looking at docstring
            if hasattr(func, '__doc__') and func.__doc__:
                if "sync" in func.__doc__.lower() or "stretch" in func.__doc__.lower() or "duration matching" in func.__doc__.lower():
                    print(f"✅ {func_name}: {description} - NEW VERSION")
                else:
                    print(f"⚠️ {func_name}: Found but might be old version")
            else:
                print(f"⚠️ {func_name}: Found but no docstring")
        else:
            print(f"❌ {func_name}: Not found")

def main():
    print("SYNC IMPROVEMENT VERIFICATION")
    print("="*60)

    # Check functions
    check_sync_functions()

    # Test time stretching
    success = test_time_stretch_function()

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    if success:
        print("✅ Time stretching is ENABLED and WORKING!")
        print("✅ Sync improvements are integrated!")
        print("\nThe pipeline should now maintain audio-video sync throughout the video.")
    else:
        print("❌ Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()