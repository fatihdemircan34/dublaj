#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal test to check the pipeline fix
"""

import json
from mp4_diarize_pipeline import _write_outputs
from pathlib import Path

# Test the fixed _write_outputs function
test_dir = Path("test_output")
test_dir.mkdir(exist_ok=True)

# Test with different timeline formats
segments = [{"start": 0, "end": 5, "text": "Test", "speaker": "A"}]
words = []

# Test 1: Timeline as list of dicts
timeline_dict = [{"start": 0, "end": 5, "mode": "speech", "speakers": ["A"], "channels": None}]
result = _write_outputs(test_dir, "test1", segments, words, "Test", "en", 5.0, timeline=timeline_dict)
print("✓ Test 1 passed: Timeline as dict list")

# Test 2: Timeline as list of tuples
timeline_tuple = [(0, 5), (6, 10)]
result = _write_outputs(test_dir, "test2", segments, words, "Test", "en", 10.0, timeline=timeline_tuple)
print("✓ Test 2 passed: Timeline as tuple list")

# Test 3: Timeline as None
result = _write_outputs(test_dir, "test3", segments, words, "Test", "en", 5.0, timeline=None)
print("✓ Test 3 passed: Timeline as None")

print("\nAll tests passed! The fix is working correctly.")