#!/usr/bin/env python3
"""
Simple Working Pipeline - Guaranteed to work with WhisperX
No complex overlap detection, just clean diarization + ASR
"""

import json
import sys
import os
from pathlib import Path

def process_audio_simple(audio_file, output_dir="./output", min_speakers=None, max_speakers=None):
    """
    Simple pipeline that just works
    """
    import whisperx
    import torch
    import gc
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 16
    
    print(f"Processing: {audio_file}")
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading Whisper model...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    
    # Load audio
    print("Loading audio...")
    audio = whisperx.load_audio(audio_file)
    
    # Transcribe
    print("Transcribing...")
    result = model.transcribe(audio, batch_size=batch_size)
    print(f"Language detected: {result['language']}")
    
    # Align
    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    
    # Clean up alignment model
    del model_a
    gc.collect()
    
    # Diarization (if HF token available)
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token:
        print("Running speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        
        # Set speaker parameters
        diarize_params = {}
        if min_speakers:
            diarize_params["min_speakers"] = min_speakers
        if max_speakers:
            diarize_params["max_speakers"] = max_speakers
        
        # Run diarization
        diarize_segments = diarize_model(audio, **diarize_params)
        
        # Assign speakers
        print("Assigning speakers to segments...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        print("Warning: No HF_TOKEN found, skipping diarization")
    
    # Process segments
    segments = []
    for seg in result["segments"]:
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        segment = {
            "text": text,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "speaker": seg.get("speaker", "SPEAKER_00")
        }
        segments.append(segment)
    
    # Sort by start time
    segments.sort(key=lambda x: x["start"])
    
    # Fix overlaps simply
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i-1]["end"]:
            # Simple fix: adjust start time
            segments[i]["start"] = segments[i-1]["end"]
    
    # Save outputs
    base_name = Path(audio_file).stem
    output_file = output_dir / f"{base_name}_output.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(json.dumps(segment, ensure_ascii=False) + '\n')
    
    print(f"\nOutput saved to: {output_file}")
    print(f"Total segments: {len(segments)}")
    
    # Print sample
    print("\nSample output (first 3 segments):")
    for seg in segments[:3]:
        print(json.dumps(seg, ensure_ascii=False))
    
    return str(output_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Working Pipeline")
    parser.add_argument("audio_file", help="Audio file to process")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("--min-speakers", type=int, help="Minimum speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum speakers")
    
    args = parser.parse_args()
    
    # Check HF token
    if not os.getenv("HF_TOKEN"):
        print("\nWarning: HF_TOKEN not set!")
        print("Speaker diarization will not work properly.")
        print("Set it with: export HF_TOKEN=your_token\n")
    
    try:
        process_audio_simple(
            args.audio_file,
            args.output,
            args.min_speakers,
            args.max_speakers
        )
        print("\n✓ Processing complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)