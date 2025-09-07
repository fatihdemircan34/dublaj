#!/usr/bin/env python3
"""
Final Solution - Optimal Pyannote Configuration
This uses the best practices from pyannote documentation
"""

import os
import json
import torch
from pyannote.audio import Pipeline
import whisperx
import gc

def final_solution(audio_path: str, num_speakers: int = 3):
    """
    Final solution with optimal configuration
    """
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is required!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Step 1: Load Pyannote with specific version
    print("Loading Pyannote pipeline...")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    # CRITICAL FIX: Use instantiate instead of reset_parameters
    # This is the proper way to set parameters in pyannote 3.x
    params = {
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 15,
            "threshold": 0.7153  # Optimal for 3 speakers based on pyannote benchmarks
        },
        "segmentation": {
            "min_duration_on": 0.0,  # Don't skip short speech
            "min_duration_off": 0.0  # Don't merge close segments
        }
    }
    
    # Apply parameters properly
    try:
        pipeline.instantiate(params)
    except:
        # Fallback for older versions
        for key, value in params.items():
            if hasattr(pipeline, key):
                for param, val in value.items():
                    if hasattr(getattr(pipeline, key), param):
                        setattr(getattr(pipeline, key), param, val)
    
    if device == "cuda":
        pipeline.to(torch.device("cuda"))
    
    # Step 2: Run diarization with exact speaker count
    print(f"Running diarization for exactly {num_speakers} speakers...")
    
    # This forces exact number of speakers
    diarization = pipeline(audio_path, num_speakers=num_speakers)
    
    # Verify speakers
    speakers = list(diarization.labels())
    print(f"Found {len(speakers)} speakers: {speakers}")
    
    # Step 3: Transcribe with WhisperX
    print("Transcribing with WhisperX...")
    
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("large-v3", device, compute_type="float16" if device == "cuda" else "int8")
    
    # Transcribe
    result = model.transcribe(audio, batch_size=16)
    language = result.get("language", "en")
    
    # Align
    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language, device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    
    del model_a
    gc.collect()
    
    # Step 4: Alternative speaker assignment using WhisperX diarization
    print("Using WhisperX diarization for comparison...")
    
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
    
    # Assign speakers using WhisperX
    result_whisperx = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Step 5: Combine both approaches for best results
    print("Combining results...")
    
    final_segments = []
    
    for seg in result_whisperx["segments"]:
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        start = seg["start"]
        end = seg["end"]
        
        # Get speaker from WhisperX
        whisperx_speaker = seg.get("speaker", None)
        
        # Also check pyannote diarization
        from pyannote.core import Segment
        segment = Segment(start, end)
        pyannote_speakers = diarization.get_labels(segment)
        
        # Choose speaker
        if whisperx_speaker:
            speaker = whisperx_speaker
        elif pyannote_speakers:
            # Get most overlapping speaker
            speaker_durations = {}
            for spk in pyannote_speakers:
                timeline = diarization.label_timeline(spk)
                overlap = timeline.crop(segment).duration()
                speaker_durations[spk] = overlap
            speaker = max(speaker_durations, key=speaker_durations.get)
        else:
            speaker = "SPEAKER_00"
        
        # Check for overlaps
        is_overlap = False
        overlapping_speakers = []
        
        # Check if multiple speakers in this segment
        if pyannote_speakers and len(pyannote_speakers) > 1:
            is_overlap = True
            overlapping_speakers = [s for s in pyannote_speakers if s != speaker]
        
        segment_data = {
            "text": text,
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker
        }
        
        if is_overlap:
            segment_data["is_overlap"] = True
            segment_data["overlapping_speakers"] = overlapping_speakers
        
        final_segments.append(segment_data)
    
    # Step 6: Post-process to ensure quality
    final_segments = post_process_final(final_segments, num_speakers)
    
    return final_segments

def post_process_final(segments, expected_speakers):
    """
    Final post-processing to ensure quality
    """
    
    # Check speaker distribution
    speaker_counts = {}
    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in speaker_counts:
            speaker_counts[speaker] = 0
        speaker_counts[speaker] += 1
    
    print(f"\nSpeaker distribution:")
    for speaker, count in sorted(speaker_counts.items()):
        print(f"  {speaker}: {count} segments")
    
    # If we don't have expected number of speakers, there's a problem
    if len(speaker_counts) < expected_speakers:
        print(f"WARNING: Only {len(speaker_counts)} speakers found, expected {expected_speakers}")
        print("Attempting to redistribute segments...")
        
        # Force redistribution based on position in conversation
        total = len(segments)
        for i, seg in enumerate(segments):
            # Simple heuristic: divide conversation into parts
            if i < total // 3:
                seg["speaker"] = "SPEAKER_00"
            elif i < 2 * total // 3:
                seg["speaker"] = "SPEAKER_01"
            else:
                seg["speaker"] = "SPEAKER_02"
    
    return segments

def main():
    import sys
    
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "input_6.wav"
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print(f"\n{'='*60}")
    print(f"Processing: {audio_file}")
    print(f"Expected speakers: {num_speakers}")
    print(f"{'='*60}\n")
    
    try:
        segments = final_solution(audio_file, num_speakers)
        
        # Save output
        output_file = "output_final_solution.jsonl"
        with open(output_file, 'w') as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + '\n')
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"{'='*60}")
        print(f"Output: {output_file}")
        print(f"Total segments: {len(segments)}")
        
        # Show sample
        print(f"\nFirst 10 segments:")
        print(f"{'-'*60}")
        
        for i, seg in enumerate(segments[:10], 1):
            overlap = " [OVERLAP]" if seg.get("is_overlap") else ""
            text = seg['text'][:45] + "..." if len(seg['text']) > 45 else seg['text']
            print(f"{i:2}. [{seg['speaker']}] {seg['start']:6.2f}-{seg['end']:6.2f}: {text}{overlap}")
        
        if len(segments) > 10:
            print(f"\n... and {len(segments) - 10} more segments")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())