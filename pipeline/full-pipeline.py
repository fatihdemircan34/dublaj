#!/usr/bin/env python3
"""
Sentence-Split Diarization Pipeline
Splits WhisperX output at natural sentence boundaries and re-assigns speakers
Perfect for handling incorrectly merged segments
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Any
import re
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentenceSplitDiarizationPipeline:
    """
    Pipeline that splits transcriptions at sentence boundaries
    and correctly assigns speakers to each part
    """
    
    def __init__(
        self,
        hf_token: str,
        device: Optional[str] = None
    ):
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models"""
        try:
            from pyannote.audio import Pipeline
            import whisperx
            
            logger.info("Loading PyAnnote diarization...")
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            if self.device == "cuda":
                self.diarization.to(torch.device("cuda"))
            
            logger.info("Loading WhisperX...")
            self.whisper_model = whisperx.load_model(
                "large-v3",
                self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            logger.info("Models loaded!")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    def process(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process audio with sentence-level speaker assignment"""
        
        logger.info(f"Processing: {audio_path}")
        
        # Step 1: Diarization
        logger.info("Step 1: Running diarization...")
        diarization_result = self._run_diarization(audio_path, num_speakers)
        
        # Step 2: Transcription with word alignment
        logger.info("Step 2: Running WhisperX...")
        transcription_result = self._run_whisperx(audio_path)
        
        # Step 3: Smart sentence splitting
        logger.info("Step 3: Splitting at sentence boundaries...")
        split_segments = self._smart_split_segments(transcription_result)
        
        # Step 4: Assign speakers to each split segment
        logger.info("Step 4: Assigning speakers to split segments...")
        final_segments = self._assign_speakers_with_validation(
            split_segments, diarization_result
        )
        
        # Step 5: Detect overlaps
        logger.info("Step 5: Detecting overlaps...")
        final_segments = self._detect_overlaps(final_segments, diarization_result)
        
        # Step 6: Post-process
        final_segments = self._post_process(final_segments)
        
        return final_segments
    
    def _run_diarization(self, audio_path: str, num_speakers: Optional[int]) -> Dict:
        """Run PyAnnote diarization"""
        
        params = {"num_speakers": num_speakers} if num_speakers else {
            "min_speakers": 1, "max_speakers": 10
        }
        
        diarization = self.diarization(audio_path, **params)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        # Find overlaps
        overlaps = []
        for i, s1 in enumerate(segments):
            for s2 in segments[i+1:]:
                if s1["speaker"] != s2["speaker"]:
                    o_start = max(s1["start"], s2["start"])
                    o_end = min(s1["end"], s2["end"])
                    if o_start < o_end:
                        overlaps.append({
                            "start": o_start,
                            "end": o_end,
                            "speakers": [s1["speaker"], s2["speaker"]]
                        })
        
        logger.info(f"Diarization: {len(set(s['speaker'] for s in segments))} speakers, "
                   f"{len(overlaps)} overlaps")
        
        return {"segments": segments, "overlaps": overlaps}
    
    def _run_whisperx(self, audio_path: str) -> Dict:
        """Run WhisperX with word alignment"""
        
        import whisperx
        
        audio = whisperx.load_audio(audio_path)
        result = self.whisper_model.transcribe(audio, batch_size=16)
        
        # Word alignment
        lang = result.get("language", "en")
        logger.info(f"Language: {lang}")
        
        model_a, metadata = whisperx.load_align_model(lang, self.device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, self.device
        )
        
        return result
    
    def _smart_split_segments(self, transcription: Dict) -> List[Dict]:
        """
        Smart splitting at natural boundaries:
        - Ellipsis (...)
        - Sentence endings (. ! ?)
        - Long pauses
        - Interjections (Mr., Mrs., etc.)
        """
        
        split_segments = []
        
        # Patterns for splitting
        ellipsis_pattern = re.compile(r'\.\.\.')
        sentence_end_pattern = re.compile(r'[.!?]+')
        interjection_pattern = re.compile(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Yes,|No,|Well,|Oh,|Okay,|Right,)')
        
        for segment in transcription["segments"]:
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            # If no word-level timestamps, can't split accurately
            if "words" not in segment or not segment["words"]:
                split_segments.append(segment)
                continue
            
            words = segment["words"]
            current_chunk = []
            current_start = None
            
            for i, word_info in enumerate(words):
                word = word_info.get("word", "").strip()
                if not word:
                    continue
                
                if current_start is None:
                    current_start = word_info.get("start", segment["start"])
                
                current_chunk.append(word_info)
                
                # Check for split points
                should_split = False
                split_reason = None
                
                # 1. Check for ellipsis
                if ellipsis_pattern.search(word):
                    should_split = True
                    split_reason = "ellipsis"
                
                # 2. Check for sentence end
                elif sentence_end_pattern.search(word):
                    should_split = True
                    split_reason = "sentence_end"
                
                # 3. Check for interjection at start of next word
                elif i < len(words) - 1:
                    next_word = words[i + 1].get("word", "")
                    if interjection_pattern.match(next_word):
                        should_split = True
                        split_reason = "interjection"
                
                # 4. Check for long pause (>0.3s)
                if i < len(words) - 1:
                    current_end = word_info.get("end", 0)
                    next_start = words[i + 1].get("start", 0)
                    if next_start - current_end > 0.3:
                        should_split = True
                        split_reason = "pause"
                
                # Create segment if splitting
                if should_split and current_chunk:
                    chunk_text = " ".join(w.get("word", "") for w in current_chunk).strip()
                    
                    # Clean up text
                    chunk_text = re.sub(r'\s+([.!?,])', r'\1', chunk_text)  # Fix punctuation spacing
                    
                    if chunk_text:
                        split_segments.append({
                            "text": chunk_text,
                            "start": current_start,
                            "end": word_info.get("end", current_start + 0.1),
                            "split_reason": split_reason,
                            "original_segment_id": transcription["segments"].index(segment)
                        })
                    
                    current_chunk = []
                    current_start = None
            
            # Handle remaining words
            if current_chunk:
                chunk_text = " ".join(w.get("word", "") for w in current_chunk).strip()
                chunk_text = re.sub(r'\s+([.!?,])', r'\1', chunk_text)
                
                if chunk_text:
                    split_segments.append({
                        "text": chunk_text,
                        "start": current_start,
                        "end": current_chunk[-1].get("end", current_start + 0.1),
                        "split_reason": "segment_end",
                        "original_segment_id": transcription["segments"].index(segment)
                    })
        
        logger.info(f"Split {len(transcription['segments'])} segments into {len(split_segments)} chunks")
        
        # Log splitting reasons
        reasons = {}
        for seg in split_segments:
            reason = seg.get("split_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        logger.info(f"Split reasons: {reasons}")
        
        return split_segments
    
    def _assign_speakers_with_validation(
        self,
        segments: List[Dict],
        diarization: Dict
    ) -> List[Dict]:
        """
        Assign speakers with validation for common patterns
        """
        
        final_segments = []
        diar_segments = diarization["segments"]
        
        for i, seg in enumerate(segments):
            text = seg["text"]
            s_start = seg["start"]
            s_end = seg["end"]
            
            # Find best matching speaker
            speaker_scores = {}
            
            for diar_seg in diar_segments:
                overlap_start = max(s_start, diar_seg["start"])
                overlap_end = min(s_end, diar_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker = diar_seg["speaker"]
                    speaker_scores[speaker] = speaker_scores.get(speaker, 0) + overlap_duration
            
            if speaker_scores:
                best_speaker = max(speaker_scores, key=speaker_scores.get)
                
                # Validation: Check for common misattribution patterns
                # Pattern 1: Interjections often switch speakers
                if text.startswith(("Mr.", "Mrs.", "Ms.", "Yes,", "No,", "Well,", "Right,")):
                    # Check if previous segment has different speaker
                    if i > 0 and final_segments:
                        prev_speaker = final_segments[-1]["speaker"]
                        if prev_speaker != best_speaker:
                            # This is likely a speaker change
                            logger.debug(f"Interjection detected, keeping speaker change: {text[:20]}")
                
                # Pattern 2: Very short segments between same speakers
                if len(text.split()) < 3 and i > 0 and i < len(segments) - 1:
                    if final_segments:
                        prev_speaker = final_segments[-1]["speaker"]
                        # Look ahead
                        next_scores = {}
                        next_seg = segments[i + 1] if i + 1 < len(segments) else None
                        if next_seg:
                            for diar_seg in diar_segments:
                                overlap = max(0, min(next_seg["end"], diar_seg["end"]) - 
                                            max(next_seg["start"], diar_seg["start"]))
                                if overlap > 0:
                                    next_scores[diar_seg["speaker"]] = overlap
                            
                            if next_scores:
                                next_speaker = max(next_scores, key=next_scores.get)
                                if prev_speaker == next_speaker and prev_speaker != best_speaker:
                                    # Likely misattribution
                                    logger.debug(f"Correcting short segment: {text}")
                                    best_speaker = prev_speaker
            else:
                # No overlap - find nearest
                min_dist = float('inf')
                best_speaker = "SPEAKER_00"
                mid = (s_start + s_end) / 2
                
                for diar_seg in diar_segments:
                    if mid >= diar_seg["start"] and mid <= diar_seg["end"]:
                        best_speaker = diar_seg["speaker"]
                        break
                    dist = min(abs(mid - diar_seg["start"]), abs(mid - diar_seg["end"]))
                    if dist < min_dist:
                        min_dist = dist
                        best_speaker = diar_seg["speaker"]
            
            segment = {
                "text": text,
                "start": round(s_start, 3),
                "end": round(s_end, 3),
                "speaker": best_speaker
            }
            
            # Keep split reason for debugging
            if "split_reason" in seg:
                segment["_split"] = seg["split_reason"]
            
            final_segments.append(segment)
        
        return final_segments
    
    def _detect_overlaps(self, segments: List[Dict], diarization: Dict) -> List[Dict]:
        """Detect and mark overlapping segments"""
        
        overlaps = diarization["overlaps"]
        
        for seg in segments:
            for overlap in overlaps:
                # Check if segment is within overlap
                o_start = max(seg["start"], overlap["start"])
                o_end = min(seg["end"], overlap["end"])
                
                if o_end > o_start:
                    # Calculate overlap percentage
                    overlap_duration = o_end - o_start
                    segment_duration = seg["end"] - seg["start"]
                    
                    if overlap_duration / segment_duration > 0.5:
                        seg["is_overlap"] = True
                        seg["overlapping_speakers"] = [
                            s for s in overlap["speakers"] if s != seg["speaker"]
                        ]
                        break
        
        return segments
    
    def _post_process(self, segments: List[Dict]) -> List[Dict]:
        """Clean up and normalize output"""
        
        if not segments:
            return segments
        
        # Sort by time
        segments = sorted(segments, key=lambda x: x["start"])
        
        # Normalize speakers
        speakers = sorted(set(seg["speaker"] for seg in segments))
        speaker_map = {s: f"SPEAKER_{i:02d}" for i, s in enumerate(speakers)}
        
        for seg in segments:
            seg["speaker"] = speaker_map[seg["speaker"]]
            if "overlapping_speakers" in seg:
                seg["overlapping_speakers"] = [
                    speaker_map.get(s, s) for s in seg["overlapping_speakers"]
                ]
            
            # Remove debug info
            if "_split" in seg:
                del seg["_split"]
        
        # Merge consecutive segments from same speaker (optional)
        merged = []
        for seg in segments:
            if merged and merged[-1]["speaker"] == seg["speaker"]:
                # Check if should merge
                gap = seg["start"] - merged[-1]["end"]
                if gap < 0.2 and not seg.get("is_overlap"):  # Small gap, same speaker
                    merged[-1]["text"] += " " + seg["text"]
                    merged[-1]["end"] = seg["end"]
                    continue
            
            merged.append(seg)
        
        return merged

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("-n", "--num-speakers", type=int)
    parser.add_argument("-o", "--output", default="a.jsonF")
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    try:
        pipeline = SentenceSplitDiarizationPipeline(args.hf_token, args.device)
        segments = pipeline.process(args.audio_file, args.num_speakers)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + '\n')
        
        print(f"\n{'='*70}")
        print("✅ SENTENCE-SPLIT DIARIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Segments: {len(segments)}")
        print(f"Speakers: {len(set(s['speaker'] for s in segments))}")
        print(f"Overlaps: {sum(1 for s in segments if s.get('is_overlap'))}")
        print(f"Output: {args.output}\n")
        
        print("First 10 segments:")
        for i, seg in enumerate(segments[:10], 1):
            ov = " [OV]" if seg.get("is_overlap") else ""
            txt = seg['text'][:50] + "..." if len(seg['text']) > 50 else seg['text']
            print(f"{i}. [{seg['speaker']}] {seg['start']:.2f}-{seg['end']:.2f}: {txt}{ov}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())