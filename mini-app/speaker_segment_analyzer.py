# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import subprocess
import logging
import json
import os

logger = logging.getLogger("speaker_analyzer")

@dataclass
class SpeakerSegment:
    speaker_id: str
    start: float
    end: float
    text: str = ""
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps_with(self, other: 'SpeakerSegment') -> bool:
        return not (self.end <= other.start or self.start >= other.end)

    def merge_with(self, other: 'SpeakerSegment') -> 'SpeakerSegment':
        return SpeakerSegment(
            speaker_id=self.speaker_id,
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            text=f"{self.text} {other.text}".strip() if self.text or other.text else "",
            confidence=min(self.confidence, other.confidence)
        )

@dataclass
class SpeakerAnalysis:
    speaker_id: str
    segments: List[SpeakerSegment] = field(default_factory=list)
    total_duration: float = 0.0
    merged_segments: List[SpeakerSegment] = field(default_factory=list)

    def add_segment(self, segment: SpeakerSegment):
        self.segments.append(segment)
        self.total_duration += segment.duration

    def merge_overlapping_segments(self, gap_threshold: float = 0.5) -> List[SpeakerSegment]:
        if not self.segments:
            return []

        sorted_segments = sorted(self.segments, key=lambda x: x.start)
        merged = [sorted_segments[0]]

        for current in sorted_segments[1:]:
            last = merged[-1]

            # Check for overlap or small gap
            if current.start <= last.end + gap_threshold:
                # Merge segments
                merged[-1] = last.merge_with(current)
            else:
                merged.append(current)

        self.merged_segments = merged
        return merged

class SpeakerSegmentAnalyzer:
    def __init__(self,
                 min_segment_duration: float = 0.2,
                 merge_gap_threshold: float = 0.5,
                 overlap_tolerance: float = 0.1):
        self.min_segment_duration = min_segment_duration
        self.merge_gap_threshold = merge_gap_threshold
        self.overlap_tolerance = overlap_tolerance
        self.speaker_analyses: Dict[str, SpeakerAnalysis] = {}

    def analyze_segments(self, segments: List[Dict]) -> Dict[str, SpeakerAnalysis]:
        """Analyze segments by speaker and merge overlapping ones"""
        self.speaker_analyses.clear()

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            if speaker not in self.speaker_analyses:
                self.speaker_analyses[speaker] = SpeakerAnalysis(speaker_id=speaker)

            segment = SpeakerSegment(
                speaker_id=speaker,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg.get("text", ""),
                confidence=float(seg.get("confidence", 1.0))
            )

            if segment.duration >= self.min_segment_duration:
                self.speaker_analyses[speaker].add_segment(segment)

        # Merge overlapping segments for each speaker
        for analysis in self.speaker_analyses.values():
            analysis.merge_overlapping_segments(self.merge_gap_threshold)

        return self.speaker_analyses

    def remove_cross_speaker_overlaps(self, segments: List[Dict]) -> List[Dict]:
        """Redistribute overlaps between different speakers instead of trimming"""
        sorted_segs = sorted(segments, key=lambda x: (x["start"], x["end"]))

        # Build overlap graph
        overlaps = []
        for i, seg in enumerate(sorted_segs):
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            speaker = seg.get("speaker", "UNKNOWN")

            for j, other in enumerate(sorted_segs):
                if i >= j:  # Skip self and already checked pairs
                    continue

                other_speaker = other.get("speaker", "UNKNOWN")
                if speaker == other_speaker:  # Same speaker overlaps handled elsewhere
                    continue

                other_start = float(other["start"])
                other_end = float(other["end"])

                # Calculate overlap
                overlap_start = max(seg_start, other_start)
                overlap_end = min(seg_end, other_end)

                if overlap_start < overlap_end - self.overlap_tolerance:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append({
                        "seg1_idx": i,
                        "seg2_idx": j,
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "duration": overlap_duration
                    })

        # Redistribute overlaps
        adjusted_segments = [seg.copy() for seg in sorted_segs]

        for overlap_info in overlaps:
            seg1 = adjusted_segments[overlap_info["seg1_idx"]]
            seg2 = adjusted_segments[overlap_info["seg2_idx"]]

            seg1_start = float(seg1["start"])
            seg1_end = float(seg1["end"])
            seg2_start = float(seg2["start"])
            seg2_end = float(seg2["end"])

            overlap_start = overlap_info["overlap_start"]
            overlap_end = overlap_info["overlap_end"]
            overlap_duration = overlap_info["duration"]

            # Redistribution strategy based on segment priorities
            seg1_duration = seg1_end - seg1_start
            seg2_duration = seg2_end - seg2_start

            # Priority based on: 1) longer segments get priority, 2) earlier start time
            seg1_priority = seg1_duration / (seg1_start + 0.001)
            seg2_priority = seg2_duration / (seg2_start + 0.001)

            if overlap_duration < 0.5:  # Short overlap - allow natural conversation
                # For short overlaps, create a crossfade effect by allowing partial overlap
                # Reduce overlap to 100ms for natural flow
                if seg1_priority > seg2_priority:
                    # seg1 has priority, push seg2 slightly
                    seg2["start"] = overlap_end - 0.1  # Allow 100ms overlap
                else:
                    # seg2 has priority, trim seg1 slightly
                    seg1["end"] = overlap_start + 0.1  # Allow 100ms overlap

            elif overlap_duration < 2.0:  # Medium overlap - redistribute
                # Split the overlap time between segments
                midpoint = (overlap_start + overlap_end) / 2

                if seg1_start < seg2_start:
                    # seg1 starts first, give it time until midpoint
                    seg1["end"] = min(seg1_end, midpoint + 0.05)  # Small overlap for continuity
                    seg2["start"] = max(seg2_start, midpoint - 0.05)
                else:
                    # seg2 starts first
                    seg2["end"] = min(seg2_end, midpoint + 0.05)
                    seg1["start"] = max(seg1_start, midpoint - 0.05)

            else:  # Large overlap - likely an error, needs careful handling
                # For large overlaps, preserve the segment with clearer boundaries
                if seg1_priority > seg2_priority:
                    # Keep seg1 intact, adjust seg2
                    if seg2_start < seg1_start:
                        seg2["end"] = seg1_start - 0.02  # Small gap
                    else:
                        seg2["start"] = seg1_end + 0.02
                else:
                    # Keep seg2 intact, adjust seg1
                    if seg1_start < seg2_start:
                        seg1["end"] = seg2_start - 0.02
                    else:
                        seg1["start"] = seg2_end + 0.02

        # Filter out segments that became too short
        result = []
        for seg in adjusted_segments:
            duration = float(seg["end"]) - float(seg["start"])
            if duration >= self.min_segment_duration:
                result.append(seg)
            else:
                logger.debug(f"Dropping short segment after redistribution: {seg.get('speaker')} [{seg['start']:.2f}-{seg['end']:.2f}]")

        return result

    def get_speaker_timeline(self, speaker_id: str) -> List[Tuple[float, float]]:
        """Get timeline of segments for a specific speaker"""
        if speaker_id not in self.speaker_analyses:
            return []

        analysis = self.speaker_analyses[speaker_id]
        return [(seg.start, seg.end) for seg in analysis.merged_segments]

    def export_analysis(self, output_path: str):
        """Export analysis results to JSON"""
        result = {}
        for speaker_id, analysis in self.speaker_analyses.items():
            result[speaker_id] = {
                "total_duration": analysis.total_duration,
                "segment_count": len(analysis.segments),
                "merged_count": len(analysis.merged_segments),
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "duration": seg.duration,
                        "text": seg.text,
                        "confidence": seg.confidence
                    }
                    for seg in analysis.merged_segments
                ]
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Analysis exported to {output_path}")
        return result


class DemucsVocalSeparator:
    def __init__(self, model: str = "htdemucs", device: str = "cpu"):
        self.model = model
        self.device = device
        self.is_available = self._check_demucs()

    def _check_demucs(self) -> bool:
        """Check if Demucs is installed"""
        try:
            result = subprocess.run(
                ["python", "-m", "demucs", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Demucs not available: {e}")
            return False

    def separate_vocals(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Separate audio into vocals and other stems"""
        if not self.is_available:
            logger.error("Demucs is not installed. Run: pip install demucs")
            return {}

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run Demucs separation
        cmd = [
            "python", "-m", "demucs.separate",
            "-n", self.model,
            "-d", self.device,
            "-o", str(output_dir),
            "--two-stems", "vocals",  # Only separate vocals
            str(audio_path)
        ]

        logger.info(f"Running Demucs separation on {audio_path.name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            logger.error(f"Demucs failed: {result.stderr}")
            raise RuntimeError(f"Demucs separation failed: {result.stderr}")

        # Find output files
        separated_dir = output_dir / self.model / audio_path.stem
        vocals_path = separated_dir / "vocals.wav"
        no_vocals_path = separated_dir / "no_vocals.wav"

        if not vocals_path.exists() or not no_vocals_path.exists():
            raise RuntimeError(f"Separation output not found in {separated_dir}")

        logger.info("Vocal separation completed successfully")
        return {
            "vocals": str(vocals_path),
            "instrumental": str(no_vocals_path),
            "output_dir": str(separated_dir)
        }

    def extract_all_stems(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """Extract all stems (vocals, drums, bass, other)"""
        if not self.is_available:
            return {}

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "demucs.separate",
            "-n", self.model,
            "-d", self.device,
            "-o", str(output_dir),
            str(audio_path)
        ]

        logger.info(f"Extracting all stems from {audio_path.name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            raise RuntimeError(f"Demucs separation failed: {result.stderr}")

        separated_dir = output_dir / self.model / audio_path.stem
        stems = {
            "vocals": separated_dir / "vocals.wav",
            "drums": separated_dir / "drums.wav",
            "bass": separated_dir / "bass.wav",
            "other": separated_dir / "other.wav"
        }

        available_stems = {}
        for name, path in stems.items():
            if path.exists():
                available_stems[name] = str(path)

        logger.info(f"Extracted stems: {list(available_stems.keys())}")
        return available_stems


class DubbingMixer:
    def __init__(self,
                 original_volume: float = 0.3,
                 dubbing_volume: float = 1.0,
                 instrumental_volume: float = 0.8):
        self.original_volume = original_volume
        self.dubbing_volume = dubbing_volume
        self.instrumental_volume = instrumental_volume

    def mix_dubbing_with_instrumental(self,
                                     dubbing_path: str,
                                     instrumental_path: str,
                                     output_path: str,
                                     sync_segments: Optional[List[Dict]] = None) -> str:
        """Mix dubbed audio with instrumental track"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Basic mixing without segment-based processing
        cmd = [
            "ffmpeg", "-i", str(dubbing_path), "-i", str(instrumental_path),
            "-filter_complex",
            f"[0:a]volume={self.dubbing_volume}[dub];"
            f"[1:a]volume={self.instrumental_volume}[inst];"
            f"[dub][inst]amix=inputs=2:duration=longest",
            "-ar", "44100", "-ac", "2",
            "-c:a", "pcm_s16le",  # Use WAV format instead of AAC
            "-y", str(output_path)
        ]

        logger.info("Mixing dubbing with instrumental")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"Audio mixing failed: {result.stderr}")

        logger.info(f"Mixed audio saved to {output_path}")
        return str(output_path)

    def create_adaptive_mix(self,
                          original_path: str,
                          dubbing_path: str,
                          instrumental_path: str,
                          output_path: str,
                          speaker_segments: Dict[str, List[Tuple[float, float]]]) -> str:
        """Create adaptive mix based on speaker segments"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create filter for adaptive mixing based on segments
        filter_parts = []

        # Start with base mixing
        filter_parts.append(f"[0:a]volume={self.original_volume}[orig]")
        filter_parts.append(f"[1:a]volume={self.dubbing_volume}[dub]")
        filter_parts.append(f"[2:a]volume={self.instrumental_volume}[inst]")

        # Mix all together
        filter_parts.append("[orig][dub][inst]amix=inputs=3:duration=longest[out]")

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-i", str(original_path),
            "-i", str(dubbing_path),
            "-i", str(instrumental_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "44100", "-ac", "2",
            "-c:a", "pcm_s16le",  # Use WAV format instead of AAC
            "-y", str(output_path)
        ]

        logger.info("Creating adaptive mix with original, dubbing, and instrumental")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode != 0:
            raise RuntimeError(f"Adaptive mixing failed: {result.stderr}")

        logger.info(f"Adaptive mix saved to {output_path}")
        return str(output_path)


def process_dubbing_pipeline(
    segments_path: str,
    original_audio_path: str,
    dubbed_audio_path: str,
    output_dir: str,
    use_demucs: bool = True
) -> Dict[str, Any]:
    """Main pipeline for processing dubbing with vocal separation"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segments
    with open(segments_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        segments = data.get("segments", [])

    # Analyze speaker segments
    analyzer = SpeakerSegmentAnalyzer()
    speaker_analyses = analyzer.analyze_segments(segments)

    # Remove cross-speaker overlaps
    clean_segments = analyzer.remove_cross_speaker_overlaps(segments)

    # Export analysis
    analysis_path = output_dir / "speaker_analysis.json"
    analyzer.export_analysis(str(analysis_path))

    result = {
        "analysis": analysis_path.as_posix(),
        "clean_segments": clean_segments,
        "speaker_timelines": {}
    }

    # Get speaker timelines
    for speaker_id in speaker_analyses:
        timeline = analyzer.get_speaker_timeline(speaker_id)
        result["speaker_timelines"][speaker_id] = timeline

    # Vocal separation with Demucs if requested
    if use_demucs:
        separator = DemucsVocalSeparator()

        if separator.is_available:
            # Separate vocals from original audio
            demucs_output = output_dir / "demucs_output"
            stems = separator.separate_vocals(
                original_audio_path,
                str(demucs_output)
            )

            if stems:
                # Mix dubbing with instrumental
                mixer = DubbingMixer()
                final_output = output_dir / "final_dubbed.wav"

                mixed_path = mixer.mix_dubbing_with_instrumental(
                    dubbed_audio_path,
                    stems["instrumental"],
                    str(final_output)
                )

                result["final_audio"] = mixed_path
                result["stems"] = stems

                # Also create adaptive mix if we have speaker timelines
                if result["speaker_timelines"]:
                    adaptive_output = output_dir / "adaptive_dubbed.wav"
                    adaptive_path = mixer.create_adaptive_mix(
                        original_audio_path,
                        dubbed_audio_path,
                        stems["instrumental"],
                        str(adaptive_output),
                        result["speaker_timelines"]
                    )
                    result["adaptive_audio"] = adaptive_path
        else:
            logger.warning("Demucs not available, skipping vocal separation")

    return result


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python speaker_segment_analyzer.py <segments.json> <original_audio> <dubbed_audio> [output_dir]")
        sys.exit(1)

    segments_file = sys.argv[1]
    original_audio = sys.argv[2]
    dubbed_audio = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "./output_analysis"

    result = process_dubbing_pipeline(
        segments_file,
        original_audio,
        dubbed_audio,
        output_dir,
        use_demucs=True
    )

    print(f"Pipeline completed. Results saved to {output_dir}")
    print(f"Analysis: {result.get('analysis')}")
    if "final_audio" in result:
        print(f"Final dubbed audio: {result['final_audio']}")
    if "adaptive_audio" in result:
        print(f"Adaptive dubbed audio: {result['adaptive_audio']}")