#!/usr/bin/env python3
"""
Sync Drift Monitor and Corrector
Monitors audio-video synchronization and provides real-time corrections
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SyncPoint:
    """Represents a synchronization checkpoint"""
    segment_id: int
    original_time: float
    actual_time: float
    drift: float
    corrected: bool = False
    correction_applied: float = 0.0

@dataclass
class DriftStatistics:
    """Statistics about drift throughout the video"""
    total_segments: int
    max_drift: float
    avg_drift: float
    drift_variance: float
    sync_points: List[SyncPoint]
    final_drift: float

class SyncDriftMonitor:
    """Monitors and corrects audio-video synchronization drift"""

    def __init__(self,
                 max_allowed_drift: float = 0.5,
                 sync_interval: int = 10,
                 correction_factor: float = 0.5):
        """
        Args:
            max_allowed_drift: Maximum drift before correction (seconds)
            sync_interval: Create sync points every N segments
            correction_factor: How much drift to correct (0-1)
        """
        self.max_allowed_drift = max_allowed_drift
        self.sync_interval = sync_interval
        self.correction_factor = correction_factor

        # Tracking
        self.sync_points: List[SyncPoint] = []
        self.current_drift = 0.0
        self.segments_processed = 0

    def add_segment(self,
                    segment_id: int,
                    original_start: float,
                    original_duration: float,
                    actual_duration: float) -> Tuple[float, bool]:
        """
        Process a segment and calculate timing adjustment.

        Args:
            segment_id: Unique segment identifier
            original_start: Original start time in video
            original_duration: Original segment duration
            actual_duration: Actual TTS output duration

        Returns:
            (adjustment_seconds, is_sync_point)
        """
        # Calculate drift for this segment
        segment_drift = actual_duration - original_duration
        self.current_drift += segment_drift

        # Determine if this is a sync point
        is_sync_point = (
            self.segments_processed > 0 and
            self.segments_processed % self.sync_interval == 0
        )

        adjustment = 0.0
        corrected = False

        if is_sync_point:
            # Check if correction is needed
            if abs(self.current_drift) > self.max_allowed_drift:
                # Calculate correction
                adjustment = -self.current_drift * self.correction_factor
                self.current_drift += adjustment
                corrected = True

                logger.info(f"Sync point {segment_id}: Drift={self.current_drift:.3f}s, "
                           f"Correction={adjustment:.3f}s")

            # Record sync point
            sync_point = SyncPoint(
                segment_id=segment_id,
                original_time=original_start,
                actual_time=original_start + self.current_drift,
                drift=self.current_drift,
                corrected=corrected,
                correction_applied=adjustment
            )
            self.sync_points.append(sync_point)

        self.segments_processed += 1
        return adjustment, is_sync_point

    def get_statistics(self) -> DriftStatistics:
        """Calculate and return drift statistics"""
        if not self.sync_points:
            return DriftStatistics(
                total_segments=self.segments_processed,
                max_drift=0.0,
                avg_drift=0.0,
                drift_variance=0.0,
                sync_points=[],
                final_drift=self.current_drift
            )

        drifts = [sp.drift for sp in self.sync_points]
        return DriftStatistics(
            total_segments=self.segments_processed,
            max_drift=max(abs(d) for d in drifts),
            avg_drift=np.mean(drifts),
            drift_variance=np.var(drifts),
            sync_points=self.sync_points,
            final_drift=self.current_drift
        )

    def export_report(self, output_path: Path):
        """Export drift analysis report"""
        stats = self.get_statistics()
        report = {
            "statistics": asdict(stats),
            "sync_points": [asdict(sp) for sp in self.sync_points],
            "configuration": {
                "max_allowed_drift": self.max_allowed_drift,
                "sync_interval": self.sync_interval,
                "correction_factor": self.correction_factor
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Sync drift report exported to {output_path}")

class AdaptiveDriftCorrector:
    """Advanced drift correction with adaptive strategies"""

    def __init__(self):
        self.history: List[float] = []
        self.prediction_window = 5

    def predict_drift(self, segment_idx: int) -> float:
        """Predict drift for upcoming segments based on history"""
        if len(self.history) < self.prediction_window:
            return 0.0

        # Use linear regression to predict trend
        recent = self.history[-self.prediction_window:]
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)

        # Predict next drift
        predicted = np.polyval(coeffs, len(recent))
        return predicted

    def calculate_preemptive_correction(self,
                                        current_drift: float,
                                        predicted_drift: float) -> float:
        """Calculate preemptive correction to prevent future drift"""
        # If drift is accelerating, apply stronger correction
        if abs(predicted_drift) > abs(current_drift):
            correction_strength = 0.7  # Stronger correction
        else:
            correction_strength = 0.5  # Normal correction

        # Calculate correction
        total_drift = current_drift + predicted_drift * 0.3  # Weight prediction
        correction = -total_drift * correction_strength

        return correction

    def update(self, drift: float):
        """Update drift history"""
        self.history.append(drift)

        # Keep history size manageable
        if len(self.history) > 100:
            self.history = self.history[-100:]

def calculate_optimal_stretch_factor(
        original_duration: float,
        tts_duration: float,
        max_stretch: float = 1.5) -> float:
    """
    Calculate optimal stretch factor with quality constraints.

    Args:
        original_duration: Target duration
        tts_duration: Actual TTS output duration
        max_stretch: Maximum stretch factor for quality

    Returns:
        Optimal stretch factor
    """
    if tts_duration == 0:
        return 1.0

    ideal_factor = original_duration / tts_duration

    # Clamp to quality limits
    if ideal_factor > max_stretch:
        return max_stretch
    elif ideal_factor < 1.0 / max_stretch:
        return 1.0 / max_stretch

    return ideal_factor

def analyze_segment_timing(segments: List[Dict]) -> Dict:
    """Analyze timing characteristics of segments"""
    if not segments:
        return {}

    durations = []
    gaps = []
    speakers = {}

    for i, seg in enumerate(segments):
        duration = seg['end'] - seg['start']
        durations.append(duration)

        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(duration)

        if i > 0:
            gap = seg['start'] - segments[i-1]['end']
            gaps.append(gap)

    analysis = {
        'total_segments': len(segments),
        'total_duration': sum(durations),
        'avg_segment_duration': np.mean(durations),
        'std_segment_duration': np.std(durations),
        'avg_gap': np.mean(gaps) if gaps else 0,
        'speakers': {}
    }

    for speaker, spk_durations in speakers.items():
        analysis['speakers'][speaker] = {
            'segments': len(spk_durations),
            'total_duration': sum(spk_durations),
            'avg_duration': np.mean(spk_durations)
        }

    return analysis

if __name__ == "__main__":
    # Test the sync monitor
    monitor = SyncDriftMonitor(max_allowed_drift=0.3, sync_interval=5)

    # Simulate segments with drift
    test_segments = [
        (1.0, 1.1),  # 0.1s drift
        (2.0, 2.2),  # 0.2s drift
        (1.5, 1.4),  # -0.1s drift
        (3.0, 3.3),  # 0.3s drift
        (2.5, 2.8),  # 0.3s drift - sync point here
        (1.8, 1.7),  # -0.1s drift
        (2.2, 2.4),  # 0.2s drift
        (1.5, 1.6),  # 0.1s drift
        (2.0, 2.0),  # 0s drift
        (3.0, 3.5),  # 0.5s drift - sync point here
    ]

    for i, (orig_dur, actual_dur) in enumerate(test_segments):
        adjustment, is_sync = monitor.add_segment(
            segment_id=i,
            original_start=i * 2.0,
            original_duration=orig_dur,
            actual_duration=actual_dur
        )

        if is_sync:
            print(f"Sync point at segment {i}: adjustment={adjustment:.3f}s")

    # Get statistics
    stats = monitor.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total segments: {stats.total_segments}")
    print(f"  Max drift: {stats.max_drift:.3f}s")
    print(f"  Average drift: {stats.avg_drift:.3f}s")
    print(f"  Final drift: {stats.final_drift:.3f}s")

    # Export report
    monitor.export_report(Path("sync_drift_report.json"))