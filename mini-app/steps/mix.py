from __future__ import annotations

import logging
from math import ceil
from pathlib import Path

from core.pipeline.base import Context

from pydub import AudioSegment, effects


logger = logging.getLogger(__name__)


class MixStep:
    """Timeline-based audio mixing with optional background ducking."""

    name = "Mix"

    def __init__(
        self,
        lufs_target: float = -14.0,
        duck_db: float = -7.0,
        pan_amount: float = 0.0,
    ) -> None:
        self.lufs_target = lufs_target
        self.duck_db = duck_db
        self.pan_amount = pan_amount

    @staticmethod
    def _load(path: Path) -> AudioSegment:
        """Load an audio file as stereo 48kHz."""
        return AudioSegment.from_file(path).set_frame_rate(48000).set_channels(2)

    @staticmethod
    def _total_ms(segments: list[dict]) -> int:
        """Compute total timeline length in milliseconds."""
        if not segments:
            return 0
        max_end = max(seg.get("end", 0.0) for seg in segments)
        return int(ceil(max_end * 1000)) + 200

    @staticmethod
    def _lufs_normalize(seg: AudioSegment, target: float = -14.0) -> AudioSegment:
        """Normalize segment to target LUFS if pyloudnorm available."""
        try:  # pragma: no cover - optional dependency
            import numpy as np
            import pyloudnorm as pyln

            samples = np.array(seg.get_array_of_samples()).astype(np.float32)
            if seg.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            samples /= 2 ** 15
            meter = pyln.Meter(seg.frame_rate)
            loudness = meter.integrated_loudness(samples)
            normed = pyln.normalize.loudness(samples, loudness, target)
            audio = AudioSegment(
                (normed * (2 ** 15)).astype(np.int16).tobytes(),
                frame_rate=seg.frame_rate,
                sample_width=2,
                channels=1,
            )
            return audio.set_channels(2)
        except Exception:  # pragma: no cover - graceful fallback
            return effects.normalize(seg)

    def _pan_for_speaker(self, spk: str | int | None) -> float:
        """Return deterministic pan for a speaker or 0 if disabled."""
        if not self.pan_amount or spk is None:
            return 0.0
        return ((hash(str(spk)) % 2000) / 1000 - 1.0) * self.pan_amount

    def run(self, ctx: Context) -> None:  # pragma: no cover - heavy audio
        artifacts = ctx["artifacts"]
        segments = artifacts.get("segments", [])
        total_ms = self._total_ms(segments)
        speech_bus = AudioSegment.silent(duration=total_ms, frame_rate=48000).set_channels(2)

        placed = 0
        for seg in segments:
            if seg.get("tts_error"):
                continue
            p = seg.get("tts_path")
            if not p:
                continue
            path = Path(p)
            if not path.exists():
                continue
            try:
                clip = self._load(path)
            except Exception:
                continue
            if len(clip) < 5:
                continue
            pan = self._pan_for_speaker(seg.get("speaker") or seg.get("speaker_id"))
            if pan:
                clip = clip.pan(pan)
            start_ms = int(float(seg.get("start", 0)) * 1000)
            speech_bus = speech_bus.overlay(clip, position=start_ms)
            placed += 1

        if placed == 0:
            raise RuntimeError("Mix: no TTS clips placed")

        bg_path = artifacts.get("music_audio") or artifacts.get("original_audio")
        if bg_path:
            bg = self._load(Path(bg_path))
            if len(bg) < total_ms:
                reps = total_ms // len(bg) + 1
                bg *= reps
            bg = bg[:total_ms]
            final = bg.overlay(speech_bus, position=0, gain_during_overlay=self.duck_db)
        else:
            final = speech_bus

        final = self._lufs_normalize(final, self.lufs_target)
        out = Path(ctx["temp_dir"]) / "final_audio.wav"
        final.export(out, format="wav")
        artifacts["final_audio"] = str(out)
        logger.info(
            "Mix: placed=%d clips, total=%.2fs -> %s",
            placed,
            total_ms / 1000.0,
            out,
        )

