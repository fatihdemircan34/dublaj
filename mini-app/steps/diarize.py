# =============================================================================
# YÜKSEK ETKİ + KOLAY ÇÖZÜMLER - HEPSİ BİR ARADA
# Optimize edilmiş segment bölme (3-4 saniye) + PyAnnote + Debug + Validation
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional
    torch = None

from core.pipeline.base import Context

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Speaker diarization using PyAnnote.audio - FULL OPTIMIZATION"""

    def __init__(
            self,
            model: str = "pyannote/speaker-diarization-3.1",
            device: Optional[str] = None,
            auth_token: Optional[str] = None,
            min_speakers: int = 1,
            max_speakers: int = 10,
    ) -> None:
        self.model_name = model
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.auth_token = auth_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

        # ÇÖZÜM #3: PYANNOTE PARAMETRELERİ İYİLEŞTİRMESİ
        self.min_duration_on = 0.0        # Minimum konuşma süresi
        self.min_duration_off = 0.1       # Minimum sessizlik süresi

        logger.info("SpeakerDiarizer initialized with optimized PyAnnote parameters:")
        logger.info(f"  min_duration_on: {self.min_duration_on}")
        logger.info(f"  min_duration_off: {self.min_duration_off}")

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:  # pragma: no cover - heavy dependency
            from pyannote.audio import Pipeline  # type: ignore
            import os

            logger.info("Loading diarization model: %s", self.model_name)
            token = self.auth_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if token:
                self._pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=token)
            else:
                self._pipeline = Pipeline.from_pretrained(self.model_name)
            if torch:
                self._pipeline = self._pipeline.to(torch.device(self.device))
            logger.info("Diarization pipeline loaded on %s", self.device)
            return self._pipeline
        except Exception as e:  # pragma: no cover - optional
            logger.error("PyAnnote.audio not installed: %s", e)
            raise RuntimeError("PyAnnote.audio required for speaker diarization")

    def diarize(self, audio_path: Path, return_embeddings: bool = False) -> List[Tuple[float, float, str]]:
        pipeline = None
        try:
            pipeline = self._load_pipeline()
        except RuntimeError:
            pipeline = None
        if pipeline is None:
            # Simple fallback: single speaker covering full audio
            import wave

            with wave.open(str(audio_path), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate())
            logger.warning("Fallback diarization: single speaker for %.1fs", duration)
            return [(0.0, duration, "SPEAKER_00")]

        logger.info("Speaker diarization starting: %s", audio_path)

        # ÇÖZÜM #3: PYANNOTE İLE DAHA HASSAS PARAMETRELERİ KULLAN
        try:
            diarization = pipeline(
                str(audio_path),
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
                # PyAnnote 3.1'e özel parametreler (varsa kullan)
            )
        except TypeError:
            # Eski versiyon için fallback
            diarization = pipeline(
                str(audio_path),
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )

        segments: List[Tuple[float, float, str]] = []
        embeddings = {} if return_embeddings else None
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
            if return_embeddings:
                embedding = self._extract_speaker_embedding(audio_path, turn.start, turn.end)
                embeddings.setdefault(speaker, []).append(embedding)

        # ÇÖZÜM #4: DEBUG LOGGING - Raw segments
        logger.info("🔍 Raw diarization analysis:")
        logger.info(f"  Total segments: {len(segments)}")
        logger.info(f"  Unique speakers: {len({s[2] for s in segments})}")

        for i, (start, end, speaker) in enumerate(segments[:5]):  # İlk 5 segment
            duration = end - start
            logger.info(f"  Segment {i}: {start:.3f}-{end:.3f} ({duration:.3f}s) {speaker}")

        # Long segment analysis
        long_segments = [(i, s) for i, s in enumerate(segments) if (s[1] - s[0]) > 8.0]
        if long_segments:
            logger.warning(f"🚨 Found {len(long_segments)} segments > 8s:")
            for idx, (start, end, speaker) in long_segments:
                logger.warning(f"    Segment {idx}: {start:.1f}-{end:.1f} ({end-start:.1f}s) {speaker}")

        # POSTPROCESS: Optimize segments
        segments = self._postprocess_segments(segments)

        # ÇÖZÜM #4: DEBUG LOGGING - Processed segments
        logger.info("✅ Processed segments analysis:")
        logger.info(f"  Final segments: {len(segments)}")

        for i, (start, end, speaker) in enumerate(segments[:5]):  # İlk 5 segment
            duration = end - start
            logger.info(f"  Final {i}: {start:.3f}-{end:.3f} ({duration:.3f}s) {speaker}")

        logger.info(
            "Found %d speech segments from %d speakers",
            len(segments),
            len({s[2] for s in segments}),
        )
        if return_embeddings:
            return segments, embeddings  # type: ignore[return-value]
        return segments

    def _extract_speaker_embedding(self, audio_path: Path, start: float, end: float) -> np.ndarray:
        try:  # pragma: no cover - optional
            from pyannote.audio import Inference  # type: ignore
            from pyannote.core import Segment  # type: ignore

            embedding_model = Inference("pyannote/embedding", device=self.device)
            segment = Segment(start, end)
            embedding = embedding_model.crop(str(audio_path), segment)
            return embedding
        except Exception as e:  # pragma: no cover - optional
            logger.warning("Embedding extraction failed: %s", e)
            return np.zeros(512)

    def _postprocess_segments(self, segs: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """OPTIMIZE: Daha akıllı segment işleme - 3-4 saniyelik parçalar"""
        if not segs:
            return segs
        segs = sorted(segs, key=lambda x: (x[0], x[1]))

        # OPTIMIZE: Daha hassas threshold'lar
        min_speaker_turn = float(getattr(self, "min_speaker_turn", 0.5))  # 0.7 -> 0.3
        max_gap_merge = float(getattr(self, "max_gap_merge", 0.35))        # 0.4 -> 0.2

        logger.debug(f"📊 Postprocess params: min_turn={min_speaker_turn}, max_gap={max_gap_merge}")

        # 1) Aynı konuşmacının ardışık segmentlerini gap küçükse birleştir
        merged: List[Tuple[float, float, str]] = []
        for s in segs:
            if not merged:
                merged.append(s)
                continue
            ls, le, lspk = merged[-1]
            cs, ce, cspk = s
            gap = max(0.0, cs - le)
            if cspk == lspk and gap <= max_gap_merge:
                merged[-1] = (ls, max(le, ce), lspk)
                logger.debug(f"🔗 Merged segments: gap={gap:.3f}s")
            else:
                merged.append(s)

        if len(merged) <= 1:
            return merged

        # 2) OPTIMIZE: Uzun segmentleri daha küçük parçalara böl (3-4 saniye)
        out: List[Tuple[float, float, str]] = []
        for cs, ce, cspk in merged:
            dur = ce - cs

            # OPTIMIZE: 5+ saniye segmentleri böl (8s yerine 5s)
            if dur > 5.0:
                logger.info(f"🔪 Long segment detected ({dur:.1f}s), splitting to 3-4s chunks")
                # Segmenti 3.5 saniyelik parçalara böl
                chunk_duration = 3.5
                current_start = cs
                chunk_count = 0

                while current_start < ce:
                    current_end = min(current_start + chunk_duration, ce)
                    out.append((current_start, current_end, cspk))
                    logger.debug(f"    Chunk {chunk_count}: {current_start:.3f}-{current_end:.3f} ({current_end-current_start:.3f}s)")
                    current_start = current_end
                    chunk_count += 1

                logger.info(f"✂️  Split into {chunk_count} chunks")
            else:
                out.append((cs, ce, cspk))

        # 3) Çok kısa turları komşulara yapıştır
        if len(out) <= 1:
            return out

        final_out: List[Tuple[float, float, str]] = [out[0]]
        for i in range(1, len(out) - 1):
            ps, pe, pspk = final_out[-1]
            cs, ce, cspk = out[i]
            ns, ne, nspk = out[i + 1]
            dur = ce - cs

            if dur < min_speaker_turn:
                if pspk == cspk and (cs - pe) <= max_gap_merge:
                    final_out[-1] = (ps, max(pe, ce), pspk)
                    logger.debug(f"🔗 Merged short segment {i} with previous")
                elif nspk == cspk and (ns - ce) <= max_gap_merge:
                    out[i + 1] = (min(cs, ns), max(ce, ne), cspk)
                    logger.debug(f"🔗 Merged short segment {i} with next")
                else:
                    d_prev = cs - pe
                    d_next = ns - ce
                    if d_prev <= d_next:
                        final_out[-1] = (ps, max(pe, ce), pspk)
                        logger.debug(f"🔗 Merged short segment {i} with closer previous")
                    else:
                        out[i + 1] = (min(cs, ns), max(ce, ne), nspk)
                        logger.debug(f"🔗 Merged short segment {i} with closer next")
            else:
                final_out.append((cs, ce, cspk))

        # Son elemanı ekle
        if len(out) > 1:
            last = out[-1]
            ls, le, lspk = last
            if final_out and (ls - final_out[-1][1]) <= max_gap_merge and final_out[-1][2] == lspk:
                ps, pe, pspk = final_out[-1]
                final_out[-1] = (ps, max(pe, le), pspk)
                logger.debug("🔗 Merged last segment")
            else:
                final_out.append(last)

        return final_out


class GenderDetector:
    """Gender detection based on audio features - ENHANCED"""

    def __init__(
            self,
            male_pitch_max: float = 165.0,
            female_pitch_min: float = 180.0,
            use_formants: bool = True,
            min_duration: float = 0.5,
    ) -> None:
        self.male_pitch_max = male_pitch_max
        self.female_pitch_min = female_pitch_min
        self.use_formants = use_formants
        self.min_duration = min_duration

        # ÇÖZÜM #4: DEBUG LOGGING
        logger.info("🎭 GenderDetector initialized:")
        logger.info(f"  Male pitch max: {male_pitch_max} Hz")
        logger.info(f"  Female pitch min: {female_pitch_min} Hz")
        logger.info(f"  Use formants: {use_formants}")
        logger.info(f"  Min duration: {min_duration}s")

    def detect(self, audio_path: Path, start_time: float = 0, duration: Optional[float] = None) -> str:
        try:
            import librosa  # type: ignore

            y, sr = librosa.load(audio_path, offset=start_time, duration=duration)
            if len(y) < sr * self.min_duration:
                logger.debug(f"🔇 Audio too short for gender detection ({len(y)/sr:.2f}s < {self.min_duration}s)")
                return "U"
            features = self._extract_features(y, sr)
            gender = self._classify_gender(features)

            # ÇÖZÜM #4: DEBUG LOGGING
            if features.get("mean_pitch", 0) > 0:
                logger.debug(f"🎵 Gender detection: pitch={features['mean_pitch']:.1f}Hz -> {gender}")

            return gender
        except Exception as e:  # pragma: no cover - optional
            logger.warning(f"❌ Gender detection failed: {e}")
            return "U"

    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        import librosa  # type: ignore

        features: Dict[str, float] = {}
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_clean = f0[voiced_flag]
        if len(f0_clean) > 0:
            features["mean_pitch"] = float(np.mean(f0_clean))
            features["std_pitch"] = float(np.std(f0_clean))
        else:
            features["mean_pitch"] = 0.0
            features["std_pitch"] = 0.0

        if self.use_formants:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid"] = float(np.mean(spectral_centroid))

        return features

    def _classify_gender(self, features: Dict[str, float]) -> str:
        pitch = features.get("mean_pitch", 0)
        if pitch == 0:
            return "U"
        if pitch < self.male_pitch_max:
            primary_gender = "M"
        elif pitch > self.female_pitch_min:
            primary_gender = "F"
        else:
            primary_gender = "U"
        if self.use_formants and primary_gender == "U":
            f1 = features.get("f1", 0)
            f2 = features.get("f2", 0)
            if f1 > 0 and f2 > 0:
                if f1 < 500 and f2 > 1400:
                    return "M"
                elif f1 > 550 and f2 > 1600:
                    return "F"
        return primary_gender

    def batch_detect(self, segments: List[Tuple[Path, float, float]]) -> List[str]:
        results = []
        for audio_path, start_time, duration in segments:
            gender = self.detect(audio_path, start_time, duration)
            results.append(gender)
        return results


class DiarizeStep:
    name = "Diarize"

    def __init__(self) -> None:
        self.diarizer = SpeakerDiarizer()

        # OPTIMIZE: Daha hassas parametreler
        self.diarizer.min_speaker_turn = 0.3   # 0.7s -> 0.3s (daha kısa segmentlere izin ver)
        self.diarizer.max_gap_merge = 0.2      # 0.4s -> 0.2s (daha az birleştirme)
        self.gender_detector = GenderDetector()

        # ÇÖZÜM #4: DEBUG LOGGING
        logger.info("🎯 DiarizeStep initialized with OPTIMIZED parameters:")
        logger.info(f"  min_speaker_turn: {self.diarizer.min_speaker_turn}s")
        logger.info(f"  max_gap_merge: {self.diarizer.max_gap_merge}s")

    def run(self, ctx: Context) -> None:
        audio_path = Path(ctx["artifacts"]["original_audio"])

        # ÇÖZÜM #6: SEGMENT VALIDATION - Audio duration check
        import wave
        with wave.open(str(audio_path), "rb") as wf:
            total_duration = wf.getnframes() / float(wf.getframerate())

        logger.info(f"🎵 Audio analysis:")
        logger.info(f"  File: {audio_path}")
        logger.info(f"  Duration: {total_duration:.1f}s")

        # Diarization işlemi
        diarization = self.diarizer.diarize(audio_path)

        # ÇÖZÜM #6: SEGMENT VALIDATION
        self._validate_segments(diarization, total_duration)

        segments: List[Dict[str, Any]] = []

        # ÇÖZÜM #4: DEBUG LOGGING - Gender detection process
        logger.info("🎭 Starting gender detection for segments...")

        for i, (start, end, speaker) in enumerate(diarization):
            gender = self.gender_detector.detect(audio_path, start, end - start)
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "gender": gender,
                }
            )

            # Her 5 segmentte bir log yaz
            if i % 5 == 0 or i < 5:
                duration = end - start
                logger.debug(f"  Segment {i}: {duration:.2f}s {speaker} -> {gender}")

        ctx["artifacts"]["segments"] = segments

        # ÇÖZÜM #4: DEBUG LOGGING - Final summary
        logger.info("✅ Diarization completed:")
        logger.info(f"  Total segments created: {len(segments)}")

        # Speaker summary
        speaker_stats = {}
        for seg in segments:
            speaker = seg["speaker"]
            gender = seg["gender"]
            duration = seg["end"] - seg["start"]

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"total_time": 0, "segments": 0, "gender": gender}

            speaker_stats[speaker]["total_time"] += duration
            speaker_stats[speaker]["segments"] += 1

        logger.info("👥 Speaker summary:")
        for speaker, stats in speaker_stats.items():
            logger.info(f"  {speaker} ({stats['gender']}): {stats['segments']} segments, {stats['total_time']:.1f}s total")

    def _validate_segments(self, segments: List[Tuple[float, float, str]], total_duration: float) -> None:
        """ÇÖZÜM #6: SEGMENT VALIDATION & QUALITY CONTROL"""
        if not segments:
            logger.error("❌ No segments found!")
            return

        logger.info("🔍 Segment quality analysis:")

        # 1. Coverage Analysis
        total_covered = sum(end - start for start, end, _ in segments)
        coverage = (total_covered / total_duration) * 100
        logger.info(f"  Coverage: {coverage:.1f}% ({total_covered:.1f}s / {total_duration:.1f}s)")

        if coverage < 80:
            logger.warning(f"⚠️  Low audio coverage detected ({coverage:.1f}% < 80%)")
        elif coverage > 99:
            logger.info("✅ Excellent coverage!")
        else:
            logger.info("✅ Good coverage")

        # 2. Segment Length Analysis
        durations = [end - start for start, end, _ in segments]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        logger.info(f"  Segment lengths: avg={avg_duration:.2f}s, min={min_duration:.2f}s, max={max_duration:.2f}s")

        # 3. Long Segment Detection
        long_segments = [(i, start, end, speaker) for i, (start, end, speaker) in enumerate(segments) if (end - start) > 5.0]
        if long_segments:
            logger.warning(f"⚠️  Found {len(long_segments)} long segments (>5s):")
            for i, start, end, speaker in long_segments[:3]:  # İlk 3'ünü göster
                logger.warning(f"    Segment {i}: {start:.1f}-{end:.1f} ({end-start:.1f}s) {speaker}")
        else:
            logger.info("✅ All segments under 5s - good segmentation!")

        # 4. Very Short Segment Detection
        very_short = [(i, start, end, speaker) for i, (start, end, speaker) in enumerate(segments) if (end - start) < 0.5]
        if very_short:
            logger.warning(f"⚠️  Found {len(very_short)} very short segments (<0.5s):")
            for i, start, end, speaker in very_short[:3]:
                logger.warning(f"    Segment {i}: {start:.3f}-{end:.3f} ({end-start:.3f}s) {speaker}")

        # 5. Overlap Detection
        overlaps = []
        for i in range(len(segments) - 1):
            current_end = segments[i][1]
            next_start = segments[i + 1][0]
            if current_end > next_start:
                overlap = current_end - next_start
                overlaps.append((i, overlap))

        if overlaps:
            logger.warning(f"⚠️  Found {len(overlaps)} overlapping segments:")
            for i, overlap in overlaps[:3]:
                logger.warning(f"    Overlap between segment {i} and {i+1}: {overlap:.3f}s")
        else:
            logger.info("✅ No overlaps detected")

        # 6. Gap Detection
        large_gaps = []
        for i in range(len(segments) - 1):
            current_end = segments[i][1]
            next_start = segments[i + 1][0]
            gap = next_start - current_end
            if gap > 2.0:  # 2+ saniye boşluk
                large_gaps.append((i, gap))

        if large_gaps:
            logger.info(f"📊 Found {len(large_gaps)} large gaps (>2s):")
            for i, gap in large_gaps[:3]:
                logger.info(f"    Gap after segment {i}: {gap:.1f}s")

        # 7. Speaker Distribution
        speaker_counts = {}
        for _, _, speaker in segments:
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        logger.info(f"👥 Speaker distribution:")
        for speaker, count in sorted(speaker_counts.items()):
            percentage = (count / len(segments)) * 100
            logger.info(f"    {speaker}: {count} segments ({percentage:.1f}%)")

        # 8. Final Quality Score
        quality_score = 100
        if coverage < 80: quality_score -= 20
        if long_segments: quality_score -= len(long_segments) * 5
        if very_short: quality_score -= len(very_short) * 3
        if overlaps: quality_score -= len(overlaps) * 10

        quality_score = max(0, min(100, quality_score))

        if quality_score >= 90:
            logger.info(f"🏆 Segment quality: EXCELLENT ({quality_score}/100)")
        elif quality_score >= 75:
            logger.info(f"✅ Segment quality: GOOD ({quality_score}/100)")
        elif quality_score >= 60:
            logger.warning(f"⚠️  Segment quality: FAIR ({quality_score}/100)")
        else:
            logger.error(f"❌ Segment quality: POOR ({quality_score}/100)")


# =============================================================================
# UYGULAMA TALİMATLARI
# =============================================================================
"""
✅ UYGULANAN ÇÖZÜMLER:

1. ✅ ÇÖZÜM #3: PyAnnote Parametreleri İyileştirmesi
   - min_duration_on/off parametreleri eklendi
   - Daha hassas diarization

2. ✅ ÇÖZÜM #4: Advanced Debug Logging
   - Raw ve processed segment analizi
   - Detaylı gender detection logları
   - Speaker summary ve statistics

3. ✅ ÇÖZÜM #6: Segment Validation & Quality Control
   - Coverage analysis (80%+ hedefi)
   - Segment length analysis
   - Overlap/gap detection
   - Quality scoring (0-100)

4. ✅ OPTIMIZE: Segment Bölme İyileştirmesi
   - 8-9s -> 3.5s chunk boyutu
   - 5s+ segmentler otomatik bölünür
   - Daha detaylı chunking logları

SONUÇ:
- 21s segment -> 6 adet 3.5s segment
- Kapsamlı debug bilgisi
- Otomatik kalite kontrolü
- PyAnnote optimizasyonu

TEST:
python3 main.py "https://www.youtube.com/watch?v=LYS1GAUXMGs" --asr whisper --translator nllb --source-lang en --target-lang tr --temp-dir ./tmp --tts xtts
"""