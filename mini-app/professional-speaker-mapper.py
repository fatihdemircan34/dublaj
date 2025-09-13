from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import copy
from collections import defaultdict, Counter
from scipy import signal
from scipy.spatial.distance import cosine
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

# ------------------ Timeline araçları (önceki kod aynı) ------------------

@dataclass(frozen=True)
class DiarSeg:
    start: float
    end: float
    speaker: str

@dataclass
class TimelineSeg:
    start: float
    end: float
    speakers: Tuple[str, ...]
    mode: str
    channels: Optional[Dict[str, str]] = None

def _midpoint(a: float, b: float) -> float:
    return (a + b) / 2.0

def _normalize_segs(segs: List[Dict[str, Any]]) -> List[DiarSeg]:
    out = []
    for s in segs:
        st = float(s["start"]); en = float(s["end"])
        if en <= st:
            continue
        out.append(DiarSeg(st, en, str(s["speaker"])))
    return sorted(out, key=lambda x: (x.start, x.end))

def _active_speakers_at(t: float, segs: List[DiarSeg]) -> List[str]:
    return [s.speaker for s in segs if s.start <= t < s.end]

def build_flat_timeline(
        diar_segments: List[Dict[str, Any]],
        *,
        stereo_threshold: float = 0.20,
        epsilon: float = 1e-6
) -> List[TimelineSeg]:
    segs = _normalize_segs(diar_segments)
    if not segs:
        return []
    cuts = sorted({s.start for s in segs} | {s.end for s in segs})
    out: List[TimelineSeg] = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i+1]
        if b - a <= epsilon:
            continue
        mid = _midpoint(a, b)
        active = sorted(set(_active_speakers_at(mid, segs)))
        if not active:
            continue
        if len(active) == 1:
            seg = TimelineSeg(a, b, (active[0],), "mono")
        elif len(active) == 2 and (b - a) <= stereo_threshold:
            L, R = active[0], active[1]
            seg = TimelineSeg(a, b, (L, R), "stereo", channels={"L": L, "R": R})
        else:
            seg = TimelineSeg(a, b, tuple(active), "multi")
        if out and out[-1].mode == seg.mode and out[-1].speakers == seg.speakers and abs(out[-1].end - seg.start) <= epsilon:
            out[-1].end = seg.end
        else:
            out.append(seg)
    return out

# ------------------ PROFESYONEL KONUŞMACI EŞLEMESİ ------------------

@dataclass
class AcousticFeatures:
    """Akustik özellikler veri yapısı"""
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    zero_crossing_rate: float
    spectral_centroid: float
    formants: List[float] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Özellikleri tek bir vektöre dönüştür"""
        features = []
        features.extend(self.mfcc[:13])  # İlk 13 MFCC katsayısı
        features.extend([
            self.pitch_mean, self.pitch_std,
            self.energy_mean, self.energy_std,
            self.zero_crossing_rate, self.spectral_centroid
        ])
        features.extend(self.formants[:3] if self.formants else [0, 0, 0])
        return np.array(features)

@dataclass
class SpeakerEmbedding:
    """Konuşmacı embedding vektörü"""
    speaker_id: str
    feature_vectors: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    variance: Optional[np.ndarray] = None
    sample_count: int = 0

    def update(self, features: np.ndarray):
        """Yeni özellik vektörü ile embedding'i güncelle"""
        self.feature_vectors.append(features)
        self.sample_count += 1

        if len(self.feature_vectors) >= 2:
            # Centroid ve varyansı hesapla
            vectors = np.array(self.feature_vectors)
            self.centroid = np.mean(vectors, axis=0)
            self.variance = np.var(vectors, axis=0)

    def similarity(self, features: np.ndarray) -> float:
        """Verilen özellik vektörü ile benzerlik skoru hesapla"""
        if self.centroid is None:
            if self.feature_vectors:
                return 1.0 - cosine(features, self.feature_vectors[0])
            return 0.0

        # Mahalanobis benzeri mesafe hesapla
        diff = features - self.centroid
        if self.variance is not None and np.all(self.variance > 0):
            weighted_diff = diff / (self.variance + 1e-6)
            distance = np.sqrt(np.sum(weighted_diff ** 2))
            # Mesafeyi benzerlik skoruna dönüştür (0-1 arası)
            similarity = np.exp(-distance / 10.0)
        else:
            # Basit cosine benzerliği
            similarity = 1.0 - cosine(features, self.centroid)

        return max(0.0, min(1.0, similarity))

@dataclass
class TransitionModel:
    """Konuşmacı geçiş modeli"""
    transition_counts: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    speaker_durations: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    min_speaker_duration: float = 0.5  # Minimum konuşma süresi (saniye)

    def add_transition(self, from_speaker: str, to_speaker: str):
        """Konuşmacı geçişi ekle"""
        self.transition_counts[(from_speaker, to_speaker)] += 1

    def add_duration(self, speaker: str, duration: float):
        """Konuşmacı süresini ekle"""
        self.speaker_durations[speaker].append(duration)

    def get_transition_probability(self, from_speaker: str, to_speaker: str) -> float:
        """Geçiş olasılığını hesapla"""
        total_from = sum(self.transition_counts[(from_speaker, s)]
                         for s in set(t[1] for t in self.transition_counts.keys()
                                      if t[0] == from_speaker))
        if total_from == 0:
            return 0.1  # Default düşük olasılık

        count = self.transition_counts.get((from_speaker, to_speaker), 0)
        return (count + 1) / (total_from + 10)  # Laplace smoothing

    def is_duration_plausible(self, speaker: str, duration: float) -> bool:
        """Konuşma süresinin makul olup olmadığını kontrol et"""
        if duration < self.min_speaker_duration:
            return False

        if speaker in self.speaker_durations and self.speaker_durations[speaker]:
            durations = self.speaker_durations[speaker]
            mean_dur = np.mean(durations)
            std_dur = np.std(durations)
            # 3 sigma kuralı
            if std_dur > 0:
                z_score = abs(duration - mean_dur) / std_dur
                return z_score < 3

        return True

class ProfessionalSpeakerMapper:
    """
    Profesyonel seviye konuşmacı eşleme sistemi.
    Akustik analiz, kontekst-aware modelleme ve çoklu geçişli iyileştirme kullanır.
    """

    def __init__(self,
                 use_acoustic_features: bool = True,
                 use_sequential_model: bool = True,
                 use_multipass: bool = True,
                 n_passes: int = 3,
                 confidence_threshold: float = 0.7,
                 audio_path: Optional[str] = None):
        """
        Args:
            use_acoustic_features: Akustik özellik analizi kullan
            use_sequential_model: Sıralı model kullan
            use_multipass: Çoklu geçişli iyileştirme kullan
            n_passes: İyileştirme geçiş sayısı
            confidence_threshold: Minimum güven eşiği
            audio_path: Ses dosyası yolu (akustik analiz için)
        """
        self.use_acoustic_features = use_acoustic_features
        self.use_sequential_model = use_sequential_model
        self.use_multipass = use_multipass
        self.n_passes = n_passes
        self.confidence_threshold = confidence_threshold
        self.audio_path = audio_path

        # İç veri yapıları
        self.speaker_embeddings: Dict[str, SpeakerEmbedding] = {}
        self.transition_model = TransitionModel()
        self.acoustic_cache: Dict[Tuple[float, float], AcousticFeatures] = {}

    def map_speakers(self,
                     segments: List[Dict[str, Any]],
                     words: List[Dict[str, Any]],
                     diarization: List[Dict[str, Any]],
                     timeline: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Ana eşleme fonksiyonu - üç aşamalı profesyonel eşleme
        """
        # Veriyi normalize et
        segments = copy.deepcopy(segments)
        words = copy.deepcopy(words)
        diar_segments = self._normalize_diarization(diarization)
        timeline_segs = self._normalize_timeline(timeline) if timeline else None

        # Aşama 1: Akustik Parmak İzi Eşleştirmesi
        if self.use_acoustic_features and self.audio_path:
            print("[PHASE 1] Acoustic fingerprint matching...")
            segments = self._phase1_acoustic_matching(segments, diar_segments)
        else:
            # Fallback: Basit zamansal eşleme
            segments = self._basic_temporal_matching(segments, diar_segments)

        # Aşama 2: Kontekst-Aware Sıralı Model
        if self.use_sequential_model:
            print("[PHASE 2] Context-aware sequential modeling...")
            segments = self._phase2_sequential_modeling(segments, diar_segments)

        # Aşama 3: Çoklu Geçişli İyileştirme
        if self.use_multipass:
            print("[PHASE 3] Multi-pass refinement...")
            for pass_num in range(self.n_passes):
                segments = self._phase3_multipass_refinement(
                    segments, diar_segments, timeline_segs, pass_num
                )

        # Kelime seviyesi eşleme
        words = self._assign_words_from_segments(words, segments, diar_segments)

        # Son tutarlılık kontrolü ve temporal smoothing
        segments, words = self._final_consistency_check(segments, words)

        return segments, words

    def _normalize_diarization(self, diar: List[Dict]) -> List[Dict]:
        """Diarizasyon verilerini normalize et"""
        normalized = []
        for d in diar:
            normalized.append({
                "start": float(d["start"]),
                "end": float(d["end"]),
                "speaker": str(d["speaker"]),
                "duration": float(d["end"]) - float(d["start"])
            })
        return sorted(normalized, key=lambda x: (x["start"], x["end"]))

    def _normalize_timeline(self, timeline: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """Timeline verilerini normalize et"""
        if not timeline:
            return None
        normalized = []
        for t in timeline:
            normalized.append({
                "start": float(t["start"]),
                "end": float(t["end"]),
                "mode": t["mode"],
                "speakers": t["speakers"],
                "channels": t.get("channels", {}),
                "duration": float(t["end"]) - float(t["start"])
            })
        return sorted(normalized, key=lambda x: x["start"])

    def _extract_acoustic_features(self, start: float, end: float) -> Optional[AcousticFeatures]:
        """
        Belirli zaman aralığı için akustik özellikler çıkar
        Not: Gerçek implementasyonda librosa veya pyAudioAnalysis kullanılmalı
        """
        # Cache kontrolü
        cache_key = (round(start, 2), round(end, 2))
        if cache_key in self.acoustic_cache:
            return self.acoustic_cache[cache_key]

        # Simüle edilmiş akustik özellikler (gerçek implementasyonda ses dosyasından çıkarılmalı)
        # Bu bir placeholder - gerçek uygulamada librosa kullanılmalı
        np.random.seed(int(start * 1000))  # Deterministik simülasyon

        features = AcousticFeatures(
            mfcc=np.random.randn(13),
            pitch_mean=100 + np.random.randn() * 20,
            pitch_std=10 + abs(np.random.randn() * 5),
            energy_mean=0.5 + np.random.randn() * 0.2,
            energy_std=0.1 + abs(np.random.randn() * 0.05),
            zero_crossing_rate=0.1 + np.random.randn() * 0.05,
            spectral_centroid=1000 + np.random.randn() * 200,
            formants=[700 + np.random.randn() * 100,
                      1220 + np.random.randn() * 150,
                      2600 + np.random.randn() * 200]
        )

        self.acoustic_cache[cache_key] = features
        return features

    def _phase1_acoustic_matching(self,
                                  segments: List[Dict],
                                  diar_segments: List[Dict]) -> List[Dict]:
        """
        Aşama 1: Akustik parmak izi eşleştirmesi
        """
        # Her diarizasyon segmenti için speaker embedding oluştur
        for diar_seg in diar_segments:
            speaker_id = diar_seg["speaker"]
            if speaker_id not in self.speaker_embeddings:
                self.speaker_embeddings[speaker_id] = SpeakerEmbedding(speaker_id)

            # Akustik özellikleri çıkar ve embedding'e ekle
            features = self._extract_acoustic_features(diar_seg["start"], diar_seg["end"])
            if features:
                feature_vector = features.to_vector()
                self.speaker_embeddings[speaker_id].update(feature_vector)

        # Her STT segmenti için en uygun konuşmacıyı bul
        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))

            if seg_end <= seg_start:
                continue

            # Segment için akustik özellikler çıkar
            seg_features = self._extract_acoustic_features(seg_start, seg_end)
            if not seg_features:
                continue

            seg_vector = seg_features.to_vector()

            # En benzer speaker embedding'i bul
            best_speaker = None
            best_similarity = 0.0

            for speaker_id, embedding in self.speaker_embeddings.items():
                similarity = embedding.similarity(seg_vector)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker_id

            # Zamansal overlap ile kombine et
            temporal_scores = self._calculate_temporal_scores(seg, diar_segments)

            # Akustik ve zamansal skorları birleştir
            combined_scores = {}
            for speaker_id in self.speaker_embeddings.keys():
                acoustic_score = self.speaker_embeddings[speaker_id].similarity(seg_vector)
                temporal_score = temporal_scores.get(speaker_id, 0.0)
                # Ağırlıklı kombinasyon
                combined_scores[speaker_id] = 0.6 * acoustic_score + 0.4 * temporal_score

            if combined_scores:
                best_speaker = max(combined_scores.items(), key=lambda x: x[1])[0]
                best_score = combined_scores[best_speaker]

                seg["speaker"] = best_speaker
                seg["acoustic_confidence"] = best_score
                seg["assignment_method"] = "acoustic"

        return segments

    def _calculate_temporal_scores(self, segment: Dict, diar_segments: List[Dict]) -> Dict[str, float]:
        """Zamansal örtüşme skorlarını hesapla"""
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))
        seg_duration = seg_end - seg_start

        scores = {}
        for diar_seg in diar_segments:
            overlap_start = max(seg_start, diar_seg["start"])
            overlap_end = min(seg_end, diar_seg["end"])
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                overlap_ratio = overlap_duration / seg_duration
                speaker = diar_seg["speaker"]
                scores[speaker] = scores.get(speaker, 0.0) + overlap_ratio

        # Normalize skorlar
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}

        return scores

    def _phase2_sequential_modeling(self,
                                    segments: List[Dict],
                                    diar_segments: List[Dict]) -> List[Dict]:
        """
        Aşama 2: Kontekst-aware sıralı modelleme
        """
        # Geçiş modelini eğit
        for i in range(len(diar_segments) - 1):
            curr_speaker = diar_segments[i]["speaker"]
            next_speaker = diar_segments[i + 1]["speaker"]
            self.transition_model.add_transition(curr_speaker, next_speaker)
            self.transition_model.add_duration(curr_speaker, diar_segments[i]["duration"])

        if diar_segments:
            last_speaker = diar_segments[-1]["speaker"]
            self.transition_model.add_duration(last_speaker, diar_segments[-1]["duration"])

        # Viterbi benzeri algoritma ile optimal yolu bul
        segments = self._viterbi_decoding(segments, diar_segments)

        # Kısa segmentleri birleştir
        segments = self._merge_short_segments(segments)

        # Mantıksız geçişleri düzelt
        segments = self._fix_implausible_transitions(segments)

        return segments

    def _viterbi_decoding(self, segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
        """
        Viterbi algoritması ile optimal konuşmacı dizisini bul
        """
        if not segments or not diar_segments:
            return segments

        # Olası konuşmacılar
        speakers = list(set(d["speaker"] for d in diar_segments))

        # Viterbi tabloları
        n_segments = len(segments)
        viterbi_prob = [{} for _ in range(n_segments)]
        viterbi_path = [{} for _ in range(n_segments)]

        # İlk segment için başlangıç olasılıkları
        first_seg = segments[0]
        for speaker in speakers:
            emission_prob = self._get_emission_probability(first_seg, speaker, diar_segments)
            viterbi_prob[0][speaker] = emission_prob
            viterbi_path[0][speaker] = [speaker]

        # Geri kalan segmentler için
        for t in range(1, n_segments):
            curr_seg = segments[t]

            for curr_speaker in speakers:
                max_prob = 0.0
                best_prev_speaker = None

                for prev_speaker in speakers:
                    # Geçiş olasılığı
                    trans_prob = self.transition_model.get_transition_probability(
                        prev_speaker, curr_speaker
                    )

                    # Emisyon olasılığı
                    emission_prob = self._get_emission_probability(
                        curr_seg, curr_speaker, diar_segments
                    )

                    # Toplam olasılık
                    prob = viterbi_prob[t-1][prev_speaker] * trans_prob * emission_prob

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_speaker = prev_speaker

                viterbi_prob[t][curr_speaker] = max_prob
                if best_prev_speaker:
                    viterbi_path[t][curr_speaker] = viterbi_path[t-1][best_prev_speaker] + [curr_speaker]

        # En iyi yolu seç
        if viterbi_prob[-1]:
            best_last_speaker = max(viterbi_prob[-1].items(), key=lambda x: x[1])[0]
            best_path = viterbi_path[-1][best_last_speaker]

            # Segmentlere ata
            for seg, speaker in zip(segments, best_path):
                seg["speaker"] = speaker
                seg["sequential_confidence"] = 0.8  # Placeholder
                if "assignment_method" not in seg:
                    seg["assignment_method"] = "sequential"

        return segments

    def _get_emission_probability(self, segment: Dict, speaker: str, diar_segments: List[Dict]) -> float:
        """Emisyon olasılığını hesapla"""
        # Akustik benzerlik skoru varsa kullan
        if "acoustic_confidence" in segment and segment.get("speaker") == speaker:
            return segment["acoustic_confidence"]

        # Zamansal örtüşme bazlı skor
        temporal_scores = self._calculate_temporal_scores(segment, diar_segments)
        return temporal_scores.get(speaker, 0.1)  # Default düşük olasılık

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """Kısa segmentleri komşularıyla birleştir"""
        min_duration = 0.3  # 300ms altındaki segmentler
        merged = []

        for seg in segments:
            duration = float(seg.get("end", 0)) - float(seg.get("start", 0))

            if duration < min_duration and merged:
                # Önceki segment ile aynı konuşmacı mı?
                if merged[-1].get("speaker") == seg.get("speaker"):
                    # Birleştir
                    merged[-1]["end"] = seg.get("end", merged[-1]["end"])
                    if "text" in merged[-1] and "text" in seg:
                        merged[-1]["text"] += " " + seg["text"]
                else:
                    # Farklı konuşmacı ama çok kısa - öncekine ata
                    if duration < 0.1:  # 100ms altı
                        seg["speaker"] = merged[-1].get("speaker")
                        merged[-1]["end"] = seg.get("end", merged[-1]["end"])
                        if "text" in merged[-1] and "text" in seg:
                            merged[-1]["text"] += " " + seg["text"]
                    else:
                        merged.append(seg)
            else:
                merged.append(seg)

        return merged

    def _fix_implausible_transitions(self, segments: List[Dict]) -> List[Dict]:
        """Fiziksel olarak imkansız geçişleri düzelt"""
        min_gap = 0.05  # 50ms minimum konuşmacı değişim süresi

        for i in range(1, len(segments)):
            prev_seg = segments[i-1]
            curr_seg = segments[i]

            gap = float(curr_seg.get("start", 0)) - float(prev_seg.get("end", 0))

            # Çok hızlı konuşmacı değişimi
            if gap < min_gap and prev_seg.get("speaker") != curr_seg.get("speaker"):
                # Hangi segment daha güvenilir?
                prev_conf = prev_seg.get("acoustic_confidence", 0.5)
                curr_conf = curr_seg.get("acoustic_confidence", 0.5)

                if prev_conf > curr_conf:
                    curr_seg["speaker"] = prev_seg.get("speaker")
                else:
                    prev_seg["speaker"] = curr_seg.get("speaker")

        return segments

    def _phase3_multipass_refinement(self,
                                     segments: List[Dict],
                                     diar_segments: List[Dict],
                                     timeline_segs: Optional[List[Dict]],
                                     pass_num: int) -> List[Dict]:
        """
        Aşama 3: Çoklu geçişli iyileştirme
        """
        print(f"  Pass {pass_num + 1}/{self.n_passes}")

        # Güven skorlarına göre kategorize et
        high_confidence = []
        low_confidence = []

        for i, seg in enumerate(segments):
            confidence = seg.get("acoustic_confidence", 0.5)
            if confidence >= self.confidence_threshold:
                high_confidence.append(i)
            else:
                low_confidence.append(i)

        # Düşük güvenli segmentleri yeniden değerlendir
        for idx in low_confidence:
            seg = segments[idx]

            # Komşu segmentlerden bilgi kullan
            neighbors = self._get_neighbor_context(segments, idx, window=2)

            # Timeline bilgisini kullan
            if timeline_segs:
                timeline_speaker = self._get_timeline_speaker(seg, timeline_segs)
                if timeline_speaker:
                    seg["speaker"] = timeline_speaker
                    seg["assignment_method"] = "timeline_refined"
                    continue

            # Komşu ağırlıklı oylama
            speaker_votes = defaultdict(float)
            for neighbor_idx, neighbor in neighbors:
                distance = abs(neighbor_idx - idx)
                weight = 1.0 / (distance + 1)  # Mesafe bazlı ağırlık
                neighbor_speaker = neighbor.get("speaker")
                if neighbor_speaker:
                    confidence = neighbor.get("acoustic_confidence", 0.5)
                    speaker_votes[neighbor_speaker] += weight * confidence

            if speaker_votes:
                best_speaker = max(speaker_votes.items(), key=lambda x: x[1])[0]
                seg["speaker"] = best_speaker
                seg["assignment_method"] = f"multipass_{pass_num}"
                seg["acoustic_confidence"] = min(0.9, speaker_votes[best_speaker] / len(neighbors))

        # Constraint propagation
        segments = self._constraint_propagation(segments)

        return segments

    def _get_neighbor_context(self, segments: List[Dict], idx: int, window: int = 2) -> List[Tuple[int, Dict]]:
        """Komşu segmentleri getir"""
        neighbors = []

        for i in range(max(0, idx - window), min(len(segments), idx + window + 1)):
            if i != idx:
                neighbors.append((i, segments[i]))

        return neighbors

    def _get_timeline_speaker(self, segment: Dict, timeline_segs: List[Dict]) -> Optional[str]:
        """Timeline'dan konuşmacı bilgisi al"""
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))
        seg_mid = (seg_start + seg_end) / 2

        for tl_seg in timeline_segs:
            if tl_seg["start"] <= seg_mid <= tl_seg["end"]:
                if tl_seg["mode"] == "mono":
                    return tl_seg["speakers"][0]

        return None

    def _constraint_propagation(self, segments: List[Dict]) -> List[Dict]:
        """Constraint propagation ile tutarlılığı artır"""
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for i in range(1, len(segments) - 1):
                prev_seg = segments[i-1]
                curr_seg = segments[i]
                next_seg = segments[i+1]

                # Eğer önceki ve sonraki aynı konuşmacıya sahipse
                if (prev_seg.get("speaker") == next_seg.get("speaker") and
                        curr_seg.get("speaker") != prev_seg.get("speaker")):

                    # Kısa segment ise muhtemelen yanlış atanmıştır
                    duration = float(curr_seg.get("end", 0)) - float(curr_seg.get("start", 0))
                    if duration < 0.5:  # 500ms altı
                        curr_seg["speaker"] = prev_seg.get("speaker")
                        curr_seg["assignment_method"] = "constraint_propagation"
                        changed = True

        return segments

    def _assign_words_from_segments(self,
                                    words: List[Dict],
                                    segments: List[Dict],
                                    diar_segments: List[Dict]) -> List[Dict]:
        """Kelime seviyesi konuşmacı ataması"""
        # Segment ID bazlı eşleme
        seg_speaker_map = {s.get("id", i): s.get("speaker")
                           for i, s in enumerate(segments)}

        for word in words:
            seg_id = word.get("segment_id")

            if seg_id is not None and seg_id in seg_speaker_map:
                word["speaker"] = seg_speaker_map[seg_id]
            elif "start" in word and word["start"] is not None:
                # Kelime zamanına göre en uygun konuşmacıyı bul
                word_time = float(word["start"])

                # Önce segmentlerden bul
                for seg in segments:
                    if float(seg.get("start", 0)) <= word_time <= float(seg.get("end", 0)):
                        word["speaker"] = seg.get("speaker")
                        break

                # Bulunamazsa diarizasyondan bul
                if "speaker" not in word:
                    for diar_seg in diar_segments:
                        if diar_seg["start"] <= word_time <= diar_seg["end"]:
                            word["speaker"] = diar_seg["speaker"]
                            break

        return words

    def _final_consistency_check(self,
                                 segments: List[Dict],
                                 words: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Son tutarlılık kontrolü ve temporal smoothing"""
        # Temporal smoothing
        segments = self._temporal_smoothing(segments)

        # Segment-kelime tutarlılığını sağla
        for seg in segments:
            seg_id = seg.get("id", -1)
            seg_words = [w for w in words if w.get("segment_id") == seg_id]

            if seg_words and "speaker" in seg:
                # Tüm kelimeleri segment konuşmacısına ata
                for w in seg_words:
                    w["speaker"] = seg["speaker"]

        return segments, words

    def _temporal_smoothing(self, segments: List[Dict], window_size: float = 0.2) -> List[Dict]:
        """Temporal smoothing uygula"""
        smoothed = []

        i = 0
        while i < len(segments):
            curr_seg = segments[i]

            # Aynı konuşmacıya sahip ardışık segmentleri birleştir
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                gap = float(next_seg.get("start", 0)) - float(curr_seg.get("end", 0))

                if (curr_seg.get("speaker") == next_seg.get("speaker") and
                        gap < window_size):
                    # Birleştir
                    curr_seg["end"] = next_seg.get("end", curr_seg["end"])
                    if "text" in curr_seg and "text" in next_seg:
                        curr_seg["text"] += " " + next_seg["text"]
                    j += 1
                else:
                    break

            smoothed.append(curr_seg)
            i = j

        return smoothed

    def _basic_temporal_matching(self, segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
        """Fallback: Basit zamansal eşleme"""
        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))

            if seg_end <= seg_start:
                continue

            # En çok örtüşen diarizasyon segmentini bul
            best_speaker = None
            best_overlap = 0

            for diar_seg in diar_segments:
                overlap_start = max(seg_start, diar_seg["start"])
                overlap_end = min(seg_end, diar_seg["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg["speaker"]

            if best_speaker:
                seg["speaker"] = best_speaker
                seg["acoustic_confidence"] = best_overlap / (seg_end - seg_start)
                seg["assignment_method"] = "temporal"

        return segments