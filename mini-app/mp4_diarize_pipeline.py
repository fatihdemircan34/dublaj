from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import copy
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Existing timeline tools and optimized speaker mapper from previous code
from pathlib import Path
import os
import json
import csv
import math
import time
import shutil
import subprocess
import requests
from openai import OpenAI
try:
    from openai import BadRequestError
except Exception:
    class BadRequestError(Exception):
        pass

# ------------------ Timeline araçları ------------------

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

def parse_simple_csv_lines(lines: List[str]) -> List[Dict[str, Any]]:
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        start_s, end_s, spk = [p.strip() for p in ln.split(",", 2)]
        out.append({"start": float(start_s), "end": float(end_s), "speaker": spk})
    return out

# ------------------ OPTİMİZE EDİLMİŞ KONUŞMACI EŞLEME SİSTEMİ ------------------

@dataclass
class SegmentMetrics:
    """Segment eşleştirme metrikleri"""
    overlap_duration: float
    overlap_ratio_stt: float
    overlap_ratio_diar: float
    iou: float
    boundary_distance: float
    confidence: float

@dataclass
class SpeakerCandidate:
    """Konuşmacı adayı"""
    speaker_id: str
    metrics: SegmentMetrics
    source: str
    weight: float = 1.0

class OptimizedSpeakerMapper:
    """
    Gerçek veri odaklı optimize edilmiş konuşmacı eşleme sistemi.
    """

    def __init__(self,
                 min_overlap_ratio: float = 0.2,
                 boundary_tolerance: float = 0.1,
                 min_segment_duration: float = 0.2,
                 use_vad_boundaries: bool = True,
                 use_timeline: bool = True,
                 neighbor_window: int = 3,
                 confidence_threshold: float = 0.6):
        self.min_overlap_ratio = min_overlap_ratio
        self.boundary_tolerance = boundary_tolerance
        self.min_segment_duration = min_segment_duration
        self.use_vad_boundaries = use_vad_boundaries
        self.use_timeline = use_timeline
        self.neighbor_window = neighbor_window
        self.confidence_threshold = confidence_threshold
        self.assignment_stats = defaultdict(int)

    def map_speakers(self,
                     segments: List[Dict[str, Any]],
                     words: List[Dict[str, Any]],
                     diarization: List[Dict[str, Any]],
                     timeline: Optional[List[Dict[str, Any]]] = None,
                     vad_regions: Optional[List[Dict[str, float]]] = None) -> Tuple[List[Dict], List[Dict]]:

        segments = copy.deepcopy(segments)
        words = copy.deepcopy(words)

        diar_segments = self._normalize_diarization(diarization)
        timeline_segs = self._normalize_timeline(timeline) if timeline else None
        vad_regions = self._normalize_vad(vad_regions) if vad_regions else None

        if self.use_vad_boundaries and vad_regions:
            segments = self._adjust_boundaries_with_vad(segments, vad_regions)

        for idx, seg in enumerate(segments):
            candidates = self._get_speaker_candidates(
                seg, idx, segments, diar_segments, timeline_segs
            )

            if candidates:
                best_candidate = self._select_best_candidate(candidates, seg, segments, idx)
                if best_candidate and best_candidate.metrics.confidence >= self.confidence_threshold:
                    seg["speaker"] = best_candidate.speaker_id
                    seg["confidence"] = best_candidate.metrics.confidence
                    seg["assignment_source"] = best_candidate.source
                    self.assignment_stats[best_candidate.source] += 1
                else:
                    if candidates:
                        seg["speaker"] = candidates[0].speaker_id
                        seg["confidence"] = candidates[0].metrics.confidence
                        seg["assignment_source"] = "low_confidence"
                        self.assignment_stats["low_confidence"] += 1

        segments = self._merge_short_segments(segments)
        segments = self._fix_isolated_segments(segments)
        words = self._assign_words_optimized(words, segments, diar_segments)
        segments, words = self._final_consistency_pass(segments, words)

        return segments, words

    def _normalize_diarization(self, diar: List[Dict]) -> List[Dict]:
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

    def _normalize_vad(self, vad_regions: List[Dict[str, float]]) -> List[Dict[str, float]]:
        normalized = []
        for v in vad_regions:
            normalized.append({
                "start": float(v["start"]),
                "end": float(v["end"]),
                "duration": float(v["end"]) - float(v["start"])
            })
        return sorted(normalized, key=lambda x: x["start"])

    def _calculate_segment_metrics(self, seg1_start: float, seg1_end: float,
                                   seg2_start: float, seg2_end: float) -> SegmentMetrics:
        overlap_start = max(seg1_start, seg2_start)
        overlap_end = min(seg1_end, seg2_end)
        overlap_duration = max(0, overlap_end - overlap_start)

        seg1_duration = seg1_end - seg1_start
        seg2_duration = seg2_end - seg2_start

        overlap_ratio_stt = overlap_duration / max(seg1_duration, 0.001)
        overlap_ratio_diar = overlap_duration / max(seg2_duration, 0.001)

        union = seg1_duration + seg2_duration - overlap_duration
        iou = overlap_duration / max(union, 0.001)

        start_distance = abs(seg1_start - seg2_start)
        end_distance = abs(seg1_end - seg2_end)
        boundary_distance = (start_distance + end_distance) / 2

        confidence = self._calculate_confidence(
            overlap_ratio_stt, overlap_ratio_diar, iou, boundary_distance
        )

        return SegmentMetrics(
            overlap_duration=overlap_duration,
            overlap_ratio_stt=overlap_ratio_stt,
            overlap_ratio_diar=overlap_ratio_diar,
            iou=iou,
            boundary_distance=boundary_distance,
            confidence=confidence
        )

    def _calculate_confidence(self, overlap_stt: float, overlap_diar: float,
                              iou: float, boundary_dist: float) -> float:
        boundary_score = max(0, 1 - (boundary_dist / 2.0))

        weights = {
            'overlap_stt': 0.35,
            'overlap_diar': 0.25,
            'iou': 0.25,
            'boundary': 0.15
        }

        confidence = (
                weights['overlap_stt'] * overlap_stt +
                weights['overlap_diar'] * overlap_diar +
                weights['iou'] * iou +
                weights['boundary'] * boundary_score
        )

        return min(1.0, max(0.0, confidence))

    def _adjust_boundaries_with_vad(self, segments: List[Dict],
                                    vad_regions: List[Dict]) -> List[Dict]:
        adjusted = []

        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))

            best_vad = None
            best_overlap = 0

            for vad in vad_regions:
                overlap = min(seg_end, vad["end"]) - max(seg_start, vad["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_vad = vad

            if best_vad:
                tolerance = self.boundary_tolerance

                if abs(seg_start - best_vad["start"]) < tolerance:
                    seg["start"] = best_vad["start"]
                if abs(seg_end - best_vad["end"]) < tolerance:
                    seg["end"] = best_vad["end"]

            adjusted.append(seg)

        return adjusted

    def _get_speaker_candidates(self, segment: Dict, seg_idx: int,
                                all_segments: List[Dict],
                                diar_segments: List[Dict],
                                timeline_segs: Optional[List[Dict]]) -> List[SpeakerCandidate]:
        candidates = []
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))

        if seg_end <= seg_start:
            return candidates

        speaker_scores = defaultdict(lambda: {'metrics': None, 'weight': 0})

        for diar_seg in diar_segments:
            metrics = self._calculate_segment_metrics(
                seg_start, seg_end,
                diar_seg["start"], diar_seg["end"]
            )

            if metrics.overlap_ratio_stt >= self.min_overlap_ratio:
                speaker = diar_seg["speaker"]

                if speaker_scores[speaker]['metrics'] is None or \
                        metrics.confidence > speaker_scores[speaker]['metrics'].confidence:
                    speaker_scores[speaker]['metrics'] = metrics
                    speaker_scores[speaker]['weight'] = 1.0

        for speaker, data in speaker_scores.items():
            if data['metrics']:
                candidates.append(SpeakerCandidate(
                    speaker_id=speaker,
                    metrics=data['metrics'],
                    source="diarization",
                    weight=data['weight']
                ))

        if self.use_timeline and timeline_segs:
            timeline_candidate = self._get_timeline_candidate(segment, timeline_segs)
            if timeline_candidate:
                candidates.append(timeline_candidate)

        neighbor_candidates = self._get_neighbor_candidates(
            segment, seg_idx, all_segments, diar_segments
        )
        candidates.extend(neighbor_candidates)

        candidates.sort(key=lambda c: c.metrics.confidence * c.weight, reverse=True)

        return candidates

    def _get_timeline_candidate(self, segment: Dict,
                                timeline_segs: List[Dict]) -> Optional[SpeakerCandidate]:
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))
        seg_mid = (seg_start + seg_end) / 2

        for tl_seg in timeline_segs:
            if tl_seg["start"] <= seg_mid <= tl_seg["end"]:
                if tl_seg["mode"] == "mono":
                    metrics = self._calculate_segment_metrics(
                        seg_start, seg_end,
                        tl_seg["start"], tl_seg["end"]
                    )
                    metrics.confidence = min(1.0, metrics.confidence * 1.2)

                    return SpeakerCandidate(
                        speaker_id=tl_seg["speakers"][0],
                        metrics=metrics,
                        source="timeline",
                        weight=1.2
                    )

        return None

    def _get_neighbor_candidates(self, segment: Dict, seg_idx: int,
                                 all_segments: List[Dict],
                                 diar_segments: List[Dict]) -> List[SpeakerCandidate]:
        candidates = []

        for offset in range(1, self.neighbor_window + 1):
            for idx in [seg_idx - offset, seg_idx + offset]:
                if 0 <= idx < len(all_segments) and idx != seg_idx:
                    neighbor = all_segments[idx]

                    if "speaker" in neighbor and neighbor.get("confidence", 0) > 0.7:
                        distance_weight = 1.0 / (abs(idx - seg_idx) + 1)

                        metrics = SegmentMetrics(
                            overlap_duration=0,
                            overlap_ratio_stt=0,
                            overlap_ratio_diar=0,
                            iou=0,
                            boundary_distance=abs(idx - seg_idx),
                            confidence=neighbor.get("confidence", 0.5) * distance_weight * 0.6
                        )

                        candidates.append(SpeakerCandidate(
                            speaker_id=neighbor["speaker"],
                            metrics=metrics,
                            source="neighbor",
                            weight=distance_weight * 0.5
                        ))

        return candidates

    def _select_best_candidate(self, candidates: List[SpeakerCandidate],
                               segment: Dict, all_segments: List[Dict],
                               seg_idx: int) -> Optional[SpeakerCandidate]:
        if not candidates:
            return None

        best = candidates[0]

        high_confidence_candidates = [c for c in candidates
                                      if c.metrics.confidence > self.confidence_threshold]

        if len(high_confidence_candidates) > 1:
            source_priority = {"timeline": 3, "diarization": 2, "neighbor": 1}

            high_confidence_candidates.sort(
                key=lambda c: (
                    c.metrics.confidence * c.weight,
                    source_priority.get(c.source, 0)
                ),
                reverse=True
            )

            best = high_confidence_candidates[0]

        return best

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]
            duration = float(current.get("end", 0)) - float(current.get("start", 0))

            if duration < self.min_segment_duration and i < len(segments) - 1:
                next_seg = segments[i + 1]
                gap = float(next_seg.get("start", 0)) - float(current.get("end", 0))

                if (current.get("speaker") == next_seg.get("speaker") and
                        gap < 0.3):
                    current["end"] = next_seg.get("end", current["end"])
                    if "text" in current and "text" in next_seg:
                        current["text"] = current["text"] + " " + next_seg["text"]
                    if "confidence" in current and "confidence" in next_seg:
                        current["confidence"] = max(current["confidence"],
                                                    next_seg["confidence"])
                    merged.append(current)
                    i += 2
                    continue

            merged.append(current)
            i += 1

        return merged

    def _fix_isolated_segments(self, segments: List[Dict]) -> List[Dict]:
        if len(segments) < 3:
            return segments

        fixed = segments.copy()

        for i in range(1, len(segments) - 1):
            prev_speaker = segments[i-1].get("speaker")
            curr_speaker = segments[i].get("speaker")
            next_speaker = segments[i+1].get("speaker")

            if (prev_speaker and next_speaker and
                    prev_speaker == next_speaker and
                    curr_speaker != prev_speaker):

                duration = float(segments[i].get("end", 0)) - float(segments[i].get("start", 0))
                curr_confidence = segments[i].get("confidence", 0.5)

                if duration < 0.5 and curr_confidence < 0.7:
                    fixed[i]["speaker"] = prev_speaker
                    fixed[i]["confidence"] = min(
                        segments[i-1].get("confidence", 0.5),
                        segments[i+1].get("confidence", 0.5)
                    ) * 0.8
                    fixed[i]["assignment_source"] = "isolation_fix"
                    self.assignment_stats["isolation_fix"] += 1

        return fixed

    def _assign_words_optimized(self, words: List[Dict], segments: List[Dict],
                                diar_segments: List[Dict]) -> List[Dict]:
        seg_speaker_map = {}
        for seg in segments:
            seg_id = seg.get("id")
            if seg_id is not None and "speaker" in seg:
                seg_speaker_map[seg_id] = seg["speaker"]

        for word in words:
            seg_id = word.get("segment_id")
            if seg_id in seg_speaker_map:
                word["speaker"] = seg_speaker_map[seg_id]
                continue

            if "start" in word and word["start"] is not None:
                word_time = float(word["start"])

                for seg in segments:
                    if float(seg.get("start", 0)) <= word_time <= float(seg.get("end", 0)):
                        if "speaker" in seg:
                            word["speaker"] = seg["speaker"]
                            break

                if "speaker" not in word:
                    for diar_seg in diar_segments:
                        if diar_seg["start"] <= word_time <= diar_seg["end"]:
                            word["speaker"] = diar_seg["speaker"]
                            break

        return words

    def _final_consistency_pass(self, segments: List[Dict],
                                words: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        for seg in segments:
            seg_id = seg.get("id")
            if seg_id is None:
                continue

            seg_words = [w for w in words if w.get("segment_id") == seg_id]

            if not seg_words:
                continue

            word_speakers = [w.get("speaker") for w in seg_words if "speaker" in w]

            if word_speakers:
                speaker_counts = Counter(word_speakers)
                most_common_speaker, count = speaker_counts.most_common(1)[0]

                if count / len(word_speakers) > 0.7:
                    if seg.get("speaker") != most_common_speaker:
                        seg["speaker"] = most_common_speaker
                        seg["assignment_source"] = "word_majority"
                        self.assignment_stats["word_majority"] += 1

                    for w in seg_words:
                        w["speaker"] = most_common_speaker

        return segments, words

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "assignment_sources": dict(self.assignment_stats),
            "total_assignments": sum(self.assignment_stats.values())
        }

# ------------------ Pipeline Utilities ------------------

MAX_BODY_BYTES = 26_214_400
SOFT_LIMIT_BYTES = 24 * 1024 * 1024

class DebugWriter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: List[Dict[str, Any]] = []
    def snap(self, tag: str, **data: Any):
        if not self.enabled:
            return
        evt = {"tag": tag, "ts": time.time(), **data}
        self.events.append(evt)
        kv = " ".join(f"{k}={v}" for k, v in data.items())
        print(f"[{tag}] {kv}")

def _require_ffmpeg():
    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            raise RuntimeError(f"{bin_name} bulunamadı. FFmpeg/FFprobe kurulu olmalı.")

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def probe_duration_seconds(path: Path) -> float:
    _require_ffmpeg()
    cp = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path)
    ])
    try:
        return float(cp.stdout.decode().strip())
    except Exception:
        return 0.0

def transcode_audio_under_limit(
        src_video: Path,
        workdir: Path,
        try_bitrates_kbps: Tuple[int, ...] = (64, 48, 32, 24, 16),
        ar: int = 16000,
        ac: int = 1,
        dbg: Optional[DebugWriter] = None,
) -> Path:
    _require_ffmpeg()
    workdir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration_seconds(src_video)
    if dbg: dbg.snap("AUDIO_PROBE", duration_sec=round(dur, 2))
    for br in try_bitrates_kbps:
        out = workdir / f"{src_video.stem}.mono{ac}_{ar//1000}kHz_{br}kbps.m4a"
        cmd = ["ffmpeg","-y","-i",str(src_video),"-vn","-ac",str(ac),"-ar",str(ar),"-c:a","aac","-b:a",f"{br}k","-movflags","+faststart",str(out)]
        _run(cmd)
        size = out.stat().st_size
        if dbg: dbg.snap("AUDIO_ENCODED", bitrate_kbps=br, size_bytes=size)
        if size <= SOFT_LIMIT_BYTES:
            return out
    return out

def split_audio_by_duration(
        audio_path: Path,
        chunk_sec: int,
        outdir: Path,
        copy_codecs: bool = True,
        dbg: Optional[DebugWriter] = None,
) -> List[Tuple[Path, float]]:
    _require_ffmpeg()
    outdir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration_seconds(audio_path)
    if duration == 0:
        raise RuntimeError("Süre okunamadı; ffprobe başarısız.")
    chunks: List[Tuple[Path, float]] = []
    n = math.ceil(duration / chunk_sec)
    for i in range(n):
        start = i * chunk_sec
        length = min(chunk_sec, duration - start)
        out = outdir / f"chunk_{i:04d}.m4a"
        cmd = ["ffmpeg","-y","-ss",f"{start:.3f}","-i",str(audio_path),"-t",f"{length:.3f}"]
        if copy_codecs:
            cmd += ["-c","copy","-movflags","+faststart"]
        else:
            cmd += ["-ac","1","-ar","16000","-c:a","aac","-b:a","48k","-movflags","+faststart"]
        cmd += [str(out)]
        _run(cmd)
        chunks.append((out, start))
    if dbg: dbg.snap("AUDIO_SPLIT", n_chunks=len(chunks), approx_chunk_sec=chunk_sec)
    return chunks

def ensure_wav_mono16k(audio_in: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    wav = outdir / f"{audio_in.stem}.mono16k.wav"
    _run(["ffmpeg","-y","-i",str(audio_in),"-ac","1","-ar","16000","-c:a","pcm_s16le",str(wav)])
    return wav

def _sdk_transcribe_verbose(client: OpenAI, path: Path, model: str, language: Optional[str], prompt: Optional[str], want_word_timestamps: bool) -> dict:
    extra = {}
    if model == "whisper-1" and want_word_timestamps:
        extra["timestamp_granularities"] = ["word","segment"]
    with path.open("rb") as f:
        tr = client.audio.transcriptions.create(
            model=model, file=f, response_format="verbose_json", language=language, prompt=prompt, **extra
        )
    for attr in ("model_dump","to_dict"):
        if hasattr(tr, attr):
            return getattr(tr, attr)()
    js = getattr(tr,"model_dump_json",None)
    return json.loads(js()) if callable(js) else json.loads(str(tr))

def _requests_transcribe_verbose(path: Path, model: str, language: Optional[str], prompt: Optional[str], want_word_timestamps: bool) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    data = [("model",model),("response_format","verbose_json")]
    if language: data.append(("language",language))
    if prompt:   data.append(("prompt",prompt))
    if want_word_timestamps and model == "whisper-1":
        data.append(("timestamp_granularities[]","word"))
        data.append(("timestamp_granularities[]","segment"))
    with path.open("rb") as f:
        files = {"file": (path.name, f, "application/octet-stream")}
        r = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data, files=files, timeout=(15,600))
    r.raise_for_status()
    return r.json()

def transcribe_file(path: Path, model: str="whisper-1", language: Optional[str]=None, prompt: Optional[str]=None, want_word_timestamps: bool=True, dbg: Optional[DebugWriter]=None) -> dict:
    client = OpenAI()
    try:
        if dbg: dbg.snap("TRANSCRIBE_SDK", file=str(path))
        return _sdk_transcribe_verbose(client, path, model, language, prompt, want_word_timestamps)
    except BadRequestError as e:
        msg = str(e).lower()
        if "something went wrong" in msg or "invalid_request_error" in msg:
            if dbg: dbg.snap("TRANSCRIBE_FALLBACK", reason="badrequest")
            return _requests_transcribe_verbose(path, model, language, prompt, want_word_timestamps)
        raise

def _vad_pyannote(wav_path: Path, dbg: DebugWriter) -> List[Dict[str, float]]:
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN tanımlı değil.")
    pipe = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=token)
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
            dbg.snap("VAD_INIT", device="cuda")
        else:
            dbg.snap("VAD_INIT", device="cpu")
    except Exception:
        dbg.snap("VAD_INIT", device="cpu-no-torch")

    t0 = time.perf_counter()
    vad_annot = pipe(str(wav_path))
    t1 = time.perf_counter()
    speech_tl = vad_annot.get_timeline().support()
    regions: List[Dict[str, float]] = []
    for seg in speech_tl:
        regions.append({"start": float(seg.start), "end": float(seg.end)})
    total = sum(r["end"] - r["start"] for r in regions)
    dbg.snap("VAD_DONE", regions=len(regions), total_speech_sec=round(total,2), secs=round(t1-t0,3))
    return regions

def _diarize_pyannote(wav_path: Path, speaker_count: Optional[int], dbg: DebugWriter) -> List[Dict[str, Any]]:
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN tanımlı değil.")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
            dbg.snap("DIAR_INIT", device="cuda")
        else:
            dbg.snap("DIAR_INIT", device="cpu")
    except Exception:
        dbg.snap("DIAR_INIT", device="cpu-no-torch")

    t0 = time.perf_counter()
    diar = pipe(str(wav_path)) if speaker_count is None else pipe(str(wav_path), num_speakers=max(1,int(speaker_count)))
    t1 = time.perf_counter()

    diar_segments: List[Dict[str, Any]] = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        diar_segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})
    diar_segments.sort(key=lambda d: d["start"])
    total_talk = sum(ds["end"] - ds["start"] for ds in diar_segments)
    dbg.snap("DIAR_DONE", diar_segments=len(diar_segments), est_speakers=len({d['speaker'] for d in diar_segments}),
             total_speaking_time_sec=round(total_talk,2), secs=round(t1-t0,3))
    return diar_segments

def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1,b1) - max(a0,b0))

def clip_segments_to_regions(segments: List[Dict[str, Any]], regions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    if not regions:
        return segments
    out: List[Dict[str, Any]] = []
    for s in segments:
        s0, s1 = float(s["start"]), float(s["end"])
        if s1 <= s0:
            continue
        for r in regions:
            r0, r1 = float(r["start"]), float(r["end"])
            ov = _overlap(s0, s1, r0, r1)
            if ov <= 0:
                continue
            out.append({"start": max(s0,r0), "end": min(s1,r1), "speaker": s["speaker"]})
    out.sort(key=lambda d: (d["start"], d["end"], d["speaker"]))
    merged: List[Dict[str, Any]] = []
    for seg in out:
        if merged and merged[-1]["speaker"] == seg["speaker"] and abs(merged[-1]["end"] - seg["start"]) < 1e-6:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged

def _normalize_verbose(raw: dict) -> Tuple[List[dict], List[dict], str, float, Optional[str]]:
    segments = list(raw.get("segments") or [])
    words: List[dict] = []
    if isinstance(raw.get("words"), list):
        words = raw["words"]
    else:
        for i, s in enumerate(segments):
            for w in (s.get("words") or []):
                ww = dict(w)
                if "segment_id" not in ww:
                    ww["segment_id"] = s.get("id", i)
                words.append(ww)

    text = raw.get("text")
    if not text:
        text = " ".join((s.get("text") or "").strip() for s in segments).strip()

    duration = raw.get("duration")
    if duration is None:
        if segments:
            try:
                duration = float(segments[-1].get("end") or 0.0)
            except Exception:
                duration = 0.0
        else:
            duration = 0.0
    else:
        try:
            duration = float(duration)
        except Exception:
            duration = 0.0

    language = raw.get("language")

    if not segments and text:
        segments = [{"id": 0, "start": 0.0, "end": duration or 0.0, "text": text}]

    return segments, words, text or "", float(duration), language

def _write_rttm(diar_segments: List[dict], rttm_path: Path, uri: Optional[str] = None):
    if uri is None:
        uri = rttm_path.stem
    with rttm_path.open("w") as f:
        for d in diar_segments:
            start = float(d["start"]); dur = float(d["end"]) - float(d["start"])
            spk = str(d["speaker"])
            line = f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk}\n"
            f.write(line)

def _write_outputs(outdir: Path, stem: str, segments: List[dict], words: List[dict], text: str, language: Optional[str], duration: float | None, diarization: Optional[List[dict]]=None, timeline: Optional[List[TimelineSeg]]=None) -> dict:
    verbose_json_path = outdir / f"{stem}.verbose.json"
    verbose = {"text": text, "language": language, "duration": duration, "segments": segments, "words": words}
    if diarization is not None:
        verbose["diarization"] = diarization
    if timeline is not None:
        verbose["timeline"] = [
            {"start": t.start, "end": t.end, "mode": t.mode, "speakers": list(t.speakers),
             "channels": t.channels if t.channels else None}
            for t in timeline
        ]
    verbose_json_path.write_text(json.dumps(verbose, ensure_ascii=False, indent=2), encoding="utf-8")

    def srt_time(t: float | None) -> str:
        t = 0.0 if t is None else max(0.0, float(t))
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        srt_lines += [str(i), f"{srt_time(seg.get('start'))} --> {srt_time(seg.get('end'))}", (seg.get("text") or "").strip(), ""]
    (outdir / f"{stem}.segments.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    with (outdir / f"{stem}.segments.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id","start","end","text","speaker"])
        for i, s in enumerate(segments):
            w.writerow([s.get("id", i), s.get("start"), s.get("end"), (s.get("text") or "").strip(), s.get("speaker")])
    with (outdir / f"{stem}.words.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["start","end","word","segment_id","speaker"])
        for kw in words:
            w.writerow([kw.get("start"), kw.get("end"), kw.get("word") or kw.get("text"), kw.get("segment_id"), kw.get("speaker")])

    if timeline is not None:
        tcsv = outdir / f"{stem}.timeline.csv"
        with tcsv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["start","end","mode","speakers","L","R"])
            for t in timeline:
                spks = "|".join(t.speakers)
                L = t.channels.get("L") if t.channels else ""
                R = t.channels.get("R") if t.channels else ""
                w.writerow([t.start, t.end, t.mode, spks, L, R])

    files = {
        "verbose_json": str(verbose_json_path),
        "segments_srt": str(outdir / f"{stem}.segments.srt"),
        "segments_csv": str(outdir / f"{stem}.segments.csv"),
        "words_csv": str(outdir / f"{stem}.words.csv"),
    }
    if timeline is not None:
        files["timeline_csv"] = str(outdir / f"{stem}.timeline.csv")
    return {"text": text, "language": language, "duration": duration, "files": files}

# ------------------ Main Pipeline with Optimized Speaker Mapping ------------------

def process_video_wordwise(
        video_path: str,
        output_dir: str,
        llm_model: str = "gpt-4o",
        stt_model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        want_word_timestamps: bool = True,
        chunk_overlap_sec: float = 0.0,
        diarize: bool = True,
        speaker_count: Optional[int] = None,
        use_vad: bool = True,
        stereo_threshold: float = 0.20,
        debug: bool = True,
        # Optimized mapper parameters
        use_optimized_mapping: bool = True,
        min_overlap_ratio: float = 0.2,
        boundary_tolerance: float = 0.1,
        use_vad_boundaries: bool = True,
        use_timeline: bool = True,
        confidence_threshold: float = 0.6,
) -> dict:
    dbg = DebugWriter(enabled=debug)
    src = Path(video_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {src}")

    # Audio processing
    audio = transcode_audio_under_limit(src, out / "_work", dbg=dbg)
    size = audio.stat().st_size
    if size > SOFT_LIMIT_BYTES:
        dur = probe_duration_seconds(audio)
        avg_kbps = max(1, int((size * 8) / max(1.0, dur) / 1000))
        safe_sec = int((SOFT_LIMIT_BYTES * 8) / (avg_kbps * 1000) * 0.95)
        safe_sec = max(30, min(900, safe_sec))
        chunks = split_audio_by_duration(audio, safe_sec, out / "_chunks", copy_codecs=True, dbg=dbg)
    else:
        chunks = [(audio, 0.0)]
        dbg.snap("AUDIO_SINGLE", size_bytes=size)

    # Transcription
    merged_segments: List[dict] = []
    merged_words: List[dict] = []
    merged_texts: List[str] = []
    lang: Optional[str] = None
    last_id = 0
    last_seg_end = 0.0

    for idx, (part_path, base_offset) in enumerate(chunks):
        effective_offset = base_offset - (chunk_overlap_sec if idx > 0 else 0.0)
        raw = transcribe_file(part_path, model=stt_model, language=language, prompt=prompt, want_word_timestamps=want_word_timestamps, dbg=dbg)
        segs, words, text, part_dur, part_lang = _normalize_verbose(raw)
        if lang is None:
            lang = part_lang
        for s in segs:
            s = dict(s); s["start"] = float(s.get("start",0.0)) + effective_offset; s["end"] = float(s.get("end",0.0)) + effective_offset
            s["id"] = last_id; last_id += 1; merged_segments.append(s); last_seg_end = max(last_seg_end, s["end"])
        for w in words:
            w = dict(w)
            if "start" in w and w["start"] is not None: w["start"] = float(w["start"]) + effective_offset
            if "end"   in w and w["end"]   is not None: w["end"]   = float(w["end"])   + effective_offset
            merged_words.append(w)
        merged_texts.append(text)

    diar_segments: Optional[List[dict]] = None
    vad_regions: Optional[List[Dict[str,float]]] = None

    # VAD and Diarization
    wav_for_diar = ensure_wav_mono16k(audio, out / "_work")
    if use_vad:
        vad_regions = _vad_pyannote(wav_for_diar, dbg)

    if diarize:
        diar_segments = _diarize_pyannote(wav_for_diar, speaker_count, dbg)
        if vad_regions:
            diar_segments = clip_segments_to_regions(diar_segments, vad_regions)

        timeline = build_flat_timeline(diar_segments, stereo_threshold=stereo_threshold)

        # SPEAKER MAPPING
        if use_optimized_mapping:
            dbg.snap("SPEAKER_MAPPING", method="optimized_real_data")

            timeline_json = None
            if timeline:
                timeline_json = [
                    {
                        "start": t.start,
                        "end": t.end,
                        "mode": t.mode,
                        "speakers": list(t.speakers),
                        "channels": t.channels if t.channels else {}
                    }
                    for t in timeline
                ]

            mapper = OptimizedSpeakerMapper(
                min_overlap_ratio=min_overlap_ratio,
                boundary_tolerance=boundary_tolerance,
                use_vad_boundaries=use_vad_boundaries,
                use_timeline=use_timeline,
                confidence_threshold=confidence_threshold
            )

            merged_segments, merged_words = mapper.map_speakers(
                merged_segments,
                merged_words,
                diar_segments,
                timeline_json,
                vad_regions
            )

            stats = mapper.get_statistics()
            dbg.snap("MAPPING_STATS", **stats)

            assigned_segments = sum(1 for s in merged_segments if "speaker" in s)
            high_conf_segments = sum(
                1 for s in merged_segments
                if s.get("confidence", 0) >= confidence_threshold
            )

            dbg.snap("MAPPING_RESULTS",
                     total_segments=len(merged_segments),
                     assigned_segments=assigned_segments,
                     high_confidence_segments=high_conf_segments,
                     assigned_words=sum(1 for w in merged_words if "speaker" in w))
        else:
            # Simple fallback
            dbg.snap("SPEAKER_MAPPING", method="simple_temporal")
            for s in merged_segments:
                s0, s1 = float(s.get("start",0.0)), float(s.get("end",0.0))
                best_spk, best_ov = None, 0.0
                for d in diar_segments:
                    ov = _overlap(s0,s1,d["start"],d["end"])
                    if ov > best_ov:
                        best_ov, best_spk = ov, d["speaker"]
                if best_spk is not None:
                    s["speaker"] = best_spk

        # Write RTTM and speaker CSV
        rttm_path = out / f"{wav_for_diar.stem}.diarization.rttm"
        _write_rttm(diar_segments, rttm_path, uri=src.stem)
        spk_csv = out / f"{src.stem}.speakers.csv"
        with spk_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["start","end","speaker"])
            for d in diar_segments:
                w.writerow([d["start"], d["end"], d["speaker"]])
        dbg.snap("DIAR_FILES", rttm=str(rttm_path), speakers_csv=str(spk_csv))
    else:
        timeline = None

    # Write outputs
    summary = _write_outputs(out, src.stem, merged_segments, merged_words, " ".join(merged_texts).strip(), lang, last_seg_end, diarization=diar_segments, timeline=timeline)
    summary["models"] = {"stt": stt_model, "llm": llm_model}
    if diar_segments:
        summary["files"]["speakers_csv"] = str(out / f"{src.stem}.speakers.csv")
        rttm_guess = out / f"{audio.stem}.diarization.rttm"
        if rttm_guess.exists():
            summary["files"]["speakers_rttm"] = str(rttm_guess)
    if timeline is not None:
        summary["files"]["timeline_csv"] = str(out / f"{src.stem}.timeline.csv")

    if debug and dbg.events:
        debug_json = out / f"{src.stem}.debug.json"
        debug_json.write_text(json.dumps(dbg.events, indent=2), encoding="utf-8")
        summary["files"]["debug"] = str(debug_json)

    return summary

# Usage example
if __name__ == "__main__":
    res = process_video_wordwise(
        video_path="sample2.mp4",
        output_dir="output2",
        stt_model="whisper-1",
        language=None,
        prompt=None,
        want_word_timestamps=True,
        chunk_overlap_sec=0.0,
        diarize=True,
        speaker_count=None,
        use_vad=True,
        stereo_threshold=0.20,
        debug=True,
        # Optimized mapping parameters
        use_optimized_mapping=True,
        min_overlap_ratio=0.2,
        boundary_tolerance=0.1,
        use_vad_boundaries=True,
        use_timeline=True,
        confidence_threshold=0.6,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))