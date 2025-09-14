from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Iterable
import numpy as np
import copy
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ===== Std & IO =====
from pathlib import Path
import os
import sys
import json
import csv
import math
import time
import shutil
import subprocess
import requests

# ===== 3rd-party =====
try:
    from openai import OpenAI
    try:
        from openai import BadRequestError
    except Exception:
        class BadRequestError(Exception):
            pass
except Exception:
    OpenAI = None  # type: ignore
    class BadRequestError(Exception):
        pass

# pydub (timeline ses birleştirme ve referans ses için)
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

# ====================== Timeline / Diar Utils (mevcut kodun) ======================

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

# ================== OPTİMİZE EDİLMİŞ KONUŞMACI EŞLEME (mevcut) ===================

@dataclass
class SegmentMetrics:
    overlap_duration: float
    overlap_ratio_stt: float
    overlap_ratio_diar: float
    iou: float
    boundary_distance: float
    confidence: float

@dataclass
class SpeakerCandidate:
    speaker_id: str
    metrics: SegmentMetrics
    source: str
    weight: float = 1.0

class OptimizedSpeakerMapper:
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

    def map_speakers(self, segments, words, diarization, timeline=None, vad_regions=None):
        segments = copy.deepcopy(segments)
        words = copy.deepcopy(words)
        diar_segments = self._normalize_diarization(diarization)
        timeline_segs = self._normalize_timeline(timeline) if timeline else None
        vad_regions = self._normalize_vad(vad_regions) if vad_regions else None
        if self.use_vad_boundaries and vad_regions:
            segments = self._adjust_boundaries_with_vad(segments, vad_regions)
        for idx, seg in enumerate(segments):
            candidates = self._get_speaker_candidates(seg, idx, segments, diar_segments, timeline_segs)
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

    # --- internal helpers (kısaltmadan korunuyor) ---
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

    def _calculate_segment_metrics(self, seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float) -> SegmentMetrics:
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
        confidence = self._calculate_confidence(overlap_ratio_stt, overlap_ratio_diar, iou, boundary_distance)
        return SegmentMetrics(overlap_duration, overlap_ratio_stt, overlap_ratio_diar, iou, boundary_distance, confidence)

    def _calculate_confidence(self, overlap_stt: float, overlap_diar: float, iou: float, boundary_dist: float) -> float:
        boundary_score = max(0, 1 - (boundary_dist / 2.0))
        weights = {'overlap_stt': 0.35, 'overlap_diar': 0.25, 'iou': 0.25, 'boundary': 0.15}
        confidence = (weights['overlap_stt'] * overlap_stt +
                      weights['overlap_diar'] * overlap_diar +
                      weights['iou'] * iou +
                      weights['boundary'] * boundary_score)
        return min(1.0, max(0.0, confidence))

    def _adjust_boundaries_with_vad(self, segments: List[Dict], vad_regions: List[Dict]) -> List[Dict]:
        adjusted = []
        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            best_vad = None; best_overlap = 0
            for vad in vad_regions:
                overlap = min(seg_end, vad["end"]) - max(seg_start, vad["start"])
                if overlap > best_overlap:
                    best_overlap = overlap; best_vad = vad
            if best_vad:
                tolerance = 0.1
                if abs(seg_start - best_vad["start"]) < tolerance:
                    seg["start"] = best_vad["start"]
                if abs(seg_end - best_vad["end"]) < tolerance:
                    seg["end"] = best_vad["end"]
            adjusted.append(seg)
        return adjusted

    def _get_speaker_candidates(self, segment: Dict, seg_idx: int, all_segments: List[Dict], diar_segments: List[Dict], timeline_segs: Optional[List[Dict]]) -> List[SpeakerCandidate]:
        candidates = []
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))
        if seg_end <= seg_start:
            return candidates
        speaker_scores = defaultdict(lambda: {'metrics': None, 'weight': 0})
        for diar_seg in diar_segments:
            metrics = self._calculate_segment_metrics(seg_start, seg_end, diar_seg["start"], diar_seg["end"])
            if metrics.overlap_ratio_stt >= self.min_overlap_ratio:
                speaker = diar_seg["speaker"]
                if speaker_scores[speaker]['metrics'] is None or metrics.confidence > speaker_scores[speaker]['metrics'].confidence:
                    speaker_scores[speaker]['metrics'] = metrics
                    speaker_scores[speaker]['weight'] = 1.0
        for speaker, data in speaker_scores.items():
            if data['metrics']:
                candidates.append(SpeakerCandidate(speaker_id=speaker, metrics=data['metrics'], source="diarization", weight=data['weight']))
        if timeline_segs:
            tlcand = self._get_timeline_candidate(segment, timeline_segs)
            if tlcand:
                candidates.append(tlcand)
        candidates.extend(self._get_neighbor_candidates(segment, seg_idx, all_segments, diar_segments))
        candidates.sort(key=lambda c: c.metrics.confidence * c.weight, reverse=True)
        return candidates

    def _get_timeline_candidate(self, segment: Dict, timeline_segs: List[Dict]) -> Optional[SpeakerCandidate]:
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))
        seg_mid = (seg_start + seg_end) / 2
        for tl_seg in timeline_segs:
            if tl_seg["start"] <= seg_mid <= tl_seg["end"]:
                if tl_seg["mode"] == "mono":
                    metrics = self._calculate_segment_metrics(seg_start, seg_end, tl_seg["start"], tl_seg["end"])
                    metrics.confidence = min(1.0, metrics.confidence * 1.2)
                    return SpeakerCandidate(speaker_id=tl_seg["speakers"][0], metrics=metrics, source="timeline", weight=1.2)
        return None

    def _get_neighbor_candidates(self, segment: Dict, seg_idx: int, all_segments: List[Dict], diar_segments: List[Dict]) -> List[SpeakerCandidate]:
        candidates = []
        for offset in range(1, 3 + 1):
            for idx in [seg_idx - offset, seg_idx + offset]:
                if 0 <= idx < len(all_segments) and idx != seg_idx:
                    neighbor = all_segments[idx]
                    if "speaker" in neighbor and neighbor.get("confidence", 0) > 0.7:
                        distance_weight = 1.0 / (abs(idx - seg_idx) + 1)
                        metrics = SegmentMetrics(0,0,0,0,abs(idx - seg_idx), neighbor.get("confidence", 0.5) * distance_weight * 0.6)
                        candidates.append(SpeakerCandidate(speaker_id=neighbor["speaker"], metrics=metrics, source="neighbor", weight=distance_weight * 0.5))
        return candidates

    def _select_best_candidate(self, candidates: List[SpeakerCandidate], segment: Dict, all_segments: List[Dict], seg_idx: int) -> Optional[SpeakerCandidate]:
        if not candidates:
            return None
        best = candidates[0]
        high = [c for c in candidates if c.metrics.confidence > self.confidence_threshold]
        if len(high) > 1:
            source_priority = {"timeline": 3, "diarization": 2, "neighbor": 1}
            high.sort(key=lambda c: (c.metrics.confidence * c.weight, source_priority.get(c.source, 0)), reverse=True)
            best = high[0]
        return best

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        merged = []; i = 0
        while i < len(segments):
            current = segments[i]
            duration = float(current.get("end", 0)) - float(current.get("start", 0))
            if duration < 0.2 and i < len(segments) - 1:
                next_seg = segments[i + 1]
                gap = float(next_seg.get("start", 0)) - float(current.get("end", 0))
                if (current.get("speaker") == next_seg.get("speaker") and gap < 0.3):
                    current["end"] = next_seg.get("end", current["end"])
                    if "text" in current and "text" in next_seg:
                        current["text"] = (current["text"] or "") + " " + (next_seg["text"] or "")
                    if "confidence" in current and "confidence" in next_seg:
                        current["confidence"] = max(current["confidence"], next_seg["confidence"])
                    merged.append(current); i += 2; continue
            merged.append(current); i += 1
        return merged

    def _fix_isolated_segments(self, segments: List[Dict]) -> List[Dict]:
        if len(segments) < 3:
            return segments
        fixed = segments.copy()
        for i in range(1, len(segments) - 1):
            prev_speaker = segments[i-1].get("speaker")
            curr_speaker = segments[i].get("speaker")
            next_speaker = segments[i+1].get("speaker")
            if (prev_speaker and next_speaker and prev_speaker == next_speaker and curr_speaker != prev_speaker):
                duration = float(segments[i].get("end", 0)) - float(segments[i].get("start", 0))
                curr_confidence = segments[i].get("confidence", 0.5)
                if duration < 0.5 and curr_confidence < 0.7:
                    fixed[i]["speaker"] = prev_speaker
                    fixed[i]["confidence"] = min(segments[i-1].get("confidence", 0.5), segments[i+1].get("confidence", 0.5)) * 0.8
                    fixed[i]["assignment_source"] = "isolation_fix"
                    self.assignment_stats["isolation_fix"] += 1
        return fixed

    def _assign_words_optimized(self, words: List[Dict], segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
        seg_speaker_map = {}
        for seg in segments:
            seg_id = seg.get("id")
            if seg_id is not None and "speaker" in seg:
                seg_speaker_map[seg_id] = seg["speaker"]
        for word in words:
            seg_id = word.get("segment_id")
            if seg_id in seg_speaker_map:
                word["speaker"] = seg_speaker_map[seg_id]; continue
            if "start" in word and word["start"] is not None:
                word_time = float(word["start"])
                found = False
                for seg in segments:
                    if float(seg.get("start", 0)) <= word_time <= float(seg.get("end", 0)):
                        if "speaker" in seg:
                            word["speaker"] = seg["speaker"]; found=True; break
                if not found:
                    for diar_seg in diar_segments:
                        if diar_seg["start"] <= word_time <= diar_seg["end"]:
                            word["speaker"] = diar_seg["speaker"]; break
        return words

    def _final_consistency_pass(self, segments: List[Dict], words: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        for seg in segments:
            seg_id = seg.get("id");
            if seg_id is None: continue
            seg_words = [w for w in words if w.get("segment_id") == seg_id]
            if not seg_words: continue
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
        return {"assignment_sources": dict(self.assignment_stats),
                "total_assignments": sum(self.assignment_stats.values())}

# ========================== Pipeline Utilities (mevcut) ===========================

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

def split_audio_by_duration(audio_path: Path, chunk_sec: int, outdir: Path, copy_codecs: bool = True, dbg: Optional[DebugWriter] = None) -> List[Tuple[Path, float]]:
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
    if OpenAI is None:
        raise RuntimeError("openai paketi bulunamadı.")
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

# -------------------- Pyannote VAD & Diarization --------------------
def _vad_pyannote(wav_path: Path, dbg: DebugWriter) -> List[Dict[str, float]]:
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN tanımlı değil.")
    pipe = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=token)
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda")); dbg.snap("VAD_INIT", device="cuda")
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
            pipe.to(torch.device("cuda")); dbg.snap("DIAR_INIT", device="cuda")
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
    text = raw.get("text") or " ".join((s.get("text") or "").strip() for s in segments).strip()
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

# ======================= XTTS & Lipsync Yardımcıları =======================

# 1) XTTSEngine import veya fallback (xtts.py arayüzüne uyumlu)
#    Eğer projendeki xtts.py import edilemezse, TTS.api ile aynı API'yi sağlayan basit bir sınıf kullanıyoruz.
_HAS_XTTS = False
try:
    # Kullanıcı projesindeki xtts.py (register_tts, BaseTTSEngine vb. olabilir)
    # Bu import başarısız olursa fallback'a düşeceğiz.
    from xtts import XTTSEngine as _ProjectXTTSEngine  # noqa: F401
    _HAS_XTTS = True
except Exception:
    _ProjectXTTSEngine = None

class _FallbackXTTSEngine:
    """Projede xtts.py yoksa basit Coqui TTS tabanlı fallback.
    xtts.py'deki `XTTSEngine.synthesize(...)` imzasını taklit eder.  :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, model_name: str="tts_models/multilingual/multi-dataset/xtts_v2", language: str="tr"):
        self.model_name = model_name
        self.language = language
        from TTS.api import TTS  # type: ignore
        self._tts = TTS(model_name)
        try:
            import torch
            self._tts.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            pass

    def synthesize(self, text: str, output_path: Path, speaker_wav: str|None=None, latents_path: str|None=None, speed: float|None=None, lang: str|None=None) -> Path:
        assert not (speaker_wav and latents_path), "speaker_wav ve latents birlikte verilemez."
        lang = (lang or self.language or "tr")
        if latents_path:
            # Latent ile inference (xtts.py ile aynı dahili arayüz)
            import torch, torchaudio
            lat = torch.load(latents_path, map_location="cpu")
            mdl = self._tts.synthesizer.tts_model
            wav = mdl.inference(
                text=text,
                language=lang,
                gpt_cond_latent=lat["gpt"],
                diffusion_conditioning=lat["diff"],
                speaker_embedding=lat["spk"],
            )
            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            torchaudio.save(str(output_path), wav.to(torch.float32), 24000)
            return Path(output_path)
        if not speaker_wav or not Path(speaker_wav).is_file():
            raise FileNotFoundError(f"Speaker WAV bulunamadı: {speaker_wav}")
        kwargs = {
            "text": text,
            "language": lang,
            "file_path": str(output_path),
            "speaker_wav": str(Path(speaker_wav).resolve())
        }
        if speed is not None:
            kwargs["speed"] = speed
        # split_sentences param'ı her versiyonda olmayabilir
        try:
            import inspect
            if "split_sentences" in __import__("inspect").signature(self._tts.tts_to_file).parameters:
                kwargs["split_sentences"] = False if len(text or "") < 120 else True
        except Exception:
            pass
        self._tts.tts_to_file(**kwargs)
        return Path(output_path)

def _load_xtts_engine(model_name: str, language: str):
    """Projeden XTTSEngine varsa onu, yoksa fallback'ı dön."""
    if _ProjectXTTSEngine is not None:
        return _ProjectXTTSEngine(model_name=model_name, language=language)  # :contentReference[oaicite:3]{index=3}
    return _FallbackXTTSEngine(model_name=model_name, language=language)

@dataclass
class XTTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "tr"
    speed: Optional[float] = None  # None -> otomatik süre esnetme

def _ensure_pydub():
    if AudioSegment is None:
        raise RuntimeError("pydub gerekli (pip install pydub)")

def _ffprobe_duration(path: Path) -> float:
    return probe_duration_seconds(path)

def _atempo_chain(speed: float) -> str:
    if speed <= 0: speed = 1.0
    chain: List[float] = []
    s = speed
    while s < 0.5:
        chain.append(0.5); s /= 0.5
    while s > 2.0:
        chain.append(2.0); s /= 2.0
    chain.append(s)
    return ",".join(f"atempo={x:.6f}" for x in chain)

def _time_stretch_to_duration(in_wav: Path, target_sec: float, out_wav: Path) -> Path:
    src_dur = max(0.001, _ffprobe_duration(in_wav))
    if target_sec <= 0:
        shutil.copyfile(in_wav, out_wav); return out_wav
    speed = src_dur / target_sec
    chain = _atempo_chain(speed)
    _run(["ffmpeg","-y","-i",str(in_wav), "-af", chain, "-ar","16000","-ac","1", str(out_wav)])
    return out_wav

def _concat_timeline_audio(segments: List[dict], seg_audio_paths: Dict[int, Path], total_len: float, out_wav: Path) -> Path:
    _ensure_pydub()
    sr = 16000
    out = AudioSegment.silent(duration=int(total_len * 1000), frame_rate=sr)
    for s in segments:
        sid = s.get("id")
        if sid is None or sid not in seg_audio_paths:
            continue
        wav = AudioSegment.from_file(seg_audio_paths[sid])
        wav = wav.set_frame_rate(sr).set_channels(1)
        start_ms = int(float(s.get("start", 0)) * 1000)
        out = out.overlay(wav, position=max(0, start_ms))
    out = out.set_frame_rate(sr).set_channels(1)
    out.export(out_wav, format="wav")
    return out_wav

def _mux_audio_to_video(video_in: Path, audio_in: Path, video_out: Path) -> Path:
    _require_ffmpeg()
    _run(["ffmpeg","-y","-i",str(video_in),"-i",str(audio_in),
          "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(video_out)])
    return video_out

def _apply_wav2lip(video_in: Path, audio_in: Path, wav2lip_repo: Optional[Path], checkpoint: Optional[Path], out_video: Path, face_det_batch: int = 16) -> Optional[Path]:
    if not wav2lip_repo or not checkpoint:
        return None
    infer = wav2lip_repo / "inference.py"
    if not infer.exists() or not checkpoint.exists():
        return None
    _require_ffmpeg()
    cmd = ["python", str(infer), "--checkpoint_path", str(checkpoint), "--face", str(video_in), "--audio", str(audio_in),
           "--outfile", str(out_video), "--pads","0","10","0","0","--nosmooth","--face_det_batch_size", str(face_det_batch)]
    try:
        _run(cmd); return out_video
    except Exception:
        return None

# ==================== Referans Ses & Latent Üretimi (BuildRef) ====================
# Bu kısım, verdiğin build_ref_voices.py mantığına uygun şekilde implement edilmiştir.  :contentReference[oaicite:4]{index=4}
def build_reference_voices(original_audio: Path,
                           segments: List[dict],
                           target_lang: str = "tr",
                           seconds: int = 9,
                           min_chunk_ms: int = 1000,
                           margin_ms: int = 150,
                           workdir: Optional[Path] = None) -> Tuple[Path, Dict[str, str]]:
    _ensure_pydub()
    if not segments:
        raise RuntimeError("Referans ses için segment yok.")
    audio = AudioSegment.from_file(original_audio)
    voices_dir = (workdir or original_audio.parent) / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    latents_map: Dict[str, str] = {}
    # XTTS modeli (latents üretimi için) – xtts.py API’siyle birebir  :contentReference[oaicite:5]{index=5}
    try:
        xtts_engine = _load_xtts_engine("tts_models/multilingual/multi-dataset/xtts_v2", target_lang)
        xtts_model = xtts_engine._tts.synthesizer.tts_model
    except Exception as e:
        xtts_model = None
        print("[XTTS] Model yüklenemedi, latent üretimi atlanacak:", e)

    groups: Dict[str, List[dict]] = defaultdict(list)
    for seg in segments:
        spk = str(seg.get("speaker") or seg.get("speaker_id") or "UNKNOWN")
        groups[spk].append(seg)

    from collections import Counter as _Counter
    for speaker, segs in groups.items():
        candidates = sorted(segs, key=lambda s: (float(s.get("end",0)) - float(s.get("start",0))), reverse=True)
        collected: List[AudioSegment] = []
        total_ms = 0
        for seg in candidates:
            start = float(seg.get("start",0)); end = float(seg.get("end",0))
            chunk_ms = int((end - start) * 1000)
            if chunk_ms < min_chunk_ms: continue
            begin = max(int(start*1000) + margin_ms, 0)
            finish = min(int(end*1000) - margin_ms, len(audio))
            if finish <= begin: continue
            chunk = audio[begin:finish]
            if chunk.dBFS == float("-inf") or chunk.dBFS < -45:  # sessiz/çok düşük SNR
                continue
            collected.append(chunk)
            total_ms += len(chunk)
            if total_ms >= seconds*1000:
                break
        if not collected:
            print(f"[REF] {speaker} için uygun parça bulunamadı."); continue
        voice = collected[0]
        for c in collected[1:]:
            voice += c
        voice = voice[:seconds*1000]
        if voice.dBFS != float("-inf"):
            voice = voice.apply_gain(-20.0 - voice.dBFS)  # normalize
        voice = voice.set_channels(1).set_frame_rate(16000)
        out_path = voices_dir / f"{speaker}.wav"
        voice.export(out_path, format="wav")
        abs_path = str(out_path.resolve())

        # cinsiyet çoğunluğu (varsa)
        genders = [seg.get("gender") for seg in segs if seg.get("gender") in ("M","F")]
        majority = _Counter(genders).most_common(1)[0][0] if genders else None
        for seg in segs:
            seg["voice_id"] = abs_path
            if majority:
                seg["gender"] = majority

        if xtts_model is not None:
            try:
                import torch as _torch
                try:
                    gpt, diff, spk = xtts_model.get_conditioning_latents(audio_path=[abs_path])
                except TypeError:
                    gpt, diff, spk = xtts_model.get_conditioning_latents(abs_path)
                lat_path = voices_dir / f"{speaker}.latents.pt"
                _torch.save({"gpt": gpt.detach().cpu(),
                             "diff": diff.detach().cpu(),
                             "spk": spk.detach().cpu()}, lat_path)
                latents_map[speaker] = str(lat_path.resolve())
                print(f"[XTTS] {speaker} latent kaydedildi -> {lat_path}")
            except Exception as e:
                print(f"[XTTS] {speaker} latent çıkarılamadı:", e)

    return voices_dir, latents_map

# ==================== Segment Bazlı XTTS -> Süre Uydurma -> Birleştirme =============
def synthesize_dub_track_xtts(segments: List[dict],
                              all_text: Optional[str],
                              voices_dir: Path,
                              latents_map: Dict[str, str],
                              target_lang: str,
                              out_dir: Path,
                              xtts_cfg: Optional[XTTSConfig] = None) -> Tuple[Path, Dict[int, Path]]:
    xtts_cfg = xtts_cfg or XTTSConfig(language=target_lang)
    tts = _load_xtts_engine(xtts_cfg.model_name, xtts_cfg.language)  # :contentReference[oaicite:6]{index=6}

    seg_audio: Dict[int, Path] = {}
    tmp_audio_dir = out_dir / "_tts_segments"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        sid = seg.get("id")
        if sid is None:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0)); end = float(seg.get("end", 0.0))
        target_len = max(0.01, end - start)

        spk = str(seg.get("speaker") or "UNKNOWN")
        spk_wav = voices_dir / f"{spk}.wav"
        lat_path = latents_map.get(spk)

        raw_out = tmp_audio_dir / f"seg_{sid:06d}.raw.wav"
        if lat_path:
            tts.synthesize(text, output_path=raw_out, latents_path=lat_path, lang=target_lang)  # :contentReference[oaicite:7]{index=7}
        else:
            if not spk_wav.exists():
                # fallback: herhangi bir referans
                fallback = next(iter(voices_dir.glob("*.wav")), None)
                if fallback is None:
                    raise RuntimeError("Referans ses bulunamadı.")
                spk_wav = fallback
            tts.synthesize(text, output_path=raw_out, speaker_wav=str(spk_wav), lang=target_lang)  # :contentReference[oaicite:8]{index=8}

        stretched = tmp_audio_dir / f"seg_{sid:06d}.fit.wav"
        if xtts_cfg.speed is not None:
            chain = _atempo_chain(xtts_cfg.speed)
            _run(["ffmpeg","-y","-i",str(raw_out), "-af", chain, "-ar","16000","-ac","1", str(stretched)])
        else:
            _time_stretch_to_duration(raw_out, target_len, stretched)
        seg_audio[sid] = stretched

    total_len = 0.0
    if segments:
        total_len = max(total_len, float(segments[-1].get("end", 0.0)))
    full_wav = out_dir / "dubbed.timeline.mono16k.wav"
    _concat_timeline_audio(segments, seg_audio, total_len, full_wav)
    return full_wav, seg_audio

# ==================== Basit Lipsync (Wav2Lip) + Fallback Mux ======================
def lipsync_or_mux(video_in: Path,
                   dub_audio_wav: Path,
                   out_dir: Path,
                   wav2lip_repo: Optional[str] = None,
                   wav2lip_ckpt: Optional[str] = None,
                   face_det_batch: int = 16) -> Tuple[Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lipsynced = out_dir / f"{video_in.stem}.lipsync.mp4"
    muxed     = out_dir / f"{video_in.stem}.dubbed.mp4"
    used_lipsync = False
    ls_path = _apply_wav2lip(
        video_in=video_in,
        audio_in=dub_audio_wav,
        wav2lip_repo=(Path(wav2lip_repo) if wav2lip_repo else None),
        checkpoint=(Path(wav2lip_ckpt) if wav2lip_ckpt else None),
        out_video=lipsynced,
        face_det_batch=face_det_batch
    )
    if ls_path and ls_path.exists():
        used_lipsync = True
        return ls_path, used_lipsync
    out_path = _mux_audio_to_video(video_in, dub_audio_wav, muxed)
    return out_path, used_lipsync

# ========================= Main Pipeline (genişletilmiş) ==========================
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
        # ------ Dublaj & Lipsync ------
        do_dub: bool = True,
        target_lang: Optional[str] = None,
        xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        xtts_speed: Optional[float] = None,
        do_lipsync: bool = True,
        wav2lip_repo: Optional[str] = None,
        wav2lip_checkpoint: Optional[str] = None
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

    for idx, (part_path, base_offset) in enumerate(chunks):
        effective_offset = base_offset - (chunk_overlap_sec if idx > 0 else 0.0)
        raw = transcribe_file(part_path, model=stt_model, language=language, prompt=prompt, want_word_timestamps=want_word_timestamps, dbg=dbg)
        segs, words, text, part_dur, part_lang = _normalize_verbose(raw)
        if lang is None:
            lang = part_lang
        for s in segs:
            s = dict(s); s["start"] = float(s.get("start",0.0)) + effective_offset; s["end"] = float(s.get("end",0.0)) + effective_offset
            s["id"] = last_id; last_id += 1; merged_segments.append(s)
        for w in words:
            w = dict(w)
            if "start" in w and w["start"] is not None: w["start"] = float(w["start"]) + effective_offset
            if "end"   in w and w["end"]   is not None: w["end"]   = float(w["end"])   + effective_offset
            merged_words.append(w)
        merged_texts.append(text)

    diar_segments: Optional[List[dict]] = None
    vad_regions: Optional[List[Dict[str,float]]] = None
    timeline: Optional[List[TimelineSeg]] = None

    # VAD & Diarization
    wav_for_diar = ensure_wav_mono16k(audio, out / "_work")
    if use_vad:
        vad_regions = _vad_pyannote(wav_for_diar, dbg)
    if diarize:
        diar_segments = _diarize_pyannote(wav_for_diar, speaker_count, dbg)
        if vad_regions:
            diar_segments = clip_segments_to_regions(diar_segments, vad_regions)
        timeline = build_flat_timeline(diar_segments, stereo_threshold=stereo_threshold)

        if use_optimized_mapping:
            dbg.snap("SPEAKER_MAPPING", method="optimized_real_data")
            timeline_json = None
            if timeline:
                timeline_json = [{"start": t.start, "end": t.end, "mode": t.mode, "speakers": list(t.speakers), "channels": t.channels or {}} for t in timeline]
            mapper = OptimizedSpeakerMapper(min_overlap_ratio=min_overlap_ratio, boundary_tolerance=boundary_tolerance, use_vad_boundaries=use_vad_boundaries, use_timeline=use_timeline, confidence_threshold=confidence_threshold)
            merged_segments, merged_words = mapper.map_speakers(merged_segments, merged_words, diar_segments, timeline_json, vad_regions)
            stats = mapper.get_statistics(); dbg.snap("MAPPING_STATS", **stats)
            assigned_segments = sum(1 for s in merged_segments if "speaker" in s)
            high_conf_segments = sum(1 for s in merged_segments if s.get("confidence", 0) >= confidence_threshold)
            dbg.snap("MAPPING_RESULTS", total_segments=len(merged_segments), assigned_segments=assigned_segments, high_confidence_segments=high_conf_segments, assigned_words=sum(1 for w in merged_words if "speaker" in w))
        else:
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

        # RTTM & speakers CSV
        rttm_path = out / f"{wav_for_diar.stem}.diarization.rttm"
        _write_rttm(diar_segments, rttm_path, uri=src.stem)
        spk_csv = out / f"{src.stem}.speakers.csv"
        with spk_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["start","end","speaker"])
            for d in diar_segments:
                w.writerow([d["start"], d["end"], d["speaker"]])
        dbg.snap("DIAR_FILES", rttm=str(rttm_path), speakers_csv=str(spk_csv))

    # ---------------- DUBLaj + LİPSYNC ENTEGRASYON ----------------
    dub_audio_wav = None
    lipsync_video = None
    lipsync_used  = False

    if do_dub:
        tlang = target_lang or (lang or "tr")
        # 1) Referans sesler + latents (build_ref_voices.py ile uyumlu)  :contentReference[oaicite:9]{index=9}
        voices_dir, latents_map = build_reference_voices(
            original_audio=wav_for_diar,
            segments=merged_segments,
            target_lang=tlang,
            workdir=out / "_work"
        )
        # 2) XTTS sentez + süre uydurma + tek iz birleştirme  :contentReference[oaicite:10]{index=10}
        dub_audio_wav, seg_audio_map = synthesize_dub_track_xtts(
            segments=merged_segments,
            all_text=" ".join(merged_texts).strip(),
            voices_dir=voices_dir,
            latents_map=latents_map,
            target_lang=tlang,
            out_dir=out,
            xtts_cfg=XTTSConfig(model_name=xtts_model_name, language=tlang, speed=xtts_speed)
        )
        # 3) Basit Lipsync (opsiyonel Wav2Lip) + fallback mux
        if do_lipsync:
            lipsync_video, lipsync_used = lipsync_or_mux(
                video_in=src,
                dub_audio_wav=dub_audio_wav,
                out_dir=out,
                wav2lip_repo=wav2lip_repo,
                wav2lip_ckpt=wav2lip_checkpoint,
                face_det_batch=16
            )
        else:
            lipsync_video, _ = lipsync_or_mux(
                video_in=src,
                dub_audio_wav=dub_audio_wav,
                out_dir=out,
                wav2lip_repo=None,
                wav2lip_ckpt=None
            )

    # ---------------- ÇIKTILAR ----------------
    last_seg_end = 0.0
    if merged_segments:
        last_seg_end = max(float(s.get("end",0.0)) for s in merged_segments)
    summary = _write_outputs(out, src.stem, merged_segments, merged_words, " ".join(merged_texts).strip(), lang, last_seg_end, diarization=diar_segments, timeline=timeline)
    summary["models"] = {"stt": stt_model, "llm": llm_model}
    if diar_segments:
        summary["files"]["speakers_csv"] = str(out / f"{src.stem}.speakers.csv")
        rttm_guess = out / f"{audio.stem}.diarization.rttm"
        if rttm_guess.exists():
            summary["files"]["speakers_rttm"] = str(rttm_guess)
    if timeline is not None:
        summary["files"]["timeline_csv"] = str(out / f"{src.stem}.timeline.csv")

    if do_dub and dub_audio_wav:
        summary.setdefault("files", {})["dub_audio_wav"] = str(dub_audio_wav)
        summary["dub"] = {
            "target_lang": target_lang or (lang or "tr"),
            "xtts_model": xtts_model_name,
            "audio_wav": str(dub_audio_wav),
            "video": str(lipsync_video) if lipsync_video else None,
            "lipsync": bool(lipsync_used),
            "voices_dir": str(voices_dir),
            "voice_latents": latents_map
        }
        if lipsync_video:
            summary["files"]["dubbed_video"] = str(lipsync_video)

    if debug and dbg.events:
        debug_json = out / f"{src.stem}.debug.json"
        debug_json.write_text(json.dumps(dbg.events, indent=2), encoding="utf-8")
        summary.setdefault("files", {})["debug"] = str(debug_json)

    return summary

# ============================= Example Run =============================
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
        # Mapping
        use_optimized_mapping=True,
        min_overlap_ratio=0.2,
        boundary_tolerance=0.1,
        use_vad_boundaries=True,
        use_timeline=True,
        confidence_threshold=0.6,
        # DUB + LIPSYNC
        do_dub=True,
        target_lang="tr",  # hedef dil
        xtts_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        xtts_speed=None,   # None -> segment süresine otomatik esnetme
        do_lipsync=True,
        wav2lip_repo=None,        # örn: "/opt/Wav2Lip"
        wav2lip_checkpoint=None   # örn: "/opt/Wav2Lip/checkpoints/Wav2Lip.pth"
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
