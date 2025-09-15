# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Iterable
import numpy as np
import copy
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import speaker analyzer and Demucs components
from speaker_segment_analyzer import (
    SpeakerSegmentAnalyzer,
    DemucsVocalSeparator,
    DubbingMixer
)

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
import logging

# Cache for FFmpeg filter availability
_FFMPEG_HAS_RUBBERBAND = None

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

# ================== LOGGING ==================
logger = logging.getLogger("miniapp")
logger.setLevel(logging.INFO)

# ================== XTTS LATENTS PATCH ==================
import torch

logger_xtts = logging.getLogger("miniapp.xtts")
logger_xtts.setLevel(logging.INFO)

@dataclass
class XttsConfig:
    tts: str = "xtts"
    sample_rate: int = 22050
    temperature: float = 0.7

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _save_tensor(t: torch.Tensor, path: str) -> None:
    _safe_mkdir(os.path.dirname(path))
    torch.save(t.detach().cpu(), path)

def get_conditioning_latents_safe(model, ref_wav_path: str):
    try:
        latents = model.get_conditioning_latents(audio_path=[ref_wav_path])
    except TypeError:
        try:
            latents = model.get_conditioning_latents([ref_wav_path])
        except Exception as e:
            raise RuntimeError(f"XTTS latents çağrısı başarısız: {e}") from e
    except Exception as e:
        raise RuntimeError(f"XTTS latents çağrısı başarısız: {e}") from e

    if not isinstance(latents, tuple):
        raise RuntimeError(f"Beklenmeyen dönüş tipi: {type(latents)}")
    if len(latents) == 3:
        gpt, diff, spk = latents
        return {"gpt": gpt, "diff": diff, "spk": spk}
    elif len(latents) == 2:
        gpt, spk = latents
        return {"gpt": gpt, "diff": None, "spk": spk}
    else:
        raise RuntimeError(f"Beklenmeyen latent sayısı: {len(latents)}")

def build_ref_voice_latents(model,
                            speakers_to_wav: Dict[str, str],
                            out_dir: str) -> Dict[str, Dict[str, str]]:
    voice_latents_paths: Dict[str, Dict[str, str]] = {}
    _safe_mkdir(out_dir)

    for speaker, wav_path in speakers_to_wav.items():
        abs_path = os.path.abspath(wav_path)
        if not os.path.isfile(abs_path):
            logger_xtts.warning("[XTTS] %s için referans wav bulunamadı: %s", speaker, abs_path)
            continue
        try:
            lat = get_conditioning_latents_safe(model, abs_path)
        except Exception as e:
            logger_xtts.warning("[XTTS] %s latent çıkarılamadı: %s", speaker, e)
            continue

        spk_dir = os.path.join(out_dir, speaker)
        _safe_mkdir(spk_dir)
        paths = {}
        gpt_path = os.path.join(spk_dir, "gpt.pt"); _save_tensor(lat["gpt"], gpt_path); paths["gpt"] = gpt_path
        spk_path = os.path.join(spk_dir, "spk.pt"); _save_tensor(lat["spk"], spk_path); paths["spk"] = spk_path
        if lat["diff"] is not None:
            diff_path = os.path.join(spk_dir, "diff.pt"); _save_tensor(lat["diff"], diff_path); paths["diff"] = diff_path
        voice_latents_paths[speaker] = paths
        logger_xtts.info("[XTTS] %s için latents kaydedildi: %s", speaker, json.dumps(paths, ensure_ascii=False))

    return voice_latents_paths

def synthesize_with_optional_latents(model,
                                     text: str,
                                     out_path: str,
                                     cfg: XttsConfig,
                                     latents_paths: Optional[Dict[str, str]] = None,
                                     speaker_wav: Optional[str] = None,
                                     language: Optional[str] = None) -> str:
    _safe_mkdir(os.path.dirname(out_path))
    if cfg.tts == "xtts" and latents_paths:
        try:
            audio = model.synthesize(
                text=text,
                latents_path=latents_paths,
                temperature=getattr(cfg, "temperature", 0.7),
                language=language,
                sample_rate=cfg.sample_rate,
            )
            _write_wav(audio, out_path, cfg.sample_rate)
            return out_path
        except Exception as e:
            logger_xtts.warning("[XTTS] latent ile synth başarısız, wav fallback: %s", e)
    if cfg.tts == "xtts" and speaker_wav:
        audio = model.synthesize(
            text=text,
            speaker_wav=speaker_wav,
            temperature=getattr(cfg, "temperature", 0.7),
            language=language,
            sample_rate=cfg.sample_rate,
        )
        _write_wav(audio, out_path, cfg.sample_rate)
        return out_path
    audio = model.synthesize(text=text, temperature=getattr(cfg, "temperature", 0.7), language=language, sample_rate=cfg.sample_rate)
    _write_wav(audio, out_path, cfg.sample_rate)
    return out_path

def _write_wav(audio, out_path: str, sr: int):
    import numpy as np
    import soundfile as sf
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    elif not isinstance(audio, (list, tuple,)):
        if isinstance(audio, dict) and "audio" in audio:
            arr = audio["audio"]
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            audio = arr
        else:
            audio = np.asarray(audio)
    audio = audio.astype("float32", copy=False)
    sf.write(out_path, audio, sr)

# ====================== Timeline / Diar Utils ======================
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

# ================== OPTİMİZE EDİLMİŞ KONUŞMACI EŞLEME ===================
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
                 min_overlap_ratio: float = 0.5,
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

    # (iç yardımcılar aynen)

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
        if not timeline: return None
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
            normalized.append({"start": float(v["start"]), "end": float(v["end"]), "duration": float(v["end"]) - float(v["start"])})
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
        confidence = (weights['overlap_stt'] * overlap_stt + weights['overlap_diar'] * overlap_diar + weights['iou'] * iou + weights['boundary'] * boundary_score)
        return min(1.0, max(0.0, confidence))

    def _adjust_boundaries_with_vad(self, segments: List[Dict], vad_regions: List[Dict]) -> List[Dict]:
        adjusted = []
        for seg in segments:
            seg_start = float(seg.get("start", 0)); seg_end = float(seg.get("end", 0))
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
        seg_start = float(segment.get("start", 0)); seg_end = float(segment.get("end", 0))
        if seg_end <= seg_start: return candidates
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
        seg_start = float(segment.get("start", 0)); seg_end = float(segment.get("end", 0))
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
        if not candidates: return None
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
        if len(segments) < 3: return segments
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

# ========================== Pipeline Utilities ===========================
MAX_BODY_BYTES = 26_214_400
SOFT_LIMIT_BYTES = 24 * 1024 * 1024

class DebugWriter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: List[Dict[str, Any]] = []
    def snap(self, tag: str, **data: Any):
        if not self.enabled: return
        evt = {"tag": tag, "ts": time.time(), **data}
        self.events.append(evt)
        kv = " ".join(f"{k}={v}" for k, v in data.items())
        print(f"[{tag}] {kv}")

def _require_ffmpeg():
    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            raise RuntimeError(f"{bin_name} bulunamadı. FFmpeg/FFprobe kurulu olmalı.")

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Stderr: {e.stderr.decode('utf-8', errors='ignore')}")
        raise

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

# -------------------- Basit & Dayanıklı Çeviri --------------------
def _openai_client_or_raise() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai paketi bulunamadı.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY tanımlı değil (çeviri için gerekli).")
    return OpenAI()

def _chat_translate_batch(client: OpenAI,
                          items: List[Tuple[int, str]],
                          src_lang: Optional[str],
                          tgt_lang: str,
                          model: str) -> List[str]:
    """
    items: [(sid, text)] – 1..N numaralı satırlar olarak prompt'a verilir.
    Geri dönüş: sadece çeviriler, satır-satır (ör: N satır).
    """
    numbered = "\n".join(f"{i+1}. {t or ''}" for i, (_, t) in enumerate(items))
    sys_msg = f"You are a professional translator specializing in dubbing. Translate to natural spoken {tgt_lang}."
    if src_lang:
        sys_msg += f" Source: {src_lang}."
    user_msg = f"Translate each line below to {tgt_lang}. Keep the same number of lines. Be concise and natural:\n{numbered}"
    try:
        rsp = _safe_chat(client, model=model, system=sys_msg, user=user_msg)
        if not rsp:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Empty translation response for batch of {len(items)} items")
            return [""] * len(items)

        out = [ln.strip() for ln in rsp.split("\n") if ln.strip() != ""]
        # Eğer sayı/numara geldiyse temizle
        cleaned: List[str] = []
        for ln in out:
            # "1. ..." veya "1) ..." veya "1 -" gibi ön ekleri temizle
            if ln and len(ln) > 2 and ln[0].isdigit() and " " in ln:
                cleaned.append(ln.split(" ", 1)[1].strip())
            else:
                cleaned.append(ln.lstrip("0123456789).:- ").strip())
        # Satır sayısı uyuşmazsa alt/üst kırp
        if len(cleaned) < len(items):
            cleaned += [""] * (len(items)-len(cleaned))
        return cleaned[:len(items)]
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Translation batch failed: {e}")
        # Return empty strings to trigger fallback
        return [""] * len(items)

def _safe_chat(client: OpenAI, *, model: str, system: str, user: str) -> str:
    # SDK değişikliklerine karşı sade kullanım
    try:
        comp = client.chat.completions.create(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0)
        return comp.choices[0].message.content or ""
    except Exception as e:
        # Bazı SDK’lar new Responses API kullanıyor olabilir; basit fallback yoksa raise
        raise

def translate_segments(segments: List[dict],
                       target_lang: str,
                       model: str = "gpt-4o-mini",
                       source_lang: Optional[str] = None,
                       batch_size: int = 30,
                       dbg: Optional[DebugWriter] = None) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Segment metinlerini hedef dile çevirir, orijinali `orig_text` alanına koyar.
    Boş veya çok kısa metinler atlanır.
    """
    import logging
    logger = logging.getLogger(__name__)

    if not segments:
        return segments, {"translated": False}

    # Google Translate kullanımını kontrol et
    use_google = model == "google" or model == "google_cloud"

    if use_google:
        # Google Cloud Translate API kullan
        try:
            from google.cloud import translate_v2 as translate
            import os
            # Credentials kontrolü
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")
            translate_client = translate.Client()
            logger.info("Using Google Cloud Translate API")
        except ImportError:
            logger.error("google-cloud-translate not installed. Run: poetry add google-cloud-translate")
            logger.info("Falling back to OpenAI")
            use_google = False
            client = _openai_client_or_raise()
        except Exception as e:
            logger.error(f"Google Cloud Translate init failed: {e}")
            logger.info("Falling back to OpenAI")
            use_google = False
            client = _openai_client_or_raise()
    else:
        client = _openai_client_or_raise()

    # Hazırla: sadece metni olanları al
    pending: List[Tuple[int, int, str]] = []  # (idx, seg_id, text)
    for i, seg in enumerate(segments):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        pending.append((i, seg.get("id", i), txt))

    if not pending:
        return segments, {"translated": False}

    # Toplu çeviri
    total = len(pending)
    done = 0

    if use_google:
        # Google Translate ile çeviri
        for k in range(0, total, batch_size):
            chunk = pending[k:k+batch_size]
            texts_to_translate = []

            # Her segment için - ARTİK KISALTMA YOK, TAM METNİ ÇEVİR
            for (idx, sid, txt) in chunk:
                # İNGİLİZCE METNİ KISALTMA! TAM HALİYLE ÇEVİR
                texts_to_translate.append(txt)

            try:
                # Google Translate batch çeviri
                # Source language dönüşümü: 'english' -> 'en', 'turkish' -> 'tr' vb.
                src_lang = source_lang or 'en'
                if src_lang == 'english':
                    src_lang = 'en'
                elif src_lang == 'turkish':
                    src_lang = 'tr'
                elif src_lang == 'spanish':
                    src_lang = 'es'
                elif src_lang == 'french':
                    src_lang = 'fr'
                elif src_lang == 'german':
                    src_lang = 'de'

                results = translate_client.translate(
                    texts_to_translate,
                    source_language=src_lang,
                    target_language=target_lang,
                    format_='text'
                )

                for (i, _, _), result in zip(chunk, results):
                    orig = segments[i].get("text", "")
                    segments[i]["orig_text"] = orig
                    tr = result.get('translatedText', '') if isinstance(result, dict) else ''

                    # Artık prefix kullanmıyoruz, sadece HTML entities temizle
                    tr = tr.replace('&#39;', "'")
                    tr = tr.replace('&quot;', '"')
                    tr = tr.replace('&amp;', '&')

                    if not tr or tr.strip() == "":
                        logger.warning(f"Translation failed for segment {i}: '{orig[:50]}...'")
                        segments[i]["text"] = orig
                        segments[i]["translation_failed"] = True
                    else:
                        # KELİME SAYISI ANALİZİ (GOOGLE TRANSLATE İÇİN)
                        orig_words = len(orig.split())
                        tr_words = len(tr.split())
                        orig_chars = len(orig)
                        tr_chars = len(tr)

                        # Oran hesaplama
                        word_ratio = tr_words / max(orig_words, 1)
                        char_ratio = tr_chars / max(orig_chars, 1)

                        # DETAYLI LOG
                        logger.info(f"📊Segment {i} Translation Analysis:")
                        logger.info(f"  EN: {orig_words} words, {orig_chars} chars")
                        logger.info(f"  TR: {tr_words} words, {tr_chars} chars")
                        logger.info(f"  Word ratio: {word_ratio:.2f}x ({'+' if word_ratio > 1 else ''}{(word_ratio-1)*100:.1f}%)")
                        logger.info(f"  Char ratio: {char_ratio:.2f}x ({'+' if char_ratio > 1 else ''}{(char_ratio-1)*100:.1f}%)")

                        if word_ratio > 1.3:
                            logger.warning(f"  ⚠️ UZAMA: Türkçe %{(word_ratio-1)*100:.0f} daha uzun!")
                        elif word_ratio < 0.7:
                            logger.warning(f"  ✅ KISALMA: Türkçe %{(1-word_ratio)*100:.0f} daha kısa!")

                        segments[i]["text"] = tr
                        segments[i]["translation_failed"] = False
                        segments[i]["orig_words"] = orig_words
                        segments[i]["tr_words"] = tr_words
                        segments[i]["word_ratio"] = word_ratio

            except Exception as e:
                logger.error(f"Google Translate batch failed: {e}")
                # Hata durumunda orijinal metinleri koru
                for (i, _, _) in chunk:
                    orig = segments[i].get("text", "")
                    segments[i]["orig_text"] = orig
                    segments[i]["text"] = orig
                    segments[i]["translation_failed"] = True

            done += len(chunk)
            if dbg: dbg.snap("TRANSLATE_BATCH", size=len(chunk), done=done, total=total, method="google")

        # GOOGLE TRANSLATE BATCH SONRASI GENEL ANALİZ
        if done == total:
            total_orig_words = sum(s.get('orig_words', 0) for s in segments)
            total_tr_words = sum(s.get('tr_words', 0) for s in segments)
            if total_orig_words > 0:
                overall_ratio = total_tr_words / total_orig_words
                logger.info("\n" + "="*60)
                logger.info("🌍 OVERALL TRANSLATION STATISTICS:")
                logger.info(f"  Total EN words: {total_orig_words}")
                logger.info(f"  Total TR words: {total_tr_words}")
                logger.info(f"  Overall ratio: {overall_ratio:.2f}x")

                # Anchor point adaylarını bul (ratio 0.8-1.2 arası)
                anchor_candidates = []
                for s in segments:
                    if 0.8 <= s.get('word_ratio', 1.0) <= 1.2:
                        anchor_candidates.append(s.get('id', -1))

                logger.info(f"\n🎯 ANCHOR CANDIDATES (ratio 0.8-1.2): {len(anchor_candidates)} segments")
                if anchor_candidates[:10]:  # İlk 10 adayı göster
                    logger.info(f"  First 10: {anchor_candidates[:10]}")
                logger.info("="*60 + "\n")
    else:
        # OpenAI ile çeviri (mevcut kod)
        for k in range(0, total, batch_size):
            chunk = pending[k:k+batch_size]
            pairs = [(sid, txt) for (_, sid, txt) in chunk]
            translations = _chat_translate_batch(client, pairs, source_lang, target_lang, model=model)
            for (i, _, _), tr in zip(chunk, translations):
                orig = segments[i].get("text", "")
                segments[i]["orig_text"] = orig
                # Çeviri boş veya None ise hata logla ve orijinali kullan
                if not tr or tr.strip() == "":
                    logger.warning(f"Translation failed for segment {i}: '{orig[:50]}...'")
                    segments[i]["text"] = orig  # Keep original if translation fails
                    segments[i]["translation_failed"] = True
                else:
                    segments[i]["text"] = tr
                    segments[i]["translation_failed"] = False
            done += len(chunk)
            if dbg: dbg.snap("TRANSLATE_BATCH", size=len(chunk), done=done, total=total, method="openai")

    return segments, {"translated": True, "target_lang": target_lang, "model": model, "count": total}

# -------------------- Pyannote VAD & Diarization --------------------
def _vad_pyannote(wav_path: Path, dbg: DebugWriter) -> List[Dict[str, float]]:
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN tanımlı değil.")
    pipe = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=token)
    try:
        import torch as _t
        if _t.cuda.is_available():
            pipe.to(_t.device("cuda")); dbg.snap("VAD_INIT", device="cuda")
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
        import torch as _t
        if _t.cuda.is_available():
            pipe.to(_t.device("cuda")); dbg.snap("DIAR_INIT", device="cuda")
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
            out.append({"start": max(s0,r0), "end": min(s1,r1), "speaker": s["speaker"], "id": s.get("id"), "text": s.get("text")})
    out.sort(key=lambda d: (d["start"], d["end"], d.get("speaker","")))
    merged: List[Dict[str, Any]] = []
    for seg in out:
        if merged and merged[-1].get("speaker") == seg.get("speaker") and abs(merged[-1]["end"] - seg["start"]) < 1e-6:
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
        verbose["timeline"] = []
        for t in timeline:
            if isinstance(t, dict):
                # If timeline item is already a dict
                verbose["timeline"].append(t)
            elif hasattr(t, 'start'):
                # If timeline item is an object with attributes
                verbose["timeline"].append({
                    "start": t.start,
                    "end": t.end,
                    "mode": t.mode,
                    "speakers": list(t.speakers) if hasattr(t, 'speakers') else [],
                    "channels": t.channels if hasattr(t, 'channels') else None
                })
            elif isinstance(t, tuple) and len(t) >= 2:
                # If timeline item is a tuple (start, end)
                verbose["timeline"].append({
                    "start": t[0],
                    "end": t[1],
                    "mode": "speech",
                    "speakers": [],
                    "channels": None
                })
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
                if isinstance(t, dict):
                    start = t.get("start", 0)
                    end = t.get("end", 0)
                    mode = t.get("mode", "speech")
                    speakers = t.get("speakers", [])
                    channels = t.get("channels", {})
                elif isinstance(t, tuple) and len(t) >= 2:
                    start, end = t[0], t[1]
                    mode = "speech"
                    speakers = []
                    channels = {}
                else:
                    start = t.start
                    end = t.end
                    mode = t.mode
                    speakers = t.speakers if hasattr(t, 'speakers') else []
                    channels = t.channels if hasattr(t, 'channels') else {}

                spks = "|".join(speakers) if speakers else ""
                L = channels.get("L", "") if channels else ""
                R = channels.get("R", "") if channels else ""
                w.writerow([start, end, mode, spks, L, R])

    files = {
        "verbose_json": str(verbose_json_path),
        "segments_srt": str(outdir / f"{stem}.segments.srt"),
        "segments_csv": str(outdir / f"{stem}.segments.csv"),
        "words_csv": str(outdir / f"{stem}.words.csv"),
    }
    if timeline is not None:
        files["timeline_csv"] = str(outdir / f"{stem}.timeline.csv")
    return {"text": text, "language": language, "duration": duration, "files": files}


# ============ NO-OVERLAP: Aynı konuşmacı segmentleri asla çakışmasın ============
def enforce_no_overlap_same_speaker(segments: List[dict], margin: float = 0.1, allow_overlap: bool = False) -> Tuple[List[dict], Dict[str, int]]:
    """
    Aynı konuşmacıya ait segmentler üst üste binmesin.
    - allow_overlap=True ise hiçbir değişiklik yapma
    - Eğer seg.start < prev_end + margin ise, seg.start'ı prev_end + margin'e ileri al.
    - Negatif/çok kısa kalan segmentleri at.
    """
    # Eğer overlap'e izin veriliyorsa, hiçbir değişiklik yapma
    if allow_overlap:
        dbg = debug_logger if 'debug_logger' in locals() else None
        if dbg:
            print("[NO_OVERLAP_ENFORCED] Segment overlap allowed - no trimming/dropping")
        return segments, {"trimmed": 0, "dropped": 0, "allow_overlap": True}

    segs = sorted(copy.deepcopy(segments), key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    by_spk: Dict[str, List[dict]] = defaultdict(list)
    for s in segs:
        spk = str(s.get("speaker") or "")
        by_spk[spk].append(s)

    trims, drops = 0, 0
    for spk, ss in by_spk.items():
        ss.sort(key=lambda s: float(s.get("start", 0.0)))
        prev_end = -1e9
        for s in ss:
            st = float(s.get("start", 0.0)); en = float(s.get("end", 0.0))
            if st < prev_end + margin:
                new_st = prev_end + margin
                # çok kısa/negatif kalıyorsa bu segmenti at
                if new_st >= en - 1e-3:
                    s["_drop"] = True
                    drops += 1
                    continue
                s["start"] = new_st
                trims += 1
            prev_end = float(s.get("end", 0.0))

    cleaned: List[dict] = []
    for s in segs:
        if s.get("_drop"):
            continue
        cleaned.append(s)
    return cleaned, {"trimmed": trims, "dropped": drops}


# ================== DEMUX & MÜZİK YATAĞI & DUB MİX (FFmpeg/Separation) ==================
def _extract_original_audio(video_in: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{video_in.stem}.orig.48k.wav"
    _run(["ffmpeg", "-y", "-i", str(video_in), "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(out)])
    return out

def _extract_music_bed(video_in: Path, workdir: Path, dbg: Optional[DebugWriter] = None) -> Tuple[Path, bool]:
    """
    Müzik yatağını çıkar:
      1) demucs varsa:  --two-stems vocals  -> no_vocals.wav
      2) spleeter varsa: 2stems -> accompaniment.wav
      3) fallback: orijinal ses (ducking ile konuşmayı basacağız)
    Dönen: (music_bed_48k_stereo, separated?)
    """
    workdir.mkdir(parents=True, exist_ok=True)
    orig = _extract_original_audio(video_in, workdir)

    # --- Demucs dene ---
    try:
        if shutil.which("demucs"):
            sepdir = workdir / "demucs_sep"
            cmd = ["demucs", "--two-stems", "vocals", "-o", str(sepdir), str(orig)]
            _run(cmd)
            candidates = list(sepdir.glob("**/no_vocals.wav"))
            if candidates:
                music = candidates[0]
                bed = workdir / f"{video_in.stem}.music_bed.48k.wav"
                _run(["ffmpeg", "-y", "-i", str(music), "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(bed)])
                if dbg: dbg.snap("MUSIC_BED", method="demucs", file=str(bed))
                return bed, True
    except Exception as e:
        if dbg: dbg.snap("MUSIC_BED_FAIL", method="demucs", err=str(e))

    # --- Spleeter dene ---
    try:
        if shutil.which("spleeter"):
            sepdir = workdir / "spleeter_sep"
            cmd = ["spleeter", "separate", "-p", "spleeter:2stems", "-o", str(sepdir), str(orig)]
            _run(cmd)
            stem_dir = sepdir / orig.stem
            for cand in [stem_dir / "accompaniment.wav", stem_dir / "accompaniment.flac"]:
                if cand.exists():
                    bed = workdir / f"{video_in.stem}.music_bed.48k.wav"
                    _run(["ffmpeg", "-y", "-i", str(cand), "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(bed)])
                    if dbg: dbg.snap("MUSIC_BED", method="spleeter", file=str(bed))
                    return bed, True
    except Exception as e:
        if dbg: dbg.snap("MUSIC_BED_FAIL", method="spleeter", err=str(e))

    # --- Fallback: orijinal ses (ducking ile bastırılacak) ---
    if dbg: dbg.snap("MUSIC_BED", method="original_audio_fallback", file=str(orig))
    return orig, False

def _mix_music_and_dub(
        video_in: Path,
        dub_audio_wav: Path,
        out_dir: Path,
        dbg: Optional[DebugWriter] = None,
        dub_gain_db: float = 0.0,
        music_gain_db: float = -2.0,
) -> Tuple[Path, Optional[Path]]:
    """
    Dublaj + müzik karışımı:
      - Müzik yatağı ayrıştırılıyorsa (Demucs/Spleeter) onu kullan.
      - Ayrışma yoksa, sidechaincompress + (hafif) mid-band azaltma ile orijinal konuşmayı bastır.
      - Çıktı: 48k/stereo WAV (mux için).
    """
    mixdir = out_dir / "_mix"
    mixdir.mkdir(parents=True, exist_ok=True)
    music_bed, separated = _extract_music_bed(video_in, mixdir, dbg)

    # Dub'u 48k/stereo normalize et
    dub48 = mixdir / f"{Path(dub_audio_wav).stem}.48k.stereo.wav"
    _run(["ffmpeg", "-y", "-i", str(dub_audio_wav), "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(dub48)])

    final = out_dir / f"{video_in.stem}.final_mix.48k.wav"

    # Ayrışma yoksa orta frekansları biraz kıs (300–3000 Hz civarı) + sidechain duck + amix
    # Not: firequalizer çoğu ffmpeg build'ında var. Yoksa filter atlanır (hata almamak için separated True olduğunda zaten eklemiyoruz).
    music_pre = "aformat=sample_rates=48000:channel_layouts=stereo"
    if not separated:
        music_pre += ",firequalizer=gain_entry='entry(300,-6);entry(1000,-8);entry(3000,-6)'"
    music_pre += f",volume={10**(music_gain_db/20):.6f}"

    filter_complex = (
        f"[0:a]{music_pre}[m];"
        f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={10**(dub_gain_db/20):.6f}[v];"
        f"[m][v]sidechaincompress=threshold=0.050:ratio=12:attack=15:release=250[duck];"
        f"[duck][v]amix=inputs=2:normalize=0:duration=longest[mix]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(music_bed),
        "-i", str(dub48),
        "-filter_complex", filter_complex,
        "-map", "[mix]",
        "-ar", "48000", "-ac", "2",
        "-c:a", "pcm_s16le",
        str(final)
    ]
    _run(cmd)
    if dbg: dbg.snap("DUB_MIX_DONE", final=str(final), separated_music=bool(separated))
    return final, (music_bed if separated else None)


# ======================= XTTS & Lipsync Yardımcıları =======================
def _looks_like_url(s: str | None) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")

def _ensure_ytdlp() -> str:
    """
    İndirme için önce yt-dlp, yoksa youtube-dl arar.
    """
    for bin_name in ("yt-dlp", "youtube-dl"):
        if shutil.which(bin_name) is not None:
            return bin_name
    raise RuntimeError("Ne 'yt-dlp' ne de 'youtube-dl' bulunamadı. Lütfen kur: pip install yt-dlp")

def _download_via_ytdlp(url: str, outdir: Path, *, prefer_mp4: bool = True, dbg: Optional[DebugWriter] = None) -> Path:
    """
    URL'den videoyu indirir ve indirilen **nihai** dosyanın tam yolunu döndürür.
    - Çıktıyı outdir'e sabitler (-P home:...)
    - after_move:filepath ile nihai yolu alır (destekten yoksunsa legacy 'filename' fallback)
    - MP4 tercihi varsa remux/merge ayarları eklenir, ancak .mkv/.webm gibi fallback çıktıları da kabul eder
    """
    outdir.mkdir(parents=True, exist_ok=True)
    ytdlp = _ensure_ytdlp()

    # Şablonu klasörsüz bırakıyoruz; klasörü -P ile belirliyoruz
    template = "%(title).200B-%(id)s.%(ext)s"

    base_cmd = [
        ytdlp,
        "--no-playlist",
        "--no-progress",
        "--newline",
        "-N", "4",
        "-P", f"home:{outdir}",
        "-o", template,
        "--no-part",           # .part dosyalarını bırakma
        "--retries", "10",
        "--fragment-retries", "10",
    ]

    if prefer_mp4:
        # Uyumluysa MP4'e remux; mümkün değilse merge-output-format da MP4 denesin (yt-dlp gerekirse MKV'ye düşer)
        base_cmd += ["--remux-video", "mp4", "--merge-output-format", "mp4"]

    if dbg: dbg.snap("YTDLP_START", url=url, outdir=str(outdir))

    # 1) Modern yol: after_move:filepath (postprocessor sonrası nihai yol)
    try:
        cmd = base_cmd + ["--print", "after_move:filepath", url]
        cp = _run(cmd)
        lines = [ln.strip() for ln in cp.stdout.decode(errors="ignore").splitlines() if ln.strip()]
        for ln in reversed(lines):
            p = Path(ln)
            if p.exists():
                if dbg: dbg.snap("YTDLP_DONE", file=str(p), size_bytes=p.stat().st_size)
                return p
    except subprocess.CalledProcessError as e:
        # yt-dlp hata verirse üstte yakalanır; legacy denemeye geçeceğiz
        pass

    # 2) Legacy yol: filename (bazı sürümlerde post-move olmayabilir)
    try:
        cmd = base_cmd + ["--print", "filename", url]
        cp = _run(cmd)
        lines = [ln.strip() for ln in cp.stdout.decode(errors="ignore").splitlines() if ln.strip()]
        for ln in reversed(lines):
            p = Path(ln)
            if p.exists():
                if dbg: dbg.snap("YTDLP_DONE", file=str(p), size_bytes=p.stat().st_size)
                return p
    except subprocess.CalledProcessError:
        # indirme başarısız olmuş olabilir; fallback glob taraması yine de denensin
        pass

    # 3) Son çare: outdir içinde en yeni video dosyasını yakala
    exts = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
    candidates = [p for p in outdir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    if candidates:
        p = max(candidates, key=lambda x: x.stat().st_mtime)
        if dbg: dbg.snap("YTDLP_GLOB_PICK", file=str(p), size_bytes=p.stat().st_size)
        return p

    raise FileNotFoundError(
        "İndirme başarısız ya da çıktı dosyası bulunamadı. "
        "ffmpeg kurulu mu ve yt-dlp güncel mi? (pip install -U yt-dlp)"
    )

_HAS_XTTS = False
try:
    from models.tts.xtts import XTTSEngine as _ProjectXTTSEngine  # noqa: F401
    _HAS_XTTS = True
except Exception:
    _ProjectXTTSEngine = None

class _FallbackXTTSEngine:
    def __init__(self, model_name: str="tts_models/multilingual/multi-dataset/xtts_v2", language: str="tr"):
        self.model_name = model_name
        self.language = language
        from TTS.api import TTS  # type: ignore
        self._tts = TTS(model_name)
        try:
            import torch as _t
            self._tts.to("cuda" if _t.cuda.is_available() else "cpu")
        except Exception:
            pass

    def synthesize(self, text: str, output_path: Path, speaker_wav: str|None=None, latents_path: str|None=None, speed: float|None=None, lang: str|None=None) -> Path:
        """
        HIZ/SÜRE OYNAMAYAN SÜRÜM:
        - speed parametresi tamamen yok sayılır.
        - latents varsa doğrudan model.inference kullanılır.
        - yoksa TTS.api ile dosyaya yazılır (split_sentences varsa yalnızca onu ayarlarız).
        """
        assert not (speaker_wav and latents_path), "speaker_wav ve latents birlikte verilemez."
        lang = (lang or self.language or "tr")

        if latents_path:
            import torch as _t, torchaudio
            lat = _t.load(latents_path, map_location="cpu")
            mdl = self._tts.synthesizer.tts_model
            wav = mdl.inference(
                text=text,
                language=lang,
                gpt_cond_latent=lat["gpt"],
                diffusion_conditioning=lat.get("diff"),
                speaker_embedding=lat["spk"],
            )
            if not isinstance(wav, _t.Tensor):
                wav = _t.tensor(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            torchaudio.save(str(output_path), wav.to(_t.float32), 24000)
            return Path(output_path)

        if not speaker_wav or not Path(speaker_wav).is_file():
            raise FileNotFoundError(f"Speaker WAV bulunamadı: {speaker_wav}")

        kwargs = {
            "text": text,
            "language": lang,
            "file_path": str(output_path),
            "speaker_wav": str(Path(speaker_wav).resolve()),
        }
        # Hız asla ayarlanmıyor; speed parametresi YOK SAYILIR.
        try:
            import inspect
            if "split_sentences" in inspect.signature(self._tts.tts_to_file).parameters:
                kwargs["split_sentences"] = False if len(text or "") < 120 else True
        except Exception:
            pass

        self._tts.tts_to_file(**kwargs)
        return Path(output_path)

def _ffmpeg_has_filter(filter_name: str) -> bool:
    """Check if ffmpeg has a specific filter available (cached for rubberband)"""
    global _FFMPEG_HAS_RUBBERBAND

    if filter_name == "rubberband" and _FFMPEG_HAS_RUBBERBAND is not None:
        return _FFMPEG_HAS_RUBBERBAND

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=5
        )
        has_filter = filter_name in result.stdout
    except Exception:
        has_filter = False

    if filter_name == "rubberband":
        _FFMPEG_HAS_RUBBERBAND = has_filter

    return has_filter

# ==================== Segment Rebalancing for Tempo Limits ====================

def _seg_duration(seg: dict) -> float:
    """Calculate segment duration"""
    return float(seg.get("end", 0)) - float(seg.get("start", 0))

def _seg_gap(seg_a: dict, seg_b: dict) -> float:
    """Calculate gap between two segments (negative if overlapping)"""
    return float(seg_b.get("start", 0)) - float(seg_a.get("end", 0))

def _tempo_needed(tts_duration: float, target_duration: float) -> float:
    """Calculate required tempo factor for time stretching"""
    target_duration = max(1e-6, target_duration)
    return max(1e-6, tts_duration / target_duration)

def merge_short_segments_same_speaker(segments: List[dict], min_duration: float = 1.0, max_gap: float = 0.5) -> List[dict]:
    """
    Aynı konuşmacının kısa segmentlerini birleştir.

    Args:
        segments: Segment listesi
        min_duration: Minimum segment süresi (saniye)
        max_gap: Birleştirme için maksimum boşluk (saniye)
    """
    import logging
    logger = logging.getLogger(__name__)

    if not segments:
        return segments

    merged = []
    i = 0
    merge_count = 0

    while i < len(segments):
        current = segments[i].copy()
        current_duration = float(current.get("end", 0)) - float(current.get("start", 0))

        # Kısa segment ise ve sonraki segmentlerle birleştirmeyi dene
        if current_duration < min_duration and i + 1 < len(segments):
            j = i + 1
            accumulated_text = current.get("text", "")

            # Aynı konuşmacının yakın segmentlerini birleştir
            while j < len(segments):
                next_seg = segments[j]
                gap = float(next_seg.get("start", 0)) - float(current.get("end", 0))

                # Aynı konuşmacı ve küçük boşluk varsa birleştir
                if (current.get("speaker") == next_seg.get("speaker") and
                    gap <= max_gap):

                    # Metni birleştir
                    accumulated_text = (accumulated_text.rstrip() + " " +
                                      next_seg.get("text", "").lstrip()).strip()

                    # Bitiş zamanını güncelle
                    current["end"] = next_seg.get("end")
                    current["text"] = accumulated_text

                    # Yeni süreyi kontrol et
                    new_duration = float(current["end"]) - float(current["start"])
                    merge_count += 1

                    # Yeterli uzunluğa ulaştıysa dur
                    if new_duration >= min_duration:
                        break

                    j += 1
                else:
                    break

            # Birleştirilmiş segmenti ekle
            merged.append(current)
            i = j + 1
        else:
            merged.append(current)
            i += 1

    if merge_count > 0:
        logger.info(f"Merged {merge_count} short segments for better tempo control")

    return merged

def _borrow_from_gaps(segments: List[dict], idx: int, need_extra: float, max_range: int = 3) -> float:
    """
    Aggressively borrow time from neighboring segments (up to max_range segments away).
    Returns amount of time gained.
    """
    import logging
    logger = logging.getLogger(__name__)

    gained = 0.0
    current = segments[idx]
    current_speaker = current.get("speaker")

    # Phase 1: Try borrowing from immediate gaps
    if idx < len(segments) - 1:
        gap = _seg_gap(segments[idx], segments[idx + 1])
        if gap > 0:
            # Leave minimal gap for aggressive borrowing
            take = min(gap - 0.01, need_extra)
            if take > 0:
                segments[idx]["end"] = float(segments[idx]["end"]) + take
                segments[idx + 1]["start"] = float(segments[idx + 1]["start"]) + take
                gained += take
                need_extra -= take

    if need_extra > 1e-6 and idx > 0:
        gap = _seg_gap(segments[idx - 1], segments[idx])
        if gap > 0:
            take = min(gap - 0.01, need_extra)
            if take > 0:
                segments[idx - 1]["end"] = float(segments[idx - 1]["end"]) + take
                gained += take
                need_extra -= take

    # Phase 2: Aggressive borrowing from same speaker's nearby segments
    if need_extra > 1e-6:
        # Look at neighbors within max_range
        for distance in range(1, max_range + 1):
            if need_extra <= 1e-6:
                break

            # Check forward neighbors
            if idx + distance < len(segments):
                neighbor = segments[idx + distance]
                if neighbor.get("speaker") == current_speaker:
                    neighbor_duration = _seg_duration(neighbor)
                    # Can compress neighbor to 1.1x speed (saving ~9% of its duration)
                    if neighbor_duration > 1.0:
                        spare_time = neighbor_duration * 0.08  # Conservative: 8% compression
                        take = min(need_extra, spare_time)
                        if take > 0:
                            # Shorten neighbor
                            neighbor["end"] = float(neighbor["end"]) - take
                            gained += take
                            need_extra -= take
                            logger.debug(f"Borrowed {take:.3f}s from segment {idx+distance}")

            # Check backward neighbors
            if need_extra > 1e-6 and idx - distance >= 0:
                neighbor = segments[idx - distance]
                if neighbor.get("speaker") == current_speaker:
                    neighbor_duration = _seg_duration(neighbor)
                    # Can compress neighbor to 1.1x speed
                    if neighbor_duration > 1.0:
                        spare_time = neighbor_duration * 0.08
                        take = min(need_extra, spare_time)
                        if take > 0:
                            # Move neighbor start later (giving us space before)
                            neighbor["start"] = float(neighbor["start"]) + take
                            gained += take
                            need_extra -= take
                            logger.debug(f"Borrowed {take:.3f}s from segment {idx-distance}")

    if gained > 0:
        logger.info(f"Segment {idx}: Borrowed total {gained:.3f}s from neighbors")

    return gained

def _can_merge_segments(seg_a: dict, seg_b: dict) -> bool:
    """Check if two segments can be merged"""
    if seg_a is None or seg_b is None:
        return False

    # Check speaker
    same_speaker = (seg_a.get("speaker") and
                   seg_a.get("speaker") == seg_b.get("speaker"))

    # More aggressive merging criteria for better tempo control
    # Merge if:
    # 1. Same speaker AND (short segment OR small gap)
    # 2. Different speaker BUT very short segment that needs extreme tempo

    dur_b = _seg_duration(seg_b)
    gap = _seg_gap(seg_a, seg_b)

    # Same speaker - merge more aggressively
    if same_speaker:
        return dur_b <= 1.5 or gap <= 0.5  # Increased thresholds

    # Different speaker - only merge if necessary for tempo
    # (very short segments that would need extreme tempo)
    return dur_b <= 0.3 and gap <= 0.1

def _merge_segments(segments: List[dict], idx: int) -> bool:
    """
    Merge segment at idx with idx+1.
    Returns True if merge was successful.
    """
    if idx >= len(segments) - 1:
        return False

    seg_a, seg_b = segments[idx], segments[idx + 1]

    if not _can_merge_segments(seg_a, seg_b):
        return False

    # Merge text
    text_a = (seg_a.get("text") or "").rstrip()
    text_b = (seg_b.get("text") or "").lstrip()
    merged_text = (text_a + (" " if text_a and text_b else "") + text_b).strip()

    # Update segment
    seg_a["text"] = merged_text
    seg_a["end"] = max(float(seg_a["end"]), float(seg_b["end"]))

    # Preserve other fields if they exist
    if "orig_text" in seg_b and "orig_text" in seg_a:
        orig_a = (seg_a.get("orig_text") or "").rstrip()
        orig_b = (seg_b.get("orig_text") or "").lstrip()
        seg_a["orig_text"] = (orig_a + (" " if orig_a and orig_b else "") + orig_b).strip()

    # Remove merged segment
    del segments[idx + 1]
    return True

def rebalance_segments_for_tempo(
    segments: List[dict],
    tts_durations: Dict[int, float],
    tempo_min: float = 0.5,
    tempo_max: float = 2.0,
    max_passes: int = 2
) -> List[dict]:
    """
    Rebalance segments to ensure tempo stays within limits.

    Args:
        segments: List of segments with id, start, end, text, speaker
        tts_durations: Dictionary mapping segment id to TTS output duration
        tempo_min: Minimum allowed tempo (default 0.75)
        tempo_max: Maximum allowed tempo (default 1.25)
        max_passes: Maximum optimization passes

    Returns:
        Rebalanced segments list
    """
    import logging
    logger = logging.getLogger(__name__)

    segments = sorted(segments, key=lambda s: float(s.get("start", 0)))

    for pass_num in range(max_passes):
        changed = False
        i = 0

        while i < len(segments):
            seg = segments[i]
            sid = seg.get("id", i)
            target_duration = _seg_duration(seg)

            # Get TTS duration or estimate
            tts_duration = float(tts_durations.get(sid, target_duration))

            # Calculate required tempo
            tempo = _tempo_needed(tts_duration, target_duration)

            # If tempo is within limits, continue
            if tempo_min <= tempo <= tempo_max:
                i += 1
                continue

            # Tempo too high (need to slow down) - extend target duration
            if tempo > tempo_max:
                # Calculate minimum target duration needed
                min_target = tts_duration / tempo_max
                need_extra = max(0.0, min_target - target_duration)

                # Try aggressive borrowing from gaps and neighbors
                gained = _borrow_from_gaps(segments, i, need_extra, max_range=3)

                # If not enough, try merging with neighbor
                if gained + 1e-6 < need_extra:
                    merged = _merge_segments(segments, i)
                    if merged:
                        logger.debug(f"Merged segment {sid} with next to reduce tempo")
                        changed = True
                        # Re-evaluate current segment
                        continue

                if gained > 0:
                    logger.debug(f"Borrowed {gained:.3f}s for segment {sid}")
                    changed = True

            # Tempo too low (need to speed up) - reduce target duration
            elif tempo < tempo_min:
                # Calculate maximum target duration allowed
                max_target = tts_duration / tempo_min
                need_cut = max(0.0, target_duration - max_target)

                if need_cut > 1e-3:
                    # Try to shorten segment end
                    new_end = float(seg["end"]) - need_cut
                    min_duration = 0.05  # Minimum segment duration

                    if new_end > float(seg["start"]) + min_duration:
                        seg["end"] = new_end
                        logger.debug(f"Shortened segment {sid} by {need_cut:.3f}s")
                        changed = True
                    else:
                        # Segment would be too short, try merging
                        if i > 0 and _can_merge_segments(segments[i-1], seg):
                            # Merge with previous
                            segments[i-1]["text"] = (
                                (segments[i-1].get("text") or "").rstrip() + " " +
                                (seg.get("text") or "").lstrip()
                            ).strip()
                            segments[i-1]["end"] = max(
                                float(segments[i-1]["end"]),
                                float(seg["end"])
                            )
                            del segments[i]
                            logger.debug(f"Merged segment {sid} with previous")
                            changed = True
                            continue
                        elif _merge_segments(segments, i):
                            logger.debug(f"Merged segment {sid} with next")
                            changed = True
                            continue

            i += 1

        if not changed:
            logger.info(f"Rebalancing converged after {pass_num + 1} passes")
            break

    return segments

def estimate_tts_duration(text: str, language: str = "tr") -> float:
    """
    Estimate TTS duration based on text length.
    Adjusted for XTTS behavior which tends to produce longer outputs.
    """
    if not text:
        return 0.0

    text = text.strip()
    char_count = len(text)
    word_count = len(text.split())

    # XTTS tends to speak slower than natural speech
    # Adjusted rates based on observed behavior
    cps_rates = {
        "tr": 10.0,  # Turkish (was 14.0, but XTTS is slower)
        "en": 11.0,  # English (was 15.0)
        "es": 10.5,  # Spanish
        "fr": 10.0,  # French
        "de": 9.5,   # German
        "it": 10.5,  # Italian
        "pt": 10.0,  # Portuguese
        "ru": 9.0,   # Russian
        "zh": 6.0,   # Chinese
        "ja": 7.5,   # Japanese
        "ko": 8.0,   # Korean
        "ar": 9.0,   # Arabic
    }

    cps = cps_rates.get(language, 10.0)

    # Base duration from character count
    base_duration = char_count / cps

    # XTTS has minimum durations for short texts
    # and adds pauses between words
    if word_count <= 2:
        # Very short texts get minimum duration
        min_duration = 0.8
    elif word_count <= 5:
        # Short texts get extra padding
        min_duration = 1.2
    else:
        # Normal texts - add pause time between words
        pause_per_word = 0.05  # 50ms average pause
        min_duration = base_duration + (word_count * pause_per_word)

    return max(min_duration, base_duration)

def _time_stretch_to_duration(in_wav: Path, target_sec: float, out_wav: Path, enable: bool = True) -> Path:
    """
    Advanced time stretching with quality preservation.
    Adjusts audio duration to match target while maintaining pitch and quality.
    """
    if not enable or target_sec <= 0:
        # Fallback: just normalize without stretching
        _run(["ffmpeg", "-y", "-i", str(in_wav), "-ar", "16000", "-ac", "1", str(out_wav)])
        return out_wav

    # Get current duration
    current_duration = _ffprobe_duration(in_wav)

    # If already close enough (within 50ms), still apply exact duration lock
    if abs(current_duration - target_sec) < 0.05:
        _run(["ffmpeg", "-y", "-i", str(in_wav),
              "-af", f"aresample=16000,apad=pad_dur={target_sec:.6f},atrim=0:{target_sec:.6f}",
              "-ac", "1", str(out_wav)])
        return out_wav

    # Calculate tempo factor (playback rate)
    # tempo > 1 = speed up (shorter duration), tempo < 1 = slow down (longer duration)
    tempo = current_duration / target_sec

    # Log the adjustment for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Time stretch: {current_duration:.3f}s -> {target_sec:.3f}s (tempo={tempo:.3f})")

    # Apply different strategies based on tempo factor
    if 0.87 <= tempo <= 1.15:
        # Small adjustment (±13%): Use atempo for best quality
        atempo = _atempo_chain(tempo)
        cmd = ["ffmpeg", "-y", "-i", str(in_wav),
               "-af", f"{atempo},aresample=16000,apad=pad_dur={target_sec:.6f},atrim=0:{target_sec:.6f}",
               "-ac", "1", str(out_wav)]
        _run(cmd)

    elif 0.5 <= tempo <= 2.0:
        # Moderate adjustment: Use rubberband if available via ffmpeg filter
        if _ffmpeg_has_filter("rubberband"):
            # Use rubberband for better quality time stretching
            cmd = ["ffmpeg", "-y", "-i", str(in_wav),
                   "-af", f"rubberband=tempo={tempo:.6f}:pitch=1.0,aresample=16000,apad=pad_dur={target_sec:.6f},atrim=0:{target_sec:.6f}",
                   "-ac", "1", str(out_wav)]
        else:
            # Fallback to atempo chain
            atempo = _atempo_chain(tempo)
            cmd = ["ffmpeg", "-y", "-i", str(in_wav),
                   "-af", f"{atempo},aresample=16000,apad=pad_dur={target_sec:.6f},atrim=0:{target_sec:.6f}",
                   "-ac", "1", str(out_wav)]
        _run(cmd)

    else:
        # Extreme adjustment: Multi-pass approach for better quality
        # First pass: Apply maximum safe stretch
        temp_wav = out_wav.parent / f"{out_wav.stem}_temp.wav"

        if tempo > 2.0:
            # Need to speed up significantly
            first_tempo = 2.0
            remaining_tempo = tempo / 2.0
        else:
            # Need to slow down significantly
            first_tempo = 0.5
            remaining_tempo = tempo / 0.5

        # First pass with intermediate resampling for quality
        atempo1 = _atempo_chain(first_tempo)
        cmd1 = ["ffmpeg", "-y", "-i", str(in_wav),
                "-af", f"{atempo1},aresample=48000",
                "-ac", "1", str(temp_wav)]
        _run(cmd1)

        # Second pass with exact duration lock
        atempo2 = _atempo_chain(remaining_tempo)
        cmd2 = ["ffmpeg", "-y", "-i", str(temp_wav),
                "-af", f"{atempo2},aresample=16000,apad=pad_dur={target_sec:.6f},atrim=0:{target_sec:.6f}",
                "-ac", "1", str(out_wav)]
        _run(cmd2)

        # Cleanup temp file
        if temp_wav.exists():
            temp_wav.unlink()

    # Verify the output duration
    final_duration = _ffprobe_duration(out_wav)
    duration_error = abs(final_duration - target_sec)

    if duration_error > 0.1:  # More than 100ms error
        logger.warning(f"Time stretch accuracy issue: target={target_sec:.3f}s, got={final_duration:.3f}s, error={duration_error:.3f}s")

    return out_wav

def _load_xtts_engine(model_name: str, language: str):
    if _ProjectXTTSEngine is not None:
        return _ProjectXTTSEngine(model_name=model_name, language=language)
    return _FallbackXTTSEngine(model_name=model_name, language=language)

@dataclass
class XTTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "tr"
    speed: Optional[float] = 1.3  # Varsayılan daha hızlı konuşma

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


def _concat_timeline_audio(segments: List[dict], seg_audio_paths: Dict[int, Path], total_len: float, out_wav: Path, breath_gap_ms: int = 200) -> Path:
    _ensure_pydub()
    sr = 16000

    # Create the output audio with proper timing
    out = AudioSegment.silent(duration=int(total_len * 1000), frame_rate=sr)

    # Global timing tracker for all segments
    global_timeline = []
    speaker_last_end = defaultdict(float)

    # First pass: Calculate actual positions with global timing
    segment_positions = []

    for s in sorted(segments, key=lambda x: float(x.get("start", 0))):
        sid = s.get("id")
        if sid is None or sid not in seg_audio_paths:
            continue

        speaker = s.get("speaker", "UNKNOWN")
        original_start = float(s.get("start", 0))
        original_end = float(s.get("end", 0))

        # Load the TTS audio
        wav = AudioSegment.from_file(seg_audio_paths[sid])
        wav = wav.set_frame_rate(sr).set_channels(1)
        tts_duration = len(wav) / 1000.0

        # Calculate adaptive breath gap (shorter for rapid exchanges, longer for topic changes)
        adaptive_gap = breath_gap_ms
        if len(segment_positions) > 0:
            prev_seg = segment_positions[-1]
            time_gap = original_start - prev_seg["original_end"]

            # Reduce gap for rapid conversation (< 0.5s gap)
            if time_gap < 0.5:
                adaptive_gap = min(50, breath_gap_ms)
            # Normal gap for regular pauses (0.5-1.5s)
            elif time_gap < 1.5:
                adaptive_gap = min(100, breath_gap_ms)
            # Keep original gap for topic changes (> 1.5s)

        # Global timing: respect original timing as much as possible
        target_start_ms = int(original_start * 1000)

        # Check for same-speaker overlap
        if speaker in speaker_last_end:
            min_start_ms = int(speaker_last_end[speaker] * 1000) + adaptive_gap
            target_start_ms = max(target_start_ms, min_start_ms)

        # Check for cross-speaker overlaps in global timeline
        for placed in global_timeline:
            placed_end_ms = placed["start_ms"] + placed["duration_ms"]
            # If there's an overlap with any previously placed segment
            if target_start_ms < placed_end_ms and (target_start_ms + len(wav)) > placed["start_ms"]:
                # For different speakers, allow slight overlap (natural conversation)
                if placed["speaker"] != speaker:
                    # Allow up to 100ms overlap for natural conversation flow
                    if target_start_ms < placed_end_ms - 100:
                        target_start_ms = placed_end_ms - 100
                else:
                    # Same speaker must not overlap
                    target_start_ms = placed_end_ms + adaptive_gap

        segment_positions.append({
            "segment": s,
            "wav": wav,
            "speaker": speaker,
            "start_ms": target_start_ms,
            "duration_ms": len(wav),
            "original_start": original_start,
            "original_end": original_end,
            "adaptive_gap": adaptive_gap
        })

        # Update global timeline
        global_timeline.append({
            "speaker": speaker,
            "start_ms": target_start_ms,
            "duration_ms": len(wav)
        })

        # Update speaker tracking
        speaker_last_end[speaker] = (target_start_ms + len(wav)) / 1000.0

    # Second pass: Place audio with overlap redistribution
    for pos_info in segment_positions:
        start_ms = pos_info["start_ms"]
        wav = pos_info["wav"]

        # Apply with crossfade for smoother transitions
        out = out.overlay(wav, position=max(0, start_ms))

    out = out.set_frame_rate(sr).set_channels(1)
    out.export(out_wav, format="wav")
    return out_wav

def _concat_timeline_audio_with_sync(segments: List[dict], seg_audio_paths: Dict[int, Path], total_len: float, out_wav: Path) -> Path:
    """
    Advanced concatenation with sync anchoring and drift compensation.
    - Places segments at exact timestamps
    - Implements sync anchors every N segments
    - Compensates for accumulated drift
    """
    _ensure_pydub()
    import logging
    logger = logging.getLogger(__name__)

    sr = 16000
    out = AudioSegment.silent(duration=int(total_len * 1000), frame_rate=sr)

    # Configuration
    SYNC_ANCHOR_INTERVAL = 10  # Create sync anchor every N segments
    MAX_ALLOWED_DRIFT = 0.5   # Maximum allowed drift in seconds
    CROSSFADE_MS = 10         # Crossfade duration for smoother transitions

    # Tracking variables
    segments_placed = 0
    accumulated_drift = 0.0
    last_anchor_time = 0.0
    speaker_last_end = defaultdict(float)

    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: float(x.get("start", 0)))

    for idx, seg in enumerate(sorted_segments):
        sid = seg.get("id")
        if sid is None or sid not in seg_audio_paths:
            continue

        speaker = seg.get("speaker", "UNKNOWN")
        original_start = float(seg.get("start", 0))
        original_end = float(seg.get("end", 0))

        # Load the audio segment
        wav = AudioSegment.from_file(seg_audio_paths[sid])
        wav = wav.set_frame_rate(sr).set_channels(1)
        actual_duration = len(wav) / 1000.0

        # Calculate placement position
        target_start = original_start

        # Apply drift compensation at sync anchors
        if segments_placed > 0 and segments_placed % SYNC_ANCHOR_INTERVAL == 0:
            # This is a sync anchor point
            if abs(accumulated_drift) > MAX_ALLOWED_DRIFT:
                # Apply gradual drift correction
                drift_correction = accumulated_drift * 0.5  # Correct 50% of drift
                target_start -= drift_correction
                accumulated_drift -= drift_correction
                logger.info(f"Sync anchor at segment {segments_placed}: Corrected {drift_correction:.3f}s drift")

            last_anchor_time = target_start

        # Ensure no same-speaker overlap
        if speaker in speaker_last_end:
            min_gap = 0.02  # 20ms minimum gap between same speaker
            min_start = speaker_last_end[speaker] + min_gap
            if target_start < min_start:
                adjustment = min_start - target_start
                target_start = min_start
                accumulated_drift += adjustment

        # Place the segment
        start_ms = int(target_start * 1000)

        # Apply crossfade if overlapping with existing audio
        if CROSSFADE_MS > 0 and start_ms > 0:
            # Check if there's audio at this position
            test_slice = out[max(0, start_ms - CROSSFADE_MS):start_ms + CROSSFADE_MS]
            if test_slice.dBFS > -60:  # There's audio here
                # Apply crossfade
                wav_with_fade = wav.fade_in(CROSSFADE_MS)
                out = out.overlay(wav_with_fade, position=max(0, start_ms))
            else:
                # No overlap, place normally
                out = out.overlay(wav, position=max(0, start_ms))
        else:
            out = out.overlay(wav, position=max(0, start_ms))

        # Update tracking
        actual_end = target_start + actual_duration
        speaker_last_end[speaker] = actual_end
        segments_placed += 1

        # Track drift
        expected_end = original_end
        segment_drift = actual_end - expected_end
        accumulated_drift += segment_drift

        # Log significant drift
        if abs(segment_drift) > 0.1:
            logger.debug(f"Segment {sid}: drift={segment_drift:.3f}s, accumulated={accumulated_drift:.3f}s")

    # Final statistics
    logger.info(f"Concatenation complete: {segments_placed} segments placed")
    logger.info(f"Final accumulated drift: {accumulated_drift:.3f}s")

    out = out.set_frame_rate(sr).set_channels(1)
    out.export(out_wav, format="wav")
    return out_wav

def _mux_audio_to_video(video_in: Path, audio_in: Path, video_out: Path) -> Path:
    _require_ffmpeg()

    # Ensure input files exist
    if not video_in.exists():
        raise FileNotFoundError(f"Video file not found: {video_in}")
    if not audio_in.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_in}")

    # Ensure output directory exists
    video_out.parent.mkdir(parents=True, exist_ok=True)

    _run(["ffmpeg","-y","-i",str(video_in),"-i",str(audio_in),
          "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k",str(video_out)])
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
    # XTTS modeli (latents üretimi için)
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
            if chunk.dBFS == float("-inf") or chunk.dBFS < -45:
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
            voice = voice.apply_gain(-20.0 - voice.dBFS)
        voice = voice.set_channels(1).set_frame_rate(16000)
        out_path = voices_dir / f"{speaker}.wav"
        voice.export(out_path, format="wav")
        abs_path = str(out_path.resolve())

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
                    lat = get_conditioning_latents_safe(xtts_model, abs_path)
                except Exception as e:
                    print(f"[XTTS] {speaker} latent çıkarılamadı:", e)
                    continue
                lat_path = voices_dir / f"{speaker}.latents.pt"
                _torch.save({"gpt": lat["gpt"].detach().cpu(),
                             "diff": (lat["diff"].detach().cpu() if lat["diff"] is not None else None),
                             "spk": lat["spk"].detach().cpu()}, lat_path)
                latents_map[speaker] = str(lat_path.resolve())
                print(f"[XTTS] {speaker} latent kaydedildi -> {lat_path}")
            except Exception as e:
                print(f"[XTTS] {speaker} latent çıkarılamadı:", e)

    return voices_dir, latents_map

# ==================== Anchor-Based Segmentation Algorithm =============
def create_anchor_based_segments(
    segments: List[dict],
    diar_segments: List[dict],
    anchors: List[Tuple[float, float]] = None
) -> List[dict]:
    """
    Anchor-Diarization birleştirme algoritması
    B = S ∪ {O_i} - Konuşmacı sınırları + Anchor noktaları
    """
    import logging
    logger = logging.getLogger(__name__)

    if not segments:
        return segments

    # Eğer anchor yoksa, otomatik anchor oluştur
    if not anchors:
        total_duration = max(s.get('end', 0) for s in segments) if segments else 0
        anchors = [(0.0, 0.0), (total_duration, total_duration)]

        # Word ratio'ya göre iyi segmentleri anchor yap (0.8-1.2 arası)
        good_segments = [s for s in segments if 0.8 <= s.get('word_ratio', 1.0) <= 1.2]

        # Her 10 saniyede bir anchor ekle (daha sık)
        for t in range(10, int(total_duration), 10):
            closest = min(good_segments, key=lambda s: abs(s['start'] - t), default=None) if good_segments else None
            if closest:
                orig_time = closest['start']
                # Word ratio'ya göre tahmini dublaj zamanı
                dub_time = orig_time * closest.get('word_ratio', 1.0)
                anchors.append((orig_time, dub_time))

        anchors = sorted(set(anchors))
        logger.info(f"🎯 Auto-generated {len(anchors)} anchor points")

    # Konuşmacı sınırlarını al
    speaker_bounds = []

    # Tek/çok konuşmacı kontrolü
    unique_speakers = set(d.get('speaker') for d in diar_segments) if diar_segments else set()

    if len(unique_speakers) <= 1:
        # TEK KONUşMACI - sadece anchor'lar kullan
        logger.info("👤 Single speaker detected - using only anchor points")
        B = sorted([O for O, _ in anchors])
    else:
        # ÇOK KONUşMACI - diarization sınırları ekle
        for i in range(len(diar_segments) - 1):
            if diar_segments[i].get('speaker') != diar_segments[i + 1].get('speaker'):
                speaker_bounds.append(diar_segments[i].get('end'))

        # B = S ∪ {O_i}
        B = sorted(set(speaker_bounds + [O for O, _ in anchors]))
        logger.info(f"👥 Multi-speaker: {len(speaker_bounds)} boundaries + {len(anchors)} anchors = {len(B)} points")

    # Piecewise linear mapping: f(t)
    def f(t: float) -> float:
        for i in range(len(anchors) - 1):
            O_i, D_i = anchors[i]
            O_next, D_next = anchors[i + 1]
            if O_i <= t <= O_next:
                if O_next - O_i > 0:
                    alpha = (D_next - D_i) / (O_next - O_i)
                    return D_i + alpha * (t - O_i)
                return D_i
        # Extrapolation
        if t > anchors[-1][0] and len(anchors) > 1:
            O_last, D_last = anchors[-1]
            O_prev, D_prev = anchors[-2]
            if O_last - O_prev > 0:
                alpha = (D_last - D_prev) / (O_last - O_prev)
                return D_last + alpha * (t - O_last)
        return t

    # Yeni segmentler
    new_segments = []
    for k in range(len(B) - 1):
        start_orig, end_orig = B[k], B[k + 1]

        # Min segment süresi (0.5s) ve Max segment süresi (10s)
        segment_duration = end_orig - start_orig
        if segment_duration < 0.5:
            continue

        # Eğer segment çok uzunsa (>10s), parçala
        if segment_duration > 10.0:
            # 10 saniyelik parçalara böl
            num_parts = int(segment_duration / 8.0) + 1
            part_duration = segment_duration / num_parts

            for part in range(num_parts):
                part_start_orig = start_orig + part * part_duration
                part_end_orig = min(start_orig + (part + 1) * part_duration, end_orig)

                # Dublaj mapping
                part_start_dub = f(part_start_orig)
                part_end_dub = f(part_end_orig)

                # Tempo hesapla
                part_orig_duration = part_end_orig - part_start_orig
                part_dub_duration = part_end_dub - part_start_dub
                part_tempo_ratio = part_dub_duration / part_orig_duration if part_orig_duration > 0 else 1.0

                # Konuşmacı bul
                speaker = 'UNKNOWN'
                if diar_segments:
                    for d in diar_segments:
                        if d['start'] <= part_start_orig < d['end']:
                            speaker = d.get('speaker', 'UNKNOWN')
                            break

                # Metni birleştir
                text = ""
                for s in segments:
                    overlap = min(part_end_orig, s.get('end', 0)) - max(part_start_orig, s.get('start', 0))
                    if overlap > 0:
                        text += s.get('text', '') + " "

                new_segments.append({
                    'id': len(new_segments),
                    'start': part_start_orig,
                    'end': part_end_orig,
                    'start_dub': part_start_dub,
                    'end_dub': part_end_dub,
                    'tempo_ratio': part_tempo_ratio,
                    'speaker': speaker,
                    'text': text.strip(),
                    'duration': part_orig_duration
                })
            continue  # Uzun segment işlendi, sonrakine geç

        # Dublaj mapping
        start_dub = f(start_orig)
        end_dub = f(end_orig)

        # Tempo hesapla: r_k = (f(b_{k+1}) - f(b_k)) / (b_{k+1} - b_k)
        orig_duration = end_orig - start_orig
        dub_duration = end_dub - start_dub
        tempo_ratio = dub_duration / orig_duration if orig_duration > 0 else 1.0

        # Konuşmacı bul
        speaker = 'UNKNOWN'
        if diar_segments:
            for d in diar_segments:
                if d['start'] <= start_orig < d['end']:
                    speaker = d.get('speaker', 'UNKNOWN')
                    break

        # Metni birleştir
        text = ""
        for s in segments:
            overlap = min(end_orig, s.get('end', 0)) - max(start_orig, s.get('start', 0))
            if overlap > 0:
                text += s.get('text', '') + " "

        new_segments.append({
            'id': k,
            'start': start_orig,
            'end': end_orig,
            'start_dub': start_dub,
            'end_dub': end_dub,
            'tempo_ratio': tempo_ratio,
            'speaker': speaker,
            'text': text.strip(),
            'duration': orig_duration
        })

    # Tempo istatistikleri
    if new_segments:
        tempos = [s['tempo_ratio'] for s in new_segments]
        avg_tempo = sum(tempos) / len(tempos)
        logger.info(f"⚛️ Tempo stats: avg={avg_tempo:.2f}, min={min(tempos):.2f}, max={max(tempos):.2f}")

        extreme = sum(1 for t in tempos if t < 0.5 or t > 2.0)
        if extreme > 0:
            logger.warning(f"⚠️ {extreme}/{len(tempos)} segments need extreme tempo!")

    return new_segments

# ==================== Segment Bazlı XTTS -> Süre Uydurma -> Birleştirme =============
def synthesize_dub_track_xtts(
        segments: List[dict],
        all_text: Optional[str],
        voices_dir: Path,
        latents_map: Dict[str, str],
        target_lang: str,
        out_dir: Path,
        xtts_cfg: Optional[XTTSConfig] = None,
        fit_to_segments: bool = False,  # TIME STRETCHING KAPALI
        use_tempo_limits: bool = True,
        aggressive_rebalance: bool = True,
        tempo_min: float = 0.5,
        tempo_max: float = 2.0
) -> Tuple[Path, Dict[int, Path]]:
    """
    Segment-based XTTS synthesis with duration matching.
    - Synthesizes TTS for each segment
    - Optionally rebalances segments to stay within tempo limits
    - Applies time stretching to match original segment duration
    - Tracks and compensates for timing drift
    """
    import logging
    logger = logging.getLogger(__name__)

    xtts_cfg = xtts_cfg or XTTSConfig(language=target_lang)
    tts = _load_xtts_engine(xtts_cfg.model_name, xtts_cfg.language)

    seg_audio: Dict[int, Path] = {}
    tmp_audio_dir = out_dir / "_tts_segments"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)

    # If using tempo limits, first estimate TTS durations and rebalance
    if use_tempo_limits and fit_to_segments:
        logger.info("Estimating TTS durations for tempo rebalancing...")

        # First, merge very short segments from same speaker
        if aggressive_rebalance:
            segments = merge_short_segments_same_speaker(segments, min_duration=1.0, max_gap=0.5)

        # Phase 1: Estimate TTS durations for all segments
        tts_durations = {}
        for seg in segments:
            sid = seg.get("id")
            if sid is None:
                continue
            text = (seg.get("text") or "").strip()
            if text:
                # Use estimation for now (could do actual TTS here for accuracy)
                estimated_duration = estimate_tts_duration(text, target_lang)
                tts_durations[sid] = estimated_duration

        # Phase 2: Aggressively rebalance segments to stay within tempo limits
        logger.info(f"Aggressively rebalancing segments for tempo limits [{tempo_min}, {tempo_max}]...")
        segments = rebalance_segments_for_tempo(
            segments=segments,
            tts_durations=tts_durations,
            tempo_min=tempo_min,
            tempo_max=tempo_max,
            max_passes=10  # More passes for aggressive optimization
        )
        logger.info(f"Rebalancing complete. {len(segments)} segments remain.")

    # Track duration statistics for monitoring
    duration_stats = {
        "total_original": 0.0,
        "total_tts": 0.0,
        "total_stretched": 0.0,
        "total_tts_natural": 0.0,  # Time stretching olmadan doğal TTS süresi
        "max_drift": 0.0,
        "segments_processed": 0,
        "tempos": []
    }

    for seg in segments:
        sid = seg.get("id")
        if sid is None:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # Get original segment duration
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        original_duration = seg_end - seg_start

        if original_duration <= 0:
            logger.warning(f"Invalid segment duration for ID {sid}: {original_duration}s")
            continue

        duration_stats["total_original"] += original_duration

        spk = str(seg.get("speaker") or "UNKNOWN")
        spk_wav = voices_dir / f"{spk}.wav"
        lat_path = latents_map.get(spk)

        # Handle very short segments specially
        if original_duration < 0.5 and len(text.strip()) < 10:
            # For very short segments, add a pause or repeat to avoid TTS producing too long audio
            logger.warning(f"Segment {sid} is very short ({original_duration:.2f}s) with text: {text}")
            # Add ellipsis to encourage TTS to produce shorter output
            if not text.strip().endswith((".", "!", "?", "...")):
                text = text.strip() + "."

        # Handle XTTS 400 token limit and check if segment needs splitting
        max_chars = 200  # Daha güvenli limit (226 yerine 200)

        # AGRESİF KARAKTER LİMİTİ: Süreye göre dinamik limit
        # Her saniye için yaklaşık 15-20 karakter (Türkçe için)
        dynamic_char_limit = int(original_duration * 20)  # 20 karakter/saniye

        # Daha kısıtı limiti kullan
        effective_limit = min(max_chars, dynamic_char_limit)

        # Eğer metin limiti aşıyorsa, kısalt
        if len(text) > effective_limit:
            logger.warning(f"Segment {sid} too long ({original_duration:.1f}s, {len(text)} chars), limit: {effective_limit} chars, will truncate")

            # Metni cümlelere böl
            sentences = []
            temp_text = text
            for sep in ['. ', '! ', '? ']:
                parts = temp_text.split(sep)
                if len(parts) > 1:
                    sentences = [p + sep.strip() for p in parts[:-1]] + [parts[-1]]
                    break
            if not sentences:
                # Virgülle böl
                sentences = text.split(', ')
                sentences = [s + ',' if i < len(sentences)-1 else s for i, s in enumerate(sentences)]

            # En uygun cümle kombinasyonunu bul
            selected_text = ""
            for i, sent in enumerate(sentences):
                if len(selected_text + sent) <= effective_limit:
                    selected_text += sent if i == 0 else " " + sent
                else:
                    break

            if not selected_text:  # Hiçbir cümle sığmadıysa ilk cümleyi kısalt
                selected_text = sentences[0][:effective_limit]

            text = selected_text.strip()
            if not text.endswith(('.',  '!', '?')):
                text = text + '.'
            logger.info(f"Segment {sid} truncated: {len(text)} chars (limit was {effective_limit})")

        elif len(text) > effective_limit:
            # Normal truncate with dynamic limit
            logger.warning(f"Segment {sid} text too long ({len(text)} chars), truncating to {effective_limit}")
            truncated = text[:effective_limit]
            # Find last sentence ending
            for sep in ['. ', '! ', '? ', ', ']:
                idx = truncated.rfind(sep)
                if idx > effective_limit * 0.7:  # At least 70% of effective length
                    text = truncated[:idx + 1]
                    break
            else:
                # No good boundary found, just truncate and add ellipsis
                text = truncated.strip() + '...'

        # Generate TTS output with error handling
        raw_out = tmp_audio_dir / f"seg_{sid:06d}.raw.wav"
        try:
            if lat_path:
                tts.synthesize(text, output_path=raw_out, latents_path=lat_path, lang=target_lang)
            else:
                if not spk_wav.exists():
                    fallback = next(iter(voices_dir.glob("*.wav")), None)
                    if fallback is None:
                        raise RuntimeError("Referans ses bulunamadı.")
                    spk_wav = fallback
                tts.synthesize(text, output_path=raw_out, speaker_wav=str(spk_wav), lang=target_lang)
        except Exception as e:
            logger.error(f"TTS failed for segment {sid}: {e}")
            logger.error(f"Text length: {len(text)} chars, text: {text[:100]}...")
            # Create silent audio as fallback directly to stretched file
            stretched = tmp_audio_dir / f"seg_{sid:06d}.fit.wav"
            _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={original_duration}",
                  "-ar", "16000", "-ac", "1", str(stretched)])
            seg_audio[sid] = stretched
            continue

        # Check if TTS generated output
        if not raw_out.exists():
            logger.warning(f"TTS output not found for segment {sid}, creating silent audio")
            # Create silent audio with original duration
            stretched = tmp_audio_dir / f"seg_{sid:06d}.fit.wav"
            _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={original_duration}",
                  "-ar", "16000", "-ac", "1", str(stretched)])
            seg_audio[sid] = stretched
            continue

        # Get TTS output duration
        try:
            tts_duration = _ffprobe_duration(raw_out)
        except Exception as e:
            logger.error(f"Failed to get duration for segment {sid}: {e}")
            # Create silent audio as fallback
            stretched = tmp_audio_dir / f"seg_{sid:06d}.fit.wav"
            _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={original_duration}",
                  "-ar", "16000", "-ac", "1", str(stretched)])
            seg_audio[sid] = stretched
            continue
        duration_stats["total_tts"] += tts_duration

        # TIME STRETCHING KAPALI - Direkt raw TTS çıktısını kullan
        # Artık .fit.wav üretmiyor, ses hızlandırma/yavaşlatma yok
        final_wav = tmp_audio_dir / f"seg_{sid:06d}.final.wav"

        # Raw TTS çıktısını direkt kullan (hızlandırma yok)
        if raw_out.exists():
            # Sadece format dönüşümü yap (hız değiştirme yok)
            _run(["ffmpeg", "-y", "-i", str(raw_out),
                  "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                  str(final_wav)])
        else:
            # TTS başarısızsa sessiz ses oluştur
            _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={original_duration}",
                  "-ar", "16000", "-ac", "1", str(final_wav)])

        # Süre bilgisi (sadece log için)
        final_duration = _ffprobe_duration(final_wav)
        natural_tempo = tts_duration / original_duration if original_duration > 0 else 1.0

        if abs(natural_tempo - 1.0) > 0.3:  # %30'dan fazla fark varsa uyar
            logger.info(f"Segment {sid}: Natural duration mismatch - TTS: {tts_duration:.2f}s, Original: {original_duration:.2f}s (ratio: {natural_tempo:.2f}x)")
            logger.info(f"  → Using natural TTS duration without speed adjustment")
        duration_stats["total_tts_natural"] += final_duration  # Stretching olmadığı için 'natural'

        # Track maximum drift
        drift = abs(final_duration - original_duration)
        if drift > duration_stats["max_drift"]:
            duration_stats["max_drift"] = drift

        seg_audio[sid] = final_wav
        duration_stats["segments_processed"] += 1

    # Log duration statistics
    if duration_stats["segments_processed"] > 0:
        logger.info(f"Duration Statistics:")
        logger.info(f"  Total Original: {duration_stats['total_original']:.2f}s")
        logger.info(f"  Total TTS: {duration_stats['total_tts']:.2f}s")
        logger.info(f"  Total TTS Natural: {duration_stats.get('total_tts_natural', duration_stats.get('total_stretched', 0)):.2f}s")
        logger.info(f"  Max Single Drift: {duration_stats['max_drift']:.2f}s")
        logger.info(f"  Natural Difference: {abs(duration_stats.get('total_tts_natural', duration_stats.get('total_stretched', 0)) - duration_stats['total_original']):.2f}s")

        # Log tempo statistics
        if duration_stats["tempos"]:
            tempos = duration_stats["tempos"]
            min_tempo = min(tempos)
            max_tempo = max(tempos)
            avg_tempo = sum(tempos) / len(tempos)
            logger.info(f"  Tempo Range: [{min_tempo:.2f}, {max_tempo:.2f}], Avg: {avg_tempo:.2f}")

            if use_tempo_limits:
                out_of_range = sum(1 for t in tempos if t < tempo_min or t > tempo_max)
                if out_of_range > 0:
                    logger.warning(f"  {out_of_range}/{len(tempos)} segments outside tempo limits")

    # Concatenate with improved timing
    total_len = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
    full_wav = out_dir / "dubbed.timeline.mono16k.wav"
    _concat_timeline_audio_with_sync(segments, seg_audio, total_len, full_wav)

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

# ========================= Main Pipeline (çeviri entegre) ==========================
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
        # ------ ÇEVİRİ ------
        do_translate: bool = True,
        translator_model: Optional[str] = "gpt-4o-mini",
        # ------ Dublaj & Lipsync ------
        do_dub: bool = True,
        target_lang: Optional[str] = None,
        xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        xtts_speed: Optional[float] = None,
        fit_to_segments: bool = False,  # TIME STRETCHING KAPALI
        do_lipsync: bool = True,
        wav2lip_repo: Optional[str] = None,
        wav2lip_checkpoint: Optional[str] = None,
        # ------ Speaker Analysis & Demucs ------
        analyze_speakers: bool = True,
        remove_overlaps: bool = True,
        use_demucs: bool = True,
        demucs_model: str = "htdemucs",
        instrumental_volume: float = 0.8,
        dubbing_volume: float = 1.0
) -> dict:
    dbg = DebugWriter(enabled=debug)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) Yerel dosya mı?
    src: Optional[Path] = None
    if video_path and not _looks_like_url(str(video_path)):
        cand = Path(video_path)
        if cand.exists():
            src = cand

    # 2) Değilse URL mi? (video_path URL ise ya da boşsa ve YOUTUBE_URL env tanımlıysa)
    if src is None:
        url_candidate: Optional[str] = None
        if _looks_like_url(str(video_path) if video_path is not None else None):
            url_candidate = str(video_path).strip()
        elif not video_path:
            env_url = os.getenv("YOUTUBE_URL", "").strip()
            if _looks_like_url(env_url):
                url_candidate = env_url

        if url_candidate:
            dl_dir = out / "_downloads"
            src = _download_via_ytdlp(url_candidate, dl_dir, dbg=dbg)
        else:
            raise FileNotFoundError(f"Girdi bulunamadı ve URL verilmedi: {video_path!r}")

        # Bu noktada src kesin Path ve mevcut


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


        # Overlap kontrolünü devre dışı bırak - boşlukları engellemek için
        merged_segments, noov_stats = enforce_no_overlap_same_speaker(merged_segments, margin=0.02, allow_overlap=True)
        if debug:
            dbg.snap("NO_OVERLAP_ENFORCED", **noov_stats)

        # ------ Speaker Analysis & Overlap Removal ------
        if analyze_speakers and merged_segments:
            analyzer = SpeakerSegmentAnalyzer(
                min_segment_duration=0.2,
                merge_gap_threshold=0.5,
                overlap_tolerance=0.1
            )

            # Analyze segments by speaker
            speaker_analyses = analyzer.analyze_segments(merged_segments)

            # Remove cross-speaker overlaps if requested
            if remove_overlaps:
                merged_segments = analyzer.remove_cross_speaker_overlaps(merged_segments)
                dbg.snap("SPEAKER_OVERLAPS_REMOVED", segment_count=len(merged_segments))

            # Export speaker analysis
            analysis_path = out / "speaker_analysis.json"
            analysis_result = analyzer.export_analysis(str(analysis_path))
            dbg.snap("SPEAKER_ANALYSIS_EXPORTED", path=str(analysis_path))

        # RTTM & speakers CSV
        rttm_path = out / f"{wav_for_diar.stem}.diarization.rttm"
        _write_rttm(diar_segments, rttm_path, uri=src.stem)
        spk_csv = out / f"{src.stem}.speakers.csv"
        with spk_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["start","end","speaker"])
            for d in diar_segments:
                w.writerow([d["start"], d["end"], d["speaker"]])
        dbg.snap("DIAR_FILES", rttm=str(rttm_path), speakers_csv=str(spk_csv))

    # ---------------- ÇEVİRİ (hedef dil seçimi ve uygulama) ----------------
    tlang = target_lang or (lang or "tr")
    translation_meta = {"translated": False}
    if do_translate and tlang:
        merged_segments, translation_meta = translate_segments(
            merged_segments, target_lang=tlang,
            model=(translator_model or llm_model),
            source_lang=lang, batch_size=30, dbg=dbg
        )
        dbg.snap("TRANSLATE_DONE", **translation_meta)

    # ---------------- DUBLaj + LİPSYNC ENTEGRASYON ----------------
    dub_audio_wav = None
    lipsync_video = None
    lipsync_used  = False

    if do_dub:
        # ANCHOR-BASED SEGMENTATION UYGULA
        anchor_segments = create_anchor_based_segments(
            segments=merged_segments,
            diar_segments=diar_segments if diar_segments else [],
            anchors=None  # Otomatik anchor oluştur
        )

        # Anchor segmentleri kullan (eğer oluşturulduysa)
        segments_to_use = anchor_segments if anchor_segments else merged_segments
        logger.info(f"Using {'anchor-based' if anchor_segments else 'original'} segments for dubbing")

        voices_dir, latents_map = build_reference_voices(
            original_audio=wav_for_diar,
            segments=segments_to_use,
            target_lang=tlang,
            workdir=out / "_work"
        )
        dub_audio_wav, seg_audio_map = synthesize_dub_track_xtts(
            segments=segments_to_use,
            all_text=" ".join(merged_texts).strip(),
            voices_dir=voices_dir,
            latents_map=latents_map,
            target_lang=tlang,
            out_dir=out,
            xtts_cfg=XTTSConfig(model_name=xtts_model_name, language=tlang, speed=xtts_speed)
        )
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

        # ------ Demucs Vocal Separation & Mixing ------
        if use_demucs and dub_audio_wav:
            separator = DemucsVocalSeparator(model=demucs_model)

            if separator.is_available:
                # Separate vocals from original audio
                demucs_output = out / "demucs_output"
                stems = separator.separate_vocals(
                    str(wav_for_diar),
                    str(demucs_output)
                )

                if stems:
                    # Mix dubbing with instrumental
                    mixer = DubbingMixer(
                        instrumental_volume=instrumental_volume,
                        dubbing_volume=dubbing_volume
                    )

                    # Create final mix with instrumental
                    final_output = out / "final_dubbed_with_music.wav"
                    mixed_path = mixer.mix_dubbing_with_instrumental(
                        str(dub_audio_wav),
                        stems["instrumental"],
                        str(final_output)
                    )
                    dbg.snap("DEMUCS_MIX_CREATED", path=str(mixed_path))

                    # Create video with the mixed audio if requested
                    if lipsync_video:
                        final_video = out / "final_dubbed_video_with_music.mp4"
                        _mux_audio_to_video(Path(lipsync_video) if isinstance(lipsync_video, str) else lipsync_video,
                                          Path(mixed_path), final_video)
                        lipsync_video = final_video
                        dbg.snap("FINAL_VIDEO_WITH_MUSIC", path=str(final_video))

                    # Also create adaptive mix if we have speaker timelines
                    if analyze_speakers and 'speaker_analyses' in locals():
                        speaker_timelines = {}
                        for speaker_id in speaker_analyses:
                            timeline = analyzer.get_speaker_timeline(speaker_id)
                            speaker_timelines[speaker_id] = timeline

                        if speaker_timelines:
                            adaptive_output = out / "adaptive_dubbed.wav"
                            adaptive_path = mixer.create_adaptive_mix(
                                str(wav_for_diar),
                                str(dub_audio_wav),
                                stems["instrumental"],
                                str(adaptive_output),
                                speaker_timelines
                            )
                            dbg.snap("ADAPTIVE_MIX_CREATED", path=str(adaptive_path))
            else:
                logger.warning("Demucs not available, skipping vocal separation")

    # ---------------- ÇIKTILAR ----------------
    last_seg_end = 0.0
    if merged_segments:
        last_seg_end = max(float(s.get("end",0.0)) for s in merged_segments)
    summary = _write_outputs(out, src.stem, merged_segments, merged_words, " ".join(merged_texts).strip(), lang, last_seg_end, diarization=diar_segments, timeline=timeline)
    summary["models"] = {"stt": stt_model, "llm": llm_model}
    summary["translation"] = {"enabled": bool(do_translate), **translation_meta, "source_lang": lang, "target_lang": tlang}
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
            "target_lang": tlang,
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
        video_path="",
        output_dir="output",
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
        min_overlap_ratio=0.5,  # Artırıldı: en az %50 örtüşme gerekli
        boundary_tolerance=0.3,  # Artırıldı: daha esnek sınır toleransı
        use_vad_boundaries=True,
        use_timeline=True,
        confidence_threshold=0.7,  # Artırıldı: daha yüksek güven eşiği
        # ÇEVİRİ
        do_translate=True,                 # <—— ÇEVİRİ ETKİN
        translator_model="google",    # Google Cloud Translate API kullan
        # DUB + LIPSYNC
        do_dub=True,
        target_lang="tr",  # hedef dil
        xtts_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        xtts_speed=1.3,   # Daha hızlı konuşma (1.3x)
        do_lipsync=True,
        wav2lip_repo=None,        # örn: "/opt/Wav2Lip"
        wav2lip_checkpoint=None,  # örn: "/opt/Wav2Lip/checkpoints/Wav2Lip.pth"
        # Speaker Analysis & Demucs
        analyze_speakers=True,
        remove_overlaps=True,
        use_demucs=True,
        demucs_model="htdemucs",
        instrumental_volume=0.8,
        dubbing_volume=1.0
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
