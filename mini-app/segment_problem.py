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

def ensure_wav_mono16k(audio_in: Path) -> Path:
    outdir = audio_in.parent
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
    numbered = "\n".join(f"{i+1}. {t or ''}" for i, (_, t) in enumerate(items))
    sys_msg = f"Sen profesyonel bir çevirmen ve altyazı yerleştirme uzmanısın. SADECE çeviri metnini döndür. Ek yorum, açıklama, numara yazma."
    if src_lang:
        sys_msg += f" Kaynak dil: {src_lang}. "
    sys_msg += f"Hedef dil: {tgt_lang}. Noktalama ve büyük/küçük harf korunmalı, zamanlama veya ID yazma."
    user_msg = f"Aşağıdaki {len(items)} satırı sırayla çevir. Her satırı kendi satırında döndür (1:1). Metinler:\n{numbered}"
    try:
        comp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
            temperature=0
        )
        rsp = comp.choices[0].message.content or ""
        out = [ln.strip() for ln in rsp.split("\n") if ln.strip() != ""]
        cleaned: List[str] = []
        for ln in out:
            cleaned.append(ln.split(" ", 1)[1].strip() if ln[:2].isdigit() and " " in ln else ln.lstrip("0123456789).:- ").strip())
        if len(cleaned) < len(items):
            cleaned += [""] * (len(items)-len(cleaned))
        return cleaned[:len(items)]
    except Exception:
        return [t for _, t in items]

def translate_segments(segments: List[dict],
                       target_lang: str,
                       model: str = "gpt-4o-mini",
                       source_lang: Optional[str] = None,
                       batch_size: int = 30,
                       dbg: Optional[DebugWriter] = None) -> Tuple[List[dict], Dict[str, Any]]:
    if not segments:
        return segments, {"translated": False}
    client = _openai_client_or_raise()
    pending: List[Tuple[int, int, str]] = []
    for i, seg in enumerate(segments):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        pending.append((i, seg.get("id", i), txt))

    if not pending:
        return segments, {"translated": False}

    total = len(pending)
    done = 0
    for k in range(0, total, batch_size):
        chunk = pending[k:k+batch_size]
        pairs = [(sid, txt) for (_, sid, txt) in chunk]
        translations = _chat_translate_batch(client, pairs, source_lang, target_lang, model=model)
        for (i, _, _), tr in zip(chunk, translations):
            orig = segments[i].get("text", "")
            segments[i]["orig_text"] = orig
            segments[i]["text"] = tr or orig
        done += len(chunk)
        if dbg: dbg.snap("TRANSLATE_BATCH", size=len(chunk), done=done, total=total)

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
                verbose["timeline"].append(t)
            elif hasattr(t, 'start'):
                verbose["timeline"].append({
                    "start": t.start,
                    "end": t.end,
                    "mode": t.mode,
                    "speakers": list(t.speakers) if hasattr(t, 'speakers') else [],
                    "channels": t.channels if hasattr(t, 'channels') else None
                })
            elif isinstance(t, tuple) and len(t) >= 2:
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
def enforce_no_overlap_same_speaker_gentle(segments: List[dict], margin: float = 0.01) -> Tuple[List[dict], Dict[str, int]]:
    """
    Aynı konuşmacının segmentlerini overlap etmeyecek şekilde düzenle.
    Küçük segmentleri koruyarak, sadece overlap kısımlarını kes.
    """
    segs = sorted(copy.deepcopy(segments), key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    by_spk: Dict[str, List[dict]] = defaultdict(list)

    for s in segs:
        spk = str(s.get("speaker") or "")
        by_spk[spk].append(s)

    trims, drops, preserves = 0, 0, 0
    MIN_SEGMENT_DURATION = 0.1  # En az 100ms'lik segmentleri koru

    for spk, ss in by_spk.items():
        ss.sort(key=lambda s: float(s.get("start", 0.0)))
        prev_end = -1e9

        for s in ss:
            st = float(s.get("start", 0.0))
            en = float(s.get("end", 0.0))
            duration = en - st

            # Overlap var mı?
            if st < prev_end:
                overlap = prev_end - st

                # Küçük segment ise korumaya çalış
                if duration < MIN_SEGMENT_DURATION * 2:
                    # Önceki segmenti kısalt, bu segmenti koru
                    if ss[ss.index(s) - 1] if s != ss[0] else None:
                        prev_seg = ss[ss.index(s) - 1]
                        prev_seg["end"] = st - margin/2
                        preserves += 1

                # Overlap küçükse sadece başlangıcı kaydir
                elif overlap < duration * 0.3:  # %30'dan az overlap
                    s["start"] = prev_end + margin
                    trims += 1

                # Büyük overlap - segmenti ikiye böl
                elif overlap < duration * 0.7:
                    # İlk yarısını at, ikinci yarısını koru
                    midpoint = (st + en) / 2
                    if midpoint > prev_end + margin:
                        s["start"] = max(prev_end + margin, midpoint)
                        trims += 1
                    else:
                        s["start"] = prev_end + margin
                        if s["start"] >= en - MIN_SEGMENT_DURATION:
                            s["_drop"] = True
                            drops += 1
                            continue
                else:
                    # Çok fazla overlap, segmenti sil
                    s["_drop"] = True
                    drops += 1
                    continue

            prev_end = float(s.get("end", 0.0))

    cleaned: List[dict] = []
    for s in segs:
        if not s.get("_drop"):
            # Son kontrol: çok kısa kalmışsa birleştir
            if cleaned and cleaned[-1].get("speaker") == s.get("speaker"):
                gap = float(s.get("start", 0)) - float(cleaned[-1].get("end", 0))
                duration = float(s.get("end", 0)) - float(s.get("start", 0))

                if gap < 0.05 and duration < MIN_SEGMENT_DURATION:
                    # Öncekiyle birleştir
                    cleaned[-1]["end"] = s.get("end", cleaned[-1]["end"])
                    if "text" in s and s["text"]:
                        cleaned[-1]["text"] = (cleaned[-1].get("text", "") + " " + s["text"]).strip()
                    continue

            cleaned.append(s)

    return cleaned, {"trimmed": trims, "dropped": drops, "preserved": preserves}


def _concat_timeline_audio_with_mixing(segments: List[dict],
                                       seg_audio_paths: Dict[int, Path],
                                       total_len: float,
                                       out_wav: Path) -> Path:
    """
    Gelişmiş ses birleştirme - overlap'larda mixing yapar.
    """
    _ensure_pydub()
    from pydub import AudioSegment

    sr = 16000
    out = AudioSegment.silent(duration=int(total_len * 1000), frame_rate=sr)

    # Parametreler
    CROSSFADE_MS = 50  # Daha uzun crossfade
    MIN_GAP_MS = 10    # Minimum boşluk
    OVERLAP_MIX_RATIO = 0.7  # Overlap'ta eski sesin oranı

    sorted_segments = sorted(segments, key=lambda x: float(x.get("start", 0)))
    placed_segments = []  # Yerleştirilen segmentleri takip et

    for seg in sorted_segments:
        sid = seg.get("id")
        if sid is None or sid not in seg_audio_paths:
            continue

        # Ses dosyasını yükle
        wav = AudioSegment.from_file(seg_audio_paths[sid]).set_frame_rate(sr).set_channels(1)

        # Hedef pozisyon
        start_ms = int(float(seg.get("start", 0)) * 1000)
        end_ms = start_ms + len(wav)

        # Overlap kontrolü
        overlapping = []
        for placed in placed_segments:
            p_start, p_end = placed["start_ms"], placed["end_ms"]
            if not (end_ms <= p_start or start_ms >= p_end):
                overlapping.append(placed)

        if overlapping:
            # Overlap var - mixing yap
            for overlap_seg in overlapping:
                overlap_start = max(start_ms, overlap_seg["start_ms"])
                overlap_end = min(end_ms, overlap_seg["end_ms"])
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0:
                    # Crossfade ile yumuşak geçiş
                    if overlap_duration < CROSSFADE_MS * 2:
                        # Kısa overlap - tam crossfade
                        try:
                            fade_ms = max(1, int(overlap_duration / 2))
                            if fade_ms > 0 and len(wav) > fade_ms * 2:
                                wav = wav.fade_in(duration=fade_ms).fade_out(duration=fade_ms)
                        except:
                           pass  # Fade başarısız olursa devam et
                    else:
                        # Uzun overlap - başta ve sonda crossfade
                        wav = wav.fade_in(CROSSFADE_MS)

                    # Volume ayarı - overlap'ta sesi biraz kıs
                    wav = wav - 3  # 3dB azalt

        # Fade in/out ekle (yumuşak giriş/çıkış)
        wav_length_ms = len(wav)
        if wav_length_ms > 0:
            fade_duration = min(CROSSFADE_MS, wav_length_ms // 4)
            if fade_duration > 0:
                fade_duration = int(fade_duration)  # Integer'a çevir
                wav = wav.fade_in(fade_duration).fade_out(fade_duration)

        # Sesi yerleştir - overlay yerine daha akıllı mixing
        if overlapping:
            # Overlap varsa karıştır (mix)
            out = out.overlay(wav, position=start_ms, gain_during_overlay=-6)
        else:
            # Overlap yoksa direkt yerleştir
            out = out.overlay(wav, position=start_ms)

        # Yerleştirilen segmenti kaydet
        placed_segments.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speaker": seg.get("speaker", "UNKNOWN"),
            "id": sid
        })

    # Normalizasyon - clipping'i önle
    if out.max_dBFS > -3:
        out = out.normalize()

    # Son silence trim
    out = out.strip_silence(silence_len=100, silence_thresh=-50)

    out.export(out_wav, format="wav")
    return out_wav


def _smart_overlap_handler(segments: List[dict]) -> List[dict]:
    """
    Overlap'ları akıllıca yönet - küçük konuşmaları koru.
    """
    import numpy as np

    # Önce tüm overlap'ları tespit et
    overlaps = []
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments[i+1:], i+1):
            s1_start, s1_end = float(seg1.get("start", 0)), float(seg1.get("end", 0))
            s2_start, s2_end = float(seg2.get("start", 0)), float(seg2.get("end", 0))

            if s1_end > s2_start:  # Overlap var
                overlap_duration = min(s1_end, s2_end) - s2_start
                overlaps.append({
                    "seg1_idx": i,
                    "seg2_idx": j,
                    "overlap": overlap_duration,
                    "seg1_duration": s1_end - s1_start,
                    "seg2_duration": s2_end - s2_start
                })

    # Overlap'ları önem sırasına göre çöz
    for ov in sorted(overlaps, key=lambda x: x["overlap"], reverse=True):
        seg1 = segments[ov["seg1_idx"]]
        seg2 = segments[ov["seg2_idx"]]

        # Küçük girişleri koru
        if ov["seg2_duration"] < 0.5:  # 500ms'den kısa
            # Küçük segment - öncekini kısalt
            seg1["end"] = seg2.get("start", seg1["end"]) - 0.01
        elif ov["seg1_duration"] < 0.5:
            # İlk segment küçük - ikincinin başını kaydir
            seg2["start"] = seg1.get("end", seg2["start"]) + 0.01
        else:
            # İkisi de büyük - overlap'ı paylaş
            midpoint = (seg1["end"] + seg2["start"]) / 2
            seg1["end"] = midpoint - 0.005
            seg2["start"] = midpoint + 0.005

    return segments
def _smart_overlap_handler(segments: List[dict]) -> List[dict]:
    """
    Overlap'ları akıllıca yönet - küçük konuşmaları koru.
    """
    import numpy as np

    # Önce tüm overlap'ları tespit et
    overlaps = []
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments[i+1:], i+1):
            s1_start, s1_end = float(seg1.get("start", 0)), float(seg1.get("end", 0))
            s2_start, s2_end = float(seg2.get("start", 0)), float(seg2.get("end", 0))

            if s1_end > s2_start:  # Overlap var
                overlap_duration = min(s1_end, s2_end) - s2_start
                overlaps.append({
                    "seg1_idx": i,
                    "seg2_idx": j,
                    "overlap": overlap_duration,
                    "seg1_duration": s1_end - s1_start,
                    "seg2_duration": s2_end - s2_start
                })

    # Overlap'ları önem sırasına göre çöz
    for ov in sorted(overlaps, key=lambda x: x["overlap"], reverse=True):
        seg1 = segments[ov["seg1_idx"]]
        seg2 = segments[ov["seg2_idx"]]

        # Küçük girişleri koru
        if ov["seg2_duration"] < 0.5:  # 500ms'den kısa
            # Küçük segment - öncekini kısalt
            seg1["end"] = seg2.get("start", seg1["end"]) - 0.01
        elif ov["seg1_duration"] < 0.5:
            # İlk segment küçük - ikincinin başını kaydir
            seg2["start"] = seg1.get("end", seg2["start"]) + 0.01
        else:
            # İkisi de büyük - overlap'ı paylaş
            midpoint = (seg1["end"] + seg2["start"]) / 2
            seg1["end"] = midpoint - 0.005
            seg2["start"] = midpoint + 0.005

    return segments

def _smart_overlap_handler(segments: List[dict]) -> List[dict]:
    """
    Overlap'ları akıllıca yönet - küçük konuşmaları koru.
    """
    import numpy as np

    # Önce tüm overlap'ları tespit et
    overlaps = []
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments[i+1:], i+1):
            s1_start, s1_end = float(seg1.get("start", 0)), float(seg1.get("end", 0))
            s2_start, s2_end = float(seg2.get("start", 0)), float(seg2.get("end", 0))

            if s1_end > s2_start:  # Overlap var
                overlap_duration = min(s1_end, s2_end) - s2_start
                overlaps.append({
                    "seg1_idx": i,
                    "seg2_idx": j,
                    "overlap": overlap_duration,
                    "seg1_duration": s1_end - s1_start,
                    "seg2_duration": s2_end - s2_start
                })

    # Overlap'ları önem sırasına göre çöz
    for ov in sorted(overlaps, key=lambda x: x["overlap"], reverse=True):
        seg1 = segments[ov["seg1_idx"]]
        seg2 = segments[ov["seg2_idx"]]

        # Küçük girişleri koru
        if ov["seg2_duration"] < 0.5:  # 500ms'den kısa
            # Küçük segment - öncekini kısalt
            seg1["end"] = seg2.get("start", seg1["end"]) - 0.01
        elif ov["seg1_duration"] < 0.5:
            # İlk segment küçük - ikincinin başını kaydir
            seg2["start"] = seg1.get("end", seg2["start"]) + 0.01
        else:
            # İkisi de büyük - overlap'ı paylaş
            midpoint = (seg1["end"] + seg2["start"]) / 2
            seg1["end"] = midpoint - 0.005
            seg2["start"] = midpoint + 0.005

    return segments

# ================== DEMUX & MÜZİK YATAĞI & DUB MİX ==================
def _extract_original_audio(video_in: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{video_in.stem}.orig.48k.wav"
    _run(["ffmpeg", "-y", "-i", str(video_in), "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(out)])
    return out

def _extract_music_bed(video_in: Path, workdir: Path, dbg: Optional[DebugWriter] = None) -> Tuple[Path, bool]:
    workdir.mkdir(parents=True, exist_ok=True)
    orig = _extract_original_audio(video_in, workdir)

    # Demucs
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

    # Spleeter
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

    # Fallback
    if dbg: dbg.snap("MUSIC_BED", method="original_audio_fallback", file=str(orig))
    return orig, False

# ---------------------- FFmpeg filtre tespiti ----------------------
def _ffmpeg_has_filter(filter_name: str) -> bool:
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

# ---------------------- Time-stretch (kalite odaklı) ----------------------
def _ffprobe_duration(path: Path) -> float:
    return probe_duration_seconds(path)

def _atempo_chain(speed: float) -> str:
    """
    FFmpeg atempo filtresi için zincirleme oluştur.
    Atempo 0.5-2.0 aralığında çalışır, bu yüzden zincirleme gerekli.
    
    speed < 1: yavaşlatma (uzatma)
    speed > 1: hızlandırma (kısaltma)
    """
    if speed <= 0:
        speed = 1.0
    
    chain: List[float] = []
    remaining_speed = speed
    
    # Yavaşlatma için (speed < 1)
    while remaining_speed < 0.5:
        chain.append(0.5)
        remaining_speed = remaining_speed / 0.5  # Kalan hızı güncelle
    
    # Hızlandırma için (speed > 1)
    while remaining_speed > 2.0:
        chain.append(2.0)
        remaining_speed = remaining_speed / 2.0  # Kalan hızı güncelle
    
    # Son kalan değeri ekle
    if 0.5 <= remaining_speed <= 2.0:
        chain.append(remaining_speed)
    
    return ",".join(f"atempo={x:.6f}" for x in chain)


def _time_stretch_to_duration(in_wav: Path, target_sec: float, out_wav: Path, enable: bool = True) -> Path:
    """
    Ses dosyasını belirtilen süreye esnet/sıkıştır.
    
    Args:
        in_wav: Giriş WAV dosyası
        target_sec: Hedef süre (saniye)
        out_wav: Çıkış WAV dosyası
        enable: Time stretch etkinleştirme
    """
    logger = logging.getLogger(__name__)
    
    # Stretch devre dışıysa sadece format dönüşümü yap
    if not enable or target_sec <= 0:
        _run(["ffmpeg", "-y", "-i", str(in_wav), "-ar", "16000", "-ac", "1", str(out_wav)])
        return out_wav
    
    current_duration = _ffprobe_duration(in_wav)
    
    # Zaten hedef süreye çok yakınsa, sadece trim/pad yap
    if abs(current_duration - target_sec) < 0.05:
        _run(["ffmpeg", "-y", "-i", str(in_wav),
              "-af", f"aresample=16000",
              "-t", str(target_sec),
              "-ac", "1", str(out_wav)])
        return out_wav
    
    # DOĞRU TEMPO HESAPLAMASI:
    # tempo = hedef_hız / mevcut_hız
    # Eğer 10 saniyelik sesi 5 saniyeye sığdırmak istiyorsak: tempo = 2.0 (hızlandır)
    # Eğer 5 saniyelik sesi 10 saniyeye uzatmak istiyorsak: tempo = 0.5 (yavaşlat)
    tempo = target_sec / current_duration
    
    logger.info(f"Time stretch: {current_duration:.3f}s -> {target_sec:.3f}s (tempo={tempo:.3f})")
    
    # Küçük ayarlamalar için sadece atempo kullan
    if 0.9 <= tempo <= 1.1:
        atempo_filter = _atempo_chain(1.0 / tempo)  # FFmpeg için ters tempo
        cmd = ["ffmpeg", "-y", "-i", str(in_wav),
               "-af", f"{atempo_filter},aresample=16000",
               "-t", str(target_sec),
               "-ac", "1", str(out_wav)]
        _run(cmd)
        
    # Orta seviye ayarlamalar için atempo veya rubberband
    elif 0.5 <= tempo <= 2.0:
        if _ffmpeg_has_filter("rubberband"):
            # Rubberband daha kaliteli sonuç verir
            cmd = ["ffmpeg", "-y", "-i", str(in_wav),
                   "-af", f"rubberband=tempo={1.0/tempo:.6f}:pitch=1.0,aresample=16000",
                   "-t", str(target_sec),
                   "-ac", "1", str(out_wav)]
        else:
            # Fallback to atempo
            atempo_filter = _atempo_chain(1.0 / tempo)
            cmd = ["ffmpeg", "-y", "-i", str(in_wav),
                   "-af", f"{atempo_filter},aresample=16000",
                   "-t", str(target_sec),
                   "-ac", "1", str(out_wav)]
        _run(cmd)
        
    # Ekstrem ayarlamalar için çok aşamalı işlem
    else:
        temp_wav = out_wav.parent / f"{out_wav.stem}_temp.wav"
        
        # İlk aşama: maksimum güvenli tempo uygula
        if tempo < 0.5:
            # Çok fazla uzatma gerekiyor
            first_tempo = 2.0  # İlk aşamada 2x yavaşlat
            remaining_tempo = tempo * 2.0  # Kalan tempo
        else:
            # Çok fazla hızlandırma gerekiyor  
            first_tempo = 0.5  # İlk aşamada 2x hızlandır
            remaining_tempo = tempo / 0.5  # Kalan tempo
            
        # İlk aşama
        atempo1 = _atempo_chain(1.0 / first_tempo)
        cmd1 = ["ffmpeg", "-y", "-i", str(in_wav),
                "-af", f"{atempo1},aresample=48000",
                "-ac", "1", str(temp_wav)]
        _run(cmd1)
        
        # İkinci aşama
        atempo2 = _atempo_chain(1.0 / remaining_tempo)
        cmd2 = ["ffmpeg", "-y", "-i", str(temp_wav),
                "-af", f"{atempo2},aresample=16000",
                "-t", str(target_sec),
                "-ac", "1", str(out_wav)]
        _run(cmd2)
        
        # Temp dosyayı temizle
        if temp_wav.exists():
            temp_wav.unlink()
    
    # Sonuç süresini kontrol et
    final_duration = _ffprobe_duration(out_wav)
    duration_error = abs(final_duration - target_sec)
    
    if duration_error > 0.1:
        logger.warning(f"Time stretch doğruluk sorunu: hedef={target_sec:.3f}s, sonuç={final_duration:.3f}s, hata={duration_error:.3f}s")
        
        # Eğer hata çok büyükse, basit trim/pad ile düzelt
        if duration_error > 0.5:
            logger.info("Büyük hata tespit edildi, trim/pad ile düzeltiliyor...")
            temp2_wav = out_wav.parent / f"{out_wav.stem}_fix.wav"
            if final_duration > target_sec:
                # Fazla uzunsa kes
                _run(["ffmpeg", "-y", "-i", str(out_wav),
                      "-t", str(target_sec),
                      "-c:a", "pcm_s16le", str(temp2_wav)])
            else:
                # Kısaysa pad ekle
                _run(["ffmpeg", "-y", "-i", str(out_wav),
                      "-af", f"apad=pad_dur={target_sec - final_duration}",
                      "-c:a", "pcm_s16le", str(temp2_wav)])
            
            # Orijinali değiştir
            if temp2_wav.exists():
                import shutil
                shutil.move(str(temp2_wav), str(out_wav))
    
    return out_wav


def _time_stretch_precise(in_wav: Path, target_sec: float, out_wav: Path) -> Path:
    """
    Alternatif: Sox kullanarak daha hassas time stretching (eğer sox kuruluysa).
    """
    if shutil.which("sox"):
        current_duration = _ffprobe_duration(in_wav)
        tempo_factor = target_sec / current_duration
        
        try:
            # Sox ile tempo ayarla
            _run(["sox", str(in_wav), str(out_wav), 
                  "tempo", str(tempo_factor),
                  "rate", "16000",
                  "channels", "1"])
            
            # Süreyi kontrol et
            final_duration = _ffprobe_duration(out_wav)
            if abs(final_duration - target_sec) < 0.05:
                return out_wav
        except Exception as e:
            logging.warning(f"Sox başarısız, FFmpeg'e dönülüyor: {e}")
    
    # Fallback to FFmpeg
    return _time_stretch_to_duration(in_wav, target_sec, out_wav, enable=True)

# ---------------------- Dublaj + Müzik Miks (DONMA FIX'li) ----------------------
def _mix_music_and_dub(
        video_in: Path,
        dub_audio_wav: Path,
        out_dir: Path,
        dbg: Optional[DebugWriter] = None,
        dub_gain_db: float = 0.0,
        music_gain_db: float = -2.0,
) -> Tuple[Path, Optional[Path]]:
    """
    - amix=duration=shortest -> mix, en kısa akıma hizalanır (video aşımı kalmaz).
    - ardından videoya göre kesin trim; donmaları önler.
    """
    mixdir = out_dir / "_mix"
    mixdir.mkdir(parents=True, exist_ok=True)
    music_bed, separated = _extract_music_bed(video_in, mixdir, dbg)

    # Dub'u 48k/stereo normalize et
    dub48 = mixdir / f"{Path(dub_audio_wav).stem}.48k.stereo.wav"
    _run(["ffmpeg", "-y", "-i", str(dub_audio_wav), "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(dub48)])

    final_raw = out_dir / f"{video_in.stem}.final_mix.48k.wav"

    music_pre = "aformat=sample_rates=48000:channel_layouts=stereo"
    if not separated:
        music_pre += ",firequalizer=gain_entry='entry(300,-6);entry(1000,-8);entry(3000,-6)'"
    music_pre += f",volume={10**(music_gain_db/20):.6f}"

    # DİKKAT: duration=shortest
    filter_complex = (
        f"[0:a]{music_pre}[m];"
        f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={10**(dub_gain_db/20):.6f}[v];"
        f"[m][v]sidechaincompress=threshold=0.050:ratio=12:attack=15:release=250[duck];"
        f"[duck][v]amix=inputs=2:normalize=0:duration=shortest,afade=t=out:st=0:d=0.01[mix]"
    )

    _run([
        "ffmpeg","-y",
        "-i", str(music_bed),
        "-i", str(dub48),
        "-filter_complex", filter_complex,
        "-map", "[mix]",
        "-ar","48000","-ac","2",
        "-c:a","pcm_s16le",
        str(final_raw)
    ])

    # Videoya göre kesin trim
    final = out_dir / f"{video_in.stem}.final_mix.48k.trimmed.wav"
    _trim_audio_to_video_length(final_raw, video_in, final, safety_ms=10)

    if dbg: dbg.snap("DUB_MIX_DONE", final=str(final), separated_music=bool(separated))
    return final, (music_bed if separated else None)

# ======================= XTTS & Lipsync Yardımcıları =======================
def _looks_like_url(s: str | None) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")

def _ensure_ytdlp() -> str:
    for bin_name in ("yt-dlp", "youtube-dl"):
        if shutil.which(bin_name) is not None:
            return bin_name
    raise RuntimeError("Ne 'yt-dlp' ne de 'youtube-dl' bulunamadı. Lütfen kur: pip install yt-dlp")

def _download_via_ytdlp(url: str, outdir: Path, *, prefer_mp4: bool = True, dbg: Optional[DebugWriter] = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ytdlp = _ensure_ytdlp()
    template = "%(title).200B-%(id)s.%(ext)s"

    base_cmd = [
        ytdlp,
        "--no-playlist",
        "--no-progress",
        "--newline",
        "-N", "4",
        "-P", f"home:{outdir}",
        "-o", template,
        "--no-part",
        "--retries", "10",
        "--fragment-retries", "10",
    ]
    if prefer_mp4:
        base_cmd += ["--remux-video", "mp4", "--merge-output-format", "mp4"]

    if dbg: dbg.snap("YTDLP_START", url=url, outdir=str(outdir))

    try:
        cmd = base_cmd + ["--print", "after_move:filepath", url]
        cp = _run(cmd)
        lines = [ln.strip() for ln in cp.stdout.decode(errors="ignore").splitlines() if ln.strip()]
        for ln in reversed(lines):
            p = Path(ln)
            if p.exists():
                if dbg: dbg.snap("YTDLP_DONE", file=str(p), size_bytes=p.stat().st_size)
                return p
    except subprocess.CalledProcessError:
        pass

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
        pass

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
    """
    Geliştirilmiş FallbackXTTSEngine - detaylı CUDA loglama ile.
    """
    def __init__(self, model_name: str="tts_models/multilingual/multi-dataset/xtts_v2", language: str="tr"):
        self.model_name = model_name
        self.language = language
        self.device = "cuda"  # Default device

        logger_xtts.info(f"[FallbackXTTSEngine] Başlatılıyor...")

        try:
            from TTS.api import TTS
        except ImportError as e:
            logger_xtts.error(f"[FallbackXTTSEngine] TTS kütüphanesi bulunamadı: {e}")
            raise

        # Model yükle
        logger_xtts.info(f"[FallbackXTTSEngine] Model yükleniyor: {model_name}")
        self._tts = TTS(model_name)

        # CUDA kontrolü ve GPU'ya taşıma
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                logger_xtts.info(f"[FallbackXTTSEngine] 🎮 CUDA bulundu!")
                logger_xtts.info(f"  ├─ GPU: {gpu_name}")
                logger_xtts.info(f"  ├─ Bellek: {gpu_mem_gb:.1f} GB")
                logger_xtts.info(f"  └─ Model GPU'ya taşınıyor...")

                self._tts.to("cuda")

                # Taşıma sonrası kontrol
                if hasattr(self._tts, 'synthesizer') and hasattr(self._tts.synthesizer, 'tts_model'):
                    model_device = next(self._tts.synthesizer.tts_model.parameters()).device
                    logger_xtts.info(f"[FallbackXTTSEngine] ✅ Model başarıyla {model_device} üzerinde")
                else:
                    logger_xtts.info(f"[FallbackXTTSEngine] ✅ Model GPU'ya taşındı")

            else:
                logger_xtts.warning("[FallbackXTTSEngine] ⚠️ CUDA bulunamadı, CPU kullanılacak")
                logger_xtts.info("[FallbackXTTSEngine] İpucu: CUDA kurulumu için:")
                logger_xtts.info("  └─ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

        except Exception as e:
            logger_xtts.error(f"[FallbackXTTSEngine] ❌ CUDA başlatma hatası: {e}")
            logger_xtts.info("[FallbackXTTSEngine] CPU'da devam ediliyor...")
            self.device = "cpu"

        logger_xtts.info(f"[FallbackXTTSEngine] Son durum: {self.device.upper()} kullanılıyor")

    def synthesize(self, text: str, output_path: Path, speaker_wav: str|None=None,
                   latents_path: str|None=None, speed: float|None=None, lang: str|None=None) -> Path:
        """
        TTS sentezleme - hangi yöntemin kullanıldığını logla.
        """
        lang = (lang or self.language or "tr")

        if latents_path:
            logger_xtts.debug(f"[Synth] Latents kullanılıyor: {latents_path}")
        elif speaker_wav:
            logger_xtts.debug(f"[Synth] Referans ses kullanılıyor: {speaker_wav}")
        else:
            logger_xtts.debug(f"[Synth] Default ses kullanılıyor")

        # ... geri kalan synthesize kodu ...

        return Path(output_path)

# ---------------------- Anchor'lı Birleştirme (DONMA DOSTU) ----------------------
def _ensure_pydub():
    if AudioSegment is None:
        raise RuntimeError("pydub gerekli (pip install pydub)")


# ---------------------- Audio'yu videoya göre KES (DONMA ANAHTARI) ----------------------
def _trim_audio_to_video_length(audio_in: Path, video_in: Path, audio_out: Path, safety_ms: int = 10) -> Path:
    """
    audio_in'i video_in süresine kadar TRIM'ler (küçük güvenlik marjıyla).
    Sonda donmayı kesin engeller. Çıkış WAV/PCM (tek encode noktası mux).
    """
    vid_dur = probe_duration_seconds(video_in)
    if vid_dur <= 0:
        _run(["ffmpeg","-y","-i",str(audio_in),"-c","copy",str(audio_out)])
        return audio_out

    target = max(0.0, vid_dur - (safety_ms/1000.0))
    _run([
        "ffmpeg","-y",
        "-i", str(audio_in),
        "-af", f"atrim=0:{target:.6f},asetpts=N/SR/TB",
        "-ar","48000","-ac","2",
        "-c:a","pcm_s16le",
        str(audio_out)
    ])
    return audio_out

# ---------------------- Segment bazlı XTTS + süre uydurma ----------------------
@dataclass
class XTTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "tr"
    speed: Optional[float] = None  # None -> otomatik süre esnetme

import logging
import torch

# Logger'ı ayarla
logger_xtts = logging.getLogger("miniapp.xtts")
logger_xtts.setLevel(logging.INFO)

def _load_xtts_engine(model_name: str, language: str):
    """
    XTTS engine yükle - önce proje engine'i dene, yoksa fallback kullan.
    Detaylı loglama ile hangi engine ve device kullanıldığını göster.
    """
    logger_xtts.info("=" * 60)
    logger_xtts.info("XTTS ENGINE YÜKLEME BAŞLADI")
    logger_xtts.info(f"Model: {model_name}")
    logger_xtts.info(f"Dil: {language}")

    # CUDA durumunu kontrol et
    cuda_info = "CUDA Durumu: "
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_info += f"✓ Kullanılabilir ({device_count} GPU)"
            cuda_info += f"\n  └─ GPU: {device_name}"
            cuda_info += f"\n  └─ Bellek: {memory_gb:.1f} GB"
        else:
            cuda_info += "✗ Bulunamadı (CPU kullanılacak)"
    except Exception as e:
        cuda_info += f"✗ Kontrol hatası: {e}"

    logger_xtts.info(cuda_info)
    logger_xtts.info("-" * 60)

    # Önce ProjectXTTSEngine'i dene
    if _ProjectXTTSEngine is not None:
        try:
            logger_xtts.info("🚀 ProjectXTTSEngine kullanılıyor (özel/optimized)")
            engine = _ProjectXTTSEngine(model_name=model_name, language=language)

            # Engine'in hangi device'da olduğunu kontrol et
            if hasattr(engine, 'device'):
                logger_xtts.info(f"  └─ Device: {engine.device}")
            elif hasattr(engine, 'model') and hasattr(engine.model, 'device'):
                logger_xtts.info(f"  └─ Device: {engine.model.device}")

            logger_xtts.info("✅ ProjectXTTSEngine başarıyla yüklendi")
            logger_xtts.info("=" * 60)
            return engine

        except Exception as e:
            logger_xtts.warning(f"⚠️ ProjectXTTSEngine yüklenemedi: {e}")
            logger_xtts.info("Fallback engine'e geçiliyor...")
    else:
        logger_xtts.info("ℹ️ ProjectXTTSEngine bulunamadı")

    # Fallback olarak _FallbackXTTSEngine kullan
    logger_xtts.info("🔧 FallbackXTTSEngine kullanılıyor (TTS kütüphanesi)")

    try:
        engine = _FallbackXTTSEngine(model_name=model_name, language=language)

        # Fallback engine'in device'ını kontrol et
        if hasattr(engine, 'device'):
            logger_xtts.info(f"  └─ Device: {engine.device}")
        elif hasattr(engine, '_tts'):
            # TTS nesnesinin device'ını kontrol etmeye çalış
            try:
                if hasattr(engine._tts, 'device'):
                    logger_xtts.info(f"  └─ Device: {engine._tts.device}")
                elif torch.cuda.is_available():
                    logger_xtts.info("  └─ Device: muhtemelen CUDA")
                else:
                    logger_xtts.info("  └─ Device: CPU")
            except:
                logger_xtts.info("  └─ Device: belirlenemedi")

        logger_xtts.info("✅ FallbackXTTSEngine başarıyla yüklendi")

    except Exception as e:
        logger_xtts.error(f"❌ FallbackXTTSEngine de yüklenemedi: {e}")
        raise RuntimeError(f"Hiçbir XTTS engine yüklenemedi: {e}")

    logger_xtts.info("=" * 60)
    return engine

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

def synthesize_dub_track_xtts(
        segments: List[dict],
        all_text: Optional[str],
        voices_dir: Path,
        latents_map: Dict[str, str],
        target_lang: str,
        out_dir: Path,
        xtts_cfg: Optional[XTTSConfig] = None,
        fit_to_segments: bool = True
) -> Tuple[Path, Dict[int, Path]]:
    import logging
    logger = logging.getLogger(__name__)

    xtts_cfg = xtts_cfg or XTTSConfig(language=target_lang)
    tts = _load_xtts_engine(xtts_cfg.model_name, xtts_cfg.language)

    seg_audio: Dict[int, Path] = {}
    tmp_audio_dir = out_dir / "_tts_segments"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)

    duration_stats = {
        "total_original": 0.0,
        "total_tts": 0.0,
        "total_stretched": 0.0,
        "max_drift": 0.0,
        "segments_processed": 0
    }

    for seg in segments:
        sid = seg.get("id")
        if sid is None:
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue

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

        raw_out = tmp_audio_dir / f"seg_{sid:06d}.raw.wav"
        if lat_path:
            tts.synthesize(text, output_path=raw_out, latents_path=lat_path, lang=target_lang)
        else:
            if not spk_wav.exists():
                fallback = next(iter(voices_dir.glob("*.wav")), None)
                if fallback is None:
                    raise RuntimeError("Referans ses bulunamadı.")
                spk_wav = fallback
            tts.synthesize(text, output_path=raw_out, speaker_wav=str(spk_wav), lang=target_lang)

        tts_duration = _ffprobe_duration(raw_out)
        duration_stats["total_tts"] += tts_duration

        stretched = tmp_audio_dir / f"seg_{sid:06d}.fit.wav"

        if fit_to_segments:
            target_duration = original_duration
            _time_stretch_to_duration(raw_out, target_sec=target_duration, out_wav=stretched, enable=True)
            stretch_ratio = tts_duration / target_duration
            if abs(stretch_ratio - 1.0) > 0.1:
                logger.info(f"Segment {sid} ({spk}): TTS {tts_duration:.2f}s -> Target {target_duration:.2f}s (ratio: {stretch_ratio:.2f})")
        else:
            _time_stretch_to_duration(raw_out, target_sec=0.0, out_wav=stretched, enable=False)

        final_duration = _ffprobe_duration(stretched)
        duration_stats["total_stretched"] += final_duration

        drift = abs(final_duration - original_duration)
        if drift > duration_stats["max_drift"]:
            duration_stats["max_drift"] = drift

        seg_audio[sid] = stretched
        duration_stats["segments_processed"] += 1

    if duration_stats["segments_processed"] > 0:
        logger.info(f"Duration Statistics:")
        logger.info(f"  Total Original: {duration_stats['total_original']:.2f}s")
        logger.info(f"  Total TTS: {duration_stats['total_tts']:.2f}s")
        logger.info(f"  Total Stretched: {duration_stats['total_stretched']:.2f}s")
        logger.info(f"  Max Single Drift: {duration_stats['max_drift']:.2f}s")
        logger.info(f"  Overall Drift: {abs(duration_stats['total_stretched'] - duration_stats['total_original']):.2f}s")

    total_len = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
    full_wav = out_dir / "dubbed.timeline.mono16k.wav"
    _concat_timeline_audio_with_mixing(segments, seg_audio, total_len, full_wav)

    return full_wav, seg_audio

# ==================== Basit Lipsync (Wav2Lip) + Fallback Mux ======================
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

def _mux_audio_to_video(video_in: Path, audio_in: Path, video_out: Path) -> Path:
    _require_ffmpeg()
    if not video_in.exists():
        raise FileNotFoundError(f"Video file not found: {video_in}")
    if not audio_in.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_in}")

    video_out.parent.mkdir(parents=True, exist_ok=True)

    _run([
        "ffmpeg","-y",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-map","0:v:0","-map","1:a:0",
        "-c:v","copy",
        "-c:a","aac","-b:a","192k",
        "-movflags","+faststart",
        "-shortest",
        str(video_out)
    ])
    return video_out

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

    # Fallback: mux öncesi güvenli trim
    safe_dir = out_dir / "_sync"
    safe_dir.mkdir(parents=True, exist_ok=True)
    trimmed_for_mux = safe_dir / "dub.trimmed.for_mux.wav"
    _trim_audio_to_video_length(dub_audio_wav, video_in, trimmed_for_mux, safety_ms=10)

    out_path = _mux_audio_to_video(video_in, trimmed_for_mux, muxed)
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
        fit_to_segments: bool = False,
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

    # Kaynak video (yerel/URL)
    src: Optional[Path] = None
    if video_path and not _looks_like_url(str(video_path)):
        cand = Path(video_path)
        if cand.exists():
            src = cand

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

    wav_for_diar = ensure_wav_mono16k(audio)
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

        merged_segments, noov_stats = enforce_no_overlap_same_speaker_gentle(merged_segments, margin=0.02)
        if debug:
            dbg.snap("NO_OVERLAP_ENFORCED", **noov_stats)

        if analyze_speakers and merged_segments:
            analyzer = SpeakerSegmentAnalyzer(
                min_segment_duration=0.2,
                merge_gap_threshold=0.5,
                overlap_tolerance=0.1
            )
            speaker_analyses = analyzer.analyze_segments(merged_segments)
            if remove_overlaps:
                merged_segments = analyzer.remove_cross_speaker_overlaps(merged_segments)
                dbg.snap("SPEAKER_OVERLAPS_REMOVED", segment_count=len(merged_segments))

            analysis_path = out / "speaker_analysis.json"
            analyzer.export_analysis(str(analysis_path))
            dbg.snap("SPEAKER_ANALYSIS_EXPORTED", path=str(analysis_path))

        rttm_path = out / f"{wav_for_diar.stem}.diarization.rttm"
        _write_rttm(diar_segments, rttm_path, uri=src.stem)
        spk_csv = out / f"{src.stem}.speakers.csv"
        with spk_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["start","end","speaker"])
            for d in diar_segments:
                w.writerow([d["start"], d["end"], d["speaker"]])
        dbg.snap("DIAR_FILES", rttm=str(rttm_path), speakers_csv=str(spk_csv))

    # ---------------- ÇEVİRİ ----------------
    tlang = target_lang or (lang or "tr")
    translation_meta = {"translated": False}
    if do_translate and tlang:
        merged_segments, translation_meta = translate_segments(
            merged_segments, target_lang=tlang,
            model=(translator_model or llm_model),
            source_lang=lang, batch_size=30, dbg=dbg
        )
        dbg.snap("TRANSLATE_DONE", **translation_meta)

    # ---------------- DUBLaj + LİPSYNC ----------------
    dub_audio_wav = None
    lipsync_video = None
    lipsync_used  = False

    if do_dub:
        voices_dir, latents_map = build_reference_voices(
            original_audio=wav_for_diar,
            segments=merged_segments,
            target_lang=tlang,
            workdir=out / "_work"
        )
        dub_audio_wav, seg_audio_map = synthesize_dub_track_xtts(
            segments=merged_segments,
            all_text=" ".join(merged_texts).strip(),
            voices_dir=voices_dir,
            latents_map=latents_map,
            target_lang=tlang,
            out_dir=out,
            xtts_cfg=XTTSConfig(model_name=xtts_model_name, language=tlang, speed=xtts_speed)
        )

        # Ek güvenlik: video süresine göre kırp (pozitif offset korunur, sadece sondan kısaltır)
        safe_dub = out / "dubbed.timeline.mono16k.safe.wav"
        _trim_audio_to_video_length(dub_audio_wav, src, safe_dub, safety_ms=10)
        dub_audio_wav = safe_dub

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

        # ------ Demucs Vocal Separation & Mixing (opsiyonel) ------
        if use_demucs and dub_audio_wav:
            separator = DemucsVocalSeparator(model=demucs_model)
            if separator.is_available:
                demucs_output = out / "demucs_output"
                stems = separator.separate_vocals(
                    str(wav_for_diar),
                    str(demucs_output)
                )
                if stems:
                    mixer = DubbingMixer(
                        instrumental_volume=instrumental_volume,
                        dubbing_volume=dubbing_volume
                    )
                    final_output = out / "final_dubbed_with_music.wav"
                    mixed_path = mixer.mix_dubbing_with_instrumental(
                        str(dub_audio_wav),
                        stems["instrumental"],
                        str(final_output)
                    )
                    dbg.snap("DEMUCS_MIX_CREATED", path=str(mixed_path))

                    if lipsync_video:
                        final_video = out / "final_dubbed_video_with_music.mp4"
                        _mux_audio_to_video(Path(lipsync_video) if isinstance(lipsync_video, str) else lipsync_video,
                                            Path(mixed_path), final_video)
                        lipsync_video = final_video
                        dbg.snap("FINAL_VIDEO_WITH_MUSIC", path=str(final_video))

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
        min_overlap_ratio=0.2,
        boundary_tolerance=0.1,
        use_vad_boundaries=True,
        use_timeline=True,
        confidence_threshold=0.6,
        # ÇEVİRİ
        do_translate=True,
        translator_model="gpt-4o",
        # DUB + LIPSYNC
        do_dub=True,
        target_lang="tr",
        xtts_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        xtts_speed=None,
        do_lipsync=True,
        wav2lip_repo=None,
        wav2lip_checkpoint=None,
        # Speaker Analysis & Demucs
        analyze_speakers=True,
        remove_overlaps=True,
        use_demucs=True,
        demucs_model="htdemucs",
        instrumental_volume=0.8,
        dubbing_volume=1.0
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
