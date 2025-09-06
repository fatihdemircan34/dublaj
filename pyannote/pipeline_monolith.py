#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monolithic pipeline that wraps diarization+ASR with pre/post steps.

What you get:
- DownloadStep → ExtractAudioStep → (optional) VoiceSeparationStep
- WhisperXDiarizeStep (VAD + Diarization only; no OSD here)
- ASRStep (faster-whisper, word-level)
- OverlapDetectionStep (OSD + overlap gating + speaker assignment smoothing)
- TranslationStep (optional; pass-through by default)
- BuildRefVoicesStep (optional, if pydub & xtts available)
- TTSStep (pluggable; or use --synth-audio to point to a ready WAV)
- PerfectMixStep (Demucs/ducking + EBU R128 normalize)
- LipSyncStep (mux audio into original video)

Usage (examples):
  python3 pipeline_monolith.py \
      --input "https://www.youtube.com/watch?v=xxxx" \
      --out-jsonl out.jsonl \
      --temp-dir ./_temp \
      --hf-token $HF_TOKEN \
      --asr-model large-v3 \
      --synth-audio ./tts_merged.wav \
      --do-mix \
      --do-lipsync

Minimal (just ASR+diarization JSONL from a WAV):
  python3 pipeline_monolith.py --audio in.wav --out-jsonl out.jsonl --hf-token $HF_TOKEN
"""

import os
import sys
import json
import argparse
import logging
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from bisect import insort

# Optional imports from your project (graceful if missing)
try:
    from download import DownloadStep as _DownloadStep  # provided by user
except Exception:
    _DownloadStep = None

try:
    from extract_audio import ExtractAudioStep as _ExtractAudioStep  # provided by user
except Exception:
    _ExtractAudioStep = None

try:
    from build_ref_voices import BuildRefVoicesStep as _BuildRefVoicesStep  # provided by user
except Exception:
    _BuildRefVoicesStep = None

try:
    from detayli_arastirma import PerfectMixStep as _PerfectMixStep  # provided by user
except Exception:
    _PerfectMixStep = None

try:
    from lipsync import LipSyncStep as _LipSyncStep  # provided by user
except Exception:
    _LipSyncStep = None

# Third‑party (required for core logic)
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from pyannote.core import Segment, Timeline
from faster_whisper import WhisperModel

# -------------------------------- logging -------------------------------- #
logger = logging.getLogger("pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ----------------------------- small helpers ----------------------------- #
def round3(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(f"{x:.3f}")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-version"], check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError("ffmpeg is required on PATH") from e


def run(cmd: List[str], timeout: Optional[int] = None) -> None:
    logger.debug("RUN: %s", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "command failed")


# ---------------------------- pipeline context --------------------------- #
Context = Dict[str, Any]

def make_ctx(temp_dir: Path, config: Dict[str, Any]) -> Context:
    return {
        "temp_dir": str(temp_dir.resolve()),
        "config": config or {},
        "artifacts": {},
        "audio": {},
        "flags": {},
    }


# ----------------------------- built-in steps ---------------------------- #
class DownloadStep:
    """Fallback downloader (uses yt-dlp) if your project DownloadStep isn't available."""
    name = "Download"

    def run(self, ctx: Context) -> None:
        if _DownloadStep is not None:
            # Delegate to your implementation
            _DownloadStep().run(ctx)
            return

        inp = ctx["config"].get("input")
        if not inp:
            logger.info("DownloadStep: no input URL/file provided; skipping.")
            return

        temp_dir = Path(ctx["temp_dir"])
        out = temp_dir / "input.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)

        if "://" in inp:
            # URL → use yt-dlp
            cmd = ["yt-dlp", "-f", ctx["config"].get("yt_format", "bv*+ba/b"), "-o", str(out), inp]
            run(cmd)
            logger.info("Downloaded -> %s", out)
        else:
            # Local file path
            src = Path(inp)
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy(src, out)
            logger.info("Copied local input -> %s", out)

        ctx.setdefault("artifacts", {})["video"] = str(out)


class ExtractAudioStep:
    """Fallback extract (ffmpeg) if your project ExtractAudioStep isn't available."""
    name = "ExtractAudio"

    def run(self, ctx: Context) -> None:
        if _ExtractAudioStep is not None:
            _ExtractAudioStep().run(ctx)
            return
        ensure_ffmpeg()
        video_path = ctx["artifacts"].get("video")
        if not video_path:
            logger.info("ExtractAudioStep: no video to extract; skipping.")
            return
        out = Path(ctx["temp_dir"]) / "original_audio.wav"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", str(ctx["config"].get("sample_rate", 16000)),
            "-c:a", "pcm_s16le",
            str(out)
        ]
        run(cmd)
        ctx["artifacts"]["original_audio"] = str(out)
        logger.info("Audio extracted -> %s", out)


class VoiceSeparationStep:
    """Optional: quickly estimate 'music_only' using simple HP/LP (Demucs is handled in PerfectMixStep)."""
    name = "VoiceSeparation"

    def run(self, ctx: Context) -> None:
        ensure_ffmpeg()
        orig = ctx["artifacts"].get("original_audio")
        if not orig:
            logger.info("VoiceSeparationStep: missing original_audio; skip.")
            return
        out = Path(ctx["temp_dir"]) / "music_estimate.wav"
        # Super light 'music bed' approximation (HP/LP), not true separation. Demucs will be used later.
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(orig),
            "-af", "highpass=f=120,lowpass=f=12000,volume=0.6",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out)
        ]
        try:
            run(cmd)
            ctx["artifacts"]["music_estimate"] = str(out)
        except Exception as e:
            logger.warning("VoiceSeparationStep failed: %s", e)


# ------------------------ VAD/OSD/DIAR/ASR helpers ----------------------- #
def build_brouhaha_vad(hf_token: str, onset: float = 0.5, offset: float = 0.5,
                       min_on: float = 0.0, min_off: float = 0.0) -> VoiceActivityDetection:
    seg_model = Model.from_pretrained("pyannote/brouhaha", use_auth_token=hf_token)
    vad = VoiceActivityDetection(segmentation=seg_model)
    vad.instantiate({
        "onset": onset,
        "offset": offset,
        "min_duration_on": min_on,
        "min_duration_off": min_off,
    })
    return vad


def build_osd(hf_token: str, min_on: float = 0.10, min_off: float = 0.10) -> OverlappedSpeechDetection:
    seg_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    osd = OverlappedSpeechDetection(segmentation=seg_model)
    osd.instantiate({
        "min_duration_on": min_on,
        "min_duration_off": min_off,
    })
    return osd


def build_diarization(hf_token: str) -> Pipeline:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline.instantiate({})
    return pipeline


def timeline_to_dict(timeline: Timeline) -> List[Dict[str, float]]:
    return [{"start": round3(seg.start), "end": round3(seg.end), "duration": round3(seg.duration)} for seg in timeline]


def diarization_to_dict(diarization) -> List[Dict[str, Any]]:
    segments = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round3(turn.start),
            "end": round3(turn.end),
            "duration": round3(turn.duration),
            "speaker": label
        })
    return segments


def _intersect_timelines(t1: Timeline, t2: Timeline) -> Timeline:
    inter = []
    for a in t1:
        for b in t2:
            ab = a & b
            if ab is not None and ab.duration > 0:
                inter.append(ab)
    return Timeline(inter).support()


def _timeline_total_duration(tl: Timeline) -> float:
    return sum(seg.duration for seg in tl)


def timeline_coverage_ratio(timeline: Timeline, word_seg: Segment) -> float:
    covered = 0.0
    for seg in timeline:
        inter = seg & word_seg
        if inter is not None:
            covered += inter.duration
    return min(1.0, covered / max(1e-6, word_seg.duration))


def timeline_overlaps(timeline: Timeline, word_seg: Segment) -> bool:
    for seg in timeline:
        if (seg & word_seg) is not None:
            return True
    return False


SMOOTH_WIN = 0.05
STICKY_RATIO = 0.5
LOCAL_TURN_BIAS = 0.6

def assign_speaker(diarization, word_seg: Segment) -> Tuple[str, float]:
    best_label = "SPEAKER_00"
    best_overlap = 0.0
    for turn, _, label in diarization.itertracks(yield_label=True):
        inter = turn & word_seg
        if inter is not None:
            dur = inter.duration
            if dur > best_overlap:
                best_overlap = dur
                best_label = label
    conf = min(1.0, best_overlap / max(1e-6, word_seg.duration))
    return best_label, conf


def assign_speaker_smooth(diarization, word_seg: Segment,
                          prev_speaker: Optional[str], is_overlap: bool) -> Tuple[str, float]:
    t_mid = 0.5 * (word_seg.start + word_seg.end)
    win = Segment(max(0.0, t_mid - SMOOTH_WIN), t_mid + SMOOTH_WIN)

    scores = defaultdict(float)
    total = 0.0
    active_turns = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        inter = (turn & win)
        if inter is not None:
            dur = inter.duration
            scores[label] += dur
            total += dur
        if turn.start <= t_mid < turn.end:
            active_turns.append((turn, label))

    if not scores:
        return assign_speaker(diarization, word_seg)

    if is_overlap and len(active_turns) >= 2:
        active_turns.sort(key=lambda x: x[0].duration)
        shortest_turn, shortest_label = active_turns[0]
        longest_turn, _ = active_turns[-1]
        if shortest_turn.duration <= LOCAL_TURN_BIAS * longest_turn.duration:
            conf = min(1.0, scores.get(shortest_label, shortest_turn.duration) / max(total, 1e-6))
            return shortest_label, conf

    candidate = max(scores.items(), key=lambda kv: kv[1])[0]
    conf = min(1.0, scores[candidate] / max(total, 1e-6))

    if is_overlap and prev_speaker is not None and candidate != prev_speaker:
        if scores[candidate] < STICKY_RATIO * scores.get(prev_speaker, 0.0):
            candidate = prev_speaker
    return candidate, conf


# ------------------------------ pipeline steps --------------------------- #
class WhisperXDiarizeStep:
    """Run VAD + Diarization (no OSD here)."""
    name = "WhisperXDiarize"

    def __init__(self, model_name: str = "large-v2", compute_type: str = "int8",
                 language: Optional[str] = None, min_chunk: float = 1.5, max_chunk: float = 3.0) -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self.language = language
        self.min_chunk = float(min_chunk)
        self.max_chunk = float(max_chunk)

    def run(self, ctx: Context) -> None:
        audio_path = ctx["config"].get("audio") or ctx["artifacts"].get("original_audio")
        if not audio_path:
            raise RuntimeError("WhisperXDiarizeStep: audio not found")

        hf_token = ctx["config"].get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError("HF token required (set --hf-token or HF_TOKEN env)")

        # VAD
        vad = build_brouhaha_vad(
            hf_token=hf_token,
            onset=float(ctx["config"].get("vad_onset", 0.5)),
            offset=float(ctx["config"].get("vad_offset", 0.5)),
            min_on=float(ctx["config"].get("vad_min_on", 0.0)),
            min_off=float(ctx["config"].get("vad_min_off", 0.0)),
        )
        vad_ann = vad(audio_path)
        vad_timeline = vad_ann.get_timeline().support()

        # Diarization
        diar = build_diarization(hf_token)
        call_params = {}
        if ctx["config"].get("min_speakers") is not None:
            call_params["min_speakers"] = int(ctx["config"]["min_speakers"])
        if ctx["config"].get("max_speakers") is not None:
            call_params["max_speakers"] = int(ctx["config"]["max_speakers"])
        diar_ann = diar(audio_path, **call_params)

        speakers = sorted({label for _, _, label in diar_ann.itertracks(yield_label=True)})
        logger.info("Diarization: %d speakers -> %s", len(speakers), ", ".join(speakers))

        # Save into context
        ctx["_vad_ann"] = vad_ann
        ctx["_diar_ann"] = diar_ann
        ctx["artifacts"]["vad_segments"] = timeline_to_dict(vad_timeline)
        ctx["artifacts"]["diarization_segments"] = diarization_to_dict(diar_ann)
        ctx["artifacts"]["speakers"] = speakers

        # Optionally dump
        temp = Path(ctx["temp_dir"])
        save_json({"segments": ctx["artifacts"]["vad_segments"]}, temp / "step1_vad.json")
        save_json({"segments": ctx["artifacts"]["diarization_segments"], "speakers": speakers}, temp / "step3_diarization.json")


class ASRStep:
    """Word-level ASR (faster-whisper) and raw JSON dump (no speaker assignment yet)."""
    name = "ASR"

    def __init__(self, model_name: str = "large-v3", device: str = "auto", compute_type: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

    def run(self, ctx: Context) -> None:
        audio_path = ctx["config"].get("audio") or ctx["artifacts"].get("original_audio")
        if not audio_path:
            raise RuntimeError("ASRStep: audio not found")
        logger.info("Loading ASR model: %s", self.model_name)
        asr = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        logger.info("Transcribing (word timestamps=true)...")
        words: List[Dict[str, Any]] = []
        segments, _info = asr.transcribe(
            audio_path,
            task="transcribe",
            vad_filter=False,
            word_timestamps=True,
            beam_size=10,
            temperature=0.0,
        )
        for seg in segments:
            if seg.words is None:
                continue
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                conf = getattr(w, "probability", None)
                conf = float(conf) if conf is not None else 0.9
                words.append({
                    "word": (w.word or "").strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                    "confidence": float(conf),
                })
        ctx["artifacts"]["words_raw"] = words
        save_json({"total_words": len(words)}, Path(ctx["temp_dir"]) / "step4_asr_meta.json")
        logger.info("ASR done: %d words", len(words))


class OverlapDetectionStep:
    """OSD + gating + (re)assign speakers to words with smoothing; mark overlap."""
    name = "OverlapDetection"

    def __init__(self, use_gpu: Optional[bool] = None) -> None:
        self.use_gpu = use_gpu

    def run(self, ctx: Context) -> None:
        audio_path = ctx["config"].get("audio") or ctx["artifacts"].get("original_audio")
        if not audio_path:
            raise RuntimeError("OverlapDetectionStep: audio not found")

        hf_token = ctx["config"].get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError("HF token required (set --hf-token or HF_TOKEN env)")

        words = ctx["artifacts"].get("words_raw") or []
        diar_ann = ctx.get("_diar_ann")
        vad_ann = ctx.get("_vad_ann")
        if diar_ann is None or vad_ann is None:
            raise RuntimeError("Need diarization & VAD from previous step.")

        # OSD
        osd = build_osd(hf_token, min_on=float(ctx["config"].get("osd_min_on", 0.10)),
                        min_off=float(ctx["config"].get("osd_min_off", 0.10)))
        osd_ann = osd(audio_path)
        osd_tl = osd_ann.get_timeline().support()
        vad_support = vad_ann.get_timeline().support()

        # Diar >= 2
        bounds = []
        for turn, _, _lbl in diar_ann.itertracks(yield_label=True):
            insort(bounds, (turn.start, +1))
            insort(bounds, (turn.end, -1))
        res = []
        active, prev_t = 0, None
        for t, delta in bounds:
            if prev_t is not None and t > prev_t:
                if active >= 2:
                    res.append(Segment(prev_t, t))
            active += delta
            prev_t = t
        diar_ge2 = Timeline(res).support()

        osd_in_vad = _intersect_timelines(osd_tl, vad_support)
        final_overlap = _intersect_timelines(osd_in_vad, diar_ge2)
        logger.info("OSD total=%.2fs | OSD∧VAD=%.2fs | Diar≥2=%.2fs | Final=%.2fs",
                    _timeline_total_duration(osd_tl),
                    _timeline_total_duration(osd_in_vad),
                    _timeline_total_duration(diar_ge2),
                    _timeline_total_duration(final_overlap))

        # Speaker assignment + overlap flag
        prev_speaker: Optional[str] = None
        results: List[Dict[str, Any]] = []
        for w in words:
            seg = Segment(float(w["start"]), float(w["end"]))
            is_ov = timeline_overlaps(final_overlap, seg)
            speaker, spk_conf = assign_speaker_smooth(diar_ann, seg, prev_speaker, is_ov)
            prev_speaker = speaker
            results.append({
                "word": w["word"],
                "start": round3(w["start"]),
                "end": round3(w["end"]),
                "confidence": round3(w["confidence"]),
                "speaker": speaker,
                "speaker_confidence": round3(spk_conf),
                "is_overlap": bool(is_ov),
            })

        ctx["artifacts"]["words"] = results
        # Also provide diarization segments for downstream steps (BuildRefVoicesStep expects this key name)
        ctx["artifacts"]["segments"] = ctx["artifacts"].get("diarization_segments", [])

        # Write JSONL if requested
        out_jsonl = ctx["config"].get("out_jsonl")
        if out_jsonl:
            with open(out_jsonl, "w", encoding="utf-8") as f:
                for obj in results:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            logger.info("Wrote word-level JSONL -> %s", out_jsonl)


class TranslationStep:
    """Optional MT: pass-through by default (set translator='googletrans' to try local lib)."""
    name = "Translation"

    def __init__(self, engine: str = "none", source_lang: str = "en", target_lang: str = "tr") -> None:
        self.engine = engine
        self.source = source_lang
        self.target = target_lang

    def run(self, ctx: Context) -> None:
        words = ctx["artifacts"].get("words")
        if not words:
            logger.info("TranslationStep: no words; skip.")
            return
        if self.engine == "none":
            ctx["artifacts"]["translated_words"] = words
            return
        if self.engine == "googletrans":
            try:
                from googletrans import Translator
                tr = Translator()
                texts = [w["word"] for w in words]
                # Batch translate in chunks of 100
                translated = []
                chunk = 100
                for i in range(0, len(texts), chunk):
                    part = texts[i:i+chunk]
                    res = tr.translate(part, src=self.source, dest=self.target)
                    for t, obj in zip(res, words[i:i+chunk]):
                        obj2 = dict(obj)
                        obj2["word_tr"] = t.text
                        translated.append(obj2)
                ctx["artifacts"]["translated_words"] = translated
                return
            except Exception as e:
                logger.warning("googletrans failed (%s); pass-through", e)
        # fallback: copy
        ctx["artifacts"]["translated_words"] = words


class TTSStep:
    """
    Simple pluggable TTS:
      - If --synth-audio is provided, just use that WAV
      - Else try XTTS (coqui TTS) if available and ref voices exist
      - Else raise informative error
    """
    name = "TTS"

    def __init__(self, tts_name: str = "xtts", tts_kw: Optional[Dict[str, Any]] = None,
                 voice_map: Optional[Dict[str, str]] = None, crossfade_ms: int = 10) -> None:
        self.tts_name = tts_name
        self.tts_kw = tts_kw or {}
        self.voice_map = voice_map or {}
        self.crossfade_ms = crossfade_ms

    def run(self, ctx: Context) -> None:
        synth_arg = ctx["config"].get("synth_audio")
        temp = Path(ctx["temp_dir"])

        if synth_arg:
            p = Path(synth_arg)
            if not p.exists():
                raise FileNotFoundError(p)
            out = temp / "tts_merged.wav"
            shutil.copy(p, out)
            ctx["artifacts"]["synth_audio"] = str(out)
            logger.info("TTS: using provided synth wav -> %s", out)
            return

        # Try XTTS if available
        try:
            from core.models.tts.xtts import XTTSEngine  # your project structure (optional)
            from pydub import AudioSegment  # for simple concatenation
        except Exception:
            raise RuntimeError(
                "No TTS audio provided and XTTS not available. "
                "Provide --synth-audio path to a WAV to proceed to mixing."
            )

        voices_dir = Path(ctx["artifacts"].get("ref_voices_dir", temp / "voices"))
        if not voices_dir.exists():
            raise RuntimeError("No reference voices: run BuildRefVoicesStep first or provide --synth-audio")

        engine = XTTSEngine(**self.tts_kw)  # type: ignore
        words = ctx["artifacts"].get("translated_words") or ctx["artifacts"].get("words")
        if not words:
            raise RuntimeError("TTS: no text to synthesize")

        # Super-simple per-word concat (demo). Real system should sentence-cluster and crossfade.
        speech = AudioSegment.silent(duration=0)
        for w in words:
            text = w.get("word_tr") or w.get("word") or ""
            spk = w.get("speaker")
            voice_override = self.voice_map.get(str(spk)) if self.voice_map else None
            ref_wav = voice_override or str(voices_dir / f"{spk}.wav")
            try:
                audio = engine.tts(text=text, speaker_wav=ref_wav)  # returns pydub AudioSegment in many setups
            except Exception as e:
                logger.warning("XTTS failed for %s (%s); fallback silence", spk, e)
                audio = AudioSegment.silent(duration=int((w["end"] - w["start"]) * 1000))
            if self.crossfade_ms > 0 and len(speech) > 0:
                speech = speech.append(audio, crossfade=self.crossfade_ms)
            else:
                speech += audio

        out = temp / "tts_merged.wav"
        speech.set_channels(1).set_frame_rate(int(ctx["config"].get("sample_rate", 16000))).export(out, format="wav")
        ctx["artifacts"]["synth_audio"] = str(out)
        logger.info("TTS: merged wav -> %s", out)


# ----------------------- Integrations: Mix & LipSync ---------------------- #
class PerfectMixStepWrapper:
    """Thin wrapper that delegates to your PerfectMixStep implementation if available."""
    name = "PerfectMix"

    def __init__(self, lufs_target: float = -14.0, duck_db: float = -7.0, pan_amount: float = 0.0, **kwargs) -> None:
        self.kwargs = dict(lufs_target=lufs_target, duck_db=duck_db, pan_amount=pan_amount, **kwargs)

    def run(self, ctx: Context) -> None:
        if _PerfectMixStep is None:
            raise RuntimeError("detayli_arastirma.PerfectMixStep not found in environment")
        # Map artifact names expected by your step
        art = ctx.setdefault("artifacts", {})
        # PerfectMixStep looks for 'original' & 'dub' wavs by heuristic; we set common keys to be safe
        art.setdefault("tts_merged", art.get("synth_audio"))
        art.setdefault("dubbed_audio", art.get("synth_audio"))
        art.setdefault("final_audio", art.get("synth_audio"))
        step = _PerfectMixStep(**self.kwargs)
        step.run(ctx)  # this writes artifacts['mixed_wav']
        # Align key for LipSync
        ctx["artifacts"]["final_audio"] = ctx["artifacts"].get("mixed_wav")


class LipSyncStepWrapper:
    name = "LipSync"
    def __init__(self, model_name: str = "simple", model_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    def run(self, ctx: Context) -> None:
        if _LipSyncStep is None:
            # Fallback: simple ffmpeg mux of audio over video
            ensure_ffmpeg()
            video = ctx["artifacts"].get("video")
            audio = ctx["artifacts"].get("final_audio") or ctx["artifacts"].get("synth_audio")
            if not (video and audio):
                logger.info("LipSync fallback: missing video or audio; skip.")
                return
            out = Path(ctx["temp_dir"]) / "final_video.mp4"
            cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", str(out)]
            run(cmd)
            ctx["artifacts"]["final_video"] = str(out)
            logger.info("Muxed video -> %s", out)
            return
        step = _LipSyncStep(self.model_name, self.model_kwargs)
        step.run(ctx)


# --------------------------------- runner -------------------------------- #
def run_steps(steps: List[Any], ctx: Context) -> None:
    for s in steps:
        try:
            logger.info("──▶ %s", getattr(s, "name", s.__class__.__name__))
            s.run(ctx)
        except Exception as e:
            logger.error("%s failed: %s", getattr(s, "name", s.__class__.__name__), e)
            raise


# ---------------------------------- CLI ---------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple end-to-end dubbing pipeline with diarization+ASR core")
    p.add_argument("--input", help="URL or path (video). If given, Download+ExtractAudio will run.")
    p.add_argument("--audio", help="Direct WAV path (skip Download+Extract)")
    p.add_argument("--out-jsonl", dest="out_jsonl", help="Word-level JSONL output path")
    p.add_argument("--temp-dir", default="./_temp", help="Working directory")
    p.add_argument("--hf-token", dest="hf_token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token")
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--asr-device", default="auto")
    p.add_argument("--asr-compute-type", default="auto")
    p.add_argument("--translator", default="none", help="none|googletrans")
    p.add_argument("--source_lang", default="en")
    p.add_argument("--target_lang", default="tr")
    p.add_argument("--tts", default="xtts")
    p.add_argument("--tts_kw", default=None, help="JSON dict with engine kwargs")
    p.add_argument("--voice_map", default=None, help="JSON dict {speaker: ref_wav} to override")
    p.add_argument("--crossfade_ms", type=int, default=10)
    p.add_argument("--synth-audio", dest="synth_audio", help="Provide a pre-synthesized WAV to test mixing")
    p.add_argument("--do-mix", action="store_true", help="Run PerfectMixStep")
    p.add_argument("--do-lipsync", action="store_true", help="Run LipSyncStep at the end")
    # VAD/OSD/diar tuning
    p.add_argument("--vad-onset", type=float, default=0.5)
    p.add_argument("--vad-offset", type=float, default=0.5)
    p.add_argument("--vad-min-on", type=float, default=0.0)
    p.add_argument("--vad-min-off", type=float, default=0.0)
    p.add_argument("--osd-min-on", type=float, default=0.10)
    p.add_argument("--osd-min-off", type=float, default=0.10)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    temp_dir = Path(args.temp_dir); temp_dir.mkdir(parents=True, exist_ok=True)

    # Build config for steps
    cfg: Dict[str, Any] = {
        "input": args.input,
        "audio": args.audio,
        "out_jsonl": args.out_jsonl,
        "temp_dir": str(temp_dir),
        "hf_token": args.hf_token,
        "sample_rate": 16000,
        "yt_format": None,
        # diar/OSD/VAD
        "vad_onset": args.vad_onset, "vad_offset": args.vad_offset,
        "vad_min_on": args.vad_min_on, "vad_min_off": args.vad_min_off,
        "osd_min_on": args.osd_min_on, "osd_min_off": args.osd_min_off,
        "min_speakers": args.min_speakers, "max_speakers": args.max_speakers,
        # TTS/MT
        "translator": args.translator,
        "source_lang": args.source_lang, "target_lang": args.target_lang,
        "tts": args.tts,
        "tts_kw": json.loads(args.tts_kw) if args.tts_kw else {},
        "voice_map": json.loads(args.voice_map) if args.voice_map else {},
        "crossfade_ms": args.crossfade_ms,
        "synth_audio": args.synth_audio,
        # PerfectMix defaults (override by editing below or via env)
        "lufs_target": -14.0, "duck_db": -7.0, "pan_amount": 0.0,
    }

    # Resolve audio: Download+ExtractAudio if --input is given and --audio is not
    steps: List[Any] = []
    if args.input and not args.audio:
        steps += [DownloadStep(), ExtractAudioStep()]
    else:
        # If --audio is provided, store as artifact for downstream steps
        if args.audio:
            ctx = make_ctx(temp_dir, cfg)
            ctx.setdefault("artifacts", {})["original_audio"] = args.audio
            # We'll pass this ctx to the first actual run below, so no extra step here
        steps += []

    # The core steps as requested by you
    steps += [
        VoiceSeparationStep(),
        WhisperXDiarizeStep(model_name="large-v2", compute_type="int8", language="en", min_chunk=1.5, max_chunk=3.0),
        ASRStep(model_name=args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type),
        OverlapDetectionStep(use_gpu=None),
        TranslationStep(cfg.get("translator", "none"), cfg.get("source_lang", "en"), cfg.get("target_lang", "tr")),
    ]

    # Optional: speaker reference voices (only if your dependency stack is available)
    if _BuildRefVoicesStep is not None:
        steps.append(_BuildRefVoicesStep(seconds=3))

    # TTS (or provide --synth-audio)
    steps.append(TTSStep(tts_name=args.tts, tts_kw=cfg.get("tts_kw"), voice_map=cfg.get("voice_map"), crossfade_ms=cfg.get("crossfade_ms", 10)))

    # Perfect mix & lipsync (optional via flags)
    if args.do_mix:
        steps.append(PerfectMixStepWrapper(lufs_target=cfg.get("lufs_target", -14.0),
                                           duck_db=cfg.get("duck_db", -7.0),
                                           pan_amount=cfg.get("pan_amount", 0.0)))
    if args.do_lipsync:
        steps.append(LipSyncStepWrapper(model_name=cfg.get("lipsync", "simple"),
                                        model_kwargs=cfg.get("lipsync_opts") or {}))

    # Run
    ctx = make_ctx(temp_dir, cfg)
    # If user passed --audio directly, seed artifact for downstream
    if args.audio:
        ctx["artifacts"]["original_audio"] = args.audio

    run_steps(steps, ctx)

    # Summary
    summary = {
        "artifacts": ctx.get("artifacts", {}),
        "speakers": ctx["artifacts"].get("speakers"),
        "stats": {
            "total_words": len(ctx["artifacts"].get("words", [])),
        }
    }
    save_json(summary, Path(args.out_jsonl).with_suffix(".summary.json") if args.out_jsonl else Path(ctx["temp_dir"]) / "pipeline_summary.json")
    logger.info("Done.")

if __name__ == "__main__":
    main()
