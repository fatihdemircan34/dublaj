#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from bisect import insort

from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from pyannote.core import Segment, Timeline

from faster_whisper import WhisperModel


# ----------------------------- yardımcılar ----------------------------- #

def round3(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(f"{x:.3f}")


def save_step_output(step_name: str, data: Dict, output_dir: str) -> None:
    """Her adımın çıktısını JSON olarak kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{step_name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 {step_name} çıktısı kaydedildi: {filepath}")


def timeline_to_dict(timeline: Timeline) -> List[Dict]:
    """Timeline'ı JSON serileştirilebilir formata çevir."""
    return [
        {"start": round3(seg.start), "end": round3(seg.end), "duration": round3(seg.duration)}
        for seg in timeline
    ]


def diarization_to_dict(diarization) -> List[Dict]:
    """Diarization sonuçlarını JSON formatına çevir."""
    segments = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round3(turn.start),
            "end": round3(turn.end),
            "duration": round3(turn.duration),
            "speaker": label
        })
    return segments


# ----------------------------- modeller ----------------------------- #

def build_brouhaha_vad(
        hf_token: str,
        onset: float = 0.5,
        offset: float = 0.5,
        min_on: float = 0.0,
        min_off: float = 0.0
) -> VoiceActivityDetection:
    """pyannote/brouhaha modelini segmentation olarak kullanıp VAD pipeline'ı kurar."""
    print("🔧 VAD modeli yükleniyor (pyannote/brouhaha)...")
    seg_model = Model.from_pretrained("pyannote/brouhaha", use_auth_token=hf_token)
    print(f"⚙️  VAD parametreleri: onset={onset}, offset={offset}, min_on={min_on}, min_off={min_off}")
    vad = VoiceActivityDetection(segmentation=seg_model)
    vad.instantiate({
        "onset": onset,
        "offset": offset,
        "min_duration_on": min_on,
        "min_duration_off": min_off,
    })
    print("✅ VAD modeli hazır")
    return vad


def build_osd(
        hf_token: str,
        onset: float = 0.7,
        offset: float = 0.7,
        min_on: float = 0.10,
        min_off: float = 0.10
) -> OverlappedSpeechDetection:
    """Overlapped Speech Detection (pyannote/segmentation-3.0) pipeline'ı."""
    print("🔧 OSD modeli yükleniyor (pyannote/segmentation-3.1)...")
    seg_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    print(f"⚙️  OSD parametreleri: onset={onset}, offset={offset}, min_on={min_on}, min_off={min_off}")
    osd = OverlappedSpeechDetection(segmentation=seg_model)
    osd.instantiate({
        "min_duration_on": min_on,
        "min_duration_off": min_off,
    })
    print("✅ OSD modeli hazır")
    return osd


def build_diarization(
        hf_token: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
) -> Pipeline:
    """speaker-diarization-3.1 pipeline'ını yükler ve instantiate eder."""
    print("🔧 Diarization modeli yükleniyor (pyannote/speaker-diarization-3.1)...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)



    pipeline.instantiate({})
    print("✅ Diarization modeli hazır")
    return pipeline


# ----------------------------- timeline yardımcıları ----------------------------- #

def timeline_coverage_ratio(timeline: Timeline, word_seg: Segment) -> float:
    """timeline ile kelime segmenti kesişim oranı."""
    covered = 0.0
    for seg in timeline:
        inter = seg & word_seg
        if inter is not None:
            covered += inter.duration
    return min(1.0, covered / max(1e-6, word_seg.duration))


def timeline_overlaps(timeline: Timeline, word_seg: Segment) -> bool:
    """timeline herhangi bir seg ile kelime segmenti kesişiyor mu?"""
    for seg in timeline:
        if (seg & word_seg) is not None:
            return True
    return False


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


def build_overlap_from_diarization(diarization, min_count: int = 2) -> Timeline:
    """Diarization’dan aynı anda >=min_count konuşmacı olan bölgeleri üret."""
    bounds = []
    for turn, _, _label in diarization.itertracks(yield_label=True):
        insort(bounds, (turn.start, +1))
        insort(bounds, (turn.end, -1))

    res: List[Segment] = []
    active = 0
    prev_t = None
    for t, delta in bounds:
        if prev_t is not None and t > prev_t:
            if active >= min_count:
                res.append(Segment(prev_t, t))
        active += delta
        prev_t = t
    return Timeline(res).support()


# ----------------------------- speaker atama ----------------------------- #

SMOOTH_WIN = 0.05      # 150 ms: midpoint etrafında pencere
STICKY_RATIO = 0.5   # overlap sırasında önceki konuşmacıyı koruma eşiği
LOCAL_TURN_BIAS = 0.6  # iki aktif turn varsa, belirgin şekilde kısa olanı tercih et (~%40 daha kısa)


def assign_speaker(diarization, word_seg: Segment) -> Tuple[str, float]:
    """
    Basit: kelime segmentiyle en çok kesişen konuşmacıyı seç.
    speaker_confidence = (en büyük kesişim) / (kelime süresi)
    """
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


def assign_speaker_smooth(
        diarization,
        word_seg: Segment,
        prev_speaker: Optional[str],
        is_overlap: bool
) -> Tuple[str, float]:
    """
    Midpoint ± SMOOTH_WIN penceresinde çoğunluk konuşmacıyı seç.
    Overlap anında: (1) aynı anda aktif turn’ler arasında lokal/kısa turn’e bias,
    (2) gerekirse önceki konuşmacıya hafif sadakat uygula.
    """
    t_mid = 0.5 * (word_seg.start + word_seg.end)
    win = Segment(max(0.0, t_mid - SMOOTH_WIN), t_mid + SMOOTH_WIN)

    scores = defaultdict(float)
    total = 0.0
    active_turns = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        # skorlar
        inter = (turn & win)
        if inter is not None:
            dur = inter.duration
            scores[label] += dur
            total += dur
        # aktif turn listesi
        if turn.start <= t_mid < turn.end:
            active_turns.append((turn, label))

    if not scores:
        return assign_speaker(diarization, word_seg)

    # Eğer overlap ve aynı anda >=2 aktif turn varsa "daha lokal/kısa" turn'ü tercih et
    if is_overlap and len(active_turns) >= 2:
        active_turns.sort(key=lambda x: x[0].duration)
        shortest_turn, shortest_label = active_turns[0]
        longest_turn, _longest_label = active_turns[-1]
        if shortest_turn.duration <= LOCAL_TURN_BIAS * longest_turn.duration:
            conf = min(1.0, scores.get(shortest_label, shortest_turn.duration) / max(total, 1e-6))
            return shortest_label, conf

    # Aksi halde pencere toplam süresine göre çoğunluk
    candidate = max(scores.items(), key=lambda kv: kv[1])[0]
    conf = min(1.0, scores[candidate] / max(total, 1e-6))

    # Overlap’ta gereksiz hoplamayı azalt: adaya karşı eski konuşmacı çok yakınsa stick
    if is_overlap and prev_speaker is not None and candidate != prev_speaker:
        if scores[candidate] < STICKY_RATIO * scores.get(prev_speaker, 0.0):
            candidate = prev_speaker
    return candidate, conf


# ----------------------------- ASR ----------------------------- #

def transcribe_words(asr_model: WhisperModel, audio_path: str) -> List[Dict]:
    """faster-whisper ile kelime zaman damgaları (iç VAD kapalı)."""
    print("🎙️  ASR transkripsiyon başlıyor...")
    results: List[Dict] = []
    segments, _info = asr_model.transcribe(
        audio_path,
        task="transcribe",
        vad_filter=False,
        word_timestamps=True,
        beam_size=10,
        temperature=0.0,
    )

    word_count = 0
    for seg in segments:
        if seg.words is None:
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            conf = getattr(w, "probability", None)
            conf = float(conf) if conf is not None else 0.9
            results.append({
                "word": w.word.strip(),
                "start": float(w.start),
                "end": float(w.end),
                "confidence": float(conf)
            })
            word_count += 1

    print(f"✅ ASR tamamlandı: {word_count} kelime tespit edildi")
    return results


# ----------------------------- main ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="brouhaha VAD + OSD + diarization + faster-whisper word-level JSONL (overlap-gated, smoothing)"
    )
    parser.add_argument("--audio", required=True, help="Girdi WAV dosyası")
    parser.add_argument("--out", required=True, help="Çıkış JSONL dosyası")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face access token (HF_TOKEN env önerilir)")
    parser.add_argument("--asr-model", default="large-v3",
                        help="faster-whisper modeli (tiny/base/small/medium/large-v3)")
    parser.add_argument("--asr-device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--asr-compute-type", default="auto",
                        help="auto|int8|int8_float16|float16|float32 ...")
    parser.add_argument("--vad-onset", type=float, default=0.0)
    parser.add_argument("--vad-offset", type=float, default=0.0)
    parser.add_argument("--vad-min-on", type=float, default=0.0)
    parser.add_argument("--vad-min-off", type=float, default=0.0)
    parser.add_argument("--osd-onset", type=float, default=0.0, help="OSD karar eşiği (konservatif: 0.7)")
    parser.add_argument("--osd-offset", type=float, default=0.0, help="OSD kapama eşiği")
    parser.add_argument("--osd-min-on", type=float, default=0.10, help="Çok kısa overlap'ları buda (s)")
    parser.add_argument("--osd-min-off", type=float, default=0.10, help="Kısa boşlukları doldur (s)")
    parser.add_argument("--require-vad", action="store_true",
                        help="Kelimenin en az %X'i VAD konuşma bölgesinde değilse dışla")
    parser.add_argument("--vad-coverage", type=float, default=0.6,
                        help="require-vad aktifse minimum VAD kapsama oranı (0-1)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Diarization min konuşmacı")
    parser.add_argument("--max-speakers", type=int, default=None, help="Diarization max konuşmacı")
    parser.add_argument("--output-dir", default=None,
                        help="Ara adım çıktıları klasörü (varsayılan: --out ile aynı dizin)")
    args = parser.parse_args()

    if not args.hf_token:
        print("❌ HF token gerekli. --hf-token ya da HF_TOKEN ortam değişkenini kullan.", file=sys.stderr)
        sys.exit(1)

    audio_path = args.audio
    output_dir = args.output_dir or (os.path.dirname(args.out) or ".")
    print(f"🎵 Audio: {audio_path}")
    print(f"📝 Çıkış: {args.out}")
    print(f"📁 Adım çıktı dizini: {output_dir}")
    print(f"🤖 ASR: {args.asr_model} (device={args.asr_device}, compute={args.asr_compute_type})")
    print("=" * 60)

    # 1) VAD
    print("📍 ADIM 1: Voice Activity Detection (VAD)")
    vad_pipeline = build_brouhaha_vad(
        hf_token=args.hf_token,
        onset=args.vad_onset,
        offset=args.vad_offset,
        min_on=args.vad_min_on,
        min_off=args.vad_min_off,
    )
    print("🔍 VAD analizi yapılıyor...")
    vad_ann = vad_pipeline(audio_path)  # Annotation
    vad_timeline = vad_ann.get_timeline().support()
    vad_segments = len(vad_timeline)
    vad_total_duration = sum(seg.duration for seg in vad_timeline)
    print(f"✅ VAD tamam: {vad_segments} segment, toplam {vad_total_duration:.2f}s")
    save_step_output("step1_vad", {
        "step": "voice_activity_detection",
        "model": "pyannote/brouhaha",
        "parameters": {
            "onset": args.vad_onset,
            "offset": args.vad_offset,
            "min_duration_on": args.vad_min_on,
            "min_duration_off": args.vad_min_off,
        },
        "results": {
            "segments": timeline_to_dict(vad_timeline),
            "total_segments": vad_segments,
            "total_duration": round3(vad_total_duration),
        },
    }, output_dir)
    print()

    # 2) OSD
    print("📍 ADIM 2: Overlapped Speech Detection (OSD)")
    osd_pipeline = build_osd(
        hf_token=args.hf_token,
        onset=args.osd_onset,
        offset=args.osd_offset,
        min_on=args.osd_min_on,
        min_off=args.osd_min_off,
    )
    print("🔍 OSD analizi yapılıyor...")
    osd_ann = osd_pipeline(audio_path)  # Annotation
    osd_timeline = osd_ann.get_timeline().support()
    osd_segments = len(osd_timeline)
    osd_total_duration = sum(seg.duration for seg in osd_timeline)
    print(f"✅ OSD tamam: {osd_segments} segment, toplam {osd_total_duration:.2f}s")
    save_step_output("step2_osd", {
        "step": "overlapped_speech_detection",
        "model": "pyannote/segmentation-3.0",
        "parameters": {
            "onset": args.osd_onset,
            "offset": args.osd_offset,
            "min_duration_on": args.osd_min_on,
            "min_duration_off": args.osd_min_off,
        },
        "results": {
            "segments": timeline_to_dict(osd_timeline),
            "total_segments": osd_segments,
            "total_duration": round3(osd_total_duration),
        },
    }, output_dir)
    print()

    # 3) Diarization
    print("📍 ADIM 3: Speaker Diarization")
    diar_pipeline = build_diarization(args.hf_token, args.min_speakers, args.max_speakers)
    print("🔍 Diarization analizi yapılıyor...")
    diar_call_params = {}
    if args.min_speakers is not None:
        diar_call_params["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        diar_call_params["max_speakers"] = args.max_speakers
    diar = diar_pipeline(audio_path, **diar_call_params)  # Annotation

    speakers = set()
    diar_segments = 0
    for turn, _, label in diar.itertracks(yield_label=True):
        speakers.add(label)
        diar_segments += 1
    print(f"✅ Diarization tamam: {len(speakers)} konuşmacı ({', '.join(sorted(speakers))}), {diar_segments} segment")
    save_step_output("step3_diarization", {
        "step": "speaker_diarization",
        "model": "pyannote/speaker-diarization-3.1",
        "parameters": {
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
        },
        "results": {
            "segments": diarization_to_dict(diar),
            "speakers": sorted(list(speakers)),
            "total_speakers": len(speakers),
            "total_segments": diar_segments,
        },
    }, output_dir)
    print()

    # 3.5) Overlap gating: OSD ∧ VAD ∧ Diar(≥2)
    print("🧰 Overlap timeline kapılanıyor: OSD ∧ VAD ∧ (≥2 konuşmacı)")
    vad_support = vad_ann.get_timeline().support()
    diar_overlap = build_overlap_from_diarization(diar, min_count=2)
    osd_in_vad = _intersect_timelines(osd_timeline, vad_support)
    final_overlap_timeline = _intersect_timelines(osd_in_vad, diar_overlap)
    print(f"  • Ham OSD süresi     : {osd_total_duration:.2f}s")
    print(f"  • OSD∧VAD süresi     : {_timeline_total_duration(osd_in_vad):.2f}s")
    print(f"  • Diar(≥2) süresi    : {_timeline_total_duration(diar_overlap):.2f}s")
    print(f"  • Nihai overlap süresi: {_timeline_total_duration(final_overlap_timeline):.2f}s")
    save_step_output("step2_5_overlap_gated", {
        "step": "overlap_gating",
        "compose": "OSD ∧ VAD ∧ (diarization>=2)",
        "results": {
            "osd_and_vad": timeline_to_dict(osd_in_vad),
            "diar_ge2": timeline_to_dict(diar_overlap),
            "final_overlap": timeline_to_dict(final_overlap_timeline),
        },
    }, output_dir)
    print()

    # 4) ASR
    print("📍 ADIM 4: Automatic Speech Recognition (ASR)")
    print(f"🔧 ASR modeli yükleniyor: {args.asr_model}...")
    asr = WhisperModel(args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type)
    print("✅ ASR modeli hazır")
    words = transcribe_words(asr, audio_path)
    save_step_output("step4_asr", {
        "step": "automatic_speech_recognition",
        "model": args.asr_model,
        "parameters": {
            "device": args.asr_device,
            "compute_type": args.asr_compute_type,
            "vad_filter": False,
            "word_timestamps": True,
            "beam_size": 10,
            "temperature": 0.0,
        },
        "results": {
            "words": words,
            "total_words": len(words),
        },
    }, output_dir)
    print()

    # 5) Kelime -> konuşmacı + overlap + VAD filtresi
    print("📍 ADIM 5: Kelime-seviye analiz ve JSONL yazımı")
    written_words = 0
    filtered_words = 0
    overlap_words = 0
    prev_speaker: Optional[str] = None
    final_results: List[Dict] = []

    with open(args.out, "w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            seg = Segment(w["start"], w["end"])

            # VAD kapsaması
            coverage = timeline_coverage_ratio(vad_timeline, seg)
            if args.require_vad and (coverage < args.vad_coverage):
                filtered_words += 1
                continue

            # overlap var mı? -> kapılanmış timeline’a göre
            is_ov = timeline_overlaps(final_overlap_timeline, seg)
            if is_ov:
                overlap_words += 1

            # konuşmacı ataması
            speaker, spk_conf = assign_speaker_smooth(diar, seg, prev_speaker, is_ov)
            prev_speaker = speaker

            obj = {
                "word": w["word"],
                "start": round3(w["start"]),
                "end": round3(w["end"]),
                "confidence": round3(w["confidence"]),
                "speaker": speaker,
                "speaker_confidence": round3(spk_conf),
                "is_overlap": bool(is_ov),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            final_results.append(obj)
            written_words += 1

            if (i + 1) % 100 == 0:
                print(f"  📝 İşlenen kelime: {i + 1}/{len(words)}")

    print(f"✅ JSONL yazımı tamamlandı: {args.out}")

    # step5 & özet
    save_step_output("step5_final", {
        "step": "final_word_level_analysis",
        "parameters": {
            "require_vad": args.require_vad,
            "vad_coverage": args.vad_coverage,
            "smooth_window": SMOOTH_WIN,
            "sticky_ratio": STICKY_RATIO,
            "local_turn_bias": LOCAL_TURN_BIAS,
        },
        "results": {
            "words": final_results,
            "statistics": {
                "total_words_from_asr": len(words),
                "written_words": written_words,
                "filtered_words": filtered_words,
                "overlap_words": overlap_words,
                "detected_speakers": len(speakers),
                "vad_segments": vad_segments,
                "osd_segments": osd_segments,
            },
        },
    }, output_dir)

    save_step_output("pipeline_summary", {
        "pipeline_summary": {
            "audio_file": audio_path,
            "output_file": args.out,
            "output_directory": output_dir,
            "models_used": {
                "vad": "pyannote/brouhaha",
                "osd": "pyannote/segmentation-3.0",
                "diarization": "pyannote/speaker-diarization-3.1",
                "asr": args.asr_model,
            },
            "final_statistics": {
                "total_words_from_asr": len(words),
                "written_words": written_words,
                "filtered_words": filtered_words,
                "overlap_words": overlap_words,
                "detected_speakers": len(speakers),
                "vad_segments": vad_segments,
                "vad_total_duration": round3(vad_total_duration),
                "osd_segments": osd_segments,
                "osd_total_duration": round3(osd_total_duration),
                "diarization_segments": diar_segments,
            },
        }
    }, output_dir)

    print("📊 ÖZET:")
    print(f"  🎙️  Toplam kelime (ASR): {len(words)}")
    print(f"  📝 Yazılan kelime: {written_words}")
    if args.require_vad:
        print(f"  🚫 VAD filtresi ile elenen: {filtered_words}")
    print(f"  🔄 Overlap tespit edilen: {overlap_words}")
    print(f"  👥 Tespit edilen konuşmacı: {len(speakers)}")
    print(f"  🎵 VAD konuşma segmenti: {vad_segments}")
    print(f"  🔄 OSD overlap segmenti: {osd_segments}")
    print("🎉 İşlem başarıyla tamamlandı!")


if __name__ == "__main__":
    main()
