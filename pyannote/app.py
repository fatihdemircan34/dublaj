import os
import sys
import json
import math
import argparse
from typing import List, Tuple, Optional
from collections import defaultdict

from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from pyannote.core import Segment

from faster_whisper import WhisperModel


def round3(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(f"{x:.3f}")


def save_step_output(step_name: str, data: dict, output_dir: str):
    """Her adımın çıktısını JSON olarak kaydet"""
    filename = f"{step_name}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 {step_name} çıktısı kaydedildi: {filepath}")


def timeline_to_dict(timeline):
    """Timeline'ı JSON serileştirilebilir formata çevir"""
    segments = []
    for seg in timeline:
        segments.append({
            "start": round3(seg.start),
            "end": round3(seg.end),
            "duration": round3(seg.duration)
        })
    return segments


def diarization_to_dict(diarization):
    """Diarization sonuçlarını JSON formatına çevir"""
    segments = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round3(turn.start),
            "end": round3(turn.end),
            "duration": round3(turn.duration),
            "speaker": label
        })
    return segments


def build_brouhaha_vad(hf_token: str,
                       onset: float = 0.5,
                       offset: float = 0.5,
                       min_on: float = 0.0,
                       min_off: float = 0.0) -> VoiceActivityDetection:
    """
    pyannote/brouhaha modelini segmentation olarak kullanıp
    VoiceActivityDetection pipeline'ını 3.x ile kurar.
    """
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


def build_osd(hf_token: str,
              onset: float = 0.7,
              offset: float = 0.7,
              min_on: float = 0.10,
              min_off: float = 0.10) -> OverlappedSpeechDetection:
    print("🔧 OSD modeli yükleniyor (pyannote/segmentation-3.0)...")
    # 3.x ile uyumlu segmentation checkpoint'i kullan
    seg_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    print(f"⚙️  OSD parametreleri: onset={onset}, offset={offset}, min_on={min_on}, min_off={min_off}")
    osd = OverlappedSpeechDetection(segmentation=seg_model)
    # 3.x OSD hiperparametreleri: onset/offset + min_duration_on/off
    osd.instantiate({
        "min_duration_on": min_on,
        "min_duration_off": min_off,
    })
    print("✅ OSD modeli hazır")
    return osd


def build_diarization(hf_token: str, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> Pipeline:
    """
    speaker-diarization-3.1 pipeline'ını yükler.
    """
    print("🔧 Diarization modeli yükleniyor (pyannote/speaker-diarization-3.1)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # Şimdilik varsayılan parametrelerle - konuşmacı kısıtlamaları post-processing'de yapılacak
    print("⚙️  Varsayılan diarization parametreleri kullanılıyor")
    if min_speakers is not None or max_speakers is not None:
        print(f"ℹ️  Konuşmacı kısıtlamaları (min={min_speakers}, max={max_speakers}) şimdilik göz ardı ediliyor")

    pipeline.instantiate({})

    print("✅ Diarization modeli hazır")
    return pipeline


def timeline_coverage_ratio(timeline, word_seg: Segment) -> float:
    """
    timeline (pyannote.core.Timeline) ile kelime segmenti kesişim oranı.
    """
    covered = 0.0
    for seg in timeline:
        inter = seg & word_seg
        if inter is not None:
            covered += inter.duration
    return min(1.0, covered / max(1e-6, word_seg.duration))


def timeline_overlaps(timeline, word_seg: Segment) -> bool:
    """
    timeline herhangi bir seg ile kelime segmenti kesişiyor mu?
    """
    for seg in timeline:
        if (seg & word_seg) is not None:
            return True
    return False


SMOOTH_WIN = 0.15  # 150 ms: midpoint etrafında pencere

def assign_speaker_smooth(diarization, word_seg: Segment,
                          prev_speaker: Optional[str],
                          is_overlap: bool) -> Tuple[str, float]:
    """
    Midpoint ±150ms penceresinde çoğunluk konuşmacıyı seç.
    Overlap anında gereksiz hoplamayı önlemek için önceki konuşmacıya sadakat uygula.
    """
    t = 0.5 * (word_seg.start + word_seg.end)
    win = Segment(max(0.0, t - SMOOTH_WIN), t + SMOOTH_WIN)

    scores = defaultdict(float)
    total = 0.0
    for turn, _, label in diarization.itertracks(yield_label=True):
        inter = (turn & win)
        if inter is not None:
            dur = inter.duration
            scores[label] += dur
            total += dur

    if not scores:
        return assign_speaker(diarization, word_seg)

    speaker = max(scores.items(), key=lambda kv: kv[1])[0]
    conf = min(1.0, scores[speaker] / max(total, 1e-6))
    if is_overlap and prev_speaker is not None and speaker != prev_speaker:
        if scores[speaker] < 1.25 * scores.get(prev_speaker, 0.0):
            speaker = prev_speaker
    return speaker, conf

def assign_speaker(diarization, word_seg: Segment) -> Tuple[str, float]:
    """
    Kelime segmentini, kesişim süresine göre en olası konuşmacıya atar.
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


def transcribe_words(asr_model: WhisperModel,
                     audio_path: str) -> List[dict]:
    """
    faster-whisper ile kelime zaman damgaları. İç VAD kapalı.
    """
    print("🎙️  ASR transkripsiyon başlıyor...")
    results = []
    segments, _info = asr_model.transcribe(
        audio_path,
        task="transcribe",
        vad_filter=False,
        word_timestamps=True,
        beam_size=5,
        temperature=0.0,
    )

    word_count = 0
    for seg in segments:
        if seg.words is None:
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            # faster-whisper Word: .word, .start, .end, .probability (bazı sürümlerde olabilir)
            conf = None
            if hasattr(w, "probability") and w.probability is not None:
                conf = float(w.probability)
            # conf yoksa segment üzerinden tahmini 0.9 ver (sabit değil, istenirse parametre yapılabilir)
            conf = conf if conf is not None else 0.9

            results.append({
                "word": w.word.strip(),
                "start": float(w.start),
                "end": float(w.end),
                "confidence": float(conf)
            })
            word_count += 1

    print(f"✅ ASR tamamlandı: {word_count} kelime tespit edildi")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="brouhaha VAD + OSD + diarization + faster-whisper word-level JSONL v4 - JSON çıktılarıyla"
    )
    parser.add_argument("--audio", required=True, help="Girdi WAV dosyası yolu")
    parser.add_argument("--out", required=True, help="Çıkış JSONL dosyası")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face access token (HF_TOKEN env değişkeni önerilir)")
    parser.add_argument("--asr-model", default="large-v3",
                        help="faster-whisper modeli (örn. tiny, base, small, medium, large-v3)")
    parser.add_argument("--asr-device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--asr-compute-type", default="auto",
                        help="auto|int8|int8_float16|float16|float32 ...")
    parser.add_argument("--vad-onset", type=float, default=0.5)
    parser.add_argument("--vad-offset", type=float, default=0.5)
    parser.add_argument("--vad-min-on", type=float, default=0.0)
    parser.add_argument("--vad-min-off", type=float, default=0.0)
    parser.add_argument("--osd-onset", type=float, default=0.7, help="OSD karar eşiği (konservatif için 0.7 önerilir)")
    parser.add_argument("--osd-offset", type=float, default=0.7, help="OSD kapama eşiği")
    parser.add_argument("--osd-min-on", type=float, default=0.10, help="Çok kısa overlap'ları buda (saniye)")
    parser.add_argument("--osd-min-off", type=float, default=0.10, help="Kısa boşlukları doldur (saniye)")
    parser.add_argument("--require-vad", action="store_true",
                        help="Kelimenin en az %X'i VAD konuşma bölgesinde değilse dışla")
    parser.add_argument("--vad-coverage", type=float, default=0.6,
                        help="require-vad aktifse minimum VAD kapsama oranı (0-1)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Diarization için minimum konuşmacı sayısı ipucu")
    parser.add_argument("--max-speakers", type=int, default=None, help="Diarization için maksimum konuşmacı sayısı ipucu")
    parser.add_argument("--output-dir", default=None, help="Adım çıktıları için klasör (belirtilmezse ana dosya dizini kullanılır)")
    args = parser.parse_args()

    if not args.hf_token:
        print("❌ HF token gerekli. --hf-token ya da HF_TOKEN ortam değişkenini kullan.", file=sys.stderr)
        sys.exit(1)

    audio_path = args.audio


    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(args.out) or "."

    print(f"🎵 Audio dosyası: {audio_path}")
    print(f"📝 Çıkış dosyası: {args.out}")
    print(f"📁 Adım çıktı dizini: {output_dir}")
    print(f"🤖 ASR modeli: {args.asr_model} (device: {args.asr_device}, compute: {args.asr_compute_type})")
    print("=" * 60)

    # 1) VAD (brouhaha)
    print("📍 ADIM 1: Voice Activity Detection (VAD)")
    vad_pipeline = build_brouhaha_vad(
        hf_token=args.hf_token,
        onset=args.vad_onset,
        offset=args.vad_offset,
        min_on=args.vad_min_on,
        min_off=args.vad_min_off
    )
    print("🔍 VAD analizi yapılıyor...")
    vad_ann = vad_pipeline(audio_path)  # Annotation
    vad_timeline = vad_ann.get_timeline().support()
    vad_segments = len(vad_timeline)
    vad_total_duration = sum(seg.duration for seg in vad_timeline)
    print(f"✅ VAD tamamlandı: {vad_segments} konuşma segmenti, toplam {vad_total_duration:.2f}s")

    # VAD çıktısını kaydet
    vad_data = {
        "step": "voice_activity_detection",
        "model": "pyannote/brouhaha",
        "parameters": {
            "onset": args.vad_onset,
            "offset": args.vad_offset,
            "min_duration_on": args.vad_min_on,
            "min_duration_off": args.vad_min_off
        },
        "results": {
            "segments": timeline_to_dict(vad_timeline),
            "total_segments": vad_segments,
            "total_duration": round3(vad_total_duration)
        }
    }
    save_step_output("step1_vad", vad_data, output_dir)
    print()

    # 2) OSD (Overlapped Speech Detection)
    print("📍 ADIM 2: Overlapped Speech Detection (OSD)")
    osd_pipeline = build_osd(
        hf_token=args.hf_token,
        onset=args.osd_onset,
        offset=args.osd_offset,
        min_on=args.osd_min_on,
        min_off=args.osd_min_off
    )
    print("🔍 OSD analizi yapılıyor...")
    osd_ann = osd_pipeline(audio_path)  # Annotation
    osd_timeline = osd_ann.get_timeline().support()
    osd_segments = len(osd_timeline)
    osd_total_duration = sum(seg.duration for seg in osd_timeline)
    print(f"✅ OSD tamamlandı: {osd_segments} overlap segmenti, toplam {osd_total_duration:.2f}s")

    # OSD çıktısını kaydet
    osd_data = {
        "step": "overlapped_speech_detection",
        "model": "pyannote/segmentation-3.0",
        "parameters": {
            "onset": args.osd_onset,
            "offset": args.osd_offset,
            "min_duration_on": args.osd_min_on,
            "min_duration_off": args.osd_min_off
        },
        "results": {
            "segments": timeline_to_dict(osd_timeline),
            "total_segments": osd_segments,
            "total_duration": round3(osd_total_duration)
        }
    }
    save_step_output("step2_osd", osd_data, output_dir)
    print()

    # 3) Diarization
    print("📍 ADIM 3: Speaker Diarization")
    diar_pipeline = build_diarization(args.hf_token, args.min_speakers, args.max_speakers)
    print("🔍 Diarization analizi yapılıyor...")
    diar = diar_pipeline(audio_path)  # Annotation
    speakers = set()
    diar_segments = 0
    for turn, _, label in diar.itertracks(yield_label=True):
        speakers.add(label)
        diar_segments += 1
    print(f"✅ Diarization tamamlandı: {len(speakers)} konuşmacı ({', '.join(sorted(speakers))}), {diar_segments} segment")

    # Diarization çıktısını kaydet
    diar_data = {
        "step": "speaker_diarization",
        "model": "pyannote/speaker-diarization-3.1",
        "parameters": {
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers
        },
        "results": {
            "segments": diarization_to_dict(diar),
            "speakers": sorted(list(speakers)),
            "total_speakers": len(speakers),
            "total_segments": diar_segments
        }
    }
    save_step_output("step3_diarization", diar_data, output_dir)
    print()

    # 4) ASR (word timestamps)
    print("📍 ADIM 4: Automatic Speech Recognition (ASR)")
    print(f"🔧 ASR modeli yükleniyor: {args.asr_model}...")
    asr = WhisperModel(args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type)
    print("✅ ASR modeli hazır")
    words = transcribe_words(asr, audio_path)

    # ASR çıktısını kaydet
    asr_data = {
        "step": "automatic_speech_recognition",
        "model": args.asr_model,
        "parameters": {
            "device": args.asr_device,
            "compute_type": args.asr_compute_type,
            "vad_filter": False,
            "word_timestamps": True,
            "beam_size": 5,
            "temperature": 0.0
        },
        "results": {
            "words": words,
            "total_words": len(words)
        }
    }
    save_step_output("step4_asr", asr_data, output_dir)
    print()

    # 5) Kelime -> konuşmacı + overlap + VAD filtresi
    print("📍 ADIM 5: Kelime-seviye analiz ve JSONL yazımı")
    print("🔄 Kelimeler işleniyor...")

    written_words = 0
    filtered_words = 0
    overlap_words = 0
    prev_speaker = None
    final_results = []

    with open(args.out, "w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            seg = Segment(w["start"], w["end"])

            # VAD kapsaması
            coverage = timeline_coverage_ratio(vad_timeline, seg)
            if args.require_vad and (coverage < args.vad_coverage):
                filtered_words += 1
                continue  # konuşma değilse atla

            # overlap var mı?
            is_ov = timeline_overlaps(osd_timeline, seg)
            if is_ov:
                overlap_words += 1

            # konuşmacı ataması + güven (smooth algorithm ile)
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

            # İlerleme göstergesi
            if (i + 1) % 100 == 0:
                print(f"  📝 İşlenen kelime: {i + 1}/{len(words)}")

    print(f"✅ JSONL yazımı tamamlandı: {args.out}")

    # Son adım çıktısını kaydet
    final_data = {
        "step": "final_word_level_analysis",
        "parameters": {
            "require_vad": args.require_vad,
            "vad_coverage": args.vad_coverage,
            "smooth_window": SMOOTH_WIN
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
                "osd_segments": osd_segments
            }
        }
    }
    save_step_output("step5_final", final_data, output_dir)
    print()

    # Özet JSON'ını kaydet
    summary_data = {
        "pipeline_summary": {
            "audio_file": audio_path,
            "output_file": args.out,
            "output_directory": output_dir,
            "models_used": {
                "vad": "pyannote/brouhaha",
                "osd": "pyannote/segmentation-3.0",
                "diarization": "pyannote/speaker-diarization-3.1",
                "asr": args.asr_model
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
                "diarization_segments": diar_segments
            }
        }
    }
    save_step_output("pipeline_summary", summary_data, output_dir)

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