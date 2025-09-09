#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
SES DOSYASI DİARİZASYON VE TRANSKRİPSİYON SİSTEMİ
=============================================================================

Bu program şunları yapar:
1. Ses dosyasını (MP4/MP3/WAV) alır
2. Kimin ne zaman konuştuğunu tespit eder (diarizasyon)
3. Konuşulanları metne döker (transkripsiyon)
4. Her kelimeyi doğru konuşmacıya atar
5. Anlamlı cümle segmentleri oluşturur

Kullanılan teknolojiler:
- PyAnnote: Konuşmacı tespiti için
- Whisper: Konuşma tanıma için
- FFmpeg: Ses formatı dönüşümü için
=============================================================================
"""

import httpx
import sys
import json
import tempfile
import argparse
import subprocess
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from bisect import insort
from datetime import datetime

import anthropic
import json
import os

import numpy as np
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from pyannote.core import Segment, Timeline, Annotation
from faster_whisper import WhisperModel
import warnings

warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")

# ========================= DEBUG LOGGER KURULUMU =========================

class ColoredFormatter(logging.Formatter):
    """Renkli log çıktıları için özel formatter"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{self.BOLD}[{record.levelname}]{self.RESET}"
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)

def setup_logger(debug_level: str = "INFO", log_file: Optional[str] = None):
    """
    Debug logger'ı yapılandır

    debug_level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: Log dosyası yolu (opsiyonel)
    """
    logger = logging.getLogger('diarization')
    logger.setLevel(getattr(logging, debug_level.upper()))

    # Console handler (renkli)
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        '%(asctime)s %(levelname)s %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (opsiyonel)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

# Global logger
logger = logging.getLogger('diarization')

# ========================= YARDIMCI FONKSİYONLAR =========================

def format_time(seconds: float) -> str:
    """Saniyeyi okunabilir formata çevir (MM:SS.mmm)"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"

def round3(x: Optional[float]) -> Optional[float]:
    """Sayıyı 3 ondalık basamağa yuvarla"""
    if x is None:
        return None
    result = round(x, 3)
    logger.debug(f"Yuvarlama: {x:.6f} → {result:.3f}")
    return result

def log_segment(seg: Segment, label: str = "Segment"):
    """Segment bilgilerini logla"""
    logger.debug(f"{label}: [{format_time(seg.start)} - {format_time(seg.end)}] "
                 f"(Süre: {seg.duration:.3f}s)")

def log_timeline(timeline: Timeline, label: str = "Timeline"):
    """Timeline bilgilerini logla"""
    total_duration = sum(s.duration for s in timeline)
    logger.info(f"{label}: {len(timeline)} segment, "
                f"Toplam süre: {total_duration:.2f}s")

    if logger.level <= logging.DEBUG:
        for i, seg in enumerate(timeline[:5]):  # İlk 5 segmenti göster
            log_segment(seg, f"  Segment {i+1}")
        if len(timeline) > 5:
            logger.debug(f"  ... ve {len(timeline)-5} segment daha")

# ========================= SES DÖNÜŞTÜRME =========================

def ffmpeg_to_wav_mono16k(src_path: str) -> str:
    """
    Ses dosyasını işlenebilir formata çevir

    NEDEN GEREKLİ?
    - PyAnnote ve Whisper modelleri 16kHz mono WAV formatında çalışır
    - MP4/MP3 gibi formatlar direkt işlenemez
    - Stereo ses mono'ya çevrilmeli (tek kanal)

    Args:
        src_path: Kaynak ses dosyası (MP4, MP3, WAV, vb.)

    Returns:
        Dönüştürülmüş WAV dosyasının yolu
    """
    logger.info(f"Ses dosyası dönüştürülüyor: {src_path}")

    # Dosya kontrolü
    if not os.path.exists(src_path):
        logger.error(f"Dosya bulunamadı: {src_path}")
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {src_path}")

    file_size = os.path.getsize(src_path) / (1024 * 1024)  # MB
    logger.info(f"Dosya boyutu: {file_size:.2f} MB")

    # Geçici WAV dosyası oluştur
    tmp_wav = tempfile.mktemp(suffix=".wav")
    logger.debug(f"Geçici dosya: {tmp_wav}")

    # FFmpeg komutu
    cmd = [
        "ffmpeg", "-y",           # Evet, üzerine yaz
        "-i", src_path,           # Girdi dosyası
        "-ac", "1",               # Audio channels: 1 (mono)
        "-ar", "16000",           # Audio rate: 16kHz
        "-vn",                    # Video no (videoyu dahil etme)
        "-f", "wav",              # Format: WAV
        tmp_wav                   # Çıktı dosyası
    ]

    logger.debug(f"FFmpeg komutu: {' '.join(cmd)}")

    try:
        # FFmpeg'i çalıştır
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # Dönüştürülen dosya boyutu
        wav_size = os.path.getsize(tmp_wav) / (1024 * 1024)
        logger.info(f"✓ Dönüştürme tamamlandı. WAV boyutu: {wav_size:.2f} MB")

        # FFmpeg çıktısını debug modunda göster
        if result.stderr and logger.level <= logging.DEBUG:
            logger.debug("FFmpeg çıktısı:")
            for line in result.stderr.split('\n')[:10]:
                if line.strip():
                    logger.debug(f"  {line}")

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg hatası: {e}")
        logger.error(f"Hata çıktısı: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        raise RuntimeError(f"FFmpeg dönüştürme hatası: {e}")

    return tmp_wav

# ========================= TIMELINE İŞLEMLERİ =========================

def timeline_to_dict(tl: Timeline) -> List[Dict]:
    """Timeline nesnesini JSON'a yazılabilir formata çevir"""
    result = []
    for seg in tl:
        result.append({
            "start": round3(seg.start),
            "end": round3(seg.end),
            "duration": round3(seg.duration)
        })

    logger.debug(f"Timeline → Dict: {len(tl)} segment dönüştürüldü")
    return result

def diarization_to_dict(di: Annotation) -> List[Dict]:
    """Diarization sonuçlarını JSON formatına çevir"""
    result = []
    speaker_times = defaultdict(float)  # Her konuşmacının toplam süresi

    for turn, _, label in di.itertracks(yield_label=True):
        result.append({
            "start": round3(turn.start),
            "end": round3(turn.end),
            "duration": round3(turn.duration),
            "speaker": label
        })
        speaker_times[label] += turn.duration

    # Konuşmacı istatistikleri
    logger.info(f"Konuşmacı istatistikleri:")
    for speaker, duration in sorted(speaker_times.items()):
        logger.info(f"  {speaker}: {duration:.1f}s ({duration/60:.1f} dakika)")

    return result

def timeline_coverage_ratio(timeline: Timeline, seg: Segment) -> float:
    """
    Bir segmentin timeline tarafından ne kadar kaplandığını hesapla

    KULLANIM AMACI:
    VAD (Voice Activity Detection) kontrolü için kullanılır.
    Bir kelimenin gerçekten konuşma içinde mi yoksa sessizlikte mi
    olduğunu anlamak için.
    """
    if seg.duration <= 0:
        return 0.0

    covered = 0.0
    overlaps = []

    for t in timeline:
        intersection = t & seg  # Kesişim
        if intersection is not None:
            covered += intersection.duration
            overlaps.append(intersection)

    ratio = min(1.0, covered / seg.duration)

    if logger.level <= logging.DEBUG and overlaps:
        logger.debug(f"Kapsama hesabı: Segment [{format_time(seg.start)}-{format_time(seg.end)}]")
        logger.debug(f"  {len(overlaps)} kesişim, Kapsama: {ratio:.2%}")

    return ratio

def timeline_overlaps(timeline: Timeline, seg: Segment) -> bool:
    """Segment ile timeline'ın kesişip kesişmediğini kontrol et"""
    for t in timeline:
        if (t & seg) is not None:
            logger.debug(f"Çakışma tespit edildi: [{format_time(seg.start)}-{format_time(seg.end)}]")
            return True
    return False

def intersect_timelines(t1: Timeline, t2: Timeline) -> Timeline:
    """
    İki timeline'ın kesişimini bul

    KULLANIM AMACI:
    Farklı algılama sistemlerinin (VAD, OSD, Diarization) sonuçlarını
    birleştirerek daha güvenilir çakışma bölgeleri bulmak için.
    """
    logger.debug(f"Timeline kesişimi hesaplanıyor: "
                 f"T1({len(t1)} seg) ∩ T2({len(t2)} seg)")

    intersections = []
    for a in t1:
        for b in t2:
            ab = a & b
            if ab is not None and ab.duration > 0:
                intersections.append(ab)

    result = Timeline(intersections).support() if intersections else Timeline()
    logger.debug(f"Kesişim sonucu: {len(result)} segment")

    return result

def build_overlap_from_diarization(diar: Annotation, min_count: int = 2) -> Timeline:
    """
    Diarization'dan çakışan konuşma bölgelerini tespit et

    NASIL ÇALIŞIR?
    1. Tüm konuşma başlangıç ve bitişlerini topla
    2. Zaman çizgisinde ilerle
    3. Her noktada kaç kişi konuşuyor say
    4. 2+ kişi konuşuyorsa = çakışma

    Bu "sweep line algoritması" olarak bilinir.
    """
    logger.info("Çakışan konuşma bölgeleri tespit ediliyor...")

    # Tüm zaman noktalarını topla
    # +1 = konuşma başlıyor, -1 = konuşma bitiyor
    bounds = []
    for turn, _, label in diar.itertracks(yield_label=True):
        insort(bounds, (turn.start, +1))  # Başlangıç
        insort(bounds, (turn.end, -1))    # Bitiş

    logger.debug(f"Toplam {len(bounds)} zaman noktası işlenecek")

    # Zaman çizgisinde ilerleyerek çakışmaları bul
    overlaps = []
    active_count = 0  # Şu anda konuşan kişi sayısı
    prev_time = None

    for time, delta in bounds:
        # Eğer 2+ kişi konuşuyorsa, bu aralığı çakışma olarak kaydet
        if prev_time is not None and time > prev_time and active_count >= min_count:
            overlap_seg = Segment(prev_time, time)
            overlaps.append(overlap_seg)

            if logger.level <= logging.DEBUG:
                logger.debug(f"  Çakışma: [{format_time(prev_time)}-{format_time(time)}] "
                             f"({active_count} konuşmacı)")

        active_count += delta
        prev_time = time

    result = Timeline(overlaps).support() if overlaps else Timeline()
    logger.info(f"✓ {len(result)} çakışma bölgesi bulundu")

    return result

# ========================= MODEL YÜKLEYİCİLER =========================

def build_vad(hf_token: str, onset=0.5, offset=0.5,
              min_on=0.0, min_off=0.0):
    """
    Voice Activity Detection (VAD) - PyAnnote 3.1.1 Uyumlu
    """
    logger.info("VAD yapılandırılıyor (PyAnnote 3.1.1)...")

    try:
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection

        logger.info("Segmentation modeli yükleniyor...")

        # PyAnnote 3.x model ismi
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=hf_token
        )
        logger.info("✓ Segmentation modeli yüklendi")

        # VAD pipeline oluştur
        vad = VoiceActivityDetection(segmentation=model)

        # PyAnnote 3.x parametreleri - onset/offset YOK!
        HYPER_PARAMETERS = {
            "min_duration_on": min_on,    # onset yerine bu
            "min_duration_off": min_off,   # offset yerine bu
        }

        try:
            vad.instantiate(HYPER_PARAMETERS)
            logger.info("✓ VAD parametreleri ayarlandı")
        except Exception as e:
            logger.warning(f"Parametreler uygulanamadı: {e}")
            logger.info("Varsayılan parametrelerle devam ediliyor...")
            vad.instantiate({})  # Boş dict ile başlat

        logger.info("✓ VAD sistemi hazır")
        return vad

    except Exception as e:
        logger.error(f"VAD modeli yüklenemedi: {e}")
        logger.warning("Alternatif: Diarization tabanlı VAD kullanılacak")

        class DiarizationBasedVAD:
            def __init__(self):
                self.is_fallback = True
                self.vad_timeline = None
                logger.info("DiarizationBasedVAD yedek sistemi aktif")

            def __call__(self, wav_path):
                logger.info("VAD bilgisi diarization'dan çıkarılacak")
                from pyannote.core import Timeline
                return Timeline()

        return DiarizationBasedVAD()

def build_osd(hf_token: str, min_on=0.10, min_off=0.10) -> OverlappedSpeechDetection:
    """
    Overlapped Speech Detection (OSD) modeli oluştur

    OSD NEDİR?
    İki veya daha fazla kişinin aynı anda konuştuğu bölümleri tespit eder.
    Kesişen konuşmaları bulmak kritik çünkü bu bölgelerde
    konuşmacı ataması zor ve hata payı yüksek.
    """
    logger.info("OSD (Çakışan Konuşma) modeli yükleniyor...")
    logger.debug(f"Parametreler: min_on={min_on}, min_off={min_off}")

    try:
        seg_model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=hf_token
        )
        logger.info("✓ OSD modeli yüklendi")

        osd = OverlappedSpeechDetection(segmentation=seg_model)
        osd.instantiate({
            "min_duration_on": min_on,
            "min_duration_off": min_off
        })

        return osd

    except Exception as e:
        logger.error(f"OSD modeli yüklenemedi: {e}")
        raise

def build_diarization(hf_token: str,
                      min_speakers: Optional[int] = None,
                      max_speakers: Optional[int] = None) -> Pipeline:
    """
    Speaker Diarization Pipeline - PyAnnote 3.1.1 Uyumlu
    """
    logger.info("Diarization pipeline yükleniyor (PyAnnote 3.1.1)...")

    if min_speakers:
        logger.info(f"Minimum konuşmacı sayısı: {min_speakers}")
    if max_speakers:
        logger.info(f"Maksimum konuşmacı sayısı: {max_speakers}")

    try:
        from pyannote.audio import Pipeline

        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        logger.info("✓ Diarization pipeline yüklendi")

        # PyAnnote 3.x'te instantiate() boş dict alıyor
        pipe.instantiate({})

        # min/max speakers için wrapper
        if min_speakers is not None or max_speakers is not None:
            class ParameterizedDiarization:
                def __init__(self, pipeline, min_spk, max_spk):
                    self.pipeline = pipeline
                    self.min_speakers = min_spk
                    self.max_speakers = max_spk

                def __call__(self, audio_file, **kwargs):
                    if self.min_speakers is not None:
                        kwargs['min_speakers'] = self.min_speakers
                    if self.max_speakers is not None:
                        kwargs['max_speakers'] = self.max_speakers
                    return self.pipeline(audio_file, **kwargs)

            logger.info(f"Konuşmacı limitleri ayarlandı")
            return ParameterizedDiarization(pipe, min_speakers, max_speakers)

        return pipe

    except Exception as e:
        logger.error(f"Diarization pipeline yüklenemedi: {e}")
        raise RuntimeError(f"Diarization başlatılamadı: {e}")

# ========================= KONUŞMACI ATAMA LOJİĞİ =========================

# Konuşmacı atama parametreleri (fine-tuning için)
SMOOTH_WIN = 0.05      # 50ms pencere - çok hızlı konuşmacı değişimlerini yumuşatır
STICKY_RATIO = 0.5     # Önceki konuşmacıya yapışma - gürültüyü azaltır
LOCAL_TURN_BIAS = 0.6  # Kısa konuşmalara öncelik - kesintileri yakalar

def assign_speaker_midwin(diar: Annotation,
                          word_seg: Segment,
                          prev_spk: Optional[str],
                          is_overlap: bool) -> Tuple[str, float]:
    """
    Kelime segmentine en uygun konuşmacıyı ata

    NASIL ÇALIŞIR?
    1. Kelimenin orta noktası etrafında küçük bir pencere aç
    2. Bu pencerede en çok konuşan kişiyi bul
    3. Çakışma varsa özel stratejiler uygula
    4. Güven skoru hesapla

    STRATEJİLER:
    - Normal durumda: En çok konuşan kazanır
    - Çakışmada: Kısa/keskin konuşmalara öncelik
    - Belirsizlikte: Önceki konuşmacıya yapış
    """
    # Kelimenin orta noktası
    tmid = (word_seg.start + word_seg.end) / 2

    # Analiz penceresi
    win = Segment(max(0.0, tmid - SMOOTH_WIN), tmid + SMOOTH_WIN)

    logger.debug(f"Kelime [{format_time(word_seg.start)}-{format_time(word_seg.end)}] "
                 f"için konuşmacı atanıyor")
    logger.debug(f"  Analiz penceresi: [{format_time(win.start)}-{format_time(win.end)}]")

    # Penceredeki her konuşmacının konuşma süresi
    scores = defaultdict(float)
    total = 0.0
    active_at_mid = []  # Orta noktada aktif konuşmacılar

    for turn, _, label in diar.itertracks(yield_label=True):
        # Pencere ile kesişim
        inter = turn & win
        if inter is not None:
            duration = inter.duration
            scores[label] += duration
            total += duration
            logger.debug(f"    {label}: {duration:.3f}s")

        # Orta noktada aktif mi?
        if turn.start <= tmid < turn.end:
            active_at_mid.append((turn, label))

    # Hiç skor yoksa (nadir durum)
    if not scores:
        logger.warning("Pencerede konuşmacı bulunamadı, tam kesişime bakılıyor")

        best_label = "SPEAKER_00"
        best_duration = 0.0

        for turn, _, label in diar.itertracks(yield_label=True):
            inter = turn & word_seg
            if inter is not None and inter.duration > best_duration:
                best_duration = inter.duration
                best_label = label

        confidence = min(1.0, best_duration / max(1e-6, word_seg.duration))
        logger.debug(f"  Sonuç: {best_label} (güven: {confidence:.2%})")
        return best_label, confidence

    # ÇAKIŞMA STRATEJİSİ
    if is_overlap and len(active_at_mid) >= 2:
        logger.debug("  ⚠ Çakışma tespit edildi, özel strateji uygulanıyor")

        # Konuşmaları sürelerine göre sırala
        active_at_mid.sort(key=lambda x: x[0].duration)
        short_turn, short_label = active_at_mid[0]
        long_turn, _ = active_at_mid[-1]

        # Kısa konuşma muhtemelen kesinti/müdahale
        if short_turn.duration <= LOCAL_TURN_BIAS * long_turn.duration:
            confidence = min(1.0, scores.get(short_label, 0) / max(total, 1e-6))
            logger.debug(f"  Kısa konuşmaya öncelik: {short_label} "
                         f"({short_turn.duration:.2f}s < {LOCAL_TURN_BIAS}*{long_turn.duration:.2f}s)")
            return short_label, confidence

    # NORMAL DURUM: En yüksek skoru seç
    candidate = max(scores.items(), key=lambda kv: kv[1])[0]
    confidence = min(1.0, scores[candidate] / max(total, 1e-6))

    logger.debug(f"  En yüksek skor: {candidate} ({scores[candidate]:.3f}s / {total:.3f}s)")

    # YAPIŞTIRMA STRATEJİSİ (smoothing)
    if is_overlap and prev_spk and candidate != prev_spk:
        prev_score = scores.get(prev_spk, 0.0)
        if scores[candidate] < STICKY_RATIO * prev_score:
            logger.debug(f"  Önceki konuşmacıya yapışılıyor: {prev_spk} "
                         f"({scores[candidate]:.3f} < {STICKY_RATIO}*{prev_score:.3f})")
            candidate = prev_spk

    logger.debug(f"  ✓ Sonuç: {candidate} (güven: {confidence:.2%})")
    return candidate, confidence

# ========================= ASR (KONUŞMA TANIMA) =========================

def transcribe_words(asr: WhisperModel,
                     audio_path: str,
                     lang: Optional[str]) -> List[Dict]:
    """
    Ses dosyasını metne çevir ve kelime zaman damgalarını al

    WHISPER NEDİR?
    OpenAI'ın geliştirdiği, 680.000 saat ses verisiyle eğitilmiş
    çok dilli konuşma tanıma modeli. 100+ dili destekler.

    KELİME ZAMAN DAMGALARI:
    Her kelimenin tam olarak ne zaman söylendiğini milisaniye
    hassasiyetinde tespit eder. Bu, konuşmacı ataması için kritik.
    """
    # Model bilgisini güvenli şekilde al
    # faster-whisper'da model boyutu farklı saklanıyor
    model_info = "Whisper Model"  # Varsayılan
    try:
        # Farklı olası attribute isimlerini dene
        if hasattr(asr, 'model_size'):
            model_info = asr.model_size
        elif hasattr(asr, 'model'):
            model_info = str(asr.model)
        else:
            # Model bilgisi bulunamazsa, sadece "Whisper" yaz
            model_info = "Whisper"
    except:
        pass

    logger.info(f"Transkripsiyon başlıyor... (Model: {model_info})")
    if lang:
        logger.info(f"Dil: {lang}")

    start_time = datetime.now()

    # Transkripsiyon parametreleri
    segments, info = asr.transcribe(
        audio_path,
        task="transcribe",           # "translate" değil, orijinal dilde tut
        vad_filter=False,            # VAD'yi kendimiz yapıyoruz
        word_timestamps=True,        # ⚠ KRİTİK: Kelime zamanlamaları
        beam_size=10,                # Daha yüksek = daha doğru ama yavaş
        temperature=0.0,             # 0 = deterministik (tutarlı sonuçlar)
        language=lang                # Dil tespitini atla, hızlandır
    )

    # Tespit edilen dil bilgisini kontrol et
    if hasattr(info, 'language') and info.language:
        logger.info(f"Tespit edilen dil: {info.language}")

    # Kelimeleri topla
    words = []
    total_segments = 0
    skipped_words = 0

    for seg in segments:
        total_segments += 1

        if not seg.words:
            logger.debug(f"Segment {total_segments}: Kelime yok, atlanıyor")
            continue

        logger.debug(f"Segment {total_segments}: {len(seg.words)} kelime")

        for word in seg.words:
            # Zaman damgası olmayan kelimeleri atla
            if word.start is None or word.end is None:
                skipped_words += 1
                logger.debug(f"  ⚠ Kelime atlandı (zaman yok): '{word.word}'")
                continue

            word_dict = {
                "word": word.word.strip(),
                "start": float(word.start),
                "end": float(word.end),
                "confidence": float(getattr(word, "probability", 0.9))
            }

            words.append(word_dict)

            # İlk birkaç kelimeyi logla (debug modunda)
            if len(words) <= 5 and logger.level <= logging.DEBUG:
                logger.debug(f"  Kelime {len(words)}: '{word_dict['word']}' "
                             f"[{format_time(word_dict['start'])}-{format_time(word_dict['end'])}] "
                             f"(güven: {word_dict['confidence']:.2%})")

    # İstatistikler
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ Transkripsiyon tamamlandı:")
    logger.info(f"  Süre: {elapsed:.1f}s")
    logger.info(f"  Segment: {total_segments}")
    logger.info(f"  Kelime: {len(words)}")
    if skipped_words > 0:
        logger.warning(f"  Atlanan: {skipped_words} kelime (zaman damgası yok)")

    return words



def create_output_byspeaker(debug_dir: str, segments: List[Dict],
                            speakers: List[str]) -> str:
    """
    Segmentleri konuşmacılara göre organize et ve ayrı bir dosya oluştur

    Bu fonksiyon, ana çıktıdaki segmentleri alır ve her konuşmacının
    ne dediğini ayrı ayrı gruplar. Kelime kelime değil, anlamlı
    cümle/segment halinde organize eder.

    Args:
        debug_dir: Debug dosyalarının bulunduğu klasör
        segments: Ana programdan gelen konuşma segmentleri (text içeren)
        speakers: Tespit edilen konuşmacı listesi

    Returns:
        Oluşturulan dosyanın yolu
    """
    output_path = os.path.join(debug_dir, "output_byspeaker.json")

    # Segmentleri konuşmacılara göre grupla
    segments_by_speaker = defaultdict(list)

    for segment in segments:
        speaker = segment["speaker"]
        # Her segmenti olduğu gibi ekle (text dahil)
        segments_by_speaker[speaker].append({
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],  # Birleştirilmiş metin
            "duration": segment["duration"],
            "word_count": segment["word_count"]
        })

    # Her konuşmacı için segmentleri zaman sırasına göre sırala
    for speaker in segments_by_speaker:
        segments_by_speaker[speaker].sort(key=lambda x: x["start"])

    # Çıktı formatını oluştur
    output_data = {
        "description": "Konuşmacılara göre organize edilmiş segmentler",
        "creation_time": datetime.now().isoformat(),
        "speakers": speakers,
        "total_segments": len(segments),
        "segments_by_speaker": {}
    }

    # Her konuşmacı için detaylı bilgi oluştur
    for speaker in speakers:
        speaker_segments = segments_by_speaker.get(speaker, [])

        if speaker_segments:
            # İstatistikler
            total_duration = sum(seg["duration"] for seg in speaker_segments)
            total_words = sum(seg["word_count"] for seg in speaker_segments)
            first_time = speaker_segments[0]["start"]
            last_time = speaker_segments[-1]["end"]

            # Tüm metinleri birleştir (opsiyonel - tam transkript için)
            full_text = " ".join(seg["text"] for seg in speaker_segments)

        else:
            total_duration = 0
            total_words = 0
            first_time = None
            last_time = None
            full_text = ""

        output_data["segments_by_speaker"][speaker] = {
            "segment_count": len(speaker_segments),
            "total_duration": round3(total_duration),
            "total_words": total_words,
            "first_time": round3(first_time) if first_time is not None else None,
            "last_time": round3(last_time) if last_time is not None else None,
            "segments": speaker_segments,  # Text içeren segmentler
            "full_text": full_text  # Tüm konuşmaların birleşimi
        }

    # Genel istatistikler
    total_duration_all = sum(
        data["total_duration"]
        for data in output_data["segments_by_speaker"].values()
    )

    output_data["statistics"] = {}
    for speaker in speakers:
        speaker_data = output_data["segments_by_speaker"][speaker]
        output_data["statistics"][speaker] = {
            "segment_percentage": round(
                100 * speaker_data["segment_count"] / max(len(segments), 1), 2
            ),
            "duration_percentage": round(
                100 * speaker_data["total_duration"] / max(total_duration_all, 1), 2
            ),
            "word_percentage": round(
                100 * speaker_data["total_words"] /
                max(sum(s["total_words"] for s in output_data["segments_by_speaker"].values()), 1), 2
            )
        }

    # Dosyayı kaydet
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Konuşmacı bazlı çıktı oluşturuldu: {output_path}")

    # İstatistikleri logla
    logger.info("Konuşmacı segment dağılımı:")
    for speaker in speakers:
        seg_count = output_data["segments_by_speaker"][speaker]["segment_count"]
        duration = output_data["segments_by_speaker"][speaker]["total_duration"]
        percentage = output_data["statistics"][speaker]["duration_percentage"]
        logger.info(f"  {speaker}: {seg_count} segment, {duration:.1f}s ({percentage}%)")

    return output_path




# ========================= SEGMENT BİRLEŞTİRME =========================

def merge_words_into_segments(tagged_words: List[Dict],
                              max_gap: float = 0.6,
                              max_len: float = 12.0) -> List[Dict]:
    """
    Kelimeleri anlamlı cümle segmentlerine birleştir

    NEDEN GEREKLİ?
    Kelime kelime çıktı okunması zor. İnsanlar cümleler halinde
    konuşur. Bu fonksiyon kelimeleri doğal cümle yapılarına dönüştürür.

    STRATEJİ:
    - Aynı konuşmacının ardışık kelimeleri birleştir
    - Uzun suskunlukta böl (max_gap)
    - Çok uzun segmentleri böl (max_len)
    """
    if not tagged_words:
        logger.warning("Birleştirilecek kelime yok")
        return []

    logger.info(f"Kelimeler cümlelere birleştiriliyor... "
                f"({len(tagged_words)} kelime)")
    logger.debug(f"Parametreler: max_gap={max_gap}s, max_len={max_len}s")

    segments = []
    current = None

    for i, word in enumerate(tagged_words):
        if current is None:
            # İlk segment
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "words": [word["word"]],
                "overlap_count": 1 if word.get("is_overlap") else 0
            }
            logger.debug(f"Yeni segment başladı: {word['speaker']} @ {format_time(word['start'])}")
            continue

        # Birleştirme kriterleri
        same_speaker = (word["speaker"] == current["speaker"])
        gap = word["start"] - current["end"]
        total_length = word["end"] - current["start"]

        gap_ok = gap <= max_gap
        length_ok = total_length <= max_len

        if logger.level <= logging.DEBUG and i < 10:  # İlk 10 kelime için detay
            logger.debug(f"  Kelime {i}: '{word['word']}' - "
                         f"Aynı konuşmacı: {same_speaker}, "
                         f"Boşluk: {gap:.2f}s, "
                         f"Toplam: {total_length:.2f}s")

        if same_speaker and gap_ok and length_ok:
            # Mevcut segmente ekle
            current["end"] = word["end"]
            current["words"].append(word["word"])
            if word.get("is_overlap"):
                current["overlap_count"] += 1
        else:
            # Yeni segment başlat
            segments.append(current)

            if not same_speaker:
                logger.debug(f"Konuşmacı değişti: {current['speaker']} → {word['speaker']}")
            elif not gap_ok:
                logger.debug(f"Uzun boşluk: {gap:.2f}s > {max_gap}s")
            elif not length_ok:
                logger.debug(f"Segment çok uzun: {total_length:.2f}s > {max_len}s")

            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "words": [word["word"]],
                "overlap_count": 1 if word.get("is_overlap") else 0
            }

    # Son segmenti ekle
    if current is not None:
        segments.append(current)

    logger.info(f"✓ {len(segments)} segment oluşturuldu")

    # Formatla ve istatistikleri hesapla
    formatted = []
    speaker_segments = defaultdict(int)

    for i, s in enumerate(segments):
        overlap_ratio = s["overlap_count"] / len(s["words"])

        formatted_seg = {
            "id": i,
            "start": round3(s["start"]),
            "end": round3(s["end"]),
            "text": " ".join(s["words"]).strip(),
            "speaker": s["speaker"],
            "duration": round3(s["end"] - s["start"]),
            "word_count": len(s["words"]),
            "overlap_ratio": round3(overlap_ratio)
        }

        formatted.append(formatted_seg)
        speaker_segments[s["speaker"]] += 1

        # İlk birkaç segmenti logla
        if i < 3 and logger.level <= logging.DEBUG:
            logger.debug(f"Segment {i}: {s['speaker']} "
                         f"[{format_time(s['start'])}-{format_time(s['end'])}] "
                         f"({len(s['words'])} kelime): "
                         f"'{' '.join(s['words'][:5])}...'")

    # Segment istatistikleri
    logger.info("Segment dağılımı:")
    for speaker, count in sorted(speaker_segments.items()):
        logger.info(f"  {speaker}: {count} segment")

    return formatted

def filter_tiny_overlaps(overlap_tl: Timeline,
                         min_duration: float = 0.5) -> Timeline:
    """
    Çok kısa çakışma bölgelerini filtrele

    NEDEN?
    Çok kısa çakışmalar genelde yanlış algılamadır.
    0.5 saniyeden kısa çakışmalar muhtemelen gürültü.
    """
    logger.debug(f"Kısa çakışmalar filtreleniyor (min: {min_duration}s)")

    original_count = len(overlap_tl)
    filtered = []
    removed = []

    for seg in overlap_tl:
        if seg.duration >= min_duration:
            filtered.append(seg)
        else:
            removed.append(seg)

    if removed:
        logger.info(f"Filtrelenen çakışmalar: {len(removed)}/{original_count}")
        for r in removed[:5]:  # İlk 5 tanesini göster
            logger.debug(f"  Kaldırıldı: [{format_time(r.start)}-{format_time(r.end)}] "
                         f"({r.duration:.3f}s)")

    return Timeline(filtered).support() if filtered else Timeline()

def post_merge_tiny_segments(segments: List[Dict],
                             min_duration: float = 2.0,
                             max_gap: float = 1.5) -> List[Dict]:
    """
    Çok kısa segmentleri komşularıyla birleştir

    NEDEN?
    Kısa segmentler (< 2s) genelde kesik cümlelerdir:
    - "Evet." (0.5s)
    - "Hmm..." (0.3s)
    - "Peki." (0.4s)

    Bunları bir sonraki segmentle birleştirmek daha doğal.
    """
    if not segments:
        return segments

    logger.info(f"Kısa segmentler birleştiriliyor (min: {min_duration}s)...")

    merged = []
    i = 0
    merge_count = 0

    while i < len(segments):
        seg = segments[i]
        duration = seg["end"] - seg["start"]

        # Kısa segment ve sonraki segment varsa
        if duration < min_duration and i + 1 < len(segments):
            next_seg = segments[i + 1]
            gap = next_seg["start"] - seg["end"]

            # Birleştirme kriterleri
            same_speaker = (seg["speaker"] == next_seg["speaker"])
            gap_ok = gap <= max_gap

            if same_speaker and gap_ok:
                # Birleştir
                merged_seg = {
                    "id": len(merged),
                    "start": seg["start"],
                    "end": next_seg["end"],
                    "text": seg["text"] + " " + next_seg["text"],
                    "speaker": seg["speaker"],
                    "duration": round3(next_seg["end"] - seg["start"]),
                    "word_count": seg.get("word_count", 0) + next_seg.get("word_count", 0)
                }

                merged.append(merged_seg)
                merge_count += 1

                logger.debug(f"Birleştirildi: Segment {seg['id']} + {next_seg['id']} "
                             f"({duration:.1f}s + {next_seg['end']-next_seg['start']:.1f}s)")

                i += 2  # İki segmenti atla
                continue

        # Birleştirme yapılamadı
        seg_copy = seg.copy()
        seg_copy["id"] = len(merged)
        merged.append(seg_copy)
        i += 1

    if merge_count > 0:
        logger.info(f"✓ {merge_count} segment birleştirildi "
                    f"({len(segments)} → {len(merged)} segment)")

    return merged
def create_output_byspeaker_from_tagged_words(debug_dir: str,
                                              tagged_words: List[Dict],
                                              speakers: List[str],
                                              max_gap: float = 1.5,
                                              max_len: float = 30.0) -> str:
    """
    Tagged words'den direkt konuşmacı bazlı çıktı oluştur

    Args:
        debug_dir: Debug dosyalarının bulunduğu klasör
        tagged_words: Konuşmacı atamalı kelimeler
        speakers: Tespit edilen konuşmacı listesi
        max_gap: Segment birleştirme için maksimum boşluk (saniye)
        max_len: Maksimum segment uzunluğu (saniye)

    Returns:
        Oluşturulan dosyanın yolu
    """
    output_path = os.path.join(debug_dir, "output_byspeaker.json")

    # Her konuşmacı için kelimeleri segmentlere dönüştür
    segments_by_speaker = {}

    for speaker in speakers:
        # Bu konuşmacıya ait kelimeleri filtrele
        speaker_words = [w for w in tagged_words if w["speaker"] == speaker]

        if not speaker_words:
            segments_by_speaker[speaker] = {"segments": []}
            continue

        # Kelimeleri segmentlere birleştir
        segments = []
        current_segment = None
        segment_id = 0

        for word in speaker_words:
            if current_segment is None:
                # İlk segment
                current_segment = {
                    "id": segment_id,
                    "start": round3(word["start"]),
                    "end": round3(word["end"]),
                    "words": [word["word"]],
                    "speaker": speaker
                }
            else:
                # Birleştirme kriterleri
                gap = word["start"] - current_segment["end"]
                total_length = word["end"] - current_segment["start"]

                if gap <= max_gap and total_length <= max_len:
                    # Mevcut segmente ekle
                    current_segment["end"] = round3(word["end"])
                    current_segment["words"].append(word["word"])
                else:
                    # Yeni segment başlat
                    # Önce mevcut segmenti kaydet
                    current_segment["text"] = " ".join(current_segment["words"])
                    current_segment["duration"] = round3(
                        current_segment["end"] - current_segment["start"]
                    )
                    current_segment["word_count"] = len(current_segment["words"])
                    del current_segment["words"]  # words alanını kaldır
                    segments.append(current_segment)

                    segment_id += 1
                    current_segment = {
                        "id": segment_id,
                        "start": round3(word["start"]),
                        "end": round3(word["end"]),
                        "words": [word["word"]],
                        "speaker": speaker
                    }

        # Son segmenti ekle
        if current_segment:
            current_segment["text"] = " ".join(current_segment["words"])
            current_segment["duration"] = round3(
                current_segment["end"] - current_segment["start"]
            )
            current_segment["word_count"] = len(current_segment["words"])
            del current_segment["words"]  # words alanını kaldır
            del current_segment["speaker"]  # speaker alanını kaldır (zaten üst seviyede var)
            segments.append(current_segment)

        # Konuşmacıya ait segmentleri kaydet
        segments_by_speaker[speaker] = {
            "segments": segments
        }

    # Dosyayı kaydet
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments_by_speaker, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Konuşmacı bazlı çıktı oluşturuldu: {output_path}")

    # İstatistikleri logla
    logger.info("Konuşmacı segment dağılımı:")
    for speaker in speakers:
        seg_count = len(segments_by_speaker[speaker]["segments"])
        if seg_count > 0:
            total_duration = sum(s["duration"] for s in segments_by_speaker[speaker]["segments"])
            total_words = sum(s["word_count"] for s in segments_by_speaker[speaker]["segments"])
            logger.info(f"  {speaker}: {seg_count} segment, {total_duration:.1f}s, {total_words} kelime")

    return output_path


def process_dialog(input_json, out_json_path):
    http_client = httpx.Client(transport=transport)

    client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            http_client=http_client
        )
        # Diarization ve input parametrelerini ekledik
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128000,
        temperature=1,
        system="Verilen ardışık cümleleri mantıksal sıraya göre düzenle ve mantıklı dialog olacak sekilde aşağıdaki kriterlere uygun bir JSON çıktısı oluştur:\n\nGereksinimler:\n1. Cümleleri mantıksal ve dilbilgisi açısından doğru bir sıraya koy\n2. Her bir cümlenin yanına ISO 8601 formatında bir zaman damgası ekle\n3. JSON formatında çıktı ver\n4. Her cümlenin benzersiz bir ID'si olmalı\n5. Cümlelerin sıralamasını ve mantıksal akışını kontrol et\n6. en öneli adim {{SEGMENT_INPUT_JSON}} uygun olmali\nJSON Şablonu:\n{\n \"sentences\": [\n {\n \"id\": \"unique_id_1\",\n \"text\": \"Düzenlenmiş cümle\",\n \"start\": 0.0,\n \"end\": 3.64,\n \"duration\": 13.7,\n \"word_count\": 52,\n \"speaker\": \"SPEAKER_00\",\n },\n ...\n ]\n}",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n\n\nBeklenen Çıktı: Mantıksal sıralanmış, zaman damgalı JSON\n\n\nGirdi: \n{{INPUT_JSON}}\n{{SEGMENT_INPUT_JSON}}"
                    }
                ]
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 51200
        }
    )

    # Çıktıyı JSON dosyasına kaydet
    try:
        processed_dialog = json.loads(message.content[0].text)

        # Diarization bilgilerini ekle
        diarization_info = {
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "total_speakers": 2,
            "language": "tr",
            "processing_time": 0.5  # saniye cinsinden
        }

        processed_dialog["diarization"] = diarization_info

        # Çıktıyı dosyaya yaz
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_dialog, f, ensure_ascii=False, indent=2)

        print(f"İşlenen dialog {out_json_path} dosyasına kaydedildi.")
        return processed_dialog

    except json.JSONDecodeError as e:
        print(f"JSON Decode Hatası: {e}")
        return None
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return None


# ========================= ANA PROGRAM =========================

def main():

    """
    Ana program akışı:
    1. Parametreleri al
    2. Ses dosyasını hazırla
    3. Modelleri yükle
    4. Analizleri yap (VAD, OSD, Diarization)
    5. Transkripsiyon yap
    6. Konuşmacıları ata
    7. Segmentleri oluştur
    8. Sonuçları kaydet
    """
    parser = argparse.ArgumentParser(
        description="🎙️ Ses Dosyası Diarizasyon ve Transkripsiyon Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  python diarize.py --input meeting.mp4 --output result.json --hf-token YOUR_TOKEN
  
Debug modu:
  python diarize.py --input audio.wav --output out.json --hf-token TOKEN --debug DEBUG --log-file debug.log
        """
    )

    # Temel parametreler
    parser.add_argument("--input", required=True,
                        help="Girdi ses dosyası (MP4/MP3/WAV)")
    parser.add_argument("--output", required=True,
                        help="Çıktı JSON dosyası")
    parser.add_argument("--hf-token",
                        default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token (veya $HF_TOKEN)")

    # Debug parametreleri
    parser.add_argument("--debug", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Debug seviyesi")
    parser.add_argument("--log-file", default=None,
                        help="Log dosyası (opsiyonel)")
    parser.add_argument("--dump-debug", default=None,
                        help="Ara sonuçları kaydet (klasör yolu)")

    # ASR parametreleri
    parser.add_argument("--asr-model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model boyutu")
    parser.add_argument("--asr-device", default="auto",
                        help="İşlemci (cpu/cuda/auto)")
    parser.add_argument("--asr-compute-type", default="auto",
                        help="Hesaplama tipi (int8/float16/auto)")
    parser.add_argument("--language", default=None,
                        help="Transkripsiyon dili (tr, en, vb.)")

    # Diarization parametreleri
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Minimum konuşmacı sayısı")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Maksimum konuşmacı sayısı")

    # Filtreleme parametreleri
    parser.add_argument("--require-vad", action="store_true",
                        help="Sadece VAD tespit edilen bölgeleri kullan")
    parser.add_argument("--vad-coverage", type=float, default=0.6,
                        help="Minimum VAD kapsama oranı (0-1)")

    args = parser.parse_args()

    # Logger kurulumu
    global logger
    logger = setup_logger(args.debug, args.log_file)

    # Başlangıç
    logger.info("="*60)
    logger.info("🎙️  SES DOSYASI DİARİZASYON VE TRANSKRİPSİYON")
    logger.info("="*60)
    logger.info(f"Girdi: {args.input}")
    logger.info(f"Çıktı: {args.output}")
    logger.info(f"Debug seviyesi: {args.debug}")

    # Token kontrolü
    if not args.hf_token:
        logger.error("HuggingFace token gerekli! --hf-token veya $HF_TOKEN")
        sys.exit(1)

    try:
        # ============ AŞAMA 1: SES HAZIRLAMA ============
        logger.info("\n" + "="*40)
        logger.info("AŞAMA 1: SES DOSYASI HAZIRLAMA")
        logger.info("="*40)

        wav_path = ffmpeg_to_wav_mono16k(args.input)

        # ============ AŞAMA 2: MODEL YÜKLEME ============
        logger.info("\n" + "="*40)
        logger.info("AŞAMA 2: MODELLERİ YÜKLEME")
        logger.info("="*40)

        vad = build_vad(args.hf_token)
        osd = build_osd(args.hf_token)
        diar = build_diarization(args.hf_token,
                                 args.min_speakers,
                                 args.max_speakers)

        logger.info(f"\nWhisper modeli yükleniyor: {args.asr_model}")
        asr = WhisperModel(args.asr_model,
                           device=args.asr_device,
                           compute_type=args.asr_compute_type)
        logger.info("✓ Tüm modeller hazır")

        # ============ AŞAMA 3: SES ANALİZİ ============
        logger.info("\n" + "="*40)
        logger.info("AŞAMA 3: SES ANALİZİ")
        logger.info("="*40)

        # 3.1 Voice Activity Detection
        logger.info("\n--- VAD (Konuşma Tespiti) ---")
        vad_ann = vad(wav_path)
        vad_timeline = vad_ann.get_timeline().support()
        log_timeline(vad_timeline, "VAD Sonucu")

        # 3.2 Overlapped Speech Detection
        logger.info("\n--- OSD (Çakışma Tespiti) ---")
        osd_ann = osd(wav_path)
        osd_timeline = osd_ann.get_timeline().support()
        log_timeline(osd_timeline, "OSD Sonucu")

        # 3.3 Speaker Diarization
        logger.info("\n--- Diarization (Konuşmacı Ayrımı) ---")
        diar_ann = diar(wav_path)

        # Konuşmacı sayısı
        speakers = list(diar_ann.labels())
        logger.info(f"Tespit edilen konuşmacı sayısı: {len(speakers)}")
        logger.info(f"Konuşmacılar: {', '.join(speakers)}")

        # 3.4 Çakışma bölgelerini belirle
        logger.info("\n--- Çakışma Analizi ---")
        diar_overlap = build_overlap_from_diarization(diar_ann, min_count=2)
        osd_in_vad = intersect_timelines(osd_timeline, vad_timeline)
        final_overlap = intersect_timelines(osd_in_vad, diar_overlap)
        final_overlap = filter_tiny_overlaps(final_overlap, min_duration=0.5)
        log_timeline(final_overlap, "Final Çakışma Bölgeleri")

        # ============ AŞAMA 4: TRANSKRİPSİYON ============
        logger.info("\n" + "="*40)
        logger.info("AŞAMA 4: TRANSKRİPSİYON")
        logger.info("="*40)

        words = transcribe_words(asr, wav_path, args.language)

        # ============ AŞAMA 5: KONUŞMACI ATAMA ============
        logger.info("\n" + "="*40)
        logger.info("AŞAMA 5: KONUŞMACI ATAMA")
        logger.info("="*40)

        prev_speaker = None
        tagged_words = []
        vad_filtered = 0
        overlap_words = 0

        for i, word in enumerate(words):
            seg = Segment(word["start"], word["end"])

            # VAD filtresi
            if args.require_vad:
                coverage = timeline_coverage_ratio(vad_timeline, seg)
                if coverage < args.vad_coverage:
                    vad_filtered += 1
                    logger.debug(f"Kelime {i} VAD filtresine takıldı: "
                                 f"'{word['word']}' (kapsama: {coverage:.2%})")
                    continue

            # Çakışma kontrolü
            is_overlap = timeline_overlaps(final_overlap, seg)
            if is_overlap:
                overlap_words += 1

            # Konuşmacı ata
            speaker, confidence = assign_speaker_midwin(
                diar_ann, seg, prev_speaker, is_overlap
            )

            prev_speaker = speaker

            tagged_words.append({
                "word": word["word"],
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker,
                "confidence": word["confidence"],
                "speaker_confidence": confidence,
                "is_overlap": is_overlap
            })

            # İlerleme göstergesi
            if (i + 1) % 100 == 0:
                logger.info(f"  İşlenen kelime: {i+1}/{len(words)}")

        logger.info(f"✓ Konuşmacı ataması tamamlandı:")
        logger.info(f"  Toplam kelime: {len(words)}")
        logger.info(f"  İşlenen: {len(tagged_words)}")
        if vad_filtered > 0:
            logger.info(f"  VAD filtresi: {vad_filtered} kelime")
        if overlap_words > 0:
            logger.info(f"  Çakışmada: {overlap_words} kelime")

        # ============ AŞAMA 6: SEGMENT OLUŞTURMA ============
        logger.info("\n" + "="*40)
        logger.info("AŞAMA 6: SEGMENT OLUŞTURMA")
        logger.info("="*40)

        segments = merge_words_into_segments(tagged_words,
                                             max_gap=1.5,
                                             max_len=30.0)

        # Küçük segmentleri birleştir
        segments = post_merge_tiny_segments(segments,
                                            min_duration=2.0,
                                            max_gap=1.5)

        # ============ AŞAMA 7: ÇIKTI KAYDETME ============
        logger.info("\n" + "="*40)
        logger.info("AŞAMA 7: SONUÇLARI KAYDETME")
        logger.info("="*40)

        # Ana çıktı
        output = {
            "metadata": {
                "input_file": args.input,
                "total_segments": len(segments),
                "speakers": speakers,
                "language": args.language or "auto",
                "model": args.asr_model,
                "processing_date": datetime.now().isoformat()
            },
            "segments": segments
        }

        input_json = diar_ann
        out_json_path = output

        result = process_dialog(input_json, out_json_path)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Ana çıktı kaydedildi: {args.output}")

        # Debug dosyaları
        if args.dump_debug:
            logger.info(f"\nDebug dosyaları kaydediliyor: {args.dump_debug}")
            os.makedirs(args.dump_debug, exist_ok=True)

            debug_files = {
                "vad.json": {
                    "description": "Voice Activity Detection sonuçları",
                    "segments": timeline_to_dict(vad_timeline)
                },
                "osd.json": {
                    "description": "Overlapped Speech Detection sonuçları",
                    "segments": timeline_to_dict(osd_timeline)
                },
                "diarization.json": {
                    "description": "Speaker Diarization sonuçları",
                    "speakers": speakers,
                    "segments": diarization_to_dict(diar_ann)
                },
                "overlap.json": {
                    "description": "Final çakışma bölgeleri",
                    "segments": timeline_to_dict(final_overlap)
                },
                "words.json": {
                    "description": "Transkribe edilmiş kelimeler",
                    "total": len(words),
                    "words": words[:100]  # İlk 100 kelime
                },
                "tagged_words.json": {
                    "description": "Konuşmacı atamalı kelimeler",
                    "total": len(tagged_words),
                    "words": tagged_words[:100]  # İlk 100 kelime
                }
            }



            for filename, data in debug_files.items():
                filepath = os.path.join(args.dump_debug, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"  ✓ {filename}")





        # Temizlik
        try:
            os.remove(wav_path)
            logger.info(f"\nGeçici dosya silindi: {wav_path}")
        except Exception as e:
            logger.warning(f"Geçici dosya silinemedi: {e}")

        # ============ ÖZET ============
        logger.info("\n" + "="*60)
        logger.info("✨ İŞLEM BAŞARIYLA TAMAMLANDI!")
        logger.info("="*60)
        logger.info(f"Çıktı dosyası: {args.output}")
        logger.info(f"Toplam segment: {len(segments)}")
        logger.info(f"Konuşmacı sayısı: {len(speakers)}")

        # Segment uzunluk istatistikleri
        if segments:
            durations = [s['duration'] for s in segments if 'duration' in s]
            if durations:
                logger.info(f"Ortalama segment süresi: {np.mean(durations):.1f}s")
                logger.info(f"En kısa segment: {min(durations):.1f}s")
                logger.info(f"En uzun segment: {max(durations):.1f}s")

    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"❌ HATA OLUŞTU!")
        logger.error(f"{'='*60}")
        logger.error(f"{type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()