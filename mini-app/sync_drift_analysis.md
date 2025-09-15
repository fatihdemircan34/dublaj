# Audio-Video Sync Drift Analysis Report

## Problem Summary
Video sonlarına doğru ses ve görüntü arasında giderek artan senkronizasyon kaybı yaşanıyor. Özellikle 10 konuşmacılı videolarda sorun daha belirgin.

## Root Causes Identified

### 1. **Time Stretching DISABLED (Primary Issue)**
```python
def _time_stretch_to_duration(in_wav: Path, target_sec: float, out_wav: Path, enable: bool = True) -> Path:
    # HIZ/SÜRE OYNAMA KAPALI SÜRÜM:
    # - target_sec ve enable göz ardı edilir
    # - sadece 16 kHz, mono normalize edilerek kopyalanır
```
**Problem:** TTS çıktıları orijinal segment sürelerine fit edilmiyor. 3 saniyelik segment için TTS 2.5 veya 3.5 saniye üretebiliyor.

### 2. **Cumulative Duration Drift**
- Her segment'te küçük süre farkları (+/- 0.2-0.5 saniye)
- 100 segment'te toplam drift: 20-50 saniye olabiliyor
- Video sonu = Başlangıçta küçük fark → Sonda büyük kayma

### 3. **Breath Gap Accumulation**
```python
breath_gap_ms: int = 200  # Default değer
```
- Her segment arası 50-200ms gap ekleniyor
- 100 segment = 5-20 saniye ekstra süre
- Adaptive gap bile sorunu tam çözmüyor

### 4. **Global Timeline Issues**
Yeni global timing tracker eklendi AMA:
```python
# Global timing: respect original timing as much as possible
target_start_ms = int(original_start * 1000)
```
- TTS süreleri orijinal sürelere uymayınca, start pozisyonları kayıyor
- Overlap redistribution timing'i daha da bozabiliyor

### 5. **No Duration Compensation**
```python
# Hız/süre esnetme YOK: sadece normalize et
stretched = tmp_audio_dir / f"seg_{sid:06d}.fit.wav"
_time_stretch_to_duration(raw_out, target_sec=0.0, out_wav=stretched, enable=False)
```
**Kritik:** `enable=False` ve `target_sec=0.0` → Hiçbir düzeltme yapılmıyor!

## Progressive Desync Pattern

```
Segment 1: Original 3.0s → TTS 3.2s → Drift: +0.2s
Segment 2: Original 2.5s → TTS 2.3s → Drift: +0.0s (cumulative)
Segment 3: Original 4.0s → TTS 4.5s → Drift: +0.5s (cumulative)
...
Segment 50: Cumulative drift: +10s
Segment 100: Cumulative drift: +25s
```

## Why It's Worse Now

Son değişikliklerle overlap redistribution ekledik:
- Overlap'leri yeniden dağıtırken segment timing'leri değişti
- Adaptive gaps ekstra delay ekliyor
- Global timeline tracker drift'i kompanse etmiyor, sadece takip ediyor

## Critical Code Sections

### 1. synthesize_dub_track_xtts (Line 1508-1561)
- TTS synthesis yapıyor
- Time stretching DISABLED
- Segment süreleri korunmuyor

### 2. _concat_timeline_audio (Line 1293-1388)
- Global timeline tracking var
- AMA TTS sürelerini düzeltmiyor
- Sadece pozisyonları ayarlıyor

### 3. _time_stretch_to_duration (Line 1254-1261)
- **COMPLETELY DISABLED**
- Sadece format conversion yapıyor
- Duration adjustment YOK

## Solution Requirements

1. **Re-enable Time Stretching**
   - TTS çıktılarını orijinal segment sürelerine fit et
   - FFmpeg atempo veya librosa/pyrubberband kullan

2. **Duration-Aware Concatenation**
   - Her segment'in hedef süresini koru
   - Drift'i segment bazında kompanse et

3. **Anchor Points**
   - Her N segment'te bir "sync point" belirle
   - Drift'i bu noktalarda sıfırla

4. **Breath Gap Optimization**
   - Statik gap yerine dinamik kompansasyon
   - Drift varsa gap'leri azalt/arttır

## Recommendation

**URGENT:** Time stretching'i yeniden aktifleştir. Bu olmadan sync problemi çözülemez.

```python
def _time_stretch_to_duration(in_wav: Path, target_sec: float, out_wav: Path, enable: bool = True) -> Path:
    if not enable or target_sec <= 0:
        # Fallback
        _run(["ffmpeg", "-y", "-i", str(in_wav), "-ar", "16000", "-ac", "1", str(out_wav)])
        return out_wav

    # ACTUAL TIME STRETCHING
    current_duration = _ffprobe_duration(in_wav)
    if abs(current_duration - target_sec) < 0.05:  # 50ms tolerance
        # Close enough, just normalize
        _run(["ffmpeg", "-y", "-i", str(in_wav), "-ar", "16000", "-ac", "1", str(out_wav)])
    else:
        # Apply time stretching
        speed = current_duration / target_sec
        atempo = _atempo_chain(speed)
        _run(["ffmpeg", "-y", "-i", str(in_wav), "-af", f"{atempo},aresample=16000", "-ac", "1", str(out_wav)])

    return out_wav
```