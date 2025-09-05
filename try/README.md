# Enhanced ASR + Diarization System

WhisperX ve Pyannote kombinasyonu ile geliştirilmiş ASR+Diarization sistemi.

## 🚀 Özellikler

- **Word-level alignment** - Her kelimenin hassas zamanlaması
- **Gelişmiş speaker assignment** - IoU algoritması ile doğru konuşmacı atama
- **Overlap detection** - Aynı anda konuşan kişileri tespit
- **Speaker smoothing** - Gürültülü speaker değişimlerini düzelt
- **Detaylı analiz** - Kapsamlı istatistikler ve raporlar

## 📦 Kurulum

```bash
# Bu script ile otomatik kurulum
./setup_enhanced_asr.sh

# Manuel kurulum
cd try
source venv/bin/activate
pip install -r requirements.txt
```

## 🔑 Gereksinimler

- Python 3.8+
- FFmpeg
- HuggingFace Token (Pyannote için)

```bash
export HF_TOKEN=your_huggingface_token
```

## 🎯 Kullanım

### Basit kullanım
```bash
cd scripts
./run_enhanced.sh -i audio.wav
```

### Parametreli kullanım
```bash
./run_enhanced.sh -i audio.wav \
  -o results \
  -m large-v2 \
  -l tr \
  --min-speakers 2 \
  --max-speakers 4
```

### Python API
```python
from src.enhanced_asr_diarization import EnhancedASRDiarization

processor = EnhancedASRDiarization(model_size="large-v2")
result = processor.process("audio.wav", "output")
```

## 📊 Çıktılar

- `speaker_timeline.jsonl` - Konuşmacı bazlı timeline
- `word_details.jsonl` - Kelime düzeyinde detaylar
- `overlaps.json` - Overlap analizi
- `statistics.json` - Detaylı istatistikler
- `subtitles.srt` - SRT altyazı dosyası

## 🔧 Konfigürasyon

`configs/config.json` dosyasını düzenleyerek varsayılan ayarları değiştirebilirsiniz.

## 🧪 Test

```bash
cd src
python test_enhanced.py
```

## 📈 Performans

| Model Size | Hız | Doğruluk | Bellek |
|------------|-----|----------|--------|
| tiny       | En hızlı | Düşük | Az |
| medium     | Orta | İyi | Orta |
| large-v2   | Yavaş | Mükemmel | Yüksek |

## 🛠️ Sorun Giderme

1. **CUDA hatası**: `--device cpu` kullanın
2. **Memory hatası**: Daha küçük model kullanın
3. **HF_TOKEN**: Token'ınızı kontrol edin
4. **FFmpeg**: Kurulu olduğundan emin olun

## 📝 Lisans

MIT License
