# ASR+Diarization Worker (GCP / Cloud Run GPU)

Bu depo, dublaj pipeline'ının **ASR (WhisperX)** + **Diarization (pyannote / NeMo MSDD / GCP STT)**
aşamalarını çalıştıran **plugin mimarili** bir AI "worker" servisidir.
- Artefact standartları: `asr_segments.jsonl`, `diar_segments.jsonl`, `speaker_profiles.json` (opsiyonel), `edl.jsonl`
- Çalıştırma: CLI veya Cloud Run Job
- Giriş: GCS URI (`gs://...`) veya yerel dosya
- Çıkış: GCS prefix veya yerel klasör

> Not: Bu repo **AI worker** katmanıdır. API ve Web UI ayrı projelerdir.

## Hızlı Başlangıç (Yerel / GPU)

```bash
# Python 3.10+ önerilir, CUDA yüklü GPU ile
pip install -e .[nemo]

# Veya NeMo olmadan
# pip install -e .

# HuggingFace token (pyannote için gerekli)
export HF_TOKEN=hf_xxx

# Basit örnek (pyannote diarization)
python -m dubbing_ai.runner \
  --audio gs://my-bucket/input/audio.wav \
  --out gs://my-bucket/outputs/job-123 \
  --diarizer pyannote \
  --domain podcast \
  --model-size medium

# GCP STT (managed) örnek
export GOOGLE_CLOUD_PROJECT=my-project
python -m dubbing_ai.runner --audio ./sample.wav --out ./out --diarizer gcp-stt --min-speakers 1 --max-speakers 4

# NeMo MSDD (telephony/overlap)
python -m dubbing_ai.runner --audio ./call.wav --out ./out --diarizer nemo-msdd --domain telephony
```

## Çıktılar
```
out/
  asr_segments.jsonl
  diar_segments.jsonl
  speaker_profiles.json   # opsiyonel
  asr_words_speaker.jsonl # kelime->konuşmacı eşlemesi
  edl.jsonl               # cut-list
```

## Docker (Cloud Run GPU)
```bash
gcloud builds submit --tag gcr.io/$PROJECT/asr-diarize-worker:latest
gcloud run deploy asr-diarize-worker --image gcr.io/$PROJECT/asr-diarize-worker:latest   --region=us-central1 --cpu=4 --memory=16Gi --gpu=1 --gpu-type=nvidia-l4   --no-allow-unauthenticated --max-instances=5
```

## Lisans / Modeller
- `pyannote.audio` için HF token ve model lisans koşullarına uyunuz.
- NeMo MSDD modelleri NVIDIA lisansı ile sunulur.
- Google STT kullanımı ücretlidir.

## Yapı
- `dubbing_ai/` ana paket
- `dubbing_ai/diarizer/` diarization plugin'leri
- `dubbing_ai/transcriber/` ASR (WhisperX) eklentisi
- `dubbing_ai/align/` kelime→konuşmacı eşlemesi ve EDL
- `dubbing_ai/utils/` GCS IO, audio yardımcıları
- `configs/` örnek NeMo konfig ve Cloud Run Job env örneği

Üretimde, FFmpeg/yt-dlp/lip-sync wrapper'larınızı API katmanından bu worker'a parametre ile bağlayabilirsiniz.
