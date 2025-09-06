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
source $(poetry env info --path)/bin/activate
# Shell plugin'ini yükle
poetry self add poetry-plugin-shell

# Artık shell komutunu kullanabilirsiniz
poetry shell
export HF_TOKEN="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"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/voiceprocess-58fbc-8e5f0264cdc0.json"

## Çıktılar
(dublaj-pipeline-py3.10) videodubb_voiceprocess_io@instance-20250904-153243:~/PycharmProjects/dublaj/pyannote$ poetry run python app-full.py --audio ../input_6.wav   --out out.jsonl   --hf-token "$HF_TOKEN"  --do-mix 
usage: app-full.py [-h] --audio AUDIO --out OUT [--hf-token HF_TOKEN] [--asr-model ASR_MODEL] [--asr-device ASR_DEVICE] [--asr-compute-type ASR_COMPUTE_TYPE] [--vad-onset VAD_ONSET]
                   [--vad-offset VAD_OFFSET] [--vad-min-on VAD_MIN_ON] [--vad-min-off VAD_MIN_OFF] [--osd-onset OSD_ONSET] [--osd-offset OSD_OFFSET] [--osd-min-on OSD_MIN_ON]
                   [--osd-min-off OSD_MIN_OFF] [--require-vad] [--vad-coverage VAD_COVERAGE] [--min-speakers MIN_SPEAKERS] [--max-speakers MAX_SPEAKERS] [--output-dir OUTPUT_DIR]
app-full.py: error: unrecognized arguments: --do-mix
(dublaj-pipeline-py3.10) videodubb_voiceprocess_io@instance-20250904-153243:~/PycharmProjects/dublaj/pyannote$ poetry run python app-full.py --audio ../input_6.wav   --out out.jsonl   --hf-token "$HF_TOKEN"  
🎵 Audio: ../input_6.wav
📝 Çıkış: out.jsonl
📁 Adım çıktı dizini: .
🤖 ASR: large-v3 (device=auto, compute=auto)
============================================================
📍 ADIM 1: Voice Activity Detection (VAD)
🔧 VAD modeli yükleniyor (pyannote/brouhaha)...
Lightning automatically upgraded your loaded checkpoint from v1.6.5 to v2.5.5. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../../../.cache/torch/pyannote/models--pyannote--brouhaha/snapshots/c93c9b537732dd50c28c0366c73f560c3a7aeb02/pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.12.1+cu102, yours is 2.3.1+cu121. Bad things might happen unless you revert torch to 1.x.
⚙️  VAD parametreleri: onset=0.0, offset=0.0, min_on=0.0, min_off=0.0
✅ VAD modeli hazır
🔍 VAD analizi yapılıyor...
✅ VAD tamam: 1 segment, toplam 137.65s
💾 step1_vad çıktısı kaydedildi: ./step1_vad.json

📍 ADIM 2: Overlapped Speech Detection (OSD)
🔧 OSD modeli yükleniyor (pyannote/segmentation-3.1)...
⚙️  OSD parametreleri: onset=0.0, offset=0.0, min_on=0.1, min_off=0.1
✅ OSD modeli hazır
🔍 OSD analizi yapılıyor...
✅ OSD tamam: 23 segment, toplam 44.58s
💾 step2_osd çıktısı kaydedildi: ./step2_osd.json

📍 ADIM 3: Speaker Diarization
🔧 Diarization modeli yükleniyor (pyannote/speaker-diarization-3.1)...
✅ Diarization modeli hazır
🔍 Diarization analizi yapılıyor...
/home/videodubb_voiceprocess_io/.cache/pypoetry/virtualenvs/dublaj-pipeline-KrFI56mV-py3.10/lib/python3.10/site-packages/pyannote/audio/models/blocks/pooling.py:104: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at ../aten/src/ATen/native/ReduceOps.cpp:1807.)
  std = sequences.std(dim=-1, correction=1)




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
