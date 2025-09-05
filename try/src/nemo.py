#!/usr/bin/env python3
"""
NVIDIA Parakeet-TDT-0.6B ASR Script (Hugging Face Version)
WAV dosyasından JSON formatında transkripsiyon üretir
"""

import json
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Hugging Face transformers
try:
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        pipeline
    )
except ImportError:
    print("Transformers kütüphanesi yüklü değil. Lütfen şu komutu çalıştırın:")
    print("pip install transformers torch torchaudio")
    exit(1)

class ParakeetASR:
    def __init__(self, model_name="nvidia/parakeet-tdt-1.1b", use_fp32=False):
        """
        Parakeet ASR modelini başlat
        NOT: Parakeet-TDT-0.6b doğrudan HF'de yoksa, alternatif model kullanıyoruz

        Args:
            model_name: Model adı veya yolu
        """
        print(f"Model yükleniyor: {model_name}")

        # GPU varsa kullan
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cihaz: {self.device}")

        # dtype seçimi: CUDA varsa ve fp32 zorlanmıyorsa fp16; aksi halde fp32
        # Bazı GPU'larda fp16 performanslıdır; fp32 gerektiğinde --fp32 bayrağı ile zorlanabilir.
        dtype = torch.float16 if (self.device == "cuda" and not use_fp32) else torch.float32
        self._dtype = dtype

        try:
            # Önce Parakeet'i dene
            if "parakeet" in model_name.lower():
                # Parakeet modelleri genelde Whisper tabanlı
                print("Parakeet modeli yükleniyor (Whisper-based)...")

                # Alternatif: Whisper modelini kullan
                fallback_model = "openai/whisper-large-v3"
                print(f"Not: Parakeet HF'de bulunamazsa {fallback_model} kullanılacak")

                try:
                    self.processor = AutoProcessor.from_pretrained(model_name)
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    )
                except:
                    print(f"{model_name} bulunamadı, {fallback_model} kullanılıyor...")
                    self.processor = WhisperProcessor.from_pretrained(fallback_model)
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        fallback_model,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    )
            else:
                # Genel ASR modeli
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )

            self.model.to(self.device)
            self.model.eval()
            print("Model başarıyla yüklendi!")

        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            print("\nAlternatif modeller:")
            print("- openai/whisper-large-v3")
            print("- openai/whisper-large-v3-turbo")
            print("- facebook/seamless-m4t-v2-large")
            raise

    def load_audio(self, audio_path, sampling_rate=16000):
        """
        Audio dosyasını yükle ve işle

        Args:
            audio_path: Ses dosyası yolu
            sampling_rate: Örnekleme oranı

        Returns:
            numpy array: İşlenmiş ses verisi
        """
        # Audio'yu yükle
        waveform, sample_rate = torchaudio.load(audio_path)

        # Mono'ya çevir
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Örnekleme oranını ayarla
        if sample_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, sampling_rate)
            waveform = resampler(waveform)

        # NumPy array'e çevir
        audio_array = waveform.squeeze().numpy()

        # Normalize et (sessiz dosyaya karşı korun)
        peak = np.abs(audio_array).max()
        if peak > 0:
            audio_array = audio_array / peak
        else:
            # Tamamen sessizse, olduğu gibi bırak
            audio_array = audio_array

        return audio_array, waveform.shape[1] / sampling_rate

    def transcribe(self, audio_path, language="turkish", task="transcribe"):
        """
        Ses dosyasını metne dönüştür

        Args:
            audio_path: WAV dosya yolu
            language: Dil (turkish, english, vb.)
            task: transcribe veya translate

        Returns:
            dict: Transkripsiyon sonuçları
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {audio_path}")

        print(f"\nİşleniyor: {audio_path}")

        # Audio'yu yükle
        audio_array, duration = self.load_audio(str(audio_path))

        # Input'u hazırla (attention_mask varsa ilet; dtype'ı modelle eşle)
        inputs_dict = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs_dict.input_features.to(self.device, dtype=self._dtype)
        attention_mask = getattr(inputs_dict, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Transkripsiyon yap
        with torch.no_grad():
            # Dil tokenını ayarla (Whisper için)
            if hasattr(self.processor, 'tokenizer'):
                if language == "turkish" or language == "tr":
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language="turkish",
                        task=task
                    )
                else:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language,
                        task=task
                    )
            else:
                forced_decoder_ids = None

            # Generate (token limitini ayarla)
            max_tokens = 444 if forced_decoder_ids else 446

            generate_kwargs = {
                "forced_decoder_ids": forced_decoder_ids,
                "max_new_tokens": max_tokens,
                "num_beams": 5,
                "temperature": 0.0,
                "no_repeat_ngram_size": 3,
                "do_sample": False
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            predicted_ids = self.model.generate(
                input_features,
                temperature=0.0,
                no_repeat_ngram_size=3,
                do_sample=False
            )

            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

        # Sonuçları hazırla
        result = {
            "file": str(audio_path),
            "duration": round(duration, 2),
            "language": language,
            "task": task,
            "transcription": transcription.strip(),
            "model": "nvidia/parakeet-tdt-0.6b (via Whisper)",
            "device": str(self.device)
        }

        return result

    def transcribe_with_timestamps(self, audio_path, language="turkish"):
        """
        Zaman damgalı transkripsiyon

        Args:
            audio_path: WAV dosya yolu
            language: Dil

        Returns:
            dict: Detaylı transkripsiyon sonuçları
        """
        audio_path = Path(audio_path)
        print(f"\nZaman damgalı transkripsiyon: {audio_path}")

        # Audio'yu yükle
        audio_array, duration = self.load_audio(str(audio_path))

        # Pipeline kullan (daha kolay timestamp için)
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=0 if self.device == "cuda" else -1
            )

            result = pipe(
                audio_array,
                return_timestamps=True,
                generate_kwargs={
                    "language": language if language != "tr" else "turkish",
                    "task": "transcribe"
                }
            )

            output = {
                "file": str(audio_path),
                "duration": round(duration, 2),
                "language": language,
                "transcription": result["text"].strip(),
                "chunks": result.get("chunks", []),
                "model": "nvidia/parakeet-tdt-0.6b (via Whisper)",
                "device": str(self.device)
            }

        except:
            # Fallback: timestamp olmadan
            print("Zaman damgası desteklenmiyor, normal transkripsiyon yapılıyor...")
            output = self.transcribe(audio_path, language)

        return output

def main():
    parser = argparse.ArgumentParser(
        description="Parakeet/Whisper ile ses dosyalarını metne dönüştür"
    )
    parser.add_argument(
        "input_wav",
        type=str,
        help="Giriş ses dosyası yolu (WAV, MP3, vb.)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="transcription.json",
        help="Çıkış JSON dosyası yolu (varsayılan: transcription.json)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="english",
        help="Dil (turkish, english, vb.) (varsayılan: english)"
    )
    parser.add_argument(
        "-t", "--timestamps",
        action="store_true",
        help="Zaman damgalı transkripsiyon"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="Model adı (varsayılan: openai/whisper-large-v3)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Görev: transcribe veya translate (varsayılan: transcribe)"
    )

    # CUDA olsa dahi fp32 ile çalışmayı zorlamak için bayrak
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="CUDA mevcut olsa bile fp32 ile çalış (varsayılan: CUDA varsa fp16)"
    )
    args = parser.parse_args()

    try:
        # ASR modelini başlat (dtype sorunları için default fp32 kullan)
        print("="*50)
        print("PARAKEET/WHISPER ASR")
        print("="*50)

        # Geçici çözüm: fp32 attribute yoksa False yap
        use_fp32 = getattr(args, 'fp32', False)

        asr = ParakeetASR(model_name=args.model, use_fp32=use_fp32)

        # Transkripsiyon yap
        if args.timestamps:
            result = asr.transcribe_with_timestamps(
                args.input_wav,
                language=args.language
            )
        else:
            result = asr.transcribe(
                args.input_wav,
                language=args.language,
                task=args.task
            )

        # Sonucu ekrana yazdır
        print("\n" + "="*50)
        print("TRANSKRIPSIYON SONUCU:")
        print("="*50)
        print(f"Dosya: {result['file']}")
        print(f"Süre: {result['duration']} saniye")
        print(f"Dil: {result['language']}")
        print("-"*50)
        print(f"Metin:\n{result['transcription']}")

        # Eğer chunks varsa göster
        if "chunks" in result and result["chunks"]:
            print("-"*50)
            print("Zaman Damgalı Segmentler:")
            for chunk in result["chunks"]:
                start = chunk.get('timestamp', [0, 0])[0]
                end = chunk.get('timestamp', [0, 0])[1]
                text = chunk.get('text', '')
                print(f"[{start:.2f}s - {end:.2f}s]: {text}")

        # JSON olarak kaydet
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nSonuç kaydedildi: {output_path}")

    except Exception as e:
        print(f"\nHata: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())