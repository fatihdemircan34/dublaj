#!/usr/bin/env python3
"""
Enhanced ASR+Diarization Test Script
"""

import os
import sys
from enhanced_asr_diarization import EnhancedASRDiarization
from utils import analyze_speaker_distribution, export_srt

def test_with_sample_audio():
    """Test fonksiyonu"""

    print("🧪 Enhanced ASR+Diarization Test")
    print("=" * 40)

    # Sample audio kontrolü
    sample_audio = "../data/sample_audio.wav"
    if not os.path.exists(sample_audio):
        print(f"⚠️ Sample audio bulunamadı: {sample_audio}")
        print("Kendi audio dosyanızla test edebilirsiniz:")
        print("python test_enhanced.py your_audio.wav")
        return

    # Processor oluştur
    processor = EnhancedASRDiarization(
        device="auto",
        model_size="medium"  # Test için daha hızlı
    )

    # İşle
    result = processor.process(
        audio_path=sample_audio,
        output_dir="../output/test_results",
        language="tr",
        min_speakers=1,
        max_speakers=4
    )

    # Analiz et
    speaker_stats = analyze_speaker_distribution(result["timeline"])

    print("\n📊 Konuşmacı Analizi:")
    for speaker, stats in speaker_stats.items():
        print(f"  {speaker}:")
        print(f"    • Toplam süre: {stats['total_duration']:.1f}s")
        print(f"    • Toplam kelime: {stats['total_words']}")
        print(f"    • Dakikada kelime: {stats['words_per_minute']:.1f}")

    # SRT export
    srt_path = "../output/test_results/subtitles.srt"
    export_srt(result["timeline"], srt_path)
    print(f"\n📝 SRT dosyası oluşturuldu: {srt_path}")

    print(f"\n✅ Test tamamlandı!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom audio file
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            processor = EnhancedASRDiarization()
            processor.process(audio_file, "../output/custom_test")
        else:
            print(f"❌ Dosya bulunamadı: {audio_file}")
    else:
        test_with_sample_audio()
