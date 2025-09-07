import json
from typing import List, Dict, Any, Optional, Tuple

def load_words(filepath: str) -> List[Dict[str, Any]]:
    """JSONL dosyasından kelimeleri yükle"""
    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words.append(json.loads(line.strip()))
    return words

def load_diarization(filepath: str) -> Dict[str, Any]:
    """Diarization JSON dosyasını yükle"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_overlapping_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Overlapping segment'leri birleştir ve düzenle.
    Aynı konuşmacının ardışık segment'lerini birleştirir.
    """
    if not segments:
        return []

    # Segment'leri başlangıç zamanına göre sırala
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    merged = []

    current = sorted_segments[0].copy()

    for next_seg in sorted_segments[1:]:
        # Aynı konuşmacı ve overlap varsa veya çok yakınsa birleştir
        if (current['speaker'] == next_seg['speaker'] and
                (current['end'] >= next_seg['start'] - 0.5)):  # 0.5 saniye tolerans
            # Segment'leri birleştir
            current['end'] = max(current['end'], next_seg['end'])
            current['duration'] = current['end'] - current['start']
        else:
            # Farklı konuşmacı veya overlap yok, mevcut segment'i kaydet
            merged.append(current)
            current = next_seg.copy()

    # Son segment'i ekle
    merged.append(current)

    return merged

def assign_words_to_speakers(words: List[Dict[str, Any]],
                             segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Kelimeleri konuşmacılara ata.
    Kelimeler zaten speaker bilgisi içeriyor, bu bilgiyi kullan.
    """
    # Konuşmacı bazlı kelime grupları oluştur
    speaker_words = {}

    for word in words:
        speaker = word.get('speaker', 'UNKNOWN')
        if speaker not in speaker_words:
            speaker_words[speaker] = []
        speaker_words[speaker].append(word)

    # Her konuşmacı için sürekli konuşma bloklarını oluştur
    speech_blocks = []

    for speaker, words_list in speaker_words.items():
        if not words_list:
            continue

        # Kelimeleri zamana göre sırala
        sorted_words = sorted(words_list, key=lambda x: x['start'])

        # Ardışık kelimeleri grupla (0.5 saniyeden fazla boşluk varsa yeni blok)
        current_block = {
            'speaker': speaker,
            'words': [sorted_words[0]],
            'start': sorted_words[0]['start'],
            'end': sorted_words[0]['end']
        }

        for word in sorted_words[1:]:
            # Eğer kelimeler arasında 0.5 saniyeden fazla boşluk varsa yeni blok başlat
            if word['start'] - current_block['end'] > 0.5:
                speech_blocks.append(current_block)
                current_block = {
                    'speaker': speaker,
                    'words': [word],
                    'start': word['start'],
                    'end': word['end']
                }
            else:
                # Mevcut bloğa ekle
                current_block['words'].append(word)
                current_block['end'] = word['end']

        # Son bloğu ekle
        speech_blocks.append(current_block)

    # Blokları zamana göre sırala
    speech_blocks.sort(key=lambda x: x['start'])

    return speech_blocks

def create_final_segments(speech_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Konuşma bloklarından final segment'leri oluştur
    """
    final_segments = []

    for idx, block in enumerate(speech_blocks):
        # Metni oluştur
        text = ' '.join([w['word'] for w in block['words']])

        segment = {
            'segment_id': idx,
            'speaker': block['speaker'],
            'start': block['start'],
            'end': block['end'],
            'duration': block['end'] - block['start'],
            'text': text.strip(),
            'word_count': len(block['words']),
            'words': block['words']
        }

        final_segments.append(segment)

    return final_segments

def create_transcript(segments: List[Dict[str, Any]],
                      include_timestamps: bool = True,
                      include_word_details: bool = False) -> str:
    """
    Segment'lerden okunabilir bir transcript oluştur
    """
    transcript_lines = []
    current_speaker = None
    current_text = []

    for segment in segments:
        if segment['text']:  # Boş segment'leri atla
            # Eğer konuşmacı değiştiyse, önceki konuşmacının metnini ekle
            if current_speaker != segment['speaker']:
                if current_speaker and current_text:
                    combined_text = ' '.join(current_text)
                    transcript_lines.append(f"{current_speaker}: {combined_text}")
                    transcript_lines.append("")  # Boş satır ekle

                current_speaker = segment['speaker']
                current_text = [segment['text']]
            else:
                # Aynı konuşmacı devam ediyor
                current_text.append(segment['text'])

    # Son konuşmacının metnini ekle
    if current_speaker and current_text:
        combined_text = ' '.join(current_text)
        transcript_lines.append(f"{current_speaker}: {combined_text}")

    return '\n'.join(transcript_lines)

def create_detailed_transcript(segments: List[Dict[str, Any]]) -> str:
    """
    Zaman damgalı detaylı transcript oluştur
    """
    transcript_lines = []

    for segment in segments:
        if segment['text']:
            line = f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['speaker']}: {segment['text']}"
            transcript_lines.append(line)

    return '\n'.join(transcript_lines)

def save_results(segments: List[Dict[str, Any]], output_prefix: str = 'step4'):
    """
    Sonuçları farklı formatlarda kaydet
    """
    # 1. Detaylı JSON (kelime bilgileriyle)
    with open(f'{output_prefix}_segments_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    # 2. Özet JSON (kelime detayları olmadan)
    segments_summary = []
    for seg in segments:
        summary = {k: v for k, v in seg.items() if k != 'words'}
        segments_summary.append(summary)

    with open(f'{output_prefix}_segments_summary.json', 'w', encoding='utf-8') as f:
        json.dump(segments_summary, f, indent=2, ensure_ascii=False)

    # 3. Detaylı transcript (zaman damgalı)
    detailed_transcript = create_detailed_transcript(segments)
    with open(f'{output_prefix}_transcript_detailed.txt', 'w', encoding='utf-8') as f:
        f.write(detailed_transcript)

    # 4. Gruplandırılmış transcript (konuşmacı bazlı)
    grouped_transcript = create_transcript(segments)
    with open(f'{output_prefix}_transcript_grouped.txt', 'w', encoding='utf-8') as f:
        f.write(grouped_transcript)

    print(f"✓ Sonuçlar kaydedildi:")
    print(f"  - {output_prefix}_segments_detailed.json (kelime detaylarıyla)")
    print(f"  - {output_prefix}_segments_summary.json (özet)")
    print(f"  - {output_prefix}_transcript_detailed.txt (zaman damgalı)")
    print(f"  - {output_prefix}_transcript_grouped.txt (konuşmacı bazlı)")

def analyze_segments(segments: List[Dict[str, Any]]):
    """
    Segment istatistiklerini analiz et ve göster
    """
    print("\n=== SEGMENT ANALİZİ ===")
    print(f"Toplam segment sayısı: {len(segments)}")

    # Konuşmacı bazlı analiz
    speaker_stats = {}
    for seg in segments:
        speaker = seg['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'segment_count': 0,
                'total_duration': 0,
                'total_words': 0,
                'segments': []
            }

        speaker_stats[speaker]['segment_count'] += 1
        speaker_stats[speaker]['total_duration'] += seg['duration']
        speaker_stats[speaker]['total_words'] += seg['word_count']
        speaker_stats[speaker]['segments'].append(seg['segment_id'])

    print("\n--- Konuşmacı İstatistikleri ---")
    for speaker, stats in sorted(speaker_stats.items()):
        print(f"\n{speaker}:")
        print(f"  - Segment sayısı: {stats['segment_count']}")
        print(f"  - Toplam süre: {stats['total_duration']:.2f} saniye")
        print(f"  - Toplam kelime: {stats['total_words']}")
        if stats['segment_count'] > 0:
            print(f"  - Ortalama kelime/segment: {stats['total_words']/stats['segment_count']:.1f}")

    # En uzun segment'ler
    print("\n--- En Uzun 5 Segment ---")
    sorted_segments = sorted(segments, key=lambda x: x['duration'], reverse=True)
    for seg in sorted_segments[:5]:
        preview = seg['text'][:50] + "..." if len(seg['text']) > 50 else seg['text']
        print(f"  {seg['segment_id']:3d}. {seg['speaker']} ({seg['duration']:.2f}s): {preview}")

def main():
    """
    Ana işleme fonksiyonu - Kelime bazlı speaker bilgisini kullan
    """
    print("=== STEP 4: KELİME SPEAKER BİLGİSİNE GÖRE GRUPLAMA ===\n")

    try:
        # Dosyaları yükle
        print("Dosyalar yükleniyor...")
        words = load_words('out_words.jsonl')
        diarization_data = load_diarization('step3_diarization.json')

        print(f"✓ {len(words)} kelime yüklendi")
        print(f"✓ {len(diarization_data['results']['segments'])} segment yüklendi")

        # Kelimelerdeki speaker bilgisini kullanarak konuşma blokları oluştur
        print("\nKelimelerdeki speaker bilgisi kullanılarak gruplandırılıyor...")
        speech_blocks = assign_words_to_speakers(words, diarization_data['results']['segments'])

        # Final segment'leri oluştur
        print("Final segment'ler oluşturuluyor...")
        final_segments = create_final_segments(speech_blocks)
        print(f"✓ {len(final_segments)} konuşma bloğu oluşturuldu")

        # Analiz
        analyze_segments(final_segments)

        # Sonuçları kaydet
        print("\nSonuçlar kaydediliyor...")
        save_results(final_segments)

        print("\n✅ İşlem başarıyla tamamlandı!")

        # Örnek çıktı göster
        print("\n=== ÖRNEK ÇIKTI (İlk 5 Segment) ===")
        for seg in final_segments[:5]:
            if seg['text']:
                print(f"\n[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}:")
                print(f"  \"{seg['text'][:100]}...\"" if len(seg['text']) > 100 else f"  \"{seg['text']}\"")
                print(f"  ({seg['word_count']} kelime)")

    except FileNotFoundError as e:
        print(f"❌ Hata: Dosya bulunamadı - {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Hata: JSON parse hatası - {e}")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()