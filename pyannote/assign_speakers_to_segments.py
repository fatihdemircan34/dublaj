#!/usr/bin/env python3
# assign_speakers_to_segments.py
# Cümle segmentlerine pyannote diarization ile konuşmacı atama

import os, json, argparse, subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math

import torch
import torchaudio

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_grad_enabled(False)

# --------------- FFmpeg: video -> mono 16k wav ---------------
def extract_mono16k(input_video: Path, out_wav: Path) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg","-y","-i",str(input_video),
        "-vn","-acodec","pcm_s16le","-ar","16000","-ac","1",str(out_wav)
    ]
    subprocess.run(cmd, capture_output=True)
    return out_wav

def load_wav_mono_16k(path: Path) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav, sr

# --------------- Geometri: süre çakışması ---------------
def overlap_dur(a_start, a_end, b_start, b_end) -> float:
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0.0, e - s)

# --------------- Pyannote diarization ---------------
def run_diarization(audio_path: Path, hf_token: str, num_speakers: int | None = None, device: str = "cpu"):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    try:
        pipeline.to("cuda")
    except Exception:
        pass
    if num_speakers is not None:
        diar = pipeline(str(audio_path), num_speakers=num_speakers)
    else:
        diar = pipeline(str(audio_path))
    # döndür: [(start, end, label), ...]
    turns = []
    for seg, _, label in diar.itertracks(yield_label=True):
        turns.append((float(seg.start), float(seg.end), str(label)))
    # zaman sıralı
    turns.sort(key=lambda x: (x[0], x[1]))
    return turns

# --------------- Label normalizasyonu ---------------
def normalize_labels(turns: List[Tuple[float,float,str]]) -> Tuple[List[Tuple[float,float,str]], Dict[str,str]]:
    # her label toplam konuşma süresi
    dur = {}
    for s,e,l in turns:
        dur[l] = dur.get(l, 0.0) + (e - s)
    # en uzun konuşandan başlayarak SPEAKER_00, 01...
    ordered = sorted(dur.items(), key=lambda kv: kv[1], reverse=True)
    mapping = { old: f"SPEAKER_{i:02d}" for i,(old,_) in enumerate(ordered) }
    norm = [(s,e,mapping[l]) for (s,e,l) in turns]
    return norm, mapping

# --------------- Segmentlere konuşmacı atama ---------------
def assign_segments_speakers(
    segments: List[Dict[str,Any]],
    turns: List[Tuple[float,float,str]],
    min_overlap_ratio: float = 0.05,   # yumuşak eşik
    context_pad: float = 0.30,         # ±0.30s bağlam
    continuity_bonus: float = 0.20     # önceki konuşmacıya küçük ödül
) -> List[Dict[str,Any]]:
    """
    Cümleleri diarization turn'lerine ata:
      - Pencereyi ±context_pad genişletip çakışmayı ölç.
      - Skor = normalized_overlap + continuity_bonus(if same as prev)
      - Çakışma yoksa cümle merkezine en yakın turn kiminse o.
    NOT: Hiç turn yoksa asla 'SPEAKER_00' dayamıyoruz; varsa mevcut 'speaker' alanını
         koruyoruz, yoksa segmenti olduğu gibi bırakıyoruz.
    """
    if not segments:
        return segments

    # diarization yoksa: dokunma (varsa mevcut speaker alanını aynen koru)
    if not turns:
        return segments

    def ov(a0, a1, b0, b1):
        s = max(a0, b0); e = min(a1, b1)
        return max(0.0, e - s)

    # İlk segment için prev_speaker = en iyi gerçek aday (asla hard-code yok)
    def best_label_for_segment(seg):
        s0 = float(seg["start"]); s1 = float(seg["end"])
        dur = max(1e-6, s1 - s0)
        ws0, ws1 = s0 - context_pad, s1 + context_pad

        overlaps = {}
        for ts, te, lab in turns:
            o = ov(ws0, ws1, ts, te)
            if o > 0:
                overlaps[lab] = overlaps.get(lab, 0.0) + o

        if overlaps:
            # normalize edilmemiş toplam çakışmaya göre ilk seçim
            return max(overlaps, key=lambda k: overlaps[k])
        else:
            # merkez en yakın turn
            center = 0.5 * (s0 + s1)
            best_lab, best_dist = None, 1e9
            for ts, te, lab in turns:
                c = 0.5 * (ts + te)
                d = abs(c - center)
                if d < best_dist:
                    best_lab, best_dist = lab, d
            return best_lab

    # İlk konuşmacıyı belirle
    first_lab = best_label_for_segment(segments[0])
    prev_speaker = first_lab if first_lab is not None else segments[0].get("speaker")

    # Eğer hala None ise, bu segmente dokunma; sonrasında continuity işleyecek.
    if prev_speaker is not None:
        segments[0]["speaker"] = prev_speaker

    # Geri kalanları ata
    for i in range(1, len(segments)):
        seg = segments[i]
        s0 = float(seg["start"]); s1 = float(seg["end"])
        dur = max(1e-6, s1 - s0)
        ws0, ws1 = s0 - context_pad, s1 + context_pad

        raw_overlap = {}
        for ts, te, lab in turns:
            o = ov(ws0, ws1, ts, te)
            if o > 0:
                raw_overlap[lab] = raw_overlap.get(lab, 0.0) + o

        if not raw_overlap:
            # çakışma yoksa: merkeze en yakın turn
            center = 0.5 * (s0 + s1)
            best_lab, best_dist = None, 1e9
            for ts, te, lab in turns:
                c = 0.5 * (ts + te)
                d = abs(c - center)
                if d < best_dist:
                    best_lab, best_dist = lab, d
            chosen = best_lab if best_lab is not None else prev_speaker or seg.get("speaker")
            if chosen is not None:
                seg["speaker"] = chosen
                prev_speaker = chosen
            # chosen None ise speaker alanını boş bırakıp geç (dokunma)
            continue

        # skor: normalize edilmiş çakışma + continuity bonusu
        scores = {}
        norm_den = dur + 2*context_pad
        for lab, o in raw_overlap.items():
            score = (o / norm_den)
            if prev_speaker is not None and lab == prev_speaker:
                score += continuity_bonus
            scores[lab] = score

        best_lab = max(scores, key=lambda k: scores[k])

        # Çok zayıf delilse ve bir önceki var ise, continuity kullan
        if (raw_overlap[best_lab] / dur) < min_overlap_ratio and prev_speaker is not None:
            seg["speaker"] = prev_speaker
        else:
            seg["speaker"] = best_lab

        prev_speaker = seg["speaker"]

    return segments

# --------------- Yakın segmentleri birleştir (opsiyonel) ---------------
def merge_adjacent_same_speaker(segments: List[Dict[str,Any]],
                                max_gap: float = 0.12) -> List[Dict[str,Any]]:
    """Aynı konuşmacıya ait ve aradaki boşluk çok küçükse birleştir."""
    if not segments:
        return segments
    merged = []
    cur = dict(segments[0])
    for nxt in segments[1:]:
        if (nxt["speaker"] == cur["speaker"]) and (float(nxt["start"]) - float(cur["end"]) <= max_gap):
            # birleştir
            cur["end"] = float(nxt["end"])
            # metni düzgün bağla
            t = (cur.get("text","") or "").rstrip()
            u = (nxt.get("text","") or "").lstrip()
            if t and u:
                cur["text"] = t + (" " if (t[-1:].isalnum() and u[:1].isalnum()) else "") + u
            else:
                cur["text"] = t + u
        else:
            merged.append(cur)
            cur = dict(nxt)
    merged.append(cur)
    # id'leri yeniden sırala
    for i, s in enumerate(merged):
        s["id"] = i
    return merged

# --------------- Ana akış ---------------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str, help="Video girdi")
    g.add_argument("--audio", type=str, help="Mono 16k wav (opsiyonel)")
    ap.add_argument("--segments_json", type=str, required=True, help="Önceden üretilmiş cümle segmentleri JSON (id/seek/start/end/text/speaker)")
    ap.add_argument("--out_json", type=str, required=True, help="Konuşmacı atanmış çıktı JSON")
    ap.add_argument("--hf_token", type=str, required=True, help="HuggingFace token (pyannote için)")
    ap.add_argument("--num_speakers", type=int, default=None, help="İstersen sabit konuşmacı sayısı ver (örn. 3)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--min_overlap_ratio", type=float, default=0.20, help="Cümle süresine göre min çakışma oranı")
    ap.add_argument("--merge_after", action="store_true", help="Aynı konuşmacı ardışık cümleleri küçük boşlukta birleştir")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # 1) Audio hazırla
    if args.audio:
        audio_path = Path(args.audio)
    else:
        audio_path = out_json.parent / "mix_16k.wav"
        extract_mono16k(Path(args.video), audio_path)

    # 2) Segmentleri oku
    with open(args.segments_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments: List[Dict[str,Any]] = data["segments"]

    # 3) Diarization
    turns_raw = run_diarization(audio_path, args.hf_token, args.num_speakers, device=args.device)
    turns_norm, mapping = normalize_labels(turns_raw)

    # 4) Atama
    assigned = assign_segments_speakers(segments, turns_norm, min_overlap_ratio=args.min_overlap_ratio)

    # 5) (Opsiyonel) yakın segmentleri birleştir
    if args.merge_after:
        assigned = merge_adjacent_same_speaker(assigned, max_gap=0.12)

    # 6) speaker alanı zaten normalize edildi. JSON’u yaz
    out = {"segments": [
        {
            "id": int(seg["id"]),
            "seek": int(seg.get("seek", 0)),
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": seg["text"],
            "speaker": seg["speaker"]
        } for seg in assigned
    ]}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n✅ bitti. {len(out['segments'])} segment yazıldı.")
    print(f"📄 çıktı: {out_json}")
    if mapping:
        print("🔖 speaker eşlemesi (pyannote → normalleştirilmiş):")
        for k,v in mapping.items():
            print(f"   {k} → {v}")

if __name__ == "__main__":
    main()
