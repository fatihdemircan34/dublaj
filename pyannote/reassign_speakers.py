#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding tabanlı speaker yeniden atama (post-process) - İyileştirilmiş versiyon
"""

import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
from collections import Counter, defaultdict

# ------------------ Argümanlar ------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="WAV/FLAC/OGG path (mono tercih)")
    p.add_argument("--in_json", required=True, help="WhisperX JSON")
    p.add_argument("--out_json", required=True, help="Çıkış JSON")
    p.add_argument("--hf_token", default=None, help="HuggingFace token (gerekliyse)")
    p.add_argument("--embedding_model", default="pyannote/embedding",
                   help="pyannote/embedding veya speechbrain/spkrec-ecapa-voxceleb")
    
    # Optimize edilmiş parametreler - kısa replikler için
    p.add_argument("--window_ms", type=int, default=400,  # 950'den düşürdük
                   help="Kelime embedding penceresi (ms)")
    p.add_argument("--maj_min", type=float, default=0.55,  # 0.6'dan düşürdük
                   help="Segment çoğunluk eşiği (0-1)")
    p.add_argument("--min_word_ms", type=int, default=80,  # 120'den düşürdük
                   help="Çok kısa kelimeleri yine de değerlendir (ms)")
    p.add_argument("--sim_threshold", type=float, default=0.5,  # 0.6'dan düşürdük
                   help="Kelime bazında düşük güven eşiği")
    p.add_argument("--refine_iters", type=int, default=3,  # 2'den arttırdık
                   help="Prototip güncelleme/yeniden atama iterasyon sayısı")
    p.add_argument("--neigh_ms", type=int, default=300,  # 500'den düşürdük
                   help="Komşuluk penceresi (±ms) komşu çoğunluk için")
    p.add_argument("--sim_margin", type=float, default=0.03,  # Aynı
                   help="Komşuya yaslamak için gerekli benzerlik marjı")
    
    # Yeni parametreler
    p.add_argument("--cross_segment", action="store_true", default=True,
                   help="Segment sınırlarında cross-check yap")
    p.add_argument("--adaptive_window", action="store_true", default=True,
                   help="Kelime uzunluğuna göre window adaptasyonu")
    
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# ------------------ Yardımcılar ------------------
def l2norm(x, eps=1e-12):
    n = np.linalg.norm(x) + eps
    return x / n

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    return wav, sr

def crop_window(sr, total_s, center_s, win_ms, adaptive=False, word_duration_ms=None):
    """
    Adaptive window: Kısa kelimeler için daha küçük window
    """
    if adaptive and word_duration_ms and word_duration_ms < 200:
        # Kısa kelimeler için window'u küçült
        win_ms = min(win_ms, word_duration_ms * 2)
    
    half = (win_ms/1000.0) / 2.0
    a = max(0.0, center_s - half)
    b = min(total_s, center_s + half)
    if b <= a:
        b = a + 0.02
    i0 = int(round(a * sr))
    i1 = int(round(b * sr))
    return i0, i1

def duration(w):
    return float(w.get("end", 0.0)) - float(w.get("start", 0.0))

def seg_time_bounds(seg):
    s = float(seg.get("start", 0.0))
    e = float(seg.get("end", s))
    return s, e

def word_mid(w, seg_start, seg_end):
    ws = float(w.get("start", seg_start))
    we = float(w.get("end", seg_end))
    return 0.5 * (ws + we)

# ------------------ Embedder Seçimi ------------------
def load_embedder(name, device, hf_token=None):
    name = (name or "").strip().lower()
    if name == "speechbrain/spkrec-ecapa-voxceleb":
        try:
            from speechbrain.pretrained import EncoderClassifier
        except Exception as e:
            raise RuntimeError("speechbrain gerekli: `pip install speechbrain`") from e
        enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        def embed_fn(wave_1xT, sr):
            if sr != 16000:
                wave_1xT = torchaudio.functional.resample(wave_1xT, sr, 16000)
            with torch.no_grad():
                e = enc.encode_batch(wave_1xT.to(device)).squeeze(0).squeeze(0).cpu().numpy()
            return l2norm(e)
        return embed_fn

    # Varsayılan: pyannote/embedding
    try:
        from pyannote.audio import Model
    except Exception as e:
        raise RuntimeError("pyannote.audio gerekli: `pip install pyannote.audio`") from e

    mdl = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token, strict=False)
    mdl = mdl.to(device).eval()

    def embed_fn(wave_1xT, sr):
        if sr != 16000:
            wave_1xT = torchaudio.functional.resample(wave_1xT, sr, 16000)
        with torch.no_grad():
            out = mdl(wave_1xT.to(device))
            if isinstance(out, dict):
                out = out.get("embedding", out)
            e = out.detach().cpu().numpy().squeeze()
        return l2norm(e)
    return embed_fn

# ------------------ İyileştirilmiş Prototip & Atama ------------------
def build_prototypes(assignments, min_samples=3):
    """
    assignments: list of (speaker, emb)
    min_samples: Prototip oluşturmak için minimum örnek sayısı
    """
    by_spk = defaultdict(list)
    for spk, emb in assignments:
        if spk:
            by_spk[spk].append(emb)
    
    protos = {}
    for spk, embs in by_spk.items():
        if len(embs) >= min_samples:  # Çok az örnek varsa prototip oluşturma
            protos[spk] = l2norm(np.mean(np.stack(embs, axis=0), axis=0))
    return protos

def nearest_proto(emb, proto_keys, protos_mat):
    sims = np.dot(protos_mat, emb)
    idx = int(np.argmax(sims))
    return proto_keys[idx], float(sims[idx])

def cross_segment_neighborhood(segments, si, wi, neigh_ms):
    """
    Segment sınırlarını da kontrol eden gelişmiş neighborhood
    """
    seg = segments[si]
    s0, e0 = seg_time_bounds(seg)
    words = seg.get("words", [])
    
    if not words:
        return None, 0.0
    
    target = words[wi]
    center = word_mid(target, s0, e0)
    left = center - (neigh_ms/1000.0)
    right = center + (neigh_ms/1000.0)
    
    c = Counter()
    total = 0.0
    
    # Mevcut segment'teki kelimeler
    for w in words:
        mid = word_mid(w, s0, e0)
        if left <= mid <= right:
            d = max(0.0, duration(w))
            spk = w.get("speaker")
            if spk:
                c[spk] += d
                total += d
    
    # Önceki segment'ten kelimeler (segment başındaysak)
    if si > 0 and wi < 3:  # İlk 3 kelime
        prev_seg = segments[si-1]
        prev_words = prev_seg.get("words", [])
        if prev_words:
            ps0, pe0 = seg_time_bounds(prev_seg)
            # Son 3 kelimeye bak
            for w in prev_words[-3:]:
                mid = word_mid(w, ps0, pe0)
                if left <= mid <= right:
                    d = max(0.0, duration(w))
                    spk = w.get("speaker")
                    if spk:
                        c[spk] += d
                        total += d
    
    # Sonraki segment'ten kelimeler (segment sonundaysak)
    if si < len(segments)-1 and wi >= len(words)-3:  # Son 3 kelime
        next_seg = segments[si+1]
        next_words = next_seg.get("words", [])
        if next_words:
            ns0, ne0 = seg_time_bounds(next_seg)
            # İlk 3 kelimeye bak
            for w in next_words[:3]:
                mid = word_mid(w, ns0, ne0)
                if left <= mid <= right:
                    d = max(0.0, duration(w))
                    spk = w.get("speaker")
                    if spk:
                        c[spk] += d
                        total += d
    
    if not c or total <= 0:
        return None, 0.0
    
    spk, dur = c.most_common(1)[0]
    return spk, dur / total

def detect_speaker_change_points(segments, threshold=0.4):
    """
    Potansiyel speaker değişim noktalarını tespit et
    """
    change_points = []
    
    for si, seg in enumerate(segments):
        words = seg.get("words", [])
        for wi in range(1, len(words)):
            prev_sim = words[wi-1].get("speaker_sim", 0)
            curr_sim = words[wi].get("speaker_sim", 0)
            
            # Her ikisi de düşük güvenliyse, muhtemel değişim noktası
            if prev_sim < threshold and curr_sim < threshold:
                change_points.append((si, wi))
    
    return change_points

# ------------------ Ana Akış ------------------
def main():
    args = parse_args()

    # Ses
    wav, sr = load_audio(args.audio)
    total_dur_s = wav.shape[-1] / sr

    # Embedder
    try:
        embed_fn = load_embedder(args.embedding_model, args.device, args.hf_token)
    except Exception as e:
        if "pyannote" in (args.embedding_model or ""):
            print(f"[WARN] {args.embedding_model} yüklenemedi ({e}). SpeechBrain'e düşüyorum...")
            embed_fn = load_embedder("speechbrain/spkrec-ecapa-voxceleb", args.device, None)
        else:
            raise

    # JSON yükle
    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", None)
    if segments is None and isinstance(data, list):
        segments = data
        root_is_list = True
    else:
        root_is_list = False

    # Kelime embedding'leri
    word_emb_cache = []
    min_len = int(sr * (args.min_word_ms/1000.0))
    min_len = max(min_len, int(sr * 0.4))
    
    with torch.no_grad():
        for si, seg in enumerate(segments):
            s0, e0 = seg_time_bounds(seg)
            words = seg.get("words", [])
            
            for wi, w in enumerate(words):
                ws = float(w.get("start", s0))
                we = float(w.get("end", e0))
                if we <= ws:
                    continue
                
                mid = 0.5*(ws+we)
                word_dur_ms = (we - ws) * 1000
                
                # Adaptive window
                i0, i1 = crop_window(
                    sr, total_dur_s, mid, args.window_ms,
                    adaptive=args.adaptive_window,
                    word_duration_ms=word_dur_ms
                )
                
                audio_chunk = wav[:, i0:i1]
            

                # Pyannote modeli için minimum chunk size kontrolü
                # 16kHz'de minimum 400ms = 6400 sample gerekiyor
                min_samples_for_model = int(0.4 * sr)  # 400ms

                if audio_chunk.shape[-1] < min_samples_for_model:
                    # Padding ekle
                    pad_needed = min_samples_for_model - audio_chunk.shape[-1]
                    audio_chunk = torch.nn.functional.pad(audio_chunk, (0, pad_needed), mode='constant', value=0)
                    
                    # Debug için
                    print(f"[DEBUG] Kelime çok kısa, padding eklendi: {audio_chunk.shape[-1]} samples")

                emb = embed_fn(audio_chunk, sr)

                word_emb_cache.append(((si, wi), emb))

    # İlk prototipler - ambiguous segment'leri farklı değerlendir
    init_assign = []
    ambiguous_embs = []
    
    for (si, wi), emb in word_emb_cache:
        seg = segments[si]
        word = seg["words"][wi]
        
        # Eğer segment ambiguous ise ve kelime güveni düşükse
        if seg.get("ambiguous_speaker") and word.get("score", 1.0) < 0.5:
            ambiguous_embs.append(emb)
        else:
            cur_spk = word.get("speaker")
            if cur_spk:
                init_assign.append((cur_spk, emb))
    
    speaker_protos = build_prototypes(init_assign)

    # Etiket yoksa veya yetersizse, K-means ile başlat
    if len(speaker_protos) < 2 and word_emb_cache:
        print("[INFO] Yetersiz prototip, K-means ile başlatılıyor...")
        from sklearn.cluster import KMeans
        
        all_embs = np.stack([e for (_, e) in word_emb_cache], axis=0)
        
        # 2-3 cluster dene
        kmeans = KMeans(n_clusters=min(3, len(all_embs)), random_state=42)
        labels = kmeans.fit_predict(all_embs)
        
        # Prototipleri oluştur
        for i in range(kmeans.n_clusters):
            cluster_embs = all_embs[labels == i]
            if len(cluster_embs) > 0:
                speaker_protos[f"SPEAKER_{i:02d}"] = l2norm(np.mean(cluster_embs, axis=0))

    if not speaker_protos:
        raise RuntimeError("Speaker prototipleri oluşturulamadı.")

    print(f"[INFO] {len(speaker_protos)} speaker prototipi oluşturuldu")

    # --------- Iteratif yeniden atama ---------
    for it in range(max(1, args.refine_iters)):
        proto_keys = list(speaker_protos.keys())
        protos_mat = np.stack([speaker_protos[k] for k in proto_keys], axis=0)

        # 1) En yakın prototipe ata + sim yaz
        for (si, wi), emb in word_emb_cache:
            spk, sim = nearest_proto(emb, proto_keys, protos_mat)
            wd = segments[si]["words"][wi]
            wd["speaker"] = spk
            wd["speaker_sim"] = sim

        # 2) Düşük sim'ler için gelişmiş komşu çoğunluğa yaslama
        changed = 0
        for (si, wi), emb in word_emb_cache:
            wd = segments[si]["words"][wi]
            sim = float(wd.get("speaker_sim", 0.0))
            
            if sim >= args.sim_threshold:
                continue  # güvenli
            
            # Cross-segment neighborhood kullan
            if args.cross_segment:
                neigh_spk, neigh_ratio = cross_segment_neighborhood(segments, si, wi, args.neigh_ms)
            else:
                from reassign_speakers import neighborhood_majority
                neigh_spk, neigh_ratio = neighborhood_majority(segments, si, wi, args.neigh_ms)
            
            if not neigh_spk or neigh_spk not in speaker_protos:
                continue
            
            neigh_proto = speaker_protos[neigh_spk]
            neigh_sim = float(np.dot(neigh_proto, emb))
            
            # Yaslama kuralı
            if neigh_sim >= (sim + args.sim_margin):
                wd["speaker"] = neigh_spk
                wd["speaker_sim"] = neigh_sim
                changed += 1

        # 3) Prototipleri güncelle
        new_assign = []
        for (si, wi), emb in word_emb_cache:
            spk = segments[si]["words"][wi].get("speaker")
            new_assign.append((spk, emb))
        
        new_protos = build_prototypes(new_assign, min_samples=2)
        
        if new_protos:
            speaker_protos = new_protos
        
        print(f"[INFO] Iteration {it+1}: {changed} kelime güncellendi")
        
        if changed == 0:
            break

    # Speaker değişim noktalarını kontrol et (opsiyonel)
    change_points = detect_speaker_change_points(segments, args.sim_threshold)
    if change_points:
        print(f"[INFO] {len(change_points)} potansiyel speaker değişim noktası tespit edildi")

    # Eşik altı kelimeleri ambiguous işaretle
    for (si, wi), _ in word_emb_cache:
        wd = segments[si]["words"][wi]
        sim = float(wd.get("speaker_sim", 0.0))
        if sim < args.sim_threshold:
            wd["ambiguous"] = True

    # Segment-level speaker'ı güncelle
    for seg in segments:
        words = seg.get("words", [])
        dur_by_spk = defaultdict(float)
        total = 0.0
        
        for w in words:
            d = max(0.0, duration(w))
            spk = w.get("speaker")
            if spk:
                dur_by_spk[spk] += d
                total += d
        
        if dur_by_spk and total > 0:
            spk, dur = max(dur_by_spk.items(), key=lambda x: x[1])
            ratio = dur / total
            
            if ratio >= args.maj_min:
                seg["speaker"] = spk
                seg["ambiguous_speaker"] = False
            else:
                # Çoklu speaker var
                seg["speaker"] = None
                seg["ambiguous_speaker"] = True
                seg["speakers"] = dict(dur_by_spk)  # Tüm speaker'ları sakla

    # Çıkış
    out_obj = segments if root_is_list else {**data, "segments": segments}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK] Yeniden atama bitti -> {args.out_json}")

if __name__ == "__main__":
    main()