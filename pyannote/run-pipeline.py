#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None):
    print("[RUN]", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Komut hata ile bitti: {' '.join(map(str, cmd))}")

def safe_run(cmd, cwd=None):
    print("[TRY]", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def segments_from(data):
    if isinstance(data, list):
        return data, True
    return data.get("segments", []), False

def quality_metrics(json_path: Path):
    try:
        data = load_json(json_path)
    except Exception:
        return {"ok": False, "amb_seg": 1e9, "empty_words": 1e9, "total_words": 0}
    segs, _ = segments_from(data)
    amb_seg = 0
    empty_words = 0
    total_words = 0
    for seg in segs:
        if seg.get("ambiguous_speaker"):
            amb_seg += 1
        for w in seg.get("words", []):
            total_words += 1
            if w.get("speaker") in (None, "", "UNKNOWN"):
                empty_words += 1
    return {"ok": True, "amb_seg": amb_seg, "empty_words": empty_words, "total_words": total_words}

def better(a, b):
    if a["amb_seg"] != b["amb_seg"]:
        return a["amb_seg"] < b["amb_seg"]
    return a["empty_words"] < b["empty_words"]

def find_and_stabilize_whisper_json(out_dir: Path, base: str) -> Path:
    # out_dir içinde olası isimler
    candidates = [
        out_dir / f"{base}_whisperx.json",
        out_dir / f"{base}.json",
        out_dir / f"{base}.en.json",
        out_dir / f"{base}_aligned.json",
        out_dir / "transcription.json",
    ]
    found = None
    for c in candidates:
        if c.exists():
            try:
                _ = load_json(c)
                found = c
                break
            except Exception:
                pass
    if not found:
        # base ile başlayan en yeni json
        for p in sorted(out_dir.glob(f"{base}*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                _ = load_json(p)
                found = p
                break
            except Exception:
                pass
    if not found:
        # klasördeki en yeni json
        for p in sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                _ = load_json(p)
                found = p
                break
            except Exception:
                pass
    if not found:
        raise SystemExit(f"WhisperX JSON bulunamadı: {out_dir}")

    print(f"[INFO] WhisperX JSON bulundu: {found}")

    stable = out_dir / f"{base}_whisperx.json"
    if stable != found:
        data = load_json(found)
        save_json(stable, data)
        print(f"[INFO] WhisperX JSON sabitlendi: {stable}")
    else:
        print("[INFO] WhisperX JSON zaten sabit isimde.")
    return stable

def main():
    ap = argparse.ArgumentParser(description="WhisperX -> (reassign/overlap/auto/both) tek komut")
    # Zorunlu
    ap.add_argument("--audio", required=True, help="Giriş ses (wav/flac/ogg)")
    ap.add_argument("--hf_token", required=True, help="HF token")
    ap.add_argument("--postproc", default="auto", choices=["reassign","overlap","both","auto"])
    ap.add_argument("--out_dir", default="out_whisperx", help="Çıktı klasörü (WhisperX + postprocess)")

    # Script konumları
    ap.add_argument("--scripts_dir", default=None,
                    help="overlap_diarize_and_fuse.py ve reassign_speakers.py klasörü (örn: pyannote)")

    # WhisperX
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float32")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--vad_onset", type=float, default=0.05)
    ap.add_argument("--vad_offset", type=float, default=0.20)
    ap.add_argument("--chunk_size", type=int, default=10)
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=4)
    ap.add_argument("--align_model", default="WAV2VEC2_ASR_LARGE_LV60K_960H")
    ap.add_argument("--interpolate_method", default="linear")
    ap.add_argument("--beam_size", type=int, default=10)
    ap.add_argument("--best_of", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--patience", type=float, default=2.0)
    ap.add_argument("--segment_resolution", default="sentence")

    # reassign
    ap.add_argument("--reassign_window_ms", type=int, default=950)
    ap.add_argument("--reassign_sim_threshold", type=float, default=0.7)
    ap.add_argument("--reassign_refine_iters", type=int, default=4)
    ap.add_argument("--reassign_neigh_ms", type=int, default=700)
    ap.add_argument("--reassign_sim_margin", type=float, default=0.03)
    ap.add_argument("--reassign_maj_min", type=float, default=0.7)

    # overlap
    ap.add_argument("--fuse_maj_min", type=float, default=0.6)
    ap.add_argument("--overlap_mark_threshold", type=float, default=0.2)
    ap.add_argument("--embed_model", default="pyannote/embedding")
    ap.add_argument("--word_window_ms", type=int, default=800)
    ap.add_argument("--proto_min_seg_ms", type=int, default=500)
    ap.add_argument("--overlap_ambiguity_share", type=float, default=0.25)
    ap.add_argument("--sim_guard", type=float, default=0.03)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.audio).stem

    # Sabit dosya yolları (hepsi out_dir içinde!)
    whisper_json_std = out_dir / f"{base}_whisperx.json"
    reassign_json    = out_dir / f"{base}_reassign.json"
    overlap_json     = out_dir / f"{base}_overlap.json"
    final_json       = out_dir / f"{base}_final.json"

    # Script path’leri
    scripts_dir = Path(args.scripts_dir) if args.scripts_dir else Path(__file__).parent
    overlap_script  = scripts_dir / "overlap_diarize_and_fuse.py"
    reassign_script = scripts_dir / "reassign_speakers.py"

    print(f"[INFO] overlap_script:  {overlap_script.resolve()}")
    print(f"[INFO] reassign_script: {reassign_script.resolve()}")
    if not overlap_script.exists():
        print(f"[WARN] Bulunamadı: {overlap_script} (overlap adımı atlanacak)")
    if not reassign_script.exists():
        print(f"[WARN] Bulunamadı: {reassign_script} (reassign adımı atlanacak)")

    # 1) WhisperX
    whisper_cmd = [
        "whisperx", args.audio,
        "--model", args.model,
        "--language", args.language,
        "--device", args.device,
        "--compute_type", args.compute_type,
        "--batch_size", str(args.batch_size),
        "--vad_onset", str(args.vad_onset),
        "--vad_offset", str(args.vad_offset),
        "--chunk_size", str(args.chunk_size),
        "--diarize",
        "--min_speakers", str(args.min_speakers),
        "--max_speakers", str(args.max_speakers),
        "--align_model", args.align_model,
        "--interpolate_method", args.interpolate_method,
        "--beam_size", str(args.beam_size),
        "--best_of", str(args.best_of),
        "--temperature", str(args.temperature),
        "--patience", str(args.patience),
        "--highlight_words", "True",
        "--segment_resolution", args.segment_resolution,
        "--hf_token", args.hf_token,
        "--output_format", "json",
        "--print_progress", "True",
        "--verbose", "True",
        "--output_dir", str(out_dir),
    ]
    run(whisper_cmd)

    # 1b) Whisper JSON’u bul & sabitle
    whisper_json_std = find_and_stabilize_whisper_json(out_dir, base)

    need_reassign = args.postproc in ("reassign", "both", "auto") and reassign_script.exists()
    need_overlap  = args.postproc in ("overlap", "both", "auto")  and overlap_script.exists()

    overlap_ok = False
    reassign_ok = False

    # 2a) OVERLAP
    if need_overlap:
        fuse_cmd = [
            sys.executable, str(overlap_script),
            "--audio", args.audio,
            "--in_json", str(whisper_json_std),
            "--out_json", str(overlap_json),
            "--hf_token", args.hf_token,
            "--min_speakers", str(args.min_speakers),
            "--max_speakers", str(args.max_speakers),
            "--use_overlap_detector",
            "--maj_min", str(args.fuse_maj_min),
            "--overlap_mark_threshold", str(args.overlap_mark_threshold),
            "--embed_model", args.embed_model,
            "--word_window_ms", str(args.word_window_ms),
            "--proto_min_seg_ms", str(args.proto_min_seg_ms),
            "--overlap_ambiguity_share", str(args.overlap_ambiguity_share),
            "--sim_guard", str(args.sim_guard),
        ]
        res = safe_run(fuse_cmd)
        overlap_ok = (res.returncode == 0)

    # 2b) REASSIGN
    if need_reassign:
        reassign_cmd = [
            sys.executable, str(reassign_script),
            "--audio", args.audio,
            "--in_json", str(whisper_json_std),
            "--out_json", str(reassign_json),
            "--hf_token", args.hf_token,
            "--embedding_model", "pyannote/embedding",
            "--window_ms", str(args.reassign_window_ms),
            "--maj_min", str(args.reassign_maj_min),
            "--sim_threshold", str(args.reassign_sim_threshold),
            "--refine_iters", str(args.reassign_refine_iters),
            "--neigh_ms", str(args.reassign_neigh_ms),
            "--sim_margin", str(args.reassign_sim_margin),
        ]
        res = safe_run(reassign_cmd)
        reassign_ok = (res.returncode == 0)

    # 3) Karar & final
    chosen = None
    if args.postproc == "both":
        if overlap_ok:
            chosen = overlap_json
        elif reassign_ok:
            chosen = reassign_json
    elif args.postproc == "overlap":
        chosen = overlap_json if overlap_ok else None
    elif args.postproc == "reassign":
        chosen = reassign_json if reassign_ok else None
    else:  # auto
        if overlap_ok and reassign_ok:
            m_overlap  = quality_metrics(overlap_json)
            m_reassign = quality_metrics(reassign_json)
            print("[METRIC overlap]:", m_overlap)
            print("[METRIC reassign]:", m_reassign)
            chosen = overlap_json if better(m_overlap, m_reassign) else reassign_json
        elif overlap_ok:
            chosen = overlap_json
        elif reassign_ok:
            chosen = reassign_json

    if not chosen:
        raise SystemExit("Ne overlap ne reassign başarılı sonuç üretmedi. Logları kontrol et.")

    final_data = load_json(chosen)
    save_json(final_json, final_data)
    print(f"[OK] Final çıktı -> {final_json}")

if __name__ == "__main__":
    main()
