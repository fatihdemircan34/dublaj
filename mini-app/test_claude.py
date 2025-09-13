"""
process_video_wordwise.py

Gereksinimler:
  pip install faster-whisper==1.2.0 pyannote.audio torch

Sistem:
  - ffmpeg + ffprobe (PATH'te olmalı)
Kimlik:
  - HF_TOKEN (gated pyannote modelleri için şart)

Özet:
  Video -> FFmpeg (16k mono) -> Whisper large-v3 (kelime zamanları, CUDA)
        -> pyannote diarization (CUDA)
        -> pyannote OverlappedSpeechDetection (CUDA, segmentation=pyannote/segmentation-3.0)
        -> Viterbi (kelime bazlı, overlap-aware)
        -> (opsiyonel) Semantik düzeltme: 'heuristic' ya da 'llm'

Çıktılar:
  - output/words_diarized.jsonl   (kelime-kelime)
  - output/dialog_sentences.json  (cümle-cümle, speaker + zaman; text boş)
  - output/transcript_words.srt   (opsiyonel, 1 kelime = 1 cue)
  - output/transcript_diarized.json (özet)
  - output/process.debug.{log,jsonl}
"""

from __future__ import annotations
import os, json, shutil, subprocess, logging, math, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch

# ------------------------- Logging & Debug -------------------------

def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _attach_file_logger(log_path: Path):
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

class DebugWriter:
    def __init__(self, out_dir: Path):
        self.jsonl = (out_dir / "process.debug.jsonl")
        self.t0 = time.perf_counter()
        self.last_ts = self.t0
        self.jsonl.write_text("", encoding="utf-8")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def _gpu_stats(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"device": "cpu"}
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        idx = torch.cuda.current_device()
        return {
            "device": f"cuda:{idx}",
            "gpu_name": torch.cuda.get_device_name(idx),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "mem_alloc_mb": round(torch.cuda.memory_allocated(idx) / (1024**2), 1),
            "mem_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024**2), 1),
            "mem_peak_mb": round(torch.cuda.max_memory_allocated(idx) / (1024**2), 1),
        }

    def snap(self, step: str, **metrics):
        now = time.perf_counter()
        delta = now - self.last_ts
        elapsed = now - self.t0
        self.last_ts = now
        rec = {
            "step": step,
            "t_delta_sec": round(delta, 3),
            "t_elapsed_sec": round(elapsed, 3),
            "gpu": self._gpu_stats(),
            **metrics
        }
        logging.info("[DEBUG:%s] %s", step, json.dumps(rec, ensure_ascii=False))
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ------------------------- Yardımcılar -------------------------

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg bulunamadı. Lütfen ffmpeg'i kurup PATH'e ekleyin.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe bulunamadı. Lütfen ffprobe'u kurup PATH'e ekleyin.")

def _to_srt_time(t: float) -> str:
    if t < 0: t = 0.0
    ms = int(round((t - int(t)) * 1000))
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _write_word_srt(words: List[Dict[str, Any]], srt_path: Path) -> None:
    with srt_path.open("w", encoding="utf-8") as f:
        for i, w in enumerate(words, 1):
            text = w["word"]
            if w.get("overlap"):
                text = f"{text} +"   # overlap görsel işareti
            f.write(f"{i}\n{_to_srt_time(w['start'])} --> {_to_srt_time(w['end'])}\n{w['speaker']}: {text}\n\n")

def _save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _ffprobe_info(media_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-show_entries", "format=duration",
        "-of", "json", str(media_path)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {}
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return {}
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    streams = data.get("streams", [])
    sr = int(streams[0].get("sample_rate", 0)) if streams else 0
    ch = int(streams[0].get("channels", 0)) if streams else 0
    return {"duration_sec": duration, "sample_rate": sr, "channels": ch}

def _extract_audio_ffmpeg(video_path: Path, out_wav: Path, sr: int = 16000, dbg: Optional[DebugWriter] = None) -> Dict[str, Any]:
    _check_ffmpeg()
    vinf = _ffprobe_info(video_path)
    if dbg: dbg.snap("FFPROBE_VIDEO", **{f"video_{k}": v for k, v in vinf.items()}, file=str(video_path))
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", str(out_wav)]
    logging.info("FFmpeg ile ses çıkarılıyor...")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    if proc.returncode != 0:
        tail = proc.stderr.splitlines()[-1] if proc.stderr else "bilinmiyor"
        raise RuntimeError(f"ffmpeg başarısız: {tail}")
    winf = _ffprobe_info(out_wav)
    size_mb = out_wav.stat().st_size / (1024**2)
    if dbg: dbg.snap("EXTRACT_AUDIO_DONE", out=str(out_wav), secs=round(t1 - t0, 3), size_mb=round(size_mb, 2), **winf)
    return winf

def _language_to_whisper(lang_code: str) -> str:
    if not lang_code: return "en"
    lc = lang_code.lower()
    if lc.startswith("en"): return "en"
    if lc.startswith("tr"): return "tr"
    return lc.split("-")[0]

# ------------------------- ASR / Diar / OSD -------------------------

def _transcribe_whisper(wav_path: Path, language_code: str, dbg: DebugWriter) -> List[Dict[str, Any]]:
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    dbg.snap("ASR_INIT", device=device, compute_type=compute_type, torch_cuda=torch.version.cuda, torch_version=torch.__version__)
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)

    logging.info("ASR (Whisper large-v3) başlıyor... device=%s, compute=%s", device, compute_type)
    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        str(wav_path),
        language=_language_to_whisper(language_code),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    t1 = time.perf_counter()

    words = []
    seg_count = 0
    for seg in segments:
        seg_count += 1
        for w in (seg.words or []):
            if w.start is not None and w.end is not None and w.word:
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                    "prob": float(getattr(w, "probability", 0.0)) if hasattr(w, "probability") and w.probability is not None else None
                })
    wav_info = _ffprobe_info(wav_path)
    wps = (len(words) / wav_info.get("duration_sec", 1.0)) if wav_info.get("duration_sec", 0) > 0 else None
    dbg.snap("ASR_DONE", wav=str(wav_path), seg_count=seg_count, word_count=len(words), secs=round(t1 - t0, 3), audio_dur_sec=round(wav_info.get("duration_sec", 0), 3), words_per_sec=(round(wps, 2) if wps else None))
    return words

def _diarize_pyannote(wav_path: Path, speaker_count: int, dbg: DebugWriter) -> List[Dict[str, Any]]:
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN tanımlı değil. Hugging Face token gerekiyor (pyannote diarization için).")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
        dbg.snap("DIAR_INIT", device="cuda")
    else:
        dbg.snap("DIAR_INIT", device="cpu")
    t0 = time.perf_counter()
    diar = pipe(str(wav_path), num_speakers=max(1, speaker_count or 1))
    t1 = time.perf_counter()

    diar_segments = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        diar_segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": spk})
    diar_segments.sort(key=lambda d: d["start"])
    total_talk = sum(ds["end"] - ds["start"] for ds in diar_segments)
    dbg.snap("DIAR_DONE", diar_segments=len(diar_segments), est_speakers=len({d['speaker'] for d in diar_segments}), total_speaking_time_sec=round(total_talk, 2), secs=round(t1 - t0, 3))
    return diar_segments

def _osd_pyannote(wav_path: Path, dbg: DebugWriter) -> List[Tuple[float, float]]:
    """
    Overlapped Speech Detection (OSD) -> overlapped segment listesi [(start,end),...]
    3.x: OverlappedSpeechDetection(segmentation='pyannote/segmentation-3.0')
    """
    from pyannote.audio.pipelines import OverlappedSpeechDetection
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN tanımlı değil. Hugging Face token gerekiyor (pyannote OSD için).")
    pipeline = OverlappedSpeechDetection(segmentation="pyannote/segmentation-3.0")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        dbg.snap("OSD_INIT", device="cuda")
    else:
        dbg.snap("OSD_INIT", device="cpu")
    HYP = {"min_duration_on": 0.08, "min_duration_off": 0.08}
    pipeline.instantiate(HYP)
    t0 = time.perf_counter()
    ann = pipeline(str(wav_path))  # Annotation: overlapped bölgeler
    t1 = time.perf_counter()

    intervals = []
    try:
        for seg in ann.itersegments():
            intervals.append((float(seg.start), float(seg.end)))
    except Exception:
        tl = ann.get_timeline()
        for seg in tl:
            intervals.append((float(seg.start), float(seg.end)))
    total_ov = sum(e - s for s, e in intervals)
    dbg.snap("OSD_DONE", intervals=len(intervals), total_overlap_sec=round(total_ov, 3), secs=round(t1 - t0, 3))
    return intervals

def _canonicalize_speakers(diar_segments: List[Dict[str, Any]], dbg: DebugWriter) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    first_seen = {}
    for ds in diar_segments:
        if ds["speaker"] not in first_seen:
            first_seen[ds["speaker"]] = ds["start"]
    ordered = sorted(first_seen.items(), key=lambda kv: kv[1])
    mapping = {old: f"SPEAKER_{i+1:02d}" for i, (old, _) in enumerate(ordered)}
    new = []
    for ds in diar_segments:
        new.append({"start": ds["start"], "end": ds["end"], "speaker": mapping.get(ds["speaker"], ds["speaker"])})
    dbg.snap("SPEAKER_CANONICALIZE", unique_in=len(first_seen), unique_out=len(set(d["speaker"] for d in new)))
    return new, mapping

# ------------------------- Kelime-bazlı atama -------------------------

@dataclass
class OverlapPolicy:
    strategy: str = "smart"     # 'smart' (OSD varsa duplicate), 'top', 'duplicate', 'mark'
    alpha: float = 1.25
    min_dup_dur: float = 0.06

@dataclass
class ViterbiParams:
    switch_penalty: float = 0.35
    center_lambda: float = 0.20
    small_gap: float = 0.12
    mid_gap: float = 0.35
    long_gap: float = 0.80

def _overlap_amount(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def _center_distance(mid: float, s: float, e: float) -> float:
    if s <= mid <= e: return 0.0
    return min(abs(mid - s), abs(mid - e))

def _speaker_set(diar_segments: List[Dict[str, Any]]) -> List[str]:
    seen = []
    for ds in diar_segments:
        spk = ds["speaker"]
        if spk not in seen:
            seen.append(spk)
    return seen

def _build_overlap_tables(words: List[Dict[str, Any]], diar_segments: List[Dict[str, Any]], speakers: List[str], dbg: DebugWriter) -> List[Dict[str, Dict[str, float]]]:
    tables: List[Dict[str, Dict[str, float]]] = []
    diar_by_spk: Dict[str, List[Dict[str, Any]]] = {s: [] for s in speakers}
    for ds in diar_segments:
        diar_by_spk[ds["speaker"]].append(ds)

    multi_over = 0
    for w in words:
        ws, we = w["start"], w["end"]
        dur = max(we - ws, 1e-6)
        mid = 0.5 * (ws + we)
        row: Dict[str, Dict[str, float]] = {}
        overlaps = 0
        for spk in speakers:
            blocks = diar_by_spk.get(spk, [])
            raw, cdist = 0.0, float("inf")
            for b in blocks:
                raw += _overlap_amount(ws, we, b["start"], b["end"])
                cdist = min(cdist, _center_distance(mid, b["start"], b["end"]))
            o_norm = min(1.0, raw / dur)
            d_norm = min(cdist / 0.4, 3.0) if math.isfinite(cdist) else 3.0
            row[spk] = {"o_raw": raw, "o_norm": o_norm, "d_norm": d_norm}
            if raw > 0:
                overlaps += 1
        if overlaps >= 2:
            multi_over += 1
        tables.append(row)
    dbg.snap("OVERLAP_TABLES_DONE", words=len(words), speakers=len(speakers), multi_speaker_overlap_words=multi_over)
    return tables

def _emission_cost(o_norm: float, d_norm: float, prob: Optional[float], center_lambda: float) -> float:
    eps = 1e-3
    p = max(eps, min(1.0, o_norm))
    if prob is not None:
        p = max(eps, min(1.0, 0.7 * p + 0.3 * prob))
    return -math.log(p) + center_lambda * d_norm

def _gap_factor(prev_end: float, cur_start: float, vp: ViterbiParams) -> float:
    gap = max(0.0, cur_start - prev_end)
    if gap < vp.small_gap: return 1.6
    if gap < vp.mid_gap: return 1.0
    if gap < vp.long_gap: return 0.7
    return 0.4

def _assign_speakers_viterbi(words: List[Dict[str, Any]], overlap_tables: List[Dict[str, Dict[str, float]]], speakers: List[str], vp: ViterbiParams, dbg: DebugWriter) -> List[str]:
    if not speakers:
        return ["SPEAKER_01"] * len(words)
    S, N = len(speakers), len(words)
    dp = [[float("inf")] * S for _ in range(N)]
    bp = [[-1] * S for _ in range(N)]

    # t=0
    for si, spk in enumerate(speakers):
        feats = overlap_tables[0][spk]
        dp[0][si] = _emission_cost(feats["o_norm"], feats["d_norm"], words[0].get("prob"), vp.center_lambda)

    # t>0
    for i in range(1, N):
        for si, spk in enumerate(speakers):
            feats = overlap_tables[i][spk]
            emit = _emission_cost(feats["o_norm"], feats["d_norm"], words[i].get("prob"), vp.center_lambda)
            best_cost, best_prev = float("inf"), -1
            for pj in range(S):
                trans = 0.0 if pj == si else vp.switch_penalty * _gap_factor(words[i-1]["end"], words[i]["start"], vp)
                cost = dp[i-1][pj] + trans + emit
                if cost < best_cost:
                    best_cost, best_prev = cost, pj
            dp[i][si], bp[i][si] = best_cost, best_prev

    path = [-1] * N
    last = min(range(S), key=lambda si: dp[N-1][si])
    path[N-1] = last
    for i in range(N-2, -1, -1):
        path[i] = bp[i+1][path[i+1]]
    assigned = [speakers[si] for si in path]
    switches = sum(1 for i in range(1, N) if assigned[i] != assigned[i-1])
    dbg.snap("VITERBI_DONE", words=N, speakers=S, switches=switches)
    return assigned

def _flag_osd_for_words(words: List[Dict[str, Any]], osd_intervals: List[Tuple[float, float]]) -> None:
    for w in words:
        mid = 0.5 * (w["start"] + w["end"])
        w["osd"] = any(s <= mid <= e for s, e in osd_intervals)

def _post_overlap_policy(words: List[Dict[str, Any]], overlap_tables: List[Dict[str, Dict[str, float]]], assigned: List[str], policy: OverlapPolicy, dbg: DebugWriter) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ambiguous_cnt, duplicated_cnt = 0, 0
    for i, w in enumerate(words):
        row = overlap_tables[i]
        ranked = sorted(((spk, row[spk]["o_raw"]) for spk in row.keys()), key=lambda kv: kv[1], reverse=True)
        best_spk, best_raw = ranked[0]
        second_spk, second_raw = (ranked[1] if len(ranked) > 1 else (None, 0.0))
        ambiguous = second_spk is not None and (best_raw < policy.alpha * second_raw)

        choose_spk = assigned[i]
        base = dict(w); base["speaker"] = choose_spk; base["overlap"] = bool(ambiguous)
        if ambiguous:
            ambiguous_cnt += 1
            base["alt_speakers"] = [spk for spk, _ in ranked[1:3]]

        allow_dup = (policy.strategy == "duplicate") or (policy.strategy == "smart" and w.get("osd"))
        if allow_dup and ambiguous and second_raw >= policy.min_dup_dur:
            out.append(base)
            clone = dict(w); clone["speaker"] = second_spk; clone["overlap"] = True; clone["alt_speakers"] = [best_spk]; clone["osd"] = w.get("osd", False)
            out.append(clone)
            duplicated_cnt += 1
            continue

        out.append(base)

    dbg.snap("POST_OVERLAP_POLICY_DONE", words=len(words), ambiguous_words=ambiguous_cnt, duplicated_words=duplicated_cnt, strategy=policy.strategy)
    return out

# ------------------------- Semantik düzeltme (opsiyonel) -------------------------

def _semantic_repair(words: List[Dict[str, Any]], window: int = 40, mode: str = "none", dbg: Optional[DebugWriter] = None, llm_model: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    mode:
      - "none"      : düzeltme yok
      - "heuristic" : noktalama/bağlaç/uzunluk çoğunluğu ile geçişleri yumuşat
      - "llm"       : OpenAI vb. API ile minimal flip (opsiyonel)
    """
    if mode == "none":
        if dbg: dbg.snap("SEMANTIC_REPAIR_SKIPPED", mode=mode)
        return words

    changed = 0

    if mode == "heuristic":
        K = 6
        i = 1
        while i < len(words):
            if words[i]["speaker"] != words[i-1]["speaker"]:
                l = max(0, i-K); r = min(len(words), i+K)
                left_block = [w["speaker"] for w in words[l:i]]
                right_block = [w["speaker"] for w in words[i:r]]
                left_major = max(set(left_block), key=left_block.count) if left_block else words[i-1]["speaker"]
                right_major = max(set(right_block), key=right_block.count) if right_block else words[i]["speaker"]
                if (i+1 < len(words)) and (words[i+1]["speaker"] == words[i-1]["speaker"]) and (words[i]["word"] not in {",", ".", "?", "!"}):
                    words[i]["speaker"] = words[i-1]["speaker"]; changed += 1
                elif left_major == words[i-1]["speaker"] and right_major == words[i-1]["speaker"]:
                    words[i]["speaker"] = words[i-1]["speaker"]; changed += 1
            i += 1
        if dbg: dbg.snap("SEMANTIC_REPAIR_DONE", mode=mode, changed=changed)
        return words

    if mode == "llm":
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            if dbg: dbg.snap("SEMANTIC_REPAIR_DISABLED", reason="openai sdk yok veya anahtar yok")
            return words

        def windowed(seq, n):
            for i in range(0, len(seq), n):
                yield i, seq[i:i+n]

        for base, chunk in windowed(words, window):
            prompt = {
                "task": "speaker_boundary_repair",
                "instruction": "Given words with timestamps and speaker tags, propose minimal index->speaker flips to maximize semantic continuity. Only flip if clearly improves coherence.",
                "words": [{"i": base+i, "t": [w["start"], w["end"]], "w": w["word"], "s": w["speaker"]} for i, w in enumerate(chunk)]
            }
            try:
                rsp = client.responses.create(model=llm_model or "gpt-4o-mini", input=json.dumps(prompt))
                txt = rsp.output_text
            except Exception:
                from openai import OpenAI as _Client
                c = _Client()
                msg = c.chat.completions.create(
                    model=llm_model or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You output only JSON: {changes:[{index:int,speaker:str}]}"},
                        {"role": "user", "content": json.dumps(prompt)}
                    ]
                )
                txt = msg.choices[0].message.content

            try:
                data = json.loads(txt)
                for ch in data.get("changes", []):
                    i = int(ch["index"]); spk = str(ch["speaker"])
                    if 0 <= i < len(words):
                        words[i]["speaker"] = spk
                        changed += 1
            except Exception:
                pass

        if dbg: dbg.snap("SEMANTIC_REPAIR_DONE", mode=mode, changed=changed)
        return words

    if dbg: dbg.snap("SEMANTIC_REPAIR_SKIPPED", mode=mode)
    return words

# ------------------------- Cümle-bazlı (diyalog) oluşturma -------------------------

_EOS_RE = re.compile(r"[.?!…]+$")
_ONLY_EOS_RE = re.compile(r"^[.?!…]+$")

def _is_eos_token(tok: Optional[str]) -> bool:
    if not tok:
        return False
    t = tok.strip()
    return bool(_ONLY_EOS_RE.fullmatch(t) or _EOS_RE.search(t))

def _words_to_dialog_sentences(
        words: List[Dict[str, Any]],
        sentence_max_pause: float = 0.7,
        sentence_max_tokens: int = 50,
        dbg: Optional[DebugWriter] = None,
) -> List[Dict[str, Any]]:
    """
    Kelime-kelime listeden (speaker etiketli) cümle-cümle diyalog listesi üretir.
    - Her konuşmacı için ayrı akış, sonra zaman çizgisinde birleştirilir.
    - Sınırlar: cümle sonu noktalaması, uzun durak (pause), aşırı token sayısı.
    Çıktı: [{'start','end','speaker','text': ''}, ...]
    """
    by_spk: Dict[str, List[Dict[str, Any]]] = {}
    for w in words:
        spk = w.get("speaker") or "SPEAKER_01"
        by_spk.setdefault(spk, []).append(w)
    for spk in by_spk:
        by_spk[spk].sort(key=lambda x: (x["start"], x["end"]))

    sentences: List[Dict[str, Any]] = []
    for spk, ws in by_spk.items():
        if not ws: continue
        cur_tokens: List[str] = []
        cur_start = ws[0]["start"]; cur_end = ws[0]["end"]
        for i, w in enumerate(ws):
            token = w.get("word", "")
            if not cur_tokens:
                cur_start = w["start"]; cur_end = w["end"]
            cur_tokens.append(token); cur_end = max(cur_end, w["end"])
            next_gap = None
            if i + 1 < len(ws):
                next_gap = ws[i+1]["start"] - w["end"]
            boundary = False
            if _is_eos_token(token): boundary = True
            if next_gap is not None and next_gap > sentence_max_pause: boundary = True
            if len(cur_tokens) >= sentence_max_tokens: boundary = True
            if boundary:
                sentences.append({
                    "start": round(float(cur_start), 3),
                    "end": round(float(cur_end), 3),
                    "speaker": spk,
                    "text": ""
                })
                cur_tokens = []
        if cur_tokens:
            sentences.append({
                "start": round(float(cur_start), 3),
                "end": round(float(cur_end), 3),
                "speaker": spk,
                "text": ""
            })
    sentences.sort(key=lambda s: (s["start"], s["end"]))
    if dbg: dbg.snap("DIALOG_SENTENCES_DONE", sentences=len(sentences))
    return sentences

# ------------------------- Sonuç yapıları -------------------------

@dataclass
class ProcessResult:
    success: bool
    engine: str
    language: str
    video_path: str
    audio_path: Optional[str]
    output_dir: str
    words_jsonl: str
    srt_path: Optional[str]
    json_path: Optional[str]
    words_count: int
    dialog_path: Optional[str] = None
    notes: Optional[str] = None

# ------------------------- Ana API -------------------------

def process_video_wordwise(
        video_path: str,
        output_dir: str,
        speaker_count: int = 2,
        keep_audio: bool = True,
        language_code: str = "en-US",
        overlap_strategy: str = "smart",   # 'smart' | 'top' | 'duplicate' | 'mark'
        overlap_alpha: float = 1.25,
        overlap_min_dup: float = 0.06,
        switch_penalty: float = 0.35,
        center_lambda: float = 0.20,
        write_word_srt: bool = False,
        semantic_mode: str = "none",       # 'none' | 'heuristic' | 'llm'
        llm_model: Optional[str] = None,
        # >>>> EKLENEN PARAMETRELER <<<<
        sentence_max_pause: float = 0.7,
        sentence_max_tokens: int = 50,
        write_dialog_json: bool = True,
) -> ProcessResult:

    _setup_logging()
    video_p = Path(video_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    _ensure_dir(out_dir)
    _attach_file_logger(out_dir / "process.debug.log")
    dbg = DebugWriter(out_dir)

    if not video_p.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_p}")

    logging.info("========== SES ÇIKARMA ==========")
    wav_path = out_dir / f"{video_p.stem}_16k.wav"
    _extract_audio_ffmpeg(video_p, wav_path, sr=16000, dbg=dbg)

    logging.info("========== ASR (WHISPER CUDA) ==========")
    words_raw = _transcribe_whisper(wav_path, language_code, dbg)

    logging.info("========== DIARIZATION (PYANNOTE CUDA) ==========")
    diar_segments_raw = _diarize_pyannote(wav_path, speaker_count, dbg)
    diar_segments, spk_map = _canonicalize_speakers(diar_segments_raw, dbg)
    speakers = _speaker_set(diar_segments)
    dbg.snap("SPEAKER_SET", speakers=speakers, count=len(speakers))

    logging.info("========== OSD (OVERLAPPED SPEECH DETECTION) ==========")
    osd_intervals = _osd_pyannote(wav_path, dbg)
    _flag_osd_for_words(words_raw, osd_intervals)
    osd_cover_words = sum(1 for w in words_raw if w.get("osd"))
    dbg.snap("OSD_FLAGGED_WORDS", osd_words=osd_cover_words, total_words=len(words_raw))

    logging.info("========== KELİME-BAZLI VITERBI ATAMA ==========")
    tables = _build_overlap_tables(words_raw, diar_segments, speakers, dbg)
    vp = ViterbiParams(switch_penalty=switch_penalty, center_lambda=center_lambda)
    assigned = _assign_speakers_viterbi(words_raw, tables, speakers, vp, dbg)

    logging.info("========== OVERLAP POLİTİKASI UYGULAMA ==========")
    policy = OverlapPolicy(strategy=overlap_strategy, alpha=overlap_alpha, min_dup_dur=overlap_min_dup)
    words_assigned = _post_overlap_policy(words_raw, tables, assigned, policy, dbg)

    logging.info("========== SEMANTİK DÜZELTME ==========")
    words_final = _semantic_repair(words_assigned, window=40, mode=semantic_mode, dbg=dbg, llm_model=llm_model)

    # Yaz: JSONL (kelime-kelime)
    jsonl_path = out_dir / "words_diarized.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for w in words_final:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    dbg.snap("WORDS_JSONL_WRITTEN", path=str(jsonl_path), mb=round(jsonl_path.stat().st_size/(1024**2), 3), words=len(words_final))

    # Opsiyonel: Word-level SRT
    srt_path = None
    if write_word_srt:
        srt_path = out_dir / "transcript_words.srt"
        _write_word_srt(words_final, srt_path)
        dbg.snap("WORD_SRT_WRITTEN", path=str(srt_path), mb=round(srt_path.stat().st_size/(1024**2), 3))

    # === CÜMLE-BAZLI DİYALOG ===
    dialog_path = None
    if write_dialog_json:
        logging.info("========== DİYALOG (CÜMLE) OLUŞTURMA ==========")
        dialog = _words_to_dialog_sentences(
            words_final,
            sentence_max_pause=sentence_max_pause,
            sentence_max_tokens=sentence_max_tokens,
            dbg=dbg,
        )
        dialog_path = out_dir / "dialog_sentences.json"
        with dialog_path.open("w", encoding="utf-8") as f:
            json.dump(dialog, f, ensure_ascii=False, indent=2)
        dbg.snap("DIALOG_JSON_WRITTEN", path=str(dialog_path), sentences=len(dialog), mb=round(dialog_path.stat().st_size/(1024**2), 3))

    # Özet JSON
    meta = {
        "video": str(video_p),
        "audio": str(wav_path),
        "engine": "whisper-large-v3 + pyannote (diar+osd) + viterbi",
        "language_code": language_code,
        "speakers": speakers,
        "speaker_map": spk_map,
        "params": {
            "overlap_strategy": overlap_strategy,
            "overlap_alpha": overlap_alpha,
            "overlap_min_dup": overlap_min_dup,
            "switch_penalty": switch_penalty,
            "center_lambda": center_lambda,
            "semantic_mode": semantic_mode,
            "llm_model": llm_model,
            "sentence_max_pause": sentence_max_pause,
            "sentence_max_tokens": sentence_max_tokens,
            "write_dialog_json": write_dialog_json,
        },
        "osd_intervals": osd_intervals[:200],
        "words_count": len(words_final),
    }
    meta_path = out_dir / "transcript_diarized.json"
    _save_json(meta, meta_path)
    dbg.snap("META_WRITTEN", path=str(meta_path), mb=round(meta_path.stat().st_size/(1024**2), 3))

    # Temizlik
    audio_path_out = str(wav_path)
    if not keep_audio and wav_path.exists():
        try:
            wav_path.unlink(); audio_path_out = None
        except Exception:
            pass

    return ProcessResult(
        success=True,
        engine="whisper-large-v3+pyannote(osd)+viterbi",
        language=language_code,
        video_path=str(video_p),
        audio_path=audio_path_out,
        output_dir=str(out_dir),
        words_jsonl=str(jsonl_path),
        srt_path=str(srt_path) if srt_path else None,
        json_path=str(meta_path),
        words_count=len(words_final),
        dialog_path=(str(dialog_path) if dialog_path else None),
        notes=None,
    )

# ------------------------- CLI -------------------------

if __name__ == "__main__":
    res = process_video_wordwise(
        video_path="sample.mp4",
        output_dir="output",
        speaker_count=3,
        keep_audio=True,
        language_code="en-US",
        overlap_strategy="top",
        overlap_alpha=1.30,
        overlap_min_dup=0.06,
        switch_penalty=0.35,
        center_lambda=0.20,
        write_word_srt=True,          # Kelime başına SRT büyük olabilir
        semantic_mode="llm",          # 'none' | 'heuristic' | 'llm'
        llm_model="gpt-4o",
        # >>> cümle-bazlı çıktı <<<
        sentence_max_pause=0.70,
        sentence_max_tokens=50,
        write_dialog_json=True,
    )
    print(json.dumps(res.__dict__, ensure_ascii=False, indent=2))
