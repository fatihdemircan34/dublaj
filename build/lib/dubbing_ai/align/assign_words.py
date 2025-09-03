from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dubbing_ai.artifact_types import ASRSegment, DiarSegment, Word, read_jsonl, write_jsonl
import math

def iou(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    s1, e1 = a; s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0

def assign_words(asr_segments: List[ASRSegment], diar_segments: List[DiarSegment],
                 overlap_second_best_threshold: float = 0.25) -> List[ASRSegment]:
    # Assign each word to best-overlapping diar segment (speaker)
    diar_spans = [(d.spk, d.start, d.end) for d in diar_segments]

    for seg in asr_segments:
        for w in seg.words:
            span = (w.s, w.e)
            scores = [(spk, iou(span, (s,e))) for (spk, s, e) in diar_spans]
            scores.sort(key=lambda x: x[1], reverse=True)
            if not scores or scores[0][1] == 0.0:
                # leave unassigned; downstream can handle
                setattr(w, "spk", None)
                continue
            best_spk, best_score = scores[0]
            setattr(w, "spk", best_spk)
            # second-best co-tagging (optional)
            if len(scores) > 1 and scores[1][1] >= overlap_second_best_threshold:
                setattr(w, "spk2", scores[1][0])
    return asr_segments

def group_words_to_edl(asr_segments: List[ASRSegment]) -> List[Dict]:
    edl = []
    cur_spk: Optional[str] = None
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_words: List[str] = []
    for seg in asr_segments:
        for w in seg.words:
            spk = getattr(w, "spk", None)
            if spk is None:
                # flush current if any
                if cur_spk is not None:
                    edl.append({"spk": cur_spk, "start": cur_start, "end": cur_end, "text": " ".join(cur_words).strip()})
                    cur_spk, cur_start, cur_end, cur_words = None, None, None, []
                continue
            if cur_spk is None:
                cur_spk, cur_start, cur_end, cur_words = spk, w.s, w.e, [w.w]
            elif spk == cur_spk and w.s <= (cur_end or w.s) + 0.3:
                cur_end = max(cur_end, w.e) if cur_end is not None else w.e
                cur_words.append(w.w)
            else:
                edl.append({"spk": cur_spk, "start": cur_start, "end": cur_end, "text": " ".join(cur_words).strip()})
                cur_spk, cur_start, cur_end, cur_words = spk, w.s, w.e, [w.w]
    if cur_spk is not None:
        edl.append({"spk": cur_spk, "start": cur_start, "end": cur_end, "text": " ".join(cur_words).strip()})
    return edl

def main_assign(asr_jsonl: str, diar_jsonl: str, out_words_jsonl: str, out_edl_jsonl: str):
    asr = read_jsonl(asr_jsonl, ASRSegment)
    diar = read_jsonl(diar_jsonl, DiarSegment)
    asr2 = assign_words(asr, diar)
    write_jsonl(out_words_jsonl, asr2)
    import json
    edl = group_words_to_edl(asr2)
    with open(out_edl_jsonl, "w", encoding="utf-8") as f:
        for item in edl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
