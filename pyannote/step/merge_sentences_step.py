import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from .utils import write_jsonl
logger = logging.getLogger("app-pipeline")

class MergeSentencesStep:
    """Kelime düzeyi -> cümle bazlı JSONL (out.jsonl)."""
    name = "MergeSentences"

    def __init__(self, gap_th: float = 1.0, out_name: str = "out.jsonl") -> None:
        self.gap_th = float(gap_th)
        self.out_name = out_name

    def run(self, ctx: Dict[str, Any]) -> None:
        words: List[Dict[str, Any]] = ctx["artifacts"].get("words") or []
        if not words:
            raise RuntimeError("MergeSentencesStep: 'words' yok. Önce AppFullCoreStep çalışmalı.")

        sentences: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None

        for w in words:
            spk = w.get("speaker")
            start = float(w["start"]); end = float(w["end"])
            token = (w.get("word") or "").strip()
            if not token:
                continue

            if cur is None:
                cur = {"text": token, "start": start, "end": end, "speaker": spk}
                continue

            if spk == cur["speaker"] and (start - float(cur["end"])) < self.gap_th:
                cur["text"] = (cur["text"] + " " + token).strip()
                cur["end"] = end
            else:
                if cur.get("text"):
                    sentences.append(cur)
                cur = {"text": token, "start": start, "end": end, "speaker": spk}

        if cur is not None and cur.get("text"):
            sentences.append(cur)

        temp = Path(ctx["temp_dir"])
        out_path = temp / self.out_name
        write_jsonl(out_path, sentences)
        logger.info("Cümle birleştirme: %d cümle -> %s", len(sentences), out_path)

        ctx["artifacts"]["sentences"] = sentences
        ctx["artifacts"]["out_jsonl"] = str(out_path)


# -------------------- 9 sn referans ses -------------------- #
