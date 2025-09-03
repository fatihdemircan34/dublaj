from __future__ import annotations
import os
from typing import Optional, List
import whisperx
import torch
from dubbing_ai.interfaces import TranscriberPlugin
from dubbing_ai.artifact_types import ASRSegment, Word, write_jsonl

class WhisperXTranscriber(TranscriberPlugin):
    name = "whisperx"

    def run(self, audio_path: str, out_dir: str, lang_hint: Optional[str] = None, model_size: str = "medium") -> str:
        os.makedirs(out_dir, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model = whisperx.load_model(model_size, device=device, compute_type=compute_type, language=lang_hint)
        result = model.transcribe(audio_path, batch_size=16, language=lang_hint)

        # Alignment
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device, return_char_alignments=False)

        # Convert to artefact
        segs: List[ASRSegment] = []
        for i, seg in enumerate(result_aligned["segments"]):
            words = []
            for w in seg.get("words", []) or []:
                if w.get("start") is None or w.get("end") is None:
                    continue
                words.append(Word(w=w["word"], s=float(w["start"]), e=float(w["end"]), p=float(w.get("probability") or 0.0)))
            segs.append(ASRSegment(
                id=f"seg_{i:04d}",
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg.get("text","").strip(),
                lang=result.get("language"),
                words=words
            ))

        out_path = os.path.join(out_dir, "asr_segments.jsonl")
        write_jsonl(out_path, segs)
        return out_path
