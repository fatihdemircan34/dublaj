from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Dict
import importlib
import os

import torch

from core.registry.asr import register_asr

import json

@register_asr("whisper")
class WhisperASR:
    """ASR model wrapper around OpenAI's Whisper."""

    def __init__(self, model: str = "base", device: str | None = None, **load_kw: Any) -> None:
        self.model_name = model
        self.load_kw = load_kw
        self.device = device or os.getenv("WHISPER_DEVICE")
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            whisper = importlib.import_module("whisper")
            self._model = whisper.load_model(
                self.model_name, device=self.device, **self.load_kw
            )
        return self._model

    def transcribe(
        self, wav_path: str, word_timestamps: bool = False
    ) -> List[Dict[str, Any]]:
        """Transcribe audio and return list of segments.

        Parameters
        ----------
        wav_path:
            Path to the audio file to transcribe.
        word_timestamps:
            If ``True``, include word-level timestamps and average log probability
            for each returned segment. Defaults to ``False`` for backwards
            compatibility.
        """

        model = self._ensure_model()
        result = model.transcribe(wav_path, word_timestamps=word_timestamps)
        segments: List[Dict[str, Any]] = []
        out = Path("out"); out.mkdir(parents=True, exist_ok=True)
        debug_json = out / f"whisper.debug.json"
        debug_json.write_text(json.dumps(result, ensure_ascii=False))
        for seg in result.get("segments", []):
            segment: Dict[str, Any] = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
            }

            if word_timestamps:
                segment["avg_logprob"] = float(seg.get("avg_logprob", 0.0))
                segment["words"] = [
                    {
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "text": w["word"].strip(),
                    }
                    for w in seg.get("words", [])
                ]

            segments.append(segment)

        return segments
