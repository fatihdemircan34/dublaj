from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class DiarizeHints:
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    domain: Optional[str] = None  # "podcast" | "meeting" | "telephony"

@dataclass
class DiarizeResult:
    diar_jsonl_path: str
    profiles_json_path: Optional[str] = None

class DiarizerPlugin:
    name: str = "base"
    def run(self, audio_path: str, out_dir: str, hints: Optional[DiarizeHints] = None) -> DiarizeResult:
        raise NotImplementedError()

class TranscriberPlugin:
    name: str = "base"
    def run(self, audio_path: str, out_dir: str, lang_hint: Optional[str] = None, model_size: str = "medium") -> str:
        """Returns path to asr_segments.jsonl"""
        raise NotImplementedError()
