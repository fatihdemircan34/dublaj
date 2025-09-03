from __future__ import annotations
from typing import Optional
from dubbing_ai.interfaces import DiarizeHints
from dubbing_ai.diarizer import PyannoteDiarizer, NemoMSDDDiarizer, GCPSTTDiarizer

def select_diarizer(name: Optional[str], hints: Optional[DiarizeHints] = None):
    if name:
        if name == "pyannote":
            return PyannoteDiarizer()
        if name == "nemo-msdd":
            return NemoMSDDDiarizer()
        if name == "gcp-stt":
            return GCPSTTDiarizer()
        raise ValueError(f"Unknown diarizer: {name}")
    # heuristic
    if hints and hints.domain in {"telephony", "meeting"}:
        return NemoMSDDDiarizer()
    if hints and hints.domain == "streaming":
        return GCPSTTDiarizer()
    return PyannoteDiarizer()
