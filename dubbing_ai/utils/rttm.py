from __future__ import annotations

# Simple RTTM parser (SPEAKER entries)
# Format: SPEAKER <uri> <chan> <start> <dur> <ortho> <stype> <name> <conf> <slat>
from typing import List, Dict
from dubbing_ai.artifact_types import DiarSegment

def parse_rttm(path: str) -> List[DiarSegment]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            # fields
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7] if len(parts) > 7 else "UNK"
            conf = float(parts[8]) if len(parts) > 8 else None
            out.append(DiarSegment(spk=spk, start=start, end=start+dur, conf=conf, overlap=False))
    return out
