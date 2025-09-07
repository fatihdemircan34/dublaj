import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List

def ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-version"], check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError("ffmpeg PATH'te bulunamadı") from e

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            rows.append(json.loads(l))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
