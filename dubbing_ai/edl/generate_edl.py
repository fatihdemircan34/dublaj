from __future__ import annotations
import json
from typing import List, Dict

def write_edl_jsonl(path: str, edl: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in edl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
