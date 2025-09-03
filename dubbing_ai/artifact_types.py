from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Iterable, Dict, Any
import json

class Word(BaseModel):
    w: str = Field(..., description="Word text")
    s: float = Field(..., description="Word start time (sec)")
    e: float = Field(..., description="Word end time (sec)")
    p: Optional[float] = Field(default=None, description="Word confidence")

class ASRSegment(BaseModel):
    id: str
    start: float
    end: float
    text: str
    lang: Optional[str] = None
    words: List[Word] = Field(default_factory=list)

class DiarSegment(BaseModel):
    spk: str
    start: float
    end: float
    conf: Optional[float] = None
    overlap: bool = False

def write_jsonl(path: str, items: Iterable[BaseModel]):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")

def read_jsonl(path: str, model):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(model.model_validate_json(line))
    return out

def write_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
