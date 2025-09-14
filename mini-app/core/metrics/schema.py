from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List

from pydantic import BaseModel, Field, validator


class TechMetrics(BaseModel):
    bitrate: int = Field(..., ge=0)
    sample_rate: int = Field(..., ge=0)


class SyncMetrics(BaseModel):
    lip_sync_score: int = Field(ge=0, le=100)
    lse_c: float
    lse_d: float


class ProsodyMetrics(BaseModel):
    pitch: float
    energy: float


class IntelligibilityMetrics(BaseModel):
    word_error_rate: float = Field(..., ge=0.0, le=1.0)
    pronunciation_score: int = Field(..., ge=0, le=100)


class SpeakerMetrics(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    gender_match: bool


class ClarityMetrics(BaseModel):
    noise: float = Field(..., ge=0.0)
    clipping: float = Field(..., ge=0.0)


class SegmentMetrics(BaseModel):
    segment_id: str
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    tech: TechMetrics
    sync: SyncMetrics
    prosody: ProsodyMetrics
    intelligibility: IntelligibilityMetrics
    speaker: SpeakerMetrics
    clarity: ClarityMetrics

    @validator("segment_id")
    def _validate_segment_id(cls, v: str) -> str:
        if not re.fullmatch(r"seg_[0-9]+", v):
            raise ValueError("segment_id must match pattern 'seg_<digits>'")
        return v

    @validator("end")
    def _validate_end(cls, v: float, values: dict) -> float:
        start = values.get("start")
        if start is not None and v < start:
            raise ValueError("end must be greater than or equal to start")
        return v


class QualityReport(BaseModel):
    video_id: str
    profile: str = "web_speech"
    generated_at_utc: datetime
    segments: List[SegmentMetrics]

    @validator("generated_at_utc")
    def _validate_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() != timedelta(0):
            raise ValueError("generated_at_utc must be timezone-aware and in UTC")
        return v
