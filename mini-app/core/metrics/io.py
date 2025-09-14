from __future__ import annotations

import json
from pathlib import Path

from .schema import QualityReport


def _model_dump_json(model: QualityReport) -> str:
    """Dump model to JSON string compatible with Pydantic v1 and v2."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=2)
    return model.json(indent=2)


def save_quality_report(report: QualityReport, filepath: str) -> None:
    """Serialize ``report`` to ``filepath`` as JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(_model_dump_json(report))


def load_quality_report(filepath: str) -> QualityReport:
    """Load a ``QualityReport`` from JSON file at ``filepath``."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return QualityReport(**data)
