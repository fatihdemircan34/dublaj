from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

from core.pipeline.base import Context
from core.registry.lipsync import get_lipsync_model

logger = logging.getLogger(__name__)


class LipSyncStep:
    """Synchronize audio with the original video and create final video."""

    name = "LipSync"

    def __init__(
        self, model_name: str = "simple", model_kwargs: Optional[Dict[str, object]] = None
    ) -> None:
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    def run(self, ctx: Context) -> None:
        artifacts = ctx["artifacts"]
        video = (
            artifacts.get("processed_video")
            or artifacts.get("video")
            or artifacts.get("input_video")
        )
        audio = artifacts.get("final_audio") or artifacts.get("synth_audio")
        if not video or not audio:
            return
        out_path = Path(ctx["temp_dir"]) / "final_video.mp4"

        model = get_lipsync_model(self.model_name, **self.model_kwargs)
        used = self.model_name
        start = time.time()
        try:
            if hasattr(model, "is_available") and not model.is_available():
                raise RuntimeError("model unavailable")
            result = model.sync(Path(video), Path(audio), out_path)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "LipSync model %s failed (%s); falling back to simple mux",
                self.model_name,
                exc,
            )
            fallback = get_lipsync_model("simple")
            result = fallback.sync(Path(video), Path(audio), out_path)
            used = "simple"
        duration = time.time() - start
        artifacts["final_video"] = str(result)
        logger.info("LipSync: model=%s time=%.2fs -> %s", used, duration, result)
