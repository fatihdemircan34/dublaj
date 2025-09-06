from __future__ import annotations

import time
from typing import List

from .base import Context, Step
from .timeline import StepTimeline


class StepRunner:
    """Sequentially execute steps and record their timeline."""

    def __init__(self, steps: List[Step]):
        self.steps = steps
        self.timeline = StepTimeline()

    def run(self, ctx: Context) -> None:
        for step in self.steps:
            t0 = time.time()
            status = "ok"
            try:
                step.run(ctx)
            except Exception as exc:  # pragma: no cover - re-raise for visibility
                status = f"err:{exc}"
                raise
            finally:
                t1 = time.time()
                self.timeline.add(step.name, t0, t1, status)
        ctx.setdefault("artifacts", {})["timeline"] = self.timeline.to_list()
