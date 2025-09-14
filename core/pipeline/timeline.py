from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class StepEvent:
    name: str
    t0: float
    t1: float
    status: str


@dataclass
class StepTimeline:
    events: List[StepEvent] = field(default_factory=list)

    def add(self, name: str, t0: float, t1: float, status: str) -> None:
        self.events.append(StepEvent(name, t0, t1, status))

    def to_list(self) -> List[Dict[str, float | str]]:
        return [event.__dict__ for event in self.events]
