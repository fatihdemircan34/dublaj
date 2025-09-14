from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict


class Context(TypedDict, total=False):
    """Shared data for all pipeline steps."""

    temp_dir: str
    config: Dict[str, Any]
    artifacts: Dict[str, Any]


class Result(TypedDict, total=False):
    pass


class Step(Protocol):
    """Pipeline step interface."""

    name: str

    def run(self, ctx: Context) -> None:
        """Run step and mutate context with produced artifacts."""
        ...
