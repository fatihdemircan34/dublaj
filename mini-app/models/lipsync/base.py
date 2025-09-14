from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseLipSyncModel(ABC):
    """Abstract base class for lip sync models."""

    @abstractmethod
    def sync(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        """Perform lip synchronization."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model can be used in the current environment."""
        ...

    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Return a list of external dependencies required by the model."""
        ...
