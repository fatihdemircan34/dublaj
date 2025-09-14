from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - gracefully handle absence
    torch = None  # type: ignore

from .base import BaseLipSyncModel
from core.registry.lipsync import register_lipsync

logger = logging.getLogger(__name__)


@register_lipsync("wav2lip")
class Wav2LipModel(BaseLipSyncModel):
    """Wav2Lip deep learning lip sync model."""

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[Path] = None,
        face_detect_path: Optional[Path] = None,
        pad_top: int = 0,
        pad_right: int = 0,
        pad_bottom: int = 10,
        pad_left: int = 0,
        resize_factor: int = 1,
        fps: Optional[int] = None,
        nosmooth: bool = False,
    ) -> None:
        self.device = device
        self.model_path = model_path or Path("./models/wav2lip/wav2lip_gan.pth")
        self.face_detect_path = face_detect_path or Path("./models/face_detection/s3fd.pth")
        self.pad_top = pad_top
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.resize_factor = resize_factor
        self.fps = fps
        self.nosmooth = nosmooth

    def is_available(self) -> bool:
        if torch is None:
            logger.debug("PyTorch not installed; Wav2Lip unavailable")
            return False
        requirements_met = (
            self.model_path.exists()
            and self.face_detect_path.exists()
            and Path("./Wav2Lip").exists()
        )
        if not requirements_met:
            logger.debug("Wav2Lip requirements not met")
            return False
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.debug("CUDA not available for Wav2Lip")
            return False
        return True

    def sync(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.is_available():
            raise RuntimeError("Wav2Lip model unavailable")
        return self._run_wav2lip_inference(video_path, audio_path, output_path)

    def _run_wav2lip_inference(
        self, video_path: Path, audio_path: Path, output_path: Path
    ) -> Path:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent  # go up to repo root
        wav2lip_script = project_root / "Wav2Lip" / "inference.py"
        logger.debug("Looking for Wav2Lip at: %s", wav2lip_script)
        if not wav2lip_script.exists():
            alternative_paths = [
                project_root / "Wav2Lip" / "inference.py",
                Path("./Wav2Lip/inference.py"),
                Path("../Wav2Lip/inference.py"),
                Path("../../Wav2Lip/inference.py"),
                Path("Wav2Lip/inference.py"),
            ]
            for alt in alternative_paths:
                logger.debug("Trying: %s", alt.resolve())
                if alt.exists():
                    wav2lip_script = alt
                    break
            else:
                raise RuntimeError("Wav2Lip inference.py not found")
        video_abs = Path(video_path).resolve()
        audio_abs = Path(audio_path).resolve()
        output_abs = Path(output_path).resolve()
        model_abs = Path(self.model_path).resolve()
        if not video_abs.exists():
            raise RuntimeError(f"Video file not found: {video_abs}")
        if not audio_abs.exists():
            raise RuntimeError(f"Audio file not found: {audio_abs}")
        if not model_abs.exists():
            raise RuntimeError(f"Model file not found: {model_abs}")
        cmd = [
            "python",
            str(wav2lip_script),
            "--checkpoint_path",
            str(model_abs),
            "--face",
            str(video_abs),
            "--audio",
            str(audio_abs),
            "--outfile",
            str(output_abs),
            "--pads",
            str(self.pad_top),
            str(self.pad_right),
            str(self.pad_bottom),
            str(self.pad_left),
            "--resize_factor",
            str(self.resize_factor),
        ]
        if self.fps is not None:
            cmd.extend(["--fps", str(self.fps)])
        if self.nosmooth:
            cmd.append("--nosmooth")
        if self.device == "cpu":
            cmd.extend(["--device", "cpu"])
        logger.info("Running Wav2Lip inference: %s", " ".join(cmd))
        wav2lip_dir = wav2lip_script.parent
        log_path = output_abs.parent / "wav2lip.log"
        with open(log_path, "w", encoding="utf-8") as logf:
            result = subprocess.run(
                cmd,
                cwd=str(wav2lip_dir),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Wav2Lip inference failed (see {log_path})")
        if not output_abs.exists():
            raise RuntimeError(f"Wav2Lip output file not created: {output_abs}")
        return output_abs

    def get_requirements(self) -> List[str]:
        return [
            "torch",
            "torchvision",
            "opencv-python",
            "numpy",
            "face-alignment",
            "Wav2Lip repository",
            "wav2lip_gan.pth model",
            "s3fd.pth face detection model",
        ]
