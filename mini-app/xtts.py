from pathlib import Path
from typing import Dict
import inspect
import logging

# --- XTTS safe load shim for PyTorch >= 2.6 ---
import torch
import torchaudio

try:  # pragma: no cover - depends on torch version
    from torch.serialization import add_safe_globals
    try:  # pragma: no cover - optional dependency
        from TTS.tts.models.xtts import XttsAudioConfig
        add_safe_globals([XttsAudioConfig])
    except Exception:
        pass
except Exception:
    pass

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):  # pragma: no cover - compatibility shim
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat
# --- /shim ---

from core.registry.tts import register_tts
from models.tts.gtts import BaseTTSEngine

logger = logging.getLogger(__name__)


@register_tts("xtts")
class XTTSEngine(BaseTTSEngine):
    """TTS engine using Coqui's XTTS v2 model."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "tr",
    ) -> None:
        self.model_name = model_name
        self.language = language
        try:
            from TTS.api import TTS  # type: ignore
            from TTS.tts.configs.xtts_config import XttsConfig

            try:
                add_safe_globals([XttsConfig])  # type: ignore[name-defined]
            except Exception:
                pass
            self._tts = TTS(model_name)
            try:
                self._tts.to("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            try:
                p = next(self._tts.synthesizer.tts_model.parameters())
                logger.info("XTTS device=%s dtype=%s", p.device, p.dtype)
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.error("Failed to load XTTS model '%s': %s", model_name, exc)
            raise RuntimeError(
                f"XTTS model '{model_name}' could not be loaded. "
                "Ensure the model path is correct and that the TTS library "
                "is up to date (`pip install -U TTS`).",
            ) from exc

    def synthesize(
        self,
        text: str,
        output_path: Path,
        speaker_wav: str | None = None,
        latents_path: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
    ) -> Path:
        """Synthesize `text` into `output_path` and return the path."""
        assert not (speaker_wav and latents_path), "speaker_wav and latents_path are mutually exclusive"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lang = lang or self.language
        if lang:
            lang = lang.lower()
            for sep in ("-", "_"):
                lang = lang.split(sep)[0]
            if lang.startswith("tr"):
                lang = "tr"
            elif len(lang) > 2:
                lang = lang[:2]

        if latents_path:
            try:
                lat = torch.load(latents_path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - runtime errors
                logger.warning("XTTS: failed to load latents %s: %s", latents_path, exc)
                raise RuntimeError("XTTS latents load failed") from exc

            logger.info(
                "XTTS: inference(language=%s, cached_latents, out=%s, text_chars=%d)",
                lang,
                output_path,
                len(text or ""),
            )
            try:
                wav, sr = self._synth_with_latents(text, lang, lat)
                torchaudio.save(str(output_path), wav, sr)
            except Exception as exc:  # pragma: no cover - runtime errors
                logger.error("XTTS synthesis failed: %s", exc)
                raise RuntimeError("XTTS synthesis failed") from exc
            logger.info("XTTS: wrote %s (sr=%d)", output_path, sr)
            return output_path

        if not speaker_wav or not Path(speaker_wav).is_file():
            raise FileNotFoundError(
                f"Speaker WAV not found: {speaker_wav}. Pass a valid reference file.",
            )

        abs_spk = str(Path(speaker_wav).resolve())
        split_threshold = 120
        split_sentences = False if (text and len(text) < split_threshold) else True

        tts_kwargs = {
            "text": text,
            "language": lang,
            "file_path": str(output_path),
            "speaker_wav": abs_spk,
            "split_sentences": split_sentences,
        }
        if speed is not None:
            tts_kwargs["speed"] = speed

        try:  # drop split_sentences if backend does not support it
            if "split_sentences" not in inspect.signature(self._tts.tts_to_file).parameters:
                tts_kwargs.pop("split_sentences", None)
        except Exception:
            pass

        logger.info(
            "XTTS: tts_to_file(language=%s, speaker_wav=%s, out=%s, text_chars=%d)",
            lang,
            abs_spk,
            output_path,
            len(text or ""),
        )
        try:
            self._tts.tts_to_file(**tts_kwargs)
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.error("XTTS synthesis failed: %s", exc)
            raise RuntimeError("XTTS synthesis failed") from exc
        logger.info("XTTS: wrote %s (sr=24000)", output_path)
        return output_path

    def _synth_with_latents(self, text: str, lang: str, lat: Dict[str, torch.Tensor]):
        mdl = self._tts.synthesizer.tts_model
        wav = mdl.inference(
            text=text,
            language=lang,
            gpt_cond_latent=lat["gpt"],
            diffusion_conditioning=lat["diff"],
            speaker_embedding=lat["spk"],
        )
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav.to(torch.float32), 24000
