from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from core.pipeline.base import Context

try:  # pragma: no cover - optional dependency
    from pydub import AudioSegment
except Exception:  # pragma: no cover - missing dependency
    AudioSegment = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..models.tts.xtts import XTTSEngine
except Exception:  # pragma: no cover - optional dependency
    XTTSEngine = None  # type: ignore

logger = logging.getLogger(__name__)


class BuildRefVoicesStep:
    """Create reference speaker WAVs from original audio."""

    name = "BuildRefVoices"

    def __init__(self, seconds: int = 9, min_chunk_ms: int = 1000, margin_ms: int = 150) -> None:
        self.target_ms = int(seconds * 1000)
        self.min_chunk_ms = min_chunk_ms
        self.margin_ms = margin_ms

    def run(self, ctx: Context) -> None:  # pragma: no cover - heavy audio
        if not AudioSegment:
            logger.error("pydub is required for BuildRefVoicesStep")
            return

        original = ctx["artifacts"].get("original_audio")
        segments = ctx["artifacts"].get("segments", [])
        if not original or not segments:
            logger.warning("Missing original audio or segments for BuildRefVoicesStep")
            return

        audio = AudioSegment.from_file(original)
        voices_dir = Path(ctx["temp_dir"]) / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)

        latents_map: Dict[str, str] = {}
        mdl = None
        if XTTSEngine:
            try:
                engine = XTTSEngine(language=ctx.get("target_lang", "tr"))
                mdl = engine._tts.synthesizer.tts_model
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("XTTS: failed to init model for latents: %s", exc)
                mdl = None

        groups: Dict[str, List[dict]] = defaultdict(list)
        for seg in segments:
            speaker = str(seg.get("speaker") or seg.get("speaker_id") or "UNKNOWN")
            groups[speaker].append(seg)

        for speaker, segs in groups.items():
            candidates = sorted(
                segs,
                key=lambda s: (s.get("end", 0) - s.get("start", 0)),
                reverse=True,
            )
            collected: List[AudioSegment] = []
            total_ms = 0
            for seg in candidates:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                chunk_ms = int((end - start) * 1000)
                if chunk_ms < self.min_chunk_ms:
                    continue
                begin = max(int(start * 1000) + self.margin_ms, 0)
                finish = min(int(end * 1000) - self.margin_ms, len(audio))
                if finish <= begin:
                    continue
                chunk = audio[begin:finish]
                if chunk.dBFS < -45:
                    continue
                collected.append(chunk)
                total_ms += len(chunk)
                if total_ms >= self.target_ms:
                    break
            if not collected:
                logger.warning("No suitable audio found for speaker %s", speaker)
                continue
            voice = collected[0]
            for c in collected[1:]:
                voice += c
            voice = voice[: self.target_ms]
            if voice.dBFS != float("-inf"):
                voice = voice.apply_gain(-20.0 - voice.dBFS)
            voice = voice.set_channels(1).set_frame_rate(16000)
            out_path = voices_dir / f"{speaker}.wav"
            voice.export(out_path, format="wav")
            abs_path = str(out_path.resolve())

            if mdl:
                try:
                    try:
                        latents = mdl.get_conditioning_latents(audio_path=[abs_path])
                    except TypeError:
                        latents = mdl.get_conditioning_latents(abs_path)

                    if isinstance(latents, tuple):
                        if len(latents) == 3:
                            gpt, diff, spk = latents
                            payload = {
                                "gpt": gpt.detach().cpu(),
                                "diff": diff.detach().cpu(),
                                "spk": spk.detach().cpu(),
                            }
                        elif len(latents) == 2:
                            gpt, spk = latents
                            payload = {
                                "gpt": gpt.detach().cpu(),
                                "spk": spk.detach().cpu(),
                            }
                        else:
                            raise ValueError(
                                f"Unexpected number of conditioning latents: {len(latents)}"
                            )
                    else:
                        raise ValueError(
                            "Unexpected return type from get_conditioning_latents"
                        )

                    lat_path = voices_dir / f"{speaker}.latents.pt"
                    torch.save(payload, lat_path)
                    latents_map[speaker] = str(lat_path.resolve())
                    logger.info("XTTS: saved latents for %s -> %s", speaker, lat_path)
                except Exception as exc:  # pragma: no cover - runtime errors
                    logger.warning(
                        "XTTS: failed to compute latents for %s: %s", speaker, exc
                    )

            genders = [seg.get("gender") for seg in segs if seg.get("gender") in ("M", "F")]
            majority: Optional[str] = None
            if genders:
                majority = Counter(genders).most_common(1)[0][0]

            for seg in segs:
                seg["voice_id"] = abs_path
                if majority:
                    seg["gender"] = majority

        ctx["artifacts"]["ref_voices_dir"] = str(voices_dir)
        ctx["artifacts"]["segments"] = segments
        ctx["artifacts"]["voice_latents"] = latents_map
