from __future__ import annotations

import logging
import re
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Tuple

from core.pipeline.base import Context
from core.registry.tts import TTS_REGISTRY
from ..utils.voice import VoiceSelector

try:  # pragma: no cover - required dependency
    from pydub import AudioSegment
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None


logger = logging.getLogger(__name__)


class TTSStep:
    """Synthesize translated text into speech."""

    def __init__(
        self,
        tts: str,
        tts_kw: dict[str, Any] | None = None,
        voice_id: str | None = None,
        speed: float | None = None,
        voice_map: dict[str, str] | None = None,
        crossfade_ms: int = 10,
    ) -> None:
        # Store configuration and expose the selected TTS engine in the step name
        # so that pipeline timelines can indicate which synthesizer was used.
        self.cfg = SimpleNamespace(
            tts=tts,
            tts_kw=tts_kw or {},
            voice_id=voice_id,
            speed=speed,
            voice_map=voice_map,
            crossfade_ms=crossfade_ms,
        )
        self.name = f"TTS({tts})"

    def run(self, ctx: Context) -> None:
        segments = ctx["artifacts"].get("segments", [])
        logger.debug("TTS step started with %d segments", len(segments))
        if not segments:
            logger.warning("No segments available for TTS; skipping step")
            return

        cfg = self.cfg
        model = TTS_REGISTRY.create(cfg.tts, **cfg.tts_kw)
        mapping = dict(cfg.voice_map) if cfg.voice_map else {}
        if cfg.voice_id:
            mapping["U"] = cfg.voice_id
        selector = VoiceSelector(
            tts_engine=cfg.tts,
            custom_mapping=mapping or None,
        )

        voice_latents = ctx["artifacts"].get("voice_latents", {})

        out_dir = Path(ctx["temp_dir"]) / "tts_segments"
        out_dir.mkdir(parents=True, exist_ok=True)

        combined = AudioSegment.silent(duration=0) if AudioSegment else None
        temp_segments: list[Path] = []
        position = 0
        ext = ".wav" if cfg.tts in ("cosyvoice3", "xtts") else ".mp3"
        crossfade_ms = int(getattr(cfg, "crossfade_ms", 10))
        for idx, seg in enumerate(segments):
            text = seg.get("translation") or seg.get("text", "")
            if not text:
                seg["voice_id"] = None
                seg["tts_start"] = position / 1000
                seg["tts_end"] = position / 1000
                continue

            gender = seg.get("gender")
            if not gender:
                speaker_id = seg.get("speaker") or seg.get("speaker_id")
                if speaker_id:
                    gender = selector.infer_gender_from_speaker(speaker_id)

            speaker = str(seg.get("speaker") or seg.get("speaker_id") or "")
            if cfg.tts == "xtts":
                voice_id = seg.get("voice_id") or ctx.get("voice_wav")
                latents_path = voice_latents.get(speaker)
            else:
                voice_id = selector.pick_voice(gender or "U")

            out_file = out_dir / f"{idx}{ext}"
            try:
                if cfg.tts == "xtts" and latents_path and Path(latents_path).exists():
                    try:
                        model.synthesize(
                            text,
                            out_file,
                            latents_path=latents_path,
                            speed=cfg.speed,
                            lang=ctx.get("target_lang", cfg.tts_kw.get("language")),
                        )
                        seg["tts_mode"] = "latent"
                    except Exception as exc:
                        logger.warning(
                            "XTTS latents failed for %s: %s", speaker, exc
                        )
                        model.synthesize(
                            text,
                            out_file,
                            speaker_wav=voice_id,
                            speed=cfg.speed,
                            lang=ctx.get("target_lang", cfg.tts_kw.get("language")),
                        )
                        seg["tts_mode"] = "wav"
                elif cfg.tts == "xtts" and voice_id and Path(voice_id).exists():
                    model.synthesize(
                        text,
                        out_file,
                        speaker_wav=voice_id,
                        speed=cfg.speed,
                        lang=ctx.get("target_lang", cfg.tts_kw.get("language")),
                    )
                    seg["tts_mode"] = "wav"
                else:
                    model.synthesize(
                        text,
                        out_file,
                        voice_id=voice_id,
                        speed=cfg.speed,
                        lang=cfg.tts_kw.get("language"),
                    )
            except Exception as exc:
                logger.error("TTS synthesis failed for segment %d: %s", idx, exc)
                seg.update(
                    {
                        "voice_id": voice_id,
                        "tts_start": position / 1000,
                        "tts_end": position / 1000,
                        "tts_error": True,
                    }
                )
                continue

            if not out_file.exists() or out_file.stat().st_size == 0:
                logger.warning(
                    "Synthesized file missing or empty for segment %d", idx
                )
                seg.update(
                    {
                        "voice_id": voice_id,
                        "tts_start": 0.0,
                        "tts_end": 0.0,
                        "tts_error": True,
                    }
                )
                continue

            if AudioSegment:
                audio = AudioSegment.from_file(out_file)
                if len(audio) == 0:
                    logger.warning("Synthesized audio empty for segment %d", idx)
                    seg.update(
                        {
                            "voice_id": voice_id,
                            "tts_start": 0.0,
                            "tts_end": 0.0,
                            "tts_error": True,
                        }
                    )
                    continue
                seg["tts_path"] = str(out_file)
                if len(combined) == 0:
                    combined = audio
                    seg["tts_start"] = 0.0
                    seg["tts_end"] = len(combined) / 1000
                else:
                    seg_ms = len(audio)
                    allowed = min(
                        crossfade_ms,
                        max(len(combined) - 1, 0),
                        max(seg_ms - 1, 0),
                    )
                    if allowed < 1:
                        start = len(combined)
                        combined = combined + audio
                    else:
                        start = len(combined) - allowed
                        combined = combined.append(audio, crossfade=allowed)
                    seg["tts_start"] = start / 1000
                    seg["tts_end"] = len(combined) / 1000
            else:
                duration = self._probe_duration(out_file)
                if duration == 0:
                    logger.warning("Synthesized audio empty for segment %d", idx)
                    seg.update(
                        {
                            "voice_id": voice_id,
                            "tts_start": 0.0,
                            "tts_end": 0.0,
                            "tts_error": True,
                        }
                    )
                    continue
                seg["tts_path"] = str(out_file)
                seg["tts_start"] = position / 1000
                position += duration
                seg["tts_end"] = position / 1000
                temp_segments.append(out_file)

            seg["voice_id"] = voice_id

        out_path = Path(ctx["temp_dir"]) / "synth_audio.wav"
        if AudioSegment and combined is not None:
            if len(combined) == 0:
                raise RuntimeError("TTS synthesis produced no audio")
            combined.export(out_path, format="wav")
        else:
            self._merge_with_ffmpeg(temp_segments, out_path)
        ctx["artifacts"]["synth_audio"] = str(out_path)
        ctx["artifacts"]["tts_segments"] = [
            {
                "voice_id": seg.get("voice_id"),
                "start": seg.get("tts_start"),
                "end": seg.get("tts_end"),
                "tts_error": seg.get("tts_error", False),
            }
            for seg in segments
        ]
        if not any(not s.get("tts_error", False) for s in segments):
            logger.error("TTS synthesis failed for all segments")
            raise RuntimeError("TTS synthesis failed for all segments")
        logger.debug("TTS step completed, output saved to %s", out_path)

    def _chunk_text(
        self,
        text: str,
        min_s: int = 8,
        max_s: int = 15,
        wps: float = 2.0,
    ) -> Iterable[Tuple[str, str]]:
        """Split text into chunks of roughly 8-15 seconds using punctuation."""
        # Break into sentences while retaining punctuation
        parts = re.split(r"(?<=[.!?,])\s+", text)
        chunks: list[Tuple[str, str]] = []
        current: list[str] = []
        current_time = 0.0
        for part in parts:
            if not part:
                continue
            words = part.split()
            est = len(words) / wps
            if current and (current_time + est) > max_s:
                joined = " ".join(current).strip()
                chunks.append((joined, joined[-1] if joined else ""))
                current = [part]
                current_time = est
            else:
                current.append(part)
                current_time += est
            if current_time >= min_s:
                joined = " ".join(current).strip()
                chunks.append((joined, joined[-1] if joined else ""))
                current = []
                current_time = 0.0
        if current:
            joined = " ".join(current).strip()
            chunks.append((joined, joined[-1] if joined else ""))
        return chunks

    @staticmethod
    def _pause_duration(punct: str) -> int:
        """Return pause duration in ms based on punctuation."""
        if punct in ".!?":
            return 700
        if punct in ",;":
            return 350
        return 0

    @staticmethod
    def _generate_silence(duration_ms: int, path: Path) -> None:
        """Create a silent audio file of the given duration."""
        path = Path(path)
        if path.suffix.lower() == ".mp3":
            cmd = [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                f"{duration_ms / 1000}",
                "-q:a",
                "9",
                "-acodec",
                "libmp3lame",
                str(path),
                "-loglevel",
                "error",
                "-y",
            ]
            subprocess.run(cmd, check=True)
        else:
            frames = int(16000 * (duration_ms / 1000))
            with wave.open(str(path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * frames)

    @staticmethod
    def _probe_duration(path: Path) -> int:
        """Return duration of an audio file in milliseconds."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            seconds = float(result.stdout.strip() or 0)
            return int(seconds * 1000)
        except Exception as exc:  # pragma: no cover - rely on torchaudio fallback
            logger = logging.getLogger(__name__)
            logger.warning("ffprobe failed for %s: %s", path, exc)
            try:
                import torchaudio

                info = torchaudio.info(str(path))
                seconds = info.num_frames / float(info.sample_rate)
                return int(seconds * 1000)
            except Exception as exc2:
                logger.error("Failed to determine duration for %s: %s", path, exc2)
                return 0

    @staticmethod
    def _merge_with_ffmpeg(inputs: list[Path], output: Path) -> None:
        """Merge audio segments into a single WAV file using ffmpeg."""
        output.parent.mkdir(parents=True, exist_ok=True)
        if not inputs:
            # Create an empty WAV file
            with wave.open(str(output), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"")
            return

        list_file = output.with_suffix(".txt")
        with list_file.open("w") as f:
            for p in inputs:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-y",
            str(output),
            "-loglevel",
            "error",
        ]
        subprocess.run(cmd, check=True)
