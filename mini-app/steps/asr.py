from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from core.pipeline.base import Context
from core.registry.asr import ASR_REGISTRY


logger = logging.getLogger(__name__)


class GapPreservingASRStep:
    """ASR step that preserves natural gaps between segments.

    This implementation uses ``ffmpeg``'s ``silencedetect`` filter to locate
    gaps in the original audio. Gaps between 0.5 and 3 seconds are preserved
    and considered when assigning transcriptions to diarized segments.  When
    Whisper is used as the ASR backend the transcription is performed with
    ``word_timestamps=True`` to enable fine‑grained splitting.  If another ASR
    model is requested the step falls back to the legacy behaviour of simply
    assigning returned texts to existing segments.
    """

    name = "ASR"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _detect_gaps(self, audio_path: Path) -> List[Dict[str, float]]:
        """Return list of gap dictionaries using ffmpeg ``silencedetect``.

        Only gaps in the range [0.5, 3] seconds are kept.  The command output is
        parsed from stderr lines that contain ``silence_start`` and
        ``silence_end`` information.
        """

        cmd = [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=noise=-30dB:d=0.5",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - ffmpeg optional
            logger.warning("silencedetect failed: %s", exc)
            return []

        gaps: List[Dict[str, float]] = []
        start: float | None = None
        for line in proc.stderr.splitlines():
            line = line.strip()
            if line.startswith("silence_start:"):
                try:
                    start = float(line.split(":", 1)[1])
                except ValueError:
                    start = None
            elif line.startswith("silence_end:") and start is not None:
                parts = line.replace("silence_end:", "").split()
                try:
                    end = float(parts[0])
                    dur = float(parts[-1])
                except ValueError:
                    start = None
                    continue
                gaps.append({"start": start, "end": end, "duration": dur})
                start = None

        preserved = [g for g in gaps if 0.5 <= g["duration"] <= 3.0]
        if preserved:
            avg = sum(g["duration"] for g in preserved) / len(preserved)
            total = sum(g["duration"] for g in preserved)
            logger.info(
                "Detected %d gaps (preserving %d, avg %.2fs, total %.2fs)",
                len(gaps),
                len(preserved),
                avg,
                total,
            )
        else:
            logger.info("No preservable gaps detected")
        return preserved

    def _extract_segment(self, audio_path: Path, start: float, end: float, out: Path) -> None:
        """Extract ``[start, end]`` from ``audio_path`` into ``out`` using ffmpeg."""

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            f"{start}",
            "-to",
            f"{end}",
            str(out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def run(self, ctx: Context) -> None:
        audio_path = Path(ctx["artifacts"]["original_audio"])
        logger.debug("ASR step started for audio: %s", audio_path)

        # Fallback for non-Whisper models ---------------------------------
        if self.model_name != "whisper":
            model = ASR_REGISTRY.create(self.model_name)
            transcripts = model.transcribe(str(audio_path))
            existing = ctx["artifacts"].get("segments")
            if existing:
                for i, seg in enumerate(existing):
                    seg["text"] = transcripts[i]["text"] if i < len(transcripts) else ""
                    seg["transcript_source"] = self.model_name
                    seg["split_method"] = "model"
            else:
                for t in transcripts:
                    t["transcript_source"] = self.model_name
                    t["split_method"] = "model"
                ctx["artifacts"]["segments"] = transcripts
            logger.debug("Generic ASR step completed")
            return

        # Whisper-based advanced path -------------------------------------
        gaps = self._detect_gaps(audio_path)

        # obtain underlying whisper model through registry to respect model config
        wrapper = ASR_REGISTRY.create(self.model_name)
        if hasattr(wrapper, "_ensure_model"):
            whisper_model = wrapper._ensure_model()  # type: ignore[attr-defined]
        else:  # pragma: no cover - should not happen for whisper
            import whisper

            whisper_model = whisper.load_model(self.model_name)

        existing = ctx["artifacts"].get("segments", [])
        final_segments: List[Dict[str, Any]] = []

        for idx, seg in enumerate(existing):
            start = float(seg["start"])
            end = float(seg["end"])

            # Extract audio snippet for the segment into temp directory
            tmp_path = Path(ctx["temp_dir"]) / f"asr_{idx:04d}.wav"
            self._extract_segment(audio_path, start, end, tmp_path)

            try:
                result = whisper_model.transcribe(str(tmp_path), word_timestamps=True)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            # Adjust timestamps to global timeline
            for wseg in result.get("segments", []):
                for w in wseg.get("words", []):
                    w["start"] += start
                    w["end"] += start

            # Determine gaps that fall inside this diarized segment
            internal = [g for g in gaps if g["start"] > start and g["end"] < end]

            if internal:
                prev = start
                for g in internal:
                    words = [
                        w.get("word", "")
                        for seg_w in result.get("segments", [])
                        for w in seg_w.get("words", [])
                        if prev <= w.get("start", 0.0) < g["start"]
                    ]
                    text = "".join(words).strip()
                    new_seg = dict(seg)
                    new_seg.update(
                        {
                            "start": prev,
                            "end": g["start"],
                            "text": text,
                            "followed_by_gap": True,
                            "gap_duration": g["duration"],
                            "split_method": "gap",
                            "transcript_source": "whisper",
                        }
                    )
                    final_segments.append(new_seg)
                    prev = g["end"]

                # trailing part after last gap
                words = [
                    w.get("word", "")
                    for seg_w in result.get("segments", [])
                    for w in seg_w.get("words", [])
                    if prev <= w.get("start", 0.0) < end
                ]
                text = "".join(words).strip()
                last_seg = dict(seg)
                last_seg.update(
                    {
                        "start": prev,
                        "end": end,
                        "text": text,
                        "followed_by_gap": False,
                        "split_method": "gap",
                        "transcript_source": "whisper",
                    }
                )
                final_segments.append(last_seg)
                continue

            # No internal gaps -------------------------------------------------
            whisper_segs = result.get("segments", [])
            if len(whisper_segs) > 1:
                # Split according to whisper's own timeline
                for wseg in whisper_segs:
                    new_seg = dict(seg)
                    s = float(wseg.get("start", 0.0)) + start
                    e = float(wseg.get("end", 0.0)) + start
                    text = wseg.get("text", "").strip()
                    gap_after = next((g for g in gaps if abs(g["start"] - e) < 0.1), None)
                    new_seg.update(
                        {
                            "start": s,
                            "end": e,
                            "text": text,
                            "followed_by_gap": bool(gap_after),
                            "gap_duration": gap_after["duration"] if gap_after else 0.0,
                            "split_method": "whisper",
                            "transcript_source": "whisper",
                        }
                    )
                    final_segments.append(new_seg)
            else:
                # Fallback: time-based single segment
                text = whisper_segs[0]["text"].strip() if whisper_segs else ""
                gap_after = next((g for g in gaps if abs(g["start"] - end) < 0.1), None)
                new_seg = dict(seg)
                new_seg.update(
                    {
                        "text": text,
                        "followed_by_gap": bool(gap_after),
                        "gap_duration": gap_after["duration"] if gap_after else 0.0,
                        "split_method": "time" if not whisper_segs else "whisper",
                        "transcript_source": "whisper",
                    }
                )
                final_segments.append(new_seg)

        ctx["artifacts"]["segments"] = final_segments

        # Validation & statistics logging
        empty = [i for i, s in enumerate(final_segments) if not s.get("text")]
        if empty:
            logger.warning("ASR produced %d empty segments", len(empty))
        logger.info(
            "ASR step completed: %d segments, %d gaps preserved", len(final_segments), len(gaps)
        )


# Backwards compatibility alias -------------------------------------------------
ASRStep = GapPreservingASRStep

