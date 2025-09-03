from __future__ import annotations
import os
from typing import Optional, List
from dubbing_ai.interfaces import DiarizerPlugin, DiarizeHints, DiarizeResult
from dubbing_ai.artifact_types import DiarSegment, write_jsonl
from pyannote.audio import Pipeline

class PyannoteDiarizer(DiarizerPlugin):
    name = "pyannote"

    def run(self, audio_path: str, out_dir: str, hints: Optional[DiarizeHints] = None) -> DiarizeResult:
        os.makedirs(out_dir, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN env is required for pyannote models.")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

        params = {}
        if hints:
            if hints.min_speakers: params["min_speakers"] = hints.min_speakers
            if hints.max_speakers: params["max_speakers"] = hints.max_speakers

        diarization = pipeline(audio_path, **params)

        segs: List[DiarSegment] = []
        # diarization: pyannote.core.Annotation
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segs.append(DiarSegment(spk=str(speaker), start=float(segment.start), end=float(segment.end), conf=None, overlap=False))

        out = os.path.join(out_dir, "diar_segments.jsonl")
        write_jsonl(out, segs)
        return DiarizeResult(diar_jsonl_path=out, profiles_json_path=None)
