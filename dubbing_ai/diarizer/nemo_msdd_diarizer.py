from __future__ import annotations
import os, tempfile, subprocess, json, uuid, shutil
from typing import Optional
from dubbing_ai.interfaces import DiarizerPlugin, DiarizeHints, DiarizeResult
from dubbing_ai.utils.rttm import parse_rttm

class NemoMSDDDiarizer(DiarizerPlugin):
    name = "nemo-msdd"

    def run(self, audio_path: str, out_dir: str, hints: Optional[DiarizeHints] = None) -> DiarizeResult:
        os.makedirs(out_dir, exist_ok=True)
        # Requires nemo-toolkit installed in environment
        # We use NeMo diarize utility module with a config yaml (provided in configs/).
        cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "configs", "nemo_msdd_telephony.yaml")
        cfg = os.path.abspath(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            wav_dir = os.path.join(tmp, "wavs")
            os.makedirs(wav_dir, exist_ok=True)
            # Copy/Link audio file into dir (NeMo expects dir manifest)
            basename = "audio.wav"
            target = os.path.join(wav_dir, basename)
            shutil.copy2(audio_path, target)

            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                f.write(json.dumps({"audio_filepath": target, "offset": 0, "duration": None, "text": "-", "label": "UNK", "uniq_id": "utt1"}) + "\n")

            out_dir_nemo = os.path.join(tmp, "out")
            os.makedirs(out_dir_nemo, exist_ok=True)

            env = os.environ.copy()
            cmd = [
                "python", "-m", "nemo.collections.asr.parts.utils.diarize",
                f"diarizer.manifest_filepath={manifest}",
                f"diarizer.out_dir={out_dir_nemo}",
                f"diarizer.msdd_model.parameters.infer_params.do_overlap=true",
            ]
            # Optional speaker hints
            if hints and hints.max_speakers:
                cmd.append(f"diarizer.clustering.parameters.max_num_speakers={hints.max_speakers}")

            # Allow overriding default config
            if os.path.exists(cfg):
                cmd.extend(["--config-path", os.path.dirname(cfg), "--config-name", os.path.basename(cfg)])

            subprocess.run(cmd, check=True)

            # NeMo outputs RTTM at out_dir_nemo/pred_rttms
            rttm_dir = os.path.join(out_dir_nemo, "pred_rttms")
            # Assume single file
            rttm_files = [os.path.join(rttm_dir, f) for f in os.listdir(rttm_dir) if f.endswith(".rttm")]
            if not rttm_files:
                raise RuntimeError("NeMo did not produce RTTM output.")
            diar_segs = parse_rttm(rttm_files[0])

        # Convert to jsonl
        diar_jsonl = os.path.join(out_dir, "diar_segments.jsonl")
        from dubbing_ai.artifact_types import write_jsonl
        write_jsonl(diar_jsonl, diar_segs)

        return DiarizeResult(diar_jsonl_path=diar_jsonl, profiles_json_path=None)
