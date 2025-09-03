from __future__ import annotations
import os, tempfile, json, click
from typing import Optional
from dubbing_ai.utils.gcs_io import ensure_local_or_download, maybe_upload
from dubbing_ai.utils.audio import to_wav_mono_16k
from dubbing_ai.transcriber.whisperx import WhisperXTranscriber
from dubbing_ai.interfaces import DiarizeHints
from dubbing_ai.dispatcher import select_diarizer
from dubbing_ai.align.assign_words import main_assign

@click.command()
@click.option("--audio", "audio_uri", required=True, help="Input audio (local path or gs:// URI)")
@click.option("--out", "out_prefix", required=True, help="Output directory or gs:// prefix")
@click.option("--diarizer", type=click.Choice(["pyannote", "nemo-msdd", "gcp-stt"]), default=None, help="Which diarizer to use")
@click.option("--domain", type=click.Choice(["podcast","meeting","telephony","streaming"]), default="podcast")
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
@click.option("--lang", type=str, default=None, help="ASR language hint, e.g., tr")
@click.option("--model-size", type=str, default="medium", help="WhisperX model size (tiny|base|small|medium|large-v2)")
def main(audio_uri: str, out_prefix: str, diarizer: Optional[str], domain: str,
         min_speakers: Optional[int], max_speakers: Optional[int], lang: Optional[str], model_size: str):
    """Run ASR (WhisperX) + Diarization (pyannote/NeMo/GCP) and produce artefacts."""
    hints = DiarizeHints(min_speakers=min_speakers, max_speakers=max_speakers, domain=domain)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import warnings
    from pyannote.audio.utils.reproducibility import ReproducibilityWarning
    warnings.filterwarnings("ignore", category=ReproducibilityWarning)

    with tempfile.TemporaryDirectory() as tmp:
        local_in = ensure_local_or_download(audio_uri, tmp)
        wav16 = to_wav_mono_16k(local_in, tmp)

        # 1) ASR
        asr_dir = os.path.join(tmp, "asr")
        os.makedirs(asr_dir, exist_ok=True)
        asr_path = WhisperXTranscriber().run(wav16, asr_dir, lang_hint=lang, model_size=model_size)

        # 2) Diarization
        dia_dir = os.path.join(tmp, "dia")
        os.makedirs(dia_dir, exist_ok=True)
        dia_plugin = select_diarizer(diarizer, hints)
        dia = dia_plugin.run(wav16, dia_dir, hints=hints)

        # 3) Assign words to speakers + EDL
        words_jsonl = os.path.join(tmp, "asr_words_speaker.jsonl")
        edl_jsonl = os.path.join(tmp, "edl.jsonl")
        main_assign(asr_path, dia.diar_jsonl_path, words_jsonl, edl_jsonl)

        # Upload/move outputs
        out_asr = maybe_upload(asr_path, out_prefix)
        out_dia = maybe_upload(dia.diar_jsonl_path, out_prefix)
        out_words = maybe_upload(words_jsonl, out_prefix)
        out_edl = maybe_upload(edl_jsonl, out_prefix)

        result = {
            "asr_segments": out_asr,
            "diar_segments": out_dia,
            "asr_words_speaker": out_words,
            "edl": out_edl,
            "diarizer": getattr(dia_plugin, "name", "unknown"),
            "domain": domain,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
