from __future__ import annotations
import os
from typing import Optional, List
from google.cloud import speech_v1p1beta1 as speech  # v1 with diarization
from dubbing_ai.interfaces import DiarizerPlugin, DiarizeHints, DiarizeResult
from dubbing_ai.artifact_types import DiarSegment, write_jsonl

def _group_words_into_spans(words):
    # words: list of (start_sec, end_sec, speakerTag)
    if not words:
        return []
    words = sorted(words, key=lambda x: x[0])
    spans = []
    cur_spk = words[0][2]
    cur_start = words[0][0]
    cur_end = words[0][1]
    for s,e,spk in words[1:]:
        if spk == cur_spk and s <= cur_end + 0.3:  # 300ms tolerance
            cur_end = max(cur_end, e)
        else:
            spans.append((cur_spk, cur_start, cur_end))
            cur_spk, cur_start, cur_end = spk, s, e
    spans.append((cur_spk, cur_start, cur_end))
    return spans

class GCPSTTDiarizer(DiarizerPlugin):
    name = "gcp-stt"

    def run(self, audio_path: str, out_dir: str, hints: Optional[DiarizeHints] = None) -> DiarizeResult:
        os.makedirs(out_dir, exist_ok=True)

        client = speech.SpeechClient()
        with open(audio_path, "rb") as f:
            content = f.read()

        diar_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=hints.min_speakers if hints and hints.min_speakers else 1,
            max_speaker_count=hints.max_speakers if hints and hints.max_speakers else 8,
        )

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="tr-TR",
            enable_word_time_offsets=True,
            diarization_config=diar_config,
            model="latest_long",
        )

        op = client.long_running_recognize(config=config, audio=audio)
        resp = op.result(timeout=3600)

        words_all = []
        for res in resp.results:
            alt = res.alternatives[0]
            for w in getattr(alt, "words", []):
                s = (w.start_time.seconds + w.start_time.nanos * 1e-9)
                e = (w.end_time.seconds + w.end_time.nanos * 1e-9)
                tag = getattr(w, "speaker_tag", 0)
                words_all.append((s,e,tag))

        spans = _group_words_into_spans(words_all)
        segs: List[DiarSegment] = [DiarSegment(spk=f"S{int(t)}", start=float(s), end=float(e), conf=None, overlap=False) for (t,s,e) in spans]

        diar_jsonl = os.path.join(out_dir, "diar_segments.jsonl")
        write_jsonl(diar_jsonl, segs)
        return DiarizeResult(diar_jsonl_path=diar_jsonl, profiles_json_path=None)
