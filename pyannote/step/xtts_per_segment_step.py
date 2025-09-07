import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger("app-pipeline")

class XTTSPerSegmentStep:
    """
    Her cümle için TTS üretir (XTTS/Coqui/Edge → sessizlik fallback) ve
    zamanlara göre hizalanmış tek WAV oluşturur.
    """
    name = "XTTSPerSegment"

    def __init__(self, tts_name: str = "xtts", tts_kw: Optional[Dict[str, Any]] = None,
                 voice_map: Optional[Dict[str, str]] = None, sample_rate: int = 16000,
                 edge_voice: Optional[str] = None) -> None:
        self.tts_name = tts_name
        self.tts_kw = tts_kw or {}
        self.voice_map = voice_map or {}
        self.sample_rate = int(sample_rate)
        self.edge_voice = edge_voice  # ör. tr-TR-AhmetNeural

        # state
        self._engine_xtts = None
        self._engine_coqui = None
        self._edge_available = False

    # ---- backend yükleyiciler ----
    def _try_xtts(self):
        try:
            from core.models.tts.xtts import XTTSEngine  # type: ignore
            self._engine_xtts = XTTSEngine(**self.tts_kw)
            logger.info("TTS backend: core.models.tts.xtts.XTTSEngine")
        except Exception as e:
            self._engine_xtts = None
            logger.warning("XTTS bulunamadı (%s).", e)

    def _try_coqui(self):
        try:
            from TTS.api import TTS  # type: ignore
            # Kullanıcı kendi modelini tts_kw ile geçebilir; aksi halde default dene
            model_name = self.tts_kw.get("model_name") or "tts_models/multilingual/multi-dataset/xtts_v2"
            self._engine_coqui = TTS(model_name=model_name)
            logger.info("TTS backend: Coqui TTS (%s)", model_name)
        except Exception as e:
            self._engine_coqui = None
            logger.warning("Coqui TTS bulunamadı (%s).", e)

    def _try_edge(self):
        try:
            import edge_tts  # type: ignore
            self._edge_available = True
            logger.info("TTS backend: edge-tts")
        except Exception as e:
            self._edge_available = False
            logger.warning("edge-tts bulunamadı (%s).", e)

    async def _edge_synthesize_wav(self, text: str, voice: str, out_path: Path) -> None:
        import edge_tts  # type: ignore
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate="+0%",
            volume="+0%",
            # WAV çıktı:
            tts_output_format="riff-16000hz-16bit-mono-pcm",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        async with await communicate.stream() as stream:
            with out_path.open("wb") as f:
                async for chunk in stream:
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])

    # ---- ana çalışma ----
    def run(self, ctx: Dict[str, Any]) -> None:
        from pydub import AudioSegment  # type: ignore

        temp = Path(ctx["temp_dir"])
        voices_dir = Path(ctx["artifacts"].get("ref_voices_dir", temp / "voices"))
        if not voices_dir.exists():
            logger.warning("Referans ses klasörü bulunamadı, TTS ses eşleme için default kullanılacak.")

        sents: List[Dict[str, Any]] = ctx["artifacts"].get("sentences_tr") or ctx["artifacts"].get("sentences") or []
        if not sents:
            raise RuntimeError("XTTS: cümle bulunamadı")

        # backendleri sırayla dene
        self._try_xtts()
        if not self._engine_xtts:
            self._try_coqui()
        if not self._engine_xtts and not self._engine_coqui:
            self._try_edge()

        seg_out_dir = temp / "tts_segments"; seg_out_dir.mkdir(parents=True, exist_ok=True)
        rendered: List[Tuple[float, AudioSegment, str]] = []  # (start_s, audio, spk)

        # Edge-tts için varsayılan ses
        target_lang = (ctx["config"].get("target_lang") or "tr").lower()
        default_edge_voice = self.edge_voice or ("tr-TR-AhmetNeural" if target_lang.startswith("tr") else "en-US-GuyNeural")

        for idx, s in enumerate(sents):
            text = (s.get("text_tr") or s.get("text") or "").strip()
            if not text:
                continue
            spk = str(s.get("speaker") or "SPEAKER_00")
            ref_wav = self.voice_map.get(spk) or str(voices_dir / f"{spk}.wav")
            start_s = float(s["start"])
            dur_ms_budget = int(max(200, (float(s["end"]) - float(s["start"])) * 1000))

            out_file = seg_out_dir / f"seg_{idx:04d}_{spk}.wav"
            audio_seg = None

            # 1) XTTS (projeye özgü)
            if self._engine_xtts:
                try:
                    audio_seg = self._engine_xtts.tts(text=text, speaker_wav=ref_wav)  # pydub.AudioSegment beklenir
                except Exception as e:
                    logger.warning("XTTS hatası (%s) -> diğer backend'e geçiliyor", e)
                    audio_seg = None

            # 2) Coqui TTS
            if (audio_seg is None) and self._engine_coqui:
                try:
                    tmp_path = out_file.with_suffix(".coqui.wav")
                    self._engine_coqui.tts_to_file(text=text, file_path=str(tmp_path))
                    audio_seg = AudioSegment.from_file(tmp_path)
                except Exception as e:
                    logger.warning("Coqui TTS hatası (%s) -> diğer backend'e geçiliyor", e)
                    audio_seg = None

            # 3) edge-tts
            if (audio_seg is None) and self._edge_available:
                try:
                    asyncio.run(self._edge_synthesize_wav(text, default_edge_voice, out_file))
                    audio_seg = AudioSegment.from_file(out_file)
                except Exception as e:
                    logger.warning("edge-tts hatası (%s) -> sessizlik fallback", e)
                    audio_seg = None

            # 4) sessizlik fallback
            if audio_seg is None:
                from pydub import AudioSegment as _AS  # type: ignore
                audio_seg = _AS.silent(duration=dur_ms_budget)

            # segment dosyası
            audio_seg.set_channels(1).set_frame_rate(self.sample_rate).export(out_file, format="wav")
            rendered.append((start_s, audio_seg, spk))

        if not rendered:
            raise RuntimeError("XTTS hiçbir segment üretemedi.")

        # Zamanlamaya göre overlay
        last_end_ms = 0
        for start_s, audio_seg, _ in rendered:
            end_ms = int(start_s * 1000) + len(audio_seg)
            last_end_ms = max(last_end_ms, end_ms)
        base = AudioSegment.silent(duration=max(1000, last_end_ms))
        for start_s, audio_seg, _ in rendered:
            base = base.overlay(audio_seg, position=int(start_s * 1000))

        out_wav = temp / "tts_merged.wav"
        base.set_channels(1).set_frame_rate(self.sample_rate).export(out_wav, format="wav")
        ctx["artifacts"]["synth_audio"] = str(out_wav)
        ctx["artifacts"]["tts_segments"] = [str(p) for p in sorted(seg_out_dir.glob("*.wav"))]
        logger.info("TTS birleşik WAV -> %s", out_wav)
