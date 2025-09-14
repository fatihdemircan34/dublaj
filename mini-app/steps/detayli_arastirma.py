"""
Detaylı Araştırma (Perfect Mix + Loudness QA)
---------------------------------------------
Bu step, dublaj kalitesini artırmak için:
  1) Orijinal sesten konuşmayı ayırır (Demucs varsa) → müzik/FX korunur
  2) Dublaj sesi ile "perfect mix" üretir:
     - music chain: highpass→lowpass→volume
     - (opsiyonel) sidechaincompress ile müziği dublaja göre kısar
     - Demucs yoksa: güvenli ducking fallback
  3) LUFS (I), True Peak (dBTP), LRA ölçer ve EBU R128 iki-geçişli normalize eder
  4) Çıktıları context'e yazar:
       artifacts.mixed_wav -> normalize edilmiş final WAV
       audio.loudness.pre/post -> ölçüm metrikleri
       flags.perfect_mix = True

Varsayılan hedefler: I=-16 LUFS, TP=-1.0 dBTP, LRA=11 LU (YouTube için güvenli).

Konfig (opsiyonel):
  config = {
    "demucs_model": "htdemucs",
    "music_volume": 0.8,
    "duck_enable": True,
    "duck_threshold": 0.015,
    "duck_ratio": 8.0,
    "duck_attack_ms": 5,
    "duck_release_ms": 250,
    "sample_rate": 16000,
    "channels": 1,
    "loudness_target_I": -16.0,
    "loudness_target_TP": -1.0,
    "loudness_target_LRA": 11.0,
  }
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Iterable

logger = logging.getLogger("detayli_arastirma")


# ──────────────────────────────────────────────────────────────────────────────
# Küçük yardımcılar
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_ok(cmd: list[str], timeout: int = 10) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def _demucs_available() -> bool:
    # "python -m demucs --help" veya "demucs --help" çalışıyor mu?
    return _cmd_ok(["python", "-m", "demucs", "--help"]) or _cmd_ok(["demucs", "--help"])


def _run(cmd: list[str], timeout: Optional[int] = None) -> None:
    logger.debug("RUN: %s", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or "command failed")


def _find_artifact_wav(artifacts: Dict[str, Any], candidates: Iterable[str]) -> Optional[Path]:
    # 1) anahtar adına bakarak
    for k, v in artifacts.items():
        if isinstance(v, str) and v.lower().endswith(".wav"):
            lk = k.lower()
            if any(c in lk for c in candidates):
                p = Path(v)
                if p.exists():
                    return p
    # 2) dosya adına bakarak
    for v in artifacts.values():
        if isinstance(v, str) and v.lower().endswith(".wav"):
            lv = Path(v).name.lower()
            if any(c in lv for c in candidates):
                p = Path(v)
                if p.exists():
                    return p
    return None


def _measure_loudness(audio_path: Path, I: float, TP: float, LRA: float) -> Dict[str, float]:
    """
    ffmpeg loudnorm ile LUFS (I), True Peak (TP), LRA ölç.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(audio_path),
        "-af", f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
        "-f", "null", "-"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"loudnorm ölçüm hatası: {r.stderr or r.stdout}")
    text = (r.stderr or "") + "\n" + (r.stdout or "")
    i1, i2 = text.rfind("{"), text.rfind("}")
    if i1 == -1 or i2 == -1 or i2 <= i1:
        raise RuntimeError("loudnorm JSON parse edilemedi")
    j = json.loads(text[i1:i2+1])
    return {
        "input_i": float(j.get("input_i", 0.0)),
        "input_tp": float(j.get("input_tp", 0.0)),
        "input_lra": float(j.get("input_lra", 0.0)),
        "input_thresh": float(j.get("input_thresh", 0.0)),
        "target_offset": float(j.get("target_offset", 0.0)),
    }


def _loudness_two_pass(audio_in: Path, audio_out: Path, *, I: float, TP: float, LRA: float,
                       sample_rate: int, channels: int) -> Dict[str, float]:
    stats = _measure_loudness(audio_in, I, TP, LRA)
    ln = (
        f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
        f":measured_I={stats['input_i']}"
        f":measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}"
        f":measured_thresh={stats['input_thresh']}"
        f":offset={stats['target_offset']}"
        f":linear=true:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(audio_in),
        "-af", ln,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        str(audio_out)
    ]
    _run(cmd)
    return _measure_loudness(audio_out, I, TP, LRA)


def _ducking_fallback(original_audio: Path, dubbed_audio: Path, output_path: Path,
                      *, music_volume: float, sample_rate: int, channels: int) -> None:
    """
    Basit ducking fallback: orijinal müziği kıs, dublajı üzerine amix et.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_audio),
        "-i", str(dubbed_audio),
        "-filter_complex", f"[0:a]volume={music_volume}[m];[m][1:a]amix=inputs=2:duration=longest:normalize=0",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    _run(cmd)


def _demucs_separate_music_only(audio_path: Path, out_dir: Path, model: str) -> Path:
    """
    Demucs ile kaynak ayırma; vokalleri atıp drums+bass+other → music_only.wav üret.
    """
    sep_out = out_dir / "demucs_sep"
    sep_out.mkdir(parents=True, exist_ok=True)

    cmd = ["python", "-m", "demucs.separate", "-n", model, "-o", str(sep_out), str(audio_path)]
    _run(cmd, timeout=900)  # 15 dk üst sınır

    base = Path(audio_path).stem
    stem_dir = sep_out / model / base
    drums = stem_dir / "drums.wav"
    bass  = stem_dir / "bass.wav"
    other = stem_dir / "other.wav"
    if not drums.exists() or not bass.exists() or not other.exists():
        raise RuntimeError("Demucs ayrıştırma eksik çıktı üretti")

    music_only = out_dir / "music_only.wav"
    cmd_mix = [
        "ffmpeg", "-y",
        "-i", str(drums),
        "-i", str(bass),
        "-i", str(other),
        "-filter_complex", "amix=inputs=3:duration=longest:normalize=0",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(music_only)
    ]
    _run(cmd_mix)
    return music_only


def _perfect_mix(music_path: Path, dubbed_path: Path, output_path: Path, *,
                 music_volume: float, duck_enable: bool,
                 duck_threshold: float, duck_ratio: float, duck_attack_ms: int, duck_release_ms: int,
                 sample_rate: int, channels: int,
                 music_highpass_hz: int = 120, music_lowpass_hz: int = 12000) -> None:
    """
    Demucs sonrası müzik ile dublajın nihai miksini yapar.
    """
    music_chain = (
        f"highpass=f={music_highpass_hz},"
        f"lowpass=f={music_lowpass_hz},"
        f"volume={music_volume}"
    )
    if duck_enable:
        filter_complex = (
            f"[0:a]{music_chain}[m0];"
            f"[m0][1:a]sidechaincompress="
            f"threshold={duck_threshold}:ratio={duck_ratio}:attack={duck_attack_ms}:release={duck_release_ms}[md];"
            f"[md][1:a]amix=inputs=2:duration=longest:normalize=0[mix]"
        )
    else:
        filter_complex = (
            f"[0:a]{music_chain}[m0];"
            f"[m0][1:a]amix=inputs=2:duration=longest:normalize=0[mix]"
        )
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(music_path),
        "-i", str(dubbed_path),
        "-filter_complex", filter_complex,
        "-map", "[mix]",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    _run(cmd, timeout=180)


# ──────────────────────────────────────────────────────────────────────────────
# Step sınıfı
# ──────────────────────────────────────────────────────────────────────────────
class PerfectMixStep:
    """
    MixStep yerine drop‑in replacement.
    artifacts.mixed_wav üretir (LipSyncStep bunu tüketir).
    """
    name = "PerfectMix"
    def __init__(self,
                 lufs_target: float = -14.0,  # pipeline default'unu koruyoruz
                 duck_db: float = -7.0,       # sadece heuristik için kullanılıyor
                 pan_amount: float = 0.0,     # kullanılmıyor (interface için)
                 **kwargs):
        self.lufs_target = lufs_target
        self.duck_db = duck_db
        self.pan_amount = pan_amount
        # opsiyonel ayarlar
        self.kwargs = kwargs

    def __call__(self, ctx: Dict[str, Any]) -> None:
        return self.run(ctx)

    def run(self, ctx: Dict[str, Any]) -> None:
        cfg = dict(ctx.get("config", {}) or {})
        temp_dir = Path(ctx.get("temp_dir", ".")).resolve()
        artifacts: Dict[str, Any] = ctx.setdefault("artifacts", {})
        audio_info: Dict[str, Any] = ctx.setdefault("audio", {})
        flags: Dict[str, Any] = ctx.setdefault("flags", {})

        # Girişleri bul
        # - orijinal ses (ExtractAudioStep çıktısı)
        orig = _find_artifact_wav(artifacts, ["original", "extract", "audio", "raw"])
        # - dublaj (TTSStep birleşik WAV)
        dub  = _find_artifact_wav(artifacts, ["tts", "dub", "synth", "merged", "voice", "speech"])

        if not orig:
            # fallback: temp_dir içinde yaygın adlar
            for name in ("extracted_audio.wav", "audio.wav", "original.wav"):
                p = temp_dir / name
                if p.exists():
                    orig = p; break
        if not dub:
            for name in ("tts_merged.wav", "dubbed_audio.wav", "tts.wav", "speech.wav"):
                p = temp_dir / name
                if p.exists():
                    dub = p; break

        if not orig or not dub:
            raise RuntimeError(
                "PerfectMixStep: giriş bulunamadı. "
                f"original={orig}, dubbed={dub}. artifacts keys={list(artifacts.keys())}"
            )

        # Hedefler (config’ten override edilebilir)
        I   = float(cfg.get("loudness_target_I",  self.lufs_target if self.lufs_target is not None else -16.0))
        TP  = float(cfg.get("loudness_target_TP", -1.0))
        LRA = float(cfg.get("loudness_target_LRA", 11.0))

        # Miks ayarları
        music_volume   = float(cfg.get("music_volume", 0.8))
        duck_enable    = bool(cfg.get("duck_enable", True))
        duck_threshold = float(cfg.get("duck_threshold", 0.015))
        duck_ratio     = float(cfg.get("duck_ratio", 8.0))
        duck_attack_ms = int(cfg.get("duck_attack_ms", 5))
        duck_release_ms= int(cfg.get("duck_release_ms", 250))
        sample_rate    = int(cfg.get("sample_rate", 16000))
        channels       = int(cfg.get("channels", 1))
        demucs_model   = str(cfg.get("demucs_model", "htdemucs"))

        # Heuristik: duck_db çok agresifse ratio biraz yükselsin
        if self.duck_db <= -9.0 and "duck_ratio" not in cfg:
            duck_ratio = 10.0

        perfect_raw  = temp_dir / "perfect_mixed_raw.wav"
        perfect_norm = temp_dir / "perfect_mixed_audio.wav"

        try:
            if _demucs_available():
                logger.info("🎵 Demucs bulundu → konuşma ayrıştırma başlıyor")
                music_only = _demucs_separate_music_only(orig, temp_dir, demucs_model)
                _perfect_mix(
                    music_only, dub, perfect_raw,
                    music_volume=music_volume,
                    duck_enable=duck_enable,
                    duck_threshold=duck_threshold,
                    duck_ratio=duck_ratio,
                    duck_attack_ms=duck_attack_ms,
                    duck_release_ms=duck_release_ms,
                    sample_rate=sample_rate,
                    channels=channels,
                )
            else:
                logger.warning("Demucs bulunamadı → ducking fallback kullanılacak")
                _ducking_fallback(
                    orig, dub, perfect_raw,
                    music_volume=0.3,
                    sample_rate=sample_rate,
                    channels=channels,
                )
        except Exception as e:
            logger.warning("Demucs/mix başarısız (%s) → ducking fallback", e)
            _ducking_fallback(
                orig, dub, perfect_raw,
                music_volume=0.3,
                sample_rate=sample_rate,
                channels=channels,
            )

        # Loudness ölç → normalize → tekrar ölç
        pre_stats  = _measure_loudness(perfect_raw, I, TP, LRA)
        post_stats = _loudness_two_pass(perfect_raw, perfect_norm, I=I, TP=TP, LRA=LRA,
                                        sample_rate=sample_rate, channels=channels)

        logger.info(
            "PerfectMixStep Loudness | Önce: I=%.1f LUFS, TP=%.1f dBTP, LRA=%.1f  |  Sonra: I=%.1f, TP=%.1f, LRA=%.1f",
            pre_stats["input_i"], pre_stats["input_tp"], pre_stats["input_lra"],
            post_stats["input_i"], post_stats["input_tp"], post_stats["input_lra"]
        )

        # artifacts güncelle — LipSyncStep'in beklediği anahtar
        artifacts["mixed_wav"] = str(perfect_norm)
        artifacts["perfect_mixed_raw_wav"] = str(perfect_raw)
        audio_info.setdefault("loudness", {})
        audio_info["loudness"]["pre"] = pre_stats
        audio_info["loudness"]["post"] = post_stats
        audio_info["loudness"]["targets"] = {"I": I, "TP": TP, "LRA": LRA}
        flags["perfect_mix"] = True

        # Not: pan_amount bu stepte kullanılmıyor (müzik/ses pan işlemleri gerekiyorsa MixStep'e özgü kalır).

