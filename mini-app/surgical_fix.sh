#!/usr/bin/env bash
set -euo pipefail
FILE="segment_problem.py"

# 1) _trim_audio_to_video_length yoksa ekle (probe_duration_seconds'un hemen arkasına)
if ! grep -q "_trim_audio_to_video_length" "$FILE"; then
  awk '
  /def probe_duration_seconds\(path: Path\) -> float:/{flag=1}
  {print $0}
  flag==1 && /^\s*return 0\.0\s*$/ && !done {
    print ""
    print "def _trim_audio_to_video_length(audio_in: Path, video_in: Path, audio_out: Path, safety_ms: int = 10) -> Path:"
    print "    vid_dur = probe_duration_seconds(video_in)"
    print "    if vid_dur <= 0:"
    print "        _run([\"ffmpeg\",\"-y\",\"-i\",str(audio_in),\"-c\",\"copy\",str(audio_out)])"
    print "        return audio_out"
    print "    target = max(0.0, vid_dur - (safety_ms/1000.0))"
    print "    _run([\"ffmpeg\",\"-y\",\"-i\",str(audio_in),\"-af\", f\"atrim=0:{target:.6f},asetpts=N/SR/TB\", \"-ar\",\"48000\",\"-ac\",\"2\",\"-c:a\",\"pcm_s16le\", str(audio_out)])"
    print "    return audio_out"
    print ""
    done=1
  }' "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
fi

# 2) _mix_music_and_dub: duration=longest -> shortest + fade, final_raw + trim
perl -0777 -pe '
  s/amix=inputs=2:normalize=0:duration=longest/mix_fix_marker/g;
  s/final = out_dir \/ f"{video_in\.stem}\.final_mix\.48k\.wav"/final_raw = out_dir \/ f"{video_in.stem}.final_mix.48k.wav"/;
  s/mix_fix_marker/amix=inputs=2:normalize=0:duration=shortest,afade=t=out:st=0:d=0.01/;
' -i "$FILE"

# Trim’i ekle (miks _run sonrası)
perl -0777 -pe '
  s/(_run\(cmd\)\s*\)\s*\n\s*if dbg: dbg\.snap\("DUB_MIX_DONE".*?return )(final)(, \(.+?\)\))/\1final, \3/g;
' -i "$FILE" # no-op guard

perl -0777 -pe '
  s/_run\(cmd\)\s*\)\s*\n(\s*)if dbg: dbg\.snap\("DUB_MIX_DONE",[^\n]*\)\s*\n(\s*)return final, \(music_bed if separated else None\)/
  _run(cmd)
  \1final = out_dir \/ f"{video_in.stem}.final_mix.48k.trimmed.wav"
  \1_trim_audio_to_video_length(final_raw, video_in, final, safety_ms=10)
  \1if dbg: dbg.snap("DUB_MIX_DONE", final=str(final), separated_music=bool(separated))
  \1return final, (music_bed if separated else None)
/sx' -i "$FILE"

# 3) _mux_audio_to_video: -shortest +faststart
perl -0777 -pe '
  s/_run\(\[(\s*)"ffmpeg","-y",\s*"-i",\s*str\(video_in\),\s*"-i",\s*str\(audio_in\),\s*"-map","0:v:0","\s*-map","1:a:0","\s*-c:v","copy","\s*-c:a","aac","\s*-b:a","192k",\s*str\(video_out\)\s*\]\)/
  _run(["ffmpeg","-y","-i",str(video_in),"-i",str(audio_in),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest",str(video_out)])
  /sx' -i "$FILE"

# 4) lipsync_or_mux fallback’te mux öncesi trim
perl -0777 -pe '
  s/return ls_path, used_lipsync\s*\n\s*out_path = _mux_audio_to_video\(video_in, dub_audio_wav, muxed\)/
  return ls_path, used_lipsync
    safe_dir = out_dir \/ "_sync"
    safe_dir.mkdir(parents=True, exist_ok=True)
    trimmed_for_mux = safe_dir \/ "dub.trimmed.for_mux.wav"
    _trim_audio_to_video_length(dub_audio_wav, video_in, trimmed_for_mux, safety_ms=10)
    out_path = _mux_audio_to_video(video_in, trimmed_for_mux, muxed)
  /sx' -i "$FILE"

# 5) process_video_wordwise: lipsync/mux öncesi güvenli trim
perl -0777 -pe '
  s/(xtts_cfg=XTTSConfig\(model_name=xtts_model_name, language=tlang, speed=xtts_speed\)\s*\)\s*\)\s*\n\s*if do_lipsync:)/\1
        safe_dub = out \/ "dubbed.timeline.mono16k.safe.wav"
        _trim_audio_to_video_length(dub_audio_wav, src, safe_dub, safety_ms=10)
        dub_audio_wav = safe_dub
        if do_lipsync:/s' -i "$FILE"

echo "[ok] surgical fix uygulandı."
