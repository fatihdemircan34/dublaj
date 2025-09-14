from __future__ import annotations

from pathlib import Path

from core.pipeline.base import Context, Step
from core.io import downloader


class DownloadStep:
    name = "Download"

    def run(self, ctx: Context) -> None:
        input_path = ctx["config"]["input"]
        # Produce a deterministic MP4 filename so downstream steps don't have to resolve %(ext)s
        out = Path(ctx["temp_dir"]) / "input.mp4"
        # Optional override from config: config.yaml → yt_format: "<yt-dlp format string>"
        yt_format = None
        cfg = ctx.get("config") or {}
        if isinstance(cfg, dict):
            yt_format = cfg.get("yt_format")

        video_path = downloader.download(
            input_path,
            str(out),
            ydl_format=yt_format,
        )
        ctx.setdefault("artifacts", {})["video"] = video_path
