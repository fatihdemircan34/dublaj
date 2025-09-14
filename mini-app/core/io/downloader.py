from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve


def download(input_path: str, output_path: str, ydl_format: Optional[str] = None) -> str:
    """Download ``input_path`` to ``output_path``.

    The helper understands a couple of different ``input_path`` formats:

    * Local file paths – simply copied to ``output_path``.
    * HTTP/HTTPS URLs – downloaded using :func:`urllib.request.urlretrieve`.
    * YouTube links – downloaded with ``yt_dlp`` if available.
    * Cloud bucket URLs (``gs://`` or ``s3://``) – fetched via ``smart_open``
      if the library is installed.

    The additional cases are implemented lazily so that unit tests, which
    only rely on local file copying, do not require extra dependencies.
    """

    parsed = urlparse(input_path)

    # YouTube links require special handling
    if "youtube.com" in input_path or "youtu.be" in input_path:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:  # pragma: no cover - import failure path
            raise RuntimeError("yt_dlp is required to download YouTube links") from exc

        ydl_opts = {"outtmpl": output_path, "merge_output_format": "mp4"}
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(input_path, download=True)
            resolved = ydl.prepare_filename(info)
        return resolved

    # Generic HTTP/HTTPS download
    if parsed.scheme in {"http", "https"}:
        urlretrieve(input_path, output_path)
        return output_path

    # Cloud bucket (gs:// or s3://)
    if parsed.scheme in {"gs", "s3"}:
        try:
            from smart_open import open as smart_open
        except ImportError as exc:  # pragma: no cover - import failure path
            raise RuntimeError("smart_open is required to download bucket links") from exc

        with smart_open(input_path, "rb") as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return output_path

    # Fallback to local file copy
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return str(dst)
