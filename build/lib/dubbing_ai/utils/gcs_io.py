from __future__ import annotations
import os
from typing import Optional
from google.cloud import storage

def is_gcs_uri(uri: str) -> bool:
    return uri.startswith("gs://")

def split_gcs_uri(uri: str):
    assert uri.startswith("gs://")
    p = uri[5:]
    bucket, _, blob = p.partition("/")
    return bucket, blob

def download_to_local(uri: str, local_path: str):
    if not is_gcs_uri(uri):
        raise ValueError("Not a GCS URI")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client = storage.Client()
    bucket_name, blob_name = split_gcs_uri(uri)
    b = client.bucket(bucket_name)
    bl = b.blob(blob_name)
    bl.download_to_filename(local_path)

def upload_file(local_path: str, gcs_uri: str):
    if not is_gcs_uri(gcs_uri):
        raise ValueError("Not a GCS URI")
    client = storage.Client()
    bucket_name, blob_name = split_gcs_uri(gcs_uri)
    b = client.bucket(bucket_name)
    bl = b.blob(blob_name)
    bl.upload_from_filename(local_path)

def ensure_local_or_download(uri_or_path: str, tmp_dir: str) -> str:
    if is_gcs_uri(uri_or_path):
        filename = os.path.basename(uri_or_path)
        local = os.path.join(tmp_dir, filename)
        download_to_local(uri_or_path, local)
        return local
    return uri_or_path

def maybe_upload(local_path: str, out_prefix: str) -> str:
    if is_gcs_uri(out_prefix):
        gcs_uri = out_prefix.rstrip("/") + "/" + os.path.basename(local_path)
        upload_file(local_path, gcs_uri)
        return gcs_uri
    # write to local dir
    os.makedirs(out_prefix, exist_ok=True)
    target = os.path.join(out_prefix, os.path.basename(local_path))
    if os.path.abspath(local_path) != os.path.abspath(target):
        import shutil
        shutil.copy2(local_path, target)
    return target
