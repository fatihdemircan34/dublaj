# NVIDIA CUDA base for GPU (Cloud Run)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Optional: reduce torch CUDA builds size via pip wheels cache hints
ENV PIP_NO_CACHE_DIR=1
RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY pyproject.toml /app/
COPY dubbing_ai /app/dubbing_ai
COPY configs /app/configs

# Install base deps
RUN pip install -e .

# For NeMo (optional): uncomment if you want NeMo in the image
# RUN pip install -e .[nemo]

# HuggingFace token will be provided at runtime as env HF_TOKEN
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "-m", "dubbing_ai.runner"]
