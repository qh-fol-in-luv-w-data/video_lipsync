# ============================================================
# LatentSync 1.5 — Audio → Lip-sync Video
# Base: CUDA 12.1 + cuDNN 8 + Ubuntu 22.04
# ============================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git wget curl ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch CUDA 12.1 ────────────────────────────────────────
RUN pip install --no-cache-dir \
        torch==2.3.1+cu121 \
        torchvision==0.18.1+cu121 \
        torchaudio==2.3.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# ── Inference deps ───────────────────────────────────────────
COPY requirements-lipsync.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-lipsync.txt

# ── App ──────────────────────────────────────────────────────
WORKDIR /app
COPY lipsync.py     ./
COPY entrypoint.sh  /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p inputs/images inputs/audios outputs

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
