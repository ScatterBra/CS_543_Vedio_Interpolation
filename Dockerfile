FROM python:3.12-slim-bookworm

WORKDIR /work

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements-cu128.txt .
RUN python -m pip install --no-cache-dir -r requirements-cu128.txt \
    && python -m pip check