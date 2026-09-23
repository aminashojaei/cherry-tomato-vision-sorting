# syntax=docker/dockerfile:1

FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker-cpu.txt ./
RUN python -m pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-docker-cpu.txt

COPY classifier/ classifier/
COPY config/ config/
COPY core/ core/
COPY crop/ crop/
COPY detector/ detector/
COPY geometry/ geometry/
COPY models/ models/
COPY pipeline/ pipeline/
COPY tracker/ tracker/
COPY utils/ utils/
COPY visualization/ visualization/
COPY run_pipeline.py ./

RUN sha256sum -c models/checksums.sha256

FROM base AS test

COPY Dockerfile compose.yaml ./
COPY tests/ tests/

RUN python -c "import cv2, lap, numpy, torch, torchvision, ultralytics, yaml" \
    && python -m unittest discover -s tests -v \
    && touch /tmp/tests-passed

FROM base AS runtime

# This copy makes the runtime stage depend on a successful test stage.
COPY --from=test /tmp/tests-passed /tmp/tests-passed

RUN mkdir -p /data/input /data/output

ENTRYPOINT ["python", "/app/run_pipeline.py"]
CMD ["--help"]
