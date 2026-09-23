# Docker CPU workflow

The Docker release targets Linux x86-64 and runs inference on CPU. Docker Desktop can run the same image on Windows and macOS; Apple Silicon hosts use x86-64 emulation.

## Prerequisites

- Docker Engine with the Compose plugin, or Docker Desktop
- At least 8 GB of free disk space for the build cache and image
- One or more supported videos in `inputs/`

## Build

```bash
docker compose build
```

The multi-stage build installs pinned CPU dependencies, verifies both model SHA-256 hashes, and executes the complete unit-test suite. The runtime stage copies a marker from the test stage, so a failed test prevents the final image from being created.

## Run all input videos

```bash
mkdir -p inputs outputs
docker compose run --rm tomato-sorting
```

Place videos in `inputs/` before running the command. Results are written to `outputs/` on the host.

## Smoke test one video

Linux and macOS:

```bash
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/inputs:/data/input:ro" \
  -v "$(pwd)/outputs:/data/output" \
  cherry-tomato-vision-sorting:cpu \
  --config /app/config/config.yaml \
  --input /data/input/video_01.mp4 \
  --output-dir /data/output/smoke-test \
  --device cpu \
  --max-frames 120
```

Windows PowerShell:

```powershell
docker run --rm --platform linux/amd64 `
  -v "${PWD}/inputs:/data/input:ro" `
  -v "${PWD}/outputs:/data/output" `
  cherry-tomato-vision-sorting:cpu `
  --config /app/config/config.yaml `
  --input /data/input/video_01.mp4 `
  --output-dir /data/output/smoke-test `
  --device cpu `
  --max-frames 120
```

## Useful checks

```bash
docker compose config
docker build --platform linux/amd64 --target test -t cherry-tomato-vision-sorting:test .
docker run --rm --platform linux/amd64 cherry-tomato-vision-sorting:cpu --help
```

The image contains the runtime source, configuration, release checkpoints, and their metadata. It excludes notebooks, documentation, demo media, development caches, and local inputs or outputs.
