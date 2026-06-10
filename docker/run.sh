#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-pkasso-webserver:latest}"
PORT="${PORT:-8001}"

docker run --rm -it \
  --publish "${PORT}:8001" \
  "${IMAGE_NAME}"
