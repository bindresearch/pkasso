#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-pkasso-webserver:latest}"
PORT="${PORT:-8001}"

docker run --rm -it \
  --env "PKASSO_ROOT_PATH=${PKASSO_ROOT_PATH:-}" \
  --publish "${PORT}:8001" \
  "${IMAGE_NAME}"
