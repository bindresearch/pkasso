#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-pkasso-webserver:latest}"
PORT="${PORT:-8001}"
GPU_ARGS="${GPU_ARGS:-}"

GPU_OPTS=()
if [[ -n "${GPU_ARGS}" ]]; then
  read -r -a GPU_OPTS <<< "${GPU_ARGS}"
fi

docker run --rm -it \
  "${GPU_OPTS[@]}" \
  --publish "${PORT}:8001" \
  "${IMAGE_NAME}"
