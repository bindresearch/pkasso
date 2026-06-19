#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-pkasso-webserver:latest}"
PORT="${PORT:-8001}"

docker run --rm -it \
  --env "PKASSO_ROOT_PATH=${PKASSO_ROOT_PATH:-}" \
  --env "PKASSO_FORWARDED_ALLOW_IPS=${PKASSO_FORWARDED_ALLOW_IPS:-${FORWARDED_ALLOW_IPS:-}}" \
  --publish "${PORT}:8001" \
  "${IMAGE_NAME}"
