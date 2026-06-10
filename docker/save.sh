#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-pkasso-webserver:latest}"
OUTPUT="${OUTPUT:-pkasso-webserver.tar}"

docker save "${IMAGE_NAME}" --output "${OUTPUT}"

printf 'Saved %s to %s\n' "${IMAGE_NAME}" "${OUTPUT}"
