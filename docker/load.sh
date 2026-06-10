#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-pkasso-webserver.tar}"

docker load --input "${INPUT}"
