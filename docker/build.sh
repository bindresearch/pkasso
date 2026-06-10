#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-pkasso-webserver:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! docker info >/dev/null 2>&1; then
  printf 'Docker is not reachable from this shell.\n' >&2
  printf 'Current user/groups: ' >&2
  id >&2
  printf 'Docker host: %s\n' "${DOCKER_HOST:-<default>}" >&2
  printf 'Docker socket: ' >&2
  ls -l /var/run/docker.sock >&2
  exit 1
fi

docker build \
  --no-cache \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "${IMAGE_NAME}" \
  "${REPO_ROOT}"
