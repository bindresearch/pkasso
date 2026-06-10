#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-pkasso-webserver:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DOCKER_BUILDKIT=1

if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  printf 'SSH_AUTH_SOCK is not set. Start/load your SSH agent before building.\n' >&2
  printf 'Try: eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519\n' >&2
  exit 1
fi

if ! ssh-add -l >/dev/null 2>&1; then
  printf 'No SSH keys are loaded in the active SSH agent.\n' >&2
  printf 'Load your GitHub key, for example: ssh-add ~/.ssh/id_ed25519\n' >&2
  exit 1
fi

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
  --ssh "default=${SSH_AUTH_SOCK}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "${IMAGE_NAME}" \
  --no-cache \
  "${REPO_ROOT}"
