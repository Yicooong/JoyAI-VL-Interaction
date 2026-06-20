#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICES_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -z "${VENV_ACTIVATE+x}" ]; then
  if [ -f "${SERVICES_DIR}/.venv/bin/activate" ]; then
    VENV_ACTIVATE="${SERVICES_DIR}/.venv/bin/activate"
  else
    VENV_ACTIVATE=".venv/bin/activate"
  fi
else
  VENV_ACTIVATE="${VENV_ACTIVATE:-}"
fi

if [ -f "${VENV_ACTIVATE}" ]; then
  source "${VENV_ACTIVATE}"
elif [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  echo "Virtual environment not found: ${VENV_ACTIVATE}" >&2
  echo "Set VENV_ACTIVATE or activate an environment before running this script." >&2
  exit 1
fi

if [ ! -f cert.pem ] || [ ! -f key.pem ]; then
  echo "SSL certificate not found, generating cert.pem/key.pem..."
  bash "${SCRIPT_DIR}/generate_cert.sh"
fi

PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m joy_interaction_webui.server \
  --ssl-cert cert.pem \
  --ssl-key key.pem \
  --host 0.0.0.0 \
  --port 8099 \
  --model streaming-infer-adapter \
  --api-base http://127.0.0.1:8070/v1 \
  "$@"
