#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MEDIAMTX_BIN="${MEDIAMTX_BIN:-$REPO_ROOT/tools/mediamtx/mediamtx}"
MEDIAMTX_CONFIG="${MEDIAMTX_CONFIG:-$REPO_ROOT/tools/mediamtx/mediamtx.yml}"

exec "$MEDIAMTX_BIN" "$MEDIAMTX_CONFIG"