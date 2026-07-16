#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

SUMMARY_PORT="${SUMMARY_PORT:-8065}"
MAIN_MODEL_PORT="${MAIN_MODEL_PORT:-7060}"
ADAPTER_PORT="${ADAPTER_PORT:-8070}"
if [[ -n "${SAFEVL_LOG_DIR:-}" ]]; then
  LOG_DIR="${LOG_DIR:-${SAFEVL_LOG_DIR}/webinfer}"
else
  LOG_DIR="${LOG_DIR:-${SERVICE_DIR}/summary_vllm_logs}"
fi
GRACE_SECONDS="${GRACE_SECONDS:-10}"

usage() {
  cat <<EOF
Usage:
  bash scripts/stop.sh all             Stop adapter, main model services, and summary service.
  bash scripts/stop.sh summary         Stop the summary vLLM service.
  bash scripts/stop.sh models          Stop the main model vLLM services.
  bash scripts/stop.sh adapter         Stop the web inference adapter.

Common overrides:
  SUMMARY_PORT=8065 MAIN_MODEL_PORT=7060 ADAPTER_PORT=8070
  LOG_DIR=${SERVICE_DIR}/summary_vllm_logs
  GRACE_SECONDS=10
EOF
}

is_pid_running() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

add_pid() {
  local pid="$1"
  local existing

  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  [[ "${pid}" != "$$" ]] || return 0

  for existing in "${PIDS_TO_KILL[@]:-}"; do
    [[ "${existing}" != "${pid}" ]] || return 0
  done

  PIDS_TO_KILL+=("${pid}")
}

add_descendants() {
  local parent="$1"
  local child

  command -v pgrep >/dev/null 2>&1 || return 0

  while IFS= read -r child; do
    add_pid "${child}"
    add_descendants "${child}"
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
}

add_pids_on_port() {
  local port="$1"
  local pid

  if command -v lsof >/dev/null 2>&1; then
    while IFS= read -r pid; do
      add_pid "${pid}"
    done < <(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)
  elif command -v fuser >/dev/null 2>&1; then
    for pid in $(fuser -n tcp "${port}" 2>/dev/null || true); do
      add_pid "${pid}"
    done
  fi
}

add_pids_by_pattern() {
  local pattern="$1"
  local pid

  command -v pgrep >/dev/null 2>&1 || return 0

  while IFS= read -r pid; do
    add_pid "${pid}"
  done < <(pgrep -f -- "${pattern}" 2>/dev/null || true)
}

kill_collected_pids() {
  local label="$1"
  local pid
  local deadline
  local still_running=()

  if [[ ${#PIDS_TO_KILL[@]} -eq 0 ]]; then
    echo "${label}: no matching processes."
    return 0
  fi

  for pid in "${PIDS_TO_KILL[@]}"; do
    add_descendants "${pid}"
  done

  echo "${label}: stopping PIDs ${PIDS_TO_KILL[*]}"
  kill "${PIDS_TO_KILL[@]}" 2>/dev/null || true

  deadline=$((SECONDS + GRACE_SECONDS))
  while (( SECONDS < deadline )); do
    still_running=()
    for pid in "${PIDS_TO_KILL[@]}"; do
      if is_pid_running "${pid}"; then
        still_running+=("${pid}")
      fi
    done

    if [[ ${#still_running[@]} -eq 0 ]]; then
      echo "${label}: stopped."
      return 0
    fi

    sleep 1
  done

  still_running=()
  for pid in "${PIDS_TO_KILL[@]}"; do
    if is_pid_running "${pid}"; then
      still_running+=("${pid}")
    fi
  done

  if [[ ${#still_running[@]} -gt 0 ]]; then
    echo "${label}: forcing PIDs ${still_running[*]}"
    kill -9 "${still_running[@]}" 2>/dev/null || true
  fi
}

stop_summary() {
  local pid_file="${LOG_DIR}/vllm_${SUMMARY_PORT}.pid"
  local pid=""
  PIDS_TO_KILL=()

  if [[ -f "${pid_file}" ]]; then
    pid="$(tr -cd '0-9' < "${pid_file}" || true)"
    if is_pid_running "${pid}"; then
      add_pid "${pid}"
    else
      echo "summary: stale pid file ${pid_file}"
    fi
  fi

  add_pids_on_port "${SUMMARY_PORT}"
  add_pids_by_pattern "vllm.entrypoints.openai.api_server.*--port ${SUMMARY_PORT}"
  kill_collected_pids "summary"

  if [[ -f "${pid_file}" ]] && ! is_pid_running "${pid:-}"; then
    rm -f "${pid_file}"
  fi
}

stop_models() {
  PIDS_TO_KILL=()

  add_pids_on_port "${MAIN_MODEL_PORT}"
  add_pids_by_pattern "vllm.entrypoints.openai.api_server.*--port ${MAIN_MODEL_PORT}"
  add_pids_by_pattern "${SCRIPT_DIR}/start_model.sh"
  add_pids_by_pattern "${SCRIPT_DIR}/start_all_models.sh"
  kill_collected_pids "models"
}

stop_adapter() {
  PIDS_TO_KILL=()

  add_pids_on_port "${ADAPTER_PORT}"
  add_pids_by_pattern "live_adapter.py.*--port ${ADAPTER_PORT}"
  add_pids_by_pattern "${SCRIPT_DIR}/start_adapter.sh"
  kill_collected_pids "adapter"
}

stop_all() {
  stop_adapter
  stop_models
  stop_summary
}

case "${ACTION}" in
  all)
    stop_all
    ;;
  summary)
    stop_summary
    ;;
  model|models)
    stop_models
    ;;
  adapter)
    stop_adapter
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
