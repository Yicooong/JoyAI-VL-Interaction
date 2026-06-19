#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$INSTALL_DIR/.." && pwd)"
SERVICES_DIR="$REPO_ROOT/services"
ASR_DIR="$SERVICES_DIR/asr"
TTS_DIR="$SERVICES_DIR/tts"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_BIN="${UV_BIN:-uv}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$INSTALL_DIR/constraints.txt}"
INSTALL_ASR=0
INSTALL_TTS=0

usage() {
  cat <<'USAGE'
Usage: install-audio-runtime.sh [options]

按 services/asr/README.md 和 services/tts/README.md 安装真实 ASR/TTS 模型服务运行环境，统一使用 Python 3.12。

Options:
  --asr       安装 ASR vLLM nightly + ASR adapter 环境。
  --tts       安装 TTS vLLM Omni + TTS adapter 环境。
  --all       安装 ASR 和 TTS 环境。
  -h, --help  显示帮助。

Environment overrides:
  PYTHON_BIN=python3.12
  UV_BIN=/path/to/uv
  CONSTRAINTS_FILE=/path/to/constraints.txt  # 仅 --tts 使用
  ASR_VENV_DIR=/path/to/asr/.venv
  TTS_VENV_DIR=/path/to/tts/.venv
USAGE
}

die() {
  echo "错误: $*" >&2
  exit 1
}

require_dir() {
  [ -d "$1" ] || die "目录不存在: $1"
}

uv_pip_install() {
  local venv_dir="$1"
  shift
  "$UV_BIN" pip install --python "$venv_dir/bin/python" "$@"
}

install_asr() {
  local venv_dir="${ASR_VENV_DIR:-$ASR_DIR/.venv}"

  require_dir "$ASR_DIR"
  "$UV_BIN" venv -p "$PYTHON_BIN" --seed "$venv_dir"
  uv_pip_install "$venv_dir" -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
  uv_pip_install "$venv_dir" "vllm[audio]"
  uv_pip_install "$venv_dir" -e "$ASR_DIR[dev]"
}

install_tts() {
  local venv_dir="${TTS_VENV_DIR:-$TTS_DIR/.venv}"

  require_dir "$TTS_DIR"
  [ -f "$CONSTRAINTS_FILE" ] || die "约束文件不存在: $CONSTRAINTS_FILE"
  "$UV_BIN" venv -p "$PYTHON_BIN" --seed "$venv_dir"
  uv_pip_install "$venv_dir" -c "$CONSTRAINTS_FILE" "vllm==0.22.0" "vllm-omni==0.22.0" --torch-backend=auto
  uv_pip_install "$venv_dir" -c "$CONSTRAINTS_FILE" -e "$TTS_DIR[dev]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --asr)
      INSTALL_ASR=1
      ;;
    --tts)
      INSTALL_TTS=1
      ;;
    --all)
      INSTALL_ASR=1
      INSTALL_TTS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "未知选项: $1"
      ;;
  esac
  shift
done

if [ "$INSTALL_ASR" -eq 0 ] && [ "$INSTALL_TTS" -eq 0 ]; then
  usage >&2
  exit 2
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  die "找不到 uv: $UV_BIN"
fi

if ! "$UV_BIN" python find "$PYTHON_BIN" >/dev/null 2>&1; then
  die "uv 找不到 Python: $PYTHON_BIN"
fi

if [ "$INSTALL_ASR" -eq 1 ]; then
  install_asr
fi

if [ "$INSTALL_TTS" -eq 1 ]; then
  install_tts
fi

echo "音频运行环境安装完成。"
echo "ASR 环境: ${ASR_VENV_DIR:-$ASR_DIR/.venv}"
echo "TTS 环境: ${TTS_VENV_DIR:-$TTS_DIR/.venv}"
