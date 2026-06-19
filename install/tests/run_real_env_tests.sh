#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$INSTALL_DIR/tests/real_envs"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_BIN="${UV_BIN:-uv}"
MAX_SUBAGENTS="${MAX_SUBAGENTS:-12}"

run_case() {
  local name="$1"
  shift
  local case_dir="$TEST_ROOT/$name"
  local venv_dir="$case_dir/.venv"
  local install_log="$case_dir/install.log"
  local verify_log="$case_dir/verify.log"

  rm -rf "$case_dir"
  mkdir -p "$case_dir"

  echo "==> 安装 $name"
  VENV_DIR="$venv_dir" PYTHON_BIN="$PYTHON_BIN" UV_BIN="$UV_BIN" "$INSTALL_DIR/install.sh" "$@" \
    >"$install_log" 2>&1

  echo "==> 验证 $name"
  "$UV_BIN" pip check --python "$venv_dir/bin/python" >"$verify_log" 2>&1
  PATH="$venv_dir/bin:$PATH" "$venv_dir/bin/python" "$INSTALL_DIR/tests/verify_real_env.py" "$name" \
    >>"$verify_log" 2>&1
}

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "找不到 uv: $UV_BIN" >&2
  exit 1
fi

if ! "$UV_BIN" python find "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "uv 找不到 Python: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$TEST_ROOT"

run_case no_with
run_case with_asr --with-asr
run_case with_tts --with-tts
run_case with_background_agent --with-background-agent --max-subagents "$MAX_SUBAGENTS"
run_case with_all --with-all --max-subagents "$MAX_SUBAGENTS"

echo "全部真实环境测试通过。"
echo "日志目录: $TEST_ROOT"
