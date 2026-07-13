#!/bin/bash
MINI_CONDA_ROOT="/mnt/shared-storage-user/liuyicong/miniconda3"
CONDA_SH="${MINI_CONDA_ROOT}/etc/profile.d/conda.sh"
# 判断conda初始化脚本是否存在
if \[ ! -f "${CONDA_SH}" \]; then
    echo "ERROR: conda.sh 文件不存在：${CONDA_SH}" >&2
    echo "请执行 find ${MINI_CONDA_ROOT} -name conda.sh 查找真实路径" >&2
    exit 1
fi
# 加载conda环境
source "${CONDA_SH}"
# 激活jd环境
conda activate jd
if \[ $? -ne 0 \]; then
    echo "ERROR: conda环境 jd 激活失败，请执行 conda env list 确认环境存在" >&2
    exit 1
fi
MAIN_GPU="${MAIN_GPU:-0}"
IFS=',' read -ra GPU_LIST <<< "${MAIN_GPU}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${#GPU_LIST[@]}}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-${DATA_PARALLEL_SIZE}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--jd-opensource--JoyAI-VL-Interaction-Preview}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-JoyAI-VL-Interaction-Preview}"
MAIN_MODEL_PORT="${MAIN_MODEL_PORT:-7060}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAIN_GPU_MEMORY_UTILIZATION="${MAIN_GPU_MEMORY_UTILIZATION:-0.75}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${MODEL_PATH}" ]] || [[ -z "$(find "${MODEL_PATH}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "主模型目录不存在或为空: ${MODEL_PATH}" >&2
    echo "请先从仓库根目录运行: ./install/download-models.sh --all" >&2
    exit 1
fi

echo "============================================================"
echo "Starting Main VLM Model (vLLM OpenAI API Server)"
echo "  Model: ${MODEL_PATH}"
echo "  Served model name: ${SERVED_MODEL_NAME}"
echo "  Port:  ${MAIN_MODEL_PORT}"
echo "  GPU:   ${MAIN_GPU}"
echo "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  Data parallel size: ${DATA_PARALLEL_SIZE}"
echo "  Local data parallel size: ${DATA_PARALLEL_SIZE_LOCAL}"
echo "  Max model len: ${MAX_MODEL_LEN}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${MAIN_GPU}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${MAIN_MODEL_PORT}" \
    --gpu-memory-utilization "${MAIN_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --allowed-local-media-path /mnt/shared-storage-user/liuyicong/JoyAI-VL-Interaction/
    


# curl -N --location \
#   'http://s-20260713094852-f6xk7.ailab-evobox.pjh-service.org.cn/v1/chat/completions' \
#   --header 'Content-Type: application/json' \
#   --data '{
#     "model": "JoyAI-VL-Interaction-Preview",
#     "temperature": 1.0,
#     "stream": false,
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           {
#             "type": "video_url",
#             "video_url": {
#               "url": "file:///mnt/shared-storage-user/liuyicong/JoyAI-VL-Interaction/videos/example.mp4"
#             }
#           },
#           {
#             "type": "text",
#             "text": "请描述这个视频"
#           }
#         ]
#       }
#     ]
#   }'