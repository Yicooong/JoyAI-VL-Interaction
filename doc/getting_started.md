# Getting Started

> 中文文档: [getting_started.zh-CN.md](getting_started.zh-CN.md)

## Prerequisites

- Linux with NVIDIA GPUs (tested on A100/H100)
- CUDA 12.x + NVIDIA driver 535+
- Python 3.12 (recommended)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip for Python package management

## Installation

Use the provided install scripts to set up all dependencies:

```bash
# Install core dependencies (webinfer + webui)
./install/install.sh --with-all

# Install ASR/TTS runtime (optional)
./install/install-audio-runtime.sh --all

# Download all model weights
./install/download-models.sh --all
```

For compatibility notes across platforms, see `install/README.md`.

`install/` is only for dependency setup, model downloads, and generated configuration.
Runtime entrypoints live under `services/`: use `services/scripts/run.sh` for orchestration, or
`services/<service>/scripts/` for component-level commands.

## Model Downloads

Download model weights before starting:

```bash
# All models: main interaction + summary + ASR + TTS
./install/download-models.sh --all
```

Default model paths:

| Model | Default Path | HuggingFace Repo |
|-------|-------------|------------------|
| Main interaction model | `/tmp/models/JoyAI-VL-Interaction-Preview` | `ydydy/JoyAI-VL-Interaction-Preview` |
| Summary model | `/tmp/models/Qwen3-VL-4B-Instruct` | `Qwen/Qwen3-VL-4B-Instruct` |
| ASR model (optional) | `/tmp/models/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-1.7B` |
| TTS model (optional) | `/tmp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |

## Minimal Setup (2 services)

Start only the core services — video inference and the web UI:

```bash
# Start webinfer in the background, then WebUI in the foreground
./services/scripts/run.sh minimal
```

`services/scripts/run.sh minimal` starts `webinfer` in the background and keeps WebUI in the foreground.
When using `services/scripts/run.sh`, startup is not complete until the final WebUI process has started.
Press `Ctrl+C` in that terminal to stop the services started by the orchestrator.

Open your browser at: `https://127.0.0.1:8099`

## Full Setup (all services)

Optional services must start **before** the web UI.

### Recommended startup order

```text
1. services/webinfer          (required)
2. ASR                        (optional — enables voice input)
3. TTS                        (optional — enables voice output)
4. background-agent           (optional — enables task delegation)
5. services/webui             (required — start last)
```

### Step-by-step

Start the full service set with one command:

```bash
./services/scripts/run.sh all
```

`services/scripts/run.sh all` starts optional services before WebUI. Use `START_ASR=0`,
`START_TTS=0`, or `START_BACKGROUND_AGENT=0` to skip optional services.
When using this orchestrator, the full service set is not ready until the final WebUI process has started.
Press `Ctrl+C` in that terminal to stop the services started by the orchestrator.

You can also start the services manually in this order:

```bash
# 1. Inference backend
(cd services/webinfer && bash scripts/run.sh all)

# 2. ASR (optional)
./services/asr/scripts/run.sh all

# 3. TTS (optional)
./services/tts/scripts/run.sh all

# 4. Background Agent (optional)
./services/background-agent/scripts/run.sh

# 5. Web UI (start last)
source services/.venv/bin/activate
(cd services/webui && bash scripts/start_server.sh)
```

## Health Checks

After startup, verify each service is running:

```bash
curl http://127.0.0.1:8070/health   # webinfer
curl http://127.0.0.1:8994/health   # ASR (optional)
curl http://127.0.0.1:8992/health   # TTS (optional)
curl http://127.0.0.1:8079/health   # background-agent (optional)
```

The web UI is accessible at `https://127.0.0.1:8099` (accept the self-signed certificate warning).

## RTSP Local Stream Testing

The WebUI can use either a webcam or an RTSP input. To test RTSP without a physical IP camera, you can run a local MediaMTX server and push a local video file with `ffmpeg`.

After the local stream is running, enter an RTSP URL such as `rtsp://127.0.0.1:8554/fire1` in the WebUI RTSP input. If WebUI is running on another machine, replace `127.0.0.1` with the machine that runs MediaMTX.

See the [RTSP Local Streaming Guide](rtsp_streaming.md) for MediaMTX download notes, helper script examples, and troubleshooting checks.

## Stopping Services

If you started with `services/scripts/run.sh minimal` or `services/scripts/run.sh all`, press `Ctrl+C`
in that terminal. To stop services from another terminal:

```bash
./services/scripts/stop.sh all
```

## Configuration

### webinfer

Key environment variables (set in start scripts or export before launch):

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON_BIN` | `python` | Python binary for vLLM and adapter |
| `VENV_ACTIVATE` | auto-detects `services/.venv` | Optional venv activate script path; set `VENV_ACTIVATE=` to use the current shell environment |
| `MODEL_PATH` | `/tmp/models/JoyAI-VL-Interaction-Preview` | Main model local path |
| `SUMMARY_MODEL_PATH` | `/tmp/models/Qwen3-VL-4B-Instruct` | Summary model local path |
| `MAIN_GPU` | `0` | Single physical GPU used by the streaming model service |
| `SUMMARY_GPU` | `1` | Single physical GPU used by the summary model service |
| `ADAPTER_PORT` | `8070` | Adapter listen port |
| `CHUNK` | `100` | Frames per memory chunk |
| `COMPRESS_EVERY_N_CHUNKS` | `5` | Chunks before long-term compression |
| `MAIN_MAX_TOKENS` | `256` | Max tokens for main model output |
| `MAIN_TEMPERATURE` | `0.8` | Sampling temperature |
| `FORCE_SILENCE_BEFORE_QUERY` | `true` | Suppress output when no user query |

### ASR

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_UPSTREAM_URL` | `http://127.0.0.1:8993/v1/audio/transcriptions` | vLLM endpoint |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Model name |
| `ASR_ADAPTER_PORT` | `8994` | Adapter listen port |
| `ASR_GPU` | `2` | Single physical GPU used by the ASR model service |
| `ASR_GPU_MEMORY_UTILIZATION` | `0.3` | vLLM GPU memory utilization cap |

### TTS

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_UPSTREAM_URL` | `ws://127.0.0.1:8991/v1/audio/speech/stream` | vLLM-Omni endpoint |
| `TTS_DEFAULT_VOICE` | `vivian` | Default speaker voice |
| `TTS_ADAPTER_PORT` | `8992` | Adapter listen port |
| `TTS_GPU` | `2` | Single physical GPU used by the TTS model service |
| `TTS_DEPLOY_CONFIG` | `services/tts/config/qwen3_tts_lowmem.yaml` | Short-reply vLLM-Omni deploy config with total TTS memory budget `0.6` |

## Troubleshooting

**Q: `webinfer` returns 502 errors**
A: The vLLM backend on port 7060 or 8065 is not ready. Wait for the model to finish loading (check logs) or verify with `curl http://127.0.0.1:7060/v1/models`.

**Q: Model never speaks (always returns `</silence>`)**
A: By default, `FORCE_SILENCE_BEFORE_QUERY=true` suppresses output when there is no user query. Either send a prompt or set this to `false`.

**Q: Browser cannot access the webcam**
A: WebRTC requires HTTPS. Ensure you access via `https://` and accept the self-signed certificate.

**Q: ASR/TTS not working after starting WebUI**
A: Optional services must be started before WebUI. Restart WebUI after launching ASR/TTS.

**Q: `ASR failed: Cannot connect to 127.0.0.1:8994`**
A: ASR adapter is not running. Start it with `./services/asr/scripts/run.sh all`, then verify `curl http://127.0.0.1:8994/health`.

**Q: ASR/TTS keeps printing `upstream ... is not ready`**
A: The model service is not ready on `8993` (ASR) or `8991` (TTS). Run `./install/download-models.sh --all`, then restart ASR/TTS and check `curl http://127.0.0.1:8993/v1/models` or `curl http://127.0.0.1:8991/v1/models`.

**Q: Out of GPU memory**
A: Check GPU allocation. By default, the streaming model uses GPU 0, the summary model uses GPU 1, and ASR/TTS both default to GPU 2. ASR uses `gpu_memory_utilization=0.3`; TTS uses the deploy config under `services/tts/config/` with a total TTS memory budget of `0.6`. Adjust the GPU and deploy config environment variables for your hardware.

**Q: `file://` images rejected by webinfer**
A: Configure `ALLOWED_LOCAL_IMAGE_ROOTS` to include your frame directory.
