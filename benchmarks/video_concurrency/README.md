# 多路视频 VLM 性能测试
ffmpeg is required for video input
这个工具以固定视频时钟模拟 1/2/4/8 路视频，并请求 OpenAI-compatible
`/v1/chat/completions`。它既可直接测试 vLLM，也可测试 webinfer adapter。

## 依赖

- 项目环境中的 `openai` Python 包
- 视频输入需要 `ffmpeg`；也可以直接传入 JPEG/PNG 图片

视频会在正式计时前按 `--fps` 预抽帧，因此磁盘解码不计入请求延迟。

### 视频长度短于测试时长时

脚本会在测试开始前使用 ffmpeg 按 `--fps` 将整个视频预抽帧，并把抽出的 JPEG
保存在内存中。正式测试期间不会再次运行 ffmpeg，也不会实时解码视频。

如果视频长度小于 `--duration`，脚本会循环使用已经抽出的帧，直到该并发档位的测试
结束。例如，10 秒视频使用 `--fps 1 --duration 60` 时，大约会预抽取 10 帧，并按
下面的方式循环六次：

```text
测试时间（秒）:  0  1  2 ...  9 | 10 11 12 ... 19 | ... | 50 51 ... 59
使用视频帧:      0  1  2 ...  9 |  0  1  2 ...  9 | ... |  0  1 ...  9
```

实现上使用 `frames[frame_index % len(frames)]` 选择帧。因此：

- 视频内容会从头循环，内存中不会复制多份相同帧。
- `adapter` 模式下 API 请求的 stream session 不会重置，仍然是同一路连续视频流；
  `vllm` 模式不向服务端发送 session。
- 请求携带的逻辑时间戳不会随视频循环而归零，而是按照
  `frame_index / fps` 持续递增；上例中的时间戳为 0～59 秒。
- 模型可能观察到周期性重复的画面。如果不希望内容重复，应提供长度不小于
  `--duration` 的视频，或者将 `--duration` 设置为不超过最短测试视频的长度。

## 快速测试

```bash
python benchmarks/video_concurrency/benchmark.py \
  --target vllm \
  --api-base http://REMOTE_HOST:7060/v1 \
  --model JoyAI-VL-Interaction-Preview \
  --video videos/example.mp4 \
  --concurrency 1,2,4,8 \
  --fps 1 \
  --duration 600 \
  --output benchmarks/video_concurrency/results/remote-vllm
```

测试 webinfer 时需要使用 `--target adapter`，并将 `--api-base` 设置为 adapter 的
`/v1` 地址。工具会为每一路自动生成独立的 `x-streaming-session`，因此各路视频上下文
不会混合。

```bash
python benchmarks/video_concurrency/benchmark.py \
  --target adapter \
  --api-base http://ADAPTER_HOST:8070/v1 \
  --model JoyAI-VL-Interaction-Preview \
  --video videos/ \
  --prompt "请观察当前画面并决定是否需要回应。" \
  --concurrency 1,2,4,8
```

使用不同视频模拟不同用户：

```bash
python benchmarks/video_concurrency/benchmark.py \
  --api-base http://127.0.0.1:7060/v1 \
  --model JoyAI-VL-Interaction-Preview \
  --video videos/a.mp4 --video videos/b.mp4 \
  --video videos/c.mp4 --video videos/d.mp4 \
  --concurrency 1,2,4,8 --duration 600
```

视频数量少于并发数时会循环使用输入视频，但 session 仍然相互独立。

`--video` 也可以直接传入目录。脚本会递归查找目录中的视频，在每个并发档位随机
分配给各路 stream。候选视频足够时不会重复；候选数少于视频路数时，会先使用完所有
候选，再随机重复。每轮开始时终端会打印 `stream → 视频路径`，相同映射也会保存在
`summary.json` 的 `selected_videos` 中：

```bash
python benchmarks/video_concurrency/benchmark.py \
  --api-base http://127.0.0.1:7060/v1 \
  --model JoyAI-VL-Interaction-Preview \
  --video videos/ \
  --concurrency 1,2,4,8 \
  --seed 2026
```

## 参数说明

### 两种测试目标

`--target` 决定请求语义，而不只是给结果增加一个标签：

| 模式 | 请求内容 | 测量范围 |
| --- | --- | --- |
| `--target vllm` | 标准 OpenAI `messages`、图片和生成参数；不发送 `x-streaming-session` 或 `frame_time_range`。 | 原始 vLLM 图文推理服务，包括客户端网络、vLLM 排队和模型推理。每帧是相互独立的请求。 |
| `--target adapter` | 在标准请求之外，发送每路唯一的 `x-streaming-session` 和持续递增的 `frame_time_range`。 | webinfer adapter 完整链路，包括 session 锁、视频上下文、chunk、记忆/摘要、主模型推理和后处理。 |

两种模式应分别使用不同输出目录，避免结果文件相互覆盖，例如
`results/remote-vllm` 和 `results/remote-adapter`。

### 服务连接参数

| 参数 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | 否 | `vllm` | 测试目标。`vllm` 只发送标准 OpenAI 图文请求，测试原始模型服务；`adapter` 额外发送每路独立 session 和连续帧时间戳，测试 webinfer 的有状态完整链路。 |
| `--streaming` | 否 | `auto` | Token timing 模式。`auto` 对原生 vLLM 启用 SSE、对 adapter 保持非流式；`on` 强制使用 SSE；`off` 只测完整响应延迟。当前 adapter 不提供 SSE，因此 `--target adapter --streaming on` 会被拒绝。 |
| `--api-base` | 是 | 无 | OpenAI-compatible API 的基础地址，必须包含 `/v1`。`--target vllm` 应填写原生 vLLM 地址（常见端口 7060）；`--target adapter` 应填写 webinfer adapter 地址（默认端口 8070）。 |
| `--vllm-metrics` | 否 | 无 | 后端 vLLM 的 Prometheus `/metrics` 完整地址。设置后，每个并发档位会在测试前后各抓取一次并计算本轮 Histogram 增量；适用于通过 adapter 测量后端 TTFT、ITL、E2E、Prefill 和 Decode。 |
| `--api-key` | 否 | `EMPTY` | API 密钥。本地 vLLM 通常不校验密钥，可以保留默认值；远程服务启用鉴权时传入真实密钥。该值不会写入结果文件。 |
| `--model` | 是 | 无 | 请求体中的模型名称，必须与服务启动时的 `--served-model-name` 一致，例如 `JoyAI-VL-Interaction-Preview`。 |

### 视频和负载参数

| 参数 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--video` | 是 | 无 | 输入视频、图片或目录路径，可多次指定。目录会被递归搜索；支持 MP4、MOV、MKV、AVI、WebM、M4V、MPEG、MPG、JPEG、PNG 和 WebP。候选素材足够时每路不重复，不足时才随机重复。 |
| `--concurrency` | 否 | `1,2,4,8` | 要依次测试的视频路数，使用英文逗号分隔。每个数值会形成一轮独立测试。例如 `1,2,4,8` 会顺序运行四轮，而不是同时运行 15 路。 |
| `--fps` | 否 | `1.0` | 每一路视频每秒调度的帧数，同时也是 ffmpeg 预抽帧频率。`1` 表示每路每秒产生一次推理机会；8 路时目标总到达率为 8 RPS。必须大于 0。 |
| `--duration` | 否 | `600.0` | 每个并发档位的正式调度时长，单位为秒。`--concurrency 1,2,4,8 --duration 600` 会运行四轮各 600 秒，纯调度时间约 40 分钟，另加预抽帧和等待尾部请求完成的时间。必须大于 0。 |
| `--prompt` | 否 | `请观察当前画面并决定是否需要回应。` | 每次请求随图片发送的文本指令。比较不同并发档位时应保持 prompt 不变，因为 prompt 长度会影响 prefill 和总体延迟。包含空格时需要使用引号。 |
| `--max-tokens` | 否 | `128` | 单次请求允许生成的最大 token 数。数值越大，最坏情况下的 decode 时间越长。为了使不同轮次可比较，建议固定该值。 |
| `--temperature` | 否 | `0.0` | 生成采样温度。默认 0 尽量减少输出随机性，使重复测试更容易比较；它不保证不同请求的输出长度完全一致。 |
| `--seed` | 否 | 随机 | 视频分配所用的随机种子。指定相同整数可以在候选文件列表不变时复现各路视频选择，例如 `--seed 2026`。 |

### 调度和超时参数

| 参数 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--timeout` | 否 | `60.0` | 单次 API 请求的客户端超时时间，单位为秒。超过该时间的请求记为错误。它是请求失败边界，不是实时性合格标准。 |
| `--deadline-ms` | 否 | `1000.0` | 实时性目标，单位为毫秒。成功返回但延迟超过该值的请求会记为 deadline miss，不会记为请求错误。 |
| `--arrival` | 否 | `burst` | 多路请求在一个帧周期内的到达方式。`staggered` 将各路均匀错开，更接近自然流量；`burst` 让所有路在同一时刻到达，用于测试瞬时压力。 |
| `--overload-policy` | 否 | `drop` | 同一路上一请求尚未完成时如何处理新帧。`drop` 丢弃新帧，与当前 WebUI 行为一致；`queue` 仍然提交请求，用于观察服务排队和延迟累积。 |

### 输出参数

| 参数 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--output` | 否 | `benchmarks/video_concurrency/results` | 保存 `requests.jsonl`、`summary.json` 和 `summary.csv` 的目录。再次使用同一目录运行会覆盖这三个同名结果文件，因此正式实验建议为每轮指定不同目录。 |

可以随时通过下面的命令查看脚本当前支持的参数：

```bash
python benchmarks/video_concurrency/benchmark.py --help
```

## 调度模式

默认参数是：

- `--arrival staggered`：同一秒内均匀错开各路请求；`burst` 表示同时请求。
- `--overload-policy drop`：某一路上一帧仍在推理时丢弃新帧，与当前 WebUI
  `_processing_lock` 的行为一致。
- `--overload-policy queue`：不丢帧，用于观察服务队列和延迟是否持续增长。
- `--deadline-ms 1000`：将超过 1 秒的成功请求标为 deadline miss。

建议正式测试使用 `drop`，另跑一组 `queue` 诊断服务最大吞吐。

## 输出

输出目录包含：

- `requests.jsonl`：每个请求和每次丢帧的原始记录。
- `summary.json`：完整配置、总体指标和逐路指标。
- `summary.csv`：适合直接导入表格绘图的 1/2/4/8 路汇总。

主要关注 `p95_latency_ms`、`effective_fps_per_stream`、`drop_rate`、
`deadline_miss_rate` 和 `completed_rps`。API key 不会写入结果文件。

### `requests.jsonl` 字段说明

`requests.jsonl` 使用 JSON Lines 格式，每行是一条独立记录。`type=request` 表示实际
发起的 API 请求，示例如下：

```json
{"type":"request","concurrency":4,"stream_id":"bench-5f344b6019-002","frame_index":58,"scheduled_s":58.5,"started_s":58.5009,"completed_s":59.1851,"schedule_lag_ms":0.9066,"latency_ms":684.1507,"deadline_missed":false,"ok":true,"status_code":200,"prompt_tokens":100,"completion_tokens":128,"image_bytes":13213,"error":null}
```

| 字段 | 说明 |
| --- | --- |
| `type` | 记录类型。`request` 表示实际发起了请求，`drop` 表示该帧因上一请求仍未完成而被客户端丢弃。 |
| `concurrency` | 当前测试档位的视频路数，例如 `4` 表示正在进行 4 路测试。 |
| `stream_id` | 压测中的逻辑视频流 ID，末尾的 `002` 表示该轮测试中的第 3 路 stream。仅在 `adapter` 模式下，它会作为 `x-streaming-session` 请求头发送给服务端。 |
| `frame_index` | 当前 stream 的逻辑帧序号，从 0 开始；它表示调度序号，不一定等于成功请求数量，因为中间可能发生丢帧。 |
| `scheduled_s` | 按固定视频时钟，该帧原计划在本轮测试开始后多少秒发出。 |
| `started_s` | 客户端实际开始 API 请求的相对时间，单位为秒。 |
| `completed_s` | 客户端收到完整 API 响应的相对时间，单位为秒。 |
| `schedule_lag_ms` | 实际开始时间相对计划时间的滞后，即 `(started_s - scheduled_s) × 1000`。该值较大通常表示压测客户端自身调度不及时。 |
| `latency_ms` | 客户端端到端 API 延迟，即 `(completed_s - started_s) × 1000`，包含图片 Base64、网络传输、服务端排队和模型推理。 |
| `ttft_ms` | Time To First Token，从请求开始到收到第一个非空 SSE token/chunk。仅流式 vLLM 请求可用；adapter 当前为 `null`。 |
| `tpot_ms` | Time Per Output Token，使用 `(流结束时间 - 首 token 时间) / (completion_tokens - 1)` 估算。仅 completion tokens 大于 1 的流式请求可用。 |
| `output_tokens_per_second` | 首 token 之后的平均输出 token 吞吐。只用于流式请求。 |
| `deadline_missed` | 成功请求的延迟是否超过 `--deadline-ms`。超过 deadline 不等同于请求失败。 |
| `ok` | OpenAI-compatible API 调用是否成功完成。 |
| `status_code` | HTTP 状态码；成功时通常为 `200`，某些连接类异常可能没有状态码。 |
| `prompt_tokens` | 服务端 usage 数据中返回的输入 token 数。具体是否包含视觉 token 取决于服务端实现。 |
| `completion_tokens` | 服务端 usage 数据中返回的输出 token 数。如果它经常等于 `--max-tokens`，应结合 `finish_reason` 判断是否被截断。 |
| `finish_reason` | 服务端返回的结束原因，例如 `stop` 或 `length`。`length` 表示触及输出 token 上限。 |
| `response_category` | 根据完整输出分类为 `silence`、`response` 或 `empty`，用于分别统计监控静默和直播解说负载。响应原文不会写入日志。 |
| `response_chars` | 完整响应的字符数，不保存文本内容。 |
| `frames_arrived_during_request` | 当前请求执行期间，按照目标 FPS 已经到达的后续帧数量。用于判断生成过程中画面向前推进了多少帧。 |
| `adapter_total_ms` | Adapter 响应中已有的 `streamingharness.timing.adapter_total_ms`；直接 vLLM 请求为 `null`。 |
| `adapter_vllm_inference_ms` | Adapter 报告的完整 main-model 调用耗时，不是 TTFT；直接 vLLM 请求为 `null`。 |
| `image_bytes` | Base64 编码前的 JPEG/图片原始字节数，不包含 Base64 约 4/3 的体积膨胀和 JSON 请求体开销。 |
| `error` | 请求失败时记录异常类型和信息；请求成功时为 `null`。 |

`scheduled_s`、`started_s` 和 `completed_s` 都来自客户端单调时钟，并以当前并发档位
开始时刻为 0 点，不是 Unix 时间戳，因此适合计算时间差，不适合直接转换成日期时间。

在 `--fps 1 --arrival staggered` 的 4 路测试中，每秒内的计划时间分别错开为
0、0.25、0.5 和 0.75 秒。因此 `stream-002` 的 `frame_index=58` 对应：

```text
scheduled_s = 58 + 2 / 4 = 58.5 秒
schedule_lag_ms = (58.5009066 - 58.5) × 1000 ≈ 0.907 ms
latency_ms = (59.1850573 - 58.5009066) × 1000 ≈ 684.151 ms
```

当使用 `--overload-policy drop` 时，被丢弃的帧使用更短的记录：

```json
{"type":"drop","concurrency":4,"stream_id":"bench-xxxx-002","frame_index":59,"scheduled_s":59.5,"reason":"previous_request_still_running"}
```

其中 `reason=previous_request_still_running` 表示同一路的上一帧请求在新帧计划时间到达时
仍未完成。由于该帧没有发起 API 请求，因此没有 latency、HTTP 状态码或 token 数据。
每轮 stream 与输入视频的对应关系保存在 `summary.json` 的 `selected_videos` 中；目前
逐请求记录不重复保存视频路径，也不保存模型响应文本，以控制日志体积。

### 采集后端 vLLM `/metrics`

主要测试 adapter 时，可以额外传入后端 vLLM 的 Prometheus 地址：

```bash
python benchmarks/video_concurrency/benchmark.py --target adapter --api-base http://ADAPTER_HOST:8070/v1 --vllm-metrics https://VLLM_HOST/metrics ...
```

Prometheus Histogram 是 vLLM 启动后的累计数据，所以 benchmark 不会只在测试结束时读取。
每个 `--concurrency` 档位都会在正式请求开始前和尾部请求全部完成后各抓取一次，使用
`after - before` 得到本轮增量。`summary.json` 每轮结果的 `vllm_metrics.metrics`
包含以下字段，单位均为毫秒：

- `time_to_first_token_ms`：后端 vLLM TTFT；
- `inter_token_latency_ms`：相邻输出 token 的延迟；
- `e2e_request_latency_ms`：vLLM 内部请求端到端延迟；
- `request_prefill_time_ms`：Prefill 阶段耗时；
- `request_decode_time_ms`：Decode 阶段耗时。

每项保存 `count`、`mean`、`p50`、`p95` 和 `p99`。`mean` 由本轮 `_sum/_count`
精确计算；分位数根据 bucket 增量做线性插值，是 Histogram 近似值。
`inter_token_latency_ms.count` 是 token 间隔数量，其他四项通常是请求数量，因此它们
不要求相等。相同数据也会展开写入 `summary.csv`，列名前缀分别为 `vllm_ttft_`、
`vllm_itl_`、`vllm_e2e_`、`vllm_prefill_` 和 `vllm_decode_`。

如果抓取失败，测试本身仍继续，错误写入该轮 `vllm_metrics.error`。如果测试期间 vLLM
重启导致计数器回退，相应指标会记录 counter reset 错误。`/metrics` 是服务级聚合数据；
为避免混入其他用户或 summarizer 请求，测试期间应独占该 vLLM，或确保指标端点只服务
目标 main model。它衡量的是 adapter 后端 vLLM 的服务端时间，不包含 adapter session
锁等待、prompt 构造、网络和后处理，因此不能直接视为完整的 adapter 端到端 TTFT。

### TTFT、TPOT 与场景分层

原生 vLLM 在默认 `--streaming auto` 下使用 SSE，汇总文件会增加：

- `ttft_ms` 的 mean/P50/P95/P99/min/max；
- `tpot_ms` 的 mean/P50/P95/P99/min/max；
- `output_tokens_per_second` 和 `completion_tokens` 分布；
- `response_categories` 中的 silence/response 数量及占比；
- `response_category_metrics` 中 silence/response 各自的 latency、TTFT、TPOT 和
  completion token 分布，避免大量 silence 掩盖解说请求；
- `mean_frames_arrived_during_request`。

当前 webinfer adapter 外部接口返回普通 JSON，不提供 SSE。Benchmark 不会用完整响应时间
伪造 TTFT/TPOT；adapter 的这两组统计会显示 `count=0`、其余值为 `null`。它仍然会统计
silence 占比、非 silence 占比、输出 token、完整端到端延迟，以及 adapter 响应中已有的
总推理 timing。要获得 adapter 后端 main model 的真实 TTFT/TPOT，需要服务端提供相应
流式时间点，本次改动没有修改 adapter。

视频监控报告应重点查看 silence rate、决策延迟、丢帧和有效 FPS；实时直播解说应重点
查看非 silence 请求的 TTFT、TPOT、完整响应延迟和 `frames_arrived_during_request`。
为了让直播测试产生足够多的非 silence 回复，应使用直播类视频和明确的解说 prompt，
例如：

```bash
--prompt "你是实时直播解说员，请持续根据当前画面给出简短、及时的解说。"
```

运行单元测试：

```bash
python -m unittest discover -s benchmarks/video_concurrency -p 'test_*.py'
```

正式测试时建议同步采集服务端 vLLM `/metrics` 和 `nvidia-smi`/DCGM 指标。该脚本的
延迟是客户端端到端 API 延迟，不包含测试开始前的视频解码，但包含图片 base64、网络、
服务端排队和模型推理。
