#!/usr/bin/env python3
"""Open-loop multi-video benchmark for an OpenAI-compatible VLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg",
    ".jpg", ".jpeg", ".png", ".webp",
}

@dataclass
class RequestRecord:
    concurrency: int
    stream_id: str
    frame_index: int
    scheduled_s: float
    started_s: float
    completed_s: float
    schedule_lag_ms: float
    latency_ms: float
    deadline_missed: bool
    ok: bool
    status_code: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    image_bytes: int
    error: str | None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    output_tokens_per_second: float | None = None
    finish_reason: str | None = None
    response_category: str = "unknown"
    response_chars: int = 0
    frames_arrived_during_request: int = 0
    adapter_total_ms: float | None = None
    adapter_vllm_inference_ms: float | None = None


@dataclass
class DropRecord:
    concurrency: int
    stream_id: str
    frame_index: int
    scheduled_s: float
    reason: str = "previous_request_still_running"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    records: list[RequestRecord],
    drops: list[DropRecord],
    *,
    concurrency: int,
    duration_s: float,
    fps: float,
) -> dict[str, Any]:
    successful = [r for r in records if r.ok]
    latencies = [r.latency_ms for r in successful]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]
    output_rates = [
        r.output_tokens_per_second
        for r in successful
        if r.output_tokens_per_second is not None
    ]
    completion_counts = [
        r.completion_tokens for r in successful if r.completion_tokens is not None
    ]
    category_counts = {
        category: sum(r.response_category == category for r in successful)
        for category in ("silence", "response", "empty")
    }
    category_metrics = {}
    for category in ("silence", "response", "empty"):
        category_records = [
            record for record in successful if record.response_category == category
        ]
        category_metrics[category] = {
            "count": len(category_records),
            "latency_ms": metric_summary(
                [record.latency_ms for record in category_records]
            ),
            "ttft_ms": metric_summary(
                [
                    record.ttft_ms
                    for record in category_records
                    if record.ttft_ms is not None
                ]
            ),
            "tpot_ms": metric_summary(
                [
                    record.tpot_ms
                    for record in category_records
                    if record.tpot_ms is not None
                ]
            ),
            "completion_tokens": metric_summary(
                [
                    record.completion_tokens
                    for record in category_records
                    if record.completion_tokens is not None
                ]
            ),
        }
    scheduled = len(records) + len(drops)
    stream_stats: dict[str, dict[str, Any]] = {}
    stream_ids = sorted({r.stream_id for r in records} | {d.stream_id for d in drops})
    for stream_id in stream_ids:
        sr = [r for r in successful if r.stream_id == stream_id]
        attempted = sum(r.stream_id == stream_id for r in records)
        dropped = sum(d.stream_id == stream_id for d in drops)
        stream_stats[stream_id] = {
            "successful": len(sr),
            "errors": attempted - len(sr),
            "dropped": dropped,
            "effective_fps": len(sr) / duration_s,
            "p95_latency_ms": percentile([r.latency_ms for r in sr], 0.95),
        }
    return {
        "concurrency": concurrency,
        "duration_s": duration_s,
        "target_fps_per_stream": fps,
        "scheduled_frames": scheduled,
        "attempted_requests": len(records),
        "successful_requests": len(successful),
        "errors": len(records) - len(successful),
        "dropped_frames": len(drops),
        "drop_rate": len(drops) / scheduled if scheduled else 0.0,
        "error_rate": (len(records) - len(successful)) / len(records) if records else 0.0,
        "completed_rps": len(successful) / duration_s,
        "effective_fps_per_stream": len(successful) / duration_s / concurrency,
        "deadline_miss_rate": (
            sum(r.deadline_missed for r in successful) / len(successful)
            if successful
            else 0.0
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "ttft_ms": metric_summary(ttfts),
        "tpot_ms": metric_summary(tpots),
        "output_tokens_per_second": metric_summary(output_rates),
        "completion_tokens": metric_summary(completion_counts),
        "response_categories": {
            **category_counts,
            "silence_rate": (
                category_counts["silence"] / len(successful) if successful else 0.0
            ),
            "response_rate": (
                category_counts["response"] / len(successful) if successful else 0.0
            ),
        },
        "response_category_metrics": category_metrics,
        "requests_with_streaming_metrics": len(ttfts),
        "mean_frames_arrived_during_request": (
            statistics.fmean(r.frames_arrived_during_request for r in successful)
            if successful
            else 0.0
        ),
        "streams": stream_stats,
    }


def metric_summary(values: list[float | int]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric) if numeric else None,
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
        "p99": percentile(numeric, 0.99),
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }


def extract_frames(source: Path, fps: float, work_dir: Path) -> list[bytes]:
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        return [source.read_bytes()]
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video input")
    source_id = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    output_dir = work_dir / f"{source.stem}-{source_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vf", f"fps={fps}", "-q:v", "3", str(output_dir / "%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    frames = [path.read_bytes() for path in sorted(output_dir.glob("*.jpg"))]
    if not frames:
        raise RuntimeError(f"no frames extracted from {source}")
    return frames


def discover_media(inputs: list[Path]) -> list[Path]:
    """Expand files/directories into a stable, de-duplicated media list."""
    candidates: list[Path] = []
    for value in inputs:
        if value.is_file():
            candidates.append(value.resolve())
        elif value.is_dir():
            candidates.extend(
                path.resolve()
                for path in sorted(value.rglob("*"))
                if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
            )
    return list(dict.fromkeys(candidates))


def select_sources(
    candidates: list[Path], concurrency: int, rng: random.Random
) -> list[Path]:
    """Prefer unique sources, repeating randomly only after exhausting them."""
    if not candidates:
        raise ValueError("no media candidates")
    selected = rng.sample(candidates, min(concurrency, len(candidates)))
    if concurrency > len(candidates):
        selected.extend(rng.choice(candidates) for _ in range(concurrency - len(candidates)))
    return selected


def image_data_url(data: bytes, source: Path) -> str:
    mime = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def use_streaming(args: argparse.Namespace) -> bool:
    if args.streaming == "on":
        return True
    if args.streaming == "off":
        return False
    return args.target == "vllm"


def response_category(text: str) -> str:
    normalized = (text or "").strip().lower()
    if not normalized:
        return "empty"
    if "</silence>" in normalized or normalized == "silence":
        return "silence"
    return "response"


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in value
        )
    return "" if value is None else str(value)


def build_request_kwargs(
    args: argparse.Namespace,
    *,
    stream_id: str,
    frame_index: int,
    frame: bytes,
    source: Path,
) -> dict[str, Any]:
    """Build a pure vLLM request or an adapter-aware stateful request."""
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(frame, source)}},
            ],
        }],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
    }
    if args.target == "adapter":
        kwargs["extra_headers"] = {"x-streaming-session": stream_id}
        kwargs["extra_body"] = {
            "frame_time_range": f"{frame_index / args.fps:.1f} seconds"
        }
    return kwargs


async def send_request(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    *,
    concurrency: int,
    stream_id: str,
    frame_index: int,
    frame: bytes,
    source: Path,
    scheduled_at: float,
    run_started_at: float,
) -> RequestRecord:
    started_at = time.perf_counter()
    status_code = None
    prompt_tokens = None
    completion_tokens = None
    error = None
    ok = False
    ttft_ms = None
    tpot_ms = None
    output_rate = None
    finish_reason = None
    response_text = ""
    adapter_total_ms = None
    adapter_vllm_inference_ms = None
    try:
        request_kwargs = build_request_kwargs(
            args,
            stream_id=stream_id,
            frame_index=frame_index,
            frame=frame,
            source=source,
        )
        if use_streaming(args):
            request_kwargs["stream"] = True
            request_kwargs["stream_options"] = {"include_usage": True}
            stream = await client.chat.completions.create(**request_kwargs)
            first_token_at = None
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                    completion_tokens = getattr(
                        usage, "completion_tokens", completion_tokens
                    )
                choices = getattr(chunk, "choices", None) or []
                for choice in choices:
                    reason = getattr(choice, "finish_reason", None)
                    if reason:
                        finish_reason = str(reason)
                    delta = getattr(choice, "delta", None)
                    content_piece = _text_value(getattr(delta, "content", None))
                    timing_piece = content_piece or _text_value(
                        getattr(delta, "reasoning_content", None)
                    )
                    if timing_piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            ttft_ms = (first_token_at - started_at) * 1000
                    if content_piece:
                        response_text += content_piece
            stream_completed_at = time.perf_counter()
            if first_token_at is not None and completion_tokens:
                decode_seconds = max(0.0, stream_completed_at - first_token_at)
                if completion_tokens > 1:
                    tpot_ms = decode_seconds * 1000 / (completion_tokens - 1)
                    if decode_seconds > 0:
                        output_rate = (completion_tokens - 1) / decode_seconds
        else:
            response = await client.chat.completions.create(**request_kwargs)
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            choices = getattr(response, "choices", None) or []
            if choices:
                finish_reason = getattr(choices[0], "finish_reason", None)
                message = getattr(choices[0], "message", None)
                response_text = _text_value(getattr(message, "content", None))
            try:
                payload = response.model_dump()
                timing = payload.get("streamingharness", {}).get("timing", {})
                adapter_total_ms = timing.get("adapter_total_ms")
                adapter_vllm_inference_ms = timing.get("vllm_inference_ms")
            except (AttributeError, TypeError):
                pass
        ok = True
        status_code = 200
    except Exception as exc:  # The record is more useful than aborting the run.
        status_code = getattr(exc, "status_code", None)
        error = f"{type(exc).__name__}: {exc}"
    completed_at = time.perf_counter()
    latency_ms = (completed_at - started_at) * 1000
    frames_arrived = max(
        0, math.floor((completed_at - scheduled_at) * args.fps + 1e-9)
    )
    return RequestRecord(
        concurrency=concurrency,
        stream_id=stream_id,
        frame_index=frame_index,
        scheduled_s=scheduled_at - run_started_at,
        started_s=started_at - run_started_at,
        completed_s=completed_at - run_started_at,
        schedule_lag_ms=(started_at - scheduled_at) * 1000,
        latency_ms=latency_ms,
        deadline_missed=latency_ms > args.deadline_ms,
        ok=ok,
        status_code=status_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        image_bytes=len(frame),
        error=error,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        output_tokens_per_second=output_rate,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_category=response_category(response_text),
        response_chars=len(response_text),
        frames_arrived_during_request=frames_arrived,
        adapter_total_ms=adapter_total_ms,
        adapter_vllm_inference_ms=adapter_vllm_inference_ms,
    )


async def run_one(
    args: argparse.Namespace,
    concurrency: int,
    frame_sets: list[tuple[Path, list[bytes]]],
) -> tuple[list[RequestRecord], list[DropRecord], dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency 'openai'; activate the project WebUI environment "
            "or install services/webui"
        ) from exc
    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)
    records: list[RequestRecord] = []
    drops: list[DropRecord] = []
    pending: dict[int, asyncio.Task[RequestRecord]] = {}
    all_tasks: list[asyncio.Task[RequestRecord]] = []
    interval = 1.0 / args.fps
    run_id = uuid.uuid4().hex[:10]
    run_started_at = time.perf_counter()
    total_ticks = math.ceil(args.duration * args.fps)

    try:
        for tick in range(total_ticks):
            for stream_index in range(concurrency):
                offset = 0.0 if args.arrival == "burst" else stream_index * interval / concurrency
                scheduled_at = run_started_at + tick * interval + offset
                delay = scheduled_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                previous = pending.get(stream_index)
                stream_id = f"bench-{run_id}-{stream_index:03d}"
                if args.overload_policy == "drop" and previous and not previous.done():
                    drops.append(DropRecord(
                        concurrency, stream_id, tick, scheduled_at - run_started_at
                    ))
                    continue
                source, frames = frame_sets[stream_index]
                task = asyncio.create_task(send_request(
                    client, args,
                    concurrency=concurrency,
                    stream_id=stream_id,
                    frame_index=tick,
                    frame=frames[tick % len(frames)],
                    source=source,
                    scheduled_at=scheduled_at,
                    run_started_at=run_started_at,
                ))
                pending[stream_index] = task
                all_tasks.append(task)
        if all_tasks:
            records.extend(await asyncio.gather(*all_tasks))
    finally:
        await client.close()
    summary = summarize(
        records, drops, concurrency=concurrency, duration_s=args.duration, fps=args.fps
    )
    return records, drops, summary


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    records: list[RequestRecord],
    drops: list[DropRecord],
    summaries: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"type": "request", **asdict(record)}, ensure_ascii=False) + "\n")
        for drop in drops:
            handle.write(json.dumps({"type": "drop", **asdict(drop)}, ensure_ascii=False) + "\n")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: json_safe(value)
            for key, value in vars(args).items()
            if key not in {"api_key"}
        },
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "concurrency", "scheduled_frames", "successful_requests", "errors",
            "dropped_frames", "drop_rate", "completed_rps", "effective_fps_per_stream",
            "deadline_miss_rate", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "p50_ttft_ms", "p95_ttft_ms", "p50_tpot_ms", "p95_tpot_ms",
            "mean_output_tokens_per_second", "mean_completion_tokens",
            "silence_rate", "response_rate", "mean_frames_arrived_during_request",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            ttft = item.get("ttft_ms", {})
            tpot = item.get("tpot_ms", {})
            output_rate = item.get("output_tokens_per_second", {})
            completion = item.get("completion_tokens", {})
            categories = item.get("response_categories", {})
            writer.writerow({
                **{field: item.get(field) for field in fields},
                "p50_latency_ms": item["latency_ms"]["p50"],
                "p95_latency_ms": item["latency_ms"]["p95"],
                "p99_latency_ms": item["latency_ms"]["p99"],
                "p50_ttft_ms": ttft.get("p50"),
                "p95_ttft_ms": ttft.get("p95"),
                "p50_tpot_ms": tpot.get("p50"),
                "p95_tpot_ms": tpot.get("p95"),
                "mean_output_tokens_per_second": output_rate.get("mean"),
                "mean_completion_tokens": completion.get("mean"),
                "silence_rate": categories.get("silence_rate"),
                "response_rate": categories.get("response_rate"),
                "mean_frames_arrived_during_request": item.get("mean_frames_arrived_during_request"),
            })


def json_safe(value: Any) -> Any:
    """Recursively convert argparse values such as list[Path] to JSON types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("vllm", "adapter"), default="vllm",
        help="Test the raw vLLM API or the stateful webinfer adapter",
    )
    parser.add_argument(
        "--streaming", choices=("auto", "on", "off"), default="auto",
        help="Use SSE for token timing; auto enables it for vLLM only",
    )
    parser.add_argument("--api-base", required=True, help="Example: http://host:7060/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True,
                        help="Video/image file or directory; repeat for multiple inputs")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--prompt", default="请实时解说画面")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed used when assigning videos to streams")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--deadline-ms", type=float, default=1000.0)
    parser.add_argument("--arrival", choices=("staggered", "burst"), default="burst")
    parser.add_argument("--overload-policy", choices=("drop", "queue"), default="drop")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/video_concurrency/results"))
    args = parser.parse_args()
    if args.fps <= 0 or args.duration <= 0:
        parser.error("--fps and --duration must be positive")
    args.concurrency = [int(value) for value in args.concurrency.split(",")]
    if not args.concurrency or any(value <= 0 for value in args.concurrency):
        parser.error("--concurrency must contain positive integers")
    missing = [str(path) for path in args.video if not path.exists()]
    if missing:
        parser.error(f"input paths not found: {', '.join(missing)}")
    if not discover_media(args.video):
        parser.error("--video inputs contain no supported media files")
    if args.target == "adapter" and args.streaming == "on":
        parser.error("adapter does not expose SSE; use --streaming auto or off")
    return args


async def async_main(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="vlm-benchmark-") as temp:
        print(f"target={args.target} api_base={args.api_base}", flush=True)
        candidates = discover_media(args.video)
        rng = random.Random(args.seed)
        frame_cache: dict[Path, list[bytes]] = {}
        all_records: list[RequestRecord] = []
        all_drops: list[DropRecord] = []
        summaries: list[dict[str, Any]] = []
        for concurrency in args.concurrency:
            print(f"running concurrency={concurrency} ...", flush=True)
            selected_sources = select_sources(candidates, concurrency, rng)
            print("selected videos:", flush=True)
            for stream_index, source in enumerate(selected_sources):
                print(f"  stream-{stream_index:03d}: {source}", flush=True)
                if source not in frame_cache:
                    frame_cache[source] = extract_frames(source, args.fps, Path(temp))
            frame_sets = [(source, frame_cache[source]) for source in selected_sources]
            records, drops, summary = await run_one(args, concurrency, frame_sets)
            summary["selected_videos"] = [str(source) for source in selected_sources]
            all_records.extend(records)
            all_drops.extend(drops)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        write_outputs(args.output, args, all_records, all_drops, summaries)
        print(f"results written to {args.output}")


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
