import argparse
import json
import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

MAX_DURATION = 320  # Maximum frame extraction duration in seconds; override with --max_duration.
FFMPEG_THREADS = 2

# Process-level duration cache to avoid repeated ffprobe calls.
_duration_cache = {}
_duration_lock = threading.Lock()
_duration_in_flight = {}

# Frame output directory locks to prevent concurrent extraction/deletion for the same out_dir.
# value: (Lock, refcount)
_dir_locks = {}
_dir_locks_guard = threading.Lock()


def get_video_duration(video_path):
    """Get the actual video duration in seconds with ffprobe, using a thread-safe process cache."""
    with _duration_lock:
        if video_path in _duration_cache:
            return _duration_cache[video_path]
        if video_path in _duration_in_flight:
            event = _duration_in_flight[video_path]
        else:
            _duration_in_flight[video_path] = threading.Event()
            event = None
    if event is not None:
        event.wait()
        return _duration_cache[video_path]

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    try:
        dur = float(result.stdout.strip())
    except (ValueError, AttributeError):
        dur = 0.0
    with _duration_lock:
        _duration_cache[video_path] = dur
        _duration_in_flight.pop(video_path).set()
    return dur


def _list_frame_paths(output_dir, expected_count=None):
    if expected_count is not None:
        return [os.path.join(output_dir, f"frame_{i:06d}.jpg") for i in range(expected_count)]
    return sorted(
        os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")
    )


def extract_frames(video_path, output_dir, fps=1.0, rewrite=False,
                   max_duration=None):
    """Extract frames from a video at the given fps.

    Returns (frame_paths, effective_duration, truncated).
    Videos longer than max_duration only use the first max_duration seconds.
    """
    if max_duration is None:
        max_duration = MAX_DURATION
    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(video_path)
    if duration <= 0:
        raise ValueError(f"ffprobe returned duration={duration} for {video_path}")

    # max_duration <= 0 means no limit.
    if max_duration and max_duration > 0:
        truncated = duration > max_duration
        effective_duration = min(duration, max_duration)
    else:
        truncated = False
        effective_duration = duration

    # Get the directory-specific lock to prevent concurrent extraction/deletion in the same directory.
    with _dir_locks_guard:
        if output_dir not in _dir_locks:
            _dir_locks[output_dir] = [threading.Lock(), 0]
        entry = _dir_locks[output_dir]
        entry[1] += 1
        dir_lock = entry[0]

    try:
        with dir_lock:
            # If frame files already exist and rewrite is not requested, return them directly.
            existing = _list_frame_paths(output_dir)
            if existing and not rewrite:
                return existing, effective_duration, truncated

            # Delete old frames in rewrite mode.
            if rewrite and existing:
                for p in existing:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
            ]
            cmd += [
                "-threads",
                str(FFMPEG_THREADS),
                "-fflags",
                "+discardcorrupt",
                "-err_detect",
                "ignore_err",
            ]
            cmd += [
                "-i",
                video_path,
            ]
            cmd += [
                "-vf",
                f"fps={fps}",
                "-q:v",
                "5",
                "-start_number",
                "0",
                "-y",
                output_pattern,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            expected_count = max(int(duration * fps), 1)
            paths = _list_frame_paths(output_dir, expected_count)
            if not paths or not os.path.exists(paths[0]):
                paths = _list_frame_paths(output_dir)
            if not paths:
                stderr = (result.stderr or "").strip()
                raise ValueError(
                    f"ffmpeg extracted 0 frames from {video_path}"
                    + (f": {stderr}" if stderr else "")
                )
            return paths, effective_duration, truncated
    finally:
        with _dir_locks_guard:
            entry[1] -= 1
            if entry[1] == 0:
                del _dir_locks[output_dir]


def parse_times(time_str):
    """Parse a time field such as '8' or '5,6,7' and return a list of ints."""
    if not time_str:
        return []
    return [int(float(t.strip())) for t in str(time_str).split(",") if t.strip()]


def convert_sample(sample, frame_dir, rewrite=False,
                   max_duration=None):
    """Convert one raw sample to inference format.

    Returns (result_dict, truncated_bool, warning_msg) or (None, False, warning_msg).
    """
    video_path = sample["video_path"]
    task_type = sample.get("task_type", "")
    source = sample.get("source", "")
    video_stem = os.path.splitext(sample["video_name"])[0]

    # Dynamically adjust fps based on video duration.
    duration = get_video_duration(video_path)
    if duration >= 160:
        fps = 1.0
    elif duration >= 64:
        fps = 2.0
    else:
        fps = 4.0

    # Extract frames, automatically truncating videos longer than max_duration.
    frame_source = str(source or "").strip("/")
    out_dir = os.path.join(frame_dir, task_type, frame_source, video_stem)
    frame_paths, effective_duration, truncated = extract_frames(
        video_path, out_dir, fps, rewrite=rewrite, max_duration=max_duration)

    # Frame extraction may cover the full video, but JSON output only uses frames for effective_duration.
    effective_n = max(int(effective_duration * fps), 1)
    used_paths = frame_paths[:effective_n]

    # Group by second: each second contains fps frames, and messages are organized by second.
    frames_per_sec = max(int(fps), 1)
    n_seconds = max(int(effective_duration), 1)

    # question_map: second_idx -> text
    question_map = {}
    for q in sample.get("question", []):
        for t in parse_times(q.get("time")):
            if t > effective_duration:
                if truncated:
                    continue
                return None, False, (
                    f"question time {t}s > video duration {effective_duration:.2f}s, "
                    f"skipping: {sample.get('video_name', video_path)}")
            si = min(t, n_seconds - 1)
            question_map[si] = q["content"]

    # response_map: second_idx -> text
    # support both flat list [{"content":..,"time":..}, ...] and
    # nested list [[{"content":..,"time":..}, ...], ...] formats
    response_map = {}
    raw_responses = sample.get("response", [])
    flat_responses = []
    for item in raw_responses:
        if isinstance(item, list):
            flat_responses.extend(item)
        else:
            flat_responses.append(item)
    for r in flat_responses:
        for t in parse_times(r.get("time")):
            if t > effective_duration:
                if truncated:
                    continue
                return None, False, (
                    f"response time {t}s > video duration {effective_duration:.2f}s, "
                    f"skipping: {sample.get('video_name', video_path)}")
            si = min(t, n_seconds - 1)
            response_map[si] = r["content"]

    # Build messages. The system prompt is added automatically by qwen3vl.py.
    # Each second has one user message containing frames_per_sec <image> tags.
    messages = []
    for sec in range(n_seconds):
        # user
        parts = []
        if sec in question_map:
            parts.append(question_map[sec])
        parts.append(f"<{sec:.1f} seconds>")
        for _ in range(frames_per_sec):
            parts.append("<image>")
        messages.append({"role": "user", "content": "\n".join(parts)})

        # assistant (ground truth)
        if sec in response_map:
            messages.append({"role": "assistant", "content": f"</response> {response_map[sec]}"})
        else:
            messages.append({"role": "assistant", "content": "</silence>"})

    # Keep only the frames actually used by the per-second grouping.
    actual_used_paths = used_paths[:n_seconds * frames_per_sec]

    return {
        "messages": messages,
        "images": actual_used_paths,
        "video_name": sample["video_name"],
        "video_path": video_path,
        "task_type": task_type,
        "source": source,
    }, truncated, None


def collect_json_files(path):
    """Collect all JSON files under the input path."""
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for name in sorted(names):
            if name.endswith(".json") and name != "example.json":
                files.append(os.path.join(root, name))
    return files


def _process_sample(sample_and_args):
    """Wrapper for concurrent workers. Returns (result, truncated, warning)."""
    (
        sample,
        frame_dir,
        rewrite,
        max_duration,
    ) = sample_and_args
    try:
        return convert_sample(sample, frame_dir, rewrite=rewrite,
                              max_duration=max_duration)
    except Exception as exc:
        return None, False, (
            f"{type(exc).__name__}: {exc}, "
            f"skipping: {sample.get('video_name', sample.get('video_path', '<unknown>'))}"
        )


def _log_sample_result(result, was_truncated, warn_msg, logger, truncated_logger,
                       max_duration):
    """Log one sample result and return (filtered_count_increment, truncated_count_increment)."""
    filtered_inc = 0
    truncated_inc = 0

    if warn_msg:
        filtered_inc = 1
        logger.warning(f"  Filtered sample: {warn_msg}")

    if result and was_truncated:
        truncated_inc = 1
        orig_dur = _duration_cache[result["video_path"]]
        truncated_logger.info(
            f"TRUNCATED: {result['video_name']} | "
            f"path={result['video_path']} | "
            f"original_duration={orig_dur:.2f}s | "
            f"used_duration={max_duration}s | "
            f"frames={len(result['images'])}")
        logger.info(
            f"  Truncated video: {result['video_name']} "
            f"({orig_dur:.2f}s -> {max_duration}s)")

    return filtered_inc, truncated_inc


def setup_logging(log_dir):
    """Set up timestamped logs and return (general_logger, truncated_logger)."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # General log.
    general_logger = logging.getLogger("convert_eval")
    general_logger.handlers.clear()
    general_logger.setLevel(logging.INFO)
    general_log_path = os.path.join(log_dir, f"convert_eval_{timestamp}.log")
    gh = logging.FileHandler(general_log_path, encoding="utf-8")
    gh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    general_logger.addHandler(gh)
    # Also output to console.
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    general_logger.addHandler(sh)

    # Truncation log for videos longer than max_duration.
    truncated_logger = logging.getLogger("truncated")
    truncated_logger.handlers.clear()
    truncated_logger.setLevel(logging.INFO)
    truncated_log_path = os.path.join(log_dir, f"truncated_videos_{timestamp}.log")
    th = logging.FileHandler(truncated_log_path, encoding="utf-8")
    th.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    truncated_logger.addHandler(th)

    general_logger.info(f"General log: {general_log_path}")
    general_logger.info(f"Truncation log: {truncated_log_path}")

    return general_logger, truncated_logger


def main():
    parser = argparse.ArgumentParser(description="Convert raw evaluation data to inference format")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file or raw_data/ directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory; preserves the original directory structure")
    parser.add_argument("--frame_dir", required=True, help="Root directory for saved frame images")
    parser.add_argument("--max_samples", type=int, default=0, help="Only process the first N samples; 0 means process all samples")
    parser.add_argument("--rewrite", action="store_true", help="Force rewrite; equivalent to specifying both --rewrite_frames and --rewrite_json")
    parser.add_argument("--rewrite_frames", action="store_true", help="Force frame re-extraction and overwrite existing frame files")
    parser.add_argument("--rewrite_json", action="store_true", help="Force output JSON regeneration and overwrite existing files")
    parser.add_argument("--workers", "-w", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--log_dir", default="logs", help="Directory for saved logs")
    parser.add_argument("--max_duration", type=int, default=MAX_DURATION,
                        help=f"Maximum frame extraction duration in seconds; longer videos are truncated (default {MAX_DURATION})")
    args = parser.parse_args()

    if args.rewrite:
        args.rewrite_frames = True
        args.rewrite_json = True

    logger, truncated_logger = setup_logging(args.log_dir)

    input_base = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    input_files = collect_json_files(args.input)
    logger.info(f"Found {len(input_files)} JSON files, using {args.workers} workers, "
                f"max_duration={args.max_duration}s")

    global_count = 0
    truncated_count = 0
    filtered_count = 0
    for fpath in tqdm(input_files, desc="Converting"):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]

        if args.max_samples > 0:
            remaining = args.max_samples - global_count
            if remaining <= 0:
                break
            data = data[:remaining]

        rel = os.path.relpath(fpath, input_base)
        out_path = os.path.join(args.output, rel)
        if not args.rewrite_json and os.path.exists(out_path):
            logger.info(f"  {rel}: output already exists, skipping (use --rewrite_json or --rewrite to overwrite)")
            continue

        # Prefetch all unique video durations in parallel in the main process.
        unique_videos = list({s["video_path"] for s in data} - set(_duration_cache))
        if unique_videos:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(get_video_duration, unique_videos))

        # Process samples concurrently.
        task_args = [
            (
                sample,
                args.frame_dir,
                args.rewrite_frames,
                args.max_duration,
            )
            for sample in data
        ]
        results = []
        if args.workers > 1 and len(data) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_sample, ta): i for i, ta in enumerate(task_args)}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"  {os.path.basename(fpath)}", leave=False):
                    r, was_truncated, warn_msg = future.result()
                    filtered_inc, truncated_inc = _log_sample_result(
                        r, was_truncated, warn_msg, logger, truncated_logger,
                        args.max_duration)
                    filtered_count += filtered_inc
                    truncated_count += truncated_inc
                    if r:
                        results.append((futures[future], r))
            # Preserve original order.
            results.sort(key=lambda x: x[0])
            results = [r for _, r in results]
        else:
            for ta in tqdm(task_args, desc=f"  {os.path.basename(fpath)}", leave=False):
                r, was_truncated, warn_msg = _process_sample(ta)
                filtered_inc, truncated_inc = _log_sample_result(
                    r, was_truncated, warn_msg, logger, truncated_logger,
                    args.max_duration)
                filtered_count += filtered_inc
                truncated_count += truncated_inc
                if r:
                    results.append(r)

        if not results:
            continue

        # Preserve directory structure: raw_data/qa/.../1.json -> data/qa/.../1.json.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        global_count += len(results)
        logger.info(f"  {rel}: {len(results)} samples -> {out_path}")

        if args.max_samples > 0 and global_count >= args.max_samples:
            break

    logger.info(f"Done. Processed {global_count} samples, "
                f"{truncated_count} videos were truncated (>{args.max_duration}s), "
                f"{filtered_count} samples were filtered")


if __name__ == "__main__":
    main()
