import unittest
import random
import json
import tempfile
from argparse import Namespace
from pathlib import Path

from benchmark import (
    DropRecord, RequestRecord, build_request_kwargs, discover_media, json_safe,
    percentile, response_category, select_sources, summarize, use_streaming,
    write_outputs,
)


class BenchmarkSummaryTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([100, 200, 300], 0.5), 200)
        self.assertEqual(percentile([], 0.5), None)

    def test_summary_includes_drops_errors_and_deadlines(self):
        base = dict(
            concurrency=2, stream_id="stream-0", frame_index=0,
            scheduled_s=0, started_s=0, completed_s=0.5, schedule_lag_ms=0,
            status_code=200, prompt_tokens=10, completion_tokens=2,
            image_bytes=100, error=None,
        )
        records = [
            RequestRecord(latency_ms=500, deadline_missed=False, ok=True, **base),
            RequestRecord(
                latency_ms=1500, deadline_missed=True, ok=False,
                **{**base, "stream_id": "stream-1", "status_code": 500, "error": "failed"},
            ),
        ]
        drops = [DropRecord(2, "stream-1", 1, 1.0)]
        result = summarize(records, drops, concurrency=2, duration_s=2, fps=1)
        self.assertEqual(result["successful_requests"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["dropped_frames"], 1)
        self.assertEqual(result["drop_rate"], 1 / 3)
        self.assertEqual(result["effective_fps_per_stream"], 0.25)

    def test_source_selection_prefers_unique_then_repeats(self):
        candidates = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
        unique = select_sources(candidates, 3, random.Random(7))
        repeated = select_sources(candidates, 5, random.Random(7))
        self.assertEqual(len(set(unique)), 3)
        self.assertEqual(len(set(repeated[:3])), 3)
        self.assertEqual(len(repeated), 5)

    def test_discover_media_expands_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "nested").mkdir()
            (root / "a.mp4").touch()
            (root / "nested" / "b.mkv").touch()
            (root / "ignore.txt").touch()
            found = discover_media([root])
        self.assertEqual([path.name for path in found], ["a.mp4", "b.mkv"])

    def test_output_config_serializes_video_path_list(self):
        args = Namespace(video=[Path("videos/a.mp4"), Path("videos/b.mp4")], api_key="secret")
        summary = {
            "concurrency": 1,
            "latency_ms": {"p50": 1, "p95": 2, "p99": 3},
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_outputs(output, args, [], [], [summary])
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["config"]["video"], ["videos/a.mp4", "videos/b.mp4"])
        self.assertNotIn("api_key", report["config"])

    def test_json_safe_handles_nested_paths(self):
        value = {"inputs": [Path("a.mp4"), {"output": Path("results")}]}
        self.assertEqual(
            json_safe(value), {"inputs": ["a.mp4", {"output": "results"}]}
        )

    def test_vllm_request_has_no_adapter_extensions(self):
        args = Namespace(
            target="vllm", model="model", prompt="prompt", max_tokens=8,
            temperature=0, timeout=10, fps=1,
        )
        kwargs = build_request_kwargs(
            args, stream_id="stream-0", frame_index=2,
            frame=b"jpeg", source=Path("frame.jpg"),
        )
        self.assertNotIn("extra_headers", kwargs)
        self.assertNotIn("extra_body", kwargs)

    def test_adapter_request_has_session_and_frame_time(self):
        args = Namespace(
            target="adapter", model="model", prompt="prompt", max_tokens=8,
            temperature=0, timeout=10, fps=2,
        )
        kwargs = build_request_kwargs(
            args, stream_id="stream-0", frame_index=3,
            frame=b"jpeg", source=Path("frame.jpg"),
        )
        self.assertEqual(kwargs["extra_headers"], {"x-streaming-session": "stream-0"})
        self.assertEqual(kwargs["extra_body"], {"frame_time_range": "1.5 seconds"})

    def test_streaming_auto_only_enables_vllm(self):
        self.assertTrue(use_streaming(Namespace(target="vllm", streaming="auto")))
        self.assertFalse(use_streaming(Namespace(target="adapter", streaming="auto")))
        self.assertFalse(use_streaming(Namespace(target="vllm", streaming="off")))

    def test_response_category(self):
        self.assertEqual(response_category("</silence>"), "silence")
        self.assertEqual(response_category("silence"), "silence")
        self.assertEqual(response_category("现场出现了一辆车"), "response")
        self.assertEqual(response_category(""), "empty")


if __name__ == "__main__":
    unittest.main()
