"""Local Codex CLI API for StreamingHarness background tasks."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_HOST = os.environ.get("CODEX_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("CODEX_API_PORT", "8079"))
DEFAULT_MAX_SUBAGENTS = int(os.environ.get("CODEX_API_MAX_SUBAGENTS", "6"))
DEFAULT_MAX_CONCURRENT_RUNS = int(os.environ.get("CODEX_API_MAX_CONCURRENT_RUNS", "2"))
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("CODEX_API_TIMEOUT_SECONDS", "600"))
DEFAULT_MAX_FRAMES = int(os.environ.get("CODEX_API_MAX_FRAMES", "50"))
DEFAULT_BACKGROUND_AGENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_PATH = DEFAULT_BACKGROUND_AGENT_DIR.parent.parent / "agent-workspace"
DEFAULT_WORKSPACE = os.environ.get(
    "CODEX_API_WORKSPACE",
    str(DEFAULT_WORKSPACE_PATH),
)
DEFAULT_CODEX_HOME = os.environ.get(
    "CODEX_HOME",
    str(DEFAULT_BACKGROUND_AGENT_DIR / "codex-home"),
)
STDERR_TAIL_BYTES = int(os.environ.get("CODEX_API_STDERR_TAIL_BYTES", "20000"))
STDOUT_MAX_BYTES = int(os.environ.get("CODEX_API_STDOUT_MAX_BYTES", str(64 * 1024 * 1024)))
STREAM_READER_LIMIT_BYTES = int(
    os.environ.get("CODEX_API_STREAM_READER_LIMIT_BYTES", str(64 * 1024 * 1024))
)


class FrameInput(BaseModel):
    image_url: str = Field(..., description="JPEG data URL")
    timestamp: float | None = None
    timestamp_kind: str | None = None
    pts: int | None = None


class SolveRequest(BaseModel):
    session_id: str
    task_id: str
    question: str
    foreground_text: str = ""
    frames: list[FrameInput] = Field(default_factory=list)
    max_subagents: int | None = None
    timeout_seconds: float | None = None


class SolveResponse(BaseModel):
    status: Literal["completed", "failed", "timeout"]
    text: str
    thread_id: str | None = None
    usage: dict[str, Any] | None = None
    duration_ms: float
    events_digest: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@dataclass
class JsonlState:
    thread_id: str | None = None
    usage: dict[str, Any] | None = None
    final_message: str = ""
    event_counts: dict[str, int] = field(default_factory=dict)
    item_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    web_searches: int = 0
    command_executions: int = 0

    def ingest(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self.errors.append(f"invalid-jsonl: {line[:200]}")
            return
        event_type = str(event.get("type") or "unknown")
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        if event_type == "thread.started":
            self.thread_id = event.get("thread_id") or self.thread_id
        elif event_type == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                self.usage = usage
        elif event_type in {"turn.failed", "error"}:
            self.errors.append(json.dumps(event, ensure_ascii=False, default=str)[:1000])

        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type") or "unknown")
            self.item_counts[item_type] = self.item_counts.get(item_type, 0) + 1
            if event_type == "item.completed" and item_type == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    self.final_message = text.strip()
            if item_type == "web_search":
                self.web_searches += 1
            elif item_type == "command_execution":
                self.command_executions += 1

    def digest(self) -> dict[str, Any]:
        return {
            "event_counts": self.event_counts,
            "item_counts": self.item_counts,
            "errors": self.errors[-5:],
            "web_searches": self.web_searches,
            "command_executions": self.command_executions,
        }


app = FastAPI(title="StreamingHarness Codex API", version="0.1.0")
_run_semaphore = asyncio.Semaphore(max(1, DEFAULT_MAX_CONCURRENT_RUNS))


@app.get("/health")
async def health() -> dict[str, Any]:
    codex_path = shutil.which("codex")
    version = ""
    if codex_path:
        try:
            proc = await asyncio.create_subprocess_exec(
                codex_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            version = stdout.decode(errors="replace").strip()
        except Exception:
            version = ""
    config_path = Path(DEFAULT_CODEX_HOME) / "config.toml"
    return {
        "status": "ok",
        "codex_path": codex_path,
        "codex_version": version,
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "workspace": DEFAULT_WORKSPACE,
        "yolo": True,
        "web_search": "live",
        "max_subagents": DEFAULT_MAX_SUBAGENTS,
        "max_concurrent_runs": DEFAULT_MAX_CONCURRENT_RUNS,
        "stdout_max_bytes": STDOUT_MAX_BYTES,
        "stream_reader_limit_bytes": STREAM_READER_LIMIT_BYTES,
    }


@app.post("/v1/solve", response_model=SolveResponse)
async def solve(request: SolveRequest) -> SolveResponse:
    async with _run_semaphore:
        return await _solve_with_codex(request)


async def _solve_with_codex(request: SolveRequest) -> SolveResponse:
    codex_path = shutil.which("codex")
    if not codex_path:
        raise HTTPException(status_code=500, detail="codex CLI not found on PATH")

    max_subagents = _bounded_int(
        request.max_subagents,
        default=DEFAULT_MAX_SUBAGENTS,
        minimum=1,
        maximum=DEFAULT_MAX_SUBAGENTS,
    )
    timeout_seconds = _bounded_float(
        request.timeout_seconds,
        default=DEFAULT_TIMEOUT_SECONDS,
        minimum=1.0,
        maximum=DEFAULT_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="streamingharness-codex-") as tmpdir:
        image_paths = _write_frame_images(_limit_frames(request.frames), Path(tmpdir))
        argv = _build_codex_argv(
            codex_path=codex_path,
            workspace=DEFAULT_WORKSPACE,
            image_paths=image_paths,
            max_subagents=max_subagents,
        )
        prompt = _build_prompt(request, max_subagents)
        state = JsonlState()
        stderr_chunks: list[bytes] = []
        stdout_bytes = 0

        codex_env = {**os.environ, "CODEX_HOME": DEFAULT_CODEX_HOME}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=STREAM_READER_LIMIT_BYTES,
            env=codex_env,
        )

        async def read_stdout() -> None:
            nonlocal stdout_bytes
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                stdout_bytes += len(line)
                if stdout_bytes > STDOUT_MAX_BYTES:
                    raise RuntimeError("codex stdout exceeded limit")
                state.ingest(line.decode(errors="replace").strip())

        async def read_stderr() -> None:
            assert proc.stderr is not None
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
                total = sum(len(item) for item in stderr_chunks)
                while total > STDERR_TAIL_BYTES and stderr_chunks:
                    removed = stderr_chunks.pop(0)
                    total -= len(removed)

        stdout_task = asyncio.create_task(read_stdout())
        stderr_task = asyncio.create_task(read_stderr())
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
            await stdout_task
            await stderr_task
        except asyncio.TimeoutError:
            _terminate_process_group(proc)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            return SolveResponse(
                status="timeout",
                text="",
                thread_id=state.thread_id,
                usage=state.usage,
                duration_ms=(time.perf_counter() - started) * 1000,
                events_digest=state.digest(),
                error=f"Codex run timed out after {timeout_seconds:.0f}s",
            )
        except Exception as err:
            _terminate_process_group(proc)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            return SolveResponse(
                status="failed",
                text=state.final_message,
                thread_id=state.thread_id,
                usage=state.usage,
                duration_ms=(time.perf_counter() - started) * 1000,
                events_digest=state.digest(),
                error=str(err),
            )

        stderr_tail = b"".join(stderr_chunks)[-STDERR_TAIL_BYTES:].decode(errors="replace")
        duration_ms = (time.perf_counter() - started) * 1000
        if proc.returncode != 0:
            return SolveResponse(
                status="failed",
                text=state.final_message,
                thread_id=state.thread_id,
                usage=state.usage,
                duration_ms=duration_ms,
                events_digest={**state.digest(), "stderr_tail": stderr_tail},
                error=f"codex exited with {proc.returncode}",
            )
        if not state.final_message:
            return SolveResponse(
                status="failed",
                text="",
                thread_id=state.thread_id,
                usage=state.usage,
                duration_ms=duration_ms,
                events_digest={**state.digest(), "stderr_tail": stderr_tail},
                error="codex returned no final agent message",
            )
        return SolveResponse(
            status="completed",
            text=state.final_message,
            thread_id=state.thread_id,
            usage=state.usage,
            duration_ms=duration_ms,
            events_digest={**state.digest(), "stderr_tail": stderr_tail},
            error=None,
        )


def _build_codex_argv(
    *,
    codex_path: str,
    workspace: str,
    image_paths: list[Path],
    max_subagents: int,
) -> list[str]:
    argv = [
        codex_path,
        "--search",
        "exec",
        "--json",
        "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        workspace,
        "-c",
        f"agents.max_threads={max_subagents}",
        "-c",
        "agents.max_depth=1",
    ]
    for image_path in image_paths:
        argv.extend(["-i", str(image_path)])
    argv.append("-")
    return argv


def _build_prompt(request: SolveRequest, max_subagents: int) -> str:
    frame_lines = []
    for index, frame in enumerate(request.frames, start=1):
        timestamp = frame.timestamp if frame.timestamp is not None else "unknown"
        timestamp_kind = frame.timestamp_kind or "unknown"
        pts = frame.pts if frame.pts is not None else "unknown"
        frame_lines.append(
            f"- Frame {index}: timestamp={timestamp} kind={timestamp_kind} pts={pts}"
        )
    frame_context = "\n".join(frame_lines) if frame_lines else "- No recent frames were provided."
    return f"""You are the Codex background solver for a real-time video assistant.

Use Chinese by default for user-facing prose unless the user explicitly asks otherwise.
Use live web search when current or external information is useful.
You may spawn at most {max_subagents} parallel subagents. Do not exceed this limit.
If you spawn subagents, wait for all of them and consolidate their useful results.
The answer is isolated background UI output. Do not write files unless the user explicitly requested an artifact and it is necessary for analysis; return the final content in the response.
For any visual deliverable request, including image generation, posters, illustrations, avatars, cartoon characters, or PPT/slides, default to imagegen / gpt-image-2 to generate real PNG/JPG assets; do not substitute Python/SVG/HTML/CSS drawings unless the user explicitly asks for code or vector output.
If you create a user-visible file artifact, save it under the current working directory. In the final response, include the existing artifact file path as plain text, not in backticks or a code block, and do not return a directory path.
At the very end of your final response, include a concise summary wrapped exactly as <summary>...</summary>. The text inside must be 1-2 Chinese sentences for the frontend summary card.
If a chart is useful, include a fenced JSON block like {{"type":"bar_chart","title":"...","labels":[],"values":[]}}.
If asked to recreate a visible webpage, return a complete static HTML document in a fenced html code block.

Session: {request.session_id}
Task: {request.task_id}
Foreground note: {request.foreground_text}
Delegated question:
{request.question}

Recent frame metadata:
{frame_context}
"""


def _write_frame_images(frames: list[FrameInput], directory: Path) -> list[Path]:
    paths = []
    for index, frame in enumerate(frames, start=1):
        data = _decode_data_url(frame.image_url)
        path = directory / f"frame-{index:04d}.jpg"
        path.write_bytes(data)
        paths.append(path)
    return paths


def _limit_frames(frames: list[FrameInput]) -> list[FrameInput]:
    if DEFAULT_MAX_FRAMES <= 0:
        return []
    return list(frames or [])[-DEFAULT_MAX_FRAMES:]


def _decode_data_url(value: str) -> bytes:
    prefix = "base64,"
    marker = value.find(prefix)
    if not value.startswith("data:image/") or marker < 0:
        raise HTTPException(status_code=400, detail="frame image_url must be an image data URL")
    try:
        return base64.b64decode(value[marker + len(prefix) :], validate=True)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"invalid frame image data: {err}") from err


def _bounded_int(value: int | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        resolved = int(value if value is not None else default)
    except (TypeError, ValueError):
        resolved = default
    return min(max(resolved, minimum), maximum)


def _bounded_float(value: float | None, *, default: float, minimum: float, maximum: float) -> float:
    try:
        resolved = float(value if value is not None else default)
    except (TypeError, ValueError):
        resolved = default
    return min(max(resolved, minimum), maximum)


def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    try:
        if proc.returncode is None and proc.pid:
            os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.terminate()
        except ProcessLookupError:
            return


def main() -> None:
    import uvicorn

    uvicorn.run("codex_api.main:app", host=DEFAULT_HOST, port=DEFAULT_PORT, reload=False)


if __name__ == "__main__":
    sys.exit(main())
