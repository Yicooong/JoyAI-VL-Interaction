
"""Background agent delegation for long-running or high-risk VLM questions."""

import asyncio
import base64
import io
import html
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import httpx
from PIL import Image

try:
    from .local_file_server import (
        LOCAL_HTML_ARTIFACT_DIR,
        local_file_url_for_path,
        rewrite_payload_local_file_links,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _html_artifact_dir = Path(os.environ.get("LIVE_VLM_HTML_ARTIFACT_DIR", "html")).expanduser()
    if not _html_artifact_dir.is_absolute():
        _html_artifact_dir = PROJECT_ROOT / _html_artifact_dir
    LOCAL_HTML_ARTIFACT_DIR = _html_artifact_dir

    def local_file_url_for_path(path_value: str) -> str:
        return ""

    def rewrite_payload_local_file_links(payload: dict) -> dict:
        return payload


def _legacy_background_env(suffix: str, default: str = "") -> str:
    legacy_name = "BACKGROUND_" + "CO" + "DEX_" + suffix
    return os.environ.get(legacy_name, default)


BACKGROUND_AGENT_API_URL = os.environ.get(
    "BACKGROUND_AGENT_API_URL",
    _legacy_background_env("API_URL", "http://127.0.0.1:8079"),
)
BACKGROUND_MODEL_PROVIDER = "Background Agent"
BACKGROUND_MODEL_NAME = os.environ.get("BACKGROUND_MODEL_NAME", "background-agent")
BACKGROUND_TIMEOUT_SECONDS = 600.0
BACKGROUND_MAX_SUBAGENTS = int(
    os.environ.get(
        "BACKGROUND_MAX_SUBAGENTS",
        os.environ.get(
            "BACKGROUND_AGENT_MAX_SUBAGENTS",
            _legacy_background_env("MAX_SUBAGENTS", "6"),
        ),
    )
)
BACKGROUND_SUMMARIZER_API_BASE = os.environ.get(
    "BACKGROUND_SUMMARIZER_API_BASE",
    os.environ.get("LIVE_VLM_API_BASE", "http://127.0.0.1:8070/v1"),
)
BACKGROUND_SUMMARIZER_MODEL = os.environ.get(
    "BACKGROUND_SUMMARIZER_MODEL",
    os.environ.get("SUMMARIZER_MODEL", "/tmp/models/Qwen3-VL-4B-Instruct"),
)
BACKGROUND_SUMMARIZER_TIMEOUT_SECONDS = float(os.environ.get("BACKGROUND_SUMMARIZER_TIMEOUT_SECONDS", "60"))
BACKGROUND_SUMMARIZER_MAX_TOKENS = int(os.environ.get("BACKGROUND_SUMMARIZER_MAX_TOKENS", "256"))
BACKGROUND_FRAME_MULTIPLIER = 2
BACKGROUND_DEFAULT_FOREGROUND_FPS = 1.0
BACKGROUND_DEFAULT_MAX_FRAMES = 100
BACKGROUND_MAX_FRAME_LIMIT = 100
BACKGROUND_FRAME_RESIZE_LONG_EDGE = 768
BACKGROUND_FRAME_JPEG_QUALITY = 82
BACKGROUND_HANDOFF_MAX_CHARS = 3500
BACKGROUND_HTML_ARTIFACT_DIR = LOCAL_HTML_ARTIFACT_DIR
BACKGROUND_DELEGATION_FOREGROUND_TEXT = (
    "</response> 这个问题需要调用后台模型，请稍等，你可以在此期间继续向我提问"
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}

_THINK_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)[^>]*>[\s\S]*?</(?:think|thinking|reasoning)>",
    flags=re.IGNORECASE,
)
_LEADING_THINK_TAG_RE = re.compile(
    r"^\s*<(?:think|thinking|reasoning)[^>]*>",
    flags=re.IGNORECASE,
)
_LOOSE_THINK_TAG_RE = re.compile(
    r"</?(?:think|thinking|reasoning)[^>]*>",
    flags=re.IGNORECASE,
)
_FINAL_ANSWER_LABEL_RE = re.compile(
    r"(?im)^\s*(?:final answer|answer|final response|最终答案|正式答案|给用户的答案)\s*[:：]\s*"
)
_RENDERABLE_FRAGMENT_RE = re.compile(
    r"<(?:body|main|section|article|aside|header|footer|nav|div|h[1-6]|p|button|ul|ol|li|span|img|figure|table)\b",
    flags=re.IGNORECASE,
)
_INTERNAL_QUESTION_MARKER_RE = re.compile(
    r"\s*\[(?:[^\]]*\bUser\s+(?:Query|Prompt|Request)\b[^\]]*|[^\]]*\bIMPORTANT\b[^\]]*)\]\s*",
    flags=re.IGNORECASE,
)
_SUMMARY_TAG_RE = re.compile(r"<summary\b[^>]*>([\s\S]*?)</summary>", flags=re.IGNORECASE)


def _shorten_log_text(value, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def _normalize_user_facing_background_terms(text: str) -> str:
    value = str(text or "")
    return value.replace("背景模型", "后台模型").replace("背景 model", "后台模型")


def clean_delegation_question_for_display(question: str) -> str:
    """Remove internal routing markers while preserving the actual user question."""
    value = str(question or "").strip()
    if not value:
        return ""
    value = _INTERNAL_QUESTION_MARKER_RE.sub(" ", value)
    value = re.sub(
        r"^\s*请(?:回答|解答)这道(?:简答题|数学题|题目|问题)[，,、\s]*"
        r"(?:并(?:提供|给出)必要的推理过程和最终答案)?[：:]\s*",
        "",
        value,
    )
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"([:：])\s+", r"\1", value)
    value = re.sub(r"\s+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _is_internal_question_marker_only(question: str) -> bool:
    value = str(question or "").strip()
    if not value:
        return False
    return not _INTERNAL_QUESTION_MARKER_RE.sub(" ", value).strip()


def strip_background_thinking(text: str) -> str:
    """Remove explicit reasoning wrapper tags before rendering or foreground handoff."""
    value = str(text or "").strip()
    if not value:
        return ""

    value = _THINK_BLOCK_RE.sub("", value).strip()
    value = _strip_unclosed_leading_think(value)
    value = _LOOSE_THINK_TAG_RE.sub("", value).strip()

    label_match = _FINAL_ANSWER_LABEL_RE.search(value)
    if label_match and label_match.start() <= 4000:
        value = value[label_match.end() :].strip()

    return value


def _strip_unclosed_leading_think(value: str) -> str:
    if not _LEADING_THINK_TAG_RE.match(value):
        return value.strip()
    body = _LEADING_THINK_TAG_RE.sub("", value, count=1).strip()
    return body


@dataclass(frozen=True)
class DelegationRequest:
    foreground_text: str
    question: str
    original_foreground_text: str = ""
    raw_text: str = ""
    raw_question: str = ""


@dataclass
class BackgroundFrame:
    image: Image.Image
    timestamp: Optional[float] = None
    timestamp_kind: Optional[str] = None
    pts: Optional[int] = None


def parse_delegation(text: str) -> Optional[DelegationRequest]:
    """Extract foreground response text and delegated question from model output."""
    if not text:
        return None

    raw_text = str(text)
    internal_call = _parse_internal_background_call(raw_text)
    if internal_call:
        return internal_call

    lower_text = raw_text.lower()
    tag_matches = [
        (lower_text.find("</delegation>"), len("</delegation>")),
        (lower_text.find("<delegation>"), len("<delegation>")),
    ]
    tag_matches = [(index, size) for index, size in tag_matches if index >= 0]
    if not tag_matches:
        return None

    tag_index, tag_size = min(tag_matches, key=lambda item: item[0])
    original_foreground_text = raw_text[:tag_index].strip()
    raw_question = raw_text[tag_index + tag_size :].strip()
    question = clean_delegation_question_for_display(raw_question)
    if not question and not _is_internal_question_marker_only(raw_question):
        return None

    return DelegationRequest(
        foreground_text=BACKGROUND_DELEGATION_FOREGROUND_TEXT,
        question=question,
        original_foreground_text=_normalize_user_facing_background_terms(original_foreground_text),
        raw_text=raw_text,
        raw_question=raw_question,
    )


def _parse_internal_background_call(text: str) -> Optional[DelegationRequest]:
    tag = "<|background_call|>"
    tag_index = text.find(tag)
    if tag_index < 0:
        return None

    original_foreground_text = text[:tag_index].strip()
    call_payload = text[tag_index + len(tag) :].strip()
    parsed_payload = _extract_json_object(call_payload)
    question = ""
    if parsed_payload:
        query = parsed_payload.get("query")
        if isinstance(query, str):
            question = query.strip()
    if not question:
        question = call_payload.strip()
    raw_question = question
    question = clean_delegation_question_for_display(raw_question)
    if not question and not _is_internal_question_marker_only(raw_question):
        return None

    return DelegationRequest(
        foreground_text=BACKGROUND_DELEGATION_FOREGROUND_TEXT,
        question=question,
        original_foreground_text=_normalize_user_facing_background_terms(original_foreground_text),
        raw_text=text,
        raw_question=raw_question,
    )


def _extract_json_object(text: str) -> Optional[dict]:
    decoder = json.JSONDecoder()
    value = str(text or "").strip()
    if not value:
        return None
    for index, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def build_background_handoff_summary(
    question: str,
    result: str,
    rich: Optional[dict] = None,
    *,
    max_chars: int = BACKGROUND_HANDOFF_MAX_CHARS,
) -> str:
    """Create a bounded digest that the interaction model can safely use."""
    answer = str(result or "").strip()
    if not answer:
        return ""

    rich = rich or extract_rich_content(answer)
    parts = [f"用户委托问题: {str(question or '').strip()}"]

    chart = rich.get("chart") if isinstance(rich, dict) else None
    if chart:
        parts.append(
            "后台产物: 柱状图"
            + (f"《{chart.get('title')}》" if chart.get("title") else "")
            + "。完整图表和数据留在后台结果气泡中，interaction 只读取自然语言摘要。"
        )

    html_text = rich.get("html") if isinstance(rich, dict) else ""
    if html_text:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
        title = html.unescape(title_match.group(1)).strip() if title_match else ""
        parts.append(
            "后台产物: 静态 HTML/网页预览"
            + (f"，标题: {title}" if title else "")
            + f"，HTML 长度约 {len(html_text)} 字符"
            + ("，可能被截断" if rich.get("html_incomplete") else "")
            + "。完整内容留在后台结果气泡中，interaction 只读取这份摘要。"
        )

    explanation = _strip_structured_blocks_for_handoff(answer).strip()
    if not explanation and not (chart or html_text):
        explanation = answer
    if explanation:
        parts.append("核心内容摘录:\n" + _truncate_handoff_text(explanation, max_chars // 2))

    return _truncate_handoff_text("\n\n".join(part for part in parts if part), max_chars)


def _strip_structured_blocks_for_handoff(text: str) -> str:
    value = str(text or "")
    value = re.sub(
        r"```[ \t]*(?:html|json)[ \t]*\r?\n[\s\S]*?(?:\r?\n```|$)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:以下|下面|上方|下方)?[^。！？!?\n]*(?:JSON|json|代码块|图表数据|绘图数据|可直接用于绘图|可视化数据|chart)[^。！？!?\n]*[。！？!?]?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    start_matches = [
        match.start()
        for match in (
            re.search(r"<!doctype\s+html\b", value, flags=re.IGNORECASE),
            re.search(r"<html\b", value, flags=re.IGNORECASE),
        )
        if match
    ]
    if start_matches:
        value = value[: min(start_matches)]
    return value.strip()


def _truncate_handoff_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\n{3,}", "\n\n", str(text or "").strip())
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 20)].rstrip() + "\n...[已截断]"


class BackgroundModelService:
    """Caches recent frames and solves delegated questions asynchronously."""

    def __init__(
        self,
        session_id: str,
        notify_callback: Optional[Callable[[dict], None]] = None,
        api_base: str = BACKGROUND_AGENT_API_URL,
        model: str = BACKGROUND_MODEL_NAME,
        max_subagents: int = BACKGROUND_MAX_SUBAGENTS,
        api_key: Optional[str] = None,
        wire_api: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        disable_response_storage: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: float = BACKGROUND_TIMEOUT_SECONDS,
        summarizer_api_base: str = BACKGROUND_SUMMARIZER_API_BASE,
        summarizer_model: str = BACKGROUND_SUMMARIZER_MODEL,
        summarizer_timeout_seconds: float = BACKGROUND_SUMMARIZER_TIMEOUT_SECONDS,
        frame_multiplier: int = BACKGROUND_FRAME_MULTIPLIER,
        max_frames: Optional[int] = None,
        foreground_fps: float = BACKGROUND_DEFAULT_FOREGROUND_FPS,
        resize_long_edge: int = BACKGROUND_FRAME_RESIZE_LONG_EDGE,
        jpeg_quality: int = BACKGROUND_FRAME_JPEG_QUALITY,
        enabled: bool = True,
    ):
        self.session_id = session_id
        self.notify_callback = notify_callback
        self.api_base = str(api_base or BACKGROUND_AGENT_API_URL).rstrip("/")
        self.model = model
        self.max_subagents = max(1, int(max_subagents or BACKGROUND_MAX_SUBAGENTS))
        self._legacy_api_key = api_key
        self._legacy_wire_api = wire_api
        self._legacy_reasoning_effort = reasoning_effort
        self._legacy_disable_response_storage = disable_response_storage
        self._legacy_max_tokens = max_tokens
        self.timeout_seconds = float(timeout_seconds)
        self.summarizer_api_base = str(summarizer_api_base or BACKGROUND_SUMMARIZER_API_BASE).rstrip("/")
        self.summarizer_model = str(summarizer_model or BACKGROUND_SUMMARIZER_MODEL)
        self.summarizer_timeout_seconds = max(1.0, float(summarizer_timeout_seconds))
        self.frame_multiplier = max(1, int(frame_multiplier))
        self.max_frame_count = self._normalize_max_frames(
            max_frames if max_frames is not None else BACKGROUND_DEFAULT_MAX_FRAMES
        )
        self.foreground_fps = self._normalize_foreground_fps(foreground_fps)
        self.resize_long_edge = self._normalize_resize_long_edge(resize_long_edge)
        self.jpeg_quality = min(max(40, int(jpeg_quality)), 95)
        self.enabled = bool(enabled)
        self._frame_buffer: deque[BackgroundFrame] = deque()
        self._active_tasks: set[asyncio.Task] = set()
        self._closed = False
        self._task_sequence = 0
        self._last_cached_frame_time: Optional[float] = None
        self._resize_frame_buffer()

    def _normalize_max_frames(self, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        try:
            max_frames = int(value)
        except (TypeError, ValueError):
            return None
        if max_frames <= 0:
            return BACKGROUND_DEFAULT_MAX_FRAMES
        return min(max_frames, BACKGROUND_MAX_FRAME_LIMIT)

    def _normalize_foreground_fps(self, value) -> float:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            return BACKGROUND_DEFAULT_FOREGROUND_FPS
        if fps <= 0:
            return BACKGROUND_DEFAULT_FOREGROUND_FPS
        return min(fps, 60.0)

    def _normalize_resize_long_edge(self, value) -> int:
        try:
            long_edge = int(value)
        except (TypeError, ValueError):
            return BACKGROUND_FRAME_RESIZE_LONG_EDGE
        if long_edge <= 0:
            return 0
        return min(max(256, long_edge), 2048)

    def _background_fps(self) -> float:
        return min(
            max(0.1, self.foreground_fps * float(self.frame_multiplier)),
            60.0,
        )

    def _sample_interval_seconds(self) -> float:
        return 1.0 / self._background_fps()

    def _target_frame_count(self) -> int:
        return min(max(1, int(self.max_frame_count or BACKGROUND_DEFAULT_MAX_FRAMES)), BACKGROUND_MAX_FRAME_LIMIT)

    def _resize_frame_buffer(self) -> None:
        maxlen = self._target_frame_count()
        if self._frame_buffer.maxlen == maxlen:
            return
        self._frame_buffer = deque(list(self._frame_buffer)[-maxlen:], maxlen=maxlen)

    def get_config(self) -> dict:
        return {
            "enabled": self.enabled,
            "api_base": self.api_base,
            "model": self.model,
            "provider": BACKGROUND_MODEL_PROVIDER,
            "web_search_enabled": True,
            "sandbox": "yolo",
            "max_subagents": self.max_subagents,
            "timeout_seconds": self.timeout_seconds,
            "summarizer_api_base": self.summarizer_api_base,
            "summarizer_model": self.summarizer_model,
            "summarizer_timeout_seconds": self.summarizer_timeout_seconds,
            "frame_multiplier": self.frame_multiplier,
            "max_frames": self.max_frame_count,
            "foreground_fps": self.foreground_fps,
            "background_fps": self._background_fps(),
            "sample_interval_seconds": self._sample_interval_seconds(),
            "target_frame_count": self._target_frame_count(),
            "cached_frame_count": len(self._frame_buffer),
            "resize_long_edge": self.resize_long_edge,
            "jpeg_quality": self.jpeg_quality,
        }

    def update_config(
        self,
        *,
        enabled: Optional[bool] = None,
        frame_multiplier: Optional[int] = None,
        max_frames: Optional[int] = None,
        foreground_fps: Optional[float] = None,
        resize_long_edge: Optional[int] = None,
    ) -> dict:
        if enabled is not None:
            self.enabled = bool(enabled)
        if frame_multiplier is not None:
            self.frame_multiplier = min(max(1, int(frame_multiplier)), 10)
        if max_frames is not None:
            self.max_frame_count = self._normalize_max_frames(max_frames)
        if foreground_fps is not None:
            self.foreground_fps = self._normalize_foreground_fps(foreground_fps)
        if resize_long_edge is not None:
            self.resize_long_edge = self._normalize_resize_long_edge(resize_long_edge)
        self._resize_frame_buffer()
        return self.get_config()

    def update_summary_api(
        self,
        *,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict:
        if api_base:
            self.summarizer_api_base = str(api_base).rstrip("/")
        if model:
            self.summarizer_model = str(model)
        return self.get_config()

    def set_foreground_frames_per_batch(self, frames_per_batch: int) -> None:
        # Kept for older callers; background sampling is based on foreground FPS.
        self._resize_frame_buffer()

    def set_foreground_sampling(
        self,
        *,
        process_interval_seconds: Optional[float] = None,
        frames_per_batch: Optional[int] = None,
    ) -> None:
        if process_interval_seconds is None:
            return
        try:
            interval = float(process_interval_seconds)
            batch = max(1, int(frames_per_batch or 1))
        except (TypeError, ValueError):
            return
        if interval <= 0:
            return
        self.foreground_fps = self._normalize_foreground_fps(batch / interval)

    def should_sample_frame(self, wall_time: Optional[float] = None) -> bool:
        if self._closed or not self.enabled:
            return False
        sample_time = float(wall_time if wall_time is not None else time.time())
        return (
            self._last_cached_frame_time is None
            or sample_time - self._last_cached_frame_time >= self._sample_interval_seconds()
        )

    def add_frame(
        self,
        image: Image.Image,
        *,
        timestamp: Optional[float] = None,
        timestamp_kind: Optional[str] = None,
        pts: Optional[int] = None,
        foreground_frames_per_batch: Optional[int] = None,
        wall_time: Optional[float] = None,
    ) -> None:
        if self._closed or image is None:
            return
        if foreground_frames_per_batch is not None:
            self.set_foreground_frames_per_batch(foreground_frames_per_batch)

        sample_time = float(wall_time if wall_time is not None else time.time())
        if not self.should_sample_frame(sample_time):
            return

        try:
            image_copy = self._prepare_frame_image(image)
        except Exception:
            logger.warning("[%s] Failed to copy background frame", self.session_id, exc_info=True)
            return

        self._last_cached_frame_time = sample_time
        self._frame_buffer.append(
            BackgroundFrame(
                image=image_copy,
                timestamp=timestamp,
                timestamp_kind=timestamp_kind,
                pts=pts,
            )
        )

    def _prepare_frame_image(self, image: Image.Image) -> Image.Image:
        image_copy = image.copy()
        if image_copy.mode != "RGB":
            image_copy = image_copy.convert("RGB")
        if self.resize_long_edge:
            width, height = image_copy.size
            long_edge = max(width, height)
            if long_edge > self.resize_long_edge:
                scale = self.resize_long_edge / float(long_edge)
                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                image_copy = image_copy.resize(new_size, resampling)
        return image_copy

    def _snapshot_frames(self) -> list[BackgroundFrame]:
        return list(self._frame_buffer)

    def _notify(self, payload: dict) -> None:
        if not self.notify_callback:
            return
        try:
            self.notify_callback(payload)
        except Exception:
            logger.warning("[%s] Failed to send background notification", self.session_id, exc_info=True)

    def handle_foreground_response(self, text: str, metrics: Optional[dict] = None) -> str:
        """Start background work when a foreground response contains delegation."""
        delegation = parse_delegation(text)
        if not delegation:
            return text
        original_question = clean_delegation_question_for_display(
            str((metrics or {}).get("user_prompt") or "")
        )
        if original_question:
            delegation = replace(delegation, question=original_question)
        if not delegation.question:
            fallback_question = original_question
            if fallback_question:
                logger.warning(
                    "[%s] Background delegation used user prompt fallback because delegated question was only an internal marker: raw_question=%s fallback=%s",
                    self.session_id,
                    _shorten_log_text(delegation.raw_question),
                    _shorten_log_text(fallback_question),
                )
                delegation = replace(delegation, question=fallback_question)
            else:
                logger.warning(
                    "[%s] Background delegation skipped: delegated question was only an internal marker and no user prompt fallback was available: raw_question=%s",
                    self.session_id,
                    _shorten_log_text(delegation.raw_question),
                )
                return delegation.foreground_text
        if not self.enabled or self._closed:
            logger.info(
                "[%s] Background delegation skipped: enabled=%s closed=%s question=%s",
                self.session_id,
                self.enabled,
                self._closed,
                _shorten_log_text(delegation.question),
            )
            return delegation.foreground_text

        self._task_sequence += 1
        task_id = f"bg-{self.session_id}-{self._task_sequence}-{uuid.uuid4().hex[:8]}"
        frames = self._snapshot_frames()
        logger.info(
            "[%s] Background delegation queued: task_id=%s frames=%s target_frames=%s question=%s",
            self.session_id,
            task_id,
            len(frames),
            self._target_frame_count(),
            _shorten_log_text(delegation.question),
        )
        task = asyncio.create_task(
            self._run_delegation_task(task_id, delegation, frames, metrics or {}),
            name=f"background-delegation:{self.session_id}:{self._task_sequence}",
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._discard_task)
        return delegation.foreground_text

    def _discard_task(self, task: asyncio.Task) -> None:
        self._active_tasks.discard(task)
        if task.cancelled():
            return
        try:
            err = task.exception()
        except asyncio.CancelledError:
            return
        if err:
            logger.warning(
                "[%s] Background delegation task ended with unhandled error: %s",
                self.session_id,
                err,
                exc_info=(type(err), err, err.__traceback__),
            )

    async def _run_delegation_task(
        self,
        task_id: str,
        delegation: DelegationRequest,
        frames: list[BackgroundFrame],
        metrics: dict,
    ) -> None:
        start_time = time.perf_counter()
        user_question = str(delegation.question or "").strip()
        try:
            await asyncio.sleep(0)
            logger.info(
                "[%s] Background delegation started: task_id=%s frames=%s model=%s",
                self.session_id,
                task_id,
                len(frames),
                self.model,
            )
            self._notify(
                {
                    "type": "background_task_started",
                    "task_id": task_id,
                    "question": user_question,
                    "foreground_text": delegation.foreground_text,
                    "original_foreground_text": delegation.original_foreground_text,
                    "context_excluded": True,
                    "frame_count": len(frames),
                    "model": self.model,
                    "metrics": metrics,
                }
            )
            result_payload = await asyncio.wait_for(
                self.solve_delegation(
                    task_id=task_id,
                    question=user_question,
                    foreground_text=delegation.foreground_text,
                    frames=frames,
                ),
                timeout=self.timeout_seconds,
            )
            solve_latency_ms = (time.perf_counter() - start_time) * 1000
            result_payload = self._normalize_background_result(result_payload)
            result = result_payload["text"]
            summary_text = result_payload.get("summary_text") or result_payload.get("background_summary") or ""
            rich = result_payload.get("rich") or extract_rich_content(result)
            rich = ensure_html_artifact(rich, task_id=task_id)
            handoff = summary_text or build_background_handoff_summary(user_question, result, rich)
            summary_latency_ms = 0.0
            latency_ms = (time.perf_counter() - start_time) * 1000
            background_metrics = result_payload.get("metrics") or {}
            logger.info(
                "[%s] Background result ready: chars=%s summary_chars=%s handoff_chars=%s solve_latency_ms=%.0f html_chars=%s html_incomplete=%s chart=%s",
                self.session_id,
                len(result or ""),
                len(summary_text or ""),
                len(handoff or ""),
                solve_latency_ms,
                len(rich.get("html") or ""),
                bool(rich.get("html_incomplete")),
                bool(rich.get("chart")),
            )
            self._notify(
                rewrite_payload_local_file_links(
                    {
                        "type": "background_result_ready",
                        "task_id": task_id,
                        "question": user_question,
                        "foreground_text": delegation.foreground_text,
                        "original_foreground_text": delegation.original_foreground_text,
                        "text": result,
                        "rich": rich,
                        "summary_text": summary_text,
                        "background_summary": summary_text or result,
                        "interaction_handoff": {
                            "digest_only": True,
                            "summary": handoff,
                            "source": "background_model",
                        },
                        "context_excluded": True,
                        "model": self.model,
                        "metrics": {
                            "latency_ms": latency_ms,
                            "solve_latency_ms": solve_latency_ms,
                            "summary_latency_ms": summary_latency_ms,
                            "frame_count": len(frames),
                            "background_service": background_metrics,
                        },
                    },
                )
            )
        except asyncio.CancelledError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "[%s] Background delegation cancelled: task_id=%s latency_ms=%.0f",
                self.session_id,
                task_id,
                latency_ms,
            )
            raise
        except Exception as err:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("[%s] Background delegation failed: %s", self.session_id, err)
            self._notify(
                {
                    "type": "background_result_error",
                    "task_id": task_id,
                    "question": user_question,
                    "foreground_text": delegation.foreground_text,
                    "original_foreground_text": delegation.original_foreground_text,
                    "error": str(err),
                    "context_excluded": True,
                    "model": self.model,
                    "metrics": {
                        "latency_ms": latency_ms,
                        "frame_count": len(frames),
                    },
                }
            )

    async def solve_delegation(
        self,
        *,
        task_id: str,
        question: str,
        foreground_text: str,
        frames: Optional[list[BackgroundFrame]] = None,
    ) -> dict:
        frames = frames if frames is not None else self._snapshot_frames()
        payload = {
            "session_id": self.session_id,
            "task_id": task_id,
            "question": str(question or ""),
            "foreground_text": str(foreground_text or ""),
            "max_subagents": self.max_subagents,
            "timeout_seconds": self.timeout_seconds,
            "frames": [self._frame_to_payload(frame) for frame in frames],
        }
        timeout = httpx.Timeout(self.timeout_seconds + 5.0, connect=10.0)
        url = f"{self.api_base}/v1/solve"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        return self._normalize_background_result(data)

    async def summarize_background_result(self, *, question: str, result: str) -> str:
        text = str(result or "").strip()
        if not text:
            return ""
        prompt = (
            "请把下面后台模型的完整回答压缩成给前端摘要卡展示的简短摘要。\n"
            "要求：\n"
            "- 使用中文。\n"
            "- 只输出摘要正文，不要标题、列表编号或客套话。\n"
            "- 1 到 2 句话，保留结论/关键步骤/产物类型。\n"
            "- 不要补充原文没有的信息。\n\n"
            f"用户问题：{str(question or '').strip()}\n\n"
            f"后台完整回答：\n{text}"
        )
        payload = {
            "model": self.summarizer_model,
            "user": f"background-summary:{self.session_id}",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": BACKGROUND_SUMMARIZER_MAX_TOKENS,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        url = f"{self.summarizer_api_base}/chat/completions"
        try:
            timeout = httpx.Timeout(self.summarizer_timeout_seconds, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"x-streaming-session": f"background-summary:{self.session_id}"},
                )
                response.raise_for_status()
                data = response.json()
            summary = self._extract_chat_completion_text(data)
            return self._normalize_summary_text(summary)
        except Exception as err:
            logger.warning(
                "[%s] Background summary model failed: %s",
                self.session_id,
                err,
            )
            return ""

    def _frame_to_payload(self, frame: BackgroundFrame) -> dict:
        return {
            "image_url": self._image_to_data_url(frame.image),
            "timestamp": frame.timestamp,
            "timestamp_kind": frame.timestamp_kind,
            "pts": frame.pts,
        }

    def _normalize_background_result(self, value) -> dict:
        if isinstance(value, str):
            text = strip_background_thinking(value)
            summary, text = self._extract_summary_tag(text)
            return {
                "text": text,
                "rich": None,
                "summary_text": summary,
                "background_summary": summary,
                "metrics": {},
            }
        if not isinstance(value, dict):
            text = str(value or "").strip()
            summary, text = self._extract_summary_tag(text)
            return {
                "text": text,
                "rich": None,
                "summary_text": summary,
                "background_summary": summary,
                "metrics": {},
            }
        status = str(value.get("status") or "completed")
        error = value.get("error")
        if status != "completed":
            raise RuntimeError(str(error or f"Background service returned status={status}"))
        text = self._normalize_text_value(value.get("text") or value.get("answer"))
        text = strip_background_thinking(text)
        tagged_summary, text = self._extract_summary_tag(text)
        if not text:
            raise RuntimeError("Background service returned empty text")
        summary_text = (
            self._normalize_summary_text(value.get("summary_text"))
            or self._normalize_summary_text(value.get("background_summary"))
            or tagged_summary
        )
        rich = value.get("rich") if isinstance(value.get("rich"), dict) else None
        existing_metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else {}
        metrics = {
            "thread_id": value.get("thread_id"),
            "usage": value.get("usage"),
            "duration_ms": value.get("duration_ms"),
            "events_digest": value.get("events_digest"),
        }
        metrics = {**metrics, **existing_metrics}
        return {
            "text": text,
            "rich": rich,
            "summary_text": summary_text,
            "background_summary": summary_text,
            "metrics": metrics,
        }

    def _extract_summary_tag(self, text: str) -> tuple[str, str]:
        value = str(text or "").strip()
        match = _SUMMARY_TAG_RE.search(value)
        if not match:
            return "", value
        summary = self._normalize_summary_text(match.group(1))
        body = (value[: match.start()] + value[match.end() :]).strip()
        return summary, body

    def _extract_chat_completion_text(self, value) -> str:
        if not isinstance(value, dict):
            return ""
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if isinstance(message, dict):
            return self._normalize_text_value(message.get("content"))
        return self._normalize_text_value(first.get("text"))

    def _normalize_summary_text(self, value) -> str:
        text = self._normalize_text_value(value)
        text = strip_background_thinking(text)
        text = re.sub(r"^\s*(?:摘要|总结|简要摘要|简述)\s*[:：]\s*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _normalize_text_value(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                text = ""
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value") or ""
                else:
                    text = (
                        getattr(item, "text", None)
                        or getattr(item, "content", None)
                        or getattr(item, "value", None)
                        or ""
                    )
                normalized = self._normalize_text_value(text)
                if normalized:
                    parts.append(normalized)
            return "\n".join(parts).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                text = self._normalize_text_value(value.get(key))
                if text:
                    return text
            return self._json_preview(value)
        return str(value).strip()

    def _json_preview(self, value) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str, indent=2).strip()
        except Exception:
            return str(value).strip()

    def _format_frame_timestamp(self, frame: BackgroundFrame) -> str:
        if frame.timestamp is None:
            return "timestamp unavailable"
        try:
            timestamp = float(frame.timestamp)
        except (TypeError, ValueError):
            return str(frame.timestamp)
        suffix = "relative seconds" if frame.timestamp_kind == "relative_seconds" else "seconds"
        return f"{timestamp:.3f} {suffix}"

    def _image_to_data_url(self, image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG", quality=self.jpeg_quality, optimize=True)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"

    async def cancel_active_requests(self, timeout: float = 2.0) -> int:
        current_task = asyncio.current_task()
        tasks = [
            task
            for task in self._active_tasks
            if task is not current_task and not task.done()
        ]
        if not tasks:
            return 0

        logger.info("[%s] Cancelling %s background delegation task(s)", self.session_id, len(tasks))
        for task in tasks:
            task.cancel()
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        return len(tasks)

    async def close(self, cancel_requests: bool = True) -> None:
        self._closed = True
        if cancel_requests:
            await self.cancel_active_requests()
        self._frame_buffer.clear()


def extract_rich_content(text: str) -> dict:
    """Extract structured rich content from a background model answer."""
    html_details = extract_html_details(text)
    rich = {
        "html": html_details["html"],
        "html_incomplete": html_details["incomplete"],
        "chart": extract_bar_chart(text),
    }
    return ensure_html_artifact(rich)


def ensure_html_artifact(rich: Optional[dict], *, task_id: str = "") -> dict:
    """Persist generated HTML so users can open a fully clickable page."""
    if not isinstance(rich, dict):
        return rich or {}
    html_text = str(rich.get("html") or "").strip()
    if not html_text:
        return rich
    if str(rich.get("html_url") or "").strip():
        return rich

    artifact_path = _write_html_artifact(html_text, task_id=task_id)
    if not artifact_path:
        return rich
    url = local_file_url_for_path(str(artifact_path))
    if not url:
        return rich
    next_rich = dict(rich)
    next_rich["html_path"] = str(artifact_path)
    next_rich["html_url"] = url
    return next_rich


def _write_html_artifact(html_text: str, *, task_id: str = "") -> Optional[Path]:
    try:
        BACKGROUND_HTML_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(task_id or "").strip()).strip("-")
        filename = f"{safe_task or uuid.uuid4().hex}.html"
        path = BACKGROUND_HTML_ARTIFACT_DIR / filename
        path.write_text(str(html_text or ""), encoding="utf-8")
        return path
    except Exception:
        logger.warning("Failed to write background HTML artifact", exc_info=True)
        return None


def extract_html_document(text: str) -> str:
    return extract_html_details(text)["html"]


def extract_html_details(text: str) -> dict:
    raw_text = html.unescape(str(text or "")).strip()
    if not raw_text:
        return {"html": "", "incomplete": False}

    explicit_candidates = []
    explicit_candidates.extend(_extract_fenced_block_candidates(raw_text, "html", include_unclosed=True))

    for block in _extract_fenced_blocks(raw_text, "json"):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("type") == "html" and isinstance(parsed.get("html"), str):
            explicit_candidates.append({"text": parsed["html"], "incomplete": False})

    best = _pick_best_html_candidate(explicit_candidates)
    if best:
        return {"html": best["html"], "incomplete": best["incomplete"]}

    best = _pick_best_html_candidate([{"text": raw_text, "incomplete": False}])
    if best:
        return {"html": best["html"], "incomplete": best["incomplete"]}
    return {"html": "", "incomplete": False}


def _pick_best_html_candidate(candidates: list[dict]) -> Optional[dict]:
    best = None
    for candidate in candidates:
        document, incomplete = _normalize_html_document_with_status(
            candidate["text"],
            source_incomplete=candidate.get("incomplete", False),
        )
        if document:
            scored = {
                "html": document,
                "incomplete": incomplete,
                "score": _score_html_candidate(document, incomplete),
            }
            if best is None or scored["score"] > best["score"]:
                best = scored
    return best


def extract_bar_chart(text: str) -> Optional[dict]:
    raw_text = str(text or "").strip()
    candidates = _extract_fenced_blocks(raw_text, "json")
    if raw_text.startswith("{") and raw_text.endswith("}"):
        candidates.insert(0, raw_text)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or parsed.get("type") != "bar_chart":
            continue
        labels = parsed.get("labels")
        values = parsed.get("values")
        if not isinstance(labels, list) or not isinstance(values, list) or len(labels) != len(values):
            continue
        try:
            numeric_values = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        return {
            "type": "bar_chart",
            "title": str(parsed.get("title") or ""),
            "labels": [str(label) for label in labels],
            "values": numeric_values,
        }
    return None


def _extract_fenced_blocks(text: str, language: str = "") -> list[str]:
    return [
        candidate["text"]
        for candidate in _extract_fenced_block_candidates(
            text,
            language,
            include_unclosed=False,
        )
    ]


def _extract_fenced_block_candidates(
    text: str,
    language: str = "",
    *,
    include_unclosed: bool = False,
) -> list[dict]:
    blocks = []
    raw_text = str(text or "")
    pattern = re.compile(r"```[ \t]*([a-zA-Z0-9_-]*)[ \t]*\r?\n?", re.MULTILINE)
    language = language.lower()
    for match in pattern.finditer(raw_text):
        lang = (match.group(1) or "").lower()
        if not language or lang == language:
            content_start = match.end()
            closing_index = raw_text.find("```", content_start)
            if closing_index >= 0:
                blocks.append(
                    {
                        "text": raw_text[content_start:closing_index].strip(),
                        "incomplete": False,
                    }
                )
            elif include_unclosed:
                blocks.append(
                    {
                        "text": raw_text[content_start:].strip(),
                        "incomplete": True,
                    }
                )
    return blocks


def _normalize_html_document(value: str) -> str:
    return _normalize_html_document_with_status(value)[0]


def _normalize_html_document_with_status(
    value: str,
    *,
    source_incomplete: bool = False,
) -> tuple[str, bool]:
    candidate = html.unescape(str(value or "")).strip()
    if not candidate:
        return "", False

    candidate = re.sub(r"^```[ \t]*(?:html)?[ \t]*\r?\n?", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"\r?\n?```[ \t]*$", "", candidate).strip()

    start_matches = [
        match.start()
        for match in (
            re.search(r"<!doctype\s+html\b", candidate, flags=re.IGNORECASE),
            re.search(r"<html\b", candidate, flags=re.IGNORECASE),
        )
        if match
    ]
    if start_matches:
        start_index = min(start_matches)
        end_match = re.search(r"</html\s*>", candidate[start_index:], flags=re.IGNORECASE)
        if end_match:
            end_index = start_index + end_match.end()
            return candidate[start_index:end_index].strip(), False
        return candidate[start_index:].strip(), True

    if re.search(r"<(head|body|main|section|style|div|h1|p)\b", candidate, flags=re.IGNORECASE):
        return (
            f"<!doctype html>\n<html><head><meta charset=\"UTF-8\"></head><body>{candidate}</body></html>",
            source_incomplete or _looks_incomplete_html(candidate),
        )

    return "", False


def _score_html_candidate(document: str, incomplete: bool) -> int:
    html_text = str(document or "")
    lower = html_text.lower()
    score = min(len(html_text), 20000)
    if not incomplete:
        score += 5000
    if "<style" in lower:
        score += 2500
    if "<body" in lower:
        score += 1500
    if _RENDERABLE_FRAGMENT_RE.search(html_text):
        score += 1500
    score += min(len(re.findall(r"<(?:div|section|main|header|nav|footer|button|h[1-6]|p|li)\b", html_text, flags=re.IGNORECASE)) * 120, 3000)
    placeholder_count = html_text.count("...") + html_text.count("…")
    score -= placeholder_count * 3000
    score -= len(re.findall(r">\s*\.\.\.\s*<", html_text)) * 5000
    return score


def _looks_incomplete_html(value: str) -> bool:
    candidate = str(value or "")
    checks = (
        (r"<html\b", r"</html\s*>"),
        (r"<body\b", r"</body\s*>"),
        (r"<head\b", r"</head\s*>"),
        (r"<style\b", r"</style\s*>"),
    )
    for open_pattern, close_pattern in checks:
        if re.search(open_pattern, candidate, flags=re.IGNORECASE) and not re.search(
            close_pattern,
            candidate,
            flags=re.IGNORECASE,
        ):
            return True
    return False
