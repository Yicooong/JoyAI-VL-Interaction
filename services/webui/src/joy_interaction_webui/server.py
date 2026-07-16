# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WebRTC Joy VL Interaction Server
Main server that handles WebRTC connections and serves the web interface
"""

import asyncio
import io
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

import aiohttp
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.mediastreams import MediaStreamError

from .vlm_service import VLMService
from .video_processor import VideoProcessorTrack
from .rtsp_track import RTSPVideoTrack
from .asr import setup_asr_routes
from .tts import setup_tts_routes
from .background_model import BackgroundModelService
from .local_file_server import setup_local_file_routes
from .websocket_video import WebSocketVideoTrack

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WEBRTC_TRANSPORT = os.environ.get("WEBRTC_TRANSPORT", "tcp").strip().lower()
if WEBRTC_TRANSPORT not in {"tcp", "udp"}:
    logger.warning(
        "Invalid WEBRTC_TRANSPORT=%r; falling back to tcp", WEBRTC_TRANSPORT
    )
    WEBRTC_TRANSPORT = "tcp"

# Global objects
relay = MediaRelay()
pcs = set()
vlm_service = None  # Kept for backwards compat; default session uses sessions["default"]
websockets = set()  # Track active WebSocket connections (all)
rtsp_tracks = {}  # Track active RTSP streams {session_id: (rtsp_track, processor_track)}
uploaded_videos = {}  # file_id -> {path, session_id}
http_video_tracks = defaultdict(set)  # session_id -> source tracks used by MJPEG responses
http_video_tasks = defaultdict(set)  # parent session_id -> active MJPEG request tasks
_safevl_cache_dir = os.environ.get("SAFEVL_CACHE_DIR", "").strip()
video_upload_dir = Path(
    os.environ.get("LIVE_VLM_UPLOAD_DIR")
    or (
        str(Path(_safevl_cache_dir).expanduser() / "webui" / "uploads")
        if _safevl_cache_dir
        else Path(tempfile.gettempdir()) / "joy-interaction-webui-uploads"
    )
).expanduser()
video_library_dir = Path(
    os.environ.get("LIVE_VLM_VIDEO_DIR", str(Path.cwd() / "videos"))
).expanduser().resolve()

# Multi-session state
default_vlm_config = {}  # Set at startup; used to create new sessions
sessions = {}  # session_id -> {"vlm_service": VLMService}
session_websockets = defaultdict(set)  # session_id -> set of ws
ws_to_session = {}  # ws -> session_id
session_peer_connections = defaultdict(set)  # session_id -> set of RTCPeerConnection
session_children = defaultdict(set)  # parent UI session -> per-video VLM sessions


def notify_session_json(session_id: str, payload: dict):
    """Send a JSON payload to WebSocket clients in this session."""
    handle_background_handoff_for_interaction(session_id, payload)
    send_to_session(session_id, json.dumps(payload, ensure_ascii=False))


def handle_background_handoff_for_interaction(session_id: str, payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("type") != "background_result_ready":
        return

    session = sessions.get(session_id)
    if not session or not session.get("vlm_service"):
        return
    handoff = payload.get("interaction_handoff")
    summary = ""
    if isinstance(handoff, dict):
        summary = str(handoff.get("summary") or "").strip()
    if not summary:
        logger.info(
            "[%s] Background result received without interaction handoff: task_id=%s",
            session_id,
            payload.get("task_id"),
        )
        return
    session["vlm_service"].queue_background_handoff(
        task_id=str(payload.get("task_id") or ""),
        question=str(payload.get("question") or ""),
        summary=summary,
    )
    logger.info(
        "[%s] Background handoff queued for interaction: task_id=%s summary_chars=%s",
        session_id,
        payload.get("task_id"),
        len(summary),
    )


def get_background_service(session_id: str):
    """Return the background model service for a session if it exists."""
    session = sessions.get(session_id)
    if not session:
        return None
    return session.get("background_service")


def get_or_create_session(session_id: str):
    """Get or create per-session state (VLM service). Thread-safe for aiohttp."""
    if session_id not in sessions:
        cfg = default_vlm_config
        sessions[session_id] = {
            "vlm_service": VLMService(
                model=cfg.get("model", "meta/llama-3.2-11b-vision-instruct"),
                api_base=cfg.get("api_base", "http://localhost:8000/v1"),
                api_key=cfg.get("api_key", "EMPTY"),
                prompt=cfg.get("prompt") or None,
                session_id=session_id,
            ),
            "background_service": BackgroundModelService(
                session_id=session_id,
                notify_callback=lambda payload, sid=session_id: notify_session_json(sid, payload),
                summarizer_api_base=cfg.get("api_base", "http://localhost:8000/v1"),
            ),
            "show_request_payload": False,
            "show_response_payload": False,
            "show_memory_state": False,
        }
        logger.info(f"Created new session: {session_id}")
    return sessions[session_id]


def send_to_session(session_id: str, message: str):
    """Send a message only to WebSocket clients in this session."""
    for ws in session_websockets.get(session_id, set()):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")


def get_session_callback(session_id: str):
    """Return a text_callback that sends VLM results only to this session."""
    _last_memory_hash = [None]

    def callback(text: str, metrics: dict):
        session = sessions.get(session_id)
        display_text = text
        if session and session.get("background_service"):
            display_text = session["background_service"].handle_foreground_response(
                text,
                metrics=metrics,
            )

        out = {"type": "vlm_response", "text": display_text, "metrics": metrics}
        if session and session.get("vlm_service"):
            svc = session["vlm_service"]
            if session.get("show_request_payload"):
                payload = svc.get_last_request_payload()
                if payload is not None:
                    out["request_payload"] = payload
            if session.get("show_response_payload"):
                payload = svc.get_last_response_payload()
                if payload is not None:
                    try:
                        out["response_payload"] = json.loads(json.dumps(payload, default=str))
                    except (TypeError, ValueError):
                        out["response_payload"] = payload
            resp = svc.get_last_response_payload()
            if resp and isinstance(resp, dict):
                sh = resp.get("streamingharness", {})
                memory = sh.get("memory") if isinstance(sh, dict) else None
                if memory:
                    mem_hash = json.dumps(memory, ensure_ascii=False, sort_keys=True)
                    if mem_hash != _last_memory_hash[0]:
                        _last_memory_hash[0] = mem_hash
                        out["memory_state"] = memory
                summarizer_timing = sh.get("summarizer_timing") if isinstance(sh, dict) else None
                if summarizer_timing:
                    out["summarizer_timing"] = summarizer_timing
        send_to_session(session_id, json.dumps(out, ensure_ascii=False))

    return callback


async def cleanup_session(session_id: str, reset_adapter: bool = True) -> dict:
    """Cancel active work and remove all server-side state for a session."""
    if not session_id:
        return {"session_id": session_id, "removed": False, "reason": "missing_session_id"}

    logger.info("[%s] Cleaning up session", session_id)

    current_task = asyncio.current_task()
    stream_tasks = [
        task
        for task in http_video_tasks.pop(session_id, set())
        if task is not current_task and not task.done()
    ]
    for task in stream_tasks:
        task.cancel()
    if stream_tasks:
        await asyncio.gather(*stream_tasks, return_exceptions=True)

    session_sockets = list(session_websockets.pop(session_id, set()))
    for ws in session_sockets:
        try:
            await ws.close()
        except Exception as e:
            logger.warning("[%s] Error closing websocket: %s", session_id, e)
        finally:
            websockets.discard(ws)
            ws_to_session.pop(ws, None)

    if session_id in rtsp_tracks:
        await _stop_rtsp_session(session_id)

    for track in list(http_video_tracks.pop(session_id, set())):
        try:
            track.stop()
        except Exception as e:
            logger.warning("[%s] Failed to stop HTTP video track: %s", session_id, e)

    # Stop all source tracks before closing their independent VLM sessions.
    for child_id in list(session_children.pop(session_id, set())):
        await cleanup_session(child_id, reset_adapter=reset_adapter)
    for children in session_children.values():
        children.discard(session_id)

    for file_id, upload in list(uploaded_videos.items()):
        if upload["session_id"] == session_id:
            try:
                upload["path"].unlink(missing_ok=True)
            except OSError as e:
                logger.warning("[%s] Failed to remove uploaded video: %s", session_id, e)
            uploaded_videos.pop(file_id, None)

    pcs_for_session = list(session_peer_connections.pop(session_id, set()))
    for pc in pcs_for_session:
        try:
            await pc.close()
        except Exception as e:
            logger.warning("[%s] Error closing peer connection: %s", session_id, e)
        finally:
            pcs.discard(pc)

    session = sessions.pop(session_id, None)
    cancelled = 0
    cancelled_background = 0
    if session and session.get("vlm_service"):
        svc = session["vlm_service"]
        cancelled = await svc.cancel_active_requests()
        if reset_adapter:
            await svc.reset_adapter_session()
        await svc.close(cancel_requests=False)
    if session and session.get("background_service"):
        bg_svc = session["background_service"]
        cancelled_background = await bg_svc.cancel_active_requests()
        await bg_svc.close(cancel_requests=False)

    logger.info(
        "[%s] Session cleanup complete: removed=%s, websockets=%s, peer_connections=%s, video_streams=%s, cancelled_vlm_tasks=%s, cancelled_background_tasks=%s",
        session_id,
        bool(session),
        len(session_sockets),
        len(pcs_for_session),
        len(stream_tasks),
        cancelled,
        cancelled_background,
    )
    return {
        "session_id": session_id,
        "removed": bool(session),
        "websockets_closed": len(session_sockets),
        "peer_connections_closed": len(pcs_for_session),
        "video_streams_closed": len(stream_tasks),
        "cancelled_vlm_tasks": cancelled,
        "cancelled_background_tasks": cancelled_background,
    }


def is_port_available(port, host="0.0.0.0"):
    """Check if a port is available for binding"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def find_process_using_port(port):
    """Find what process is using a port (Linux/Unix only)"""
    try:
        # Try lsof first (more reliable)
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip().split()[0]
            # Get process name
            name_result = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="], capture_output=True, text=True, timeout=2
            )
            if name_result.returncode == 0:
                return f"PID {pid} ({name_result.stdout.strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lsof not available, try netstat
        try:
            result = subprocess.run(
                ["netstat", "-tulpn"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTEN" in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        return parts[-1]  # PID/Program name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return "unknown process"


def find_available_port(start_port=8080, max_attempts=10):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None


async def detect_local_service_and_model():
    """
    Auto-detect available local VLM services and select a model
    Returns: (api_base, model_name) or (None, None) if no service found
    """
    services = [
        ("http://localhost:11434/v1", "Ollama"),
        ("http://localhost:8000/v1", "vLLM"),
        ("http://localhost:30000/v1", "SGLang"),
    ]

    for api_base, service_name in services:
        try:
            # Try to connect to the service
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{api_base}/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            # Prefer vision models
                            vision_keywords = ["vision", "llava", "llama-3.2", "gemini"]
                            for model in models:
                                model_id = model.get("id", "")
                                if any(keyword in model_id.lower() for keyword in vision_keywords):
                                    logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                                    logger.info(f"   Selected model: {model_id}")
                                    return (api_base, model_id)

                            # If no vision model found, use the first one
                            model_id = models[0].get("id", "")
                            logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                            logger.info(
                                f"   Selected model: {model_id} (vision model preferred but not found)"
                            )
                            return (api_base, model_id)
        except Exception as e:
            logger.debug(f"Service {service_name} not available at {api_base}: {e}")
            continue

    return (None, None)


async def index(request):
    """Serve the main HTML page"""
    content = open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r").read()
    content = content.replace("__WEBRTC_TRANSPORT__", WEBRTC_TRANSPORT)
    return web.Response(content_type="text/html", text=content)


async def models(request):
    """Return available models from the VLM API"""
    try:
        # Check if custom API base and key are provided in query params
        api_base = request.rel_url.query.get("api_base")
        api_key = request.rel_url.query.get("api_key")

        if api_base:
            # Query models from the provided API endpoint
            from openai import AsyncOpenAI

            temp_client = AsyncOpenAI(base_url=api_base, api_key=api_key if api_key else "EMPTY")
            models_response = await temp_client.models.list()
            models_list = [
                {"id": model.id, "name": model.id, "current": False}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
        else:
            # Use default session's VLM service (backwards compat when no api_base in query)
            default_svc = get_or_create_session("default")["vlm_service"]
            models_response = await default_svc.client.models.list()
            models_list = [
                {"id": model.id, "name": model.id, "current": model.id == default_svc.model}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        # Return current model as fallback
        if sessions.get("default"):
            default_svc = sessions["default"]["vlm_service"]
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "models": [
                            {"id": default_svc.model, "name": default_svc.model, "current": True}
                        ]
                    }
                ),
            )
        return web.Response(
            content_type="application/json", text=json.dumps({"models": [], "error": str(e)})
        )


async def detect_services(request):
    """Detect available local VLM services"""
    services = [
        {"name": "Ollama", "url": "http://localhost:11434/v1", "port": 11434, "path": "/api/tags"},
        {"name": "vLLM", "url": "http://localhost:8000/v1", "port": 8000, "path": "/v1/models"},
        {"name": "SGLang", "url": "http://localhost:30000/v1", "port": 30000, "path": "/v1/models"},
    ]

    detected = []

    async def check_service(service):
        """Check if a service is running by probing its endpoint"""
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)  # 1 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"http://localhost:{service['port']}{service['path']}"
                async with session.get(url) as response:
                    if response.status in [200, 404]:  # 404 is ok, means server is running
                        logger.info(f"Detected {service['name']} at {service['url']}")
                        return service
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        return None

    # Check all services concurrently
    results = await asyncio.gather(*[check_service(s) for s in services])
    detected = [s for s in results if s is not None]

    # Default to NVIDIA API Catalog if no local services found
    if not detected:
        detected.append(
            {
                "name": "NVIDIA API Catalog",
                "url": "https://integrate.api.nvidia.com/v1",
                "port": None,
                "path": None,
                "requires_key": True,
            }
        )

    return web.Response(
        content_type="application/json",
        text=json.dumps({"detected": detected, "default": detected[0] if detected else None}),
    )


async def websocket_handler(request):
    """Handle WebSocket connections for text updates. Supports ?session_id= for multi-session."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Session ID from query or generate new (client should send same id in /offer)
    session_id = request.query.get("session_id", "").strip() or str(uuid.uuid4())
    ws_to_session[ws] = session_id
    session_websockets[session_id].add(ws)
    websockets.add(ws)
    logger.info(
        f"WebSocket client connected. session_id={session_id}, total clients: {len(websockets)}"
    )

    session = get_or_create_session(session_id)
    svc = session["vlm_service"]

    try:
        # Send initial message with current server configuration (include session_id if we generated it)
        await ws.send_json(
            {
                "type": "status",
                "text": "Connected to server",
                "status": "Ready",
                "session_id": session_id,
            }
        )

        # Send current server configuration for this session
        from .video_processor import VideoProcessorTrack as _VPT

        background_service = session.get("background_service")
        await ws.send_json(
            {
                "type": "server_config",
                "model": svc.model,
                "api_base": svc.api_base,
                "prompt": svc.prompt,
                "process_interval": _VPT.process_interval_seconds,
                "frames_per_batch": _VPT.frames_per_batch,
                "webrtc_transport": WEBRTC_TRANSPORT,
                "background_model": background_service.get_config()
                if background_service
                else None,
                "session_id": session_id,
            }
        )

        # Keep connection alive and handle incoming messages
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Re-resolve session in case it was recreated
                    svc = get_or_create_session(session_id)["vlm_service"]

                    if data.get("type") == "update_prompt":
                        new_prompt = data.get("prompt", "").strip()
                        if svc:
                            svc.update_prompt(new_prompt)
                            for child_id in session_children.get(session_id, set()):
                                child = sessions.get(child_id)
                                if child and child.get("vlm_service"):
                                    child["vlm_service"].update_prompt(new_prompt)
                            logger.info(f"[{session_id}] Prompt updated: {new_prompt}")

                            await ws.send_json(
                                {
                                    "type": "prompt_updated",
                                    "prompt": new_prompt,
                                }
                            )

                    elif data.get("type") == "update_model":
                        new_model = data.get("model", "").strip()
                        api_base = data.get("api_base", "").strip()
                        api_key = data.get("api_key", "").strip()

                        if new_model and svc:
                            svc.model = new_model
                            if api_base:
                                svc.update_api_settings(api_base, api_key if api_key else None)
                                bg_svc = get_background_service(session_id)
                                if bg_svc:
                                    bg_svc.update_summary_api(api_base=svc.api_base)
                                logger.info(
                                    f"[{session_id}] Model updated: {new_model}, API: {api_base}"
                                )
                            else:
                                logger.info(f"[{session_id}] Model updated: {new_model}")

                            for child_id in session_children.get(session_id, set()):
                                child = sessions.get(child_id)
                                if not child or not child.get("vlm_service"):
                                    continue
                                child_svc = child["vlm_service"]
                                child_svc.model = new_model
                                if api_base:
                                    child_svc.update_api_settings(
                                        api_base, api_key if api_key else None
                                    )

                            await ws.send_json(
                                {
                                    "type": "model_updated",
                                    "model": new_model,
                                    "api_base": svc.api_base,
                                }
                            )

                    elif data.get("type") == "update_processing":
                        interval_sec = data.get("process_interval", 1.0)
                        try:
                            interval_sec = float(interval_sec)
                            if 0.1 <= interval_sec <= 60.0:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.process_interval_seconds
                                VideoProcessorTrack.process_interval_seconds = interval_sec
                                bg_svc = get_background_service(session_id)
                                if bg_svc:
                                    bg_svc.set_foreground_sampling(
                                        process_interval_seconds=interval_sec,
                                        frames_per_batch=VideoProcessorTrack.frames_per_batch,
                                    )
                                logger.info(
                                    f"[{session_id}] Processing interval updated: {old_value} → {interval_sec}s"
                                )

                                await ws.send_json(
                                    {
                                        "type": "processing_updated",
                                        "process_interval": interval_sec,
                                        "background_model": bg_svc.get_config()
                                        if bg_svc
                                        else None,
                                    }
                                )
                            else:
                                logger.warning(
                                    f"Processing interval out of range (0.1-60): {interval_sec}"
                                )
                        except ValueError:
                            logger.error(f"Invalid processing interval: {interval_sec}")

                    elif data.get("type") == "update_frames_per_batch":
                        fpb = data.get("frames_per_batch", 1)
                        try:
                            fpb = int(fpb)
                            if 1 <= fpb <= 30:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.frames_per_batch
                                VideoProcessorTrack.frames_per_batch = fpb
                                bg_svc = get_background_service(session_id)
                                if bg_svc:
                                    bg_svc.set_foreground_sampling(
                                        process_interval_seconds=VideoProcessorTrack.process_interval_seconds,
                                        frames_per_batch=fpb,
                                    )
                                logger.info(
                                    f"[{session_id}] Frames per batch updated: {old_value} → {fpb}"
                                )

                                await ws.send_json(
                                    {
                                        "type": "frames_per_batch_updated",
                                        "frames_per_batch": fpb,
                                        "background_model": bg_svc.get_config()
                                        if bg_svc
                                        else None,
                                    }
                                )
                            else:
                                logger.warning(
                                    f"Frames per batch out of range (1-30): {fpb}"
                                )
                        except ValueError:
                            logger.error(f"Invalid frames per batch: {fpb}")

                    elif data.get("type") == "update_background_config":
                        bg_svc = get_background_service(session_id)
                        if bg_svc:
                            try:
                                config = bg_svc.update_config(
                                    enabled=data.get("enabled")
                                    if "enabled" in data
                                    else None,
                                    frame_multiplier=data.get("frame_multiplier")
                                    if "frame_multiplier" in data
                                    else None,
                                    max_frames=data.get("max_frames")
                                    if "max_frames" in data
                                    else None,
                                    foreground_fps=data.get("foreground_fps")
                                    if "foreground_fps" in data
                                    else None,
                                    resize_long_edge=data.get("resize_long_edge")
                                    if "resize_long_edge" in data
                                    else None,
                                )
                                logger.info(
                                    "[%s] Background model config updated: %s",
                                    session_id,
                                    config,
                                )
                                await ws.send_json(
                                    {
                                        "type": "background_config_updated",
                                        "background_model": config,
                                    }
                                )
                            except (TypeError, ValueError) as err:
                                await ws.send_json(
                                    {
                                        "type": "background_result_error",
                                        "task_id": "",
                                        "error": f"Invalid background config: {err}",
                                    }
                                )

                    elif data.get("type") == "set_debug":
                        session_data = get_or_create_session(session_id)
                        if "show_request_payload" in data:
                            session_data["show_request_payload"] = bool(
                                data["show_request_payload"]
                            )
                        if "show_response_payload" in data:
                            session_data["show_response_payload"] = bool(
                                data["show_response_payload"]
                            )
                        if "show_memory_state" in data:
                            session_data["show_memory_state"] = bool(
                                data["show_memory_state"]
                            )
                        logger.debug(
                            f"[{session_id}] Debug: request_payload="
                            f"{session_data.get('show_request_payload')}, response_payload="
                            f"{session_data.get('show_response_payload')}, memory_state="
                            f"{session_data.get('show_memory_state')}"
                        )

                    elif data.get("type") == "reset_session":
                        logger.info(f"[{session_id}] Client requested adapter session reset")
                        asyncio.create_task(svc.reset_adapter_session())

                    elif data.get("type") == "cleanup_session":
                        logger.info(f"[{session_id}] Client requested session cleanup")
                        asyncio.create_task(cleanup_session(session_id))
                        await ws.close()
                        break

                    elif data.get("type") == "update_max_latency":
                        max_latency = data.get("max_latency", 0.0)
                        try:
                            max_latency = float(max_latency)
                            if 0 <= max_latency <= 10.0:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.max_frame_latency
                                VideoProcessorTrack.max_frame_latency = max_latency
                                status = "disabled" if max_latency == 0 else f"{max_latency:.1f}s"
                                old_status = "disabled" if old_value == 0 else f"{old_value:.1f}s"
                                logger.info(
                                    f"[{session_id}] Max frame latency updated: {old_status} → {status}"
                                )

                                await ws.send_json(
                                    {"type": "max_latency_updated", "max_latency": max_latency}
                                )
                            else:
                                logger.warning(f"Max latency out of range (0-10.0): {max_latency}")
                        except ValueError:
                            logger.error(f"Invalid max latency value: {max_latency}")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    finally:
        session_sockets = session_websockets.get(session_id)
        if session_sockets is not None:
            session_sockets.discard(ws)
            if not session_sockets:
                session_websockets.pop(session_id, None)
        ws_to_session.pop(ws, None)
        websockets.discard(ws)
        logger.info(
            f"WebSocket client disconnected. session_id={session_id}, total clients: {len(websockets)}"
        )

    return ws


async def video_websocket_handler(request):
    """Receive browser camera frames over TCP instead of WebRTC/ICE (UDP)."""
    ws = web.WebSocketResponse(max_msg_size=4 * 1024**2)
    await ws.prepare(request)

    session_id = request.query.get("session_id", "").strip()
    if not session_id:
        await ws.close(code=1008, message=b"Missing session_id")
        return ws

    session = get_or_create_session(session_id)
    source_track = WebSocketVideoTrack()
    processor_track = VideoProcessorTrack(
        source_track,
        session["vlm_service"],
        text_callback=get_session_callback(session_id),
        background_service=session.get("background_service"),
    )

    async def consume_frames():
        while True:
            await processor_track.recv()

    consumer = asyncio.create_task(consume_frames())
    logger.info("[%s] Browser video WebSocket connected (TCP)", session_id)
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                try:
                    await source_track.put_jpeg(msg.data)
                except Exception as error:
                    logger.warning("[%s] Invalid WebSocket video frame: %s", session_id, error)
            elif msg.type == web.WSMsgType.ERROR:
                logger.error("[%s] Video WebSocket error: %s", session_id, ws.exception())
    finally:
        source_track.stop()
        processor_track.stop()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        logger.info("[%s] Browser video WebSocket disconnected", session_id)

    return ws


def broadcast_text_update(text: str, metrics: dict):
    """Broadcast text update and metrics to all connected WebSocket clients"""
    if not websockets:
        return

    message = json.dumps({"type": "vlm_response", "text": text, "metrics": metrics})

    # Send to all connected clients
    dead_websockets = set()
    for ws in websockets:
        try:
            # Use asyncio to send without blocking
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to websocket: {e}")
            dead_websockets.add(ws)

    # Clean up dead connections
    websockets.difference_update(dead_websockets)


async def offer(request):
    """Handle WebRTC offer from webcam or RTSP."""
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    rtsp_url = params.get("rtsp_url")  # Optional RTSP URL for IP camera mode
    session_id = params.get("session_id", "default")

    session = get_or_create_session(session_id)
    session_vlm = session["vlm_service"]
    background_service = session.get("background_service")
    session_callback = get_session_callback(session_id)

    # Create RTCPeerConnection with STUN servers for Docker/NAT compatibility
    config = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        ]
    )
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)
    session_peer_connections[session_id].add(pc)

    # Store RTSP track for cleanup
    rtsp_cleanup_track = None

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            # Clean up RTSP track if exists
            if rtsp_cleanup_track:
                rtsp_cleanup_track.stop()
                logger.info("RTSP track stopped on connection close")
            await pc.close()
            pcs.discard(pc)
            session_pcs = session_peer_connections.get(session_id)
            if session_pcs is not None:
                session_pcs.discard(pc)
                if not session_pcs:
                    session_peer_connections.pop(session_id, None)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            logger.error("ICE connection failed - check firewall/NAT settings")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state: {pc.iceGatheringState}")

    # If RTSP URL provided, create RTSP track instead of waiting for browser track
    if rtsp_url:
        logger.info(f"[{session_id}] Creating RTSP track for: {rtsp_url}")
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
            rtsp_cleanup_track = rtsp_track  # Store for cleanup

            # Wait for initial connection to get stream info
            await asyncio.sleep(0.5)

            # Wrap RTSP track with relay first (same pattern as webcam)
            relayed_rtsp = relay.subscribe(rtsp_track)

            processor_track = VideoProcessorTrack(
                relayed_rtsp,
                session_vlm,
                text_callback=session_callback,
                background_service=background_service,
            )

            # Add processor directly to peer connection
            pc.addTrack(processor_track)
            logger.info("Added RTSP processor track to peer connection")

        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )
    else:
        # Webcam mode: wait for browser to send track
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")

            if track.kind == "video":
                # Create processor track with this session's VLM and session-scoped callback
                processor_track = VideoProcessorTrack(
                    relay.subscribe(track),
                    session_vlm,
                    text_callback=session_callback,
                    background_service=background_service,
                )

                # Add processed track back to connection
                pc.addTrack(processor_track)
                logger.info("Added processed video track back to peer connection")

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.kind} ended")

    # Handle offer
    await pc.setRemoteDescription(offer_sdp)

    # Create answer - this must happen after tracks are added
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"Created answer with {len(pc.getTransceivers())} transceivers")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def session_cleanup(request):
    """Cancel active VLM work and remove a session."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": "Missing session_id parameter"}),
        )

    reset_adapter = bool(data.get("reset_adapter", True))
    result = await cleanup_session(session_id, reset_adapter=reset_adapter)
    return web.Response(content_type="application/json", text=json.dumps(result))


async def upload_video(request):
    """Stream an MP4 upload to server disk and return an opaque file id."""
    session_id = (request.query.get("session_id") or "").strip()
    if not session_id:
        return web.json_response({"error": "Missing session_id parameter"}, status=400)

    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "video" or not field.filename:
        return web.json_response({"error": "Missing video file"}, status=400)
    if not field.filename.lower().endswith(".mp4"):
        return web.json_response({"error": "Only .mp4 files are supported"}, status=400)

    video_upload_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    path = video_upload_dir / f"{file_id}.mp4"
    size = 0
    try:
        with path.open("wb") as output:
            while chunk := await field.read_chunk(size=1024 * 1024):
                output.write(chunk)
                size += len(chunk)
        if size == 0:
            raise ValueError("The uploaded file is empty")
    except Exception as e:
        path.unlink(missing_ok=True)
        return web.json_response({"error": f"Failed to save upload: {e}"}, status=400)

    uploaded_videos[file_id] = {"path": path, "session_id": session_id}
    logger.info("[%s] Saved MP4 upload %s (%s bytes)", session_id, file_id, size)
    return web.json_response({"file_id": file_id, "size": size})


def resolve_video_library_root(requested_directory: str = "") -> Path | None:
    """Resolve a user-selected server directory, falling back to the configured library."""
    try:
        root = (
            Path(requested_directory).expanduser().resolve(strict=True)
            if requested_directory
            else video_library_dir.resolve(strict=True)
        )
    except (OSError, RuntimeError):
        return None
    return root if root.is_dir() else None


def resolve_library_video(relative_path: str, library_root: Path) -> Path | None:
    """Resolve an MP4 path strictly inside the selected server video library."""
    if not relative_path:
        return None
    try:
        candidate = (library_root / relative_path).resolve(strict=True)
        candidate.relative_to(library_root)
    except (OSError, RuntimeError, ValueError):
        return None
    if not candidate.is_file() or candidate.suffix.lower() != ".mp4":
        return None
    return candidate


async def list_library_videos(request):
    """List reusable MP4 files from a selected server-side video directory."""
    requested_directory = (request.query.get("directory") or "").strip()
    library_root = resolve_video_library_root(requested_directory)
    if library_root is None:
        return web.json_response(
            {"error": "The server video directory does not exist or is not a directory"},
            status=400,
        )
    videos = []
    try:
        paths = sorted(
            (path for path in library_root.rglob("*") if path.suffix.lower() == ".mp4"),
            key=lambda path: str(path).lower(),
        )
        for path in paths[:2000]:
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            try:
                relative_path = resolved.relative_to(library_root).as_posix()
            except ValueError:
                continue
            stat = resolved.stat()
            videos.append(
                {
                    "path": relative_path,
                    "name": resolved.name,
                    "size": stat.st_size,
                }
            )
    except OSError as e:
        return web.json_response({"error": f"Failed to scan video directory: {e}"}, status=500)
    return web.json_response({"directory": str(library_root), "videos": videos})


async def stream_uploaded_video(request):
    """Process an uploaded MP4 and return frames as an HTTP multipart MJPEG stream."""
    session_id = (request.query.get("session_id") or "").strip()
    file_id = (request.query.get("file_id") or "").strip()
    server_path = (request.query.get("server_path") or "").strip()
    requested_directory = (request.query.get("library_dir") or "").strip()
    loop_video = request.query.get("loop", "false").lower() in {"1", "true", "yes"}
    video_id = (request.query.get("video_id") or "1").strip()
    if not video_id.isdecimal() or len(video_id) > 9 or int(video_id) < 1:
        return web.json_response({"error": "Invalid video_id"}, status=400)
    video_id = str(int(video_id))
    upload = uploaded_videos.get(file_id) if file_id else None
    if upload and upload["session_id"] != session_id:
        upload = None
    library_root = resolve_video_library_root(requested_directory) if not upload else None
    if upload:
        source_path = upload["path"]
    elif library_root:
        source_path = resolve_library_video(server_path, library_root)
    else:
        source_path = None
    if not session_id or source_path is None:
        return web.json_response({"error": "Video file was not found"}, status=404)

    parent_session = get_or_create_session(session_id)
    child_session_id = f"{session_id}:video-{video_id}"
    session_children[session_id].add(child_session_id)
    session = get_or_create_session(child_session_id)
    parent_vlm = parent_session["vlm_service"]
    child_vlm = session["vlm_service"]
    child_vlm.model = parent_vlm.model
    child_vlm.update_api_settings(parent_vlm.api_base, parent_vlm.api_key)
    child_vlm.update_prompt(parent_vlm.prompt or "")

    def video_callback(text: str, metrics: dict):
        send_to_session(
            session_id,
            json.dumps(
                {
                    "type": "vlm_response",
                    "video_id": video_id,
                    "text": text,
                    "metrics": metrics,
                },
                ensure_ascii=False,
            ),
        )
    try:
        pending_player = MediaPlayer(str(source_path), loop=False)
        if pending_player.video is None:
            raise ValueError("The uploaded MP4 does not contain a video track")
    except Exception as e:
        logger.exception("[%s] Failed to open uploaded MP4", session_id)
        return web.json_response({"error": f"Failed to open uploaded MP4: {e}"}, status=400)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )
    await response.prepare(request)
    request_task = asyncio.current_task()
    if request_task is not None:
        http_video_tasks[session_id].add(request_task)
    logger.info("[%s] Started HTTP MJPEG stream video=%s (loop=%s)", session_id, video_id, loop_video)

    try:
        while True:
            source_track = None
            processor_track = None
            frames_in_pass = 0
            try:
                # Reopening the player for every pass also resets its playback clock.
                # MediaPlayer(loop=True) reuses that clock after seeking to zero and can
                # race through subsequent passes until the preview appears frozen.
                player = pending_player or MediaPlayer(str(source_path), loop=False)
                pending_player = None
                source_track = player.video
                if source_track is None:
                    raise ValueError("The uploaded MP4 does not contain a video track")
                processor_track = VideoProcessorTrack(
                    source_track,
                    session["vlm_service"],
                    text_callback=video_callback,
                    background_service=session.get("background_service"),
                )
                http_video_tracks[session_id].add(source_track)

                while True:
                    frame = await processor_track.recv()
                    frames_in_pass += 1
                    image_buffer = io.BytesIO()
                    frame.to_image().save(image_buffer, format="JPEG", quality=85)
                    jpeg = image_buffer.getvalue()
                    await response.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode("ascii")
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
            except (MediaStreamError, StopAsyncIteration):
                if not loop_video or frames_in_pass == 0:
                    break
                logger.debug("[%s] Reopening video=%s for loop playback", session_id, video_id)
            finally:
                if processor_track is not None:
                    processor_track.stop()
                if source_track is not None:
                    source_track.stop()
                    tracks = http_video_tracks.get(session_id)
                    if tracks is not None:
                        tracks.discard(source_track)
                        if not tracks:
                            http_video_tracks.pop(session_id, None)
    except (ValueError, OSError) as e:
        logger.warning("[%s] Uploaded MP4 stream failed: %s", session_id, e)
    except (ConnectionResetError, asyncio.CancelledError):
        logger.info("[%s] HTTP video stream ended", session_id)
    finally:
        if request_task is not None:
            tasks = http_video_tasks.get(session_id)
            if tasks is not None:
                tasks.discard(request_task)
                if not tasks:
                    http_video_tasks.pop(session_id, None)

    return response


async def rtsp_start(request):
    """
    Start RTSP stream processing.

    Accepts RTSP URL and creates a video processing pipeline.

    POST /api/rtsp/start
    Body: {"rtsp_url": "rtsp://...", "session_id": "optional-id"}
    """
    try:
        data = await request.json()
        rtsp_url = data.get("rtsp_url")
        parent_session_id = str(data.get("parent_session_id") or "").strip()
        video_id = str(data.get("video_id") or "").strip()
        if parent_session_id:
            if not video_id.isdecimal() or len(video_id) > 9 or int(video_id) < 1:
                return web.json_response({"error": "Invalid video_id"}, status=400)
            video_id = str(int(video_id))
            session_id = f"{parent_session_id}:video-{video_id}"
        else:
            session_id = data.get("session_id", "default")
        stream_to_browser = bool(data.get("stream_to_browser", False))

        if not rtsp_url:
            logger.warning("RTSP start request missing rtsp_url")
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Missing rtsp_url parameter"}),
            )

        # Check if session already exists
        if session_id in rtsp_tracks:
            logger.warning(f"RTSP session {session_id} already exists, stopping it first")
            await _stop_rtsp_session(session_id)

        logger.info(f"Starting RTSP stream for session {session_id}")

        # Create RTSP video track
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )

        # Create processor track with this session's VLM and session-scoped callback
        session = get_or_create_session(session_id)
        if parent_session_id:
            parent_session = get_or_create_session(parent_session_id)
            session_children[parent_session_id].add(session_id)
            parent_vlm = parent_session["vlm_service"]
            child_vlm = session["vlm_service"]
            child_vlm.model = parent_vlm.model
            child_vlm.update_api_settings(parent_vlm.api_base, parent_vlm.api_key)
            child_vlm.update_prompt(parent_vlm.prompt or "")
        session_vlm = session["vlm_service"]
        background_service = session.get("background_service")
        if parent_session_id:

            def session_callback(text: str, metrics: dict):
                send_to_session(
                    parent_session_id,
                    json.dumps(
                        {
                            "type": "vlm_response",
                            "video_id": video_id,
                            "text": text,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                    ),
                )
        else:
            session_callback = get_session_callback(session_id)
        processor_track = VideoProcessorTrack(
            rtsp_track,
            session_vlm,
            text_callback=session_callback,
            background_service=background_service,
        )

        # Start background task to consume frames
        async def consume_frames():
            """Background task to continuously pull frames from processor track"""
            try:
                while not rtsp_track._stopped:
                    try:
                        _ = await processor_track.recv()
                        # Frame is processed, just discard it (VLM analysis happens in recv())
                    except StopAsyncIteration:
                        logger.info(f"RTSP stream {session_id} ended")
                        break
                    except Exception as e:
                        logger.error(f"Error consuming RTSP frame for {session_id}: {e}")
                        break
            finally:
                logger.info(f"Frame consumption stopped for {session_id}")

        # The browser preview endpoint consumes the processor directly so it can
        # return each processed frame as MJPEG over HTTP/TCP.
        frame_task = None if stream_to_browser else asyncio.create_task(consume_frames())

        # Store reference with frame task
        rtsp_tracks[session_id] = (rtsp_track, processor_track, frame_task)

        # Get stream stats
        stats = rtsp_track.get_stats()

        logger.info(
            f"RTSP stream started: {session_id} - {stats.get('codec')} "
            f"{stats.get('width')}x{stats.get('height')}"
        )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "started", "session_id": session_id, "stream_info": stats}),
        )

    except Exception as e:
        logger.error(f"Error starting RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def stream_rtsp_video(request):
    """Return an already prepared RTSP session as an HTTP/TCP MJPEG stream."""
    session_id = (request.query.get("session_id") or "").strip()
    entry = rtsp_tracks.get(session_id)
    if not session_id or entry is None:
        return web.json_response({"error": "RTSP session was not found"}, status=404)

    rtsp_track, processor_track, frame_task = entry
    if frame_task is not None:
        return web.json_response(
            {"error": "RTSP session is not a browser stream"}, status=409
        )

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )
    await response.prepare(request)
    logger.info("[%s] Started RTSP preview over HTTP/TCP", session_id)
    last_preview_at = 0.0
    try:
        while True:
            frame = await processor_track.recv()
            now = time.monotonic()
            if now - last_preview_at < 0.1:
                continue
            last_preview_at = now
            image_buffer = io.BytesIO()
            frame.to_image().save(image_buffer, format="JPEG", quality=85)
            jpeg = image_buffer.getvalue()
            await response.write(
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
    except (MediaStreamError, StopAsyncIteration, ConnectionResetError, asyncio.CancelledError):
        logger.info("[%s] RTSP HTTP preview ended", session_id)
    finally:
        if session_id in rtsp_tracks:
            await _stop_rtsp_session(session_id)

    return response


async def rtsp_stop(request):
    """
    Stop RTSP stream processing.

    POST /api/rtsp/stop
    Body: {"session_id": "optional-id"}
    """
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")

        await _stop_rtsp_session(session_id)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "stopped", "session_id": session_id}),
        )

    except Exception as e:
        logger.error(f"Error stopping RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_status(request):
    """
    Get status of all RTSP streams.

    GET /api/rtsp/status
    """
    try:
        status_list = []

        for session_id, (rtsp_track, processor_track, frame_task) in rtsp_tracks.items():
            stats = rtsp_track.get_stats()
            status_list.append(
                {
                    "session_id": session_id,
                    "connected": stats.get("connected"),
                    "frames_received": stats.get("frames_received"),
                    "stream_info": {
                        "codec": stats.get("codec"),
                        "width": stats.get("width"),
                        "height": stats.get("height"),
                        "fps": stats.get("fps"),
                    },
                }
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"active_streams": len(rtsp_tracks), "streams": status_list}),
        )

    except Exception as e:
        logger.error(f"Error getting RTSP status: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def _stop_rtsp_session(session_id: str):
    """Helper function to stop an RTSP session"""
    if session_id in rtsp_tracks:
        rtsp_track, processor_track, frame_task = rtsp_tracks[session_id]

        # Signal stop first so _read_frame exits early on its next iteration
        rtsp_track._stopped = True

        # Cancel frame consumption task
        if frame_task and not frame_task.done():
            frame_task.cancel()
            try:
                await frame_task
            except asyncio.CancelledError:
                pass

        # Stop tracks (rtsp_track.stop acquires _read_lock to wait for
        # any in-flight executor thread before closing the container)
        try:
            processor_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping processor track: {e}")

        try:
            rtsp_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping RTSP track: {e}")

        # Remove from tracking
        del rtsp_tracks[session_id]
        logger.info(f"RTSP stream stopped: {session_id}")
    else:
        logger.warning(f"RTSP session {session_id} not found")


async def on_startup(app):
    """Initialize resources on server startup"""
    logger.info("Server startup complete")


async def on_shutdown(app):
    """Cleanup on server shutdown"""

    logger.info("Shutting down server...")

    # Close all websockets and clear session state
    for ws in list(websockets):
        await ws.close()
    websockets.clear()
    session_websockets.clear()
    ws_to_session.clear()
    session_peer_connections.clear()

    # Close all RTSP streams
    for session_id in list(rtsp_tracks.keys()):
        await _stop_rtsp_session(session_id)
    logger.info("RTSP streams closed")

    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    for session_id, session in list(sessions.items()):
        svc = session.get("vlm_service")
        if svc:
            await svc.close()
        bg_svc = session.get("background_service")
        if bg_svc:
            await bg_svc.close()
        sessions.pop(session_id, None)
    logger.info("VLM sessions closed")

    logger.info("Cleanup complete")


async def create_app(test_mode=False):
    """
    Create and configure the aiohttp web application.

    Args:
        test_mode: If True, use test configuration

    Returns:
        Configured web.Application instance
    """
    # Create web application
    app = web.Application(client_max_size=2 * 1024**3)
    app.router.add_get("/", index)
    app.router.add_get("/models", models)
    app.router.add_get("/detect-services", detect_services)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/ws/video", video_websocket_handler)
    setup_asr_routes(app)
    setup_tts_routes(app)
    setup_local_file_routes(app)
    app.router.add_post("/offer", offer)
    app.router.add_post("/api/session/cleanup", session_cleanup)
    app.router.add_post("/api/video/upload", upload_video)
    app.router.add_get("/api/video/library", list_library_videos)
    app.router.add_get("/api/video/stream", stream_uploaded_video)

    # RTSP endpoints
    app.router.add_post("/api/rtsp/start", rtsp_start)
    app.router.add_post("/api/rtsp/stop", rtsp_stop)
    app.router.add_get("/api/rtsp/status", rtsp_status)
    app.router.add_get("/api/rtsp/stream", stream_rtsp_video)

    # Serve static files (images, etc.)
    # Always serve from static/images within the package (works for both pip and dev installs)
    images_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    images_dir = os.path.abspath(images_dir)

    if os.path.exists(images_dir):
        app.router.add_static("/images", images_dir, name="images")
        logger.info(f"Serving static files from: {images_dir}")
    else:
        logger.warning(f"⚠️  Static images directory not found: {images_dir}")

    # Serve favicon files
    favicon_dir = os.path.join(os.path.dirname(__file__), "static", "favicon")
    favicon_dir = os.path.abspath(favicon_dir)

    if os.path.exists(favicon_dir):
        app.router.add_static("/favicon", favicon_dir, name="favicon")
        logger.info(f"Serving favicon files from: {favicon_dir}")
    else:
        logger.warning(f"⚠️  Favicon directory not found: {favicon_dir}")

    if not test_mode:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)

    return app


def get_app_config_dir():
    """Get the application config directory following OS conventions"""
    import os
    from pathlib import Path

    # Follow XDG Base Directory spec on Linux, use OS-appropriate paths elsewhere
    if os.name == "posix":
        if "darwin" in os.sys.platform.lower():
            # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "joy-vl-interaction"
        else:
            # Linux/Unix (including Jetson)
            config_dir = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "joy-vl-interaction"
            )
    else:
        # Windows
        config_dir = Path(os.environ.get("APPDATA", Path.home())) / "joy-vl-interaction"

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def generate_self_signed_cert(cert_path="cert.pem", key_path="key.pem"):
    """Generate a self-signed SSL certificate if it doesn't exist"""
    import subprocess
    import os

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return True

    logger.info("🔐 Generating self-signed SSL certificate...")
    logger.info(f"   Saving to: {os.path.dirname(os.path.abspath(cert_path)) or '.'}")
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-out",
                cert_path,
                "-keyout",
                key_path,
                "-days",
                "365",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"✅ Generated {cert_path} and {key_path}")
        return True
    except FileNotFoundError:
        logger.warning("⚠️  openssl not found - cannot auto-generate certificates")
        logger.warning(
            "⚠️  Install openssl: sudo apt install openssl (Linux) or brew install openssl (Mac)"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Failed to generate certificates: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    import ssl
    from . import __version__

    parser = argparse.ArgumentParser(
        description="WebRTC Joy VL Interaction - Real-time vision model interaction",
        epilog="Examples:\n"
        "  vLLM:    python -m joy_interaction_webui.server --model llama-3.2-11b-vision-instruct --api-base http://localhost:8000/v1\n"
        "  SGLang:  python -m joy_interaction_webui.server --model llama-3.2-11b-vision-instruct --api-base http://localhost:30000/v1\n"
        "  Ollama:  python -m joy_interaction_webui.server --model llava:7b --api-base http://localhost:11434/v1\n"
        "  HTTPS:   python -m joy_interaction_webui.server --model llava:7b --api-base http://localhost:11434/v1 --ssl-cert cert.pem --ssl-key key.pem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind to (default: 8090)")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Automatically find available port if default is taken",
    )
    parser.add_argument(
        "--model", help="VLM model name (optional, will auto-detect if not specified)"
    )
    parser.add_argument(
        "--api-base", help="VLM API base URL (optional, will auto-detect or use NVIDIA NGC)"
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key - use 'EMPTY' for local servers, required for NVIDIA NGC/OpenAI (default: EMPTY)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Initial prompt to send to VLM (default: empty, waits for user input)",
    )
    parser.add_argument(
        "--video-dir",
        default=os.environ.get("LIVE_VLM_VIDEO_DIR", str(Path.cwd() / "videos")),
        help="Server directory containing reusable MP4 files (env: LIVE_VLM_VIDEO_DIR)",
    )
    # Get default SSL cert paths (platform-specific)
    default_config_dir = get_app_config_dir()
    default_cert_path = str(default_config_dir / "cert.pem")
    default_key_path = str(default_config_dir / "key.pem")

    parser.add_argument("--process-interval", type=float, default=1.0, help="Processing interval in seconds (default: 1.0)")
    parser.add_argument("--frames-per-batch", type=int, default=1, help="Number of frames to batch per VLM inference (default: 1). E.g., 2 means capture 2 frames within each process-interval and send them together.")
    parser.add_argument(
        "--ssl-cert",
        default=None,  # Will be set to config dir if not specified
        help=f"Path to SSL certificate file (default: {default_cert_path}, auto-generated if missing)",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,  # Will be set to config dir if not specified
        help=f"Path to SSL private key file (default: {default_key_path}, auto-generated if missing)",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Disable SSL (not recommended - webcam requires HTTPS)",
    )

    args = parser.parse_args()

    # Cloud deployment: env overrides for default API base, model, and frame interval
    if os.environ.get("LIVE_VLM_API_BASE"):
        if not args.api_base:
            args.api_base = os.environ.get("LIVE_VLM_API_BASE").strip()
            logger.info(f"Using API base from env: {args.api_base}")
    if os.environ.get("LIVE_VLM_DEFAULT_MODEL"):
        if not args.model:
            args.model = os.environ.get("LIVE_VLM_DEFAULT_MODEL").strip()
            logger.info(f"Using default model from env: {args.model}")
    if os.environ.get("LIVE_VLM_PROCESS_INTERVAL"):
        try:
            args.process_interval = float(os.environ.get("LIVE_VLM_PROCESS_INTERVAL"))
            logger.info(f"Using process_interval from env: {args.process_interval}s")
        except ValueError:
            pass
    if os.environ.get("LIVE_VLM_FRAMES_PER_BATCH"):
        try:
            args.frames_per_batch = int(os.environ.get("LIVE_VLM_FRAMES_PER_BATCH"))
            logger.info(f"Using frames_per_batch from env: {args.frames_per_batch}")
        except ValueError:
            pass

    # Set default SSL cert paths to config directory if not specified
    if args.ssl_cert is None:
        config_dir = get_app_config_dir()
        args.ssl_cert = str(config_dir / "cert.pem")
    if args.ssl_key is None:
        config_dir = get_app_config_dir()
        args.ssl_key = str(config_dir / "key.pem")

    # Auto-detect service and model if not specified
    api_base = args.api_base
    model = args.model
    api_key = args.api_key

    if not model or not api_base:
        logger.info("No model/API specified, auto-detecting local services...")
        detected_api_base, detected_model = asyncio.run(detect_local_service_and_model())

        if detected_api_base and detected_model:
            if not api_base:
                api_base = detected_api_base
            if not model:
                model = detected_model
        else:
            # Fall back to NVIDIA NGC
            logger.warning("⚠️  No local VLM service found (Ollama, vLLM, SGLang)")
            logger.info("📡 Falling back to NVIDIA API Catalog")
            logger.info("   You'll need an API key from: https://build.nvidia.com")
            if not api_base:
                api_base = "https://integrate.api.nvidia.com/v1"
            if not model:
                model = (
                    os.environ.get("LIVE_VLM_DEFAULT_MODEL") or "meta/llama-3.2-11b-vision-instruct"
                ).strip()
                if os.environ.get("LIVE_VLM_DEFAULT_MODEL"):
                    logger.info(f"Using default model from env: {model}")
            if api_key == "EMPTY":
                logger.warning("⚠️  API key required for NVIDIA API Catalog")
                logger.warning("   Set with: --api-key YOUR_API_KEY")
                logger.warning("   Or use WebUI to configure API settings after starting")

    # Initialize VLM service and default session for multi-session support
    global vlm_service, default_vlm_config, video_library_dir
    video_library_dir = Path(args.video_dir).expanduser().resolve()
    logger.info("Server video library: %s", video_library_dir)
    vlm_service = VLMService(model=model, api_base=api_base, api_key=api_key, prompt=args.prompt)
    default_vlm_config = {
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "prompt": args.prompt,
    }
    sessions["default"] = {
        "vlm_service": vlm_service,
        "background_service": BackgroundModelService(
            session_id="default",
            notify_callback=lambda payload: notify_session_json("default", payload),
            summarizer_api_base=api_base,
        ),
        "show_request_payload": False,
        "show_response_payload": False,
        "show_memory_state": False,
    }

    # Log initialization with better formatting
    service_name = "Local" if "localhost" in api_base or "127.0.0.1" in api_base else "Cloud"
    logger.info("Initialized VLM service:")
    logger.info(f"  Model: {model}")
    logger.info(f"  API: {api_base} ({service_name})")
    logger.info(f"  Prompt: {args.prompt}")

    # Update frame processing rate in VideoProcessorTrack if needed
    # (This is a bit hacky but works for this demo)
    VideoProcessorTrack.process_interval_seconds = args.process_interval
    VideoProcessorTrack.frames_per_batch = args.frames_per_batch

    # Create web application using create_app
    app = asyncio.run(create_app(test_mode=False))

    # Setup SSL (auto-generate certificates if needed)
    ssl_context = None
    protocol = "http"
    if not args.no_ssl:
        # Try to auto-generate if certificates don't exist
        if not os.path.exists(args.ssl_cert) or not os.path.exists(args.ssl_key):
            success = generate_self_signed_cert(args.ssl_cert, args.ssl_key)
            if not success:
                # FAIL FAST - SSL is required for webcam access
                logger.error("")
                logger.error("❌ Cannot start server without SSL certificates")
                logger.error("❌ Webcam access requires HTTPS!")
                logger.error("")
                logger.error("🔧 To fix, install openssl:")
                logger.error("   Linux/Jetson: sudo apt install openssl")
                logger.error("   macOS: brew install openssl")
                logger.error("")
                logger.error("   Then restart the server")
                logger.error("")
                logger.error(
                    "⚠️  Or run with --no-ssl if you don't need camera access (not recommended)"
                )
                logger.error("")
                sys.exit(1)

        # Load certificates (they must exist at this point)
        if os.path.exists(args.ssl_cert) and os.path.exists(args.ssl_key):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
            protocol = "https"
            logger.info("SSL enabled - using HTTPS")
        else:
            # This should never happen, but just in case
            logger.error("❌ SSL certificates missing after generation - unexpected error")
            sys.exit(1)
    else:
        logger.warning("⚠️  SSL disabled with --no-ssl flag")
        logger.warning("⚠️  Webcam access will NOT work without HTTPS!")

    # Get network addresses
    import socket
    import subprocess

    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Access the server at:")
    logger.info(f"  Local:   {protocol}://localhost:{args.port}")

    # Get network interfaces - try multiple methods for cross-platform support
    network_ips = []

    # Method 1: hostname -I (Linux)
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                # Filter out loopback and docker bridges (172.17.x.x)
                if not ip.startswith("127.") and not ip.startswith("172.17."):
                    network_ips.append(ip)
    except Exception:
        pass

    # Method 2: Socket method (cross-platform fallback)
    if not network_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip != "127.0.0.1":
                network_ips.append(ip)
        except Exception:
            pass

    # Display all found network IPs
    for ip in network_ips:
        logger.info(f"  Network: {protocol}://{ip}:{args.port}")

    logger.info("=" * 70)
    logger.info("")
    logger.info("Press Ctrl+C to stop")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nReceived signal to terminate. Shutting down gracefully...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


def stop():
    """Stop the running joy-vl-interaction server"""
    import sys
    import time

    try:
        import psutil
    except ImportError:
        logger.error("psutil is required for the stop command")
        logger.error("Install it with: pip install joy-vl-interaction[dev]")
        sys.exit(1)

    print("Stopping Joy VL Interaction server...")

    # Find and kill processes running joy_vl_interaction.server
    found = False
    killed = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "joy_vl_interaction.server" in cmdline_str or "joy-vl-interaction" in cmdline_str:
                    # Don't kill the stop command itself
                    if "stop" not in cmdline_str:
                        found = True
                        print(f"  Stopping process {proc.info['pid']}: {proc.info['name']}")
                        proc.terminate()
                        killed.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not found:
        print("✓ No running server found")
        return

    # Wait for graceful shutdown
    time.sleep(2)

    # Force kill if still running
    for proc in killed:
        try:
            if proc.is_running():
                print(f"  Force killing process {proc.pid}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Final verification
    time.sleep(1)
    still_running = False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "joy_vl_interaction.server" in cmdline_str or "joy-vl-interaction" in cmdline_str:
                    if "stop" not in cmdline_str:
                        still_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if still_running:
        print("❌ Failed to stop server")
        sys.exit(1)
    else:
        print("✓ Server stopped successfully")


if __name__ == "__main__":
    main()
