# SPDX-License-Identifier: Apache-2.0

"""Joy VL Interaction TTS websocket adapter for the local speech model."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_UPSTREAM_URL = "ws://127.0.0.1:8991/v1/audio/speech/stream"
DEFAULT_VOICE = "vivian"
DEFAULT_SAMPLE_RATE = 24000
WEBUI_OUTPUT_FORMAT = "pcm16"
VLLM_RESPONSE_FORMAT = "pcm"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 60.0

logger = logging.getLogger("joyvl_tts_adapter")


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


class UpstreamWebSocket(Protocol):
    async def send(self, message: str | bytes) -> None:
        ...

    async def recv(self) -> str | bytes:
        ...

    async def close(self) -> None:
        ...


ConnectUpstream = Callable[[str], Awaitable[UpstreamWebSocket]]


@dataclass(frozen=True)
class Settings:
    upstream_url: str = DEFAULT_UPSTREAM_URL
    model: str = MODEL_NAME
    default_voice: str = DEFAULT_VOICE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    output_format: str = WEBUI_OUTPUT_FORMAT
    request_timeout: float = DEFAULT_TIMEOUT_SECONDS
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT_SECONDS


async def connect_vllm_websocket(url: str) -> UpstreamWebSocket:
    return await websockets.connect(
        url,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    )


def load_settings_from_env() -> Settings:
    return Settings(
        upstream_url=env_value("TTS_UPSTREAM_URL", default=DEFAULT_UPSTREAM_URL),
        model=env_value("TTS_MODEL", default=MODEL_NAME),
        default_voice=env_value("TTS_DEFAULT_VOICE", default=DEFAULT_VOICE),
        sample_rate=int(
            env_value("TTS_SAMPLE_RATE", default=str(DEFAULT_SAMPLE_RATE))
        ),
        output_format=env_value("TTS_OUTPUT_FORMAT", default=WEBUI_OUTPUT_FORMAT),
        request_timeout=float(
            env_value("TTS_REQUEST_TIMEOUT", default=str(DEFAULT_TIMEOUT_SECONDS))
        ),
        idle_timeout=float(
            env_value("TTS_IDLE_TIMEOUT", default=str(DEFAULT_IDLE_TIMEOUT_SECONDS))
        ),
    )


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def resolve_voice(voice: str | None, default_voice: str) -> str:
    candidate = (voice or "").strip()
    if not candidate or candidate == "default":
        return default_voice
    return candidate


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def build_session_config(
    config: dict[str, Any],
    *,
    settings: Settings,
    emotion: str | None = None,
) -> dict[str, Any]:
    output_format = str(config.get("output_audio_format") or settings.output_format)
    sample_rate = int(config.get("sample_rate") or settings.sample_rate)
    if output_format != WEBUI_OUTPUT_FORMAT:
        raise ValueError("Only pcm16 output_audio_format is supported")
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError("Only 24000 sample_rate is supported by the local TTS model")

    instructions = str(config.get("instructions") or "").strip()
    emotion_text = str(emotion or "").strip()
    if emotion_text:
        instructions = f"{emotion_text} {instructions}".strip()

    return {
        "type": "session.config",
        "model": settings.model,
        "voice": resolve_voice(config.get("voice"), settings.default_voice),
        "response_format": VLLM_RESPONSE_FORMAT,
        "instructions": instructions,
        "temperature": float(config.get("temperature", 0.7)),
        "max_new_tokens": int(config.get("max_tokens", 1024)),
        "stream_audio": True,
    }


async def send_error(client_ws: WebSocket, message: str) -> None:
    await client_ws.send_text(json_dumps({"type": "error", "error": message}))


async def recv_json_from_client(client_ws: WebSocket) -> dict[str, Any] | None:
    message = await client_ws.receive()
    if message.get("type") == "websocket.disconnect":
        return None
    text = message.get("text")
    if text is None:
        raise ValueError("Expected a text websocket message")
    return json.loads(text)


async def forward_upstream_audio(
    *,
    client_ws: WebSocket,
    upstream_ws: UpstreamWebSocket,
    idle_timeout: float,
) -> None:
    while True:
        upstream_message = await asyncio.wait_for(upstream_ws.recv(), timeout=idle_timeout)
        if isinstance(upstream_message, bytes):
            if upstream_message:
                await client_ws.send_bytes(upstream_message)
            continue

        try:
            event = json.loads(upstream_message)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON upstream event")
            continue

        event_type = event.get("type")
        if event_type == "audio.done":
            if event.get("error"):
                await send_error(client_ws, f"vLLM audio generation failed: {event}")
                return
            continue
        if event_type == "session.done":
            await client_ws.send_text(json_dumps({"type": "response.done"}))
            return
        if event_type == "error":
            detail = event.get("error") or event.get("message") or event
            await send_error(client_ws, f"vLLM upstream error: {detail}")
            return

        logger.debug("Upstream event: %s", event_type)


async def run_tts_session(
    *,
    client_ws: WebSocket,
    upstream_url: str,
    settings: Settings,
    connect_upstream: ConnectUpstream,
) -> None:
    first_message = await recv_json_from_client(client_ws)
    if first_message is None:
        return
    if "config" not in first_message:
        await send_error(client_ws, "Expected initial config message")
        return

    config = dict(first_message.get("config") or {})
    buffered_text: list[str] = []
    emotion: str | None = None

    while True:
        next_message = await recv_json_from_client(client_ws)
        if next_message is None:
            return

        message_type = next_message.get("type")
        if message_type == "input_text.append":
            buffered_text.append(str(next_message.get("text") or ""))
            emotion = str(next_message.get("emotion") or emotion or "").strip() or None
            continue
        if message_type == "input_text.commit":
            break

        await send_error(client_ws, f"Unsupported client event before commit: {message_type}")
        return

    text = normalize_text("".join(buffered_text))
    if not text:
        await send_error(client_ws, "Text must not be empty")
        return

    try:
        session_config = build_session_config(config, settings=settings, emotion=emotion)
    except (TypeError, ValueError) as err:
        await send_error(client_ws, str(err))
        return

    upstream_ws: UpstreamWebSocket | None = None
    try:
        upstream_ws = await connect_upstream(upstream_url)
        await upstream_ws.send(json_dumps(session_config))
        await upstream_ws.send(json_dumps({"type": "input.text", "text": text}))
        await upstream_ws.send(json_dumps({"type": "input.done"}))
        await forward_upstream_audio(
            client_ws=client_ws,
            upstream_ws=upstream_ws,
            idle_timeout=settings.idle_timeout,
        )
    finally:
        if upstream_ws is not None:
            await upstream_ws.close()


def create_app(
    settings: Settings | None = None,
    connect_upstream: ConnectUpstream = connect_vllm_websocket,
) -> FastAPI:
    settings = settings or load_settings_from_env()
    app = FastAPI(title="JoyVL TTS Adapter", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": settings.model,
                "upstream_url": settings.upstream_url,
                "default_voice": settings.default_voice,
                "sample_rate": settings.sample_rate,
                "output_format": settings.output_format,
            }
        )

    @app.websocket("/ws/tts")
    async def tts_websocket(client_ws: WebSocket) -> None:
        await client_ws.accept()
        try:
            await asyncio.wait_for(
                run_tts_session(
                    client_ws=client_ws,
                    upstream_url=settings.upstream_url,
                    settings=settings,
                    connect_upstream=connect_upstream,
                ),
                timeout=settings.request_timeout,
            )
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except asyncio.TimeoutError:
            logger.warning("TTS request timed out")
            try:
                await send_error(client_ws, "TTS request timed out")
            except RuntimeError:
                pass
        except (OSError, websockets.WebSocketException) as err:
            logger.warning("Upstream websocket failed: %s", err)
            try:
                await send_error(client_ws, f"TTS upstream failed: {err}")
            except RuntimeError:
                pass
        except Exception as err:
            logger.exception("TTS adapter failed")
            try:
                await send_error(client_ws, f"TTS adapter failed: {err}")
            except RuntimeError:
                pass
        finally:
            try:
                await client_ws.close()
            except RuntimeError:
                pass

    return app


async def run_smoke_test(args: argparse.Namespace) -> int:
    config = {
        "config": {
            "modalities": ["text", "audio"],
            "voice": args.voice,
            "instructions": args.instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": WEBUI_OUTPUT_FORMAT,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    }
    append = {
        "type": "input_text.append",
        "text": args.text,
        "emotion": args.emotion,
        "reqid": "smoke",
    }
    commit = {"type": "input_text.commit", "reqid": "smoke"}

    pcm = bytearray()
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json_dumps(config))
        await ws.send(json_dumps(append))
        await ws.send(json_dumps(commit))
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
            if isinstance(message, bytes):
                pcm.extend(message)
                continue
            event = json.loads(message)
            if event.get("type") == "response.done":
                break
            if event.get("type") == "error":
                raise RuntimeError(event.get("error") or event)

    output = Path(args.output)
    output.write_bytes(bytes(pcm))
    if not pcm:
        raise RuntimeError("Smoke test produced no audio")
    if len(pcm) % 2:
        raise RuntimeError(f"Smoke test produced odd pcm byte count: {len(pcm)}")
    print(f"Wrote {len(pcm)} bytes of pcm16 audio to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JoyVL TTS adapter")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the FastAPI adapter")
    serve.add_argument(
        "--host",
        default=env_value("TTS_ADAPTER_HOST", default="0.0.0.0"),
    )
    serve.add_argument(
        "--port",
        type=int,
        default=int(env_value("TTS_ADAPTER_PORT", default="8992")),
    )
    serve.add_argument("--reload", action="store_true")

    smoke = subparsers.add_parser("smoke", help="Run an end-to-end adapter smoke test")
    smoke.add_argument("--url", default="ws://127.0.0.1:8992/ws/tts")
    smoke.add_argument("--text", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--voice", default="default")
    smoke.add_argument("--emotion", default="{{高兴}}")
    smoke.add_argument(
        "--instructions",
        default="You are a helpful AI assistant. Please answer in Chinese.",
    )
    smoke.add_argument("--temperature", type=float, default=0.7)
    smoke.add_argument("--max-tokens", type=int, default=1024)
    smoke.add_argument("--timeout", type=float, default=120.0)

    parser.add_argument("--host", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "smoke":
        return asyncio.run(run_smoke_test(args))

    if args.command is None:
        args.command = "serve"
        args.host = args.host or env_value("TTS_ADAPTER_HOST", default="0.0.0.0")
        args.port = args.port or int(env_value("TTS_ADAPTER_PORT", default="8992"))
        args.reload = False

    if args.command == "serve":
        uvicorn.run(
            "tts_adapter:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
