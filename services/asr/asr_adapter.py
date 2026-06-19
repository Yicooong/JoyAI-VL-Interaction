# SPDX-License-Identifier: Apache-2.0

"""Joy VL Interaction ASR websocket adapter for the local speech model."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import struct
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:8993/v1/audio/transcriptions"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TIMEOUT_SECONDS = 120.0
FRAME_HEADER = struct.Struct(">iii")

logger = logging.getLogger("joyvl_asr_adapter")


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class Settings:
    upstream_url: str = DEFAULT_UPSTREAM_URL
    model: str = MODEL_NAME
    sample_rate: int = DEFAULT_SAMPLE_RATE
    request_timeout: float = DEFAULT_TIMEOUT_SECONDS


Transcriber = Callable[[bytes, int, Settings], Awaitable[str]]


def load_settings_from_env() -> Settings:
    return Settings(
        upstream_url=env_value("ASR_UPSTREAM_URL", default=DEFAULT_UPSTREAM_URL),
        model=env_value("ASR_MODEL", default=MODEL_NAME),
        sample_rate=int(
            env_value("ASR_SAMPLE_RATE", default=str(DEFAULT_SAMPLE_RATE))
        ),
        request_timeout=float(
            env_value("ASR_REQUEST_TIMEOUT", default=str(DEFAULT_TIMEOUT_SECONDS))
        ),
    )


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_request_header(headers: dict[str, str], default_sample_rate: int) -> dict[str, Any]:
    raw = headers.get("request") or "{}"
    try:
        request = json.loads(raw)
    except json.JSONDecodeError:
        request = {}
    request.setdefault("reqid", uuid.uuid4().hex)
    request.setdefault("sid", "browser-room")
    request["sample_rate"] = int(request.get("sample_rate") or default_sample_rate)
    return request


def parse_audio_frame(frame: bytes) -> tuple[int, bytes]:
    if len(frame) < FRAME_HEADER.size:
        raise ValueError("ASR frame is shorter than 12-byte header")
    seqid, _status, _reserved = FRAME_HEADER.unpack(frame[: FRAME_HEADER.size])
    return seqid, frame[FRAME_HEADER.size :]


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def extract_transcription_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


async def transcribe_with_vllm(wav_bytes: bytes, sample_rate: int, settings: Settings) -> str:
    del sample_rate
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        response = await client.post(
            settings.upstream_url,
            data={"model": settings.model},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
        response.raise_for_status()
        payload = response.json()
    return extract_transcription_text(payload).strip()


def build_asr_result(
    *,
    reqid: str,
    text: str,
    event_type: str = "IS_FINAL",
    code: int = 0,
    msg: str = "ok",
) -> dict[str, Any]:
    return {
        "code": code,
        "msg": msg,
        "mid": reqid,
        "asr_response": {
            "event_type": event_type,
            "recognition_result": {
                "hypothesis": [
                    {
                        "text": text,
                        "confidence": None,
                    }
                ]
            },
        },
    }


async def send_asr_result(
    websocket: WebSocket,
    *,
    reqid: str,
    text: str,
    code: int = 0,
    msg: str = "ok",
) -> None:
    await websocket.send_text(
        json_dumps(build_asr_result(reqid=reqid, text=text, code=code, msg=msg))
    )


async def handle_asr_websocket(
    websocket: WebSocket,
    *,
    settings: Settings,
    transcriber: Transcriber,
) -> None:
    request = parse_request_header(dict(websocket.headers), settings.sample_rate)
    sample_rate = int(request["sample_rate"])
    reqid = str(request["reqid"])
    pcm = bytearray()

    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return
        if message.get("bytes") is None:
            continue

        try:
            seqid, audio = parse_audio_frame(message["bytes"])
        except ValueError as err:
            await send_asr_result(websocket, reqid=reqid, text="", code=400, msg=str(err))
            return

        if audio:
            pcm.extend(audio)
        if seqid < 0:
            break

    if not pcm:
        await send_asr_result(websocket, reqid=reqid, text="", code=400, msg="empty audio")
        return
    if len(pcm) % 2:
        await send_asr_result(
            websocket,
            reqid=reqid,
            text="",
            code=400,
            msg="pcm16 audio byte length must be even",
        )
        return

    try:
        wav_bytes = pcm16_to_wav_bytes(bytes(pcm), sample_rate)
        text = await transcriber(wav_bytes, sample_rate, settings)
    except Exception as err:
        logger.exception("ASR transcription failed")
        await send_asr_result(websocket, reqid=reqid, text="", code=500, msg=str(err))
        return

    await send_asr_result(websocket, reqid=reqid, text=text)


def create_app(
    settings: Settings | None = None,
    transcriber: Transcriber = transcribe_with_vllm,
) -> FastAPI:
    settings = settings or load_settings_from_env()
    app = FastAPI(title="JoyVL ASR Adapter", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": settings.model,
                "upstream_url": settings.upstream_url,
                "sample_rate": settings.sample_rate,
            }
        )

    @app.websocket("/ws/asr")
    async def asr_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await asyncio.wait_for(
                handle_asr_websocket(
                    websocket,
                    settings=settings,
                    transcriber=transcriber,
                ),
                timeout=settings.request_timeout,
            )
        except WebSocketDisconnect:
            logger.info("ASR client disconnected")
        except asyncio.TimeoutError:
            logger.warning("ASR request timed out")
            try:
                request = parse_request_header(dict(websocket.headers), settings.sample_rate)
                await send_asr_result(
                    websocket,
                    reqid=str(request["reqid"]),
                    text="",
                    code=504,
                    msg="ASR request timed out",
                )
            except RuntimeError:
                pass
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app


def read_wav_as_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("smoke wav must be mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError("smoke wav must be pcm16")
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())
    return pcm, sample_rate


async def websockets_connect(url: str, headers: dict[str, str]):
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


async def run_smoke_test(args: argparse.Namespace) -> int:
    pcm, sample_rate = read_wav_as_pcm(Path(args.wav))
    request = {
        "sid": "smoke",
        "reqid": f"smoke-{uuid.uuid4().hex[:12]}",
        "sample_rate": sample_rate,
    }
    headers = {
        "request": json_dumps(request),
        "recognize": json_dumps({"do_partial_result": False}),
    }
    frame = FRAME_HEADER.pack(-1, 0, 0) + pcm

    async with await websockets_connect(args.url, headers) as ws:
        await ws.send(frame)
        while True:
            payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
            response = payload.get("asr_response") or {}
            event_type = response.get("event_type")
            if event_type in {"IS_FINAL", "IS_END"}:
                hypothesis = (
                    response.get("recognition_result", {}).get("hypothesis") or [{}]
                )[0]
                print(hypothesis.get("text", ""))
                return 0
            if payload.get("code", 0) != 0:
                raise RuntimeError(payload.get("msg") or payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JoyVL ASR adapter")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the FastAPI adapter")
    serve.add_argument(
        "--host",
        default=env_value("ASR_ADAPTER_HOST", default="0.0.0.0"),
    )
    serve.add_argument(
        "--port",
        type=int,
        default=int(env_value("ASR_ADAPTER_PORT", default="8994")),
    )
    serve.add_argument("--reload", action="store_true")

    smoke = subparsers.add_parser("smoke", help="Run an adapter smoke test")
    smoke.add_argument("--url", default="ws://127.0.0.1:8994/ws/asr")
    smoke.add_argument("--wav", required=True)
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
        args.host = args.host or env_value("ASR_ADAPTER_HOST", default="0.0.0.0")
        args.port = args.port or int(env_value("ASR_ADAPTER_PORT", default="8994"))
        args.reload = False

    if args.command == "serve":
        uvicorn.run(
            "asr_adapter:create_app",
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
