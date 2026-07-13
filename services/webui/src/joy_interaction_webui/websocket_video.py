"""Video track fed by JPEG frames received over a WebSocket."""

import asyncio
import io
import time
from fractions import Fraction

import av
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from PIL import Image


class WebSocketVideoTrack(VideoStreamTrack):
    """Keep only the newest browser frame to avoid building up latency."""

    def __init__(self) -> None:
        super().__init__()
        self._frames: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._started_at = time.monotonic()

    @staticmethod
    def _decode_jpeg(data: bytes) -> av.VideoFrame:
        with Image.open(io.BytesIO(data)) as image:
            rgb = np.asarray(image.convert("RGB"))
        return av.VideoFrame.from_ndarray(rgb, format="rgb24")

    async def put_jpeg(self, data: bytes) -> None:
        if self.readyState != "live":
            return
        frame = await asyncio.to_thread(self._decode_jpeg, data)
        frame.time_base = Fraction(1, 1000)
        frame.pts = int((time.monotonic() - self._started_at) * 1000)
        if self._frames.full():
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._frames.put_nowait(frame)

    async def recv(self) -> av.VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError
        frame = await self._frames.get()
        if frame is None:
            raise MediaStreamError
        return frame

    def stop(self) -> None:
        if self.readyState == "live":
            super().stop()
            if self._frames.full():
                try:
                    self._frames.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._frames.put_nowait(None)
