import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


WEBINFER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBINFER_DIR))

from live_adapter import SessionState, StreamingInferAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_reset_session_cancels_foreground_and_summary_tasks():
    adapter = StreamingInferAdapter.__new__(StreamingInferAdapter)
    state = SessionState(session_id="session-1")
    foreground_started = asyncio.Event()
    foreground_cancelled = asyncio.Event()

    async def foreground_request():
        foreground_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            foreground_cancelled.set()

    foreground_task = asyncio.create_task(foreground_request())
    summary_task = asyncio.create_task(asyncio.Event().wait())
    state.async_pending_summary_jobs.append({"task": summary_task})
    adapter.sessions = {"session-1": state}
    adapter._active_request_tasks = {"session-1": {foreground_task}}
    adapter._flush_session_outputs = AsyncMock()

    await foreground_started.wait()
    removed, cancelled_requests = await adapter._reset_session("session-1")

    assert removed is True
    assert cancelled_requests == 1
    assert state.cancel_event.is_set()
    assert foreground_task.cancelled()
    assert foreground_cancelled.is_set()
    assert summary_task.cancelled()
    assert "session-1" not in adapter.sessions
    assert "session-1" not in adapter._active_request_tasks
    adapter._flush_session_outputs.assert_awaited_once_with(state)


@pytest.mark.asyncio
async def test_reset_missing_session_still_cancels_registered_request():
    adapter = StreamingInferAdapter.__new__(StreamingInferAdapter)
    request_task = asyncio.create_task(asyncio.Event().wait())
    adapter.sessions = {}
    adapter._active_request_tasks = {"session-2": {request_task}}
    adapter._flush_session_outputs = AsyncMock()

    removed, cancelled_requests = await adapter._reset_session("session-2")

    assert removed is False
    assert cancelled_requests == 1
    assert request_task.cancelled()
    adapter._flush_session_outputs.assert_not_awaited()
