from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from ..jobs import TERMINAL, JobNotFound
from .deps import State

router = APIRouter(tags=["jobs"])
_POLL_SECONDS = 0.05


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, state: State) -> dict:
    return state.jobs.get(job_id).snapshot()


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str, state: State) -> dict:
    return state.jobs.cancel(job_id).snapshot()


@router.websocket("/ws/jobs/{job_id}")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        job = websocket.app.state.vsp.jobs.get(job_id)
    except JobNotFound:
        await websocket.close(code=4404, reason="unknown job")
        return
    sent = 0
    while True:
        terminal = job.status in TERMINAL  # read status BEFORE events: the final event precedes the status flip
        for event in job.events_since(sent):
            await websocket.send_json(event)
            sent += 1
        if terminal:
            break
        await asyncio.sleep(_POLL_SECONDS)
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):  # the client may already be gone
        await websocket.close()
