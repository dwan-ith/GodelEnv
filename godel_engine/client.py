"""
Godel Env client for persistent remote sessions over the OpenEnv websocket API.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional
from urllib.parse import urlparse, urlunparse

import websockets

from godel_engine.models import GodelAction, GodelObservation, GodelState, GodelStepResult


def _to_ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/ws", "", "", ""))


def _step_result_from_ws(payload: dict) -> GodelStepResult:
    data = payload["data"]
    obs = data["observation"]
    done = bool(data.get("done", False))
    return GodelStepResult(
        observation=GodelObservation(**obs),
        reward=float(data.get("reward", 0.0) or 0.0),
        terminated=done,
        truncated=False,
        info={
            "grading_source": obs.get("grading_source"),
            "grading_error": obs.get("grading_error"),
        },
    )


class GodelEngineEnv:
    """Persistent remote client backed by the OpenEnv `/ws` session API."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_url = _to_ws_url(self.base_url)
        self._ws = None

    async def __aenter__(self):
        self._ws = await websockets.connect(self.ws_url)
        return self

    async def __aexit__(self, *args):
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass
            await self._ws.close()
            self._ws = None

    def _ensure_ws(self):
        if self._ws is None:
            raise RuntimeError(
                "Use GodelEngineEnv as an async context manager: "
                "`async with GodelEngineEnv(...) as client:`"
            )

    async def _send_and_receive(self, message: dict) -> dict:
        self._ensure_ws()
        await self._ws.send(json.dumps(message))
        response = json.loads(await self._ws.recv())
        if response.get("type") == "error":
            raise RuntimeError(response.get("data", {}).get("message", "Unknown websocket error"))
        return response

    async def reset(
        self,
        task_type: str = "",
        difficulty: str = "",
        task_id: str = "",
    ) -> GodelStepResult:
        data = {}
        if task_type:
            data["task_type"] = task_type
        if difficulty:
            data["difficulty"] = difficulty
        if task_id:
            data["task_id"] = task_id
        response = await self._send_and_receive({"type": "reset", "data": data})
        return _step_result_from_ws(response)

    async def step(self, action: GodelAction) -> GodelStepResult:
        response = await self._send_and_receive(
            {"type": "step", "data": action.model_dump(mode="json")}
        )
        return _step_result_from_ws(response)

    async def state(self) -> GodelState:
        response = await self._send_and_receive({"type": "state"})
        return GodelState(**response["data"])

    def sync(self) -> "SyncGodelEngineEnv":
        return SyncGodelEngineEnv(self)


class SyncGodelEngineEnv:
    """Synchronous wrapper around GodelEngineEnv."""

    def __init__(self, async_env: GodelEngineEnv):
        self._async_env = async_env
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async_env.__aenter__())
        return self

    def __exit__(self, *args):
        if self._loop:
            self._loop.run_until_complete(self._async_env.__aexit__(*args))
            self._loop.close()

    def reset(self, **kwargs) -> GodelStepResult:
        return self._loop.run_until_complete(self._async_env.reset(**kwargs))

    def step(self, action: GodelAction) -> GodelStepResult:
        return self._loop.run_until_complete(self._async_env.step(action))

    def state(self) -> GodelState:
        return self._loop.run_until_complete(self._async_env.state())
