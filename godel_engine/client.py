"""
Gödel Engine — EnvClient for remote access.

Usage (async — recommended):
    async with GodelEngineEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(GodelAction(solution="..."))

Usage (sync):
    with GodelEngineEnv(base_url="http://localhost:8000").sync() as client:
        result = client.reset()
        result = client.step(GodelAction(solution="..."))
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import httpx

from godel_engine.models import (
    GodelAction,
    GodelObservation,
    GodelState,
    GodelStepResult,
)


class GodelEngineEnv:
    """
    Async HTTP client for interacting with a remote Gödel Engine server.
    Implements the OpenEnv EnvClient interface.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self):
        if self._client is None:
            raise RuntimeError(
                "Use GodelEngineEnv as an async context manager: "
                "`async with GodelEngineEnv(...) as client:`"
            )

    async def reset(self, task_type: str = "", difficulty: str = "") -> GodelStepResult:
        """Reset the environment and get initial observation."""
        self._ensure_client()
        body = {}
        if task_type:
            body["task_type"] = task_type
        if difficulty:
            body["difficulty"] = difficulty

        resp = await self._client.post("/reset", json=body)
        resp.raise_for_status()
        data = resp.json()
        return GodelStepResult(**data)

    async def step(self, action: GodelAction) -> GodelStepResult:
        """Submit an action and get the result."""
        self._ensure_client()
        resp = await self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        data = resp.json()
        return GodelStepResult(**data)

    async def state(self) -> GodelState:
        """Get episode metadata."""
        self._ensure_client()
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return GodelState(**resp.json())

    # ── Sync wrapper ──────────────────────────────────────────────────

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
