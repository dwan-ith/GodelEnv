"""
Demo-only helper routes.

The canonical environment API is provided by openenv-core at `/reset`, `/step`,
`/state`, and `/ws`. This router only exposes helper endpoints used by the
dashboard demo, such as server-side AutoAgent action generation.
"""
from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel

from godel_engine.agent import AutoAgent
from godel_engine.environment import GodelEnvironment
from godel_engine.models import GodelAction
from godel_engine.provider_runtime import describe_provider_configs


router = APIRouter(prefix="/demo", tags=["demo"])


class DemoActRequest(BaseModel):
    task_prompt: str
    current_solution: str
    task_type: str
    current_strategy: str = ""
    recent_failures: list[str] = []
    downstream_scores: dict[str, float] = {}


@router.post("/act", response_model=GodelAction)
async def demo_act(req: DemoActRequest) -> GodelAction:
    """Generate the next action for the dashboard demo."""
    task_cls = GodelEnvironment.TASKS.get(req.task_type)
    task = task_cls() if task_cls else None
    rubrics = task._get_rubrics() if task else {}

    agent = AutoAgent()
    return await agent.act(
        task_prompt=req.task_prompt,
        current_solution=req.current_solution,
        rubrics=rubrics,
        task_type=req.task_type,
        strategy_text=req.current_strategy,
        recent_failures=req.recent_failures,
        downstream_scores=req.downstream_scores,
    )


@router.get("/provider-status")
async def provider_status() -> dict:
    """Return non-secret hybrid runtime diagnostics for the dashboard/demo."""
    return {
        "grading_mode": os.getenv("GODEL_GRADING_MODE", "auto"),
        "strategy_eval_mode": os.getenv("GODEL_STRATEGY_EVAL_MODE", "auto"),
        "providers": describe_provider_configs(),
    }
