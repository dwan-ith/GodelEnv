"""
server/routers/api.py — Thin FastAPI wrapper over godel_engine.
The real logic lives in godel_engine.environment and godel_engine.agent.
"""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel

from godel_engine.models import GodelAction, GodelStepResult, GodelState
from godel_engine.environment import GodelEnvironment
from godel_engine.agent import AutoAgent
from server.deps import get_env, get_ws_manager, ConnectionManager

router = APIRouter()


class ResetRequest(BaseModel):
    task_type: Optional[str] = None
    difficulty: Optional[str] = None


@router.post("/reset", response_model=GodelStepResult)
async def reset(
    req: Optional[ResetRequest] = Body(None),
    env: GodelEnvironment = Depends(get_env),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    task_type = req.task_type if req else None
    difficulty = req.difficulty if req else None
    result = await env.reset(task_type=task_type, difficulty=difficulty)

    await manager.broadcast({
        "type": "reset",
        "data": {
            "task_id": result.observation.task_id,
            "task_type": result.observation.task_type,
            "prompt": result.observation.task_prompt,
            "initial_score": result.observation.total_score,
            "initial_solution": result.observation.current_solution,
        },
    })
    return result


@router.post("/step", response_model=GodelStepResult)
async def step(
    action: GodelAction,
    env: GodelEnvironment = Depends(get_env),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    result = await env.step(action)

    await manager.broadcast({
        "type": "step",
        "data": {
            "step": result.observation.step,
            "score": result.observation.total_score,
            "solution": result.observation.current_solution,
            "rubrics": result.observation.rubric_scores.scores,
            "feedback": result.observation.rubric_scores.feedback,
            "edit_type": action.edit_type,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "reward": result.reward,
        },
    })
    return result


@router.post("/run", response_model=GodelStepResult)
async def run_auto_step(
    env: GodelEnvironment = Depends(get_env),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """Run one LLM auto-improvement step using AutoAgent."""
    if env.current_task is None:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")

    agent = AutoAgent()
    action = await agent.act(
        task_prompt=env.current_instance.prompt,
        current_solution=env.current_solution,
        rubrics=env.current_task._get_rubrics(),
        task_type=env.current_task.name,
    )
    result = await env.step(action)

    await manager.broadcast({
        "type": "step",
        "data": {
            "step": result.observation.step,
            "score": result.observation.total_score,
            "solution": result.observation.current_solution,
            "rubrics": result.observation.rubric_scores.scores,
            "feedback": result.observation.rubric_scores.feedback,
            "edit_type": action.edit_type,
            "strategy_note": action.strategy_note,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "reward": result.reward,
        },
    })
    return result


@router.get("/state", response_model=GodelState)
async def get_state(env: GodelEnvironment = Depends(get_env)):
    return env.state()
