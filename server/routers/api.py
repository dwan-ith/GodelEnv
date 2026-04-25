"""
Demo-only helper routes.

The canonical environment API is provided by openenv-core at `/reset`, `/step`,
`/state`, and `/ws`. This router only exposes helper endpoints used by the
dashboard demo, such as server-side AutoAgent action generation.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from godel_engine.agent import AutoAgent
from godel_engine.environment import GodelEnvironment
from godel_engine.models import GodelAction
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    describe_provider_configs,
    describe_provider_environment,
)


router = APIRouter(prefix="/demo", tags=["demo"])


class DemoActRequest(BaseModel):
    task_prompt: str
    current_solution: str
    task_type: str
    current_strategy: str = ""
    recent_failures: list[str] = []
    downstream_scores: dict[str, float] = {}


class DemoActResponse(BaseModel):
    """Extended response that includes the action plus diagnostics."""

    solution: str
    edit_type: str
    strategy_note: str
    strategy_patch: dict[str, Any] | None = None
    agent_source: str
    agent_provider: str | None
    agent_error: str | None
    is_llm_generated: bool


@router.post("/act")
async def demo_act(req: DemoActRequest) -> DemoActResponse:
    """Generate the next action for the dashboard demo with full diagnostics."""
    require_llm = os.getenv("GODEL_REQUIRE_LLM", "").lower() in ("1", "true", "yes")

    task_cls = GodelEnvironment.TASKS.get(req.task_type)
    task = task_cls() if task_cls else None
    rubrics = task._get_rubrics() if task else {}

    agent = AutoAgent()
    action = await agent.act(
        task_prompt=req.task_prompt,
        current_solution=req.current_solution,
        rubrics=rubrics,
        task_type=req.task_type,
        strategy_text=req.current_strategy,
        recent_failures=req.recent_failures,
        downstream_scores=req.downstream_scores,
    )

    is_llm = agent.last_source.startswith("llm:")

    if require_llm and not is_llm:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM required but unavailable",
                "source": agent.last_source,
                "provider": agent.last_provider,
                "reason": agent.last_error,
            },
        )

    patch_dict = None
    if action.strategy_patch:
        patch_dict = {
            "improved_strategy": action.strategy_patch.improved_strategy,
            "diff_description": action.strategy_patch.diff_description,
            "hypothesis": action.strategy_patch.hypothesis,
            "target_weaknesses": action.strategy_patch.target_weaknesses,
        }

    return DemoActResponse(
        solution=action.solution,
        edit_type=action.edit_type.value if hasattr(action.edit_type, "value") else str(action.edit_type),
        strategy_note=action.strategy_note,
        strategy_patch=patch_dict,
        agent_source=agent.last_source,
        agent_provider=agent.last_provider,
        agent_error=agent.last_error,
        is_llm_generated=is_llm,
    )


@router.get("/provider-status")
async def provider_status() -> dict:
    """Return non-secret hybrid runtime diagnostics for the dashboard/demo."""
    return {
        "grading_mode": os.getenv("GODEL_GRADING_MODE", "auto"),
        "strategy_eval_mode": os.getenv("GODEL_STRATEGY_EVAL_MODE", "auto"),
        "env_presence": describe_provider_environment(),
        "providers": describe_provider_configs(),
    }


@router.post("/provider-status/reset")
async def reset_provider_status() -> dict:
    """Clear runtime provider circuit breakers without exposing credentials."""
    ProviderCircuitBreaker.reset()
    return {
        "ok": True,
        "providers": describe_provider_configs(),
    }


@router.get("/provider-test")
async def provider_test() -> dict:
    """
    Test LLM provider connectivity by making a simple API call.
    Returns detailed diagnostics about what succeeded/failed.
    """
    from godel_engine.provider_runtime import load_provider_configs
    from godel_engine.llm_json import parse_llm_json_object
    from openai import AsyncOpenAI
    import asyncio

    results = []
    configs = load_provider_configs()

    for config in configs:
        if not config.api_key:
            results.append({
                "provider": config.name,
                "status": "skipped",
                "reason": "no API key",
            })
            continue

        if ProviderCircuitBreaker.is_disabled(config.name):
            results.append({
                "provider": config.name,
                "status": "disabled",
                "reason": ProviderCircuitBreaker.reason(config.name),
            })
            continue

        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url if config.base_url else None,
        )

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.model_name,
                    messages=[
                        {"role": "user", "content": "Reply with exactly: {\"test\": \"ok\"}"}
                    ],
                    max_tokens=50,
                ),
                timeout=30,
            )
            content = response.choices[0].message.content or ""
            results.append({
                "provider": config.name,
                "model": config.model_name,
                "base_url": config.base_url,
                "status": "success",
                "raw_response": content[:200],
            })
        except Exception as e:
            results.append({
                "provider": config.name,
                "model": config.model_name,
                "base_url": config.base_url,
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })

    return {
        "env_presence": describe_provider_environment(),
        "test_results": results,
    }


@router.get("/provider-test-full")
async def provider_test_full() -> dict:
    """
    Test LLM with a realistic JSON prompt to see raw output.
    """
    from godel_engine.provider_runtime import load_provider_configs
    from godel_engine.llm_json import parse_llm_json_object
    from openai import AsyncOpenAI
    import asyncio
    import json

    configs = load_provider_configs()
    hf_config = next((c for c in configs if c.name == 'huggingface'), None)
    
    if not hf_config or not hf_config.api_key:
        return {"error": "No HuggingFace provider configured"}

    client = AsyncOpenAI(
        api_key=hf_config.api_key,
        base_url=hf_config.base_url,
    )

    prompt = '''You are an AI agent. Return raw JSON (no markdown).
{"solution": "your answer", "edit_type": "rewrite", "strategy_note": "explanation"}'''

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=hf_config.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                max_tokens=500,
            ),
            timeout=60,
        )
        content = response.choices[0].message.content or ""
        
        # Try parsing
        parse_result = None
        parse_error = None
        try:
            parse_result = parse_llm_json_object(content)
        except json.JSONDecodeError as e:
            parse_error = str(e)

        return {
            "raw_content": content,
            "content_length": len(content),
            "parse_result": parse_result,
            "parse_error": parse_error,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {str(e)}",
        }
