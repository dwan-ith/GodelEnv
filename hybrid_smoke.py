"""
Hybrid runtime smoke test for GodelEnv.

This script is intentionally small and non-secret. It helps confirm that the
environment is actually using an external LLM provider instead of silently
falling back to deterministic mode.

Usage:
    python hybrid_smoke.py
    python hybrid_smoke.py --require-llm
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv

from godel_engine.agent import AutoAgent
from godel_engine.environment import GodelEnvironment
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    describe_provider_configs,
)


load_dotenv(override=False)


async def run_smoke(require_llm: bool) -> dict:
    os.environ.setdefault("GODEL_GRADING_MODE", "auto")
    os.environ.setdefault("GODEL_STRATEGY_EVAL_MODE", "auto")
    ProviderCircuitBreaker.reset()

    env = GodelEnvironment(seed=42)
    reset_result = await env.reset(task_type="strategy_optimization", task_id="godel02", seed=42)
    task = env.current_task

    agent = AutoAgent()
    action = await agent.act(
        task_prompt=reset_result.observation.task_prompt,
        current_solution=reset_result.observation.current_solution,
        rubrics=task._get_rubrics() if task else {},
        task_type=reset_result.observation.task_type,
        strategy_text=reset_result.observation.current_strategy,
        recent_failures=reset_result.observation.recent_failures,
        downstream_scores=reset_result.observation.downstream_scores,
    )
    step_result = await env.step(action)

    patch_diagnostics = (
        step_result.patch_decision.diagnostics if step_result.patch_decision else {}
    )
    report = {
        "modes": {
            "grading": os.getenv("GODEL_GRADING_MODE", "auto"),
            "strategy_eval": os.getenv("GODEL_STRATEGY_EVAL_MODE", "auto"),
        },
        "providers": describe_provider_configs(),
        "agent_source": agent.last_source,
        "agent_provider": agent.last_provider,
        "agent_error": agent.last_error,
        "grading_source": step_result.info.get("grading_source"),
        "grading_error": step_result.info.get("grading_error"),
        "reward": step_result.reward,
        "score": step_result.observation.total_score,
        "patch_proposed": action.strategy_patch is not None,
        "patch_accepted": step_result.patch_decision.accepted if step_result.patch_decision else False,
        "strategy_eval_source_counts": patch_diagnostics.get("child_source_counts", {}),
        "strategy_eval_last_error": patch_diagnostics.get("last_error"),
        "circuit_breaker": {
            "disabled": bool(
                [provider for provider in describe_provider_configs() if provider.get("disabled")]
            ),
            "reason": ProviderCircuitBreaker.reason(),
        },
    }

    if require_llm:
        used_llm_for_agent = str(report["agent_source"]).startswith("llm:")
        used_llm_for_grading = str(report["grading_source"]).startswith("llm:")
        used_llm_for_strategy_eval = any(
            str(source).startswith("llm:")
            for source in report["strategy_eval_source_counts"]
        )
        if not (used_llm_for_agent or used_llm_for_grading or used_llm_for_strategy_eval):
            raise RuntimeError(
                "Hybrid smoke test did not observe any live LLM-backed path. "
                "Check provider credentials, network access, or provider status."
            )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hybrid-mode smoke test for GodelEnv")
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if no agent, grader, or strategy-eval call reaches a live provider.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = asyncio.run(run_smoke(require_llm=args.require_llm))
    print(json.dumps(result, indent=2))
