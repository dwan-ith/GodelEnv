"""
Hybrid runtime smoke test for GodelEnv.

This script is intentionally small and non-secret. It helps confirm that the
environment is actually using an external LLM provider instead of silently
falling back to deterministic mode.

Usage:
    python hybrid_smoke.py
    python hybrid_smoke.py --require-llm
    python hybrid_smoke.py --require-llm --require-coevolution
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from godel_engine.agent import AutoAgent
from godel_engine.environment import GodelEnvironment
from godel_engine.provider_runtime import (
    ProviderCircuitBreaker,
    describe_provider_configs,
)


load_dotenv(override=False)


async def run_smoke(require_llm: bool, require_coevolution: bool = False) -> dict:
    os.environ.setdefault("GODEL_GRADING_MODE", "auto")
    os.environ.setdefault("GODEL_STRATEGY_EVAL_MODE", "llm")
    os.environ.setdefault("GODEL_AGENT_MODE", "llm")
    os.environ.setdefault("GODEL_ALLOW_DETERMINISTIC_FALLBACK", "0")
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
    environment_decision = step_result.environment_patch_decision
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
        "patch_improvement": (
            step_result.patch_decision.improvement if step_result.patch_decision else 0.0
        ),
        "strategy_eval_source_counts": patch_diagnostics.get("child_source_counts", {}),
        "strategy_eval_last_error": patch_diagnostics.get("last_error"),
        "environment_patch_proposed": action.environment_patch is not None,
        "environment_patch_accepted": bool(environment_decision and environment_decision.accepted),
        "environment_learning_value": (
            environment_decision.learning_value if environment_decision else 0.0
        ),
        "environment_rejection_reasons": (
            environment_decision.rejection_reasons if environment_decision else []
        ),
        "environment_generation": step_result.observation.environment_generation,
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
        if not report["patch_proposed"]:
            raise RuntimeError(
                "Self-improvement smoke test did not produce a StrategyPatch."
            )
        if report["patch_proposed"] and not used_llm_for_strategy_eval:
            raise RuntimeError(
                "StrategyPatch was proposed, but held-out strategy evaluation did not use an LLM."
            )

    if require_coevolution:
        if not str(report["agent_source"]).startswith("llm:"):
            raise RuntimeError("Coevolution smoke requires an LLM-backed proposer.")
        if not report["patch_proposed"] or not report["environment_patch_proposed"]:
            raise RuntimeError(
                "Coevolution smoke requires the LLM to propose both a StrategyPatch "
                "and an EnvironmentPatch in the same action."
            )
        if step_result.patch_decision is None or environment_decision is None:
            raise RuntimeError("Both coevolution Governors must return explicit decisions.")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hybrid-mode smoke test for GodelEnv")
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if no agent, grader, or strategy-eval call reaches a live provider.",
    )
    parser.add_argument(
        "--require-coevolution",
        action="store_true",
        help="Require an LLM action containing both model and environment mutations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the smoke report as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = asyncio.run(
        run_smoke(
            require_llm=args.require_llm,
            require_coevolution=args.require_coevolution,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
