"""
Godel Env -- Before/After Demo Script

Demonstrates the improvement of a trained model vs a base model
by running episodes on the GodelEnv and showing side-by-side results.

Usage:
    python demo.py
    python demo.py --base-model Qwen/Qwen2.5-7B-Instruct --trained-model checkpoints/final
    python demo.py --space-url https://litterarum-godelenv.hf.space
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo")

# Import env components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from godel_engine.environment import GodelEnvironment
from godel_engine.agent import AutoAgent
from godel_engine.models import GodelAction, EditType


def print_banner(text: str, width: int = 70):
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_reward_breakdown(breakdown):
    print(f"    task_score_delta:  {breakdown.task_score_delta:+.4f}")
    print(f"    format_compliance: {breakdown.format_compliance:+.4f}")
    print(f"    step_cost:         {breakdown.step_cost:+.4f}")
    print(f"    anti_hack_penalty: {breakdown.anti_hack_penalty:+.4f}")
    print(f"    process_reward:    {breakdown.process_reward:+.4f}")
    print(f"    TOTAL:             {breakdown.total:+.4f}")


async def run_demo_episode(
    env: GodelEnvironment,
    agent: AutoAgent,
    task_type: str,
    label: str,
    seed: int = 42,
) -> dict:
    """Run a single demo episode and return metrics."""
    result = await env.reset(task_type=task_type, seed=seed)
    obs = result.observation

    print(f"\n  [{label}] Task: {task_type} | Difficulty: {obs.difficulty}")
    print(f"  [{label}] Initial score: {obs.total_score:.3f}")
    print(f"  [{label}] Prompt: {obs.task_prompt[:80]}...")

    steps = 0
    while not (result.terminated or result.truncated):
        action = await agent.act(
            task_prompt=obs.task_prompt,
            current_solution=obs.current_solution,
            rubrics=env.current_task._get_rubrics(),
            task_type=task_type,
            strategy_text=obs.current_strategy,
            recent_failures=obs.recent_failures,
            downstream_scores=obs.downstream_scores,
        )
        result = await env.step(action)
        obs = result.observation
        steps += 1

        reason = result.info.get("reason", "running")
        violations = result.info.get("guard_violations", [])
        print(f"  [{label}] Step {steps}: score={obs.total_score:.3f} | "
              f"reward={result.reward:+.4f} | reason={reason}")

        if violations:
            print(f"  [{label}]   Guards: {'; '.join(violations)}")

        # Show reward breakdown on last step
        if result.terminated or result.truncated:
            print(f"  [{label}] Reward Breakdown:")
            print_reward_breakdown(result.reward_breakdown)

    return {
        "label": label,
        "task": task_type,
        "initial_score": env.initial_score,
        "final_score": obs.total_score,
        "delta": obs.total_score - env.initial_score,
        "steps": steps,
        "reason": result.info.get("reason", "unknown"),
        "difficulty": env.episode_difficulty,
    }


async def main(args):
    print_banner("GODEL ENV -- BEFORE / AFTER DEMO")

    env = GodelEnvironment(seed=42)
    agent = AutoAgent()

    tasks = ["factual_qa", "code_improvement", "strategy_optimization"]
    results = []

    # Run baseline episodes
    print_banner("BASELINE MODEL EPISODES")
    for task in tasks:
        metrics = await run_demo_episode(env, agent, task, "BASE", seed=42)
        results.append(metrics)

    # Summary table
    print("\n")
    print_banner("RESULTS SUMMARY")
    print(f"\n  {'Label':<10} {'Task':<25} {'Init':>6} {'Final':>6} {'Delta':>7} {'Steps':>6} {'Reason':<15}")
    print(f"  {'-'*10} {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*15}")
    for r in results:
        print(f"  {r['label']:<10} {r['task']:<25} {r['initial_score']:>6.3f} "
              f"{r['final_score']:>6.3f} {r['delta']:>+7.3f} {r['steps']:>6} {r['reason']:<15}")

    # Curriculum stats
    print(f"\n  Curriculum State:")
    stats = env.curriculum.get_stats()
    for key, val in stats.items():
        print(f"    {key}: {val}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("  To train a model: python train.py --dry-run")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GodelEnv Before/After Demo")
    parser.add_argument("--space-url", type=str, default=None,
                        help="If set, runs against remote Space instead of local env")
    args = parser.parse_args()
    asyncio.run(main(args))
