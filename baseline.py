"""
Gödel Env — Standalone Baseline Inference Script.

Runs a full RL training loop directly in Python — no web server required.
The agent (LLM) interacts with GodelEnvironment via the standard
step() / reset() / state() API.

Usage:
    python baseline.py
    python baseline.py --tasks factual_qa code_improvement reasoning
    python baseline.py --episodes 5 --seed 42 --output results.json
"""
from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from typing import List, Optional

# Force UTF-8 output on Windows to avoid charmap codec errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(override=True)

# ── Allow running from project root without `pip install -e .` ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from godel_engine.environment import GodelEnvironment
from godel_engine.agent import AutoAgent
from godel_engine.models import GodelAction, EditType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("baseline")

# ── Default task suite (easy → medium → hard) ──────────────────────────
ALL_TASKS = [
    "factual_qa",        # easy   — open-ended explanation
    "alignment_qa",      # easy   — AI safety concepts
    "code_improvement",  # medium — Python correctness + tests
    "python_optimized",  # medium — performance + docs
    "reasoning",         # medium — multi-step logic
    "adr_writing",       # hard   — structured technical writing
    "strategy_optimization",  # hard — meta-strategy synthesis
]


# ── Helpers ─────────────────────────────────────────────────────────────

def print_banner(text: str, width: int = 60):
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_step(step: int, score: float, reward: float, note: str):
    bar_len = 30
    filled = int(score * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  step {step:02d} | [{bar}] {score:.3f} | reward={reward:+.4f} | {note[:50]}")


# ── Core eval loop ───────────────────────────────────────────────────────

async def run_episode(
    env: GodelEnvironment,
    agent: AutoAgent,
    task_type: str,
    episode_num: int,
    seed: Optional[int],
) -> dict:
    """Run a single RL episode and return metrics."""
    ep_seed = (seed + episode_num) if seed is not None else None
    result = await env.reset(task_type=task_type, seed=ep_seed)
    obs = result.observation

    initial_score = obs.total_score
    logger.info(f"[Ep {episode_num}] task={task_type} | initial_score={initial_score:.3f}")
    logger.info(f"  Prompt: {obs.task_prompt[:80]}...")

    step_records = []
    cumulative_reward = 0.0

    while not (result.terminated or result.truncated):
        inferred_rubrics = {k: f"Optimize {k}" for k in obs.rubric_scores.scores.keys()}
        action = await agent.act(
            task_prompt=obs.task_prompt,
            current_solution=obs.current_solution,
            rubrics=inferred_rubrics,
            task_type=task_type,
            strategy_text=obs.current_strategy,
            recent_failures=obs.recent_failures,
            downstream_scores=obs.downstream_scores,
        )
        result = await env.step(action)
        obs = result.observation
        cumulative_reward += result.reward

        print_step(obs.step, obs.total_score, result.reward, action.strategy_note)
        step_records.append({
            "step": obs.step,
            "score": obs.total_score,
            "reward": result.reward,
            "edit_type": action.edit_type,
            "strategy_note": action.strategy_note,
        })

    final_score = obs.total_score
    delta = final_score - initial_score
    state = env.state()

    return {
        "episode": episode_num,
        "task_type": task_type,
        "initial_score": initial_score,
        "final_score": final_score,
        "delta": delta,
        "steps": obs.step,
        "cumulative_reward": cumulative_reward,
        "best_score": state.best_score,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "trajectory": step_records,
    }


async def run_baseline(
    tasks: List[str],
    episodes: int,
    seed: Optional[int],
    output: Optional[str],
):
    print_banner("GÖDEL ENV — BASELINE EVALUATION")
    print(f"  Tasks:    {tasks}")
    print(f"  Episodes: {episodes}")
    print(f"  Seed:     {seed}")
    print()

    env = GodelEnvironment(seed=seed)
    agent = AutoAgent()

    if not agent.clients:
        logger.warning("No API clients configured — agent will return no-op actions.")

    all_results = []
    t0 = time.time()

    for ep_idx in range(episodes):
        task = tasks[ep_idx % len(tasks)]
        print(f"\n—— Episode {ep_idx + 1}/{episodes} | {task} ——")
        try:
            metrics = await run_episode(env, agent, task, ep_idx + 1, seed)
            all_results.append(metrics)
            print(
                f"  ✓ Done | final={metrics['final_score']:.3f} "
                f"Δ={metrics['delta']:+.3f} | steps={metrics['steps']}"
            )
        except Exception as e:
            logger.error(f"Episode {ep_idx + 1} failed: {e}")
            all_results.append({"episode": ep_idx + 1, "task_type": task, "error": str(e)})

    elapsed = time.time() - t0

    # ── Summary table ──────────────────────────────────────────────────
    print()
    print_banner("RESULTS SUMMARY")
    successful = [r for r in all_results if "error" not in r]
    if successful:
        avg_init = sum(r["initial_score"] for r in successful) / len(successful)
        avg_final = sum(r["final_score"] for r in successful) / len(successful)
        avg_delta = sum(r["delta"] for r in successful) / len(successful)

        print(f"  Episodes run:   {len(all_results)}")
        print(f"  Successful:     {len(successful)}")
        print(f"  Avg init score: {avg_init:.4f}")
        print(f"  Avg final score:{avg_final:.4f}")
        print(f"  Avg Δ score:    {avg_delta:+.4f}")
        print(f"  Wall time:      {elapsed:.1f}s")
        print()

        # Per-task breakdown
        from collections import defaultdict
        by_task = defaultdict(list)
        for r in successful:
            by_task[r["task_type"]].append(r)

        print(f"  {'Task':<25} {'Episodes':>9} {'Avg Score':>10} {'Avg Δ':>8}")
        print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8}")
        for task_name, records in sorted(by_task.items()):
            t_avg_score = sum(r["final_score"] for r in records) / len(records)
            t_avg_delta = sum(r["delta"] for r in records) / len(records)
            print(f"  {task_name:<25} {len(records):>9} {t_avg_score:>10.4f} {t_avg_delta:>+8.4f}")

    print("=" * 60)

    # ── Save JSON output ───────────────────────────────────────────────
    if output:
        with open(output, "w") as f:
            json.dump({
                "config": {"tasks": tasks, "episodes": episodes, "seed": seed},
                "summary": {
                    "avg_initial": avg_init if successful else None,
                    "avg_final": avg_final if successful else None,
                    "avg_delta": avg_delta if successful else None,
                    "wall_time_seconds": elapsed,
                },
                "episodes": all_results,
            }, f, indent=2)
        print(f"\n  Results saved → {output}")


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation on Gödel Env."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        choices=ALL_TASKS,
        help="Which tasks to evaluate (default: all).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=len(ALL_TASKS),
        help="Total episodes to run (cycles through tasks).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )
    args = parser.parse_args()
    asyncio.run(run_baseline(args.tasks, args.episodes, args.seed, args.output))
