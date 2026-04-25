"""
GodelEnv 2.0 — Recursive Self-Improvement Demo

This demo showcases the core recursive self-improvement loop:
1. Agent observes current strategy + downstream scores + failures
2. Agent proposes a StrategyPatch (mutation)
3. Governor evaluates parent vs child on held-out domains
4. Patch is accepted or rejected based on multi-objective utility
5. Accepted patches become the new current strategy

Unlike the legacy demo which focused on answer improvement, this demo
shows the recursive ladder of improvement.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv


load_dotenv(override=False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo_recursive")


def print_banner(text: str, width: int = 80, char: str = "="):
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_patch_decision(decision, step: int):
    """Print a formatted patch decision."""
    status = "ACCEPTED" if decision.accepted else "REJECTED"
    color = "\033[92m" if decision.accepted else "\033[91m"  # Green or Red
    reset = "\033[0m"

    print(f"\n  [{step}] Patch {color}{status}{reset}")
    print(f"      Parent utility: {decision.parent_utility:.4f}")
    print(f"      Child utility:  {decision.child_utility:.4f}")
    print(f"      Improvement:    {decision.improvement:+.4f}")
    print(f"      Tasks evaluated: {decision.tasks_evaluated}")
    print(f"      Regressions: {decision.regression_count}")

    if decision.rejection_reasons:
        print(f"      Reasons: {'; '.join(decision.rejection_reasons)}")

    if decision.axis_scores:
        print("      Axis scores:")
        for axis, value in decision.axis_scores.items():
            print(f"        {axis}: {value:.3f}")


def print_lineage(lineage: list[str], current_id: str):
    """Print the strategy lineage."""
    print("\n  Strategy Lineage:")
    for i, strat_id in enumerate(lineage):
        marker = "→ " if strat_id == current_id else "  "
        indent = "  " * i
        print(f"    {indent}{marker}{strat_id}")


def print_leaderboard(leaderboard: list[dict], top_n: int = 5):
    """Print the strategy leaderboard."""
    print("\n  Strategy Leaderboard (by Elo):")
    print(f"    {'Rank':<6}{'ID':<15}{'Elo':<10}{'Fitness':<10}{'Gen':<6}{'Evals':<8}")
    print(f"    {'-'*6}{'-'*15}{'-'*10}{'-'*10}{'-'*6}{'-'*8}")
    for i, entry in enumerate(leaderboard[:top_n], 1):
        print(
            f"    {i:<6}"
            f"{entry['id']:<15}"
            f"{entry['elo']:<10.1f}"
            f"{entry['fitness']:<10.3f}"
            f"{entry['generation']:<6}"
            f"{entry['total_evaluations']:<8}"
        )


async def run_recursive_demo(max_episodes: int = 3, steps_per_episode: int = 5):
    """Run the recursive self-improvement demo."""
    from godel_engine.agent import AutoAgent
    from godel_engine.recursive_environment import RecursiveSelfImprovementEnv

    print_banner("GODELENV 2.0 — RECURSIVE SELF-IMPROVEMENT DEMO")
    print("""
  This demo shows the recursive self-improvement loop:
  - Strategy patches are proposed by the agent
  - Governor evaluates parent vs child on held-out domains
  - Only patches that improve multi-objective utility are accepted
  - Lineage and Elo track strategy evolution over time
""")

    env = RecursiveSelfImprovementEnv(seed=42, max_steps=steps_per_episode)
    agent = AutoAgent()

    total_patches_proposed = 0
    total_patches_accepted = 0

    for episode in range(max_episodes):
        print_banner(f"EPISODE {episode + 1}/{max_episodes}", char="-")

        result = await env.reset()
        obs = result.observation

        print(f"\n  Starting strategy: {obs.strategy_id}")
        print(f"  Generation: {obs.strategy_generation}")
        print(f"  Initial utility: {obs.total_score:.4f}")
        print(f"  Downstream scores: {json.dumps(obs.downstream_scores, indent=4)}")

        step = 0
        while not (result.terminated or result.truncated):
            step += 1
            print(f"\n  --- Step {step}/{obs.budget_remaining + step} ---")

            # Agent proposes a patch
            action = await agent.act(
                task_prompt=obs.task_prompt,
                current_solution="",
                rubrics={},
                task_type="strategy_improvement",
                strategy_text=obs.current_strategy,
                recent_failures=obs.recent_failures,
                downstream_scores=obs.downstream_scores,
            )

            agent_mode = "LLM" if agent.last_source.startswith("llm:") else "HEURISTIC"
            print(f"  Agent mode: {agent_mode}")
            if agent.last_error:
                print(f"  Agent note: {agent.last_error[:80]}...")

            if action.strategy_patch:
                print(f"  Patch hypothesis: {action.strategy_patch.hypothesis[:80]}...")
                print(f"  Target weaknesses: {action.strategy_patch.target_weaknesses[:3]}")

            # Execute the patch
            result = await env.step(action)
            obs = result.observation
            total_patches_proposed += 1

            if result.patch_decision:
                print_patch_decision(result.patch_decision, step)
                if result.patch_decision.accepted:
                    total_patches_accepted += 1

            print(f"  Current utility: {obs.total_score:.4f}")
            print(f"  Reward: {result.reward:+.4f}")

        # Episode summary
        state = env.state()
        print(f"\n  Episode finished: {result.info.get('reason', 'unknown')}")
        print(f"  Final utility: {state.current_score:.4f}")
        print(f"  Patches proposed: {state.patches_proposed}")
        print(f"  Patches accepted: {state.patches_accepted}")
        print(f"  Cumulative reward: {state.cumulative_reward:.4f}")

        # Show lineage
        print_lineage(state.strategy_lineage, obs.strategy_id)

    # Final summary
    print_banner("DEMO SUMMARY")
    print(f"""
  Total episodes: {max_episodes}
  Total patches proposed: {total_patches_proposed}
  Total patches accepted: {total_patches_accepted}
  Acceptance rate: {total_patches_accepted / total_patches_proposed * 100:.1f}%
""")

    # Show final leaderboard
    leaderboard = env.get_leaderboard()
    print_leaderboard(leaderboard)

    # Show Governor stats
    gov_stats = env.governor.get_stats()
    print(f"\n  Governor Stats:")
    print(f"    Total decisions: {gov_stats['total_decisions']}")
    print(f"    Acceptance rate: {gov_stats['acceptance_rate']:.1%}")

    print_banner("DEMO COMPLETE")
    print("""
  The recursive self-improvement loop demonstrates:
  1. Strategies evolve through accepted patches
  2. Governor rejects patches that don't improve held-out performance
  3. Lineage tracking shows strategy ancestry
  4. Elo ratings rank strategies by relative performance

  This is a practical approximation of Gödel-machine-style self-improvement:
  - Self-modification is explicit (StrategyPatch)
  - Improvement proposals are testable (held-out evaluation)
  - Acceptance depends on objective evidence (Governor)
""")


def main():
    parser = argparse.ArgumentParser(
        description="GodelEnv 2.0 Recursive Self-Improvement Demo"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Maximum steps per episode",
    )
    args = parser.parse_args()

    asyncio.run(run_recursive_demo(
        max_episodes=args.episodes,
        steps_per_episode=args.steps,
    ))


if __name__ == "__main__":
    main()
