"""
Rollout and training helpers for the recursive self-improvement environment.

This module provides tools for training an IMPROVER model that proposes
StrategyPatch mutations, rather than just improving final answers.

The key difference from standard rollout:
- Prompts include full strategy context and downstream performance
- Completions are StrategyPatch proposals (JSON)
- Rewards come from Governor accept/reject decisions on held-out evaluation
- Training teaches the model to generate better patches, not better answers
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from godel_engine.async_utils import run_async
from godel_engine.models import EditType, GodelAction, StrategyPatch
from godel_engine.recursive_environment import RecursiveSelfImprovementEnv


logger = logging.getLogger("godel_env.recursive_rollout")


# ── Prompt Templates ─────────────────────────────────────────────────

IMPROVER_SYSTEM_PROMPT = """\
You are an IMPROVER agent inside GodelEnv, a recursive self-improvement environment.

Your job is to propose mutations to a reasoning strategy that will improve downstream performance.
You will see:
1. The current strategy text
2. Recent failure cases
3. Downstream scores per evaluation domain
4. Recent patch history (accepted/rejected)

You must output a valid JSON StrategyPatch:
{
  "improved_strategy": "The complete text of your improved reasoning strategy",
  "diff_description": "What you changed and why",
  "hypothesis": "Your testable prediction about why this will improve performance",
  "target_weaknesses": ["weakness1", "weakness2"]
}

IMPORTANT:
- Your patch will be TESTED on held-out tasks you cannot see
- The Governor will REJECT patches that:
  - Don't improve overall utility
  - Cause too many regressions on individual domains
  - Have high variance across domains
  - Cause catastrophic drops on any single task
- Accepted patches become the new current strategy
- Your goal is to propose patches that ACTUALLY improve held-out performance

Focus on targeted, defensible improvements. Generic or vague patches will fail.
"""


def build_improver_prompt(
    current_strategy: str,
    downstream_scores: Dict[str, float],
    recent_failures: List[str],
    patch_history: List[Dict[str, Any]],
    current_utility: float,
    budget_remaining: int,
) -> str:
    """Build the prompt for the improver model."""
    scores_text = "\n".join(
        f"  - {domain}: {score:.3f}"
        for domain, score in sorted(downstream_scores.items())
    )

    failures_text = "\n".join(
        f"  - {f}" for f in recent_failures[-5:]
    ) if recent_failures else "  (none recorded)"

    history_text = ""
    if patch_history:
        for entry in patch_history[-3:]:
            status = "ACCEPTED" if entry.get("accepted") else "REJECTED"
            reasons = entry.get("reasons", [])
            reasons_str = f" ({'; '.join(reasons)})" if reasons else ""
            history_text += f"  - Step {entry.get('step', '?')}: {status}{reasons_str}\n"
    else:
        history_text = "  (no prior patches)"

    return f"""\
CURRENT STRATEGY:
{current_strategy}

DOWNSTREAM SCORES (by evaluation domain):
{scores_text}

CURRENT UTILITY: {current_utility:.4f}
BUDGET REMAINING: {budget_remaining} steps

RECENT FAILURES:
{failures_text}

RECENT PATCH HISTORY:
{history_text}

Analyze the current strategy's weaknesses and propose a targeted improvement.
Output your StrategyPatch as valid JSON:
"""


def parse_patch_completion(completion: str) -> Optional[StrategyPatch]:
    """Parse a model completion into a StrategyPatch."""
    # Try to extract JSON
    stripped = completion.strip()

    # Handle markdown code blocks
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                json_lines.append(line)
        stripped = "\n".join(json_lines)

    # Find JSON object
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        json_str = stripped[start:end + 1]
        try:
            data = json.loads(json_str)
            return StrategyPatch(
                improved_strategy=str(data.get("improved_strategy", "")),
                diff_description=str(data.get("diff_description", "")),
                hypothesis=str(data.get("hypothesis", "")),
                target_weaknesses=list(data.get("target_weaknesses", [])),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Failed to parse patch JSON: %s", e)

    return None


# ── Rollout Functions ────────────────────────────────────────────────

def collect_recursive_prompts(
    *,
    num_prompts: int = 32,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Collect prompts for training the improver model.

    Unlike standard prompt collection, these prompts include full strategy
    context for proposing patches.
    """
    env = RecursiveSelfImprovementEnv(seed=seed)
    dataset: List[Dict[str, Any]] = []

    async def _collect() -> None:
        for i in range(num_prompts):
            result = await env.reset(seed=seed + i)
            obs = result.observation

            prompt = build_improver_prompt(
                current_strategy=obs.current_strategy,
                downstream_scores=obs.downstream_scores,
                recent_failures=obs.recent_failures,
                patch_history=obs.patch_history,
                current_utility=obs.total_score,
                budget_remaining=obs.budget_remaining,
            )

            dataset.append({
                "prompt": IMPROVER_SYSTEM_PROMPT + "\n\n" + prompt,
                "strategy_id": obs.strategy_id,
                "current_utility": obs.total_score,
                "downstream_scores": dict(obs.downstream_scores),
                "episode_id": obs.episode_id,
            })

    run_async(_collect())
    return dataset


def recursive_reward_func(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward function for GRPO training on patch proposals.

    This executes each completion as a StrategyPatch in the environment
    and returns the reward from the Governor's decision.
    """
    env = RecursiveSelfImprovementEnv()
    rewards: List[float] = []

    async def _evaluate() -> None:
        for prompt, completion in zip(prompts, completions):
            try:
                # Reset environment
                await env.reset()

                # Parse completion as patch
                patch = parse_patch_completion(completion)
                if patch is None:
                    rewards.append(-0.1)  # Penalty for unparseable output
                    continue

                # Execute the patch
                action = GodelAction(
                    solution="",
                    edit_type=EditType.REWRITE,
                    strategy_note="GRPO training rollout",
                    strategy_patch=patch,
                )
                result = await env.step(action)
                rewards.append(result.reward)

            except Exception as e:
                logger.warning("Recursive rollout failed: %s", e)
                rewards.append(-0.1)

    run_async(_evaluate())
    return rewards


def patch_acceptance_reward_func(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Binary reward function: +1 if patch accepted, -0.5 if rejected.

    Simpler signal that focuses purely on Governor acceptance.
    """
    env = RecursiveSelfImprovementEnv()
    rewards: List[float] = []

    async def _evaluate() -> None:
        for prompt, completion in zip(prompts, completions):
            try:
                await env.reset()
                patch = parse_patch_completion(completion)
                if patch is None:
                    rewards.append(-0.5)
                    continue

                action = GodelAction(
                    solution="",
                    edit_type=EditType.REWRITE,
                    strategy_note="GRPO acceptance check",
                    strategy_patch=patch,
                )
                result = await env.step(action)

                if result.patch_decision and result.patch_decision.accepted:
                    rewards.append(1.0)
                else:
                    rewards.append(-0.5)

            except Exception as e:
                logger.warning("Acceptance check failed: %s", e)
                rewards.append(-0.5)

    run_async(_evaluate())
    return rewards


def improvement_reward_func(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Continuous reward based on utility improvement.

    Reward = improvement * 10 (scaled for clearer gradients)
    """
    env = RecursiveSelfImprovementEnv()
    rewards: List[float] = []

    async def _evaluate() -> None:
        for prompt, completion in zip(prompts, completions):
            try:
                await env.reset()
                patch = parse_patch_completion(completion)
                if patch is None:
                    rewards.append(0.0)
                    continue

                action = GodelAction(
                    solution="",
                    edit_type=EditType.REWRITE,
                    strategy_note="GRPO improvement measure",
                    strategy_patch=patch,
                )
                result = await env.step(action)

                if result.patch_decision:
                    improvement = result.patch_decision.improvement
                    rewards.append(improvement * 10.0)
                else:
                    rewards.append(0.0)

            except Exception as e:
                logger.warning("Improvement measure failed: %s", e)
                rewards.append(0.0)

    run_async(_evaluate())
    return rewards


# ── Training Reward Functions (for TRL GRPO) ─────────────────────────

RECURSIVE_REWARD_FUNCS = [
    recursive_reward_func,
    patch_acceptance_reward_func,
    improvement_reward_func,
]


# ── SFT Data Generation ──────────────────────────────────────────────

def generate_sft_data_for_patches(
    num_examples: int = 100,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generate SFT training data for patch proposals.

    Uses the heuristic policy to generate example patches, which can
    then be used for supervised fine-tuning before GRPO.
    """
    from godel_engine.heuristic_policy import build_heuristic_strategy_patch

    env = RecursiveSelfImprovementEnv(seed=seed)
    data: List[Dict[str, str]] = []

    async def _generate() -> None:
        for i in range(num_examples):
            result = await env.reset(seed=seed + i)
            obs = result.observation

            # Generate a heuristic patch
            patch = build_heuristic_strategy_patch(
                strategy_text=obs.current_strategy,
                recent_failures=obs.recent_failures,
                downstream_scores=obs.downstream_scores,
            )

            # Build the prompt
            prompt = IMPROVER_SYSTEM_PROMPT + "\n\n" + build_improver_prompt(
                current_strategy=obs.current_strategy,
                downstream_scores=obs.downstream_scores,
                recent_failures=obs.recent_failures,
                patch_history=obs.patch_history,
                current_utility=obs.total_score,
                budget_remaining=obs.budget_remaining,
            )

            # Build the completion (the patch as JSON)
            completion = json.dumps({
                "improved_strategy": patch.improved_strategy,
                "diff_description": patch.diff_description,
                "hypothesis": patch.hypothesis,
                "target_weaknesses": patch.target_weaknesses,
            }, indent=2)

            data.append({
                "prompt": prompt,
                "completion": completion,
            })

    run_async(_generate())
    return data
