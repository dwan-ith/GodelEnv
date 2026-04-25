"""
Godel Env -- Rollout function for TRL GRPOTrainer integration.

Provides make_rollout_func() that creates a rollout function compatible
with TRL's GRPOTrainer(rollout_func=...) interface.

The rollout function:
  1. Resets the remote GodelEnv (via HTTP client)
  2. Generates a completion using the trainer's model
  3. Parses the completion into a GodelAction
  4. Steps through the environment
  5. Collects multi-channel rewards

Works against a remote HF Space deployment of GodelEnv.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from godel_engine.models import GodelAction, EditType

logger = logging.getLogger("godel_env.rollout")


def parse_completion_to_action(completion: str) -> GodelAction:
    """
    Parse a model completion into a GodelAction.

    Tries to extract JSON from the completion. Falls back to treating
    the entire completion as the solution text.
    """
    # Try to find JSON in the completion
    json_match = re.search(r'\{[^{}]*"solution"[^{}]*\}', completion, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            solution = data.get("solution", completion)
            edit_type_str = data.get("edit_type", "rewrite").upper()
            try:
                edit_type = EditType[edit_type_str]
            except KeyError:
                edit_type = EditType.REWRITE
            return GodelAction(
                solution=str(solution),
                edit_type=edit_type,
                strategy_note=str(data.get("strategy_note", "RL rollout")),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Try markdown code block extraction
    code_match = re.search(r'```(?:python)?\s*\n?(.*?)```', completion, re.DOTALL)
    if code_match:
        return GodelAction(
            solution=code_match.group(1).strip(),
            edit_type=EditType.REWRITE,
            strategy_note="Extracted from code block",
        )

    # Fallback: use the entire completion as the solution
    return GodelAction(
        solution=completion.strip(),
        edit_type=EditType.REWRITE,
        strategy_note="Raw completion used as solution",
    )


def build_prompt(task_prompt: str, current_solution: str, rubric_feedback: str) -> str:
    """Build the prompt string sent to the model during rollout."""
    return (
        f"You are a self-improving AI agent. Improve the solution below.\n\n"
        f"TASK:\n{task_prompt}\n\n"
        f"CURRENT SOLUTION:\n{current_solution}\n\n"
        f"FEEDBACK:\n{rubric_feedback}\n\n"
        f"Output your improved solution as a JSON object with a 'solution' key.\n"
    )


def extract_reward_channels(step_result) -> Dict[str, float]:
    """Extract individual reward channels from a GodelStepResult for TRL."""
    breakdown = step_result.reward_breakdown
    return {
        "task_score_delta": breakdown.task_score_delta,
        "format_compliance": breakdown.format_compliance,
        "step_cost": breakdown.step_cost,
        "anti_hack_penalty": breakdown.anti_hack_penalty,
        "process_reward": breakdown.process_reward,
        "total_reward": breakdown.total,
        "env_score": step_result.observation.total_score,
    }


# ---------------------------------------------------------------------------
# Reward functions for GRPOTrainer(reward_funcs=[...])
# ---------------------------------------------------------------------------

def task_reward_func(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    """Primary reward: task score delta from the environment."""
    return kwargs.get("task_score_delta", [0.0] * len(completions))


def format_reward_func(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    """Reward for format compliance."""
    return kwargs.get("format_compliance", [0.0] * len(completions))


def guard_reward_func(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    """Penalty from anti-reward-hacking guards."""
    return kwargs.get("anti_hack_penalty", [0.0] * len(completions))


def score_reward_func(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    """Reward based on absolute environment score (incentivizes high scores)."""
    scores = kwargs.get("env_score", [0.0] * len(completions))
    # Scale to [-0.5, 0.5] range centered at 0.5
    return [s - 0.5 for s in scores]


ALL_REWARD_FUNCS = [
    task_reward_func,
    format_reward_func,
    guard_reward_func,
    score_reward_func,
]
