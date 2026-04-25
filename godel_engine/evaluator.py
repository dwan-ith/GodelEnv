"""Compatibility wrapper for strategy-level downstream evaluation.

The canonical implementation lives in `godel_engine.strategy_evaluator`.
This module is kept only for older imports; it delegates instead of maintaining
a second evaluator that can drift from the real environment path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from godel_engine.evolution import Strategy
from godel_engine.strategy_evaluator import StrategyEvaluator


class DownstreamEvaluator:
    """Backward-compatible facade over `StrategyEvaluator`."""

    def __init__(
        self,
        env_factory,
        *,
        acceptance_tasks_per_family: int = 2,
        seed: int = 42,
    ):
        self.env_factory = env_factory
        self.acceptance_tasks_per_family = acceptance_tasks_per_family
        self.seed = seed
        self._evaluator = StrategyEvaluator(
            seed=seed,
            max_cases=max(1, acceptance_tasks_per_family * 4),
        )

    async def evaluate_strategy(
        self,
        strategy: Strategy,
        task_families: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a strategy against held-out tasks and return multi-axis scores.
        """
        env = self.env_factory()
        tasks = env.tasks
        if task_families is not None:
            tasks = {name: tasks[name] for name in task_families if name in tasks}

        axis_scores, per_case_scores, diagnostics = await self._evaluator.evaluate(
            tasks,
            strategy.policy_text,
            episode_id=getattr(env, "episode_id", "") or strategy.id,
        )
        return {
            **axis_scores,
            "per_task": diagnostics.get("per_family", {}),
            "per_case": per_case_scores,
            "total_evaluations": len(per_case_scores),
            "diagnostics": diagnostics,
        }
