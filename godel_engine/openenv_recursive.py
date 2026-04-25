"""
OpenEnv wrapper for the RecursiveSelfImprovementEnv.

This wraps the recursive self-improvement environment to be compatible with
the OpenEnv HTTP/WebSocket protocol while maintaining the core recursive
self-improvement semantics.
"""
from __future__ import annotations

from typing import Any, Optional

from openenv.core.base.environment import Environment

from godel_engine.async_utils import run_async
from godel_engine.models import (
    EditType,
    GodelAction,
    GodelObservation,
    GodelState,
    StrategyPatch,
)
from godel_engine.recursive_environment import RecursiveSelfImprovementEnv


class RecursiveOpenEnvEnvironment(Environment):
    """
    OpenEnv-compliant wrapper for RecursiveSelfImprovementEnv.

    This is the production entry point for the recursive self-improvement
    environment. All episodes are strategy improvement episodes.
    """

    def __init__(self) -> None:
        self._env: Optional[RecursiveSelfImprovementEnv] = None

    @property
    def env(self) -> RecursiveSelfImprovementEnv:
        if self._env is None:
            self._env = RecursiveSelfImprovementEnv()
        return self._env

    def reset(
        self,
        seed: Optional[int] = None,
        strategy_id: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Reset the environment and start a new recursive self-improvement episode."""
        result = run_async(self.env.reset(seed=seed, strategy_id=strategy_id))
        return self._step_result_to_dict(result)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Process a StrategyPatch proposal.

        Expected action format:
        {
            "strategy_patch": {
                "improved_strategy": "full text of improved strategy",
                "diff_description": "what changed and why",
                "hypothesis": "why this should improve performance",
                "target_weaknesses": ["weakness1", "weakness2"]
            }
        }

        Legacy format (solution only) is still supported but will result in
        a null patch proposal.
        """
        godel_action = self._parse_action(action)
        result = run_async(self.env.step(godel_action))
        return self._step_result_to_dict(result)

    def state(self) -> dict[str, Any]:
        """Return the current episode state."""
        state = self.env.state()
        return state.model_dump()

    def _parse_action(self, action: dict[str, Any]) -> GodelAction:
        """Parse an action dictionary into a GodelAction."""
        patch = None
        if "strategy_patch" in action and action["strategy_patch"]:
            patch_data = action["strategy_patch"]
            patch = StrategyPatch(
                improved_strategy=str(patch_data.get("improved_strategy", "")),
                diff_description=str(patch_data.get("diff_description", "")),
                hypothesis=str(patch_data.get("hypothesis", "")),
                target_weaknesses=list(patch_data.get("target_weaknesses", [])),
            )

        edit_type = EditType.REWRITE
        if "edit_type" in action:
            try:
                edit_type = EditType[str(action["edit_type"]).upper()]
            except KeyError:
                pass

        return GodelAction(
            solution=str(action.get("solution", "")),
            edit_type=edit_type,
            strategy_note=str(action.get("strategy_note", "")),
            strategy_patch=patch,
        )

    def _step_result_to_dict(self, result) -> dict[str, Any]:
        """Convert a GodelStepResult to an OpenEnv-compatible dict."""
        obs = result.observation
        patch_decision = None
        if result.patch_decision:
            patch_decision = result.patch_decision.model_dump()

        return {
            "observation": obs.model_dump(),
            "reward": result.reward,
            "reward_breakdown": result.reward_breakdown.model_dump(),
            "done": result.terminated or result.truncated,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "info": result.info,
            "patch_decision": patch_decision,
        }


# OpenEnv-compatible action and observation classes
class RecursiveAction:
    """Action schema for the recursive self-improvement environment."""

    def __init__(
        self,
        strategy_patch: Optional[dict[str, Any]] = None,
        solution: str = "",
        edit_type: str = "rewrite",
        strategy_note: str = "",
    ):
        self.strategy_patch = strategy_patch
        self.solution = solution
        self.edit_type = edit_type
        self.strategy_note = strategy_note

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_patch": self.strategy_patch,
            "solution": self.solution,
            "edit_type": self.edit_type,
            "strategy_note": self.strategy_note,
        }


class RecursiveObservation:
    """Observation schema for the recursive self-improvement environment."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecursiveObservation":
        obs = cls()
        for key, value in data.items():
            setattr(obs, key, value)
        return obs
