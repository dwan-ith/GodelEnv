"""
OpenEnv-core compatible models for the Godel environment server.

The existing `godel_engine.models` are used internally by the package and by the
dashboard/server routers. OpenEnv-core's `HTTPEnvServer` expects Action /
Observation / State subclasses from `openenv.core.env_server.types`.

We keep these models separate to avoid breaking internal callers while still
exposing a standards-compliant OpenEnv server.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from godel_engine.models import StrategyPatch


class GodelOpenEnvAction(Action):
    solution: str = Field(..., description="Full replacement solution submitted by the agent.")
    edit_type: str = Field(default="rewrite")
    strategy_note: str = Field(default="")
    strategy_patch: StrategyPatch | None = Field(
        default=None,
        description="Optional recursive self-improvement patch to evaluate on held-out tasks.",
    )


class GodelOpenEnvObservation(Observation):
    # Observation base provides: done: bool, reward: float|int|bool|None, metadata: dict
    episode_id: str = ""
    task_id: str = ""
    task_type: str = ""
    difficulty: str = ""
    task_prompt: str = ""
    current_solution: str = ""
    total_score: float = 0.0
    rubric_scores: dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    max_steps: int = 10
    improvement_history: list[float] = Field(default_factory=list)
    feedback_summary: str = ""
    grading_source: str = ""
    grading_error: str | None = None
    current_strategy: str = ""
    strategy_id: str = ""
    strategy_generation: int = 0
    strategy_elo: float = 1000.0
    recent_failures: list[str] = Field(default_factory=list)
    downstream_scores: dict[str, float] = Field(default_factory=dict)
    patch_history: list[dict[str, Any]] = Field(default_factory=list)
    budget_remaining: int = 0
    reward_breakdown: dict[str, Any] = Field(default_factory=dict)
    patch_decision: dict[str, Any] | None = None


class GodelOpenEnvState(State):
    episode_id: str | None = None
    step_count: int = 0
    current_score: float = 0.0
    best_score: float = 0.0
    initial_score: float = 0.0
    total_cost: float = 0.0
    cumulative_reward: float = 0.0
    improvement_trajectory: list[float] = Field(default_factory=list)
    patches_proposed: int = 0
    patches_accepted: int = 0
    patches_rejected: int = 0
    strategy_lineage: list[str] = Field(default_factory=list)
    current_strategy_elo: float = 1000.0

