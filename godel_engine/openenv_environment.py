"""
OpenEnv-core `Environment` wrapper for the existing async `GodelEnvironment`.

OpenEnv-core servers are Gym-style and (by default) synchronous. The base class
provides async wrappers, but the canonical server wrapper (`HTTPEnvServer`)
expects sync `reset()` / `step()` methods.

We therefore:
- call the existing async environment via `run_async()`
- return an OpenEnv `Observation` subclass that contains done/reward fields
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment

from godel_engine.async_utils import run_async
from godel_engine.environment import GodelEnvironment
from godel_engine.models import EditType, GodelAction
from godel_engine.openenv_models import (
    GodelOpenEnvAction,
    GodelOpenEnvObservation,
    GodelOpenEnvState,
)


class GodelOpenEnvEnvironment(Environment[GodelOpenEnvAction, GodelOpenEnvObservation, GodelOpenEnvState]):
    """OpenEnv-core Environment that delegates to `godel_engine.environment.GodelEnvironment`."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, seed: int | None = None):
        super().__init__()
        self._env = GodelEnvironment(seed=seed)
        self._state = GodelOpenEnvState()

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> GodelOpenEnvObservation:
        # OpenEnv passes seed/episode_id; our env supports seed and generates its own episode_id.
        task_type = kwargs.get("task_type")
        difficulty = kwargs.get("difficulty")
        task_id = kwargs.get("task_id")

        result = run_async(
            self._env.reset(
                task_type=task_type,
                difficulty=difficulty,
                task_id=task_id,
                seed=seed,
            )
        )
        obs = result.observation
        self._state = GodelOpenEnvState(
            episode_id=obs.episode_id,
            step_count=obs.step,
            current_score=obs.total_score,
            best_score=obs.total_score,
            initial_score=obs.total_score,
            total_cost=0.0,
            cumulative_reward=0.0,
            improvement_trajectory=list(obs.improvement_history),
        )

        return GodelOpenEnvObservation(
            done=False,
            reward=result.reward,
            metadata={"info": result.info},
            episode_id=obs.episode_id,
            task_id=obs.task_id,
            task_type=obs.task_type,
            difficulty=obs.difficulty,
            task_prompt=obs.task_prompt,
            current_solution=obs.current_solution,
            total_score=obs.total_score,
            rubric_scores=obs.rubric_scores.model_dump(mode="json"),
            step=obs.step,
            max_steps=obs.max_steps,
            improvement_history=list(obs.improvement_history),
            feedback_summary=obs.feedback_summary,
            grading_source=result.info.get("grading_source", "deterministic"),
            grading_error=result.info.get("grading_error"),
            current_strategy=obs.current_strategy,
            strategy_id=obs.strategy_id,
            strategy_generation=obs.strategy_generation,
            strategy_elo=obs.strategy_elo,
            recent_failures=list(obs.recent_failures),
            downstream_scores=dict(obs.downstream_scores),
            patch_history=list(obs.patch_history),
            budget_remaining=obs.budget_remaining,
            reward_breakdown=result.reward_breakdown.model_dump(mode="json"),
            patch_decision=None,
        )

    def step(self, action: GodelOpenEnvAction, timeout_s: float | None = None, **kwargs: Any) -> GodelOpenEnvObservation:
        edit_type_raw = (action.edit_type or "rewrite").strip().upper()
        edit_type = (
            EditType[edit_type_raw]
            if edit_type_raw in EditType.__members__
            else EditType.REWRITE
        )
        
        # Handle strategy_patch as either StrategyPatch object or dict
        strategy_patch = action.strategy_patch
        if isinstance(strategy_patch, dict):
            from godel_engine.models import StrategyPatch
            strategy_patch = StrategyPatch(**strategy_patch)
        
        internal_action = GodelAction(
            solution=action.solution,
            edit_type=edit_type,
            strategy_note=action.strategy_note or "",
            strategy_patch=strategy_patch,
        )

        result = run_async(self._env.step(internal_action))
        obs = result.observation
        state = self._env.state()
        self._state = GodelOpenEnvState(
            episode_id=state.episode_id,
            step_count=state.step_count,
            current_score=state.current_score,
            best_score=state.best_score,
            initial_score=state.initial_score,
            total_cost=state.total_cost,
            cumulative_reward=state.cumulative_reward,
            improvement_trajectory=list(state.improvement_trajectory),
            patches_proposed=state.patches_proposed,
            patches_accepted=state.patches_accepted,
            patches_rejected=state.patches_rejected,
            strategy_lineage=list(state.strategy_lineage),
            current_strategy_elo=state.current_strategy_elo,
        )

        done = bool(result.terminated or result.truncated)
        return GodelOpenEnvObservation(
            done=done,
            reward=result.reward,
            metadata={
                "info": result.info,
                "reward_breakdown": result.reward_breakdown.model_dump(mode="json"),
                "patch_decision": result.patch_decision.model_dump(mode="json")
                if result.patch_decision
                else None,
                "terminated": result.terminated,
                "truncated": result.truncated,
            },
            episode_id=obs.episode_id,
            task_id=obs.task_id,
            task_type=obs.task_type,
            difficulty=obs.difficulty,
            task_prompt=obs.task_prompt,
            current_solution=obs.current_solution,
            total_score=obs.total_score,
            rubric_scores=obs.rubric_scores.model_dump(mode="json"),
            step=obs.step,
            max_steps=obs.max_steps,
            improvement_history=list(obs.improvement_history),
            feedback_summary=obs.feedback_summary,
            grading_source=result.info.get("grading_source", "deterministic"),
            grading_error=result.info.get("grading_error"),
            current_strategy=obs.current_strategy,
            strategy_id=obs.strategy_id,
            strategy_generation=obs.strategy_generation,
            strategy_elo=obs.strategy_elo,
            recent_failures=list(obs.recent_failures),
            downstream_scores=dict(obs.downstream_scores),
            patch_history=list(obs.patch_history),
            budget_remaining=obs.budget_remaining,
            reward_breakdown=result.reward_breakdown.model_dump(mode="json"),
            patch_decision=result.patch_decision.model_dump(mode="json")
            if result.patch_decision
            else None,
        )

    @property
    def state(self) -> GodelOpenEnvState:
        return self._state

