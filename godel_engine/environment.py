"""
Gödel Env — Core OpenEnv-compatible Environment.
Lives in the godel_engine package so it can be used standalone
without a web server. The FastAPI server imports this as a library.
"""
from __future__ import annotations
import random
import uuid
from typing import Any, Optional, Dict

from godel_engine.models import (
    GodelAction,
    GodelObservation,
    GodelStepResult,
    RubricScores,
    GodelState,
)
from godel_engine.evolution import DarwinPool, HuxleyTracker, Strategy
from godel_engine.tasks.alignment_qa import AlignmentQATask
from godel_engine.tasks.python_optimized import PythonOptimizedTask
from godel_engine.tasks.adr_writing import ADRWritingTask
from godel_engine.tasks.strategy_optimization import StrategyOptimizationTask
from godel_engine.tasks.factual_qa import FactualQATask
from godel_engine.tasks.code_improvement import CodeImprovementTask
from godel_engine.tasks.reasoning import ReasoningTask


class GodelEnvironment:
    """
    OpenEnv-compatible environment for Gödel Eng.
    Implements reset() / step() / state() and can be used directly
    in Python without any web server.

    Usage:
        env = GodelEnvironment()
        result = await env.reset(task_type="factual_qa")
        result = await env.step(GodelAction(solution="..."))
        state = env.state()
    """

    TASKS = {
        "factual_qa": FactualQATask,
        "code_improvement": CodeImprovementTask,
        "reasoning": ReasoningTask,
        "alignment_qa": AlignmentQATask,
        "python_optimized": PythonOptimizedTask,
        "adr_writing": ADRWritingTask,
        "strategy_optimization": StrategyOptimizationTask,
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        # Instantiate all tasks once — expensive graders are shared
        self.tasks = {name: cls() for name, cls in self.TASKS.items()}
        self.pool = DarwinPool(rng=self.rng)
        self.huxley = HuxleyTracker()

        # Episode state — reset on every reset()
        self.current_task = None
        self.current_instance = None
        self.current_solution: str = ""
        self.current_score: float = 0.0
        self.episode_id: str = ""
        self.step_count: int = 0
        self.max_steps: int = 10
        self.improvement_history: list[float] = []
        self.initial_score: float = 0.0
        self.current_strategy = None

    # ── Public OpenEnv API ──────────────────────────────────────────────

    async def reset(
        self,
        task_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GodelStepResult:
        """Initialize a new episode and return the first observation."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.improvement_history = []

        # Select task
        if task_type and task_type in self.tasks:
            self.current_task = self.tasks[task_type]
        else:
            self.current_task = self.tasks[self.rng.choice(list(self.tasks.keys()))]

        self.current_instance = self.current_task.sample(self.rng)
        self.current_solution = self.current_instance.initial_solution
        self.current_strategy = self.pool.select()

        # Grade initial (baseline) solution — fallback gracefully if LLM unavailable
        try:
            score, rubrics, fb = await self.current_task.grade(
                self.current_instance, self.current_solution
            )
        except Exception:
            score = 0.0
            rubrics = {k: 0.0 for k in self.current_task._get_rubrics()}
            fb = {k: "Baseline grading deferred." for k in rubrics}
            
        # Ensure scores are strictly between 0 and 1 for OpenEnv validation
        score = max(0.001, min(0.999, float(score)))
        rubrics = {k: max(0.001, min(0.999, float(v))) for k, v in rubrics.items()}
        
        self.current_score = score
        self.initial_score = score

        return GodelStepResult(
            observation=self._build_obs(rubrics, fb),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"msg": "reset", "episode_id": self.episode_id},
        )

    async def step(self, action: GodelAction) -> GodelStepResult:
        """Submit an agent action and receive the graded result."""
        if self.current_task is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() before step()."
            )

        self.step_count += 1
        previous_score = self.current_score
        self.current_solution = action.solution

        # Grade new solution
        score, rubrics, fb = await self.current_task.grade(
            self.current_instance, self.current_solution
        )
        
        # Ensure scores are strictly between 0 and 1 for OpenEnv validation
        score = max(0.001, min(0.999, float(score)))
        rubrics = {k: max(0.001, min(0.999, float(v))) for k, v in rubrics.items()}
        
        self.current_score = score

        # Partial-progress reward: raw score delta minus a small step cost
        delta = score - previous_score
        self.improvement_history.append(delta)
        reward = delta - 0.005  # Step cost encourages efficiency

        # Termination logic
        plateau = (
            self.step_count >= 3
            and all(abs(d) < 0.005 for d in self.improvement_history[-3:])
        )
        terminated = plateau
        truncated = self.step_count >= self.max_steps

        # Strategy evolution bookkeeping
        if terminated or truncated:
            self.current_strategy.record_performance(score)
            if score > self.initial_score + 0.05:
                child = Strategy(
                    id=f"strat_{uuid.uuid4().hex[:6]}",
                    parent_id=self.current_strategy.id,
                    generation=self.current_strategy.generation + 1,
                )
                self.pool.add_strategy(child)
                self.huxley.record_lineage(self.current_strategy.id, child.id)
            self.huxley.compute_cmp(self.pool)

        return GodelStepResult(
            observation=self._build_obs(rubrics, fb),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "delta": delta,
                "step": self.step_count,
                "action_type": action.edit_type,
                "strategy_note": action.strategy_note,
            },
        )

    def state(self) -> GodelState:
        """Return episode-level metadata without advancing the episode."""
        score_trajectory = [self.initial_score] + [
            self.initial_score + sum(self.improvement_history[: i + 1])
            for i in range(len(self.improvement_history))
        ]
        return GodelState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            current_score=self.current_score,
            best_score=max(score_trajectory) if score_trajectory else self.initial_score,
            initial_score=self.initial_score,
            total_cost=self.step_count * 0.005,
            cumulative_reward=sum(self.improvement_history) - self.step_count * 0.005,
            improvement_trajectory=self.improvement_history,
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _build_obs(self, rubrics: Dict, feedback: Dict) -> GodelObservation:
        weights = {k: 1.0 / len(rubrics) for k in rubrics} if rubrics else {}
        return GodelObservation(
            episode_id=self.episode_id,
            task_id=self.current_instance.task_id,
            task_type=self.current_task.name,
            difficulty=self.current_task.difficulty,
            task_prompt=self.current_instance.prompt,
            current_solution=self.current_solution,
            total_score=self.current_score,
            rubric_scores=RubricScores(
                scores=rubrics,
                weights=weights,
                feedback=feedback,
            ),
            step=self.step_count,
            max_steps=self.max_steps,
            improvement_history=list(self.improvement_history),
            feedback_summary=" | ".join(str(v) for v in feedback.values()),
        )
