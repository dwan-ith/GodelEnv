"""
Godel Env -- Core OpenEnv-compatible Environment.
Lives in the godel_engine package so it can be used standalone
without a web server. The FastAPI server imports this as a library.
"""
from __future__ import annotations
import random
import uuid
import logging
from typing import Any, Optional, Dict

from godel_engine.models import (
    GodelAction,
    GodelObservation,
    GodelStepResult,
    RewardBreakdown,
    RubricScores,
    GodelState,
)
from godel_engine.evolution import DarwinPool, HuxleyTracker, Strategy
from godel_engine.guards import run_all_guards
from godel_engine.curriculum import CurriculumController, DIFFICULTY_LADDER
from godel_engine.tasks.alignment_qa import AlignmentQATask
from godel_engine.tasks.python_optimized import PythonOptimizedTask
from godel_engine.tasks.adr_writing import ADRWritingTask
from godel_engine.tasks.strategy_optimization import StrategyOptimizationTask
from godel_engine.tasks.factual_qa import FactualQATask
from godel_engine.tasks.code_improvement import CodeImprovementTask
from godel_engine.tasks.reasoning import ReasoningTask

logger = logging.getLogger("godel_env.environment")


class GodelEnvironment:
    """
    OpenEnv-compatible environment for Godel Env.
    Implements reset() / step() / state() and can be used directly
    in Python without any web server.

    Features:
      - 7 task types across 4 difficulty levels
      - Multi-channel reward breakdown (task, format, guards, process)
      - Anti-reward-hacking guards (length, repetition, forbidden patterns)
      - Automatic curriculum learning (escalation / de-escalation)
      - Explicit termination reasons

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

    # Map difficulty labels to task names
    TASKS_BY_DIFFICULTY = {
        "easy": ["factual_qa", "alignment_qa"],
        "medium": ["code_improvement", "python_optimized", "reasoning"],
        "hard": ["adr_writing"],
        "godel": ["strategy_optimization"],
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        # Instantiate all tasks once -- expensive graders are shared
        self.tasks = {name: cls() for name, cls in self.TASKS.items()}
        self.pool = DarwinPool(rng=self.rng)
        self.huxley = HuxleyTracker()
        self.curriculum = CurriculumController()

        # Episode state -- reset on every reset()
        self.current_task = None
        self.current_instance = None
        self.current_solution: str = ""
        self.current_score: float = 0.0
        self.episode_id: str = ""
        self.step_count: int = 0
        self.max_steps: int = 10
        self.improvement_history: list[float] = []
        self.initial_score: float = 0.0
        self.initial_solution: str = ""
        self.current_strategy = None
        self.episode_difficulty: str = ""

    # -- Public OpenEnv API ---------------------------------------------------

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

        # Determine difficulty (curriculum auto or manual override)
        if difficulty and difficulty in DIFFICULTY_LADDER:
            chosen_difficulty = difficulty
        else:
            chosen_difficulty = self.curriculum.suggest_difficulty()
        self.episode_difficulty = chosen_difficulty

        # Select task
        if task_type and task_type in self.tasks:
            self.current_task = self.tasks[task_type]
        else:
            # Pick a random task matching the chosen difficulty
            eligible = self.TASKS_BY_DIFFICULTY.get(chosen_difficulty, list(self.tasks.keys()))
            chosen_name = self.rng.choice(eligible)
            self.current_task = self.tasks[chosen_name]

        self.current_instance = self.current_task.sample(self.rng)
        self.current_solution = self.current_instance.initial_solution
        self.initial_solution = self.current_instance.initial_solution
        self.current_strategy = self.pool.select()

        # Grade initial (baseline) solution -- fallback gracefully if LLM unavailable
        try:
            score, rubrics, fb = await self.current_task.grade(
                self.current_instance, self.current_solution
            )
        except Exception:
            score = 0.0
            rubrics = {k: 0.0 for k in self.current_task._get_rubrics()}
            fb = {k: "Baseline grading deferred." for k in rubrics}

        # Ensure scores are strictly between 0 and 1 for OpenEnv validation
        score = self._clamp(score)
        rubrics = {k: self._clamp(v) for k, v in rubrics.items()}

        self.current_score = score
        self.initial_score = score

        return GodelStepResult(
            observation=self._build_obs(rubrics, fb),
            reward=0.0,
            reward_breakdown=RewardBreakdown(),
            terminated=False,
            truncated=False,
            info={
                "msg": "reset",
                "episode_id": self.episode_id,
                "difficulty": self.episode_difficulty,
                "curriculum": self.curriculum.get_stats(),
            },
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
        try:
            score, rubrics, fb = await self.current_task.grade(
                self.current_instance, self.current_solution
            )
        except Exception as e:
            logger.warning(f"Grading failed at step {self.step_count}: {e}")
            score = previous_score
            rubrics = {k: self._clamp(previous_score) for k in self.current_task._get_rubrics()}
            fb = {k: f"Grading error: {type(e).__name__}" for k in rubrics}

        # Clamp scores to open interval (0, 1)
        score = self._clamp(score)
        rubrics = {k: self._clamp(v) for k, v in rubrics.items()}
        self.current_score = score

        # -- Multi-channel reward computation ---------------------------------
        delta = score - previous_score
        self.improvement_history.append(delta)

        # Run anti-reward-hacking guards
        guard_result = run_all_guards(
            solution=action.solution,
            initial_solution=self.initial_solution,
            task_type=self.current_task.name,
            current_score=score,
            previous_score=previous_score,
        )

        # Format compliance check
        format_score = self._check_format_compliance(action, self.current_task.name)

        # Build reward breakdown
        breakdown = RewardBreakdown(
            task_score_delta=delta,
            format_compliance=format_score,
            length_penalty=0.0,  # Captured inside guard penalty
            step_cost=-0.005,
            anti_hack_penalty=guard_result.penalty,
            process_reward=0.0,  # Populated by process supervision if enabled
        )
        breakdown.compute_total()
        reward = breakdown.total

        # -- Termination logic ------------------------------------------------
        plateau = (
            self.step_count >= 3
            and all(abs(d) < 0.005 for d in self.improvement_history[-3:])
        )
        success = score >= 0.95
        guard_halt = guard_result.penalty <= -0.8  # Severe violation

        terminated = plateau or success or guard_halt
        truncated = self.step_count >= self.max_steps

        # Determine reason
        if guard_halt:
            reason = f"guard_violation: {'; '.join(guard_result.violations)}"
        elif success:
            reason = "success"
        elif plateau:
            reason = "plateau"
        elif truncated:
            reason = "max_steps"
        else:
            reason = "running"

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

            # Record outcome for curriculum
            self.curriculum.record_outcome(self.episode_difficulty, score)

        return GodelStepResult(
            observation=self._build_obs(rubrics, fb),
            reward=reward,
            reward_breakdown=breakdown,
            terminated=terminated,
            truncated=truncated,
            info={
                "delta": delta,
                "step": self.step_count,
                "action_type": action.edit_type,
                "strategy_note": action.strategy_note,
                "reason": reason,
                "guard_violations": guard_result.violations,
                "guard_penalty": guard_result.penalty,
                "difficulty": self.episode_difficulty,
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

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        """Clamp a score to the open interval (0, 1) for OpenEnv validation."""
        return max(lo, min(hi, float(value)))

    def _check_format_compliance(self, action: GodelAction, task_type: str) -> float:
        """
        Check whether the agent's output follows expected formatting.
        Returns a small bonus [0.0, 0.02] for compliance, 0 otherwise.
        """
        solution = action.solution

        if task_type in ("code_improvement", "python_optimized"):
            # Code tasks: solution should contain a function definition
            if "def " in solution:
                return 0.02
            return 0.0

        if task_type == "adr_writing":
            # ADR tasks: should have section headers
            headers = ["# ", "## ", "Context", "Decision", "Consequences"]
            matches = sum(1 for h in headers if h.lower() in solution.lower())
            return 0.02 if matches >= 2 else 0.0

        if task_type == "strategy_optimization":
            # Strategy tasks: should have numbered steps
            if any(f"{i}." in solution for i in range(1, 6)):
                return 0.02
            return 0.0

        # Default: any non-empty, reasonably structured answer
        if len(solution.strip()) > 50:
            return 0.01
        return 0.0

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
