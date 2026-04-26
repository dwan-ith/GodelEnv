"""
Core Gödel Env Environment.

The fundamental loop is now recursive self-improvement:
  1. Agent observes current strategy + downstream performance + failures + budget
  2. Agent proposes a StrategyPatch (mutation to its reasoning policy)
  3. Environment evaluates parent vs child on held-out task bundles
  4. Governor accepts/rejects based on multi-objective utility
  5. Registry records lineage, Elo updates, accepted/rejected patches

The existing task families (factual_qa, code_improvement, reasoning, etc.)
become the **evaluation substrate** — they are the downstream problems used
to judge whether a strategy mutation actually improved capability.

Legacy mode: if the agent submits a plain GodelAction without a strategy_patch,
the environment falls back to direct answer-improvement mode for backward
compatibility with existing training infrastructure.
"""
from __future__ import annotations

import logging
import random
import uuid
from typing import Dict, List, Optional

from godel_engine.challenge_pool import ChallengePool
from godel_engine.curriculum import CurriculumController, DIFFICULTY_LADDER
from godel_engine.evolution import (
    DEFAULT_STRATEGY_TEXT,
    Governor,
    GovernorConfig,
    HuxleyTracker,
    Strategy,
    StrategyRegistry,
)
from godel_engine.guards import run_all_guards, run_strategy_guards
from godel_engine.models import (
    EditType,
    GodelAction,
    GodelObservation,
    GodelState,
    GodelStepResult,
    PatchDecision,
    RewardBreakdown,
    RubricScores,
    StrategyPatch,
)
from godel_engine.strategy_evaluator import StrategyEvaluator
from godel_engine.tasks.adr_writing import ADRWritingTask
from godel_engine.tasks.alignment_qa import AlignmentQATask
from godel_engine.tasks.code_improvement import CodeImprovementTask
from godel_engine.tasks.factual_qa import FactualQATask
from godel_engine.tasks.python_optimized import PythonOptimizedTask
from godel_engine.tasks.reasoning import ReasoningTask
from godel_engine.tasks.strategy_optimization import StrategyOptimizationTask


logger = logging.getLogger("godel_env.environment")


class GodelEnvironment:
    """
    OpenEnv-style environment for recursive strategy self-improvement.

    The agent's job is to propose StrategyPatch mutations that improve
    downstream performance as measured by the held-out evaluator and
    accepted by the Governor.
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

    TASKS_BY_DIFFICULTY = {
        "easy": ["factual_qa", "alignment_qa"],
        "medium": ["code_improvement", "python_optimized", "reasoning"],
        "hard": ["adr_writing"],
        "godel": ["strategy_optimization"],
    }

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.tasks = {name: cls() for name, cls in self.TASKS.items()}
        self.registry = StrategyRegistry(rng=self.rng)
        self.huxley = HuxleyTracker()
        self.governor = Governor()
        self.curriculum = CurriculumController()
        self.strategy_evaluator = StrategyEvaluator(seed=seed or 42)
        self.challenge_pool = ChallengePool()

        # Legacy aliases for backward compat
        self.pool = self.registry

        # Episode state
        self.current_task = None
        self.current_instance = None
        self.current_solution = ""
        self.current_score = 0.0
        self.initial_score = 0.0
        self.initial_solution = ""
        self.episode_id = ""
        self.step_count = 0
        self.max_steps = 10
        self.improvement_history: list[float] = []
        self.reward_history: list[float] = []
        self.current_strategy: Optional[Strategy] = None
        self.episode_difficulty = ""

        # GodelEnv state
        self.patches_proposed = 0
        self.patches_accepted = 0
        self.patches_rejected = 0
        self.patch_history: list[dict] = []

    async def reset(
        self,
        task_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GodelStepResult:
        """Initialize a new episode and return the first observation."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.improvement_history = []
        self.reward_history = []
        self.patches_proposed = 0
        self.patches_accepted = 0
        self.patches_rejected = 0
        self.patch_history = []

        chosen_task = self.tasks.get(task_type) if task_type else None
        if chosen_task is not None:
            self.current_task = chosen_task
            chosen_difficulty = (
                difficulty if difficulty in DIFFICULTY_LADDER else chosen_task.difficulty
            )
        else:
            chosen_difficulty = (
                difficulty
                if difficulty in DIFFICULTY_LADDER
                else self.curriculum.suggest_difficulty()
            )
            eligible = self.TASKS_BY_DIFFICULTY.get(
                chosen_difficulty, list(self.tasks.keys())
            )
            chosen_name = self.rng.choice(eligible)
            self.current_task = self.tasks[chosen_name]

        self.episode_difficulty = chosen_difficulty
        self.current_instance = self.current_task.sample(self.rng, task_id=task_id)
        self.current_solution = self.current_instance.initial_solution
        self.initial_solution = self.current_instance.initial_solution

        # Select a strategy from the registry
        self.current_strategy = self.registry.select()

        try:
            score, rubrics, feedback = await self.current_task.grade(
                self.current_instance, self.current_solution
            )
        except Exception as exc:
            logger.exception("Initial grading failed for %s", self.current_task.name)
            self.current_task.last_grading_source = "deterministic"
            self.current_task.last_grading_error = f"initial grading exception: {type(exc).__name__}"
            score = 0.0
            rubrics = {name: 0.0 for name in self.current_task._get_rubrics()}
            feedback = {name: f"Initial grading error: {type(exc).__name__}" for name in rubrics}

        self.current_score = self._clamp(score)
        self.initial_score = self.current_score
        rubrics = {name: self._clamp(value) for name, value in rubrics.items()}

        return GodelStepResult(
            observation=self._build_obs(rubrics, feedback),
            reward=0.0,
            reward_breakdown=RewardBreakdown(),
            terminated=False,
            truncated=False,
            info={
                "msg": "reset",
                "episode_id": self.episode_id,
                "task_type": self.current_task.name,
                "task_id": self.current_instance.task_id,
                "difficulty": self.episode_difficulty,
                "grading_source": getattr(
                    self.current_task, "last_grading_source", "deterministic"
                ),
                "grading_error": getattr(self.current_task, "last_grading_error", None),
                "curriculum": self.curriculum.get_stats(),
                "registry": self.registry.get_stats(),
            },
        )

    async def step(self, action: GodelAction) -> GodelStepResult:
        """
        Submit an action and receive a graded result.

        GodelEnv dual-mode:
        - If action.strategy_patch is provided, this is a strategy-level step:
          the environment evaluates parent vs child on downstream tasks.
        - Otherwise, this is a legacy answer-improvement step.
        """
        if self.current_task is None or self.current_instance is None:
            raise RuntimeError("Call reset() before step().")

        self.step_count += 1

        # ── Strategy-level step (GodelEnv primary mode) ──
        if action.strategy_patch is not None:
            return await self._strategy_step(action)

        # ── Legacy answer-improvement step ──
        return await self._answer_step(action)

    async def _strategy_step(self, action: GodelAction) -> GodelStepResult:
        """
        Process a strategy patch proposal.

        1. Create a child strategy from the patch
        2. Evaluate both parent and child on downstream tasks
        3. Run strategy-level anti-hacking guards
        4. Governor decides accept/reject
        5. Update registry, Elo, lineage
        """
        patch = action.strategy_patch
        self.patches_proposed += 1

        # Create child strategy
        child = Strategy(
            id=f"strat_{uuid.uuid4().hex[:6]}",
            policy_text=patch.improved_strategy,
            parent_id=self.current_strategy.id,
            generation=self.current_strategy.generation + 1,
            fitness=self.current_strategy.fitness,
            history=self.current_strategy.history.copy(),
            patch_description=patch.diff_description,
            patch_hypothesis=patch.hypothesis,
        )

        # Evaluate parent and child on held-out tasks
        parent_scores, parent_per_task, parent_diagnostics = await self._evaluate_strategy_downstream(
            self.current_strategy
        )
        child_scores, child_per_task, child_diagnostics = await self._evaluate_strategy_downstream(child)

        # Run strategy-level guards
        guard_result = run_strategy_guards(
            patch.improved_strategy,
            parent_per_task,
            child_per_task,
        )

        # Governor decision
        regression_count = sum(
            1
            for task in set(parent_per_task) | set(child_per_task)
            if child_per_task.get(task, 0.0) < parent_per_task.get(task, 0.0) - 0.01
        )
        if guard_result.passed:
            decision = self.governor.decide(
                parent_scores, child_scores,
                parent_per_task, child_per_task,
            )
        else:
            decision = {
                "accepted": False,
                "parent_utility": self.governor.compute_utility(parent_scores),
                "child_utility": self.governor.compute_utility(child_scores),
                "improvement": 0.0,
                "rejection_reasons": guard_result.violations,
                "regression_count": regression_count,
                "tasks_evaluated": len(child_per_task),
            }

        patch_decision = PatchDecision(
            accepted=decision["accepted"],
            parent_utility=decision["parent_utility"],
            child_utility=decision["child_utility"],
            improvement=decision.get("improvement", 0.0),
            axis_scores=child_scores,
            rejection_reasons=decision.get("rejection_reasons", []),
            tasks_evaluated=decision.get("tasks_evaluated", 0),
            regression_count=decision.get("regression_count", 0),
            diagnostics={
                "parent_source_counts": parent_diagnostics.get("source_counts", {}),
                "child_source_counts": child_diagnostics.get("source_counts", {}),
                "providers": child_diagnostics.get("providers", []),
                "last_error": child_diagnostics.get("last_error") or parent_diagnostics.get("last_error"),
            },
        )

        # Apply decision
        if decision["accepted"]:
            self.patches_accepted += 1
            self.registry.add_strategy(child)
            self.huxley.record_lineage(self.current_strategy.id, child.id)
            self.registry.update_elo(child.id, self.current_strategy.id)
            self.current_strategy = child
            patch_reward = 0.1 + decision.get("improvement", 0.0) * 0.5
        else:
            self.patches_rejected += 1
            self.registry.record_rejected_patch(self.current_strategy.id, decision)
            self.registry.update_elo(self.current_strategy.id, child.id)
            patch_reward = -0.05

        self.curriculum.record_meta_patch_outcome(bool(decision["accepted"]))
        self.registry.compute_cmp()

        # Record patch in history
        self.patch_history.append({
            "step": self.step_count,
            "accepted": decision["accepted"],
            "improvement": decision.get("improvement", 0.0),
            "reasons": decision.get("rejection_reasons", []),
        })

        # Compute reward
        child_utility = decision.get("child_utility", 0.0)
        parent_utility = decision.get("parent_utility", 0.0)
        delta = child_utility - parent_utility
        self.improvement_history.append(delta)

        breakdown = RewardBreakdown(
            task_score_delta=delta,
            format_compliance=0.01 if len(patch.improved_strategy) > 50 else 0.0,
            step_cost=-0.005,
            anti_hack_penalty=guard_result.penalty,
            patch_quality=patch_reward,
            generalization_score=child_scores.get("generalization", 0.0) * 0.1,
            robustness_score=child_scores.get("robustness", 0.0) * 0.05,
            stability_score=child_scores.get("stability", 0.0) * 0.05,
        )
        reward = breakdown.compute_total()
        self.reward_history.append(reward)
        self.current_score = self._clamp(child_utility)

        # Termination logic
        plateau = (
            self.step_count >= 3
            and all(abs(d) < 0.005 for d in self.improvement_history[-3:])
        )
        budget_exhausted = self.step_count >= self.max_steps
        terminated = plateau
        truncated = budget_exhausted

        if terminated or truncated:
            self.current_strategy.record_performance(self.current_score)
            self.curriculum.record_outcome(self.episode_difficulty, self.current_score)

        reason = "plateau" if plateau else ("max_steps" if truncated else "running")

        # Build observation with strategy context
        rubrics = child_per_task
        feedback = {
            task: f"Score: {score:.3f}" for task, score in child_per_task.items()
        }

        self._ingest_agent_challenge(action)

        return GodelStepResult(
            observation=self._build_obs(rubrics, feedback),
            reward=reward,
            reward_breakdown=breakdown,
            terminated=terminated,
            truncated=truncated,
            patch_decision=patch_decision,
            info={
                "delta": delta,
                "step": self.step_count,
                "action_type": action.edit_type,
                "strategy_note": action.strategy_note,
                "reason": reason,
                "guard_violations": guard_result.violations,
                "guard_penalty": guard_result.penalty,
                "difficulty": self.episode_difficulty,
                "task_type": self.current_task.name,
                "task_id": self.current_instance.task_id,
                "patch_accepted": decision["accepted"],
                "patch_improvement": decision.get("improvement", 0.0),
                "patches_proposed": self.patches_proposed,
                "patches_accepted": self.patches_accepted,
                "patches_rejected": self.patches_rejected,
                "strategy_id": self.current_strategy.id,
                "strategy_elo": self.current_strategy.elo,
                "strategy_generation": self.current_strategy.generation,
                "registry_stats": self.registry.get_stats(),
                "strategy_eval_mode": self.strategy_evaluator.mode,
                "strategy_eval_source_counts": child_diagnostics.get("source_counts", {}),
                "strategy_eval_last_error": child_diagnostics.get("last_error"),
                "self_play": {
                    "parent_utility": parent_utility,
                    "child_utility": child_utility,
                    "child_beats_parent_utility": child_utility > parent_utility,
                    "governor_accepted": bool(decision["accepted"]),
                },
                "challenge_pool": self.challenge_pool.as_stats(),
                "agent_challenges_mixed_in_eval": child_diagnostics.get("agent_challenges_mixed", 0),
            },
        )

    async def _answer_step(self, action: GodelAction) -> GodelStepResult:
        """Legacy answer-improvement step for backward compatibility."""
        previous_score = self.current_score
        self.current_solution = action.solution

        try:
            score, rubrics, feedback = await self.current_task.grade(
                self.current_instance, self.current_solution
            )
        except Exception as exc:
            logger.exception("Grading failed for %s", self.current_task.name)
            self.current_task.last_grading_source = "deterministic"
            self.current_task.last_grading_error = f"step grading exception: {type(exc).__name__}"
            score = previous_score
            rubrics = {
                name: previous_score for name in self.current_task._get_rubrics()
            }
            feedback = {
                name: f"Step grading error: {type(exc).__name__}"
                for name in rubrics
            }

        score = self._clamp(score)
        rubrics = {name: self._clamp(value) for name, value in rubrics.items()}
        self.current_score = score

        delta = score - previous_score
        self.improvement_history.append(delta)

        guard_result = run_all_guards(
            solution=action.solution,
            initial_solution=self.initial_solution,
            task_type=self.current_task.name,
            current_score=score,
            previous_score=previous_score,
        )
        format_score = self._check_format_compliance(action, self.current_task.name)
        process_reward = self._compute_process_reward(action, delta, format_score, guard_result.penalty)

        breakdown = RewardBreakdown(
            task_score_delta=delta,
            format_compliance=format_score,
            length_penalty=0.0,
            step_cost=-0.005,
            anti_hack_penalty=guard_result.penalty,
            process_reward=process_reward,
        )
        reward = breakdown.compute_total()
        self.reward_history.append(reward)

        plateau = (
            self.step_count >= 3
            and all(abs(change) < 0.005 for change in self.improvement_history[-3:])
        )
        success = score >= 0.95
        guard_halt = guard_result.penalty <= -0.8

        terminated = plateau or success or guard_halt
        truncated = self.step_count >= self.max_steps

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

        if terminated or truncated:
            self.current_strategy.record_performance(score, self.current_task.name)
            if score > self.initial_score + 0.05:
                child = Strategy(
                    id=f"strat_{uuid.uuid4().hex[:6]}",
                    policy_text=self.current_strategy.policy_text,
                    parent_id=self.current_strategy.id,
                    generation=self.current_strategy.generation + 1,
                    fitness=self.current_strategy.fitness,
                    history=self.current_strategy.history.copy(),
                )
                self.registry.add_strategy(child)
                self.huxley.record_lineage(self.current_strategy.id, child.id)
            self.registry.compute_cmp()
            self.curriculum.record_outcome(self.episode_difficulty, score)

        self._ingest_agent_challenge(action)

        return GodelStepResult(
            observation=self._build_obs(rubrics, feedback),
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
                "task_type": self.current_task.name,
                "task_id": self.current_instance.task_id,
                "grading_source": getattr(
                    self.current_task, "last_grading_source", "deterministic"
                ),
                "grading_error": getattr(self.current_task, "last_grading_error", None),
                "challenge_pool": self.challenge_pool.as_stats(),
            },
        )

    def _ingest_agent_challenge(self, action: GodelAction) -> None:
        """Append a validated agent-authored challenge to the pool for future held-out eval."""
        if action.agent_challenge is None:
            return
        self.challenge_pool.try_add(
            task_type=action.agent_challenge.task_type,
            prompt=action.agent_challenge.prompt,
            source_episode=self.episode_id,
        )

    async def _evaluate_strategy_downstream(
        self,
        strategy: Strategy,
    ) -> tuple[Dict[str, float], Dict[str, float], dict]:
        """
        Evaluate a strategy on held-out downstream tasks.

        Returns:
            (axis_scores, per_task_scores, diagnostics)
        """
        axis_scores, per_case_scores, diagnostics = await self.strategy_evaluator.evaluate(
            self.tasks,
            strategy.policy_text,
            episode_id=self.episode_id,
            current_task_id=self.current_instance.task_id if self.current_instance else "",
            challenge_pool=self.challenge_pool,
        )

        for task_type, score in diagnostics.get("per_family", {}).items():
            strategy.record_performance(score, task_type)
            if score < 0.45:
                strategy.record_failure(
                    f"{task_type} remains weak under strategy {strategy.id} (score {score:.3f})"
                )
        strategy.metadata["last_eval"] = diagnostics
        return axis_scores, per_case_scores, diagnostics

    def state(self) -> GodelState:
        """Return the current episode state."""
        score_trajectory = [self.initial_score]
        running = self.initial_score
        for change in self.improvement_history:
            running += change
            score_trajectory.append(running)

        return GodelState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            current_score=self.current_score,
            best_score=max(score_trajectory) if score_trajectory else self.initial_score,
            initial_score=self.initial_score,
            total_cost=self.step_count * 0.005,
            cumulative_reward=sum(self.reward_history),
            improvement_trajectory=list(self.improvement_history),
            patches_proposed=self.patches_proposed,
            patches_accepted=self.patches_accepted,
            patches_rejected=self.patches_rejected,
            strategy_lineage=self.registry.get_lineage_chain(self.current_strategy.id) if self.current_strategy else [],
            current_strategy_elo=self.current_strategy.elo if self.current_strategy else 1000.0,
        )

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _check_format_compliance(self, action: GodelAction, task_type: str) -> float:
        solution = action.solution

        if task_type in {"code_improvement", "python_optimized"}:
            return 0.02 if "def " in solution else 0.0

        if task_type == "adr_writing":
            headers = ["status", "context", "decision", "consequences", "alternatives"]
            matches = sum(1 for header in headers if header in solution.lower())
            return 0.02 if matches >= 4 else 0.01 if matches >= 2 else 0.0

        if task_type == "strategy_optimization":
            has_sections = (
                "## improved strategy" in solution.lower()
                and "## demonstration" in solution.lower()
            )
            return 0.02 if has_sections else 0.0

        return 0.01 if len(solution.strip()) > 50 else 0.0

    def _compute_process_reward(
        self,
        action: GodelAction,
        delta: float,
        format_score: float,
        guard_penalty: float,
    ) -> float:
        if guard_penalty < 0:
            return 0.0

        edit_bonus = {
            EditType.ADD_REASONING: 0.01,
            EditType.FIX_ERRORS: 0.015,
            EditType.RESTRUCTURE: 0.01,
            EditType.SYNTHESIZE: 0.015,
            EditType.REFINE: 0.005,
        }.get(action.edit_type, 0.0)

        process = max(0.0, delta) * 0.3 + format_score * 0.5 + edit_bonus
        return min(0.05, process)

    def _build_obs(self, rubrics: Dict[str, float], feedback: Dict[str, str]) -> GodelObservation:
        weights = {name: 1.0 / len(rubrics) for name in rubrics} if rubrics else {}
        feedback_summary = " | ".join(
            f"{name}: {text}" for name, text in feedback.items()
        )

        # Strategy context for GodelEnv
        strategy_text = self.current_strategy.policy_text if self.current_strategy else ""
        strategy_id = self.current_strategy.id if self.current_strategy else ""
        strategy_gen = self.current_strategy.generation if self.current_strategy else 0
        strategy_elo = self.current_strategy.elo if self.current_strategy else 1000.0
        recent_failures = self.current_strategy.failure_cases[-5:] if self.current_strategy else []
        downstream = self.current_strategy.get_downstream_summary() if self.current_strategy else {}

        return GodelObservation(
            episode_id=self.episode_id,
            task_id=self.current_instance.task_id if self.current_instance else "",
            task_type=self.current_task.name if self.current_task else "",
            difficulty=self.episode_difficulty,
            task_prompt=self.current_instance.prompt if self.current_instance else "",
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
            feedback_summary=feedback_summary,
            # GodelEnv fields
            current_strategy=strategy_text,
            strategy_id=strategy_id,
            strategy_generation=strategy_gen,
            strategy_elo=strategy_elo,
            recent_failures=recent_failures,
            downstream_scores=downstream,
            patch_history=self.patch_history[-5:],
            budget_remaining=self.max_steps - self.step_count,
            agent_challenges_queued=len(self.challenge_pool.items),
            curriculum_level=self.curriculum.current_difficulty,
        )
