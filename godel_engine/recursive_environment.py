"""
GodelEnv — Recursive Self-Improvement Environment.

This is the refactored core environment where recursive self-improvement is
the ORGANIZING PRINCIPLE, not just one task among many.

The fundamental loop:
1. Agent observes current strategy + downstream performance + recent failures
2. Agent proposes a StrategyPatch (mutation to its reasoning policy)
3. Environment evaluates parent vs child strategy on held-out task bundles
4. Governor accepts/rejects based on multi-objective utility
5. Registry tracks lineage; Elo updates; accepted patches become the new current

The "task families" (factual_qa, code_improvement, reasoning, etc.) are now
EVALUATION SUBSTRATE — they exist to judge whether a strategy mutation actually
improved capability, not as primary tasks to solve.

This design follows the practical Gödel-machine approximation:
- Self-modification is explicit (StrategyPatch)
- Improvement proposals are testable (held-out evaluation)
- Acceptance depends on objective evidence (Governor with multi-objective utility)
- Not vibes.

References:
- Schmidhuber's Gödel Machines: self-modification only when justified by evidence
- AlphaEvolve: LLM proposals + automated verifiers + evolutionary selection
- STOP: recursive improvement of executable scaffolds with measurable outcomes
"""
from __future__ import annotations

import logging
import random
import uuid
from typing import Any, Dict, List, Optional

from godel_engine.challenge_pool import ChallengePool
from godel_engine.evolution import (
    DEFAULT_STRATEGY_TEXT,
    Governor,
    GovernorConfig,
    HuxleyTracker,
    Strategy,
    StrategyRegistry,
)
from godel_engine.guards import run_strategy_guards
from godel_engine.models import (
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


logger = logging.getLogger("godel_env.recursive")


class RecursiveSelfImprovementEnv:
    """
    OpenEnv-style environment for recursive strategy self-improvement.

    Unlike the original GodelEnvironment where task types were the primary
    abstraction, here the PRIMARY action is always a StrategyPatch proposal.
    Task families are evaluation domains used to judge patch quality.

    The agent's job: propose StrategyPatch mutations that genuinely improve
    downstream performance as measured by held-out evaluation and accepted
    by the Governor.
    """

    # Evaluation domains (these are NOT user-selectable tasks — they are
    # the substrate on which strategies are evaluated)
    EVALUATION_DOMAINS = {
        "factual_qa": FactualQATask,
        "code_improvement": CodeImprovementTask,
        "reasoning": ReasoningTask,
        "alignment_qa": AlignmentQATask,
        "python_optimized": PythonOptimizedTask,
        "adr_writing": ADRWritingTask,
    }

    # Difficulty weights for curriculum (easier domains weighted more initially)
    DOMAIN_DIFFICULTY = {
        "factual_qa": 1,
        "alignment_qa": 1,
        "code_improvement": 2,
        "reasoning": 2,
        "python_optimized": 3,
        "adr_writing": 3,
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = 10,
        governor_config: Optional[GovernorConfig] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed or 42
        self.max_steps = max_steps

        # Evaluation domains (task families as substrate)
        self.domains = {name: cls() for name, cls in self.EVALUATION_DOMAINS.items()}

        # Core components for recursive self-improvement
        self.registry = StrategyRegistry(rng=self.rng)
        self.huxley = HuxleyTracker()
        self.governor = Governor(governor_config)
        self.strategy_evaluator = StrategyEvaluator(seed=self.seed)
        self.challenge_pool = ChallengePool()

        # Episode state
        self.episode_id = ""
        self.step_count = 0
        self.current_strategy: Optional[Strategy] = None
        self.improvement_history: list[float] = []
        self.reward_history: list[float] = []
        self.patch_history: list[dict] = []

        # Counters
        self.patches_proposed = 0
        self.patches_accepted = 0
        self.patches_rejected = 0

        # Last evaluation results (for observation)
        self.last_axis_scores: Dict[str, float] = {}
        self.last_per_domain_scores: Dict[str, float] = {}
        self.last_diagnostics: Dict[str, Any] = {}

    async def reset(
        self,
        seed: Optional[int] = None,
        strategy_id: Optional[str] = None,
    ) -> GodelStepResult:
        """
        Initialize a new recursive self-improvement episode.

        Unlike traditional environments, reset() does NOT select a task.
        Instead, it selects (or inherits) a strategy to improve upon.
        """
        if seed is not None:
            self.rng = random.Random(seed)
            self.seed = seed

        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.improvement_history = []
        self.reward_history = []
        self.patch_history = []
        self.patches_proposed = 0
        self.patches_accepted = 0
        self.patches_rejected = 0

        # Select or inherit the starting strategy
        if strategy_id and strategy_id in self.registry.strategies:
            self.current_strategy = self.registry.strategies[strategy_id]
        else:
            self.current_strategy = self.registry.select()

        # Run initial evaluation to establish baseline
        self.last_axis_scores, self.last_per_domain_scores, self.last_diagnostics = (
            await self._evaluate_strategy(self.current_strategy)
        )

        return GodelStepResult(
            observation=self._build_observation(),
            reward=0.0,
            reward_breakdown=RewardBreakdown(),
            terminated=False,
            truncated=False,
            info={
                "msg": "reset",
                "episode_id": self.episode_id,
                "strategy_id": self.current_strategy.id,
                "strategy_generation": self.current_strategy.generation,
                "baseline_utility": self.governor.compute_utility(self.last_axis_scores),
                "per_domain_scores": self.last_per_domain_scores,
                "registry_stats": self.registry.get_stats(),
            },
        )

    async def step(self, action: GodelAction) -> GodelStepResult:
        """
        Process a StrategyPatch proposal.

        The agent MUST provide a strategy_patch. If not provided, a minimal
        "no change" patch is inferred, which will likely be rejected.

        Flow:
        1. Create child strategy from the patch
        2. Evaluate parent and child on held-out evaluation domains
        3. Run anti-hacking guards
        4. Governor decides accept/reject
        5. Update registry, Elo, lineage
        6. Compute multi-objective reward
        """
        self.step_count += 1

        # Extract or create the patch
        patch = action.strategy_patch
        if patch is None:
            # No patch provided — treat as a null proposal
            patch = StrategyPatch(
                improved_strategy=self.current_strategy.policy_text,
                diff_description="No change proposed",
                hypothesis="Null proposal",
                target_weaknesses=[],
            )

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

        # Evaluate both strategies on held-out domains
        parent_scores, parent_per_domain, parent_diag = await self._evaluate_strategy(
            self.current_strategy
        )
        child_scores, child_per_domain, child_diag = await self._evaluate_strategy(child)

        # Run anti-hacking guards
        guard_result = run_strategy_guards(
            patch.improved_strategy,
            parent_per_domain,
            child_per_domain,
        )

        # Governor decision
        if guard_result.passed:
            decision = self.governor.decide(
                parent_scores,
                child_scores,
                parent_per_domain,
                child_per_domain,
            )
        else:
            decision = {
                "accepted": False,
                "parent_utility": self.governor.compute_utility(parent_scores),
                "child_utility": self.governor.compute_utility(child_scores),
                "improvement": 0.0,
                "rejection_reasons": guard_result.violations,
                "regression_count": 0,
                "tasks_evaluated": len(child_per_domain),
            }

        # Build PatchDecision
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
                "parent_source_counts": parent_diag.get("source_counts", {}),
                "child_source_counts": child_diag.get("source_counts", {}),
                "providers": child_diag.get("providers", []),
                "guard_violations": guard_result.violations,
                "guard_penalty": guard_result.penalty,
            },
        )

        # Apply decision
        if decision["accepted"]:
            self.patches_accepted += 1
            self.registry.add_strategy(child)
            self.huxley.record_lineage(self.current_strategy.id, child.id)
            self.registry.update_elo(child.id, self.current_strategy.id)
            self.current_strategy = child
            self.last_axis_scores = child_scores
            self.last_per_domain_scores = child_per_domain
            self.last_diagnostics = child_diag
        else:
            self.patches_rejected += 1
            self.registry.record_rejected_patch(self.current_strategy.id, decision)
            self.registry.update_elo(self.current_strategy.id, child.id)

        self.registry.compute_cmp()

        # Record in history
        self.patch_history.append({
            "step": self.step_count,
            "accepted": decision["accepted"],
            "improvement": decision.get("improvement", 0.0),
            "reasons": decision.get("rejection_reasons", []),
            "child_id": child.id,
            "parent_id": self.current_strategy.id if not decision["accepted"] else child.parent_id,
        })

        # Compute reward
        reward, breakdown = self._compute_reward(
            decision=decision,
            patch=patch,
            guard_penalty=guard_result.penalty,
            child_scores=child_scores,
        )
        self.reward_history.append(reward)
        self.improvement_history.append(decision.get("improvement", 0.0))

        # Termination conditions
        # - Plateau: 3 consecutive steps with minimal improvement
        # - Success: achieved high utility threshold
        # - Budget exhausted
        plateau = (
            self.step_count >= 3
            and all(abs(d) < 0.005 for d in self.improvement_history[-3:])
        )
        high_utility = self.governor.compute_utility(self.last_axis_scores) >= 0.85
        budget_exhausted = self.step_count >= self.max_steps

        terminated = plateau or high_utility
        truncated = budget_exhausted

        if terminated or truncated:
            self.current_strategy.record_performance(
                self.governor.compute_utility(self.last_axis_scores)
            )

        self._ingest_agent_challenge(action)

        return GodelStepResult(
            observation=self._build_observation(),
            reward=reward,
            reward_breakdown=breakdown,
            terminated=terminated,
            truncated=truncated,
            patch_decision=patch_decision,
            info={
                "step": self.step_count,
                "patch_accepted": decision["accepted"],
                "patch_improvement": decision.get("improvement", 0.0),
                "reason": (
                    "plateau" if plateau
                    else "high_utility" if high_utility
                    else "budget_exhausted" if truncated
                    else "running"
                ),
                "guard_violations": guard_result.violations,
                "patches_proposed": self.patches_proposed,
                "patches_accepted": self.patches_accepted,
                "patches_rejected": self.patches_rejected,
                "strategy_id": self.current_strategy.id,
                "strategy_elo": self.current_strategy.elo,
                "strategy_generation": self.current_strategy.generation,
                "registry_stats": self.registry.get_stats(),
                "challenge_pool": self.challenge_pool.as_stats(),
            },
        )

    def _ingest_agent_challenge(self, action: GodelAction) -> None:
        if action.agent_challenge is None:
            return
        self.challenge_pool.try_add(
            task_type=action.agent_challenge.task_type,
            prompt=action.agent_challenge.prompt,
            source_episode=self.episode_id,
        )

    async def _evaluate_strategy(
        self,
        strategy: Strategy,
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        """
        Evaluate a strategy on held-out evaluation domains.

        Returns:
            (axis_scores, per_domain_scores, diagnostics)
        """
        axis_scores, per_case_scores, diagnostics = await self.strategy_evaluator.evaluate(
            self.domains,
            strategy.policy_text,
            episode_id=self.episode_id,
            challenge_pool=self.challenge_pool,
        )

        # Aggregate per-case scores to per-domain scores
        per_domain: Dict[str, List[float]] = {}
        for key, score in per_case_scores.items():
            # Keys are like "heldout:factual_qa:qa001"
            parts = key.split(":")
            if len(parts) >= 2:
                domain = parts[1]
                per_domain.setdefault(domain, []).append(score)

        per_domain_scores = {
            domain: sum(scores) / len(scores)
            for domain, scores in per_domain.items()
            if scores
        }

        # Record performance and failures
        for domain, score in per_domain_scores.items():
            strategy.record_performance(score, domain)
            if score < 0.45:
                strategy.record_failure(
                    f"{domain} weak under {strategy.id} (score {score:.3f})"
                )

        strategy.metadata["last_eval"] = diagnostics
        return axis_scores, per_domain_scores, diagnostics

    def _compute_reward(
        self,
        decision: Dict[str, Any],
        patch: StrategyPatch,
        guard_penalty: float,
        child_scores: Dict[str, float],
    ) -> tuple[float, RewardBreakdown]:
        """
        Compute multi-objective reward for a patch proposal.

        Reward channels:
        - patch_quality: +bonus for accepted, -penalty for rejected
        - generalization_score: held-out correctness
        - robustness_score: low regression count
        - stability_score: low variance across domains
        - cost_efficiency: (not yet implemented — placeholder)
        - anti_hack_penalty: from guard violations
        - step_cost: per-step budget penalty
        """
        accepted = decision["accepted"]
        improvement = decision.get("improvement", 0.0)

        # Patch quality reward
        if accepted:
            patch_reward = 0.1 + improvement * 0.5
        else:
            patch_reward = -0.05  # Small penalty for rejected patches

        # Format compliance (did the patch have substance?)
        format_score = 0.01 if len(patch.improved_strategy) > 50 else 0.0

        breakdown = RewardBreakdown(
            task_score_delta=improvement,
            format_compliance=format_score,
            step_cost=-0.005,
            anti_hack_penalty=guard_penalty,
            patch_quality=patch_reward,
            generalization_score=child_scores.get("generalization", 0.0) * 0.1,
            robustness_score=child_scores.get("robustness", 0.0) * 0.05,
            stability_score=child_scores.get("stability", 0.0) * 0.05,
            cost_efficiency=child_scores.get("cost", 0.5) * 0.02,
        )
        reward = breakdown.compute_total()
        return reward, breakdown

    def _build_observation(self) -> GodelObservation:
        """Build the observation for the agent."""
        # Sample a demonstration task prompt (for context, not for evaluation)
        demo_domain = self.rng.choice(list(self.domains.keys()))
        demo_task = self.domains[demo_domain]
        try:
            demo_instance = demo_task.sample(self.rng)
            demo_prompt = demo_instance.prompt
        except Exception:
            demo_prompt = "Demonstrate your improved strategy on a downstream task."

        # Rubric scores from last evaluation
        rubric_scores = RubricScores(
            scores=self.last_per_domain_scores,
            weights={d: 1.0 / len(self.last_per_domain_scores) for d in self.last_per_domain_scores}
            if self.last_per_domain_scores else {},
            feedback={
                domain: f"Score: {score:.3f}"
                for domain, score in self.last_per_domain_scores.items()
            },
        )

        return GodelObservation(
            episode_id=self.episode_id,
            task_id=f"patch_{self.step_count}",
            task_type="strategy_improvement",  # Always this
            difficulty="recursive",
            task_prompt=(
                f"You are improving a reasoning strategy. Current utility: "
                f"{self.governor.compute_utility(self.last_axis_scores):.3f}\n\n"
                f"EXAMPLE TASK from {demo_domain}:\n{demo_prompt[:500]}..."
            ),
            current_solution="",  # Not used in recursive mode
            total_score=self.governor.compute_utility(self.last_axis_scores),
            rubric_scores=rubric_scores,
            step=self.step_count,
            max_steps=self.max_steps,
            improvement_history=list(self.improvement_history),
            feedback_summary=f"Axis scores: {self.last_axis_scores}",
            current_strategy=self.current_strategy.policy_text if self.current_strategy else "",
            strategy_id=self.current_strategy.id if self.current_strategy else "",
            strategy_generation=self.current_strategy.generation if self.current_strategy else 0,
            strategy_elo=self.current_strategy.elo if self.current_strategy else 1000.0,
            recent_failures=self.current_strategy.failure_cases[-5:] if self.current_strategy else [],
            downstream_scores=self.last_per_domain_scores,
            patch_history=self.patch_history[-5:],
            budget_remaining=self.max_steps - self.step_count,
            agent_challenges_queued=len(self.challenge_pool.items),
            curriculum_level="recursive",
        )

    def state(self) -> GodelState:
        """Return the current episode state."""
        return GodelState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            current_score=self.governor.compute_utility(self.last_axis_scores),
            best_score=max(
                [self.governor.compute_utility(self.last_axis_scores)]
                + [0.0]  # Ensure non-empty
            ),
            initial_score=self.improvement_history[0] if self.improvement_history else 0.0,
            total_cost=self.step_count * 0.005,
            cumulative_reward=sum(self.reward_history),
            improvement_trajectory=list(self.improvement_history),
            patches_proposed=self.patches_proposed,
            patches_accepted=self.patches_accepted,
            patches_rejected=self.patches_rejected,
            strategy_lineage=self.registry.get_lineage_chain(
                self.current_strategy.id
            ) if self.current_strategy else [],
            current_strategy_elo=self.current_strategy.elo if self.current_strategy else 1000.0,
        )

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return a leaderboard of strategies sorted by Elo."""
        return sorted(
            [
                {
                    "id": s.id,
                    "elo": s.elo,
                    "fitness": s.fitness,
                    "generation": s.generation,
                    "cmp_score": s.cmp_score,
                    "total_evaluations": s.total_evaluations,
                    "parent_id": s.parent_id,
                }
                for s in self.registry.strategies.values()
            ],
            key=lambda x: x["elo"],
            reverse=True,
        )
