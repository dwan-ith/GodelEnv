"""
Core Gödel Engine Logic — Strategy Registry, Governor, and Evolutionary Layers.

GodelEnv 2.0: Self-modification is explicit. Improvement proposals are testable.
Acceptance depends on objective evidence, not vibes.

- StrategyRegistry: stores accepted strategies with full lineage, Elo ratings,
  ablation records, and failure cases. Replaces the old DarwinPool.
- Governor: decides accept/reject based on multi-objective utility comparison
  between parent and child on held-out evaluations.
- HuxleyTracker: tracks Clade-Metaproductivity (CMP) — strategies that produce
  better descendants are valued higher than strategies that are merely good.
"""
from __future__ import annotations

import math
import random
import uuid
from collections import defaultdict
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field


# ── Default reasoning strategy ───────────────────────────────────────

DEFAULT_STRATEGY_TEXT = """\
REASONING STRATEGY v1.0

1. DECOMPOSE: Break the problem into atomic claims, assumptions, and decision points.
2. EVIDENCE: For each claim, identify supporting evidence and note gaps.
3. COUNTER: Generate at least one counterargument or alternative hypothesis.
4. UNCERTAINTY: Explicitly mark confidence levels (high/medium/low) for each claim.
5. VERIFY: Run a self-check — does the answer actually address the original question?
6. SYNTHESIZE: Combine verified claims into a final answer.
7. REFLECT: Note what the strategy handled well and what it missed, for future improvement.
"""


# ── Strategy ─────────────────────────────────────────────────────────

@dataclass
class Strategy:
    """A reasoning policy with full lineage and performance tracking."""

    id: str
    policy_text: str = DEFAULT_STRATEGY_TEXT
    parent_id: Optional[str] = None
    generation: int = 0
    fitness: float = 0.0
    elo: float = 1000.0
    cmp_score: float = 0.0  # Clade-Metaproductivity
    history: List[float] = field(default_factory=list)
    per_task_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    failure_cases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Patch that created this strategy (None for root)
    patch_description: Optional[str] = None
    patch_hypothesis: Optional[str] = None

    def record_performance(self, score: float, task_type: str = ""):
        self.history.append(score)
        self.fitness = sum(self.history) / len(self.history)
        if task_type:
            self.per_task_scores[task_type].append(score)

    def record_failure(self, description: str, max_failures: int = 20):
        self.failure_cases.append(description)
        if len(self.failure_cases) > max_failures:
            self.failure_cases = self.failure_cases[-max_failures:]

    def get_downstream_summary(self) -> Dict[str, float]:
        """Mean score per task family."""
        return {
            task: sum(scores) / len(scores)
            for task, scores in self.per_task_scores.items()
            if scores
        }

    def get_weaknesses(self, threshold: float = 0.3) -> List[str]:
        """Task families where mean score is below threshold."""
        return [
            task for task, scores in self.per_task_scores.items()
            if scores and (sum(scores) / len(scores)) < threshold
        ]

    @property
    def total_evaluations(self) -> int:
        return len(self.history)


# ── Elo System ───────────────────────────────────────────────────────

def compute_elo_delta(
    winner_elo: float,
    loser_elo: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Compute Elo rating changes after a head-to-head comparison."""
    expected_winner = 1.0 / (1.0 + 10.0 ** ((loser_elo - winner_elo) / 400.0))
    expected_loser = 1.0 - expected_winner
    winner_delta = k * (1.0 - expected_winner)
    loser_delta = k * (0.0 - expected_loser)
    return winner_delta, loser_delta


# ── Governor ─────────────────────────────────────────────────────────

@dataclass
class GovernorConfig:
    """Configuration for the patch acceptance Governor."""

    # Minimum improvement required for acceptance
    min_improvement: float = 0.01

    # Maximum fraction of tasks allowed to regress
    max_regression_fraction: float = 0.2

    # Maximum acceptable score variance across held-out tasks
    max_variance: float = 0.15

    # Weights for multi-objective utility
    correctness_weight: float = 0.35
    generalization_weight: float = 0.25
    robustness_weight: float = 0.20
    cost_weight: float = 0.10
    stability_weight: float = 0.10


class Governor:
    """
    Decides whether a proposed StrategyPatch should be accepted.

    Acceptance depends on objective evidence from held-out evaluation:
    1. Child must have higher multi-objective utility than parent.
    2. Child must not regress on too many individual tasks.
    3. Child must have acceptable variance (stability).

    This is the "empirical proof" that replaces formal verification
    in the Gödel machine paradigm.
    """

    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig()
        self.decision_log: List[Dict[str, Any]] = []

    def compute_utility(self, scores: Dict[str, float]) -> float:
        """Compute weighted multi-objective utility from axis scores."""
        cfg = self.config
        return (
            cfg.correctness_weight * scores.get("correctness", 0.0)
            + cfg.generalization_weight * scores.get("generalization", 0.0)
            + cfg.robustness_weight * scores.get("robustness", 0.0)
            + cfg.cost_weight * scores.get("cost", 0.0)
            + cfg.stability_weight * scores.get("stability", 0.0)
        )

    def decide(
        self,
        parent_scores: Dict[str, float],
        child_scores: Dict[str, float],
        per_task_parent: Dict[str, float],
        per_task_child: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Make an accept/reject decision based on multi-axis evaluation.

        Returns a dictionary with:
        - accepted: bool
        - parent_utility, child_utility, improvement: floats
        - rejection_reasons: list of strings
        - per-axis breakdown
        """
        parent_utility = self.compute_utility(parent_scores)
        child_utility = self.compute_utility(child_scores)
        improvement = child_utility - parent_utility

        rejection_reasons: List[str] = []

        # Gate 1: Minimum improvement threshold
        if improvement < self.config.min_improvement:
            rejection_reasons.append(
                f"Insufficient improvement: {improvement:.4f} < {self.config.min_improvement}"
            )

        # Gate 2: Regression check — child must not regress on too many tasks
        regression_count = 0
        total_tasks = 0
        for task in set(per_task_parent) | set(per_task_child):
            p = per_task_parent.get(task, 0.0)
            c = per_task_child.get(task, 0.0)
            total_tasks += 1
            if c < p - 0.01:  # Small tolerance
                regression_count += 1

        if total_tasks > 0:
            regression_fraction = regression_count / total_tasks
            if regression_fraction > self.config.max_regression_fraction:
                rejection_reasons.append(
                    f"Too many regressions: {regression_count}/{total_tasks} "
                    f"({regression_fraction:.0%} > {self.config.max_regression_fraction:.0%})"
                )

        # Gate 3: Variance / stability check
        child_values = list(per_task_child.values())
        if len(child_values) >= 2:
            mean_c = sum(child_values) / len(child_values)
            variance = sum((v - mean_c) ** 2 for v in child_values) / len(child_values)
            if variance > self.config.max_variance:
                rejection_reasons.append(
                    f"High variance: {variance:.4f} > {self.config.max_variance}"
                )

        accepted = len(rejection_reasons) == 0

        decision = {
            "accepted": accepted,
            "parent_utility": parent_utility,
            "child_utility": child_utility,
            "improvement": improvement,
            "rejection_reasons": rejection_reasons,
            "regression_count": regression_count,
            "tasks_evaluated": total_tasks,
            "axis_scores": {
                "parent": parent_scores,
                "child": child_scores,
            },
        }

        self.decision_log.append(decision)
        return decision


# ── Strategy Registry ────────────────────────────────────────────────

class StrategyRegistry:
    """
    Stores accepted strategies with full lineage tracking.

    Replaces the old DarwinPool with a richer system:
    - Lineage: every strategy knows its parent chain
    - Elo: strategies are rated via head-to-head comparisons
    - CMP: Huxleyan Clade-Metaproductivity tracks which lineages
      produce the best descendants
    - Ablation: tracks which patches were rejected and why
    """

    def __init__(self, max_size: int = 50, rng: Optional[random.Random] = None):
        self.max_size = max_size
        self.rng = rng or random.Random()
        self.strategies: Dict[str, Strategy] = {}
        self.lineage: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.rejected_patches: List[Dict[str, Any]] = []

        # Seed with the default strategy
        root = Strategy(id="strat_root", generation=0, elo=1000.0)
        self.add_strategy(root)

    def add_strategy(self, strategy: Strategy):
        """Add a strategy to the registry."""
        self.strategies[strategy.id] = strategy
        if strategy.parent_id:
            self.lineage.setdefault(strategy.parent_id, []).append(strategy.id)

        # Prune if over capacity — remove lowest Elo, never remove root
        if len(self.strategies) > self.max_size:
            worst = min(
                (s for s in self.strategies.values() if s.id != "strat_root"),
                key=lambda s: s.elo + s.cmp_score,
                default=None,
            )
            if worst:
                del self.strategies[worst.id]

    def record_rejected_patch(self, parent_id: str, decision: Dict[str, Any]):
        """Log a rejected patch for analysis."""
        self.rejected_patches.append({
            "parent_id": parent_id,
            **decision,
        })
        # Keep only the most recent rejections
        if len(self.rejected_patches) > 100:
            self.rejected_patches = self.rejected_patches[-100:]

    def select(self, tournament_size: int = 3) -> Strategy:
        """Tournament selection weighted by Elo + CMP."""
        candidates = self.rng.sample(
            list(self.strategies.values()),
            min(tournament_size, len(self.strategies)),
        )
        return max(candidates, key=lambda s: s.elo + s.cmp_score)

    def get_best(self) -> Strategy:
        """Return the highest-rated strategy."""
        return max(self.strategies.values(), key=lambda s: s.elo)

    def get_lineage_chain(self, strategy_id: str) -> List[str]:
        """Return the full ancestry chain from root to this strategy."""
        chain = [strategy_id]
        current = self.strategies.get(strategy_id)
        while current and current.parent_id and current.parent_id in self.strategies:
            chain.append(current.parent_id)
            current = self.strategies[current.parent_id]
        return list(reversed(chain))

    def update_elo(self, winner_id: str, loser_id: str):
        """Update Elo ratings after a head-to-head comparison."""
        winner = self.strategies.get(winner_id)
        loser = self.strategies.get(loser_id)
        if winner and loser:
            w_delta, l_delta = compute_elo_delta(winner.elo, loser.elo)
            winner.elo += w_delta
            loser.elo += l_delta

    def compute_cmp(self):
        """Compute Clade-Metaproductivity for all strategies."""
        for parent_id in list(self.strategies.keys()):
            descendants = self._get_all_descendants(parent_id)
            if descendants:
                self.strategies[parent_id].cmp_score = (
                    sum(s.fitness for s in descendants) / len(descendants)
                )

    def _get_all_descendants(self, parent_id: str) -> List[Strategy]:
        """BFS to find all descendants of a strategy."""
        descendants = []
        queue = list(self.lineage.get(parent_id, []))
        visited = set()
        while queue:
            child_id = queue.pop(0)
            if child_id in visited:
                continue
            visited.add(child_id)
            if child_id in self.strategies:
                descendants.append(self.strategies[child_id])
            queue.extend(self.lineage.get(child_id, []))
        return descendants

    def get_stats(self) -> Dict[str, Any]:
        """Return registry statistics for logging."""
        if not self.strategies:
            return {}
        best = self.get_best()
        return {
            "total_strategies": len(self.strategies),
            "best_strategy_id": best.id,
            "best_elo": best.elo,
            "best_fitness": best.fitness,
            "max_generation": max(s.generation for s in self.strategies.values()),
            "total_rejected": len(self.rejected_patches),
        }


# ── Legacy compatibility aliases ─────────────────────────────────────
# These allow old code to import DarwinPool / HuxleyTracker without breaking

DarwinPool = StrategyRegistry

class HuxleyTracker:
    """Legacy wrapper — CMP is now computed directly by StrategyRegistry."""

    def __init__(self):
        self.lineage: Dict[str, List[str]] = {}

    def record_lineage(self, parent_id: str, child_id: str):
        self.lineage.setdefault(parent_id, []).append(child_id)

    def compute_cmp(self, pool):
        if isinstance(pool, StrategyRegistry):
            pool.compute_cmp()
