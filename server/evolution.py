"""
Core Gödel Engine Logic (Darwin and Huxley layers).
"""
from __future__ import annotations
import random
import uuid
import time
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

@dataclass
class Strategy:
    id: str
    parent_id: Optional[str] = None
    generation: int = 0
    fitness: float = 0.0
    cmp_score: float = 0.0  # Clade-Metaproductivity
    history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_performance(self, score: float):
        self.history.append(score)
        # Simple moving average for fitness
        self.fitness = sum(self.history) / len(self.history)

class DarwinPool:
    """Manages the population of strategies."""
    def __init__(self, max_size: int = 20, rng: Optional[random.Random] = None):
        self.max_size = max_size
        self.rng = rng or random.Random()
        self.strategies: Dict[str, Strategy] = {}
        # Seed with a default strategy
        self.add_strategy(Strategy(id="strat_default", generation=0))

    def add_strategy(self, strategy: Strategy):
        self.strategies[strategy.id] = strategy
        if len(self.strategies) > self.max_size:
            # Remove lowest fitness
            worst = min(self.strategies.values(), key=lambda s: s.fitness)
            del self.strategies[worst.id]

    def select(self) -> Strategy:
        """Tournament selection."""
        candidates = self.rng.sample(list(self.strategies.values()), min(3, len(self.strategies)))
        return max(candidates, key=lambda s: s.fitness + s.cmp_score) # Consider CMP in selection

class HuxleyTracker:
    """Tracks metaproductivity (CMP) across generations."""
    def __init__(self):
        self.lineage: Dict[str, List[str]] = {} # parent -> children

    def record_lineage(self, parent_id: str, child_id: str):
        if parent_id not in self.lineage:
            self.lineage[parent_id] = []
        self.lineage[parent_id].append(child_id)

    def compute_cmp(self, pool: DarwinPool):
        """
        Assign higher value to strategies that produced better descendants.
        Assign CMP = avg(fitness of all descendants).
        """
        for parent_id in pool.strategies:
            descendants = self._get_all_descendants(parent_id, pool)
            if descendants:
                pool.strategies[parent_id].cmp_score = sum(s.fitness for s in descendants) / len(descendants)

    def _get_all_descendants(self, parent_id: str, pool: DarwinPool) -> List[Strategy]:
        descendants = []
        queue = self.lineage.get(parent_id, [])
        while queue:
            child_id = queue.pop(0)
            if child_id in pool.strategies:
                descendants.append(pool.strategies[child_id])
            queue.extend(self.lineage.get(child_id, []))
        return descendants
