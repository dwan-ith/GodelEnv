"""
Automatic Curriculum Controller for Godel Env.

Tracks rolling agent success rate per difficulty level and automatically
escalates/de-escalates to keep the agent in the optimal learning zone.

Rules:
  - If success_rate > escalation_threshold for last N episodes at current
    level, advance to the next difficulty.
  - If success_rate < deescalation_threshold, step back.
  - Manual override via reset(difficulty=...) always takes precedence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# Ordered from easiest to hardest
DIFFICULTY_LADDER = ["easy", "medium", "hard", "godel"]


@dataclass
class CurriculumController:
    """Automatic difficulty progression based on rolling success rate."""

    window_size: int = 10
    escalation_threshold: float = 0.6
    deescalation_threshold: float = 0.2
    success_score_threshold: float = 0.7   # Score >= this counts as "success"

    # Internal state
    current_level_idx: int = 0
    _history: dict = field(default_factory=dict)

    def __post_init__(self):
        # Rolling window of success/failure per difficulty level
        for level in DIFFICULTY_LADDER:
            self._history[level] = deque(maxlen=self.window_size)

    @property
    def current_difficulty(self) -> str:
        return DIFFICULTY_LADDER[self.current_level_idx]

    def suggest_difficulty(self) -> str:
        """Return the current recommended difficulty level."""
        return self.current_difficulty

    def record_outcome(self, difficulty: str, final_score: float):
        """Record an episode outcome and potentially adjust difficulty."""
        success = final_score >= self.success_score_threshold

        if difficulty in self._history:
            self._history[difficulty].append(success)

        # Only auto-adjust if the recorded difficulty matches current level
        if difficulty != self.current_difficulty:
            return

        window = self._history[difficulty]
        if len(window) < 3:
            return  # Not enough data yet

        success_rate = sum(window) / len(window)

        # Escalate
        if (
            success_rate >= self.escalation_threshold
            and self.current_level_idx < len(DIFFICULTY_LADDER) - 1
        ):
            self.current_level_idx += 1

        # De-escalate
        elif (
            success_rate <= self.deescalation_threshold
            and self.current_level_idx > 0
        ):
            self.current_level_idx -= 1

    def get_stats(self) -> dict:
        """Return curriculum statistics for logging/dashboard."""
        stats = {
            "current_difficulty": self.current_difficulty,
            "level_index": self.current_level_idx,
        }
        for level in DIFFICULTY_LADDER:
            window = self._history.get(level, [])
            rate = sum(window) / len(window) if window else 0.0
            stats[f"{level}_success_rate"] = round(rate, 3)
            stats[f"{level}_episodes"] = len(window)
        return stats

    def reset_to(self, difficulty: str):
        """Manually set the curriculum to a specific difficulty level."""
        if difficulty in DIFFICULTY_LADDER:
            self.current_level_idx = DIFFICULTY_LADDER.index(difficulty)
