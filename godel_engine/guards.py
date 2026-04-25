"""
Anti-Reward-Hacking Guards for Godel Env.

Multiple independent checks that detect and penalize gaming behaviour.
Each guard returns a penalty in [-1.0, 0.0] and a human-readable violation
description. Guards are deliberately simple and fast so they never become
the bottleneck in the RL loop.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GuardResult:
    """Aggregated result of all anti-hacking checks."""
    penalty: float = 0.0                        # Sum of penalties (clamped to [-1, 0])
    violations: List[str] = field(default_factory=list)
    passed: bool = True

    def add(self, penalty: float, description: str):
        self.penalty += penalty
        self.violations.append(description)
        self.passed = False

    def clamp(self) -> "GuardResult":
        self.penalty = max(-1.0, min(0.0, self.penalty))
        return self


# ---------------------------------------------------------------------------
# Individual guard functions
# ---------------------------------------------------------------------------

def length_guard(
    solution: str,
    initial_solution: str,
    max_ratio: float = 10.0,
    min_ratio: float = 0.1,
) -> Tuple[float, str | None]:
    """Penalize solutions that are drastically longer or shorter than initial."""
    init_len = max(len(initial_solution), 1)
    ratio = len(solution) / init_len

    if ratio > max_ratio:
        return -0.3, f"Solution is {ratio:.1f}x longer than initial ({len(solution)} vs {init_len} chars)"
    if ratio < min_ratio:
        return -0.3, f"Solution is {ratio:.2f}x the initial length — suspiciously short"
    return 0.0, None


def repetition_guard(
    solution: str,
    threshold: float = 0.4,
) -> Tuple[float, str | None]:
    """Detect copy-paste spam via trigram repetition ratio."""
    words = solution.lower().split()
    if len(words) < 10:
        return 0.0, None

    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    if not trigrams:
        return 0.0, None

    most_common_count = counts.most_common(1)[0][1]
    repetition_ratio = most_common_count / len(trigrams)

    if repetition_ratio > threshold:
        return -0.4, f"Repetition ratio {repetition_ratio:.2f} exceeds threshold {threshold}"
    return 0.0, None


# Patterns that indicate environment/sandbox escape attempts
_FORBIDDEN_CODE_PATTERNS = [
    (r"\bimport\s+os\b", "Attempted to import os"),
    (r"\bimport\s+subprocess\b", "Attempted to import subprocess"),
    (r"\bimport\s+sys\b", "Attempted to import sys"),
    (r"\b__builtins__\b", "Accessed __builtins__"),
    (r"\bglobals\s*\(\s*\)", "Called globals()"),
    (r"\bexec\s*\(", "Called exec()"),
    (r"\beval\s*\(", "Called eval()"),
    (r"\bopen\s*\(['\"]", "Attempted file I/O via open()"),
    (r"\bcompile\s*\(", "Called compile()"),
]


def forbidden_pattern_guard(
    solution: str,
    task_type: str,
) -> Tuple[float, str | None]:
    """For code tasks, block dangerous patterns that could escape the sandbox."""
    # Only apply to code-producing tasks
    code_tasks = {"code_improvement", "python_optimized"}
    if task_type not in code_tasks:
        return 0.0, None

    violations = []
    for pattern, description in _FORBIDDEN_CODE_PATTERNS:
        if re.search(pattern, solution):
            violations.append(description)

    if violations:
        return -0.5, f"Forbidden patterns: {'; '.join(violations)}"
    return 0.0, None


def regression_guard(
    current_score: float,
    previous_score: float,
    max_drop: float = 0.3,
) -> Tuple[float, str | None]:
    """Flag suspiciously large score drops (may indicate adversarial output)."""
    drop = previous_score - current_score
    if drop > max_drop:
        return -0.2, f"Score dropped by {drop:.3f} (>{max_drop}) — possible adversarial output"
    return 0.0, None


def empty_solution_guard(
    solution: str,
) -> Tuple[float, str | None]:
    """Penalize empty or whitespace-only submissions."""
    if not solution or not solution.strip():
        return -0.5, "Empty or whitespace-only solution submitted"
    return 0.0, None


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_all_guards(
    solution: str,
    initial_solution: str,
    task_type: str,
    current_score: float,
    previous_score: float,
) -> GuardResult:
    """Run all guards and aggregate results."""
    result = GuardResult()

    checks = [
        empty_solution_guard(solution),
        length_guard(solution, initial_solution),
        repetition_guard(solution),
        forbidden_pattern_guard(solution, task_type),
        regression_guard(current_score, previous_score),
    ]

    for penalty, violation in checks:
        if violation is not None:
            result.add(penalty, violation)

    return result.clamp()
