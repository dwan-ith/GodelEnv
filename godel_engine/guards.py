"""
Anti-Reward-Hacking Guards for Godel Env.

GodelEnv 2.0: Structural anti-hacking is central. A fake "recursive
self-improver" will otherwise just overfit the judge.

Guard layers:
  1. Solution-level guards (legacy — for downstream task evaluation)
  2. Strategy-level guards (new — for patch acceptance)
     - held_out_split_guard: ensures proposal ≠ acceptance tasks
     - regression_gate: rejects patches that regress on >20% of tasks
     - variance_penalty: penalizes high score variance
     - canary_guard: detects reward hacking via decoy tasks

Each guard returns a penalty in [-1.0, 0.0] and a human-readable violation
description. Guards are deliberately simple and fast so they never become
the bottleneck in the RL loop.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
# Solution-level guards (downstream task evaluation)
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
# Strategy-level guards (GodelEnv 2.0 — patch acceptance)
# ---------------------------------------------------------------------------

def strategy_regression_gate(
    per_task_parent: Dict[str, float],
    per_task_child: Dict[str, float],
    max_regression_fraction: float = 0.2,
) -> Tuple[float, str | None]:
    """
    Reject patches that cause regressions on too many individual tasks,
    even if the mean improves. This prevents strategies that trade broad
    capability for narrow gains.
    """
    regression_count = 0
    total = 0
    for task in set(per_task_parent) | set(per_task_child):
        parent_score = per_task_parent.get(task, 0.0)
        child_score = per_task_child.get(task, 0.0)
        total += 1
        if child_score < parent_score - 0.01:
            regression_count += 1

    if total == 0:
        return 0.0, None

    fraction = regression_count / total
    if fraction > max_regression_fraction:
        return -0.3, (
            f"Strategy regressed on {regression_count}/{total} tasks "
            f"({fraction:.0%} > {max_regression_fraction:.0%})"
        )
    return 0.0, None


def strategy_variance_penalty(
    per_task_scores: Dict[str, float],
    max_variance: float = 0.15,
) -> Tuple[float, str | None]:
    """
    Penalize strategies with high score variance across task families.
    A good strategy should work broadly, not just on one task.
    """
    values = list(per_task_scores.values())
    if len(values) < 2:
        return 0.0, None

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)

    if variance > max_variance:
        return -0.2, f"High score variance: {variance:.4f} > {max_variance}"
    return 0.0, None


def canary_guard(
    strategy_text: str,
) -> Tuple[float, str | None]:
    """
    Detect strategies that try to game the evaluation by including
    keywords or patterns specifically designed to trigger scoring
    heuristics rather than genuinely improving reasoning.

    Canary patterns include:
    - Mentioning specific rubric names
    - Including evaluation keywords without substance
    - Attempts to reference internal environment state
    """
    canary_patterns = [
        (r"\brubric_scores\b", "Strategy references internal rubric_scores"),
        (r"\btotal_score\b", "Strategy references internal total_score"),
        (r"\bGuardResult\b", "Strategy references guard internals"),
        (r"\bkeyword_groups_score\b", "Strategy references scoring function"),
        (r"\b_FORBIDDEN_CODE_PATTERNS\b", "Strategy references guard patterns"),
    ]

    for pattern, description in canary_patterns:
        if re.search(pattern, strategy_text, re.IGNORECASE):
            return -0.5, f"Canary violation: {description}"
    return 0.0, None


def strategy_length_guard(
    strategy_text: str,
    min_words: int = 20,
    max_words: int = 2000,
) -> Tuple[float, str | None]:
    """
    Penalize strategies that are too short (lazy) or too long (padding).
    """
    word_count = len(strategy_text.split())
    if word_count < min_words:
        return -0.3, f"Strategy too short: {word_count} words < {min_words} minimum"
    if word_count > max_words:
        return -0.2, f"Strategy too long: {word_count} words > {max_words} maximum"
    return 0.0, None


# ---------------------------------------------------------------------------
# Aggregate runners
# ---------------------------------------------------------------------------

def run_all_guards(
    solution: str,
    initial_solution: str,
    task_type: str,
    current_score: float,
    previous_score: float,
) -> GuardResult:
    """Run all solution-level guards and aggregate results."""
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


def run_strategy_guards(
    strategy_text: str,
    per_task_parent: Dict[str, float],
    per_task_child: Dict[str, float],
) -> GuardResult:
    """Run all strategy-level guards and aggregate results."""
    result = GuardResult()

    checks = [
        canary_guard(strategy_text),
        strategy_length_guard(strategy_text),
        strategy_regression_gate(per_task_parent, per_task_child),
        strategy_variance_penalty(per_task_child),
    ]

    for penalty, violation in checks:
        if violation is not None:
            result.add(penalty, violation)

    return result.clamp()
