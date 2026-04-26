"""
Backward-compatible wrappers for the deterministic fallback policy.

Historically this module contained a large heuristic action generator. The
current implementation delegates to the reference-grounded deterministic solver
so fallback behavior stays aligned with real task semantics instead of generic
boilerplate templates.
"""
from __future__ import annotations

from typing import Sequence

from godel_engine.deterministic_solver import (
    build_reference_action,
    build_reference_solution,
    build_reference_strategy_patch,
)
from godel_engine.models import GodelAction, StrategyPatch


def build_heuristic_solution(
    task_prompt: str,
    task_type: str,
    strategy_text: str | None = None,
) -> str:
    return build_reference_solution(
        task_prompt=task_prompt,
        task_type=task_type,
        strategy_text=strategy_text,
    )


def build_heuristic_action(
    task_prompt: str,
    task_type: str,
    strategy_text: str | None = None,
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
) -> GodelAction:
    return build_reference_action(
        task_prompt=task_prompt,
        task_type=task_type,
        strategy_text=strategy_text,
        recent_failures=recent_failures,
        downstream_scores=downstream_scores,
    )


def build_heuristic_strategy_patch(
    strategy_text: str | None = None,
    recent_failures: Sequence[str] | None = None,
    downstream_scores: dict[str, float] | None = None,
) -> StrategyPatch:
    return build_reference_strategy_patch(
        strategy_text=strategy_text,
        recent_failures=recent_failures,
        downstream_scores=downstream_scores,
    )
