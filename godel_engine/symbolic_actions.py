from __future__ import annotations

from godel_engine.heuristic_policy import build_heuristic_action, build_heuristic_solution
from godel_engine.models import EditType, GodelAction


ACTION_DIRECT_BEST = "<A_DIRECT_BEST>"
ACTION_PATCH_BALANCED = "<A_PATCH_BALANCED>"
ACTION_NOOP = "<A_NOOP>"


ACTION_DESCRIPTIONS = {
    ACTION_DIRECT_BEST: "Expand the task using the strongest deterministic task-specific solver.",
    ACTION_PATCH_BALANCED: "Propose a balanced strategy patch with verification, reflection, and downstream demonstration.",
    ACTION_NOOP: "Keep the current draft mostly unchanged. This is a weak fallback action.",
}


def allowed_action_tokens(task_type: str) -> list[str]:
    if task_type == "strategy_optimization":
        return [ACTION_PATCH_BALANCED, ACTION_DIRECT_BEST, ACTION_NOOP]
    return [ACTION_DIRECT_BEST, ACTION_NOOP]


def preferred_action_token(task_type: str) -> str:
    if task_type == "strategy_optimization":
        return ACTION_PATCH_BALANCED
    return ACTION_DIRECT_BEST


def find_action_token(text: str) -> str | None:
    for token in ACTION_DESCRIPTIONS:
        if token in text:
            return token
    return None


def build_action_from_token(
    token: str,
    *,
    task_prompt: str,
    task_type: str,
    current_solution: str = "",
    strategy_text: str = "",
) -> GodelAction:
    if token == ACTION_NOOP:
        return GodelAction(
            solution=current_solution.strip() or "No change proposed.",
            edit_type=EditType.REFINE,
            strategy_note="Expanded from compact action token: noop",
        )

    if token == ACTION_PATCH_BALANCED and task_type == "strategy_optimization":
        action = build_heuristic_action(
            task_prompt,
            task_type,
            strategy_text=strategy_text,
        )
        action.strategy_note = "Expanded from compact action token: balanced patch"
        return action

    return GodelAction(
        solution=build_heuristic_solution(
            task_prompt,
            task_type,
            strategy_text=strategy_text,
        ),
        edit_type=EditType.RESTRUCTURE
        if task_type not in {"code_improvement", "python_optimized"}
        else EditType.FIX_ERRORS,
        strategy_note="Expanded from compact action token: direct best",
    )
